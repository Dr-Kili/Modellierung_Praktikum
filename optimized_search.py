"""
Optimierte Suche nach erregbaren Konfigurationen im Suel et al. Parameterbereich (bs: 0.6-1.0, bk: 0.05-0.1)
für das B. subtilis Kompetenzmodell.

Dieses Skript führt eine hochauflösende Parametersuche durch und analysiert die gefundenen
erregbaren Konfigurationen.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import competence_circuit_analysis as comp_model
from stochastic_simulation import analyze_stochastic_competence

def create_results_directory():
    """
    Erstellt ein Ergebnisverzeichnis mit Zeitstempel.
    
    Returns:
        str: Pfad zum erstellten Verzeichnis
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"results_suel_search_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def is_excitable(params, K_range=np.linspace(0, 1.5, 300), S_range=np.linspace(0, 1.5, 300), verbose=False):
    """
    Prüft genauer, ob ein Parametersatz zu einem erregbaren System führt.
    
    Args:
        params: Parametersatz
        K_range: ComK-Wertebereich für Analyse
        S_range: ComS-Wertebereich für Analyse
        verbose: Wenn True, werden detaillierte Informationen ausgegeben
        
    Returns:
        bool: True, wenn das System erregbar ist
        dict: Zusätzliche Informationen zum System
    """
    # Fixpunkte finden
    fps = comp_model.find_fixed_points(params)
    
    if not fps:
        if verbose:
            print("Keine Fixpunkte gefunden.")
        return False, {}
    
    # Fixpunkte klassifizieren
    fp_types = [comp_model.classify_fixed_point(fp[0], fp[1], params) for fp in fps]
    
    # Anzahl stabiler und instabiler Fixpunkte zählen
    stable_fps = [fp for i, fp in enumerate(fps) if 'Stabil' in fp_types[i]]
    unstable_fps = [fp for i, fp in enumerate(fps) if 'Instabil' in fp_types[i]]
    saddle_fps = [fp for i, fp in enumerate(fps) if 'Sattel' in fp_types[i]]
    
    if verbose:
        print(f"Gefundene Fixpunkte: {len(fps)}")
        print(f"Stabile Fixpunkte: {len(stable_fps)}")
        print(f"Instabile Fixpunkte: {len(unstable_fps)}")
        print(f"Sattelpunkte: {len(saddle_fps)}")
        for i, (fp, fp_type) in enumerate(zip(fps, fp_types)):
            print(f"  FP{i+1}: ({fp[0]:.4f}, {fp[1]:.4f}) - {fp_type}")
    
    # Ein erregbares System sollte haben:
    # 1. Genau einen stabilen Fixpunkt (vegetativer Zustand)
    # 2. Mindestens einen Sattelpunkt
    # 3. Idealerweise einen instabilen Fixpunkt (Kompetenzzustand)
    # 4. Der stabile Fixpunkt sollte bei niedrigem ComK sein
    
    # Strikte Version der Prüfung (genau 1 stabil, 1 Sattel, 1 instabil)
    strict_excitable = (len(stable_fps) == 1 and 
                       len(saddle_fps) == 1 and 
                       len(unstable_fps) == 1 and
                       stable_fps[0][0] < 0.3)  # Stabiler Punkt mit niedrigem ComK
    
    # Weniger strikte Version (1 stabil bei niedrigem ComK, Rest egal)
    loose_excitable = (len(stable_fps) == 1 and
                     len(saddle_fps) >= 1 and
                     len(fps) >= 3 and
                     stable_fps[0][0] < 0.3)  # Stabiler Punkt mit niedrigem ComK
    
    # Zusätzliche Informationen sammeln
    info = {
        'fixed_points': fps,
        'fp_types': fp_types,
        'stable_fps': stable_fps,
        'unstable_fps': unstable_fps,
        'saddle_fps': saddle_fps,
        'strict_excitable': strict_excitable,
        'loose_excitable': loose_excitable
    }
    
    return strict_excitable or loose_excitable, info

def analyze_standard_parameters(results_dir):
    """
    Analysiert die Standardparameter und bestimmt, ob sie zu einem erregbaren System führen.
    
    Args:
        results_dir: Verzeichnis für Ergebnisse
        
    Returns:
        dict: Informationen zum Standardparametersatz
    """
    # Standardparameter laden
    params = comp_model.default_params()
    
    print("\n=== ANALYSE DER STANDARDPARAMETER ===")
    print(f"Standardparameter: bs={params['bs']}, bk={params['bk']}")
    
    # Prüfen, ob das System erregbar ist
    is_exc, info = is_excitable(params, verbose=True)
    
    if is_exc:
        print("Das Standardsystem ist ERREGBAR!")
    else:
        print("Das Standardsystem ist NICHT erregbar nach unseren Kriterien.")
    
    # Visualisierung der Standardparameter
    K_range = np.linspace(0, 1.5, 300)
    S_range = np.linspace(0, 12.0, 300)  # Höherer Bereich für ComS
    
    fig, ax = comp_model.plot_phase_diagram(
        params, K_range, S_range,
        "Phasendiagramm des Standardsystems",
        fixed_points=info['fixed_points']
    )
    
    plt.savefig(os.path.join(results_dir, 'standard_phase_diagram.png'), dpi=300)
    plt.close()
    
    # Zeitreihe simulieren
    t, K, S = comp_model.simulate_system(params, initial_conditions=[0.01, 0.5], t_max=200)
    fig, ax, comp_periods = comp_model.plot_time_series(
        t, K, S, threshold=0.5,
        title="Zeitreihe des Standardsystems"
    )
    
    plt.savefig(os.path.join(results_dir, 'standard_time_series.png'), dpi=300)
    plt.close()
    
    return {'params': params, 'is_excitable': is_exc, 'info': info}

def perform_suel_search(results_dir, n_points=100):
    """
    Führt eine hochauflösende Suche im Suel et al. Parameterbereich durch.
    
    Args:
        results_dir: Verzeichnis für Ergebnisse
        n_points: Anzahl der Punkte pro Dimension (höhere Auflösung)
        
    Returns:
        list: Liste erregbarer Konfigurationen
    """
    print("\n=== HOCHAUFLÖSENDE SUCHE IM SUEL ET AL. BEREICH ===")
    
    # Standardparameter laden
    std_params = comp_model.default_params()
    
    # Suel et al. Parameterbereich
    bs_range = np.linspace(0.6, 1.0, n_points)  # Bereich aus Suel et al.
    bk_range = np.linspace(0.05, 0.1, n_points)  # Bereich aus Suel et al.
    
    print(f"Suchbereich: bs=[{min(bs_range):.4f}, {max(bs_range):.4f}], bk=[{min(bk_range):.4f}, {max(bk_range):.4f}]")
    print(f"Auflösung: {n_points}x{n_points} Punkte (insgesamt {n_points*n_points} Parameterkombinationen)")
    
    # Liste für erregbare Konfigurationen
    excitable_configs = []
    
    # Fortschrittsanzeige einrichten
    total_points = len(bs_range) * len(bk_range)
    progress_interval = max(1, total_points // 20)  # 5% Schritte
    points_checked = 0
    excitable_count = 0
    
    # Gittersuche im definierten Bereich
    for i, bs in enumerate(bs_range):
        for j, bk in enumerate(bk_range):
            # Parameter-Set erstellen
            params = std_params.copy()
            params['bs'] = bs
            params['bk'] = bk
            
            # Prüfen, ob System erregbar ist
            is_exc, info = is_excitable(params)
            
            if is_exc:
                excitable_count += 1
                excitable_configs.append({
                    'bs': bs,
                    'bk': bk,
                    'params': params,
                    'fixed_points': info['fixed_points'],
                    'fp_types': info['fp_types'],
                    'index': (i, j)  # Speichern der Grid-Position für spätere Analyse
                })
            
            # Fortschritt aktualisieren
            points_checked += 1
            if points_checked % progress_interval == 0:
                progress_percent = points_checked / total_points * 100
                print(f"Fortschritt: {progress_percent:.1f}% ({excitable_count} erregbare Konfigurationen gefunden)")
    
    print(f"\nParametersuche abgeschlossen:")
    print(f"  Überprüfte Parameterkombinationen: {points_checked}")
    print(f"  Gefundene erregbare Konfigurationen: {excitable_count}")
    
    # Ergebnisse speichern
    with open(os.path.join(results_dir, 'excitable_configs.pkl'), 'wb') as f:
        pickle.dump(excitable_configs, f)
    
    # Visualisierung der erregbaren Konfigurationen
    visualize_search_results(excitable_configs, bs_range, bk_range, results_dir)
    
    return excitable_configs

def visualize_search_results(excitable_configs, bs_range, bk_range, results_dir):
    """
    Visualisiert die Suchergebnisse im Parameterraum.
    
    Args:
        excitable_configs: Liste erregbarer Konfigurationen
        bs_range: Bereich der bs-Werte
        bk_range: Bereich der bk-Werte
        results_dir: Verzeichnis für Ergebnisse
    """
    # Verzeichnis für Nullklinen erstellen
    nullclines_dir = os.path.join(results_dir, 'example_configs')
    os.makedirs(nullclines_dir, exist_ok=True)
    
    # Karte des Parameterraums erstellen
    plt.figure(figsize=(12, 10))
    
    # Excitable-Regionen im Parameterraum anzeigen
    if excitable_configs:
        bs_values = [config['bs'] for config in excitable_configs]
        bk_values = [config['bk'] for config in excitable_configs]
        plt.scatter(bs_values, bk_values, c='red', s=50, alpha=0.7, label='Erregbare Konfigurationen')
    
    # Standardwerte markieren
    std_params = comp_model.default_params()
    plt.scatter([std_params['bs']], [std_params['bk']], c='blue', s=200, marker='*', label='Standard')
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Erregbare Konfigurationen im Suel et al. Parameterbereich')
    plt.xlim(min(bs_range), max(bs_range))
    plt.ylim(min(bk_range), max(bk_range))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'excitable_map.png'), dpi=300)
    plt.close()
    
    # Einige Beispiel-Nullklinen visualisieren
    if excitable_configs:
        # Wähle eine repräsentative Teilmenge der Konfigurationen
        num_examples = min(5, len(excitable_configs))
        indices = np.linspace(0, len(excitable_configs)-1, num_examples).astype(int)
        
        for i, idx in enumerate(indices):
            config = excitable_configs[idx]
            bs = config['bs']
            bk = config['bk']
            params = config['params']
            
            # Nullklinen-Diagramm erstellen
            fig, ax = comp_model.plot_example_nullclines(
                params, 
                f"Nullklinen für bs={bs:.4f}, bk={bk:.4f}"
            )
            plt.savefig(os.path.join(nullclines_dir, f'nullclines_example_{i+1}.png'), dpi=300)
            plt.close()
            
            # Phasendiagramm mit Vektorfeld erstellen
            fig, ax = comp_model.plot_phase_diagram(
                params, 
                np.linspace(0, 1.0, 200), 
                np.linspace(0, 1.5, 200),
                f"Phasendiagramm für bs={bs:.4f}, bk={bk:.4f}",
                fixed_points=config['fixed_points'],
                show_vector_field=True
            )
            plt.savefig(os.path.join(nullclines_dir, f'phase_diagram_example_{i+1}.png'), dpi=300)
            plt.close()
            
            # Zeitreihen simulieren
            t, K, S = comp_model.simulate_system(params, initial_conditions=[0.01, 0.5], t_max=200)
            fig, ax, comp_periods = comp_model.plot_time_series(
                t, K, S, threshold=0.5,
                title=f"Zeitreihe für bs={bs:.4f}, bk={bk:.4f}"
            )
            plt.savefig(os.path.join(nullclines_dir, f'time_series_example_{i+1}.png'))
            plt.close()
    
    # Parameter-Verteilungen speichern
    if excitable_configs:
        with open(os.path.join(results_dir, 'excitable_params.txt'), 'w') as f:
            f.write("Erregbare Parameter-Konfigurationen im Suel et al. Bereich\n")
            f.write("===================================================\n\n")
            
            f.write(f"Insgesamt gefundene Konfigurationen: {len(excitable_configs)}\n\n")
            
            bs_values = [config['bs'] for config in excitable_configs]
            bk_values = [config['bk'] for config in excitable_configs]
            
            f.write(f"bs-Bereich: [{min(bs_values):.4f}, {max(bs_values):.4f}], Mittelwert: {np.mean(bs_values):.4f}\n")
            f.write(f"bk-Bereich: [{min(bk_values):.4f}, {max(bk_values):.4f}], Mittelwert: {np.mean(bk_values):.4f}\n\n")
            
            f.write("Einzelne Parameter-Konfigurationen:\n")
            for i, config in enumerate(excitable_configs):
                if i < 20:  # Beschränke die Anzahl auf 20 für bessere Lesbarkeit
                    f.write(f"Konfiguration {i+1}:\n")
                    f.write(f"  bs = {config['bs']:.6f}, bk = {config['bk']:.6f}\n")
                    f.write(f"  Fixpunkte: {len(config['fixed_points'])}\n")
                    for j, (fp, fp_type) in enumerate(zip(config['fixed_points'], config['fp_types'])):
                        f.write(f"    FP{j+1}: ({fp[0]:.4f}, {fp[1]:.4f}) - {fp_type}\n")
                    f.write("\n")
            
            if len(excitable_configs) > 20:
                f.write(f"... und {len(excitable_configs) - 20} weitere Konfigurationen.")

def select_representative_configs(excitable_configs, results_dir, num_configs=5):
    """
    Wählt repräsentative erregbare Konfigurationen für weitere Analyse aus.
    
    Args:
        excitable_configs: Liste erregbarer Konfigurationen
        results_dir: Verzeichnis für Ergebnisse
        num_configs: Anzahl der auszuwählenden Konfigurationen
        
    Returns:
        list: Liste ausgewählter Konfigurationen
    """
    # Standardparameter laden
    std_params = comp_model.default_params()
    
    # Standard-Konfiguration erstellen
    standard_config = {
        'bs': std_params['bs'],
        'bk': std_params['bk'],
        'params': std_params.copy(),
        'name': 'Standard'
    }
    
    # Liste ausgewählter Konfigurationen
    selected_configs = [standard_config]
    
    if not excitable_configs:
        print("Keine erregbaren Konfigurationen zum Auswählen vorhanden.")
        return selected_configs
    
    # Wenn wir zu wenige Konfigurationen haben, nehmen wir alle
    if len(excitable_configs) <= num_configs:
        for i, config in enumerate(excitable_configs):
            selected_configs.append({
                'bs': config['bs'],
                'bk': config['bk'],
                'params': config['params'].copy(),
                'name': f'Erregbar {i+1}'
            })
        return selected_configs
    
    # bs- und bk-Werte aller erregbaren Konfigurationen
    all_bs = np.array([config['bs'] for config in excitable_configs])
    all_bk = np.array([config['bk'] for config in excitable_configs])
    
    # Minimum und Maximum für bs und bk
    bs_min, bs_max = np.min(all_bs), np.max(all_bs)
    bk_min, bk_max = np.min(all_bk), np.max(all_bk)
    
    # Diverse Konfigurationen auswählen
    
    # Konfiguration mit bs nahe dem Minimum
    idx_min_bs = np.argmin(all_bs)
    selected_configs.append({
        'bs': excitable_configs[idx_min_bs]['bs'],
        'bk': excitable_configs[idx_min_bs]['bk'],
        'params': excitable_configs[idx_min_bs]['params'].copy(),
        'name': 'Niedrige ComS-Expression'
    })
    
    # Konfiguration mit bs nahe dem Maximum
    idx_max_bs = np.argmax(all_bs)
    selected_configs.append({
        'bs': excitable_configs[idx_max_bs]['bs'],
        'bk': excitable_configs[idx_max_bs]['bk'],
        'params': excitable_configs[idx_max_bs]['params'].copy(),
        'name': 'Hohe ComS-Expression'
    })
    
    # Konfiguration mit bk nahe dem Minimum
    idx_min_bk = np.argmin(all_bk)
    selected_configs.append({
        'bs': excitable_configs[idx_min_bk]['bs'],
        'bk': excitable_configs[idx_min_bk]['bk'],
        'params': excitable_configs[idx_min_bk]['params'].copy(),
        'name': 'Schwaches ComK-Feedback'
    })
    
    # Konfiguration mit bk nahe dem Maximum
    idx_max_bk = np.argmax(all_bk)
    selected_configs.append({
        'bs': excitable_configs[idx_max_bk]['bs'],
        'bk': excitable_configs[idx_max_bk]['bk'],
        'params': excitable_configs[idx_max_bk]['params'].copy(),
        'name': 'Starkes ComK-Feedback'
    })
    
    # Konfigurationen visualisieren
    plt.figure(figsize=(10, 8))
    
    # Alle erregbaren Konfigurationen
    plt.scatter(all_bs, all_bk, c='lightgray', s=30, alpha=0.5, label='Alle erregbaren Konfigurationen')
    
    # Ausgewählte Konfigurationen
    for config in selected_configs:
        plt.scatter([config['bs']], [config['bk']], s=100, label=config['name'])
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Ausgewählte repräsentative Konfigurationen')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'selected_configs.png'), dpi=300)
    plt.close()
    
    # In Textdatei speichern
    with open(os.path.join(results_dir, 'selected_configs.txt'), 'w') as f:
        f.write("Ausgewählte repräsentative Konfigurationen\n")
        f.write("=========================================\n\n")
        
        for config in selected_configs:
            f.write(f"{config['name']}:\n")
            f.write(f"  bs = {config['bs']:.6f}, bk = {config['bk']:.6f}\n\n")
    
    return selected_configs

def run_stochastic_analysis(selected_configs, results_dir):
    """
    Führt stochastische Simulationen für die ausgewählten Konfigurationen durch.
    
    Args:
        selected_configs: Liste ausgewählter Konfigurationen
        results_dir: Verzeichnis für Ergebnisse
    """
    # Parameter-Sets für stochastische Analyse vorbereiten
    param_sets = [config['params'] for config in selected_configs]
    param_names = [config['name'] for config in selected_configs]
    
    # Noise-Parameter
    noise_params = {
        'theta': 1.0,    # Rückkehrrate zum Mittelwert
        'mu': 0.0,       # Mittelwert (Rauschen ist um Null zentriert)
        'sigma': 0.4,    # Rauschstärke 
        'dt': 0.1        # Zeitschritt für numerische Integration
    }
    
    # Unterverzeichnis für stochastische Simulationen
    stochastic_dir = os.path.join(results_dir, 'stochastic_simulations')
    os.makedirs(stochastic_dir, exist_ok=True)
    
    # Stochastische Simulationen durchführen
    stochastic_results = analyze_stochastic_competence(
        comp_model.model_odes,
        param_sets,
        noise_params,
        stochastic_dir,
        n_simulations=20,     # Anzahl der Simulationen pro Parameter-Set
        t_max=500,           # Maximale Simulationszeit
        dt=0.01,             # Zeitschritt
        threshold=0.5,        # Kompetenzschwelle für dimensionslose Parameter
        param_names=param_names  # Namen der Parameter-Sets
    )
    
    # Ergebnisse speichern
    with open(os.path.join(stochastic_dir, 'stochastic_results.pkl'), 'wb') as f:
        pickle.dump(stochastic_results, f)
    
    return stochastic_results

def analyze_parameter_effects(selected_configs, stochastic_results, results_dir):
    """
    Analysiert die Auswirkungen der Parameter bs und bk auf verschiedene Eigenschaften.
    
    Args:
        selected_configs: Liste ausgewählter Konfigurationen
        stochastic_results: Ergebnisse der stochastischen Simulationen
        results_dir: Verzeichnis für Ergebnisse
    """
    # Wenn keine Ergebnisse vorhanden sind, abbrechen
    if not stochastic_results:
        print("Keine stochastischen Ergebnisse für die Analyse vorhanden.")
        return
    
    # Unterverzeichnis für die Analyse
    analysis_dir = os.path.join(results_dir, 'parameter_effects')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Daten aus den Ergebnissen extrahieren
    bs_values = []
    bk_values = []
    mean_durations = []
    cv_durations = []
    mean_rise_times = []
    cv_rise_times = []
    init_probabilities = []
    names = []
    
    for i, config in enumerate(selected_configs):
        param_id = f"param_set_{i+1}"
        if param_id in stochastic_results:
            result = stochastic_results[param_id]
            
            bs_values.append(config['bs'])
            bk_values.append(config['bk'])
            names.append(config['name'])
            
            mean_durations.append(result['mean_duration'])
            cv_durations.append(result['cv_duration'])
            mean_rise_times.append(result['mean_rise_time'])
            cv_rise_times.append(result['cv_rise_time'])
            init_probabilities.append(result['init_probability'])
    
    # Parameter-Einflüsse visualisieren
    
    # Durations vs bs und bk
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bs_values, bk_values, 
               c=mean_durations, s=100, cmap='plasma', alpha=0.8,
               vmin=min(mean_durations) * 0.8 if mean_durations else 0, 
               vmax=max(mean_durations) * 1.2 if mean_durations else 1)
    
    plt.colorbar(scatter, label='Mittlere Kompetenzdauer')
    
    # Labels für jeden Punkt hinzufügen
    for i, name in enumerate(names):
        plt.annotate(name, (bs_values[i], bk_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Einfluss von bs und bk auf die Kompetenzdauer')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, 'bs_bk_vs_duration.png'), dpi=300)
    plt.close()
    
    # Rise time vs bs und bk
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bs_values, bk_values, 
               c=mean_rise_times, s=100, cmap='viridis', alpha=0.8,
               vmin=min(mean_rise_times) * 0.8 if mean_rise_times else 0, 
               vmax=max(mean_rise_times) * 1.2 if mean_rise_times else 1)
    
    plt.colorbar(scatter, label='Mittlere Anstiegszeit')
    
    # Labels für jeden Punkt hinzufügen
    for i, name in enumerate(names):
        plt.annotate(name, (bs_values[i], bk_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Einfluss von bs und bk auf die Anstiegszeit')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, 'bs_bk_vs_rise_time.png'), dpi=300)
    plt.close()
    
    # Initiation probability vs bs und bk
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(bs_values, bk_values, 
               c=init_probabilities, s=100, cmap='YlOrRd', alpha=0.8,
               vmin=min(init_probabilities) * 0.8 if init_probabilities else 0, 
               vmax=max(init_probabilities) * 1.2 if init_probabilities else 1)
    
    plt.colorbar(scatter, label='Wahrscheinlichkeit der Initiierung')
    
    # Labels für jeden Punkt hinzufügen
    for i, name in enumerate(names):
        plt.annotate(name, (bs_values[i], bk_values[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('bs (ComS Expressionsrate)')
    plt.ylabel('bk (ComK Feedback-Stärke)')
    plt.title('Einfluss von bs und bk auf die Initiierungswahrscheinlichkeit')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(analysis_dir, 'bs_bk_vs_initiation.png'), dpi=300)
    plt.close()
    
    # Einzelne Parameter-Einflüsse analysieren
    if len(bs_values) > 1:
        # bs vs Kompetenzdauer
        plt.figure(figsize=(10, 6))
        plt.scatter(bs_values, mean_durations, s=80, c='blue', alpha=0.7)
        
        # Trendlinie hinzufügen
        z = np.polyfit(bs_values, mean_durations, 1)
        p = np.poly1d(z)
        bs_range = np.linspace(min(bs_values), max(bs_values), 100)
        plt.plot(bs_range, p(bs_range), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Labels für jeden Punkt hinzufügen
        for i, name in enumerate(names):
            plt.annotate(name, (bs_values[i], mean_durations[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('bs (ComS Expressionsrate)')
        plt.ylabel('Mittlere Kompetenzdauer')
        plt.title('Einfluss der ComS Expressionsrate auf die Kompetenzdauer')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'bs_vs_duration.png'))
        plt.close()
        
        # bk vs Anstiegszeit
        plt.figure(figsize=(10, 6))
        plt.scatter(bk_values, mean_rise_times, s=80, c='green', alpha=0.7)
        
        # Trendlinie hinzufügen
        z = np.polyfit(bk_values, mean_rise_times, 1)
        p = np.poly1d(z)
        bk_range = np.linspace(min(bk_values), max(bk_values), 100)
        plt.plot(bk_range, p(bk_range), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Labels für jeden Punkt hinzufügen
        for i, name in enumerate(names):
            plt.annotate(name, (bk_values[i], mean_rise_times[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('bk (ComK Feedback-Stärke)')
        plt.ylabel('Mittlere Anstiegszeit')
        plt.title('Einfluss der ComK Feedback-Stärke auf die Anstiegszeit')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'bk_vs_rise_time.png'))
        plt.close()
        
        # bk vs Initiierung
        plt.figure(figsize=(10, 6))
        plt.scatter(bk_values, init_probabilities, s=80, c='red', alpha=0.7)
        
        # Trendlinie hinzufügen
        z = np.polyfit(bk_values, init_probabilities, 1)
        p = np.poly1d(z)
        bk_range = np.linspace(min(bk_values), max(bk_values), 100)
        plt.plot(bk_range, p(bk_range), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        
        # Labels für jeden Punkt hinzufügen
        for i, name in enumerate(names):
            plt.annotate(name, (bk_values[i], init_probabilities[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('bk (ComK Feedback-Stärke)')
        plt.ylabel('Initiierungswahrscheinlichkeit')
        plt.title('Einfluss der ComK Feedback-Stärke auf die Initiierungswahrscheinlichkeit')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(analysis_dir, 'bk_vs_initiation.png'))
        plt.close()
    
    # Statistik in Textdatei speichern
    with open(os.path.join(analysis_dir, 'parameter_effects_summary.txt'), 'w') as f:
        f.write("Zusammenfassung der Parametereffekte\n")
        f.write("==================================\n\n")
        
        f.write("Name\tbs\tbk\tKompetenzdauer\tAnstiegszeit\tInitiierung\n")
        for i, name in enumerate(names):
            f.write(f"{name}\t{bs_values[i]:.4f}\t{bk_values[i]:.4f}\t")
            f.write(f"{mean_durations[i]:.2f}\t{mean_rise_times[i]:.2f}\t")
            f.write(f"{init_probabilities[i]:.4f}\n")
        
        # Korrelationskoeffizienten
        if len(bs_values) > 1:
            f.write("\nKorrelationskoeffizienten:\n")
            corr_bs_duration = np.corrcoef(bs_values, mean_durations)[0, 1]
            corr_bs_rise = np.corrcoef(bs_values, mean_rise_times)[0, 1]
            corr_bs_init = np.corrcoef(bs_values, init_probabilities)[0, 1]
            
            corr_bk_duration = np.corrcoef(bk_values, mean_durations)[0, 1]
            corr_bk_rise = np.corrcoef(bk_values, mean_rise_times)[0, 1]
            corr_bk_init = np.corrcoef(bk_values, init_probabilities)[0, 1]
            
            f.write(f"bs vs Kompetenzdauer: {corr_bs_duration:.4f}\n")
            f.write(f"bs vs Anstiegszeit: {corr_bs_rise:.4f}\n")
            f.write(f"bs vs Initiierung: {corr_bs_init:.4f}\n\n")
            
            f.write(f"bk vs Kompetenzdauer: {corr_bk_duration:.4f}\n")
            f.write(f"bk vs Anstiegszeit: {corr_bk_rise:.4f}\n")
            f.write(f"bk vs Initiierung: {corr_bk_init:.4f}\n")

def main():
    """
    Hauptfunktion für die optimierte Suche im Suel et al. Parameterbereich.
    """
    # Ergebnisverzeichnis erstellen
    results_dir = create_results_directory()
    print(f"Alle Ergebnisse werden im Verzeichnis '{results_dir}' gespeichert")
    
    # 1. Standardparameter analysieren
    std_info = analyze_standard_parameters(results_dir)
    
    # 2. Hochauflösende Suche im Suel et al. Bereich
    excitable_configs = perform_suel_search(results_dir, n_points=100)  # 100x100 = 10.000 Punkte
    
    # Wenn keine erregbaren Konfigurationen gefunden wurden, beenden
    if not excitable_configs:
        print("Keine erregbaren Konfigurationen gefunden. Analyse beendet.")
        return
    
    # 3. Repräsentative Konfigurationen auswählen
    selected_configs = select_representative_configs(excitable_configs, results_dir, num_configs=5)
    
    # 4. Stochastische Simulationen durchführen
    stochastic_results = run_stochastic_analysis(selected_configs, results_dir)
    
    # 5. Parametereffekte analysieren
    analyze_parameter_effects(selected_configs, stochastic_results, results_dir)
    
    print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert in: {results_dir}")

if __name__ == "__main__":
    main()