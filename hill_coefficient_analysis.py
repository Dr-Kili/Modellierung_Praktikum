"""
Hill-Koeffizienten-Analyse für das B. subtilis Kompetenzmodell mit direkter Erfassung der Grenzwerte.

Dieses Skript untersucht die Robustheit des erregbaren Bereichs im bk-bs-Parameterraum
bei verschiedenen Kombinationen von Hill-Koeffizienten (n und p).
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import competence_circuit_analysis as comp_model
import nullcline_visualization as null_vis

def create_results_directory():
    """
    Erstellt ein Ergebnisverzeichnis mit Zeitstempel.
    
    Returns:
        str: Pfad zum erstellten Verzeichnis
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"results_hill_analysis_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def is_excitable(params, K_range=np.linspace(0, 1.5, 300), S_range=np.linspace(0, 1.5, 300), verbose=False):
    """
    Prüft, ob ein Parametersatz zu einem erregbaren System führt.
    
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

def analyze_hill_coefficient_combinations(results_dir, grid_size=50, extended_search=True):
    """
    Analysiert die Robustheit des erregbaren Bereichs bei verschiedenen Hill-Koeffizienten.
    
    Args:
        results_dir: Verzeichnis für Ergebnisse
        grid_size: Auflösung des bk-bs-Gitters
        extended_search: Wenn True, wird ein zweiter, größerer Parameterbereich durchsucht,
                         falls im ersten Bereich keine erregbaren Konfigurationen gefunden werden
        
    Returns:
        dict: Ergebnisse für alle Hill-Koeffizient-Kombinationen
    """
    print("\n=== ANALYSE DER HILL-KOEFFIZIENTEN-KOMBINATIONEN ===")
    
    # Erstelle Unterverzeichnisse für Bilder und Textdateien
    plots_dir = os.path.join(results_dir, 'plots')
    data_dir = os.path.join(results_dir, 'data')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Mögliche Hill-Koeffizienten
    n_values = [2, 3, 4, 5]  # ComK-Aktivierung
    p_values = [2, 3, 4, 5]  # ComS-Repression
    
    # Feinere Parameterbereiche für bk und bs basierend auf dem optimized_search
    # Bereiche sind bewusst eng gewählt, um die kritische Region besser aufzulösen
    bs_range = np.linspace(0.6, 1.0, grid_size)  # ComS-Expressionsrate
    bk_range = np.linspace(0.05, 0.1, grid_size)  # ComK-Feedback-Stärke
    
    # Erweiterte Suchbereiche
    ext_bs_range = np.linspace(0.1, 2.0, grid_size)  # Breiterer bs-Bereich
    ext_bk_range = np.linspace(0.01, 0.3, grid_size)  # Breiterer bk-Bereich
    
    # Standardparameter laden
    std_params = comp_model.default_params()
    
    # Ergebnisse für jede Kombination von Hill-Koeffizienten
    all_results = {}
    
    # Für jede Kombination der Hill-Koeffizienten
    for n in n_values:
        for p in p_values:
            print(f"\nAnalysiere Hill-Koeffizienten: n={n}, p={p}")
            
            # Ergebnis-Dictionary für diese Kombination
            result = {
                'n': n,
                'p': p,
                'excitable_count': 0,
                'excitable_configs': [],
                'excitable_grid': np.zeros((grid_size, grid_size), dtype=bool),
                'bs_min': float('inf'),  # Direkte Erfassung der Minimal-/Maximalwerte
                'bs_max': float('-inf'),
                'bk_min': float('inf'),
                'bk_max': float('-inf')
            }
            
            # Fortschrittsanzeige einrichten
            total_points = grid_size * grid_size
            progress_interval = max(1, total_points // 10)  # 10% Schritte
            points_checked = 0
            
            # Gittersuche im definierten Bereich
            for i, bs in enumerate(bs_range):
                for j, bk in enumerate(bk_range):
                    # Parameter-Set erstellen
                    params = std_params.copy()
                    params['n'] = n
                    params['p'] = p
                    params['bs'] = bs
                    params['bk'] = bk
                    
                    # Prüfen, ob System erregbar ist
                    is_exc, info = is_excitable(params)
                    
                    if is_exc:
                        result['excitable_count'] += 1
                        result['excitable_configs'].append({
                            'bs': bs,
                            'bk': bk,
                            'params': params,
                            'fixed_points': info['fixed_points'],
                            'fp_types': info['fp_types'],
                            'index': (i, j)  # Speichern der Grid-Position
                        })
                        result['excitable_grid'][i, j] = True
                        
                        # Direkte Aktualisierung der Grenzwerte
                        result['bs_min'] = min(result['bs_min'], bs)
                        result['bs_max'] = max(result['bs_max'], bs)
                        result['bk_min'] = min(result['bk_min'], bk)
                        result['bk_max'] = max(result['bk_max'], bk)
                    
                    # Fortschritt aktualisieren
                    points_checked += 1
                    if points_checked % progress_interval == 0:
                        progress_percent = points_checked / total_points * 100
                        print(f"  Fortschritt: {progress_percent:.1f}% ({result['excitable_count']} erregbare Konfigurationen gefunden)")
            
            # Wenn keine erregbaren Konfigurationen gefunden wurden, setze Grenzwerte zurück
            if result['excitable_count'] == 0:
                result['bs_min'] = result['bk_min'] = None
                result['bs_max'] = result['bk_max'] = None
            
            # Erweiterte Suche durchführen (unabhängig davon, ob im Standard-Bereich erregbare Konfigurationen gefunden wurden)
            if extended_search:
                print(f"  Führe erweiterte Suche durch...")
                
                # Grid für erweiterte Suche
                ext_excitable_grid = np.zeros((grid_size, grid_size), dtype=bool)
                ext_result = {
                    'n': n,
                    'p': p,
                    'extended_excitable_count': 0,
                    'extended_excitable_configs': [],
                    'extended_excitable_grid': ext_excitable_grid,
                    'extended_bs_range': ext_bs_range,
                    'extended_bk_range': ext_bk_range,
                    'extended_bs_min': float('inf'),
                    'extended_bs_max': float('-inf'),
                    'extended_bk_min': float('inf'),
                    'extended_bk_max': float('-inf')
                }
                
                # Fortschrittsanzeige zurücksetzen
                points_checked = 0
                
                # Erweiterte Gittersuche
                for i, bs in enumerate(ext_bs_range):
                    for j, bk in enumerate(ext_bk_range):
                        # Parameter-Set erstellen
                        params = std_params.copy()
                        params['n'] = n
                        params['p'] = p
                        params['bs'] = bs
                        params['bk'] = bk
                        
                        # Prüfen, ob System erregbar ist
                        is_exc, info = is_excitable(params)
                        
                        if is_exc:
                            ext_result['extended_excitable_count'] += 1
                            ext_result['extended_excitable_configs'].append({
                                'bs': bs,
                                'bk': bk,
                                'params': params,
                                'fixed_points': info['fixed_points'],
                                'fp_types': info['fp_types'],
                                'index': (i, j),  # Speichern der Grid-Position
                                'extended_search': True  # Markieren als aus erweiterter Suche
                            })
                            ext_excitable_grid[i, j] = True
                            
                            # Direkte Aktualisierung der Grenzwerte für erweiterte Suche
                            ext_result['extended_bs_min'] = min(ext_result['extended_bs_min'], bs)
                            ext_result['extended_bs_max'] = max(ext_result['extended_bs_max'], bs)
                            ext_result['extended_bk_min'] = min(ext_result['extended_bk_min'], bk)
                            ext_result['extended_bk_max'] = max(ext_result['extended_bk_max'], bk)
                        
                        # Fortschritt aktualisieren
                        points_checked += 1
                        if points_checked % (progress_interval * 5) == 0:  # Weniger Updates bei der erweiterten Suche
                            progress_percent = points_checked / total_points * 100
                            print(f"  Erw. Suche Fortschritt: {progress_percent:.1f}% ({ext_result['extended_excitable_count']} erregbare Konfigurationen gefunden)")
                
                # Wenn keine erregbaren Konfigurationen gefunden wurden, setze Grenzwerte zurück
                if ext_result['extended_excitable_count'] == 0:
                    ext_result['extended_bs_min'] = ext_result['extended_bk_min'] = None
                    ext_result['extended_bs_max'] = ext_result['extended_bk_max'] = None
                
                # Erweiterte Ergebnisse in das Hauptergebnis integrieren
                result.update({
                    'has_extended_search': True,
                    'extended_excitable_count': ext_result['extended_excitable_count'],
                    'extended_excitable_configs': ext_result['extended_excitable_configs'],
                    'extended_excitable_grid': ext_result['extended_excitable_grid'],
                    'extended_bs_range': ext_result['extended_bs_range'],
                    'extended_bk_range': ext_result['extended_bk_range'],
                    'extended_bs_min': ext_result['extended_bs_min'],
                    'extended_bs_max': ext_result['extended_bs_max'],
                    'extended_bk_min': ext_result['extended_bk_min'],
                    'extended_bk_max': ext_result['extended_bk_max']
                })
            
            # Visualisierung der Ergebnisse mit einem einzigen Plot 
            # (Standard, wenn vorhanden, sonst erweitert, wenn vorhanden)
            visualize_hill_coefficient_result(n, p, result, bs_range, bk_range, plots_dir, data_dir)
            
            print(f"\nErgebnis für n={n}, p={p}:")
            print(f"  Erregbare Konfigurationen (Standard): {result['excitable_count']} von {total_points} ({result['excitable_count']/total_points*100:.2f}%)")
            
            if result['excitable_count'] > 0:
                print(f"  Standard-Bereich: bs: [{result['bs_min']:.4f}, {result['bs_max']:.4f}], bk: [{result['bk_min']:.4f}, {result['bk_max']:.4f}]")
            
            if 'has_extended_search' in result:
                print(f"  Erregbare Konfigurationen (Erweitert): {result['extended_excitable_count']} von {total_points} ({result['extended_excitable_count']/total_points*100:.2f}%)")
                
                if result['extended_excitable_count'] > 0:
                    print(f"  Erweiterter Bereich: bs: [{result['extended_bs_min']:.4f}, {result['extended_bs_max']:.4f}], bk: [{result['extended_bk_min']:.4f}, {result['extended_bk_max']:.4f}]")
            
            # Ergebnis für diese Kombination speichern
            all_results[f'n{n}_p{p}'] = result
    
    # Vergleichende Visualisierung erstellen
    visualize_hill_coefficient_comparison(all_results, bs_range, bk_range, plots_dir, data_dir)
    
    # Ergebnisse speichern
    with open(os.path.join(data_dir, 'hill_coefficient_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    
    # Zusätzlich eine Textdatei mit den Grenzwerten erstellen
    with open(os.path.join(data_dir, 'boundary_values.txt'), 'w') as f:
        f.write("Grenzen der erregbaren Bereiche für verschiedene Hill-Koeffizienten\n")
        f.write("=================================================================\n\n")
        
        f.write("Standard-Bereich (bs: 0.6-1.0, bk: 0.05-0.1):\n")
        f.write("Kombination | bs_min  | bs_max  | bs_range | bk_min  | bk_max  | bk_range | Fläche   | Anzahl\n")
        f.write("---------------------------------------------------------------------------\n")
        
        for n in n_values:
            for p in p_values:
                key = f'n{n}_p{p}'
                if key in all_results:
                    result = all_results[key]
                    
                    if result['excitable_count'] > 0:
                        bs_min = result['bs_min']
                        bs_max = result['bs_max']
                        bs_range_val = bs_max - bs_min
                        bk_min = result['bk_min']
                        bk_max = result['bk_max']
                        bk_range_val = bk_max - bk_min
                        area = bs_range_val * bk_range_val
                        
                        f.write(f"n={n}, p={p} | {bs_min:.4f} | {bs_max:.4f} | {bs_range_val:.4f} | {bk_min:.4f} | {bk_max:.4f} | {bk_range_val:.4f} | {area:.6f} | {result['excitable_count']}\n")
                    else:
                        f.write(f"n={n}, p={p} | -       | -       | -        | -       | -       | -        | -        | 0\n")
        
        f.write("\n\nErweiterter Bereich (bs: 0.1-2.0, bk: 0.01-0.3):\n")
        f.write("Kombination | bs_min  | bs_max  | bs_range | bk_min  | bk_max  | bk_range | Fläche   | Anzahl\n")
        f.write("---------------------------------------------------------------------------\n")
        
        for n in n_values:
            for p in p_values:
                key = f'n{n}_p{p}'
                if key in all_results and 'has_extended_search' in all_results[key]:
                    result = all_results[key]
                    
                    if result['extended_excitable_count'] > 0:
                        bs_min = result['extended_bs_min']
                        bs_max = result['extended_bs_max']
                        bs_range_val = bs_max - bs_min
                        bk_min = result['extended_bk_min']
                        bk_max = result['extended_bk_max']
                        bk_range_val = bk_max - bk_min
                        area = bs_range_val * bk_range_val
                        
                        f.write(f"n={n}, p={p} | {bs_min:.4f} | {bs_max:.4f} | {bs_range_val:.4f} | {bk_min:.4f} | {bk_max:.4f} | {bk_range_val:.4f} | {area:.6f} | {result['extended_excitable_count']}\n")
                    else:
                        f.write(f"n={n}, p={p} | -       | -       | -        | -       | -       | -        | -        | 0\n")
    
    return all_results

def visualize_hill_coefficient_result(n, p, result, bs_range, bk_range, plots_dir, data_dir):
    """
    Visualisiert den erregbaren Bereich für eine Hill-Koeffizienten-Kombination.
    Erstellt für jede Kombination entsprechende Plots:
    - excitable_region wenn im Standardbereich erregbare Konfigurationen existieren
    - extended_excitable_region wenn im erweiterten Bereich erregbare Konfigurationen existieren
    - no_excitable_region wenn in keinem Bereich erregbare Konfigurationen existieren
    
    Args:
        n: ComK-Hill-Koeffizient
        p: ComS-Hill-Koeffizient
        result: Ergebnis für diese Kombination
        bs_range: Bereich der bs-Werte (Standard)
        bk_range: Bereich der bk-Werte (Standard)
        plots_dir: Verzeichnis für Plotdateien
        data_dir: Verzeichnis für Textdateien
    """
    # Unterverzeichnisse für Hill-Koeffizienten erstellen
    hill_plots_dir = os.path.join(plots_dir, 'hill_coefficient_plots')
    hill_data_dir = os.path.join(data_dir, 'hill_coefficient_data')
    os.makedirs(hill_plots_dir, exist_ok=True)
    os.makedirs(hill_data_dir, exist_ok=True)
    
    # Prüfen, ob im Standard-Bereich erregbare Konfigurationen vorhanden sind
    has_standard_excitable = (result['excitable_count'] > 0 and 
                             'bs_min' in result and 
                             result['bs_min'] is not None)
    
    # Prüfen, ob im erweiterten Bereich erregbare Konfigurationen vorhanden sind
    has_extended_excitable = ('has_extended_search' in result and 
                            result['extended_excitable_count'] > 0 and 
                            'extended_bs_min' in result and 
                            result['extended_bs_min'] is not None)
    
    # 1. Standard-Bereich visualisieren, wenn erregbare Konfigurationen vorhanden sind
    if has_standard_excitable:
        bs_plot_range = [0.6, 1.0]
        bk_plot_range = [0.05, 0.1]
        title_prefix = "Erregbarer Bereich"
        filename_prefix = "excitable_region"
        
        excitable_grid = result['excitable_grid']
        bs_min = result['bs_min']
        bs_max = result['bs_max']
        bk_min = result['bk_min']
        bk_max = result['bk_max']
        count = result['excitable_count']
        
        # Standard-Plot erstellen
        plot_single_region(hill_plots_dir, hill_data_dir, n, p,
                          bs_range, bk_range, excitable_grid,
                          bs_min, bs_max, bk_min, bk_max, count,
                          bs_plot_range, bk_plot_range,
                          title_prefix, filename_prefix, is_extended=False)
    
    # 2. Erweiterten Bereich visualisieren, wenn erregbare Konfigurationen vorhanden sind
    if has_extended_excitable:
        bs_plot_range = [0.1, 2.0]
        bk_plot_range = [0.01, 0.3]
        title_prefix = "Erweiterter erregbarer Bereich"
        filename_prefix = "extended_excitable_region"
        
        excitable_grid = result['extended_excitable_grid']
        bs_min = result['extended_bs_min']
        bs_max = result['extended_bs_max']
        bk_min = result['extended_bk_min']
        bk_max = result['extended_bk_max']
        count = result['extended_excitable_count']
        
        # Erweiterten Plot erstellen
        plot_single_region(hill_plots_dir, hill_data_dir, n, p,
                          result['extended_bs_range'], result['extended_bk_range'], excitable_grid,
                          bs_min, bs_max, bk_min, bk_max, count,
                          bs_plot_range, bk_plot_range,
                          title_prefix, filename_prefix, is_extended=True)
    
    # 3. "Keine erregbaren Konfigurationen" visualisieren, wenn in keinem Bereich welche gefunden wurden
    if not has_standard_excitable and not has_extended_excitable:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'Keine erregbaren Konfigurationen gefunden', 
              ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.xlabel('bk (ComK Feedback-Stärke)')
        plt.ylabel('bs (ComS Expressionsrate)')
        plt.title(f'Hill-Koeffizienten n={n}, p={p}')
        
        # Standard-Achsenbereiche anzeigen
        plt.xlim([0.05, 0.1])
        plt.ylim([0.6, 1.0])
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(hill_plots_dir, f'no_excitable_region_n{n}_p{p}.png'), dpi=300)
        
        # Textdatei mit Hinweis erstellen
        with open(os.path.join(hill_data_dir, f'no_excitable_region_n{n}_p{p}.txt'), 'w') as f:
            f.write(f"Keine erregbaren Konfigurationen für n={n}, p={p}\n")
            f.write("=================================================================\n\n")
            f.write("Weder im Standard-Bereich noch im erweiterten Bereich wurden erregbare Konfigurationen gefunden.\n")
        
        plt.close()

def plot_single_region(hill_plots_dir, hill_data_dir, n, p, 
                      bs_range, bk_range, excitable_grid,
                      bs_min, bs_max, bk_min, bk_max, count,
                      bs_plot_range, bk_plot_range,
                      title_prefix, filename_prefix, is_extended):
    """
    Erstellt einen einzelnen Plot für einen erregbaren Bereich.
    
    Args:
        hill_plots_dir: Verzeichnis für die Plots
        hill_data_dir: Verzeichnis für die Textdateien
        n, p: Hill-Koeffizienten
        bs_range, bk_range: Bereiche der bs- und bk-Werte für die Heatmap
        excitable_grid: Grid der erregbaren Konfigurationen
        bs_min, bs_max, bk_min, bk_max: Grenzwerte des erregbaren Bereichs
        count: Anzahl der erregbaren Konfigurationen
        bs_plot_range, bk_plot_range: Grenzen für die Plotachsen
        title_prefix, filename_prefix: Präfixe für Titel und Dateinamen
        is_extended: True, wenn es sich um den erweiterten Bereich handelt
    """
    # Figur erstellen
    plt.figure(figsize=(10, 8))
    
    # Heatmap des erregbaren Bereichs
    extent = [min(bk_range), max(bk_range), min(bs_range), max(bs_range)]
    plt.imshow(excitable_grid, 
              extent=extent,
              origin='lower', aspect='auto', cmap='Blues', alpha=0.6)
    
    # Min/Max-Linien mit genauen Werten
    plt.axhline(y=bs_min, color='orange', linestyle='--', alpha=0.7, 
               label=f'bs_min = {bs_min:.4f}')
    plt.axhline(y=bs_max, color='orange', linestyle='--', alpha=0.7, 
               label=f'bs_max = {bs_max:.4f}')
    plt.axvline(x=bk_min, color='green', linestyle='--', alpha=0.7, 
               label=f'bk_min = {bk_min:.4f}')
    plt.axvline(x=bk_max, color='green', linestyle='--', alpha=0.7, 
               label=f'bk_max = {bk_max:.4f}')
    
    # Standardparameter markieren
    std_params = comp_model.default_params()
    plt.scatter([std_params['bk']], [std_params['bs']], c='red', s=100, marker='*', label='Standard')
    
    # Berechne die Größe des Bereichs
    bs_range_val = bs_max - bs_min
    bk_range_val = bk_max - bk_min
    area = bs_range_val * bk_range_val
    
    # Beschriftungen
    plt.xlabel('bk (ComK Feedback-Stärke)')
    plt.ylabel('bs (ComS Expressionsrate)')
    plt.title(f'{title_prefix} für n={n}, p={p}\n'
             f'Bereich: bs: [{bs_min:.4f}, {bs_max:.4f}], bk: [{bk_min:.4f}, {bk_max:.4f}]\n'
             f'Fläche: {area:.6f}, Anzahl: {count}')
    
    # Legende außerhalb rechts oben platzieren
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Achsenbereiche setzen
    plt.xlim(bk_plot_range)
    plt.ylim(bs_plot_range)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Speichern der Bilddatei
    plt.savefig(os.path.join(hill_plots_dir, f'{filename_prefix}_n{n}_p{p}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Textdatei mit den Grenzen erstellen
    with open(os.path.join(hill_data_dir, f'{filename_prefix}_n{n}_p{p}.txt'), 'w') as f:
        f.write(f"Grenzen des {title_prefix}s für n={n}, p={p}\n")
        f.write("=================================================================\n\n")
        f.write(f"bs_min: {bs_min:.6f}\n")
        f.write(f"bs_max: {bs_max:.6f}\n")
        f.write(f"bs_range: {bs_range_val:.6f}\n\n")
        f.write(f"bk_min: {bk_min:.6f}\n")
        f.write(f"bk_max: {bk_max:.6f}\n")
        f.write(f"bk_range: {bk_range_val:.6f}\n\n")
        f.write(f"Fläche: {area:.6f}\n")
        f.write(f"Anzahl erregbarer Konfigurationen: {count}\n")
        
        # Zusätzliche Informationen für erweiterten Bereich
        if is_extended:
            f.write("\nHinweis: Diese Ergebnisse stammen aus einer erweiterten Suche im Bereich\n")
            f.write(f"bs: [{bs_plot_range[0]}, {bs_plot_range[1]}], bk: [{bk_plot_range[0]}, {bk_plot_range[1]}]\n")

def visualize_hill_coefficient_comparison(all_results, bs_range, bk_range, plots_dir, data_dir):
    """
    Erstellt vergleichende Visualisierungen für alle Hill-Koeffizienten-Kombinationen,
    unter Berücksichtigung sowohl der Standard- als auch der erweiterten Ergebnisse.
    
    Args:
        all_results: Ergebnisse für alle Kombinationen
        bs_range: Bereich der bs-Werte (Standard)
        bk_range: Bereich der bk-Werte (Standard)
        plots_dir: Verzeichnis für Plotdateien
        data_dir: Verzeichnis für Textdateien
    """
    # Unterverzeichnisse für Vergleiche
    comparison_plots_dir = os.path.join(plots_dir, 'comparisons')
    comparison_data_dir = os.path.join(data_dir, 'comparisons')
    os.makedirs(comparison_plots_dir, exist_ok=True)
    os.makedirs(comparison_data_dir, exist_ok=True)
    
    # 1. Vergleich der Größe des erregbaren Bereichs
    n_values = sorted(set([all_results[key]['n'] for key in all_results]))
    p_values = sorted(set([all_results[key]['p'] for key in all_results]))
    
    # Matrix für die Anzahl erregbarer Konfigurationen im Standardbereich
    excitable_counts = np.zeros((len(n_values), len(p_values)))
    
    # Matrizen für Erweiterungen
    standard_counts = np.zeros((len(n_values), len(p_values)))
    extended_counts = np.zeros((len(n_values), len(p_values)))
    total_counts = np.zeros((len(n_values), len(p_values)))
    
    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            key = f'n{n}_p{p}'
            if key in all_results:
                # Standard-Bereich (für ursprüngliche Visualisierungen)
                excitable_counts[i, j] = all_results[key]['excitable_count']
                
                # Für neue Visualisierungen
                standard_counts[i, j] = all_results[key]['excitable_count']
                
                # Erweiterter Bereich
                if 'has_extended_search' in all_results[key] and 'extended_excitable_count' in all_results[key]:
                    extended_counts[i, j] = all_results[key]['extended_excitable_count']
                
                # Gesamtzahl (entweder Standard, oder wenn kein Standard dann Extended)
                if standard_counts[i, j] > 0:
                    total_counts[i, j] = standard_counts[i, j]
                else:
                    total_counts[i, j] = extended_counts[i, j]
    
    # URSPRÜNGLICHE VISUALISIERUNGEN
    
    # Heatmap der Anzahl erregbarer Konfigurationen (nur Standard-Bereich)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(excitable_counts, cmap='viridis')
    
    # Beschriftungen
    plt.colorbar(im, label='Anzahl erregbarer Konfigurationen')
    plt.xlabel('p (ComS-Repression Hill-Koeffizient)')
    plt.ylabel('n (ComK-Aktivierung Hill-Koeffizient)')
    plt.title('Vergleich der Größe des erregbaren Bereichs')
    
    # Achsenbeschriftungen
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    
    # Werte in die Zellen eintragen
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(excitable_counts[i, j])}", 
                          ha="center", va="center", color="w", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_heatmap.png'), dpi=300)
    plt.close()
    
    # 2. Barplot der Größe des erregbaren Bereichs (nur Standard-Bereich)
    plt.figure(figsize=(12, 6))
    
    # Daten für den Barplot vorbereiten
    x_labels = [f'n={n}, p={p}' for n in n_values for p in p_values]
    counts = [all_results[f'n{n}_p{p}']['excitable_count'] for n in n_values for p in p_values]
    
    # Farbkodierung basierend auf n-Werten
    colors = plt.cm.tab10(np.array([i for i in range(len(n_values)) for _ in range(len(p_values))]) % 10)
    
    # Barplot erstellen
    bars = plt.bar(np.arange(len(x_labels)), counts, color=colors)
    
    # Beschriftungen
    plt.xlabel('Hill-Koeffizienten (n, p)')
    plt.ylabel('Anzahl erregbarer Konfigurationen')
    plt.title('Größe des erregbaren Bereichs für verschiedene Hill-Koeffizienten')
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Legende für n-Werte
    handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(i % 10)) for i in range(len(n_values))]
    labels = [f'n={n}' for n in n_values]
    plt.legend(handles, labels, title='ComK-Aktivierung')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_barplot.png'), dpi=300)
    plt.close()
    
    # 3. Visualisierung der exakten Grenzen in einem Barplot (nur Standard-Bereich)
    plt.figure(figsize=(14, 10))
    
    # Vorbereitung der Daten für die Bereiche
    labels = []
    bs_ranges = []
    bk_ranges = []
    areas = []
    
    for n in n_values:
        for p in p_values:
            key = f'n{n}_p{p}'
            label = f'n={n}, p={p}'
            
            if key in all_results and all_results[key]['excitable_count'] > 0:
                result = all_results[key]
                
                bs_min = result['bs_min']
                bs_max = result['bs_max']
                bk_min = result['bk_min']
                bk_max = result['bk_max']
                
                bs_range_val = bs_max - bs_min
                bk_range_val = bk_max - bk_min
                area = bs_range_val * bk_range_val
                
                labels.append(label)
                bs_ranges.append(bs_range_val)
                bk_ranges.append(bk_range_val)
                areas.append(area)
    
    # Mehrere Subplots erstellen
    plt.subplot(3, 1, 1)
    plt.bar(np.arange(len(labels)), bs_ranges, color='orange')
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('bs_max - bs_min')
    plt.title('Breite des erregbaren Bereichs in bs-Dimension')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.bar(np.arange(len(labels)), bk_ranges, color='green')
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('bk_max - bk_min')
    plt.title('Breite des erregbaren Bereichs in bk-Dimension')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.bar(np.arange(len(labels)), areas, color='blue')
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Fläche')
    plt.title('Gesamtfläche des erregbaren Bereichs')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_ranges_barplot.png'), dpi=300)
    plt.close()
    
    # 4. Phasendiagramm mit direkten Grenzmarkierungen (Standard und Erweitert)
    # Erstellen eines 2x2 Grids für n=2,3 und p=4,5 (falls vorhanden)
    if 2 in n_values and 3 in n_values and 4 in p_values and 5 in p_values:
        plt.figure(figsize=(16, 14))
        
        # Unterplots erstellen
        for i, n in enumerate([2, 3]):
            for j, p in enumerate([4, 5]):
                key = f'n{n}_p{p}'
                
                if key in all_results:
                    # Subplot für diese Kombination
                    ax = plt.subplot(2, 2, i*2+j+1)
                    
                    # Standardpriorität: Verwende Standard-Bereich, wenn vorhanden, sonst erweitert
                    if all_results[key]['excitable_count'] > 0:
                        # Standard-Bereich und Grid verwenden
                        excitable_grid = all_results[key]['excitable_grid']
                        bs_range_to_plot = bs_range
                        bk_range_to_plot = bk_range
                        bs_min = all_results[key]['bs_min']
                        bs_max = all_results[key]['bs_max']
                        bk_min = all_results[key]['bk_min']
                        bk_max = all_results[key]['bk_max']
                        count = all_results[key]['excitable_count']
                        is_standard = True
                    elif ('has_extended_search' in all_results[key] and 
                          all_results[key]['extended_excitable_count'] > 0):
                        # Erweiterten Bereich verwenden
                        bs_range_to_plot = all_results[key]['extended_bs_range']
                        bk_range_to_plot = all_results[key]['extended_bk_range']
                        excitable_grid = all_results[key]['extended_excitable_grid']
                        bs_min = all_results[key]['extended_bs_min']
                        bs_max = all_results[key]['extended_bs_max']
                        bk_min = all_results[key]['extended_bk_min']
                        bk_max = all_results[key]['extended_bk_max']
                        count = all_results[key]['extended_excitable_count']
                        is_standard = False
                    else:
                        # Keine erregbaren Konfigurationen
                        ax.text(0.5, 0.5, 'Keine erregbaren\nKonfigurationen', 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=14)
                        ax.set_title(f'n={n}, p={p}')
                        continue
                    
                    # Heatmap des erregbaren Bereichs
                    extent = [min(bk_range_to_plot), max(bk_range_to_plot), 
                             min(bs_range_to_plot), max(bs_range_to_plot)]
                    ax.imshow(excitable_grid, 
                             extent=extent,
                             origin='lower', aspect='auto', 
                             cmap='Blues' if is_standard else 'Greens', alpha=0.6)
                    
                    # Min/Max-Linien
                    ax.axhline(y=bs_min, color='orange', linestyle='--', alpha=0.7)
                    ax.axhline(y=bs_max, color='orange', linestyle='--', alpha=0.7)
                    ax.axvline(x=bk_min, color='green', linestyle='--', alpha=0.7)
                    ax.axvline(x=bk_max, color='green', linestyle='--', alpha=0.7)
                    
                    # Standardparameter markieren
                    std_params = comp_model.default_params()
                    ax.scatter([std_params['bk']], [std_params['bs']], c='red', s=100, marker='*')
                    
                    # Informationen zur Größe des Bereichs
                    bs_range_val = bs_max - bs_min
                    bk_range_val = bk_max - bk_min
                    area = bs_range_val * bk_range_val
                    
                    # Titel mit Informationen
                    region_type = "Standard" if is_standard else "Erweitert"
                    ax.set_title(f'n={n}, p={p}: {region_type}, {count} Konfigurationen\n'
                               f'bs: [{bs_min:.3f}, {bs_max:.3f}], bk: [{bk_min:.3f}, {bk_max:.3f}]')
                    
                    # Achsenbeschriftungen
                    ax.set_xlabel('bk (ComK Feedback-Stärke)')
                    ax.set_ylabel('bs (ComS Expressionsrate)')
                    ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_plots_dir, 'phase_diagram_with_boundaries.png'), dpi=300)
        plt.close()
    
    # NEUE VISUALISIERUNGEN FÜR STANDARD UND ERWEITERTEN BEREICH
    
    # 5. Heatmaps für Standard, Erweitert und Kombiniert
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(standard_counts, cmap='Blues')
    plt.colorbar(im1, label='Anzahl (Standard-Bereich)')
    plt.xlabel('p (ComS-Repression)')
    plt.ylabel('n (ComK-Aktivierung)')
    plt.title('Standard-Bereich')
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(standard_counts[i, j])}", 
                          ha="center", va="center", color="w" if standard_counts[i, j] > standard_counts.max()/2 else "black", 
                          fontweight='bold')
    
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(extended_counts, cmap='Greens')
    plt.colorbar(im2, label='Anzahl (Erweiterter Bereich)')
    plt.xlabel('p (ComS-Repression)')
    plt.ylabel('n (ComK-Aktivierung)')
    plt.title('Erweiterter Bereich')
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(extended_counts[i, j])}", 
                          ha="center", va="center", color="w" if extended_counts[i, j] > extended_counts.max()/2 else "black", 
                          fontweight='bold')
    
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(total_counts, cmap='viridis')
    plt.colorbar(im3, label='Anzahl (Kombiniert)')
    plt.xlabel('p (ComS-Repression)')
    plt.ylabel('n (ComK-Aktivierung)')
    plt.title('Beste (Standard oder Erweitert)')
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            # Visuelle Markierung für Standard/Erweitert
            marker = "*" if standard_counts[i, j] > 0 else "+" if extended_counts[i, j] > 0 else ""
            text = plt.text(j, i, f"{int(total_counts[i, j])}{marker}", 
                          ha="center", va="center", color="w" if total_counts[i, j] > total_counts.max()/2 else "black", 
                          fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_heatmaps_combined.png'), dpi=300)
    plt.close()
    
    # 6. Barplot der Größe des erregbaren Bereichs für Standard- und erweiterten Bereich
    plt.figure(figsize=(14, 8))
    
    # Daten für den Barplot vorbereiten
    x_labels = [f'n={n}, p={p}' for n in n_values for p in p_values]
    x_pos = np.arange(len(x_labels))
    width = 0.35  # Breite der Balken
    
    # Standard-Bereich
    standard_bars = [all_results[f'n{n}_p{p}'].get('excitable_count', 0) for n in n_values for p in p_values]
    
    # Erweiterter Bereich
    extended_bars = [all_results[f'n{n}_p{p}'].get('extended_excitable_count', 0) if 'has_extended_search' in all_results[f'n{n}_p{p}'] else 0 
                   for n in n_values for p in p_values]
    
    # Farbkodierung basierend auf n-Werten
    colors_standard = plt.cm.Blues(np.linspace(0.5, 0.9, len(n_values)).repeat(len(p_values)))
    colors_extended = plt.cm.Greens(np.linspace(0.5, 0.9, len(n_values)).repeat(len(p_values)))
    
    # Barplot erstellen
    plt.bar(x_pos - width/2, standard_bars, width, color=colors_standard, label='Standard-Bereich')
    plt.bar(x_pos + width/2, extended_bars, width, color=colors_extended, label='Erweiterter Bereich')
    
    # Beschriftungen
    plt.xlabel('Hill-Koeffizienten (n, p)')
    plt.ylabel('Anzahl erregbarer Konfigurationen')
    plt.title('Vergleich der Größe des erregbaren Bereichs für Standard und Erweitert')
    plt.xticks(x_pos, x_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Legende
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_barplot_comparison.png'), dpi=300)
    plt.close()
    
    # 7. Visualisierung der Grenzen in Barplots für Standard und erweiterten Bereich
    plt.figure(figsize=(16, 12))
    
    # Vorbereitung der Daten für die Bereiche
    std_labels = []
    std_bs_ranges = []
    std_bk_ranges = []
    std_areas = []
    
    ext_labels = []
    ext_bs_ranges = []
    ext_bk_ranges = []
    ext_areas = []
    
    for n in n_values:
        for p in p_values:
            key = f'n{n}_p{p}'
            label = f'n={n}, p={p}'
            
            # Standard-Bereich
            if key in all_results and all_results[key]['excitable_count'] > 0:
                result = all_results[key]
                
                bs_min = result['bs_min']
                bs_max = result['bs_max']
                bk_min = result['bk_min']
                bk_max = result['bk_max']
                
                bs_range_val = bs_max - bs_min
                bk_range_val = bk_max - bk_min
                area = bs_range_val * bk_range_val
                
                std_labels.append(label)
                std_bs_ranges.append(bs_range_val)
                std_bk_ranges.append(bk_range_val)
                std_areas.append(area)
            
            # Erweiterter Bereich
            if (key in all_results and 'has_extended_search' in all_results[key] and 
                all_results[key]['extended_excitable_count'] > 0):
                result = all_results[key]
                
                bs_min = result['extended_bs_min']
                bs_max = result['extended_bs_max']
                bk_min = result['extended_bk_min']
                bk_max = result['extended_bk_max']
                
                bs_range_val = bs_max - bs_min
                bk_range_val = bk_max - bk_min
                area = bs_range_val * bk_range_val
                
                ext_labels.append(label)
                ext_bs_ranges.append(bs_range_val)
                ext_bk_ranges.append(bk_range_val)
                ext_areas.append(area)
    
    # Layout für Standard-Bereich
    plt.subplot(2, 3, 1)
    plt.bar(range(len(std_labels)), std_bs_ranges, color='royalblue')
    plt.xticks(range(len(std_labels)), std_labels, rotation=45, ha='right')
    plt.ylabel('bs_max - bs_min')
    plt.title('Standard: bs-Bereich')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.bar(range(len(std_labels)), std_bk_ranges, color='royalblue')
    plt.xticks(range(len(std_labels)), std_labels, rotation=45, ha='right')
    plt.ylabel('bk_max - bk_min')
    plt.title('Standard: bk-Bereich')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.bar(range(len(std_labels)), std_areas, color='royalblue')
    plt.xticks(range(len(std_labels)), std_labels, rotation=45, ha='right')
    plt.ylabel('Fläche')
    plt.title('Standard: Gesamtfläche')
    plt.grid(axis='y', alpha=0.3)
    
    # Layout für erweiterten Bereich
    plt.subplot(2, 3, 4)
    plt.bar(range(len(ext_labels)), ext_bs_ranges, color='green')
    plt.xticks(range(len(ext_labels)), ext_labels, rotation=45, ha='right')
    plt.ylabel('bs_max - bs_min')
    plt.title('Erweitert: bs-Bereich')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.bar(range(len(ext_labels)), ext_bk_ranges, color='green')
    plt.xticks(range(len(ext_labels)), ext_labels, rotation=45, ha='right')
    plt.ylabel('bk_max - bk_min')
    plt.title('Erweitert: bk-Bereich')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.bar(range(len(ext_labels)), ext_areas, color='green')
    plt.xticks(range(len(ext_labels)), ext_labels, rotation=45, ha='right')
    plt.ylabel('Fläche')
    plt.title('Erweitert: Gesamtfläche')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_ranges_barplot_separated.png'), dpi=300)
    plt.close()
    
    # 8. Textdatei mit zusammenfassenden Informationen
    with open(os.path.join(comparison_data_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("Zusammenfassung der Ergebnisse für alle Hill-Koeffizienten-Kombinationen\n")
        f.write("================================================================\n\n")
        
        f.write("Standard-Bereich (bs: 0.6-1.0, bk: 0.05-0.1):\n")
        f.write("Kombination | Anzahl | bs_range | bk_range | Fläche\n")
        f.write("--------------------------------------------------------\n")
        
        for i, n in enumerate(n_values):
            for j, p in enumerate(p_values):
                key = f'n{n}_p{p}'
                if key in all_results:
                    count = all_results[key].get('excitable_count', 0)
                    
                    if count > 0:
                        bs_min = all_results[key]['bs_min']
                        bs_max = all_results[key]['bs_max']
                        bk_min = all_results[key]['bk_min']
                        bk_max = all_results[key]['bk_max']
                        
                        bs_range_val = bs_max - bs_min
                        bk_range_val = bk_max - bk_min
                        area = bs_range_val * bk_range_val
                        
                        f.write(f"n={n}, p={p} | {count:5d} | {bs_range_val:.4f} | {bk_range_val:.4f} | {area:.6f}\n")
                    else:
                        f.write(f"n={n}, p={p} | {0:5d} | -       | -       | -\n")
        
        f.write("\n\nErweiterter Bereich (bs: 0.1-2.0, bk: 0.01-0.3):\n")
        f.write("Kombination | Anzahl | bs_range | bk_range | Fläche\n")
        f.write("--------------------------------------------------------\n")
        
        for i, n in enumerate(n_values):
            for j, p in enumerate(p_values):
                key = f'n{n}_p{p}'
                if key in all_results and 'has_extended_search' in all_results[key]:
                    count = all_results[key].get('extended_excitable_count', 0)
                    
                    if count > 0:
                        bs_min = all_results[key]['extended_bs_min']
                        bs_max = all_results[key]['extended_bs_max']
                        bk_min = all_results[key]['extended_bk_min']
                        bk_max = all_results[key]['extended_bk_max']
                        
                        bs_range_val = bs_max - bs_min
                        bk_range_val = bk_max - bk_min
                        area = bs_range_val * bk_range_val
                        
                        f.write(f"n={n}, p={p} | {count:5d} | {bs_range_val:.4f} | {bk_range_val:.4f} | {area:.6f}\n")
                    else:
                        f.write(f"n={n}, p={p} | {0:5d} | -       | -       | -\n")
    
def analyze_standard_parameters(results_dir):
    """
    Analysiert die Erregbarkeit des Systems mit Standardwerten für bs und bk,
    variiert aber die Hill-Koeffizienten n und p.
    
    Args:
        results_dir: Verzeichnis für Ergebnisse
    """
    print("\n=== ANALYSE DER HILL-KOEFFIZIENTEN BEI STANDARD BS UND BK WERTEN ===")
    
    # Erstelle Unterverzeichnisse
    nullclines_dir = os.path.join(results_dir, 'standard_parameters')
    os.makedirs(nullclines_dir, exist_ok=True)
    
    # Standardparameter laden
    std_params = comp_model.default_params()
    bs = std_params['bs']  # Standardwert 0.82
    bk = std_params['bk']  # Standardwert 0.07
    
    print(f"Feste Parameter: bs={bs:.4f}, bk={bk:.4f}")
    
    # Mögliche Hill-Koeffizienten
    n_values = [2, 3, 4, 5]  # ComK-Aktivierung
    p_values = [2, 3, 4, 5]  # ComS-Repression
    
    # Matrix für Erregbarkeit
    excitable_counts = np.zeros((len(n_values), len(p_values)), dtype=int)
    
    # Für jede Kombination der Hill-Koeffizienten
    for n_idx, n in enumerate(n_values):
        for p_idx, p in enumerate(p_values):
            # Parameter-Set erstellen
            params = std_params.copy()
            params['n'] = n
            params['p'] = p
            
            # Prüfen, ob System erregbar ist
            is_exc, info = is_excitable(params)
            
            # Wenn System erregbar ist, erstelle ein Nullklinen-Diagramm
            if is_exc:
                excitable_counts[n_idx, p_idx] = 1
                print(f"System mit n={n}, p={p} ist erregbar!")
                
                # Nullklinen-Diagramm erstellen
                output_path = os.path.join(nullclines_dir, f'nullclines_n{n}_p{p}.png')
                title = f"Erregbares System: n={n}, p={p}, bs={bs:.4f}, bk={bk:.4f}"
                
                # Nullklinen zeichnen
                null_vis.plot_nullclines(
                    params,
                    info['fixed_points'],
                    output_path,
                    title=title,
                    K_range=[0.0, 1.0]
                )
    
    # Ergebnisse in Textdatei speichern
    with open(os.path.join(nullclines_dir, 'excitability_results.txt'), 'w') as f:
        f.write("Ergebnisse der Erregbarkeitsanalyse bei bs={:.4f}, bk={:.4f}\n".format(bs, bk))
        f.write("=============================================================\n\n")
        f.write("   |")
        for p in p_values:
            f.write(f" p={p} |")
        f.write("\n")
        
        for n_idx, n in enumerate(n_values):
            f.write(f"n={n} |")
            for p_idx, p in enumerate(p_values):
                is_exc = excitable_counts[n_idx, p_idx]
                f.write("  1  |" if is_exc else " 0 |")
            f.write("\n")
        
        # Gesamtanzahl erregbarer Zustände
        total_excitable = np.sum(excitable_counts)
        f.write(f"\nGesamtanzahl erregbarer Zustände: {total_excitable} von {len(n_values)*len(p_values)}\n")
    
    print(f"\nZusammenfassung:")
    print(f"  Anzahl erregbarer Zustände: {np.sum(excitable_counts)}")
    
    return excitable_counts
def find_optimal_hill_coefficients():
    """
    Hauptfunktion für die Hill-Koeffizienten-Analyse.
    """
    # Ergebnisverzeichnis erstellen
    results_dir = create_results_directory()
    print(f"Alle Ergebnisse werden im Verzeichnis '{results_dir}' gespeichert")
    
    # Analyse mit Standardwerten von bs und bk
    print("\nAnalyse mit Standardwerten von bs und bk:")
    analyze_standard_parameters(results_dir)

    # Hill-Koeffizienten analysieren - mit erweiterter Suche falls nötig
    all_results = analyze_hill_coefficient_combinations(results_dir, grid_size=50, extended_search=True)

    # Visualisierung der Nullklinen-Diagramme
    null_vis.create_nullcline_diagrams(all_results, results_dir)
    
    # Optimale Hill-Koeffizienten identifizieren
    n_values = sorted(set([all_results[key]['n'] for key in all_results]))
    p_values = sorted(set([all_results[key]['p'] for key in all_results]))
    
    max_count = 0
    optimal_n = None
    optimal_p = None
    
    for n in n_values:
        for p in p_values:
            key = f'n{n}_p{p}'
            if key in all_results and all_results[key]['excitable_count'] > max_count:
                max_count = all_results[key]['excitable_count']
                optimal_n = n
                optimal_p = p
    
    if optimal_n is not None and optimal_p is not None:
        print(f"\nOptimale Hill-Koeffizienten für maximale Robustheit:")
        print(f"  n (ComK-Aktivierung) = {optimal_n}")
        print(f"  p (ComS-Repression) = {optimal_p}")
        print(f"  Größe des erregbaren Bereichs: {max_count} Konfigurationen")
        
        # Eine zusätzliche Analyse für die optimalen Koeffizienten
        key = f'n{optimal_n}_p{optimal_p}'
        
        if 'extended_search' in all_results[key] and all_results[key]['extended_search']:
            print(f"  Hinweis: Diese Kombination benötigte eine erweiterte Suche.")
            bs_min = all_results[key]['extended_bs_min']
            bs_max = all_results[key]['extended_bs_max']
            bk_min = all_results[key]['extended_bk_min']
            bk_max = all_results[key]['extended_bk_max']
        else:
            bs_min = all_results[key]['bs_min']
            bs_max = all_results[key]['bs_max']
            bk_min = all_results[key]['bk_min']
            bk_max = all_results[key]['bk_max']
        
        bs_range_val = bs_max - bs_min
        bk_range_val = bk_max - bk_min
        area = bs_range_val * bk_range_val
        
        print(f"  Erregbarer Bereich: bs: [{bs_min:.4f}, {bs_max:.4f}], bk: [{bk_min:.4f}, {bk_max:.4f}]")
        print(f"  Breite des Bereichs: bs: {bs_range_val:.4f}, bk: {bk_range_val:.4f}")
        print(f"  Fläche des Bereichs: {area:.6f}")
            
        # Ein paar Beispiel-Konfigurationen anzeigen
        if all_results[key]['excitable_configs']:
            print("\nBeispiel-Parameter für erregbare Systeme mit dieser Kombination:")
            for i, config in enumerate(all_results[key]['excitable_configs'][:3]):  # Zeige bis zu 3 Beispiele
                print(f"  Beispiel {i+1}: bs = {config['bs']:.4f}, bk = {config['bk']:.4f}")
    else:
        print("\nKeine erregbaren Konfigurationen gefunden.")
    
    # Unterverzeichnisse für Plots und Daten
    plots_dir = os.path.join(results_dir, 'plots')
    data_dir = os.path.join(results_dir, 'data')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Ergebnisse als Histogram der erregbaren Konfigurationen pro Hill-Koeffizienten-Kombination darstellen
    plt.figure(figsize=(12, 8))
    bars = []
    bar_labels = []
    
    for n in n_values:
        for p in p_values:
            key = f'n{n}_p{p}'
            if key in all_results:
                bars.append(all_results[key]['excitable_count'])
                bar_labels.append(f'n={n}, p={p}')
    
    plt.bar(range(len(bars)), bars, color='skyblue')
    plt.xticks(range(len(bars)), bar_labels, rotation=45, ha='right')
    plt.xlabel('Hill-Koeffizienten (n, p)')
    plt.ylabel('Anzahl erregbarer Konfigurationen')
    plt.title('Robustheit der erregbaren Dynamik für verschiedene Hill-Koeffizienten')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'hill_coefficients_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\nAnalyse abgeschlossen. Ergebnisse gespeichert in: {results_dir}")

if __name__ == "__main__":
    find_optimal_hill_coefficients()