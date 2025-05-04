"""
Nullklinen-Visualisierung für das B. subtilis Kompetenzmodell.

Dieses Modul bietet Funktionen zur Visualisierung von Nullklinen und Phasenraumdiagrammen
für verschiedene Parameterkonfigurationen des B. subtilis Kompetenzmodells.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import competence_circuit_analysis as comp_model

def plot_nullclines(params, fixed_points, output_path, 
                   title=None, K_range=None, S_range=None):
    """
    Zeichnet die Nullklinen für einen gegebenen Parametersatz.
    
    Args:
        params: Parametersatz für das Kompetenzmodell
        fixed_points: Liste von Fixpunkten
        output_path: Pfad zum Speichern der Abbildung
        title: Optional, Titel für die Abbildung
        K_range: Optional, Bereich für ComK-Achse
        S_range: Optional, Bereich für ComS-Achse
    """
    # Benutze immer 0.0 als Startpunkt für die K-Achse
    if K_range is None:
        K_range = [0.0, 1.0]
    
    # Erstelle ein feines Gitter für K
    K_grid = np.linspace(K_range[0], K_range[1], 500)
    
    # Berechne Nullklinen
    S_from_K, S_from_S = comp_model.nullclines(K_grid, params)
    
    # Entferne ungültige Werte (negativ oder unendlich)
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    # Bestimme S-Achsenbereich basierend auf den Nullklinen und Fixpunkten
    if S_range is None:
        max_s_nullcline = max(np.max(S_from_K[valid_K]) if np.any(valid_K) else 0,
                           np.max(S_from_S[valid_S]) if np.any(valid_S) else 0)
        max_s_fixpoints = max([fp[1] for fp in fixed_points]) if fixed_points else 0
        max_s = max(max_s_nullcline, max_s_fixpoints) * 1.2
        S_range = [0.0, max_s]
    
    # Erstelle Figure und Achsen
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Zeichne Nullklinen mit den Standardfarben
    ax.plot(K_grid[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullkline', linewidth=2)
    ax.plot(K_grid[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullkline', linewidth=2)
    
    # Zeichne Fixpunkte
    for i, fp in enumerate(fixed_points):
        K, S = fp
        fp_type = comp_model.classify_fixed_point(K, S, params)
        
        # Definiere Farben und Marker für Fixpunkte
        if 'Stabil' in fp_type:
            color = 'g'
            label = f'FP{i+1}: Stabiler Knoten ({K:.3f}, {S:.3f})'
        elif 'Sattel' in fp_type:
            color = 'y'
            label = f'FP{i+1}: Sattelpunkt ({K:.3f}, {S:.3f})'
        elif 'Instabil' in fp_type:
            color = 'r'
            label = f'FP{i+1}: Instabiler Knoten ({K:.3f}, {S:.3f})'
        else:
            color = 'gray'
            label = f'FP{i+1}: {fp_type} ({K:.3f}, {S:.3f})'
        
        ax.plot(K, S, 'o', color=color, markersize=10, label=label)
    
    # Achsenbereiche explizit setzen
    ax.set_xlim(K_range)
    ax.set_ylim(S_range)
    
    # Beschriftungen
    ax.set_xlabel('ComK Konzentration')
    ax.set_ylabel('ComS Konzentration')
    
    # Verwende Standard-Titel mit Hill-Koeffizienten und Parameterwerten
    if title is None:
        n = params.get('n', 2)
        p = params.get('p', 5)
        bs = params.get('bs', 0.82)
        bk = params.get('bk', 0.07)
        title = f"Nullklinen für n={n}, p={p}, bs={bs:.4f}, bk={bk:.4f}"
    
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Diagramm speichern
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return fig, ax

def create_nullcline_diagrams(all_results, results_dir):
    """
    Erstellt Nullklinen-Diagramme für alle erregbaren Konfigurationen aus den Hill-Koeffizienten-Analysen.
    
    Args:
        all_results: Dictionary mit Ergebnissen der Hill-Koeffizienten-Analyse
        results_dir: Hauptverzeichnis für Ergebnisse
    """
    print("\n=== ERSTELLE NULLKLINEN-DIAGRAMME ===")
    
    # Erstelle Unterverzeichnis für Nullklinen
    nullclines_dir = os.path.join(results_dir, 'nullclines')
    os.makedirs(nullclines_dir, exist_ok=True)
    
    # Für jede Hill-Koeffizienten-Kombination
    for key in all_results:
        # Extrahiere n und p aus dem Schlüssel
        n, p = map(int, key.replace('n', '').replace('p', '').split('_'))
        result = all_results[key]
        
        # Erstelle Unterverzeichnis für diese Hill-Koeffizienten-Kombination
        hill_dir = os.path.join(nullclines_dir, f'n{n}_p{p}')
        os.makedirs(hill_dir, exist_ok=True)
        
        # Verarbeite Standard-Bereich
        if result['excitable_count'] > 0:
            process_excitable_configs(result['excitable_configs'], n, p, hill_dir, 'standard')
        
        # Verarbeite erweiterten Bereich
        if 'has_extended_search' in result and result['extended_excitable_count'] > 0:
            process_excitable_configs(result['extended_excitable_configs'], n, p, hill_dir, 'extended')
    
    print("  Nullklinen-Diagramme erstellt")

def process_excitable_configs(configs, n, p, output_dir, region_type):
    """
    Verarbeitet eine Liste von erregbaren Konfigurationen und erstellt Nullklinen-Diagramme.
    
    Args:
        configs: Liste der erregbaren Konfigurationen
        n: ComK Hill-Koeffizient
        p: ComS Hill-Koeffizient
        output_dir: Ausgabeverzeichnis
        region_type: Typ des Bereichs ('standard' oder 'extended')
    """
    # Wähle repräsentative Konfigurationen aus
    if len(configs) <= 5:
        # Wenn weniger als 5 Konfigurationen, verwende alle
        selected_configs = configs
    else:
        # Sonst wähle 5 gut verteilte aus
        indices = [0, len(configs)//4, len(configs)//2, 3*len(configs)//4, len(configs)-1]
        selected_configs = [configs[i] for i in indices]
    
    prefix = "extended_" if region_type == "extended" else ""
    region_name = "Erweitert" if region_type == "extended" else "Standard"
    print(f"  Zeichne {len(selected_configs)} Nullklinen-Diagramme für n={n}, p={p} ({region_name})")
    
    # Erstelle ein Nullklinen-Diagramm für jede ausgewählte Konfiguration
    for i, config in enumerate(selected_configs):
        params = config['params']
        bs = config['bs']
        bk = config['bk']
        fixed_points = config['fixed_points']
        
        # Diagramm-Titel
        title = f"Nullklinen für n={n}, p={p}, bs={bs:.4f}, bk={bk:.4f}"
        if region_type == "extended":
            title += " (Erweitert)"
        
        # Ausgabepfad
        output_path = os.path.join(output_dir, f'{prefix}nullclines_{i+1}_bs{bs:.4f}_bk{bk:.4f}.png')
        
        # Nullklinen zeichnen - K-Bereich immer bei 0.0 starten
        plot_nullclines(
            params,
            fixed_points,
            output_path,
            title=title,
            K_range=[0.0, 1.0]  # Explizit 0.0 als Startpunkt
        )