"""
Parameter space search functions for competence circuit analysis.

This module contains functions for searching parameter spaces to find
excitable configurations of the B. subtilis competence circuit model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import is_excitable, save_results
from visualization import plot_nullclines, plot_excitable_map, plot_parameter_histograms

def search_hill_coefficient_space(model_odes, classify_fixed_point_func, find_fixed_points_func, 
                                results_dir, grid_size=50, extended_search=True):
    """
    Analyzes the robustness of the excitable region for different Hill coefficients.
    
    Args:
        model_odes: Function defining the ODEs
        classify_fixed_point_func: Function to classify fixed points
        find_fixed_points_func: Function to find fixed points
        results_dir: Directory to save results
        grid_size: Resolution of the bk-bs grid
        extended_search: Whether to perform an extended search if no excitable configurations are found
        
    Returns:
        dict: Results for all Hill coefficient combinations
    """
    print("\n=== ANALYZING HILL COEFFICIENT COMBINATIONS ===")
    
    # Create subdirectories for images and text files
    plots_dir = os.path.join(results_dir, 'plots')
    data_dir = os.path.join(results_dir, 'data')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Possible Hill coefficients
    n_values = [2, 3, 4, 5]  # ComK activation
    p_values = [2, 3, 4, 5]  # ComS repression
    
    # Parameter ranges for bk and bs based on refined searching
    bs_range = np.linspace(0.6, 1.0, grid_size)  # ComS expression rate
    bk_range = np.linspace(0.05, 0.1, grid_size)  # ComK feedback strength
    
    # Extended search ranges
    ext_bs_range = np.linspace(0.1, 2.0, grid_size)  # Wider bs range
    ext_bk_range = np.linspace(0.01, 0.3, grid_size)  # Wider bk range
    
    # Load standard parameters
    from competence_circuit_analysis import default_params
    std_params = default_params()
    
    # Results for each combination of Hill coefficients
    all_results = {}
    
    # For each combination of Hill coefficients
    for n in n_values:
        for p in p_values:
            print(f"\nAnalyzing Hill coefficients: n={n}, p={p}")
            
            # Result dictionary for this combination
            result = {
                'n': n,
                'p': p,
                'excitable_count': 0,
                'excitable_configs': [],
                'excitable_grid': np.zeros((grid_size, grid_size), dtype=bool),
                'bs_min': float('inf'),  # Direct tracking of min/max values
                'bs_max': float('-inf'),
                'bk_min': float('inf'),
                'bk_max': float('-inf')
            }
            
            # Set up progress indicator
            total_points = grid_size * grid_size
            progress_interval = max(1, total_points // 10)  # 10% steps
            points_checked = 0
            
            # Grid search in defined range
            for i, bs in enumerate(bs_range):
                for j, bk in enumerate(bk_range):
                    # Create parameter set
                    params = std_params.copy()
                    params['n'] = n
                    params['p'] = p
                    params['bs'] = bs
                    params['bk'] = bk
                    
                    # Check if system is excitable
                    is_exc, info = is_excitable(params, model_odes, 
                                             find_fixed_points_func=find_fixed_points_func,
                                             classify_fixed_point_func=classify_fixed_point_func)
                    
                    if is_exc:
                        result['excitable_count'] += 1
                        result['excitable_configs'].append({
                            'bs': bs,
                            'bk': bk,
                            'params': params,
                            'fixed_points': info['fixed_points'],
                            'fp_types': info['fp_types'],
                            'index': (i, j)  # Store grid position
                        })
                        result['excitable_grid'][i, j] = True
                        
                        # Update boundary values
                        result['bs_min'] = min(result['bs_min'], bs)
                        result['bs_max'] = max(result['bs_max'], bs)
                        result['bk_min'] = min(result['bk_min'], bk)
                        result['bk_max'] = max(result['bk_max'], bk)
                    
                    # Update progress
                    points_checked += 1
                    if points_checked % progress_interval == 0:
                        progress_percent = points_checked / total_points * 100
                        print(f"  Progress: {progress_percent:.1f}% ({result['excitable_count']} excitable configurations found)")
            
            # If no excitable configurations found, reset boundary values
            if result['excitable_count'] == 0:
                result['bs_min'] = result['bk_min'] = None
                result['bs_max'] = result['bk_max'] = None
            
            # Perform extended search if requested
            if extended_search:
                print(f"  Performing extended search...")
                
                # Grid for extended search
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
                
                # Reset progress
                points_checked = 0
                
                # Extended grid search
                for i, bs in enumerate(ext_bs_range):
                    for j, bk in enumerate(ext_bk_range):
                        # Create parameter set
                        params = std_params.copy()
                        params['n'] = n
                        params['p'] = p
                        params['bs'] = bs
                        params['bk'] = bk
                        
                        # Check if system is excitable
                        is_exc, info = is_excitable(params, model_odes,
                                                find_fixed_points_func=find_fixed_points_func,
                                                classify_fixed_point_func=classify_fixed_point_func)
                        
                        if is_exc:
                            ext_result['extended_excitable_count'] += 1
                            ext_result['extended_excitable_configs'].append({
                                'bs': bs,
                                'bk': bk,
                                'params': params,
                                'fixed_points': info['fixed_points'],
                                'fp_types': info['fp_types'],
                                'index': (i, j),  # Store grid position
                                'extended_search': True  # Mark as from extended search
                            })
                            ext_excitable_grid[i, j] = True
                            
                            # Update boundary values for extended search
                            ext_result['extended_bs_min'] = min(ext_result['extended_bs_min'], bs)
                            ext_result['extended_bs_max'] = max(ext_result['extended_bs_max'], bs)
                            ext_result['extended_bk_min'] = min(ext_result['extended_bk_min'], bk)
                            ext_result['extended_bk_max'] = max(ext_result['extended_bk_max'], bk)
                        
                        # Update progress
                        points_checked += 1
                        if points_checked % (progress_interval * 5) == 0:  # Less frequent updates for extended search
                            progress_percent = points_checked / total_points * 100
                            print(f"  Ext. search progress: {progress_percent:.1f}% ({ext_result['extended_excitable_count']} excitable configurations found)")
                
                # If no excitable configurations found, reset boundary values
                if ext_result['extended_excitable_count'] == 0:
                    ext_result['extended_bs_min'] = ext_result['extended_bk_min'] = None
                    ext_result['extended_bs_max'] = ext_result['extended_bk_max'] = None
                
                # Integrate extended results into main result
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
            
            # Visualize results
            visualize_hill_coefficient_result(n, p, result, bs_range, bk_range, plots_dir, data_dir)
            
            print(f"\nResults for n={n}, p={p}:")
            print(f"  Excitable configurations (Standard): {result['excitable_count']} of {total_points} ({result['excitable_count']/total_points*100:.2f}%)")
            
            if result['excitable_count'] > 0:
                print(f"  Standard range: bs: [{result['bs_min']:.4f}, {result['bs_max']:.4f}], bk: [{result['bk_min']:.4f}, {result['bk_max']:.4f}]")
            
            if 'has_extended_search' in result:
                print(f"  Excitable configurations (Extended): {result['extended_excitable_count']} of {total_points} ({result['extended_excitable_count']/total_points*100:.2f}%)")
                
                if result['extended_excitable_count'] > 0:
                    print(f"  Extended range: bs: [{result['extended_bs_min']:.4f}, {result['extended_bs_max']:.4f}], bk: [{result['extended_bk_min']:.4f}, {result['extended_bk_max']:.4f}]")
            
            # Store result for this combination
            all_results[f'n{n}_p{p}'] = result
    
    # Create comparative visualization
    visualize_hill_coefficient_comparison(all_results, bs_range, bk_range, plots_dir, data_dir)
    
    # Save results
    save_results(all_results, 'hill_coefficient_results.pkl', data_dir)
    
    # Create text file with boundary values
    create_boundary_values_summary(all_results, n_values, p_values, data_dir)
    
    return all_results

def visualize_hill_coefficient_result(n, p, result, bs_range, bk_range, plots_dir, data_dir):
    """
    Visualizes the excitable region for a Hill coefficient combination.
    
    Args:
        n: ComK Hill coefficient
        p: ComS Hill coefficient
        result: Result dictionary for this combination
        bs_range: Range of bs values (standard)
        bk_range: Range of bk values (standard)
        plots_dir: Directory for plot files
        data_dir: Directory for text files
    """
    # Create subdirectories for Hill coefficient plots
    hill_plots_dir = os.path.join(plots_dir, 'hill_coefficient_plots')
    hill_data_dir = os.path.join(data_dir, 'hill_coefficient_data')
    os.makedirs(hill_plots_dir, exist_ok=True)
    os.makedirs(hill_data_dir, exist_ok=True)
    
    # Check if excitable configurations exist in standard range
    has_standard_excitable = (result['excitable_count'] > 0 and 
                             'bs_min' in result and 
                             result['bs_min'] is not None)
    
    # Check if excitable configurations exist in extended range
    has_extended_excitable = ('has_extended_search' in result and 
                            result['extended_excitable_count'] > 0 and 
                            'extended_bs_min' in result and 
                            result['extended_bs_min'] is not None)
    
    # 1. Visualize standard range if excitable configurations exist
    if has_standard_excitable:
        bs_plot_range = [0.6, 1.0]
        bk_plot_range = [0.05, 0.1]
        title_prefix = "Excitable Region"
        filename_prefix = "excitable_region"
        
        excitable_grid = result['excitable_grid']
        bs_min = result['bs_min']
        bs_max = result['bs_max']
        bk_min = result['bk_min']
        bk_max = result['bk_max']
        count = result['excitable_count']
        
        # Create standard plot
        plot_single_region(hill_plots_dir, hill_data_dir, n, p,
                          bs_range, bk_range, excitable_grid,
                          bs_min, bs_max, bk_min, bk_max, count,
                          bs_plot_range, bk_plot_range,
                          title_prefix, filename_prefix, is_extended=False)
    
    # 2. Visualize extended range if excitable configurations exist
    if has_extended_excitable:
        bs_plot_range = [0.1, 2.0]
        bk_plot_range = [0.01, 0.3]
        title_prefix = "Extended Excitable Region"
        filename_prefix = "extended_excitable_region"
        
        excitable_grid = result['extended_excitable_grid']
        bs_min = result['extended_bs_min']
        bs_max = result['extended_bs_max']
        bk_min = result['extended_bk_min']
        bk_max = result['extended_bk_max']
        count = result['extended_excitable_count']
        
        # Create extended plot
        plot_single_region(hill_plots_dir, hill_data_dir, n, p,
                          result['extended_bs_range'], result['extended_bk_range'], excitable_grid,
                          bs_min, bs_max, bk_min, bk_max, count,
                          bs_plot_range, bk_plot_range,
                          title_prefix, filename_prefix, is_extended=True)
    
    # 3. Create "No excitable configurations" plot if none found
    if not has_standard_excitable and not has_extended_excitable:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'No excitable configurations found', 
              ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.xlabel('bk (ComK Feedback Strength)')
        plt.ylabel('bs (ComS Expression Rate)')
        plt.title(f'Hill Coefficients n={n}, p={p}')
        
        # Standard axis ranges
        plt.xlim([0.05, 0.1])
        plt.ylim([0.6, 1.0])
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(hill_plots_dir, f'no_excitable_region_n{n}_p{p}.png'), dpi=300)
        
        # Create text file with note
        with open(os.path.join(hill_data_dir, f'no_excitable_region_n{n}_p{p}.txt'), 'w') as f:
            f.write(f"No excitable configurations for n={n}, p={p}\n")
            f.write("=================================================================\n\n")
            f.write("No excitable configurations found in either standard or extended range.\n")
        
        plt.close()

def plot_single_region(hill_plots_dir, hill_data_dir, n, p, 
                      bs_range, bk_range, excitable_grid,
                      bs_min, bs_max, bk_min, bk_max, count,
                      bs_plot_range, bk_plot_range,
                      title_prefix, filename_prefix, is_extended):
    """
    Creates a single plot for an excitable region.
    
    Args:
        hill_plots_dir: Directory for plot files
        hill_data_dir: Directory for text files
        n, p: Hill coefficients
        bs_range, bk_range: Ranges of bs and bk values for heatmap
        excitable_grid: Grid of excitable configurations
        bs_min, bs_max, bk_min, bk_max: Boundary values of excitable region
        count: Number of excitable configurations
        bs_plot_range, bk_plot_range: Ranges for plot axes
        title_prefix, filename_prefix: Prefixes for title and filename
        is_extended: True if this is the extended range
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Heatmap of excitable region
    extent = [min(bk_range), max(bk_range), min(bs_range), max(bs_range)]
    plt.imshow(excitable_grid, 
              extent=extent,
              origin='lower', aspect='auto', cmap='Blues', alpha=0.6)
    
    # Min/Max lines with exact values
    plt.axhline(y=bs_min, color='orange', linestyle='--', alpha=0.7, 
               label=f'bs_min = {bs_min:.4f}')
    plt.axhline(y=bs_max, color='orange', linestyle='--', alpha=0.7, 
               label=f'bs_max = {bs_max:.4f}')
    plt.axvline(x=bk_min, color='green', linestyle='--', alpha=0.7, 
               label=f'bk_min = {bk_min:.4f}')
    plt.axvline(x=bk_max, color='green', linestyle='--', alpha=0.7, 
               label=f'bk_max = {bk_max:.4f}')
    
    # Mark standard parameters
    from competence_circuit_analysis import default_params
    std_params = default_params()
    plt.scatter([std_params['bk']], [std_params['bs']], c='red', s=100, marker='*', label='Standard')
    
    # Calculate region size
    bs_range_val = bs_max - bs_min
    bk_range_val = bk_max - bk_min
    area = bs_range_val * bk_range_val
    
    # Labels
    plt.xlabel('bk (ComK Feedback Strength)')
    plt.ylabel('bs (ComS Expression Rate)')
    plt.title(f'{title_prefix} for n={n}, p={p}\n'
             f'Range: bs: [{bs_min:.4f}, {bs_max:.4f}], bk: [{bk_min:.4f}, {bk_max:.4f}]\n'
             f'Area: {area:.6f}, Count: {count}')
    
    # Legend at top-right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Set axis ranges
    plt.xlim(bk_plot_range)
    plt.ylim(bs_plot_range)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(hill_plots_dir, f'{filename_prefix}_n{n}_p{p}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create text file with boundary values
    with open(os.path.join(hill_data_dir, f'{filename_prefix}_n{n}_p{p}.txt'), 'w') as f:
        f.write(f"Boundaries of {title_prefix} for n={n}, p={p}\n")
        f.write("=================================================================\n\n")
        f.write(f"bs_min: {bs_min:.6f}\n")
        f.write(f"bs_max: {bs_max:.6f}\n")
        f.write(f"bs_range: {bs_range_val:.6f}\n\n")
        f.write(f"bk_min: {bk_min:.6f}\n")
        f.write(f"bk_max: {bk_max:.6f}\n")
        f.write(f"bk_range: {bk_range_val:.6f}\n\n")
        f.write(f"Area: {area:.6f}\n")
        f.write(f"Number of excitable configurations: {count}\n")
        
        # Additional information for extended range
        if is_extended:
            f.write("\nNote: These results are from an extended search in the range\n")
            f.write(f"bs: [{bs_plot_range[0]}, {bs_plot_range[1]}], bk: [{bk_plot_range[0]}, {bk_plot_range[1]}]\n")

def visualize_hill_coefficient_comparison(all_results, bs_range, bk_range, plots_dir, data_dir):
    """
    Creates comparative visualizations for all Hill coefficient combinations.
    
    Args:
        all_results: Dictionary of results for all combinations
        bs_range: Range of bs values (standard)
        bk_range: Range of bk values (standard)
        plots_dir: Directory for plot files
        data_dir: Directory for text files
    """
    # Create subdirectories for comparisons
    comparison_plots_dir = os.path.join(plots_dir, 'comparisons')
    comparison_data_dir = os.path.join(data_dir, 'comparisons')
    os.makedirs(comparison_plots_dir, exist_ok=True)
    os.makedirs(comparison_data_dir, exist_ok=True)
    
    # Extract arrays of Hill coefficients
    n_values = sorted(set([all_results[key]['n'] for key in all_results]))
    p_values = sorted(set([all_results[key]['p'] for key in all_results]))
    
    # Matrix for excitable counts in standard range
    excitable_counts = np.zeros((len(n_values), len(p_values)))
    
    # Matrices for extensions
    standard_counts = np.zeros((len(n_values), len(p_values)))
    extended_counts = np.zeros((len(n_values), len(p_values)))
    total_counts = np.zeros((len(n_values), len(p_values)))
    
    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            key = f'n{n}_p{p}'
            if key in all_results:
                # Standard range (for original visualizations)
                excitable_counts[i, j] = all_results[key]['excitable_count']
                
                # For new visualizations
                standard_counts[i, j] = all_results[key]['excitable_count']
                
                # Extended range
                if 'has_extended_search' in all_results[key] and 'extended_excitable_count' in all_results[key]:
                    extended_counts[i, j] = all_results[key]['extended_excitable_count']
                
                # Total count (prefer standard, fall back to extended)
                if standard_counts[i, j] > 0:
                    total_counts[i, j] = standard_counts[i, j]
                else:
                    total_counts[i, j] = extended_counts[i, j]
    
    # ORIGINAL VISUALIZATIONS
    
    # Heatmap of excitable configurations count (standard range only)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(excitable_counts, cmap='viridis')
    
    # Labels
    plt.colorbar(im, label='Number of Excitable Configurations')
    plt.xlabel('p (ComS Repression Hill Coefficient)')
    plt.ylabel('n (ComK Activation Hill Coefficient)')
    plt.title('Comparison of Excitable Region Size')
    
    # Axis labels
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    
    # Add values to cells
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(excitable_counts[i, j])}", 
                          ha="center", va="center", color="w", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_heatmap.png'), dpi=300)
    plt.close()
    
    # Bar plot of excitable region size (standard range only)
    plt.figure(figsize=(12, 6))
    
    # Prepare data for bar plot
    x_labels = [f'n={n}, p={p}' for n in n_values for p in p_values]
    counts = [all_results[f'n{n}_p{p}']['excitable_count'] for n in n_values for p in p_values]
    
    # Color coding based on n values
    colors = plt.cm.tab10(np.array([i for i in range(len(n_values)) for _ in range(len(p_values))]) % 10)
    
    # Create bar plot
    bars = plt.bar(np.arange(len(x_labels)), counts, color=colors)
    
    # Labels
    plt.xlabel('Hill Coefficients (n, p)')
    plt.ylabel('Number of Excitable Configurations')
    plt.title('Size of Excitable Region for Different Hill Coefficients')
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Legend for n values
    handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(i % 10)) for i in range(len(n_values))]
    labels = [f'n={n}' for n in n_values]
    plt.legend(handles, labels, title='ComK Activation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_barplot.png'), dpi=300)
    plt.close()
    
    # NEW VISUALIZATIONS FOR STANDARD AND EXTENDED RANGES
    
    # Heatmaps for Standard, Extended, and Combined
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(standard_counts, cmap='Blues')
    plt.colorbar(im1, label='Count (Standard Range)')
    plt.xlabel('p (ComS Repression)')
    plt.ylabel('n (ComK Activation)')
    plt.title('Standard Range')
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(standard_counts[i, j])}", 
                          ha="center", va="center", color="w" if standard_counts[i, j] > standard_counts.max()/2 else "black", 
                          fontweight='bold')
    
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(extended_counts, cmap='Greens')
    plt.colorbar(im2, label='Count (Extended Range)')
    plt.xlabel('p (ComS Repression)')
    plt.ylabel('n (ComK Activation)')
    plt.title('Extended Range')
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            text = plt.text(j, i, f"{int(extended_counts[i, j])}", 
                          ha="center", va="center", color="w" if extended_counts[i, j] > extended_counts.max()/2 else "black", 
                          fontweight='bold')
    
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(total_counts, cmap='viridis')
    plt.colorbar(im3, label='Count (Combined)')
    plt.xlabel('p (ComS Repression)')
    plt.ylabel('n (ComK Activation)')
    plt.title('Best (Standard or Extended)')
    plt.xticks(np.arange(len(p_values)), p_values)
    plt.yticks(np.arange(len(n_values)), n_values)
    for i in range(len(n_values)):
        for j in range(len(p_values)):
            # Visual marking for Standard/Extended
            marker = "*" if standard_counts[i, j] > 0 else "+" if extended_counts[i, j] > 0 else ""
            text = plt.text(j, i, f"{int(total_counts[i, j])}{marker}", 
                          ha="center", va="center", color="w" if total_counts[i, j] > total_counts.max()/2 else "black", 
                          fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_plots_dir, 'excitable_count_heatmaps_combined.png'), dpi=300)
    plt.close()

def create_boundary_values_summary(all_results, n_values, p_values, data_dir):
    """
    Creates a text file with boundary values for all Hill coefficient combinations.
    
    Args:
        all_results: Dictionary of results for all combinations
        n_values: List of n values
        p_values: List of p values
        data_dir: Directory to save the text file
    """
    with open(os.path.join(data_dir, 'boundary_values.txt'), 'w') as f:
        f.write("Boundaries of Excitable Regions for Different Hill Coefficients\n")
        f.write("=================================================================\n\n")
        
        f.write("Standard Range (bs: 0.6-1.0, bk: 0.05-0.1):\n")
        f.write("Combination | bs_min  | bs_max  | bs_range | bk_min  | bk_max  | bk_range | Area     | Count\n")
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
        
        f.write("\n\nExtended Range (bs: 0.1-2.0, bk: 0.01-0.3):\n")
        f.write("Combination | bs_min  | bs_max  | bs_range | bk_min  | bk_max  | bk_range | Area     | Count\n")
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

def search_suel_parameter_space(model_odes, classify_fixed_point_func, find_fixed_points_func,
                             results_dir, n_points=100):
    """
    Performs a high-resolution search in the Suel et al. parameter space.
    
    Args:
        model_odes: Function defining the ODEs
        classify_fixed_point_func: Function to classify fixed points
        find_fixed_points_func: Function to find fixed points
        results_dir: Directory to save results
        n_points: Number of points per dimension (higher resolution)
        
    Returns:
        list: List of excitable configurations
    """
    print("\n=== HIGH-RESOLUTION SEARCH IN SUEL ET AL. PARAMETER SPACE ===")
    
    # Load standard parameters
    from competence_circuit_analysis import default_params
    std_params = default_params()
    
    # Suel et al. parameter range
    bs_range = np.linspace(0.6, 1.0, n_points)  # Range from Suel et al.
    bk_range = np.linspace(0.05, 0.1, n_points)  # Range from Suel et al.
    
    print(f"Search range: bs=[{min(bs_range):.4f}, {max(bs_range):.4f}], bk=[{min(bk_range):.4f}, {max(bk_range):.4f}]")
    print(f"Resolution: {n_points}x{n_points} points (total: {n_points*n_points} parameter combinations)")
    
    # List for excitable configurations
    excitable_configs = []
    
    # Set up progress indicator
    total_points = len(bs_range) * len(bk_range)
    progress_interval = max(1, total_points // 20)  # 5% steps
    points_checked = 0
    excitable_count = 0
    
    # Grid search in defined range
    for i, bs in enumerate(bs_range):
        for j, bk in enumerate(bk_range):
            # Create parameter set
            params = std_params.copy()
            params['bs'] = bs
            params['bk'] = bk
            
            # Check if system is excitable
            is_exc, info = is_excitable(params, model_odes,
                                     find_fixed_points_func=find_fixed_points_func,
                                     classify_fixed_point_func=classify_fixed_point_func)
            
            if is_exc:
                excitable_count += 1
                excitable_configs.append({
                    'bs': bs,
                    'bk': bk,
                    'params': params,
                    'fixed_points': info['fixed_points'],
                    'fp_types': info['fp_types'],
                    'index': (i, j)  # Store grid position for later analysis
                })
            
            # Update progress
            points_checked += 1
            if points_checked % progress_interval == 0:
                progress_percent = points_checked / total_points * 100
                print(f"Progress: {progress_percent:.1f}% ({excitable_count} excitable configurations found)")
    
    print(f"\nParameter search completed:")
    print(f"  Checked parameter combinations: {points_checked}")
    print(f"  Found excitable configurations: {excitable_count}")
    
    # Save results
    save_results(excitable_configs, 'excitable_configs.pkl', results_dir)
    
    # Visualize excitable configurations
    visualize_search_results(excitable_configs, bs_range, bk_range, results_dir)
    
    return excitable_configs

def visualize_search_results(excitable_configs, bs_range, bk_range, results_dir):
    """
    Visualizes the search results in parameter space.
    
    Args:
        excitable_configs: List of excitable configurations
        bs_range: Range of bs values
        bk_range: Range of bk values
        results_dir: Directory to save results
    """
    # Create directory for example configurations
    example_dir = os.path.join(results_dir, 'example_configs')
    os.makedirs(example_dir, exist_ok=True)
    
    # Create map of parameter space
    plot_excitable_map(excitable_configs, bs_range, bk_range, 
                      os.path.join(results_dir, 'excitable_map.png'),
                      title='Excitable Configurations in Suel et al. Parameter Space')
    
    # Visualize example nullclines
    if excitable_configs:
        # Select a representative subset of configurations
        num_examples = min(5, len(excitable_configs))
        indices = np.linspace(0, len(excitable_configs)-1, num_examples).astype(int)
        
        from competence_circuit_analysis import plot_example_nullclines, plot_phase_diagram, simulate_system, plot_time_series
        
        for i, idx in enumerate(indices):
            config = excitable_configs[idx]
            bs = config['bs']
            bk = config['bk']
            params = config['params']
            
            # Create nullclines diagram
            fig, ax = plot_example_nullclines(
                params, 
                f"Nullclines for bs={bs:.4f}, bk={bk:.4f}"
            )
            plt.savefig(os.path.join(example_dir, f'nullclines_example_{i+1}.png'), dpi=300)
            plt.close()
            
            # Create phase diagram with vector field
            fig, ax = plot_phase_diagram(
                params, 
                np.linspace(0, 1.0, 200), 
                np.linspace(0, 1.5, 200),
                f"Phase Diagram for bs={bs:.4f}, bk={bk:.4f}",
                fixed_points=config['fixed_points'],
                show_vector_field=True
            )
            plt.savefig(os.path.join(example_dir, f'phase_diagram_example_{i+1}.png'), dpi=300)
            plt.close()
            
            # Simulate time series          
            t, K, S = simulate_system(params, initial_conditions=[0.01, 0.5], t_max=200)
            fig, ax, comp_periods = plot_time_series(
                t, K, S, threshold=0.5,
                title=f"Time Series for bs={bs:.4f}, bk={bk:.4f}"
            )
            plt.savefig(os.path.join(example_dir, f'time_series_example_{i+1}.png'))
            plt.close()
    
    # Save parameter distributions
    if excitable_configs:
        with open(os.path.join(results_dir, 'excitable_params.txt'), 'w') as f:
            f.write("Excitable Parameter Configurations in Suel et al. Range\n")
            f.write("===================================================\n\n")
            
            f.write(f"Total configurations found: {len(excitable_configs)}\n\n")
            
            bs_values = [config['bs'] for config in excitable_configs]
            bk_values = [config['bk'] for config in excitable_configs]
            
            f.write(f"bs range: [{min(bs_values):.4f}, {max(bs_values):.4f}], Mean: {np.mean(bs_values):.4f}\n")
            f.write(f"bk range: [{min(bk_values):.4f}, {max(bk_values):.4f}], Mean: {np.mean(bk_values):.4f}\n\n")
            
            f.write("Individual parameter configurations:\n")
            for i, config in enumerate(excitable_configs):
                if i < 20:  # Limit to 20 for readability
                    f.write(f"Configuration {i+1}:\n")
                    f.write(f"  bs = {config['bs']:.6f}, bk = {config['bk']:.6f}\n")
                    f.write(f"  Fixed points: {len(config['fixed_points'])}\n")
                    for j, (fp, fp_type) in enumerate(zip(config['fixed_points'], config['fp_types'])):
                        f.write(f"    FP{j+1}: ({fp[0]:.4f}, {fp[1]:.4f}) - {fp_type}\n")
                    f.write("\n")
            
            if len(excitable_configs) > 20:
                f.write(f"... and {len(excitable_configs) - 20} more configurations.")

def select_representative_configs(excitable_configs, standard_config=None, num_configs=5):
    """
    Selects representative excitable configurations for further analysis.
    
    Args:
        excitable_configs: List of excitable configurations
        standard_config: Optional standard configuration to include
        num_configs: Number of configurations to select
        
    Returns:
        list: Selected configurations
    """
    # Import standard parameters if not provided
    if standard_config is None:
        from competence_circuit_analysis import default_params
        std_params = default_params()
        
        # Create standard configuration
        standard_config = {
            'bs': std_params['bs'],
            'bk': std_params['bk'],
            'params': std_params.copy(),
            'name': 'Standard'
        }
    
    # List of selected configurations
    selected_configs = [standard_config]
    
    if not excitable_configs:
        print("No excitable configurations to select from.")
        return selected_configs
    
    # If we have too few configurations, take all of them
    if len(excitable_configs) <= num_configs:
        for i, config in enumerate(excitable_configs):
            selected_configs.append({
                'bs': config['bs'],
                'bk': config['bk'],
                'params': config['params'].copy(),
                'name': f'Excitable {i+1}'
            })
        return selected_configs
    
    # Extract bs and bk values
    all_bs = np.array([config['bs'] for config in excitable_configs])
    all_bk = np.array([config['bk'] for config in excitable_configs])
    
    # Find minimum and maximum values
    bs_min, bs_max = np.min(all_bs), np.max(all_bs)
    bk_min, bk_max = np.min(all_bk), np.max(all_bk)
    
    # Select diverse configurations
    
    # Configuration with bs near minimum
    idx_min_bs = np.argmin(all_bs)
    selected_configs.append({
        'bs': excitable_configs[idx_min_bs]['bs'],
        'bk': excitable_configs[idx_min_bs]['bk'],
        'params': excitable_configs[idx_min_bs]['params'].copy(),
        'name': 'Low ComS Expression'
    })
    
    # Configuration with bs near maximum
    idx_max_bs = np.argmax(all_bs)
    selected_configs.append({
        'bs': excitable_configs[idx_max_bs]['bs'],
        'bk': excitable_configs[idx_max_bs]['bk'],
        'params': excitable_configs[idx_max_bs]['params'].copy(),
        'name': 'High ComS Expression'
    })
    
    # Configuration with bk near minimum
    idx_min_bk = np.argmin(all_bk)
    selected_configs.append({
        'bs': excitable_configs[idx_min_bk]['bs'],
        'bk': excitable_configs[idx_min_bk]['bk'],
        'params': excitable_configs[idx_min_bk]['params'].copy(),
        'name': 'Weak ComK Feedback'
    })
    
    # Configuration with bk near maximum
    idx_max_bk = np.argmax(all_bk)
    selected_configs.append({
        'bs': excitable_configs[idx_max_bk]['bs'],
        'bk': excitable_configs[idx_max_bk]['bk'],
        'params': excitable_configs[idx_max_bk]['params'].copy(),
        'name': 'Strong ComK Feedback'
    })
    
    # Create visualization of selected configurations
    plt.figure(figsize=(10, 8))
    
    # All excitable configurations
    plt.scatter(all_bs, all_bk, c='lightgray', s=30, alpha=0.5, label='All excitable configurations')
    
    # Selected configurations
    for config in selected_configs:
        plt.scatter([config['bs']], [config['bk']], s=100, label=config['name'])
    
    plt.xlabel('bs (ComS Expression Rate)')
    plt.ylabel('bk (ComK Feedback Strength)')
    plt.title('Selected Representative Configurations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return selected_configs

def find_excitable_configurations(model_odes, classify_fixed_point_func, find_fixed_points_func,
                               params_base, results_dir, n_samples=10000):
    """
    Systematically searches for parameter configurations that generate excitable systems.
    
    Args:
        model_odes: Function defining the ODEs
        classify_fixed_point_func: Function to classify fixed points
        find_fixed_points_func: Function to find fixed points
        params_base: Base parameters
        results_dir: Directory to save results
        n_samples: Number of parameter combinations to test
        
    Returns:
        list: Parameter tuples that generate excitable systems
    """
    print(f"Starting search for excitable configurations with {n_samples} samples...")
    
    # Parameter ranges based on the Schultz paper
    # Note: Ranges are adjusted to avoid zero values and ensure a more realistic search space
    # Schultz et al. ak: 0.0-0.2 and all others 0.0-1.0
    param_ranges = {
        'n': [2, 3, 4, 5],             # Hill coefficient for ComK (discrete values)
        'p': [2, 3, 4, 5],             # Hill coefficient for ComS (discrete values)
        'ak': np.linspace(0.001, 0.1, 20),  # basal ComK expression 
        'bk': np.linspace(0.04, 0.5, 20),   # ComK feedback
        'bs': np.linspace(0.1, 1.0, 20),    # ComS expression
        'k0': np.linspace(0.1, 1.0, 20),    # ComK activation
        'k1': np.linspace(0.1, 1.0, 20)     # ComS repression
    }
    
    excitable_configs = []
    tested_configs = set()  # To avoid duplicates
    
    # Progress display
    progress_interval = max(1, n_samples // 20)  # 5% steps
    
    for i in range(n_samples):
        # Show progress
        if i % progress_interval == 0:
            print(f"Progress: {i/n_samples*100:.1f}% ({len(excitable_configs)} excitable systems found)")
        
        # Generate parameters with uniform distribution
        params_test = params_base.copy()
        
        # Hill coefficients
        params_test['n'] = np.random.choice(param_ranges['n'])
        params_test['p'] = np.random.choice(param_ranges['p'])
        
        # Continuous parameters - uniform in defined range
        params_test['ak'] = np.random.uniform(0.001, 0.05)
        params_test['bk'] = np.random.uniform(0.05, 0.5)
        params_test['bs'] = np.random.uniform(0.6, 1.3)
        params_test['k0'] = np.random.uniform(0.2, 0.8)
        params_test['k1'] = np.random.uniform(0.2, 0.8)
        
        # Key parameters for hash to avoid duplicates
        config_key = (params_test['n'], params_test['p'], 
                     round(params_test['ak'], 3), round(params_test['bk'], 3), 
                     round(params_test['bs'], 3), round(params_test['k0'], 3), 
                     round(params_test['k1'], 3))
        
        if config_key in tested_configs:
            continue  # Already tested, skip
        
        tested_configs.add(config_key)
        
        # Check if system is excitable
        is_exc, info = is_excitable(params_test, model_odes,
                                 find_fixed_points_func=find_fixed_points_func,
                                 classify_fixed_point_func=classify_fixed_point_func)
        
        if is_exc:
            # Fixed points for excitable system
            fps = info['fixed_points']
            fp_types = info['fp_types']
            
            # Check for the right pattern: 1 stable, 1 saddle, and 1 unstable fixed point
            stable_fps = [fp for i, fp in enumerate(fps) if 'Stabil' in fp_types[i]]
            
            if stable_fps and stable_fps[0][0] < 0.2:  # Stable point with low ComK
                # Save parameters for excitable system
                excitable_configs.append((
                    params_test['n'], params_test['p'], 
                    params_test['ak'], params_test['bk'], 
                    params_test['bs'], params_test['k0'], 
                    params_test['k1']
                ))
    
    print(f"Search completed: {len(excitable_configs)} excitable configurations found.")
    
    # Create histograms and visualizations
    if excitable_configs:
        fig, axes, param_values = plot_parameter_histograms(excitable_configs, params_base)
        plt.savefig(os.path.join(results_dir, 'parameter_histograms.png'))
        plt.close()
        
        # Save detailed results
        save_results(excitable_configs, 'excitable_configs.pkl', results_dir)
        
        # Save as CSV
        save_excitable_configs_as_csv(excitable_configs, results_dir)
    
    return excitable_configs

def save_excitable_configs_as_csv(excitable_configs, results_dir):
    """
    Saves excitable configurations to a CSV file.
    
    Args:
        excitable_configs: List of excitable configuration tuples
        results_dir: Directory to save results
    """
    # Create DataFrame
    columns = ['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']
    df = pd.DataFrame(excitable_configs, columns=columns)
    
    # Save to CSV
    csv_path = os.path.join(results_dir, 'excitable_configs.csv')
    df.to_csv(csv_path, index=False)
    print(f"Excitable configurations saved to {csv_path}")

def select_diverse_excitable_configs(excitable_configs, n_configs=10):
    """
    Selects a diverse set of excitable configurations using K-means clustering.
    
    Args:
        excitable_configs: List of excitable configuration tuples
        n_configs: Number of configurations to select
        
    Returns:
        list: Selected excitable configurations
    """
    if len(excitable_configs) <= n_configs:
        return excitable_configs
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn not installed. Using simple selection method instead.")
        indices = np.linspace(0, len(excitable_configs) - 1, n_configs).astype(int)
        return [excitable_configs[i] for i in indices]
    
    # Convert to numpy array for clustering
    configs_array = np.array(excitable_configs)
    
    # Standardize the features to have zero mean and unit variance
    scaler = StandardScaler()
    scaled_configs = scaler.fit_transform(configs_array)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_configs, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_configs)
    
    # Get the indices of configurations closest to cluster centers
    selected_indices = []
    for i in range(n_configs):
        cluster_members = np.where(cluster_labels == i)[0]
        if len(cluster_members) > 0:
            # Find the member closest to the cluster center
            cluster_center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(scaled_configs[cluster_members] - cluster_center, axis=1)
            closest_idx = cluster_members[np.argmin(distances)]
            selected_indices.append(closest_idx)
    
    # Return the selected configurations
    return [excitable_configs[i] for i in selected_indices]