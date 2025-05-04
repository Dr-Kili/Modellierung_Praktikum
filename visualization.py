"""
Visualization functions for competence circuit analysis.

This module contains functions for creating various plots and visualizations
for the B. subtilis competence circuit model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_nullclines(params, fixed_points, output_path, model_nullclines=None,
                   title=None, K_range=None, S_range=None):
    """
    Draws nullclines for a given parameter set.
    
    Args:
        params: Parameter dictionary for the competence model
        fixed_points: List of fixed points
        output_path: Path to save the figure
        model_nullclines: Function to calculate nullclines (if None, imports from comp_model)
        title: Optional title for the figure
        K_range: Optional range for ComK axis
        S_range: Optional range for ComS axis
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects
    """
    # Import comp_model only if function isn't provided
    if model_nullclines is None:
        import competence_circuit_analysis as comp_model
        model_nullclines = comp_model.nullclines
        classify_fixed_point = comp_model.classify_fixed_point
    else:
        # If model_nullclines is provided, we need classify_fixed_point too
        from competence_circuit_analysis import classify_fixed_point
    
    # Always use 0.0 as starting point for K-axis
    if K_range is None:
        K_range = [0.0, 1.0]
    
    # Create a fine grid for K
    K_grid = np.linspace(K_range[0], K_range[1], 500)
    
    # Calculate nullclines
    S_from_K, S_from_S = model_nullclines(K_grid, params)
    
    # Remove invalid values (negative or infinite)
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    # Determine S-axis range based on nullclines and fixed points
    if S_range is None:
        max_s_nullcline = max(np.max(S_from_K[valid_K]) if np.any(valid_K) else 0,
                           np.max(S_from_S[valid_S]) if np.any(valid_S) else 0)
        max_s_fixpoints = max([fp[1] for fp in fixed_points]) if fixed_points else 0
        max_s = max(max_s_nullcline, max_s_fixpoints) * 1.2
        S_range = [0.0, max_s]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw nullclines with standard colors
    ax.plot(K_grid[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullcline', linewidth=2)
    ax.plot(K_grid[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullcline', linewidth=2)
    
    # Draw fixed points
    for i, fp in enumerate(fixed_points):
        K, S = fp
        fp_type = classify_fixed_point(K, S, params)
        
        # Define colors and markers for fixed points
        if 'Stabil' in fp_type:
            color = 'g'
            label = f'FP{i+1}: Stable Node ({K:.3f}, {S:.3f})'
        elif 'Sattel' in fp_type:
            color = 'y'
            label = f'FP{i+1}: Saddle Point ({K:.3f}, {S:.3f})'
        elif 'Instabil' in fp_type:
            color = 'r'
            label = f'FP{i+1}: Unstable Node ({K:.3f}, {S:.3f})'
        else:
            color = 'gray'
            label = f'FP{i+1}: {fp_type} ({K:.3f}, {S:.3f})'
        
        ax.plot(K, S, 'o', color=color, markersize=10, label=label)
    
    # Explicitly set axis ranges
    ax.set_xlim(K_range)
    ax.set_ylim(S_range)
    
    # Labels
    ax.set_xlabel('ComK Concentration')
    ax.set_ylabel('ComS Concentration')
    
    # Use standard title with Hill coefficients and parameter values if not provided
    if title is None:
        n = params.get('n', 2)
        p = params.get('p', 5)
        bs = params.get('bs', 0.82)
        bk = params.get('bk', 0.07)
        title = f"Nullclines for n={n}, p={p}, bs={bs:.4f}, bk={bk:.4f}"
    
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    return fig, ax

def plot_phase_diagram(params, K_range, S_range, title="Phase Diagram", 
                     fixed_points=None, show_vector_field=True, trajectories=None,
                     model_odes=None, model_nullclines=None, classify_fixed_point=None):
    """
    Creates a phase diagram for the competence circuit model.
    
    Args:
        params: Parameter dictionary
        K_range: Range of K values [min, max] or array
        S_range: Range of S values [min, max] or array
        title: Title for the plot
        fixed_points: List of fixed points to mark on the diagram
        show_vector_field: Whether to show the vector field
        trajectories: List of (K, S) trajectories to plot
        model_odes: Function defining the ODEs (if None, imports from comp_model)
        model_nullclines: Function to calculate nullclines (if None, imports from comp_model)
        classify_fixed_point: Function to classify fixed points (if None, imports from comp_model)
        
    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects
    """
    # Import comp_model only if functions aren't provided
    if model_odes is None or model_nullclines is None or classify_fixed_point is None:
        import competence_circuit_analysis as comp_model
        if model_odes is None:
            model_odes = comp_model.model_odes
        if model_nullclines is None:
            model_nullclines = comp_model.nullclines
        if classify_fixed_point is None:
            classify_fixed_point = comp_model.classify_fixed_point
    
    # Ensure K_range and S_range are arrays
    if isinstance(K_range, list):
        K_range = np.linspace(K_range[0], K_range[1], 100)
    if isinstance(S_range, list):
        S_range = np.linspace(S_range[0], S_range[1], 100)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate nullclines
    S_from_K, S_from_S = model_nullclines(K_range, params)
    
    # Mask for valid values (non-negative and finite)
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    # Plot nullclines
    ax.plot(K_range[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullcline (dK/dt = 0)')
    ax.plot(K_range[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullcline (dS/dt = 0)')
    
    # Plot vector field if requested
    if show_vector_field:
        K_mesh, S_mesh = np.meshgrid(K_range, S_range)
        dK = np.zeros_like(K_mesh)
        dS = np.zeros_like(S_mesh)
        
        for i in range(len(K_range)):
            for j in range(len(S_range)):
                dK[j, i], dS[j, i] = model_odes(0, [K_mesh[j, i], S_mesh[j, i]], params)
        
        # Normalize for better visualization
        magnitude = np.sqrt(dK**2 + dS**2)
        magnitude = np.where(magnitude == 0, 1e-10, magnitude)  # Prevent division by zero
        dK_norm = dK / magnitude
        dS_norm = dS / magnitude
        
        # Vector field with reduced density
        skip = 5  # Skip points for better readability
        ax.quiver(K_mesh[::skip, ::skip], S_mesh[::skip, ::skip], 
                dK_norm[::skip, ::skip], dS_norm[::skip, ::skip],
                color='lightgray', pivot='mid', scale=30, zorder=0)
    
    # Plot trajectories if provided
    if trajectories is not None:
        for i, traj in enumerate(trajectories):
            K_traj, S_traj = traj
            ax.plot(K_traj, S_traj, '-', linewidth=1, alpha=0.7, 
                   color=plt.cm.rainbow(i/len(trajectories)))  # Different colors
    
    # Plot fixed points if available
    if fixed_points:
        # Color mapping for different types
        colors = {
            "Stabiler Knoten": "go",
            "Stabiler Fokus": "go",
            "Instabiler Knoten": "ro",
            "Instabiler Fokus": "ro",
            "Sattelpunkt": "yo",
            "Zentrum": "co",
            "Nicht-hyperbolisch": "mo",
            "Unklassifiziert": "ko"
        }
        
        for i, fp in enumerate(fixed_points):
            K, S = fp
            if K >= 0 and S >= 0 and K <= max(K_range) and S <= max(S_range):
                fp_type = classify_fixed_point(K, S, params)
                ax.plot(K, S, colors.get(fp_type, "ko"), markersize=10, 
                       label=f'FP{i+1}: {fp_type} ({K:.3f}, {S:.3f})')
    
    ax.set_xlim([min(K_range), max(K_range)])
    ax.set_ylim([min(S_range), max(S_range)])
    ax.set_xlabel('ComK Concentration')
    ax.set_ylabel('ComS Concentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig, ax

def plot_time_series(t, K, S, threshold=0.5, title="Time Series of Competence Dynamics"):
    """
    Plots time series of ComK and ComS concentrations.
    
    Args:
        t: Time array
        K: ComK concentration array
        S: ComS concentration array
        threshold: Threshold for competence
        title: Title for the plot
        
    Returns:
        tuple: (fig, ax, competence_periods) Figure, axes, and list of competence periods
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, K, 'b-', label='ComK')
    ax.plot(t, S, 'g-', label='ComS')

    # Automatically adjust axis ranges
    y_max = max(max(K), max(S)) * 1.1  # 10% buffer
    ax.set_ylim(0, y_max)
    
    # Mark competence events when ComK > threshold
    competence_mask = K > threshold
    competence_periods = []
    
    if competence_mask.any():
        # Find transitions (0->1: start, 1->0: end)
        transitions = np.diff(competence_mask.astype(int))
        start_indices = np.where(transitions == 1)[0]
        end_indices = np.where(transitions == -1)[0]
        
        # Handle special cases
        if competence_mask[0]:
            start_indices = np.insert(start_indices, 0, 0)
        if competence_mask[-1]:
            end_indices = np.append(end_indices, len(competence_mask) - 1)
        
        # Ensure we have the same number of start and end points
        n = min(len(start_indices), len(end_indices))
        for i in range(n):
            start_t = t[start_indices[i]]
            end_t = t[end_indices[i]]
            competence_periods.append((start_t, end_t))
            ax.axvspan(start_t, end_t, alpha=0.2, color='red')
    
    # Show the threshold
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Competence Threshold')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Concentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    # Display competence times in the plot
    if competence_periods:
        competence_durations = [end-start for start, end in competence_periods]
        avg_duration = np.mean(competence_durations)
        text_str = f"Number of competence events: {len(competence_periods)}\nAverage duration (Tc): {avg_duration:.2f}"
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top')
    
    return fig, ax, competence_periods

def plot_stochastic_comparison(stochastic_results, output_dir):
    """
    Creates comparison plots for stochastic simulation results.
    
    Args:
        stochastic_results: Dictionary of stochastic simulation results
        output_dir: Directory to save the plots
    """
    if not stochastic_results:
        print("No stochastic results to visualize")
        return
        
    # Extract data for plotting
    param_ids = list(stochastic_results.keys())
    param_labels = [stochastic_results[pid]['name'] for pid in param_ids]
    median_durations = [stochastic_results[pid]['median_duration'] for pid in param_ids]
    cv_durations = [stochastic_results[pid]['cv_duration'] for pid in param_ids]
    median_rise_times = [stochastic_results[pid]['median_rise_time'] for pid in param_ids]
    cv_rise_times = [stochastic_results[pid]['cv_rise_time'] for pid in param_ids]
    
    # Update Duration and Rise Time comparison (side by side)
    plt.figure(figsize=(14, 6))
    
    # Median Values
    plt.subplot(1, 2, 1)
    bar_width = 0.3
    index = np.arange(len(param_ids))
    
    # Plot median values with stronger colors
    bars1 = plt.bar(index - bar_width/2, median_durations, bar_width, color='blue', 
                  alpha=0.9, label='Median Duration')
    bars2 = plt.bar(index + bar_width/2, median_rise_times, bar_width, color='orange', 
                  alpha=0.9, label='Median Rise Time')
    
    plt.xlabel('Parameter Set')
    plt.ylabel('Time')
    plt.title('Comparison of Median Times')
    plt.xticks(index, param_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Place legend outside the first plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Coefficient of Variation
    plt.subplot(1, 2, 2)
    bars3 = plt.bar(index - bar_width/2, cv_durations, bar_width, color='blue', 
                  alpha=0.7, label='Total Duration')
    bars4 = plt.bar(index + bar_width/2, cv_rise_times, bar_width, color='orange', 
                  alpha=0.7, label='Rise Time')
    plt.xlabel('Parameter Set')
    plt.ylabel('Coefficient of Variation')
    plt.title('Timing Variability')
    plt.xticks(index, param_labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Place legend outside the second plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_comparison.png'), bbox_inches='tight')
    plt.close()

def plot_parameter_histograms(excitable_configs, params_base):
    """
    Create histograms of parameters that lead to excitable systems.
    
    Args:
        excitable_configs: List of tuples with excitable parameter configurations
        params_base: Base parameter set
        
    Returns:
        tuple: (fig, axes, param_values) Figure, axes, and parameter values dictionary
    """
    # Create figure with 2x4 subplots for the different parameters
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Parameters and their ranges
    param_names = ['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']
    
    # Standard values from the paper
    standard_params = {
        'n': 2,
        'p': 5,
        'ak': 0.004,
        'bk': 0.07,
        'bs': 0.82,
        'k0': 0.2,
        'k1': 0.222
    }
    
    # Extract parameters from excitable configurations
    param_values = {
        name: [cfg[i] for cfg in excitable_configs] 
        for i, name in enumerate(param_names)
    }
    
    # Plot the histograms
    for i, param in enumerate(param_names):
        if param in param_values and param_values[param]:
            # More bins for better visualization
            n_bins = min(30, len(param_values[param]) // 5)
            axes[i].hist(param_values[param], bins=n_bins, alpha=0.7)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {param}')
            
            # Show standard value
            std_val = standard_params[param]
            axes[i].axvline(std_val, color='b', linestyle='-', 
                          label=f'Standard: {std_val:.3f}')
            
            # Show mean and median for parameters other than Hill coefficients
            if param not in ['n', 'p']:
                mean_val = np.mean(param_values[param])
                median_val = np.median(param_values[param])
                axes[i].axvline(mean_val, color='r', linestyle='--', 
                              label=f'Mean: {mean_val:.3f}')
                axes[i].axvline(median_val, color='g', linestyle=':', 
                              label=f'Median: {median_val:.3f}')
            
            axes[i].legend(fontsize='small')
    
    plt.tight_layout()
    
    return fig, axes, param_values

def plot_excitable_map(excitable_configs, bs_range, bk_range, output_path, 
                      title="Excitable Configurations"):
    """
    Creates a map of excitable configurations in the bs-bk parameter space.
    
    Args:
        excitable_configs: List of excitable configuration dictionaries
        bs_range: Range of bs values
        bk_range: Range of bk values
        output_path: Path to save the figure
        title: Title for the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Excitable regions in parameter space
    if excitable_configs:
        bs_values = [config['bs'] for config in excitable_configs]
        bk_values = [config['bk'] for config in excitable_configs]
        plt.scatter(bs_values, bk_values, c='red', s=50, alpha=0.7, 
                  label='Excitable Configurations')
    
    # Mark standard values
    import competence_circuit_analysis as comp_model
    std_params = comp_model.default_params()
    plt.scatter([std_params['bs']], [std_params['bk']], c='blue', s=200, marker='*', 
              label='Standard')
    
    plt.xlabel('bs (ComS Expression Rate)')
    plt.ylabel('bk (ComK Feedback Strength)')
    plt.title(title)
    plt.xlim(min(bs_range), max(bs_range))
    plt.ylim(min(bk_range), max(bk_range))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_hill_coefficient_comparison(all_results, output_dir):
    """
    Creates visualizations comparing excitable regions for different Hill coefficients.
    
    Args:
        all_results: Dictionary of results for different Hill coefficient combinations
        output_dir: Directory to save the visualizations
    """
    # Extract arrays of Hill coefficients
    n_values = sorted(set([all_results[key]['n'] for key in all_results]))
    p_values = sorted(set([all_results[key]['p'] for key in all_results]))
    
    # Create heatmap of excitable count
    excitable_counts = np.zeros((len(n_values), len(p_values)))
    
    for i, n in enumerate(n_values):
        for j, p in enumerate(p_values):
            key = f'n{n}_p{p}'
            if key in all_results:
                excitable_counts[i, j] = all_results[key]['excitable_count']
    
    # Heatmap
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
    plt.savefig(os.path.join(output_dir, 'excitable_count_heatmap.png'), dpi=300)
    plt.close()
    
    # Bar plot of excitable counts
    plt.figure(figsize=(12, 6))
    
    # Prepare data for bar plot
    x_labels = [f'n={n}, p={p}' for n in n_values for p in p_values]
    counts = [all_results[f'n{n}_p{p}']['excitable_count'] 
             for n in n_values for p in p_values]
    
    # Color coding based on n values
    colors = plt.cm.tab10(np.array([i for i in range(len(n_values)) 
                                 for _ in range(len(p_values))]) % 10)
    
    # Create bar plot
    bars = plt.bar(np.arange(len(x_labels)), counts, color=colors)
    
    # Labels
    plt.xlabel('Hill Coefficients (n, p)')
    plt.ylabel('Number of Excitable Configurations')
    plt.title('Size of Excitable Region for Different Hill Coefficients')
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Legend for n values
    handles = [plt.Rectangle((0,0),1,1, color=plt.cm.tab10(i % 10)) 
              for i in range(len(n_values))]
    labels = [f'n={n}' for n in n_values]
    plt.legend(handles, labels, title='ComK Activation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'excitable_count_barplot.png'), dpi=300)
    plt.close()

def plot_duration_distribution(all_durations, all_rise_times, output_dir):
    """
    Analyzes and plots distributions of competence durations.
    
    Args:
        all_durations: Dictionary of durations by parameter set
        all_rise_times: Dictionary of rise times by parameter set
        output_dir: Directory to save the plots
    """
    # 1. Combined histogram of durations
    plt.figure(figsize=(14, 8))
    bins = np.linspace(0, 50, 25)  # Consistent bins across parameter sets
    
    # Use a colormap for different parameter sets
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_durations)))
    
    for i, (param_name, durations) in enumerate(all_durations.items()):
        plt.hist(durations, bins=bins, alpha=0.5, color=colors[i], 
               label=f'{param_name} (n={len(durations)})')
    
    plt.xlabel('Competence Duration')
    plt.ylabel('Frequency')
    plt.title('Distribution of Competence Durations Across Parameter Sets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'combined_duration_histogram.png'))
    plt.close()
    
    # 2. Combined histogram of rise times
    plt.figure(figsize=(14, 8))
    rise_bins = np.linspace(0, 20, 20)  # Consistent bins for rise times
    
    for i, (param_name, rise_times) in enumerate(all_rise_times.items()):
        plt.hist(rise_times, bins=rise_bins, alpha=0.5, color=colors[i], 
               label=f'{param_name} (n={len(rise_times)})')
    
    plt.xlabel('Rise Time (Time to Maximum ComK)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Rise Times Across Parameter Sets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'combined_risetime_histogram.png'))
    plt.close()
    
    # 3. Combined scatter plot of rise time vs duration
    plt.figure(figsize=(14, 8))
    
    for i, (param_name, durations) in enumerate(all_durations.items()):
        rise_times = all_rise_times[param_name]
        plt.scatter(rise_times, durations, alpha=0.7, color=colors[i], 
                  label=f'{param_name} (n={len(durations)})')
        
        # Add trendline if we have enough points
        if len(durations) > 3:
            z = np.polyfit(rise_times, durations, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(rise_times), max(rise_times), 100)
            plt.plot(x_range, p(x_range), '--', color=colors[i], alpha=0.7)
    
    plt.xlabel('Rise Time')
    plt.ylabel('Total Duration')
    plt.title('Relationship Between Rise Time and Competence Duration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'combined_rise_vs_duration.png'))
    plt.close()

def plot_parameter_correlations(correlation_df, output_dir):
    """
    Creates bar charts showing correlation between parameters and competence dynamics.
    
    Args:
        correlation_df: DataFrame with parameter correlation data
        output_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    
    # Get data from DataFrame
    param_labels = correlation_df['Parameter'].values
    duration_corrs = correlation_df['Correlation with Median Duration'].values
    rise_corrs = correlation_df['Correlation with Median Rise Time'].values
    
    # Create short parameter labels for plotting
    param_labels_short = [label.split(' ')[0] for label in param_labels]
    
    # Duration correlations
    plt.subplot(2, 1, 1)
    bars = plt.bar(param_labels_short, duration_corrs)
    
    # Color bars by sign
    for i, bar in enumerate(bars):
        if duration_corrs[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Correlation')
    plt.title('Parameter Influence on Median Competence Duration')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Rise time correlations
    plt.subplot(2, 1, 2)
    bars = plt.bar(param_labels_short, rise_corrs)
    
    # Color bars by sign
    for i, bar in enumerate(bars):
        if rise_corrs[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.ylabel('Correlation')
    plt.title('Parameter Influence on Median Rise Time')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_correlation_barplot.png'))
    plt.close()