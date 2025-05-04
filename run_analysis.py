import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import competence_circuit_analysis as comp_model
from stochastic_simulation import analyze_stochastic_competence

def print_and_save_parameters(params, results_dir, name="Standard Parameters"):
    """
    Prints and saves the parameters to a text file.
    
    Args:
        params: Parameter dictionary
        results_dir: Directory to save results
        name: Name for the parameter set
    """
    # Print parameters to console
    print(f"\n{name}:")
    print("="*50)
    
    # Core parameters
    print("\nCore Parameters:")
    for param_name in ['ak', 'bk', 'bs', 'k0', 'k1', 'n', 'p']:
        if param_name in params:
            print(f"  {param_name}: {params[param_name]}")
    
    # Extended parameters if present
    if 'GammaK' in params:
        print("\nExtended Parameters (Suel et al.):")
        for param_name in ['GammaK', 'GammaS', 'lambdaK', 'lambdaS', 'deltaK', 'deltaS']:
            if param_name in params:
                print(f"  {param_name}: {params[param_name]}")
    
    # Save to file
    filename = name.lower().replace(" ", "_") + ".txt"
    with open(os.path.join(results_dir, filename), 'w') as f:
        f.write(f"{name}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Core Parameters:\n")
        for param_name in ['ak', 'bk', 'bs', 'k0', 'k1', 'n', 'p']:
            if param_name in params:
                f.write(f"  {param_name}: {params[param_name]}\n")
        
        if 'GammaK' in params:
            f.write("\nExtended Parameters (Suel et al.):\n")
            for param_name in ['GammaK', 'GammaS', 'lambdaK', 'lambdaS', 'deltaK', 'deltaS']:
                if param_name in params:
                    f.write(f"  {param_name}: {params[param_name]}\n")

def create_results_directory():
    """
    Creates a results directory with timestamp.
    
    Returns:
        str: Path to the created directory
    """
    # Create timestamp in format YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Directory name with "results_" and timestamp
    dir_name = f"competence_analysis_{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)
    
    return dir_name

def analyze_excitable_configurations(results_dir, excitable_configs=None):
    """
    Analyze competence circuit with various parameter values forming an excitable system.
    
    Args:
        results_dir: Base directory for results
        excitable_configs: Optional pre-computed excitable configurations
    
    Returns:
        list: Excitable configurations found (if excitable_configs=None)
    """
    # Load standard parameters
    params = comp_model.default_params()
    
    # Print and save parameter details
    print_and_save_parameters(params, results_dir, "Standard Dimensionless Parameters")
    
    # K and S ranges for analysis
    K_range = np.linspace(0, 1, 200)
    S_range = np.linspace(0, 1, 200)
    
    # Find fixed points in the standard system
    fixed_points = comp_model.find_fixed_points(params)
    print("\nFixed points found in system:")
    for i, fp in enumerate(fixed_points):
        K, S = fp
        fp_type = comp_model.classify_fixed_point(K, S, params)
        print(f"  Fixed point {i+1}: K = {K:.4f}, S = {S:.4f} - Type: {fp_type}")
    
    # Create phase diagram with trajectories
    if fixed_points:
        S_mean = np.mean([fp[1] for fp in fixed_points])
        trajectories = []
        
        # Choose initial conditions based on fixed points
        initial_conditions = [
            [0.01, S_mean * 0.8],
            [0.05, S_mean * 0.9],
            [0.1, S_mean * 1.0],
            [0.15, S_mean * 1.1]
        ]
        
        for init in initial_conditions:
            t, K, S = comp_model.simulate_system(params, initial_conditions=init, t_max=100)
            trajectories.append((K, S))
        
        # Create phase diagram
        fig_phase, ax_phase = comp_model.plot_phase_diagram(
            params, K_range, S_range,
            "Phase Diagram of Standard Parameters",
            fixed_points, 
            trajectories=trajectories)
        
        plt.savefig(os.path.join(results_dir, 'phase_diagram.png'))
        plt.close()
    
    # Simulate time series with standard parameters
    t, K, S = comp_model.simulate_system(params, initial_conditions=[0.01, 0.2], t_max=100)
    fig_time, ax_time, comp_periods = comp_model.plot_time_series(t, K, S, 
                                           title="Time Series of Standard Parameters")
    plt.savefig(os.path.join(results_dir, 'time_series.png'))
    plt.close()
    
    # Find excitable configurations if not provided
    if excitable_configs is None:
        print("\nSearching for excitable system configurations...")
        excitable_configs = comp_model.find_excitable_configurations(params, n_samples=500000)
        
        # Save configs for reuse
        with open(os.path.join(results_dir, 'excitable_configs.pkl'), 'wb') as f:
            pickle.dump(excitable_configs, f)
    else:
        print(f"\nUsing {len(excitable_configs)} provided excitable configurations")

    if excitable_configs:
        # Create histograms
        print("\nCreating histograms of parameters for excitable systems...")
        fig, axes, param_values = comp_model.plot_excitable_parameter_histograms(excitable_configs, params)
        plt.savefig(os.path.join(results_dir, 'parameter_histograms.png'), dpi=300)
        plt.close()
        
        # Save mean parameter values to file
        mean_params = {param: np.mean(values) for param, values in param_values.items()}
        with open(os.path.join(results_dir, 'mean_excitable_params.txt'), 'w') as f:
            f.write(f"Mean parameters for excitable systems:\n")
            for param, value in mean_params.items():
                f.write(f"{param}: {value:.4f}\n")
        
        # Create a CSV table with all excitable configurations
        excitable_df = pd.DataFrame(columns=['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1'])
        for i, config in enumerate(excitable_configs):
            row = {}
            for j, param_name in enumerate(['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']):
                row[param_name] = config[j]
            excitable_df = pd.concat([excitable_df, pd.DataFrame([row])], ignore_index=True)
        
        # Save excitable configurations to CSV
        excitable_df.to_csv(os.path.join(results_dir, 'excitable_configs.csv'), index=False)
        
        # Draw example nullclines from a sample excitable configuration
        if len(excitable_configs) > 0:
            sample_config_idx = len(excitable_configs) // 2
            sample_params = params.copy()
            for i, param_name in enumerate(['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']):
                sample_params[param_name] = excitable_configs[sample_config_idx][i]
            
            fig_nullclines, ax_nullclines = comp_model.plot_example_nullclines(
                sample_params, 
                f"Nullclines for an Excitable System"
            )
            plt.savefig(os.path.join(results_dir, 'excitable_system_nullclines.png'), dpi=300)
            plt.close()
            
            # Time series for the excitable system
            t_excite, K_excite, S_excite = comp_model.simulate_system(
                sample_params, initial_conditions=[0.01, 0.2], t_max=200)
            
            _, _, comp_periods_excite = comp_model.plot_time_series(
                t_excite, K_excite, S_excite,
                title=f"Time Series of Excitable System")
            
            plt.savefig(os.path.join(results_dir, 'excitable_system_time_series.png'))
            plt.close()
        
        # Visualize excitable configurations in parameter space (bk vs bs)
        fig_excite, ax_excite = plt.subplots(figsize=(8, 6))
        
        # Extract bk and bs from configurations (at positions 3 and 4)
        excite_bk = [config[3] for config in excitable_configs]
        excite_bs = [config[4] for config in excitable_configs]
        
        ax_excite.scatter(excite_bk, excite_bs, c='red', s=50, alpha=0.7, label='Excitable Configurations')
        ax_excite.set_xlabel('bk (ComK feedback strength)')
        ax_excite.set_ylabel('bs (ComS expression rate)')
        ax_excite.set_title(f'Excitable System Configurations')
        ax_excite.grid(True)
        
        # Add the standard parameters as a reference point
        ax_excite.scatter([params['bk']], [params['bs']], c='blue', s=100, marker='*', 
                         label='Standard Parameters')
        
        ax_excite.legend()
        plt.savefig(os.path.join(results_dir, 'excitable_configurations.png'))
        plt.close()
        
    return excitable_configs

def select_diverse_excitable_configs(excitable_configs, n_configs=10):
    """
    Select a diverse set of excitable configurations for detailed analysis.
    
    Args:
        excitable_configs: List of excitable configurations
        n_configs: Number of configurations to select
        
    Returns:
        list: Selected excitable configurations
    """
    if len(excitable_configs) <= n_configs:
        return excitable_configs
    
    # Convert to numpy array for easier manipulation
    configs_array = np.array(excitable_configs)
    
    # Extract parameter ranges to normalize
    param_mins = np.min(configs_array, axis=0)
    param_maxs = np.max(configs_array, axis=0)
    param_ranges = param_maxs - param_mins
    
    # Avoid division by zero
    param_ranges = np.where(param_ranges > 0, param_ranges, 1)
    
    # Normalize the configurations
    normalized_configs = (configs_array - param_mins) / param_ranges
    
    # Use K-means to find cluster centers
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_configs, random_state=42)
    kmeans.fit(normalized_configs)
    
    # Get indices of configurations closest to cluster centers
    selected_indices = []
    for center in kmeans.cluster_centers_:
        # Calculate Euclidean distance to center
        distances = np.sqrt(np.sum((normalized_configs - center)**2, axis=1))
        # Find index of closest configuration
        closest_idx = np.argmin(distances)
        selected_indices.append(closest_idx)
    
    # Return selected configurations
    return [excitable_configs[i] for i in selected_indices]

def create_parameter_table(stochastic_param_sets, param_names, results_dir):
    """
    Creates a table of parameter values for each parameter set with descriptive labels.
    
    Args:
        stochastic_param_sets: List of parameter dictionaries
        param_names: List of names for each parameter set
        results_dir: Directory to save results
    """
    # Create a directory for parameter analysis
    param_dir = os.path.join(results_dir, 'parameter_analysis')
    os.makedirs(param_dir, exist_ok=True)
    
    # Standard parameters for reference
    std_params = comp_model.default_params()
    
    # Create a DataFrame to store parameter values
    df = pd.DataFrame(columns=['Parameter Set', 'Description', 'n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1'])
    
    # Fill the DataFrame with parameter values
    for i, params in enumerate(stochastic_param_sets):
        # Create a row for this parameter set
        row = {
            'Parameter Set': param_names[i],
            'Description': get_parameter_description(params, std_params),
            'n': params.get('n', '-'),
            'p': params.get('p', '-'),
            'ak': params.get('ak', '-'),
            'bk': params.get('bk', '-'),
            'bs': params.get('bs', '-'),
            'k0': params.get('k0', '-'),
            'k1': params.get('k1', '-')
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(param_dir, 'parameter_table.csv'), index=False)
    
    # Create a more readable text file with the same information
    with open(os.path.join(param_dir, 'parameter_table.txt'), 'w') as f:
        f.write("Parameter Sets for Stochastic Simulations\n")
        f.write("========================================\n\n")
        
        for i, params in enumerate(stochastic_param_sets):
            f.write(f"{i+1}. {param_names[i]}\n")
            f.write(f"   Description: {get_parameter_description(params, std_params)}\n")
            f.write(f"   n (ComK Hill): {params.get('n', '-')}\n")
            f.write(f"   p (ComS Hill): {params.get('p', '-')}\n")
            f.write(f"   ak (ComK basal): {params.get('ak', '-')}\n")
            f.write(f"   bk (ComK feedback): {params.get('bk', '-')}\n")
            f.write(f"   bs (ComS expression): {params.get('bs', '-')}\n")
            f.write(f"   k0 (ComK activation): {params.get('k0', '-')}\n")
            f.write(f"   k1 (ComS repression): {params.get('k1', '-')}\n\n")
    
    return df

def get_parameter_description(params, std_params):
    """
    Generates a descriptive label for a parameter set based on how it differs from standard parameters.
    
    Args:
        params: Parameter dictionary
        std_params: Standard parameter dictionary for comparison
        
    Returns:
        str: Descriptive label
    """
    descriptions = []
    
    # Check each parameter for significant differences
    if params.get('n', 0) > std_params.get('n', 0) * 1.2:
        descriptions.append("High ComK Hill")
    elif params.get('n', 0) < std_params.get('n', 0) * 0.8:
        descriptions.append("Low ComK Hill")
        
    if params.get('p', 0) > std_params.get('p', 0) * 1.2:
        descriptions.append("High ComS Hill")
    elif params.get('p', 0) < std_params.get('p', 0) * 0.8:
        descriptions.append("Low ComS Hill")
        
    if params.get('ak', 0) > std_params.get('ak', 0) * 1.2:
        descriptions.append("High ComK Basal")
    elif params.get('ak', 0) < std_params.get('ak', 0) * 0.8:
        descriptions.append("Low ComK Basal")
        
    if params.get('bk', 0) > std_params.get('bk', 0) * 1.2:
        descriptions.append("High ComK Feedback")
    elif params.get('bk', 0) < std_params.get('bk', 0) * 0.8:
        descriptions.append("Low ComK Feedback")
        
    if params.get('bs', 0) > std_params.get('bs', 0) * 1.2:
        descriptions.append("High ComS Expression")
    elif params.get('bs', 0) < std_params.get('bs', 0) * 0.8:
        descriptions.append("Low ComS Expression")
    
    # If no significant differences, it's similar to standard
    if not descriptions:
        return "Similar to Standard"
    
    return ", ".join(descriptions)

def run_stochastic_analysis(excitable_configs, results_dir):
    """
    Run stochastic analysis with optimized noise amplification factor
    and more parameter sets including the requested variations.
    
    Args:
        excitable_configs: List of excitable configurations
        results_dir: Directory to save results
        
    Returns:
        dict: Results of stochastic simulations
    """
    print("\n=== PERFORMING STOCHASTIC SIMULATIONS WITH OPTIMIZED NOISE AMPLIFICATION ===")
    
    # Create directory for stochastic analysis
    stochastic_dir = os.path.join(results_dir, 'stochastic_analysis')
    os.makedirs(stochastic_dir, exist_ok=True)
    
    # Load standard parameters
    params = comp_model.default_params()
    
    # Prepare parameter sets for stochastic analysis
    stochastic_param_sets = []
    param_names = []
    
    # Add standard parameters
    stochastic_param_sets.append(params.copy())
    param_names.append("Standard Parameters")
    
    # Select 10 diverse excitable configurations (will be used to create 10 parameter sets)
    if excitable_configs and len(excitable_configs) >= 10:
        selected_configs = select_diverse_excitable_configs(excitable_configs, n_configs=10)
        
        for i, config in enumerate(selected_configs):
            config_params = params.copy()
            for j, param_name in enumerate(['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']):
                config_params[param_name] = config[j]
            stochastic_param_sets.append(config_params)
            param_names.append(f"Excitable Config {i+1}")
    
    # Add the specifically requested parameter variations
    
    # Set with higher basal ComK expression
    high_comk_basal = params.copy()
    high_comk_basal['ak'] = params['ak'] * 2.0
    stochastic_param_sets.append(high_comk_basal)
    param_names.append("High ComK Basal (2x)")
    
    # Set with higher ComK feedback
    high_comk_feedback = params.copy() 
    high_comk_feedback['bk'] = params['bk'] * 1.5
    stochastic_param_sets.append(high_comk_feedback)
    param_names.append("High ComK Feedback (1.5x)")
    
    # Set with higher ComS expression
    high_coms_expression = params.copy() 
    high_coms_expression['bs'] = params['bs'] * 1.5
    stochastic_param_sets.append(high_coms_expression)
    param_names.append("High ComS Expression (1.5x)")
    
    # Set with lower ComS expression
    low_coms_expression = params.copy() 
    low_coms_expression['bs'] = params['bs'] * 0.75
    stochastic_param_sets.append(low_coms_expression)
    param_names.append("Low ComS Expression (0.75x)")
    
    # Create and save parameter table
    param_table = create_parameter_table(stochastic_param_sets, param_names, results_dir)
    
    # Noise parameters with amplification factor of 10 based on prior analysis
    noise_params = {
        'theta': 1.0,      # Mean reversion rate
        'mu': 0.0,         # Mean value (noise is centered around zero)
        'sigma': 0.3,      # Noise amplitude
        'dt': 0.1          # Time step for numerical integration
    }
    
    # Run stochastic simulations with enhanced analysis
    stochastic_results = analyze_stochastic_competence(
        comp_model.model_odes,
        stochastic_param_sets,
        noise_params,
        stochastic_dir,
        n_simulations=50,     # 50 simulations per parameter set
        t_max=500,            # Longer simulation time
        dt=0.01,              # Time step
        threshold=0.5,        # Competence threshold
        param_names=param_names
    )
    
    # Save stochastic results for later analysis
    if stochastic_results:
        with open(os.path.join(stochastic_dir, 'stochastic_results.pkl'), 'wb') as f:
            pickle.dump(stochastic_results, f)
        
        # Print summary of stochastic results
        print(f"\nStochastic simulation results summary:")
        for param_id, result in stochastic_results.items():
            param_name = result.get('name', param_id)
            print(f"  {param_name}:")
            print(f"    Mean duration = {result['mean_duration']:.2f}")
            print(f"    Median duration = {result['median_duration']:.2f}")
            print(f"    Duration CV = {result['cv_duration']:.2f}")
            print(f"    Mean rise time = {result['mean_rise_time']:.2f}")
            print(f"    Initiation probability = {result['init_probability']:.4f}")
        
        # Create CSV with excitable configurations duration statistics
        create_excitable_configs_statistics_csv(stochastic_results, stochastic_dir)
    
    return stochastic_results

def create_excitable_configs_statistics_csv(stochastic_results, stochastic_dir):
    """
    Create a CSV file with statistics for all excitable configurations.
    
    Args:
        stochastic_results: Results from stochastic simulations
        stochastic_dir: Directory to save results
    """
    # Collect data for excitable configurations
    excitable_stats = []
    
    for param_id, result in stochastic_results.items():
        param_name = result.get('name', param_id)
        
        # Skip if not an excitable configuration
        if not param_name.startswith('Excitable Config'):
            continue
        
        # Extract parameters
        params = result.get('params', {})
        
        # Get statistics
        mean_duration = result.get('mean_duration', 0)
        median_duration = result.get('median_duration', 0)
        mean_rise_time = result.get('mean_rise_time', 0)
        median_rise_time = result.get('median_rise_time', 0)
        event_count = len(result.get('all_durations', []))
        
        # Add to list
        excitable_stats.append({
            'Parameter Set': param_name,
            'n': params.get('n', '-'),
            'p': params.get('p', '-'),
            'ak': params.get('ak', '-'),
            'bk': params.get('bk', '-'),
            'bs': params.get('bs', '-'),
            'k0': params.get('k0', '-'),
            'k1': params.get('k1', '-'),
            'Mean Duration': mean_duration,
            'Median Duration': median_duration,
            'Mean Rise Time': mean_rise_time,
            'Median Rise Time': median_rise_time,
            'Event Count': event_count
        })
    
    # Create DataFrame and save to CSV
    if excitable_stats:
        excitable_df = pd.DataFrame(excitable_stats)
        excitable_df.to_csv(os.path.join(stochastic_dir, 'excitable_configs_statistics.csv'), index=False)

def analyze_duration_distribution(stochastic_results, results_dir):
    """
    Analyze the distribution of competence durations and create 
    visualizations focused on understanding their patterns.
    
    Args:
        stochastic_results: Results from stochastic simulations
        results_dir: Directory to save results
    """
    if not stochastic_results:
        print("No stochastic results to analyze")
        return
        
    # Create directory for duration analysis
    duration_dir = os.path.join(results_dir, 'duration_analysis')
    os.makedirs(duration_dir, exist_ok=True)
    
    # Collect all durations by parameter set
    all_durations = {}
    all_rise_times = {}
    all_rise_to_duration_ratios = {}
    
    for param_id, result in stochastic_results.items():
        param_name = result.get('name', param_id)
        durations = result.get('all_durations', [])
        rise_times = result.get('all_rise_times', [])
        
        if durations:
            all_durations[param_name] = durations
            all_rise_times[param_name] = rise_times
            
            # Calculate rise to duration ratios on a per-event basis
            ratios = [rise/duration for rise, duration in zip(rise_times, durations)]
            all_rise_to_duration_ratios[param_name] = ratios
    
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
    plt.savefig(os.path.join(duration_dir, 'combined_duration_histogram.png'))
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
    plt.savefig(os.path.join(duration_dir, 'combined_risetime_histogram.png'))
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
    plt.savefig(os.path.join(duration_dir, 'combined_rise_vs_duration.png'))
    plt.close()
    
    # 4. Boxplot comparisons
    plt.figure(figsize=(14, 8))
    
    # Durations boxplot
    plt.subplot(1, 2, 1)
    duration_data = [all_durations[param] for param in all_durations.keys()]
    plt.boxplot(duration_data, labels=list(all_durations.keys()))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Duration')
    plt.title('Competence Durations')
    plt.grid(True, alpha=0.3)
    
    # Rise times boxplot
    plt.subplot(1, 2, 2)
    rise_data = [all_rise_times[param] for param in all_rise_times.keys()]
    plt.boxplot(rise_data, labels=list(all_rise_times.keys()))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Rise Time')
    plt.title('Competence Rise Times')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(duration_dir, 'boxplot_comparison.png'))
    plt.close()
    
    # 5. Kernel Density Estimation of Duration Distributions
    plt.figure(figsize=(14, 8))
    
    # Use KDE to get a smooth estimate of the probability distribution
    from scipy.stats import gaussian_kde
    
    for i, (param_name, durations) in enumerate(all_durations.items()):
        if len(durations) > 5:  # Need enough points for KDE
            # Use Gaussian KDE for smooth density estimation
            kde = gaussian_kde(durations)
            x_range = np.linspace(0, 50, 200)
            plt.plot(x_range, kde(x_range), '-', linewidth=2, color=colors[i], 
                   label=f'{param_name} (n={len(durations)})')
    
    plt.xlabel('Competence Duration')
    plt.ylabel('Probability Density')
    plt.title('Kernel Density Estimation of Competence Duration Distributions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(duration_dir, 'duration_kde.png'))
    plt.close()
    
    # 6. Save statistics to CSV file
    # Prepare data for CSV
    stats_data = []
    for param_name in all_durations.keys():
        durations = all_durations[param_name]
        rise_times = all_rise_times[param_name]
        ratios = all_rise_to_duration_ratios[param_name]
        
        mean_dur = np.mean(durations)
        median_dur = np.median(durations)
        cv_dur = np.std(durations) / mean_dur if mean_dur > 0 else 0
        
        mean_rise = np.mean(rise_times)
        median_rise = np.median(rise_times)
        cv_rise = np.std(rise_times) / mean_rise if mean_rise > 0 else 0
        
        mean_ratio = np.mean(ratios)
        median_ratio = np.median(ratios)
        
        stats_data.append({
            'Parameter Set': param_name,
            'Count': len(durations),
            'Mean Duration': mean_dur,
            'Median Duration': median_dur,
            'CV Duration': cv_dur,
            'Mean Rise': mean_rise,
            'Median Rise': median_rise,
            'CV Rise': cv_rise,
            'Mean Ratio': mean_ratio,
            'Median Ratio': median_ratio
        })
    
    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(duration_dir, 'duration_statistics.csv'), index=False)

def analyze_parameter_influence_on_excitable(stochastic_results, results_dir):
    """
    Analyze how each parameter influences median duration and median rise time
    specifically for excitable configurations.
    
    Args:
        stochastic_results: Results from stochastic simulations
        results_dir: Directory to save results
    """
    if not stochastic_results:
        print("No stochastic results to analyze")
        return
    
    # Create directory for parameter influence analysis
    influence_dir = os.path.join(results_dir, 'parameter_influence')
    os.makedirs(influence_dir, exist_ok=True)
    
    # Collect data on excitable configurations only
    excitable_data = []
    
    for param_id, result in stochastic_results.items():
        param_name = result.get('name', param_id)
        
        # Only consider excitable configurations
        if not param_name.startswith('Excitable Config'):
            continue
            
        params = result.get('params', {})
        
        # Skip if we don't have the parameters or no competence events
        if not params or result.get('median_duration', 0) == 0:
            continue
        
        # Add to list
        excitable_data.append({
            'Parameter Set': param_name,
            'n': params.get('n', 0),
            'p': params.get('p', 0),
            'ak': params.get('ak', 0),
            'bk': params.get('bk', 0),
            'bs': params.get('bs', 0),
            'k0': params.get('k0', 0),
            'k1': params.get('k1', 0),
            'median_duration': result.get('median_duration', 0),
            'median_rise_time': result.get('median_rise_time', 0)
        })
    
    # If no excitable data, exit
    if not excitable_data:
        print("No excitable configurations with competence events")
        return
    
    # Convert to DataFrame
    excitable_df = pd.DataFrame(excitable_data)
    
    # Save to CSV
    excitable_df.to_csv(os.path.join(influence_dir, 'excitable_parameter_influence.csv'), index=False)
    
    # Parameters to analyze
    param_names = ['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']
    param_labels = ['ComK Hill (n)', 'ComS Hill (p)', 'ComK Basal (ak)', 
                   'ComK Feedback (bk)', 'ComS Expression (bs)', 
                   'ComK Activation (k0)', 'ComS Repression (k1)']
    
    # Metrics to analyze
    metrics = ['median_duration', 'median_rise_time']
    metric_labels = ['Median Duration', 'Median Rise Time']
    
    # Create scatter plots for each parameter's influence on median duration and rise time
    for param_idx, param in enumerate(param_names):
        plt.figure(figsize=(12, 10))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 1, i+1)
            
            # Extract values
            x = excitable_df[param].values
            y = excitable_df[metric].values
            labels = excitable_df['Parameter Set'].values
            
            # Scatter plot
            plt.scatter(x, y, alpha=0.7, s=80)
            
            # Add labels for each point
            for j, label in enumerate(labels):
                plt.annotate(label, (x[j], y[j]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8)
            
            # Add trendline if we have enough points
            if len(x) > 3:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(min(x) * 0.9, max(x) * 1.1, 100)
                    plt.plot(x_range, p(x_range), 'r--', 
                           label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
                    plt.legend()
                    
                    # Calculate correlation coefficient
                    corr = np.corrcoef(x, y)[0, 1]
                    plt.text(0.05, 0.95, f"Correlation: {corr:.2f}", 
                           transform=plt.gca().transAxes, 
                           fontsize=10, verticalalignment='top')
                except:
                    pass  # Skip trendline if fitting fails
            
            # Labels
            plt.xlabel(param_labels[param_idx])
            plt.ylabel(metric_labels[i])
            plt.title(f'Effect of {param_labels[param_idx]} on {metric_labels[i]}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(influence_dir, f'excitable_{param}_influence.png'))
        plt.close()
    
    # Create summary of parameter correlations with median duration and rise time
    correlation_data = []
    
    for param_idx, param in enumerate(param_names):
        param_corrs = []
        
        for metric in metrics:
            # Extract values
            x = excitable_df[param].values
            y = excitable_df[metric].values
            
            # Calculate correlation if we have enough points
            if len(x) > 3:
                corr = np.corrcoef(x, y)[0, 1]
                param_corrs.append(corr)
            else:
                param_corrs.append(0)
        
        correlation_data.append({
            'Parameter': param_labels[param_idx],
            'Correlation with Median Duration': param_corrs[0],
            'Correlation with Median Rise Time': param_corrs[1]
        })
    
    # Create DataFrame and save to CSV
    correlation_df = pd.DataFrame(correlation_data)
    correlation_df.to_csv(os.path.join(influence_dir, 'parameter_correlations.csv'), index=False)
    
    # Create bar chart of parameter correlations
    plt.figure(figsize=(12, 8))
    
    # Duration correlations
    plt.subplot(2, 1, 1)
    duration_corrs = correlation_df['Correlation with Median Duration'].values
    param_labels_short = [label.split(' ')[0] for label in param_labels]
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
    rise_corrs = correlation_df['Correlation with Median Rise Time'].values
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
    plt.savefig(os.path.join(influence_dir, 'parameter_correlation_barplot.png'))
    plt.close()

def main():
    # Create results directory
    results_dir = create_results_directory()
    print(f"All results will be saved in directory '{results_dir}'")

    # Analyze excitable configurations
    print("\n=== ANALYZING EXCITABLE CONFIGURATIONS ===")
    excitable_configs = analyze_excitable_configurations(results_dir)
    
    # Run stochastic analysis with the optimized noise amplification
    stochastic_results = run_stochastic_analysis(excitable_configs, results_dir)
    
    # Analyze distribution of competence durations
    print("\n=== ANALYZING DISTRIBUTIONS OF COMPETENCE DURATIONS ===")
    analyze_duration_distribution(stochastic_results, results_dir)
    
    # Analyze parameter influence on dynamics for excitable configurations
    print("\n=== ANALYZING PARAMETER INFLUENCE ON DYNAMICS OF EXCITABLE CONFIGURATIONS ===")
    analyze_parameter_influence_on_excitable(stochastic_results, results_dir)
    
    print("\nAnalysis completed. Results saved in images and statistics files.")

if __name__ == "__main__":
    main()