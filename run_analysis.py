"""
Main script for running competence circuit analyses.

This script ties together the various analysis modules to run comprehensive
simulations and analyses of the B. subtilis competence circuit model.
It allows for customization of which analyses to run and their parameters.
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json

# Import refactored modules
import competence_circuit_analysis as comp_model
from helpers import create_results_directory, print_and_save_parameters, save_results, is_excitable
from visualization import plot_phase_diagram, plot_time_series, plot_parameter_correlations
from simulation import simulate_system, analyze_stochastic_competence, analyze_amplification_factors
from parameter_search import find_excitable_configurations, select_representative_configs
from parameter_search import search_hill_coefficient_space, search_suel_parameter_space
from parameter_search import select_diverse_excitable_configs

# Define default analysis parameters
DEFAULT_PARAMS = {
    # General parameters
    "output_prefix": "competence_analysis",
    
    # Which analyses to run (True/False)
    "run_standard_analysis": True,
    "run_excitable_search": True,
    "run_hill_analysis": True,
    "run_suel_search": True,
    "run_stochastic_analysis": True,
    "run_amplification_analysis": True,
    
    # Standard parameter analysis
    "standard_t_max": 100,
    
    # Excitable configuration search
    "excitable_n_samples": 50000,
    
    # Hill coefficient analysis
    "hill_grid_size": 50,
    "hill_extended_search": True,
    
    # Suel parameter space search
    "suel_n_points": 100,
    
    # Stochastic analysis
    "stochastic_n_simulations": 50,
    "stochastic_t_max": 500,
    "stochastic_amplification": 10,
    
    # Amplification factor analysis
    "amplification_factors": [1, 3, 5, 7, 10],
    "amplification_n_simulations": 50,
    "amplification_t_max": 500
}

def analyze_standard_parameters(results_dir, params):
    """
    Analyzes the standard parameter set for the competence circuit model.
    
    Args:
        results_dir: Directory to save results
        params: Analysis parameters dictionary
        
    Returns:
        dict: Information about the standard parameter set
    """
    # Load standard parameters
    std_params = comp_model.default_params()
    
    print("\n=== ANALYZING STANDARD PARAMETERS ===")
    
    # Print and save parameter details
    print_and_save_parameters(std_params, results_dir, "Standard Parameters")
    
    # K and S ranges for analysis
    K_range = np.linspace(0, 1, 200)
    S_range = np.linspace(0, 1, 200)
    
    # Find fixed points in the standard system
    fixed_points = comp_model.find_fixed_points(std_params)
    print("\nFixed points found in system:")
    for i, fp in enumerate(fixed_points):
        K, S = fp
        fp_type = comp_model.classify_fixed_point(K, S, std_params)
        print(f"  Fixed point {i+1}: K = {K:.4f}, S = {S:.4f} - Type: {fp_type}")
    
    # Check if system is excitable
    is_exc, info = is_excitable(std_params, comp_model.model_odes,
                             find_fixed_points_func=comp_model.find_fixed_points,
                             classify_fixed_point_func=comp_model.classify_fixed_point)
    
    if is_exc:
        print("The standard system is EXCITABLE!")
    else:
        print("The standard system is NOT excitable according to our criteria.")
    
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
            t, K, S = simulate_system(comp_model.model_odes, std_params, 
                                    initial_conditions=init, 
                                    t_max=params["standard_t_max"])
            trajectories.append((K, S))
        
        # Create phase diagram
        fig_phase, ax_phase = plot_phase_diagram(
            std_params, K_range, S_range,
            "Phase Diagram of Standard Parameters",
            fixed_points, 
            trajectories=trajectories,
            model_odes=comp_model.model_odes,
            model_nullclines=comp_model.nullclines,
            classify_fixed_point=comp_model.classify_fixed_point)
        
        plt.savefig(os.path.join(results_dir, 'phase_diagram.png'))
        plt.close()
    
    # Simulate time series with standard parameters
    t, K, S = simulate_system(comp_model.model_odes, std_params, 
                            initial_conditions=[0.01, 0.2], 
                            t_max=params["standard_t_max"])
    fig_time, ax_time, comp_periods = plot_time_series(t, K, S, 
                                           title="Time Series of Standard Parameters")
    plt.savefig(os.path.join(results_dir, 'time_series.png'))
    plt.close()
    
    return {'params': std_params, 'is_excitable': is_exc, 'info': info}

def run_excitable_configuration_analysis(results_dir, params):
    """
    Analyzes the competence circuit with various parameter values to find
    excitable system configurations.
    
    Args:
        results_dir: Directory to save results
        params: Analysis parameters dictionary
        
    Returns:
        list: Excitable configurations found
    """
    print("\n=== RUNNING EXCITABLE CONFIGURATION ANALYSIS ===")
    
    # Load standard parameters
    std_params = comp_model.default_params()
    
    # Create directory for excitable configurations
    excitable_dir = os.path.join(results_dir, 'excitable_configurations')
    os.makedirs(excitable_dir, exist_ok=True)
    
    # Search for excitable configurations
    excitable_configs = find_excitable_configurations(
        comp_model.model_odes,
        comp_model.classify_fixed_point,
        comp_model.find_fixed_points,
        std_params, excitable_dir, 
        n_samples=params["excitable_n_samples"])
    
    return excitable_configs

def run_hill_coefficient_analysis(results_dir, params):
    """
    Analyzes the effect of different Hill coefficients on the
    robustness of the excitable region in the parameter space.
    
    Args:
        results_dir: Directory to save results
        params: Analysis parameters dictionary
        
    Returns:
        dict: Results for all Hill coefficient combinations
    """
    print("\n=== RUNNING HILL COEFFICIENT ANALYSIS ===")
    
    # Create directory for Hill coefficient analysis
    hill_dir = os.path.join(results_dir, 'hill_coefficient_analysis')
    os.makedirs(hill_dir, exist_ok=True)
    
    # Analyze Hill coefficient combinations
    hill_results = search_hill_coefficient_space(
        comp_model.model_odes,
        comp_model.classify_fixed_point,
        comp_model.find_fixed_points,
        hill_dir, 
        grid_size=params["hill_grid_size"], 
        extended_search=params["hill_extended_search"])
    
    return hill_results

def run_suel_parameter_search(results_dir, params):
    """
    Performs high-resolution search in the Suel et al. parameter space.
    
    Args:
        results_dir: Directory to save results
        params: Analysis parameters dictionary
        
    Returns:
        tuple: (excitable_configs, representative_configs)
    """
    print("\n=== RUNNING SUEL PARAMETER SPACE SEARCH ===")
    
    # Create directory for Suel parameter search
    suel_dir = os.path.join(results_dir, 'suel_parameter_search')
    os.makedirs(suel_dir, exist_ok=True)
    
    # Perform search
    excitable_configs = search_suel_parameter_space(
        comp_model.model_odes,
        comp_model.classify_fixed_point,
        comp_model.find_fixed_points,
        suel_dir, 
        n_points=params["suel_n_points"])
    
    # If we found excitable configurations, select representative ones
    if excitable_configs:
        representative_configs = select_representative_configs(excitable_configs)
        
        # Save representative configurations
        save_results(representative_configs, 'representative_configs.pkl', suel_dir)
    else:
        representative_configs = []
    
    return excitable_configs, representative_configs

def run_stochastic_analysis(results_dir, params, excitable_configs=None):
    """
    Runs stochastic simulations with optimized noise amplification.
    
    Args:
        results_dir: Directory to save results
        params: Analysis parameters dictionary
        excitable_configs: Optional list of excitable configurations
        
    Returns:
        dict: Results of stochastic simulations
    """
    print("\n=== RUNNING STOCHASTIC ANALYSIS ===")
    
    # Create directory for stochastic analysis
    stochastic_dir = os.path.join(results_dir, 'stochastic_analysis')
    os.makedirs(stochastic_dir, exist_ok=True)
    
    # Load standard parameters
    std_params = comp_model.default_params()
    
    # Prepare parameter sets for stochastic analysis
    stochastic_param_sets = []
    param_names = []
    
    # Add standard parameters
    stochastic_param_sets.append(std_params.copy())
    param_names.append("Standard Parameters")
    
    # Add excitable configurations if provided
    if excitable_configs and len(excitable_configs) >= 10:
        # Select 10 diverse excitable configurations
        selected_excitable_configs = select_diverse_excitable_configs(excitable_configs, n_configs=10)
        
        for i, config in enumerate(selected_excitable_configs):
            config_params = std_params.copy()
            for j, param_name in enumerate(['n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1']):
                config_params[param_name] = config[j]
            stochastic_param_sets.append(config_params)
            param_names.append(f"Excitable Config {i+1}")
    
    # Add specifically requested parameter variations
    
    # Set with higher basal ComK expression
    high_comk_basal = std_params.copy()
    high_comk_basal['ak'] = std_params['ak'] * 2.0
    stochastic_param_sets.append(high_comk_basal)
    param_names.append("High ComK Basal (2x)")
    
    # Set with higher ComK feedback
    high_comk_feedback = std_params.copy() 
    high_comk_feedback['bk'] = std_params['bk'] * 1.5
    stochastic_param_sets.append(high_comk_feedback)
    param_names.append("High ComK Feedback (1.5x)")
    
    # Set with higher ComS expression
    high_coms_expression = std_params.copy() 
    high_coms_expression['bs'] = std_params['bs'] * 1.5
    stochastic_param_sets.append(high_coms_expression)
    param_names.append("High ComS Expression (1.5x)")
    
    # Set with lower ComS expression
    low_coms_expression = std_params.copy() 
    low_coms_expression['bs'] = std_params['bs'] * 0.75
    stochastic_param_sets.append(low_coms_expression)
    param_names.append("Low ComS Expression (0.75x)")
    
    # Create and save parameter table
    create_parameter_table(stochastic_param_sets, param_names, stochastic_dir)
    
    # Noise parameters with amplification factor based on parameters
    noise_params = {
        'theta': 1.0,      # Mean reversion rate
        'mu': 0.0,         # Mean value (noise is centered around zero)
        'sigma': 0.3,      # Noise amplitude
        'dt': 0.1          # Time step for numerical integration
    }
    
    # Run stochastic simulations
    stochastic_results = analyze_stochastic_competence(
        comp_model.model_odes,
        stochastic_param_sets,
        noise_params,
        stochastic_dir,
        n_simulations=params["stochastic_n_simulations"],
        t_max=params["stochastic_t_max"],
        dt=0.01,              # Time step
        threshold=0.5,        # Competence threshold
        param_names=param_names,
        amplification_factor=params["stochastic_amplification"]
    )
    
    # Analyze distribution of competence durations
    analyze_duration_distribution(stochastic_results, stochastic_dir)
    
    # Analyze parameter influence on competence dynamics
    analyze_parameter_influence(stochastic_results, stochastic_dir)
    
    return stochastic_results

def run_amplification_analysis(results_dir, params):
    """
    Analyzes the effect of different noise amplification factors on competence dynamics.
    
    Args:
        results_dir: Directory to save results
        params: Analysis parameters dictionary
        
    Returns:
        dict: Results for each amplification factor
    """
    print("\n=== RUNNING NOISE AMPLIFICATION FACTOR ANALYSIS ===")
    
    # Create directory for amplification analysis
    amp_dir = os.path.join(results_dir, 'amplification_analysis')
    os.makedirs(amp_dir, exist_ok=True)
    
    # Load standard parameters
    std_params = comp_model.default_params()
    
    # Run amplification analysis
    amp_results = analyze_amplification_factors(
        comp_model.model_odes,
        std_params,
        amp_dir,
        amplification_factors=params["amplification_factors"],
        n_simulations=params["amplification_n_simulations"],
        t_max=params["amplification_t_max"]
    )
    
    return amp_results

def create_parameter_table(stochastic_param_sets, param_names, results_dir):
    """
    Creates a table of parameter values for each parameter set.
    
    Args:
        stochastic_param_sets: List of parameter dictionaries
        param_names: List of names for each parameter set
        results_dir: Directory to save results
    """
    # Create directory for parameter analysis
    param_dir = os.path.join(results_dir, 'parameter_analysis')
    os.makedirs(param_dir, exist_ok=True)
    
    # Standard parameters for reference
    std_params = comp_model.default_params()
    
    # Create DataFrame for parameter values
    df = pd.DataFrame(columns=['Parameter Set', 'Description', 'n', 'p', 'ak', 'bk', 'bs', 'k0', 'k1'])
    
    # Fill DataFrame with parameter values
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
    
    # Save DataFrame to CSV
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
    Generates a descriptive label for a parameter set based on how it differs
    from standard parameters.
    
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

def analyze_duration_distribution(stochastic_results, results_dir):
    """
    Analyzes the distribution of competence durations and creates
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
    
    # Create visualizations
    from visualization import plot_duration_distribution
    plot_duration_distribution(all_durations, all_rise_times, duration_dir)
    
    # Save statistics to CSV file
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

def analyze_parameter_influence(stochastic_results, results_dir):
    """
    Analyzes how each parameter influences competence dynamics,
    especially for excitable configurations.
    
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
    
    # Create scatter plots for each parameter's influence
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
    
    # Create summary of parameter correlations
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
    plot_parameter_correlations(correlation_df, influence_dir)

def load_parameters(config_file=None):
    """
    Loads parameters from a JSON configuration file, or returns defaults if none specified.
    
    Args:
        config_file: Path to JSON configuration file (optional)
        
    Returns:
        dict: Parameters dictionary
    """
    # Start with default parameters
    params = DEFAULT_PARAMS.copy()
    
    # If config file provided, load and override defaults
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_params = json.load(f)
            
            # Update defaults with loaded parameters
            params.update(config_params)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            print("Using default parameters instead")
    
    return params

def save_parameters(params, output_path):
    """
    Saves parameters to a JSON file.
    
    Args:
        params: Parameters dictionary
        output_path: Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(params, indent=4, fp=f)
    print(f"Saved parameters to {output_path}")

def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="B. subtilis Competence Circuit Analysis")
    
    # Config file argument
    parser.add_argument('-c', '--config', type=str, help='Path to JSON configuration file')
    
    # Module selection arguments
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--standard', action='store_true', help='Run standard parameter analysis')
    parser.add_argument('--excitable', action='store_true', help='Run excitable configuration search')
    parser.add_argument('--hill', action='store_true', help='Run Hill coefficient analysis')
    parser.add_argument('--suel', action='store_true', help='Run Suel parameter space search')
    parser.add_argument('--stochastic', action='store_true', help='Run stochastic analysis')
    parser.add_argument('--amplification', action='store_true', help='Run amplification factor analysis')
    
    # Parameter override arguments
    parser.add_argument('--output-prefix', type=str, help='Prefix for output directory')
    parser.add_argument('--excitable-samples', type=int, help='Number of samples for excitable search')
    parser.add_argument('--hill-grid-size', type=int, help='Grid size for Hill coefficient search')
    parser.add_argument('--suel-points', type=int, help='Number of points for Suel search')
    parser.add_argument('--stochastic-sims', type=int, help='Number of stochastic simulations')
    parser.add_argument('--stochastic-time', type=float, help='Maximum time for stochastic simulations')
    parser.add_argument('--amp-sims', type=int, help='Number of amplification simulations')
    parser.add_argument('--amp-time', type=float, help='Maximum time for amplification simulations')
    
    return parser.parse_args()

def update_params_from_args(params, args):
    """
    Updates parameter dictionary from command-line arguments.
    
    Args:
        params: Parameters dictionary
        args: Parsed command-line arguments
        
    Returns:
        dict: Updated parameters dictionary
    """
    # Check if specific modules were requested
    if args.all or args.standard or args.excitable or args.hill or args.suel or args.stochastic or args.amplification:
        # Default all to False and then enable only requested ones
        params["run_standard_analysis"] = False
        params["run_excitable_search"] = False
        params["run_hill_analysis"] = False
        params["run_suel_search"] = False
        params["run_stochastic_analysis"] = False
        params["run_amplification_analysis"] = False
        
        # Enable requested modules
        if args.all:
            params["run_standard_analysis"] = True
            params["run_excitable_search"] = True
            params["run_hill_analysis"] = True
            params["run_suel_search"] = True
            params["run_stochastic_analysis"] = True
            params["run_amplification_analysis"] = True
        else:
            if args.standard:
                params["run_standard_analysis"] = True
            if args.excitable:
                params["run_excitable_search"] = True
            if args.hill:
                params["run_hill_analysis"] = True
            if args.suel:
                params["run_suel_search"] = True
            if args.stochastic:
                params["run_stochastic_analysis"] = True
            if args.amplification:
                params["run_amplification_analysis"] = True
    
    # Update other parameters if specified
    if args.output_prefix:
        params["output_prefix"] = args.output_prefix
    if args.excitable_samples:
        params["excitable_n_samples"] = args.excitable_samples
    if args.hill_grid_size:
        params["hill_grid_size"] = args.hill_grid_size
    if args.suel_points:
        params["suel_n_points"] = args.suel_points
    if args.stochastic_sims:
        params["stochastic_n_simulations"] = args.stochastic_sims
    if args.stochastic_time:
        params["stochastic_t_max"] = args.stochastic_time
    if args.amp_sims:
        params["amplification_n_simulations"] = args.amp_sims
    if args.amp_time:
        params["amplification_t_max"] = args.amp_time
    
    return params

def main():
    """
    Main function to run the comprehensive analysis of the competence circuit model.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load parameters from config file or defaults
    params = load_parameters(args.config)
    
    # Update parameters from command-line arguments
    params = update_params_from_args(params, args)
    
    # Create results directory
    results_dir = create_results_directory(params["output_prefix"])
    print(f"All results will be saved in: {results_dir}")
    
    # Save parameters used for this run
    save_parameters(params, os.path.join(results_dir, 'analysis_parameters.json'))
    
    # Variables to store analysis results
    std_info = None
    excitable_configs = None
    hill_results = None
    suel_configs = None
    representative_configs = None
    stochastic_results = None
    amp_results = None
    
    # Analyze standard parameters
    if params["run_standard_analysis"]:
        std_info = analyze_standard_parameters(results_dir, params)
    
    # Run excitable configuration analysis
    if params["run_excitable_search"]:
        excitable_configs = run_excitable_configuration_analysis(results_dir, params)
    
    # Run hill coefficient analysis
    if params["run_hill_analysis"]:
        hill_results = run_hill_coefficient_analysis(results_dir, params)
    
    # Run Suel parameter space search
    if params["run_suel_search"]:
        suel_configs, representative_configs = run_suel_parameter_search(results_dir, params)
    
    # Run stochastic analysis
    if params["run_stochastic_analysis"]:
        stochastic_results = run_stochastic_analysis(results_dir, params, excitable_configs)
    
    # Run amplification factor analysis
    if params["run_amplification_analysis"]:
        amp_results = run_amplification_analysis(results_dir, params)
    
    print("\nAnalysis completed. Results saved in images and statistics files.")
    print(f"Results directory: {results_dir}")
    
    return {
        'results_dir': results_dir,
        'std_info': std_info,
        'excitable_configs': excitable_configs,
        'hill_results': hill_results,
        'suel_configs': suel_configs,
        'representative_configs': representative_configs,
        'stochastic_results': stochastic_results,
        'amp_results': amp_results
    }

if __name__ == "__main__":
    main()