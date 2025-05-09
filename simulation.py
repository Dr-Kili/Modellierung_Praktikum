"""
Simulation functions for competence circuit analysis.

This module contains functions for simulating the competence circuit model,
including both deterministic and stochastic simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helpers import identify_competence_events, generate_ou_noise
from visualization import plot_time_series, plot_stochastic_comparison

def simulate_system(model_odes, params, initial_conditions=[0.01, 0.2], t_max=200, dt=0.01):
    """
    Simulates the system deterministically over time.
    
    Args:
        model_odes: Function defining the ODEs
        params: Model parameters
        initial_conditions: Initial state [K, S]
        t_max: Maximum simulation time
        dt: Time step
        
    Returns:
        tuple: (time_array, ComK_array, ComS_array)
    """
    steps = int(t_max/dt)
    t = np.linspace(0, t_max, steps)
    
    # Arrays to store concentrations
    K = np.zeros(steps)
    S = np.zeros(steps)
    
    # Set initial conditions
    K[0] = initial_conditions[0]
    S[0] = initial_conditions[1]
    
    # Integrate ODEs using Euler method
    for i in range(1, steps):
        dK, dS = model_odes(t[i-1], [K[i-1], S[i-1]], params)
        K[i] = K[i-1] + dK * dt
        S[i] = S[i-1] + dS * dt
        
        # Ensure non-negative values
        K[i] = max(0, K[i])
        S[i] = max(0, S[i])
    
    return t, K, S

def simulate_system_with_noise(model_odes, params, noise_params, 
                             amplification_factor=10, initial_comS_boost=False,
                             initial_conditions=[0.01, 0.2], t_max=200, dt=0.01):
    """
    Simulates the system with Ornstein-Uhlenbeck noise with a specific amplification factor.
    
    Args:
        model_odes: Function defining the deterministic ODEs
        params: Model parameters
        noise_params: Noise parameters (theta, mu, sigma, dt)
        amplification_factor: Factor to amplify noise by
        initial_comS_boost: Whether to apply an initial 10x boost to ComS
        initial_conditions: Initial state [K, S]
        t_max: Maximum simulation time
        dt: Time step
        
    Returns:
        tuple: (time array, ComK array, ComS array)
    """
    steps = int(t_max/dt)
    t = np.linspace(0, t_max, steps)
    K = np.zeros(steps)
    S = np.zeros(steps)
    
    # Initial conditions - optionally applying the ComS boost
    K[0] = initial_conditions[0]
    S[0] = initial_conditions[1] * (10 if initial_comS_boost else 1)
    
    # OU process for noise in ComS
    noise = 0.0  # Initial noise value
    
    for i in range(1, steps):
        # Calculate regular dynamics
        dK, dS = model_odes(t[i-1], [K[i-1], S[i-1]], params)
        
        # Update noise
        noise = noise + noise_params['theta'] * (noise_params['mu'] - noise) * dt + \
                noise_params['sigma'] * np.sqrt(dt) * np.random.normal()
        
        # Update system state
        K[i] = K[i-1] + dK * dt
        
        # Add noise to ComS with the specified amplification factor
        S[i] = S[i-1] + dS * dt + noise * dt * amplification_factor
        
        # Ensure non-negative values
        K[i] = max(0, K[i])
        S[i] = max(0, S[i])
    
    return t, K, S

def analyze_stochastic_competence(model_odes, params_list, noise_params, results_dir, 
                                 n_simulations=20, t_max=1000, dt=0.01, threshold=0.5,
                                 param_names=None, amplification_factor=10,
                                 initial_conditions=[0.01, 0.2], initial_coms_boost=False):
    """
    Analyzes competence dynamics with stochastic noise for multiple parameter sets.
    Calculates both competence duration and rise time statistics.
    
    Args:
        model_odes: Function defining the deterministic ODEs
        params_list: List of parameter sets to analyze
        noise_params: Noise parameters
        results_dir: Directory to save results
        n_simulations: Number of simulations per parameter set
        t_max: Maximum simulation time
        dt: Time step
        threshold: Competence threshold
        param_names: Optional list of names for the parameter sets
        amplification_factor: Factor to amplify noise
        initial_conditions: Initial state [K, S]
        initial_coms_boost: Whether to apply a 10x boost to initial ComS
        
    Returns:
        dict: Statistics of competence events for each parameter set
    """
    # Directory for stochastic results
    stochastic_dir = os.path.join(results_dir, 'stochastic_simulations')
    os.makedirs(stochastic_dir, exist_ok=True)
    
    # Results dictionary
    results = {}
    
    # For each parameter set
    for idx, params in enumerate(params_list):
        param_id = f"param_set_{idx+1}"
        
        # Use provided name if available
        if param_names and idx < len(param_names):
            param_name = param_names[idx]
        else:
            param_name = f"Parameter Set {idx+1}"
            
        print(f"Analyzing {param_name} ({idx+1}/{len(params_list)})")
        print_parameter_details(params, param_name, idx+1)
        
        if initial_coms_boost:
            print(f"  Applying 10x boost to initial ComS concentration")
        
        # Create subdirectory for this parameter set
        param_dir = os.path.join(stochastic_dir, param_id)
        os.makedirs(param_dir, exist_ok=True)
        
        # Collect statistics
        all_durations = []
        all_rise_times = []
        all_events = []
        all_initiations = []  # Count of initiations per simulation
        all_peak_values = []
        
        # Multiple simulations for this parameter set
        for sim in range(n_simulations):
            # Run simulation with noise
            t, K, S = simulate_system_with_noise(
                model_odes, params, noise_params,
                amplification_factor=amplification_factor,
                initial_comS_boost=initial_coms_boost,
                initial_conditions=initial_conditions, 
                t_max=t_max, dt=dt
            )
            
            # Identify competence events and calculate rise times
            events, durations, rise_times, peaks = identify_competence_events(t, K, threshold)
            
            # Track statistics
            all_events.extend(events)
            all_durations.extend(durations)
            all_rise_times.extend(rise_times)
            all_initiations.append(len(events))
            all_peak_values.extend(peaks)
            
            # Save every few simulations or any with competence events
            if sim % 5 == 0 or len(events) > 0:
                save_stochastic_simulation_plot(t, K, S, events, rise_times, threshold,
                                            param_name, sim, param_dir)
        
        # Calculate statistics if we observed any competence events
        if all_durations:
            # Generate and save statistics
            stats = calculate_competence_statistics(
                all_durations, all_rise_times, all_initiations, n_simulations)
            
            # Add parameters to statistics
            stats['name'] = param_name
            stats['all_durations'] = all_durations
            stats['all_rise_times'] = all_rise_times
            stats['all_peak_values'] = all_peak_values
            stats['params'] = params
            
            # Store results
            results[param_id] = stats
            
            # Create and save visualizations
            save_competence_distribution_plots(all_durations, all_rise_times, 
                                            param_name, stats, param_dir)
        else:
            print(f"No competence events detected for {param_name}")
            results[param_id] = {
                'name': param_name,
                'mean_duration': 0,
                'median_duration': 0,
                'std_duration': 0,
                'cv_duration': 0,
                'mean_rise_time': 0,
                'median_rise_time': 0,
                'std_rise_time': 0,
                'cv_rise_time': 0,
                'init_probability': 0,
                'all_durations': [],
                'all_rise_times': [],
                'all_peak_values': [],
                'params': params
            }
    
    # Create summary visualizations across all parameter sets
    if results:
        plot_stochastic_comparison(results, stochastic_dir)
        save_summary_statistics(results, stochastic_dir)
        
    return results

def print_parameter_details(params, param_name=None, index=None):
    """
    Prints detailed information about a parameter set.
    
    Args:
        params: Parameter dictionary
        param_name: Optional name for the parameter set
        index: Optional index of the parameter set
    """
    header = f"Parameter Set: {param_name if param_name else f'#{index if index is not None else 1}'}"
    print("\n" + "="*80)
    print(f"{header:^80}")
    print("="*80)
    
    # Standard parameters section
    print("\nCore Parameters:")
    print(f"  ak (ComK basal rate): {params.get('ak', 'N/A')}")
    print(f"  bk (ComK feedback strength): {params.get('bk', 'N/A')}")
    print(f"  bs (ComS expression rate): {params.get('bs', 'N/A')}")
    print(f"  k0 (ComK half-activation): {params.get('k0', 'N/A')}")
    print(f"  k1 (ComS repression threshold): {params.get('k1', 'N/A')}")
    print(f"  n (ComK Hill coefficient): {params.get('n', 'N/A')}")
    print(f"  p (ComS Hill coefficient): {params.get('p', 'N/A')}")
    
    # Check if the parameter set includes Suel-specific parameters
    if 'GammaK' in params:
        print("\nExtended Parameters (Suel et al.):")
        print(f"  GammaK: {params.get('GammaK', 'N/A')}")
        print(f"  GammaS: {params.get('GammaS', 'N/A')}")
        print(f"  lambdaK: {params.get('lambdaK', 'N/A')}")
        print(f"  lambdaS: {params.get('lambdaS', 'N/A')}")
        print(f"  deltaK: {params.get('deltaK', 'N/A')}")
        print(f"  deltaS: {params.get('deltaS', 'N/A')}")
    
    print("-"*80)

def calculate_competence_statistics(all_durations, all_rise_times, all_initiations, n_simulations):
    """
    Calculates statistics for competence events.
    
    Args:
        all_durations: List of competence durations
        all_rise_times: List of rise times
        all_initiations: List of initiations per simulation
        n_simulations: Number of simulations
        
    Returns:
        dict: Statistics dictionary
    """
    # Duration statistics
    mean_duration = np.mean(all_durations)
    median_duration = np.median(all_durations)
    std_duration = np.std(all_durations)
    cv_duration = std_duration / mean_duration if mean_duration > 0 else 0
    
    # Rise time statistics
    mean_rise_time = np.mean(all_rise_times)
    median_rise_time = np.median(all_rise_times)
    std_rise_time = np.std(all_rise_times)
    cv_rise_time = std_rise_time / mean_rise_time if mean_rise_time > 0 else 0
    
    # Initiation probability (average events per simulation)
    init_probability = np.mean(all_initiations)
    
    return {
        'mean_duration': mean_duration,
        'median_duration': median_duration,
        'std_duration': std_duration,
        'cv_duration': cv_duration,
        'mean_rise_time': mean_rise_time,
        'median_rise_time': median_rise_time,
        'std_rise_time': std_rise_time,
        'cv_rise_time': cv_rise_time,
        'init_probability': init_probability
    }

def save_stochastic_simulation_plot(t, K, S, events, rise_times, threshold, 
                                 param_name, sim_number, output_dir):
    """
    Saves a plot of a stochastic simulation.
    
    Args:
        t: Time array
        K: ComK concentration array
        S: ComS concentration array
        events: List of competence events (start_time, end_time)
        rise_times: List of rise times
        threshold: Competence threshold
        param_name: Name of the parameter set
        sim_number: Simulation number
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, K, 'b-', label='ComK')
    # Mark competence events
    for j, (start, end) in enumerate(events):
        plt.axvspan(start, end, alpha=0.2, color='red', label='Competence' if j==0 else "")
        
        # Mark rise time
        if j < len(rise_times):
            # Calculate the rise time
            max_t = start + rise_times[j]
            plt.axvspan(start, max_t, alpha=0.4, color='orange', 
                      label='Rise Time' if j==0 else "")
    
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Competence Threshold')
    plt.ylabel('ComK Concentration')
    plt.title(f'Stochastic Simulation {sim_number+1} for {param_name}')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, S, 'g-', label='ComS')
    plt.xlabel('Time')
    plt.ylabel('ComS Concentration')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'stochastic_sim_{sim_number+1}.png'))
    plt.close()

def save_competence_distribution_plots(all_durations, all_rise_times, param_name, stats, output_dir):
    """
    Saves plots of competence duration and rise time distributions.
    
    Args:
        all_durations: List of competence durations
        all_rise_times: List of competence rise times
        param_name: Name of the parameter set
        stats: Dictionary of statistics
        output_dir: Directory to save the plots
    """
    # Histogram of competence durations
    plt.figure(figsize=(10, 6))
    bins = max(10, min(20, len(all_durations)//2))
    plt.hist(all_durations, bins=bins, alpha=0.7, color='blue')
    plt.axvline(x=stats['mean_duration'], color='r', linestyle='--', 
              label=f'Mean: {stats["mean_duration"]:.2f}')
    plt.axvline(x=stats['median_duration'], color='g', linestyle='-', 
              label=f'Median: {stats["median_duration"]:.2f}')
    plt.xlabel('Competence Duration')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Competence Durations for {param_name} (CV: {stats["cv_duration"]:.2f})')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'duration_histogram.png'))
    plt.close()
    
    # Histogram of rise times
    plt.figure(figsize=(10, 6))
    bins = max(10, min(20, len(all_rise_times)//2))
    plt.hist(all_rise_times, bins=bins, alpha=0.7, color='orange')
    plt.axvline(x=stats['mean_rise_time'], color='r', linestyle='--', 
              label=f'Mean: {stats["mean_rise_time"]:.2f}')
    plt.axvline(x=stats['median_rise_time'], color='g', linestyle='-', 
              label=f'Median: {stats["median_rise_time"]:.2f}')
    plt.xlabel('Rise Time (Time to Saturation)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Rise Times for {param_name} (CV: {stats["cv_rise_time"]:.2f})')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rise_time_histogram.png'))
    plt.close()
    
    # Scatterplot of rise time vs duration
    plt.figure(figsize=(10, 6))
    plt.scatter(all_rise_times, all_durations, alpha=0.7)
    plt.xlabel('Rise Time (Time to Saturation)')
    plt.ylabel('Competence Duration')
    plt.title(f'Relationship Between Rise Time and Competence Duration for {param_name}')
    plt.grid(True)
    # Add trend line
    if len(all_rise_times) > 1:
        z = np.polyfit(all_rise_times, all_durations, 1)
        p = np.poly1d(z)
        plt.plot(sorted(all_rise_times), p(sorted(all_rise_times)), "r--", 
               label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
        plt.legend()
    plt.savefig(os.path.join(output_dir, 'rise_vs_duration.png'))
    plt.close()
    
    # Save statistics to text file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write(f"{param_name}\n")
        f.write(f"{'='*len(param_name)}\n\n")
        f.write(f"Number of competence events: {len(all_durations)}\n")
        f.write(f"Mean duration: {stats['mean_duration']:.2f}\n")
        f.write(f"Median duration: {stats['median_duration']:.2f}\n")
        f.write(f"Standard deviation of duration: {stats['std_duration']:.2f}\n")
        f.write(f"Coefficient of variation (duration): {stats['cv_duration']:.2f}\n")
        f.write(f"Mean rise time: {stats['mean_rise_time']:.2f}\n")
        f.write(f"Median rise time: {stats['median_rise_time']:.2f}\n")
        f.write(f"Standard deviation of rise time: {stats['std_rise_time']:.2f}\n")
        f.write(f"Coefficient of variation (rise time): {stats['cv_rise_time']:.2f}\n")
        f.write(f"Probability of initiation: {stats['init_probability']:.4f}\n")
        
def save_summary_statistics(results, output_dir):
    """
    Saves summary statistics for all parameter sets.
    
    Args:
        results: Dictionary of results from stochastic simulations
        output_dir: Directory to save the summary
    """
    # Save summary to text file
    with open(os.path.join(output_dir, 'stochastic_summary.txt'), 'w') as f:
        f.write("Stochastic Simulation Results\n")
        f.write("==========================\n\n")
        
        f.write("Parameter Set\tMedian Duration\tDuration CV\tMedian Rise\tRise CV\tInitiation\n")
        for param_id in results:
            name = results[param_id]['name']
            median_duration = results[param_id]['median_duration']
            dur_cv = results[param_id]['cv_duration']
            median_rise = results[param_id]['median_rise_time']
            rise_cv = results[param_id]['cv_rise_time']
            init = results[param_id]['init_probability']
            
            f.write(f"{name}\t{median_duration:.2f}\t{dur_cv:.2f}\t{median_rise:.2f}\t{rise_cv:.2f}\t{init:.4f}\n")
    
    # Create a CSV with summary data
    summary_data = []
    for param_id in results:
        summary_data.append({
            'Parameter Set': results[param_id]['name'],
            'Median Duration': results[param_id]['median_duration'],
            'Duration CV': results[param_id]['cv_duration'],
            'Median Rise Time': results[param_id]['median_rise_time'],
            'Rise Time CV': results[param_id]['cv_rise_time'],
            'Initiation Probability': results[param_id]['init_probability'],
            'Total Events': len(results[param_id]['all_durations'])
        })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'stochastic_summary.csv'), index=False)
    print(f"Stochastic summary saved to CSV: {os.path.join(output_dir, 'stochastic_summary.csv')}")

def analyze_amplification_factors(model_odes, params, results_dir, 
                               amplification_factors=[1, 3, 5, 7, 10], 
                               n_simulations=50, t_max=500,
                               initial_conditions=[0.01, 0.2],
                               initial_coms_boost=False):
    """
    Analyzes the effect of different noise amplification factors on competence dynamics.
    
    Args:
        model_odes: Function defining the deterministic ODEs
        params: Model parameters
        results_dir: Directory to save results
        amplification_factors: List of amplification factors to test
        n_simulations: Number of simulations per amplification factor
        t_max: Maximum simulation time
        initial_conditions: Initial state [K, S]
        initial_coms_boost: Whether to apply a 10x boost to initial ComS
        
    Returns:
        dict: Results for each amplification factor
    """
    print("\n=== ANALYZING EFFECT OF NOISE AMPLIFICATION FACTORS ===")
    
    # Noise parameters
    noise_params = {
        'theta': 1.0,      # Mean reversion rate
        'mu': 0.0,         # Mean value (noise is centered around zero)
        'sigma': 0.3,      # Noise amplitude
        'dt': 0.1          # Time step for numerical integration
    }
    
    # Time step for simulation
    dt = 0.01
    
    # Threshold for competence
    threshold = 0.5
    
    # Results dictionary
    results = {}
    
    # Create directory for amplification analysis
    amp_dir = os.path.join(results_dir, 'amplification_analysis')
    os.makedirs(amp_dir, exist_ok=True)
    
    # For each amplification factor
    for amp_factor in amplification_factors:
        print(f"\nAnalyzing amplification factor: {amp_factor}")
        
        # Create directory for this amplification factor
        factor_dir = os.path.join(amp_dir, f'factor_{amp_factor}')
        os.makedirs(factor_dir, exist_ok=True)
        
        # Collect statistics
        all_durations = []
        all_rise_times = []
        all_events = []
        all_initiations = []  # Count of initiations per simulation
        all_peak_values = []
        
        # Multiple simulations for this amplification factor
        for sim in range(n_simulations):
            # Progress indicator
            if (sim + 1) % 10 == 0:
                print(f"  Simulation {sim + 1}/{n_simulations}")
                
            # Run simulation with noise - with optional initial ComS boost
            t, K, S = simulate_system_with_noise(
                model_odes, params, noise_params,
                amplification_factor=amp_factor,
                initial_comS_boost=initial_coms_boost, 
                initial_conditions=initial_conditions, 
                t_max=t_max, dt=dt
            )
            
            # Identify competence events and calculate rise times
            events, durations, rise_times, peaks = identify_competence_events(t, K, threshold)
            
            # Track statistics
            all_events.extend(events)
            all_durations.extend(durations)
            all_rise_times.extend(rise_times)
            all_initiations.append(len(events))
            all_peak_values.extend(peaks)
            
            # Save every 10th simulation as an example
            if sim % 10 == 0:
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(t, K, 'b-', label='ComK')
                # Mark competence events
                for j, (start, end) in enumerate(events):
                    plt.axvspan(start, end, alpha=0.2, color='red', label='Competence' if j==0 else "")
                    
                plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Competence Threshold')
                plt.ylabel('ComK Concentration')
                plt.title(f'Simulation with Amplification Factor {amp_factor} (Sim {sim+1})')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(t, S, 'g-', label='ComS')
                plt.xlabel('Time')
                plt.ylabel('ComS Concentration')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(factor_dir, f'simulation_{sim+1}.png'))
                plt.close()
        
        # Calculate statistics if we observed any competence events
        if all_durations:
            mean_duration = np.mean(all_durations)
            median_duration = np.median(all_durations)
            std_duration = np.std(all_durations)
            cv_duration = std_duration / mean_duration if mean_duration > 0 else 0
            
            mean_rise_time = np.mean(all_rise_times)
            median_rise_time = np.median(all_rise_times)
            std_rise_time = np.std(all_rise_times)
            cv_rise_time = std_rise_time / mean_rise_time if mean_rise_time > 0 else 0
            
            # Average number of competence events per simulation
            avg_events_per_sim = np.mean(all_initiations)
            
            # Store results
            results[amp_factor] = {
                'mean_duration': mean_duration,
                'median_duration': median_duration,
                'std_duration': std_duration,
                'cv_duration': cv_duration,
                'mean_rise_time': mean_rise_time,
                'median_rise_time': median_rise_time,
                'std_rise_time': std_rise_time,
                'cv_rise_time': cv_rise_time,
                'avg_events_per_sim': avg_events_per_sim,
                'all_durations': all_durations,
                'all_rise_times': all_rise_times,
                'all_peak_values': all_peak_values,
                'total_events': len(all_durations)
            }
            
            # Histogram of competence durations
            plt.figure(figsize=(10, 6))
            bins = max(10, min(30, len(all_durations)//5))
            plt.hist(all_durations, bins=bins, alpha=0.7, color='blue')
            plt.axvline(x=mean_duration, color='r', linestyle='--', 
                      label=f'Mean: {mean_duration:.2f}')
            plt.axvline(x=median_duration, color='g', linestyle='-', 
                      label=f'Median: {median_duration:.2f}')
            plt.xlabel('Competence Duration')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Competence Durations (Amplification Factor: {amp_factor})\n'
                    f'CV: {cv_duration:.2f}, Events: {len(all_durations)}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(factor_dir, 'duration_histogram.png'))
            plt.close()
            
            # Save statistics to text file
            with open(os.path.join(factor_dir, 'statistics.txt'), 'w') as f:
                f.write(f"Amplification Factor: {amp_factor}\n")
                f.write("============================\n\n")
                f.write(f"Number of simulations: {n_simulations}\n")
                f.write(f"Total competence events: {len(all_durations)}\n")
                
                f.write("Duration Statistics:\n")
                f.write(f"  Mean: {mean_duration:.2f}\n")
                f.write(f"  Median: {median_duration:.2f}\n")
                f.write(f"  Standard deviation: {std_duration:.2f}\n")
                f.write(f"  Coefficient of variation: {cv_duration:.2f}\n\n")
                
                f.write("Rise Time Statistics:\n")
                f.write(f"  Mean: {mean_rise_time:.2f}\n")
                f.write(f"  Median: {median_rise_time:.2f}\n")
                f.write(f"  Standard deviation: {std_rise_time:.2f}\n")
                f.write(f"  Coefficient of variation: {cv_rise_time:.2f}\n")
        else:
            print(f"  No competence events detected for amplification factor {amp_factor}")
            results[amp_factor] = {
                'mean_duration': 0,
                'median_duration': 0,
                'std_duration': 0,
                'cv_duration': 0,
                'mean_rise_time': 0,
                'median_rise_time': 0,
                'std_rise_time': 0,
                'cv_rise_time': 0,
                'avg_events_per_sim': 0,
                'all_durations': [],
                'all_rise_times': [],
                'all_peak_values': [],
                'total_events': 0
            }
    
    # Create comparative visualizations
    create_amplification_comparison_plots(results, amp_dir)
    
    return results

def create_amplification_comparison_plots(results, output_dir):
    """
    Creates comparative visualizations of results across different amplification factors.
    
    Args:
        results: Dictionary of results for each amplification factor
        output_dir: Directory to save visualizations
    """
    # Extract data for plotting
    amp_factors = sorted(results.keys())
    mean_durations = [results[factor]['mean_duration'] for factor in amp_factors]
    median_durations = [results[factor]['median_duration'] for factor in amp_factors]
    cv_durations = [results[factor]['cv_duration'] for factor in amp_factors]
    mean_rise_times = [results[factor]['mean_rise_time'] for factor in amp_factors]
    cv_rise_times = [results[factor]['cv_rise_time'] for factor in amp_factors]
    events_per_sim = [results[factor]['avg_events_per_sim'] for factor in amp_factors]
    total_events = [results[factor]['total_events'] for factor in amp_factors]
    
    # 1. Bar plot of total events and events per simulation
    plt.figure(figsize=(12, 6))
    
    # Create the primary y-axis for total events
    ax1 = plt.gca()
    bars1 = ax1.bar(np.array(amp_factors) - 0.15, total_events, width=0.3, 
                  color='blue', alpha=0.7, label='Total Events')
    ax1.set_xlabel('Noise Amplification Factor')
    ax1.set_ylabel('Total Number of Competence Events', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create a secondary y-axis for events per simulation
    ax2 = ax1.twinx()
    bars2 = ax2.bar(np.array(amp_factors) + 0.15, events_per_sim, width=0.3, 
                  color='red', alpha=0.7, label='Avg Events per Simulation')
    ax2.set_ylabel('Average Events per Simulation', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add labels above bars
    for bar, value in zip(bars1, total_events):
        if value > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value)}', ha='center', va='bottom', color='blue')
            
    for bar, value in zip(bars2, events_per_sim):
        if value > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}', ha='center', va='bottom', color='red')
    
    # Add a title and adjust layout
    plt.title('Effect of Noise Amplification on Competence Event Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'event_frequency_comparison.png'))
    plt.close()
    
    # 2. Comparison of mean and median durations
    plt.figure(figsize=(12, 6))
    
    # Plot mean durations
    plt.bar(np.array(amp_factors) - 0.15, mean_durations, width=0.3, 
          color='blue', alpha=0.7, label='Mean Duration')
    
    # Plot median durations
    plt.bar(np.array(amp_factors) + 0.15, median_durations, width=0.3, 
          color='green', alpha=0.7, label='Median Duration')
    
    plt.xlabel('Noise Amplification Factor')
    plt.ylabel('Duration')
    plt.title('Effect of Noise Amplification on Competence Duration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_comparison.png'))
    plt.close()
    
    # 3. Comparison of coefficient of variation (noise)
    plt.figure(figsize=(12, 6))
    
    # Plot CV of durations
    plt.bar(np.array(amp_factors) - 0.15, cv_durations, width=0.3, 
          color='blue', alpha=0.7, label='Duration CV')
    
    # Plot CV of rise times
    plt.bar(np.array(amp_factors) + 0.15, cv_rise_times, width=0.3, 
          color='orange', alpha=0.7, label='Rise Time CV')
    
    plt.xlabel('Noise Amplification Factor')
    plt.ylabel('Coefficient of Variation')
    plt.title('Effect of Noise Amplification on Variability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'variability_comparison.png'))
    plt.close()
    
    # 4. Overlapping histograms of durations for different amplification factors
    # Only include factors with events
    factors_with_events = [factor for factor in amp_factors if results[factor]['total_events'] > 0]
    
    if len(factors_with_events) > 1:  # Only create if we have at least two factors with events
        plt.figure(figsize=(14, 8))
        
        # Define color map
        colors = plt.cm.viridis(np.linspace(0, 1, len(factors_with_events)))
        
        # Plot histograms
        for i, factor in enumerate(factors_with_events):
            durations = results[factor]['all_durations']
            if durations:
                # Normalize to make areas comparable despite different event counts
                weights = np.ones_like(durations) / len(durations)
                plt.hist(durations, bins=30, alpha=0.5, 
                       weights=weights, color=colors[i], 
                       label=f'Factor {factor} (n={len(durations)})')
        
        plt.xlabel('Competence Duration')
        plt.ylabel('Normalized Frequency')
        plt.title('Normalized Distribution of Competence Durations by Amplification Factor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'duration_distributions.png'))
        plt.close()
    
    # 5. Save summary statistics to file
    with open(os.path.join(output_dir, 'amplification_summary.txt'), 'w') as f:
        f.write("Summary of Noise Amplification Factor Analysis\n")
        f.write("============================================\n\n")
        
        # Create a table header
        f.write("Amp Factor | Total Events | Events/Sim | Mean Duration | Median Duration | Duration CV | Rise Time CV\n")
        f.write("-" * 100 + "\n")
        
        # Write data for each amplification factor
        for i, factor in enumerate(amp_factors):
            f.write(f"{factor:9d} | {total_events[i]:12d} | {events_per_sim[i]:9.2f} | ")
            f.write(f"{mean_durations[i]:13.2f} | {median_durations[i]:15.2f} | ")
            f.write(f"{cv_durations[i]:11.2f} | {cv_rise_times[i]:11.2f}\n")
    
    # Create dataframe for CSV export
    summary_df = pd.DataFrame({
        'Amplification Factor': amp_factors,
        'Total Events': total_events,
        'Events/Sim': events_per_sim,
        'Mean Duration': mean_durations,
        'Median Duration': median_durations,
        'Duration CV': cv_durations,
        'Mean Rise Time': mean_rise_times,
        'Median Rise Time': [results[factor]['median_rise_time'] for factor in amp_factors],
        'Rise Time CV': cv_rise_times
    })
    
    # Save to CSV
    summary_df.to_csv(os.path.join(output_dir, 'amplification_summary.csv'), index=False)