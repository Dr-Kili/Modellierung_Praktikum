import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def generate_ou_noise(theta, mu, sigma, dt):
    """
    Generates one step of Ornstein-Uhlenbeck noise.
    
    Args:
        theta: Mean reversion rate
        mu: Mean value
        sigma: Noise amplitude
        dt: Time step
        
    Returns:
        float: noise value
    """
    return np.random.normal(0, sigma * np.sqrt(dt))

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

def simulate_system_with_noise(model_odes, params, noise_params, initial_conditions=[0.01, 0.2], t_max=200, dt=0.01):
    """
    Simulates the system with Ornstein-Uhlenbeck noise.
    
    Args:
        model_odes: Function defining the deterministic ODEs
        params: Model parameters
        noise_params: Noise parameters (theta, mu, sigma, dt)
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
    
    # Initial conditions - set higher initial ComS to increase chance of competence
    K[0] = initial_conditions[0]
    S[0] = initial_conditions[1]
    
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
        
        # Add stronger noise to ComS to help trigger competence events
        S[i] = S[i-1] + dS * dt + noise * dt * 10  # Amplified noise effect
        
        # Ensure non-negative values
        K[i] = max(0, K[i])
        S[i] = max(0, S[i])
    
    return t, K, S

def identify_competence_events(t, K, threshold=0.5):
    """
    Identifies competence events and calculates both total duration and rise time to maximum ComK.
    
    Args:
        t: Time array
        K: ComK concentration array
        threshold: Threshold for competence
        
    Returns:
        list: List of (start_time, end_time) tuples for each competence event
        list: List of competence durations
        list: List of rise times (time from threshold crossing to maximum ComK)
        list: List of peak values for ComK during competence
    """
    competence_mask = K > threshold
    competence_events = []
    competence_durations = []
    rise_times = []
    peak_values = []
    
    if not competence_mask.any():
        return competence_events, competence_durations, rise_times, peak_values
    
    # Find transitions (0->1: start, 1->0: end)
    transitions = np.diff(competence_mask.astype(int))
    start_indices = np.where(transitions == 1)[0]
    end_indices = np.where(transitions == -1)[0]
    
    # Handle special cases
    if competence_mask[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if competence_mask[-1]:
        end_indices = np.append(end_indices, len(competence_mask) - 1)
    
    # Ensure we have matching start and end points
    n = min(len(start_indices), len(end_indices))
    
    for i in range(n):
        start_idx = start_indices[i]
        end_idx = end_indices[i]
        start_t = t[start_idx]
        end_t = t[end_idx]
        
        # Only count events that are long enough (avoid noise artifacts)
        if end_t - start_t > 1.0:  # Minimum duration threshold
            # Extract the ComK signal during this competence event
            event_K = K[start_idx:end_idx+1]
            event_t = t[start_idx:end_idx+1]
            
            # Find peak value and its position
            peak_value = np.max(event_K)
            peak_idx = np.argmax(event_K)
            peak_t = event_t[peak_idx]
            
            # Calculate rise time from threshold crossing to maximum
            rise_time = peak_t - start_t
            
            # Store results
            competence_events.append((start_t, end_t))
            competence_durations.append(end_t - start_t)
            rise_times.append(rise_time)
            peak_values.append(peak_value)
    
    return competence_events, competence_durations, rise_times, peak_values

# Diese Funktion ersetzt den entsprechenden Teil in stochastic_simulation.py

def analyze_stochastic_competence(model_odes, params_list, noise_params, results_dir, 
                                 n_simulations=20, t_max=1000, dt=0.01, threshold=0.5,
                                 param_names=None):
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
                initial_conditions=[0.01, 0.2], 
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
            
            # Save every other simulation as an example or any with competence events
            if sim % 2 == 0 or len(events) > 0:
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
                plt.title(f'Stochastic Simulation {sim+1} for {param_name}')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(t, S, 'g-', label='ComS')
                plt.xlabel('Time')
                plt.ylabel('ComS Concentration')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(param_dir, f'stochastic_sim_{sim+1}.png'))
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
            
            # Probability of initiation (average events per simulation)
            init_probability = np.mean(all_initiations)
            
            # Store results
            results[param_id] = {
                'name': param_name,
                'mean_duration': mean_duration,
                'median_duration': median_duration,
                'std_duration': std_duration,
                'cv_duration': cv_duration,
                'mean_rise_time': mean_rise_time,
                'median_rise_time': median_rise_time,
                'std_rise_time': std_rise_time,
                'cv_rise_time': cv_rise_time,
                'init_probability': init_probability,
                'all_durations': all_durations,
                'all_rise_times': all_rise_times,
                'all_peak_values': all_peak_values,
                'params': params
            }
            
            # Histogram of competence durations
            plt.figure(figsize=(10, 6))
            bins = max(10, min(20, len(all_durations)//2))
            plt.hist(all_durations, bins=bins, alpha=0.7, color='blue')
            plt.axvline(x=mean_duration, color='r', linestyle='--', 
                      label=f'Mean: {mean_duration:.2f}')
            plt.axvline(x=median_duration, color='g', linestyle='-', 
                      label=f'Median: {median_duration:.2f}')
            plt.xlabel('Competence Duration')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Competence Durations for {param_name} (CV: {cv_duration:.2f})')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(param_dir, 'duration_histogram.png'))
            plt.close()
            
            # Histogram of rise times
            plt.figure(figsize=(10, 6))
            bins = max(10, min(20, len(all_rise_times)//2))
            plt.hist(all_rise_times, bins=bins, alpha=0.7, color='orange')
            plt.axvline(x=mean_rise_time, color='r', linestyle='--', 
                      label=f'Mean: {mean_rise_time:.2f}')
            plt.axvline(x=median_rise_time, color='g', linestyle='-', 
                      label=f'Median: {median_rise_time:.2f}')
            plt.xlabel('Rise Time (Time to Saturation)')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of Rise Times for {param_name} (CV: {cv_rise_time:.2f})')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(param_dir, 'rise_time_histogram.png'))
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
            plt.savefig(os.path.join(param_dir, 'rise_vs_duration.png'))
            plt.close()
            
            # Save statistics to text file
            with open(os.path.join(param_dir, 'statistics.txt'), 'w') as f:
                f.write(f"{param_name}\n")
                f.write(f"{'='*len(param_name)}\n\n")
                f.write(f"Number of simulations: {n_simulations}\n")
                f.write(f"Total competence events: {len(all_durations)}\n")
                f.write(f"Mean duration: {mean_duration:.2f}\n")
                f.write(f"Median duration: {median_duration:.2f}\n")
                f.write(f"Standard deviation of duration: {std_duration:.2f}\n")
                f.write(f"Coefficient of variation (duration): {cv_duration:.2f}\n")
                f.write(f"Mean rise time: {mean_rise_time:.2f}\n")
                f.write(f"Median rise time: {median_rise_time:.2f}\n")
                f.write(f"Standard deviation of rise time: {std_rise_time:.2f}\n")
                f.write(f"Coefficient of variation (rise time): {cv_rise_time:.2f}\n")
                f.write(f"Probability of initiation: {init_probability:.4f}\n")
                
                f.write("\nParameters:\n")
                f.write("  Core Parameters:\n")
                for param_name in ['ak', 'bk', 'bs', 'k0', 'k1', 'n', 'p']:
                    if param_name in params:
                        f.write(f"    {param_name}: {params[param_name]}\n")
                
                # Add extended parameters if present
                if 'GammaK' in params:
                    f.write("\n  Extended Parameters (Suel et al.):\n")
                    for param_name in ['GammaK', 'GammaS', 'lambdaK', 'lambdaS', 'deltaK', 'deltaS']:
                        if param_name in params:
                            f.write(f"    {param_name}: {params[param_name]}\n")
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
    
# Dieser Teil ersetzt die Erstellung der time_comparison-Grafik in der analyze_stochastic_competence-Funktion

    # Create summary plot for all parameter sets
    if results:
        param_ids = list(results.keys())
        param_labels = [results[pid]['name'] for pid in param_ids]
        median_durations = [results[pid]['median_duration'] for pid in param_ids]
        cv_durations = [results[pid]['cv_duration'] for pid in param_ids]
        median_rise_times = [results[pid]['median_rise_time'] for pid in param_ids]
        cv_rise_times = [results[pid]['cv_rise_time'] for pid in param_ids]
        
        # Update Duration and Rise Time comparison (side by side)
        plt.figure(figsize=(14, 6))
        
        # Median Values
        plt.subplot(1, 2, 1)
        bar_width = 0.3
        index = np.arange(len(param_ids))
        
        # Plot median values with stronger colors
        bars1 = plt.bar(index - bar_width/2, median_durations, bar_width, color='blue', alpha=0.9, label='Median Duration')
        bars2 = plt.bar(index + bar_width/2, median_rise_times, bar_width, color='orange', alpha=0.9, label='Median Rise Time')
        
        plt.xlabel('Parameter Set')
        plt.ylabel('Time')
        plt.title('Comparison of Median Times')
        plt.xticks(index, param_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Place legend outside the first plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        # Coefficient of Variation
        plt.subplot(1, 2, 2)
        bars3 = plt.bar(index - bar_width/2, cv_durations, bar_width, color='blue', alpha=0.7, label='Total Duration')
        bars4 = plt.bar(index + bar_width/2, cv_rise_times, bar_width, color='orange', alpha=0.7, label='Rise Time')
        plt.xlabel('Parameter Set')
        plt.ylabel('Coefficient of Variation')
        plt.title('Timing Variability')
        plt.xticks(index, param_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Place legend outside the second plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(os.path.join(stochastic_dir, 'timing_comparison.png'), bbox_inches='tight')
        plt.close()
        
        # Save summary to text file
        with open(os.path.join(stochastic_dir, 'stochastic_summary.txt'), 'w') as f:
            f.write("Stochastische Simulationsergebnisse\n")
            f.write("==================================\n\n")
            
            f.write("Parameter Set\tMedian Dauer\tDauer CV\tMedian Anstieg\tAnstieg CV\tInitiierung\n")
            for pid in param_ids:
                name = results[pid]['name']
                median_duration = results[pid]['median_duration']
                dur_cv = results[pid]['cv_duration']
                median_rise = results[pid]['median_rise_time']
                rise_cv = results[pid]['cv_rise_time']
                init = results[pid]['init_probability']
                
                f.write(f"{name}\t{median_duration:.2f}\t{dur_cv:.2f}\t{median_rise:.2f}\t{rise_cv:.2f}\t{init:.4f}\n")
        
        # Create a CSV with summary data
        summary_data = []
        for pid in param_ids:
            summary_data.append({
                'Parameter Set': results[pid]['name'],
                'Median Duration': results[pid]['median_duration'],
                'Duration CV': results[pid]['cv_duration'],
                'Median Rise Time': results[pid]['median_rise_time'],
                'Rise Time CV': results[pid]['cv_rise_time'],
                'Initiation Probability': results[pid]['init_probability'],
                'Total Events': len(results[pid]['all_durations'])
            })
        
        # Create DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(stochastic_dir, 'stochastic_summary.csv'), index=False)
        print(f"Stochastic summary saved to CSV: {os.path.join(stochastic_dir, 'stochastic_summary.csv')}")

    return results

# Example usage (for documentation):
if __name__ == "__main__":
    print("This module provides functions for stochastic simulation of the competence circuit.")
    print("It can analyze both total competence duration and rise time to ComK saturation.")
    print("Import and use these functions in your main analysis script.")