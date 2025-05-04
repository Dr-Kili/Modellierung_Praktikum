import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import competence_circuit_analysis as comp_model
from stochastic_simulation import identify_competence_events

def create_results_directory():
    """
    Creates a results directory with timestamp.
    
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"noise_amplification_analysis_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name

def simulate_system_with_noise(model_odes, params, noise_params, 
                             amplification_factor, initial_comS_boost=False,
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

def analyze_amplification_factors(results_dir, n_simulations=50, t_max=500):
    """
    Analyzes the effect of different noise amplification factors on competence dynamics.
    
    Args:
        results_dir: Directory to save results
        n_simulations: Number of simulations per amplification factor
        t_max: Maximum simulation time
    """
    print("\n=== ANALYZING EFFECT OF NOISE AMPLIFICATION FACTORS ===")
    
    # Standard parameters
    params = comp_model.default_params()
    
    # Noise parameters
    noise_params = {
        'theta': 1.0,      # Mean reversion rate
        'mu': 0.0,         # Mean value (noise is centered around zero)
        'sigma': 0.3,      # Noise amplitude
        'dt': 0.1          # Time step for numerical integration
    }
    
    # Amplification factors to test
    amplification_factors = [1, 3, 5, 7, 10]
    
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
                
            # Run simulation with noise - no initial ComS boost
            t, K, S = simulate_system_with_noise(
                comp_model.model_odes, params, noise_params,
                amplification_factor=amp_factor,
                initial_comS_boost=False, 
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
            #avg_events_per_sim = np.mean(all_initiations)
            
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
    create_comparative_visualizations(results, amp_dir)
    
    # Save results
    with open(os.path.join(results_dir, 'amplification_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
        
    return results

def create_comparative_visualizations(results, output_dir):
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

def main():
    """
    Main function to run the analysis of amplification factors.
    """
    # Create results directory
    results_dir = create_results_directory()
    print(f"Results will be saved in: {results_dir}")
    
    # Analyze effect of different amplification factors
    results = analyze_amplification_factors(
        results_dir, 
        n_simulations=100,  # Number of simulations per amplification factor
        t_max=500          # Maximum simulation time
    )
    
    print("\nAnalysis completed. Summary of results:")
    for factor, data in results.items():
        print(f"Amplification Factor {factor}:")
        print(f"  Total events: {data['total_events']}")
        print(f"  Average events per simulation: {data['avg_events_per_sim']:.2f}")
        print(f"  Mean duration: {data['mean_duration']:.2f}")
        print(f"  Duration CV: {data['cv_duration']:.2f}")
    
    print(f"\nDetailed results and visualizations saved in: {results_dir}")

if __name__ == "__main__":
    main()