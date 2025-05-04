import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# Standard model parameters based on the excitable system in the paper
# Equations (S10)-(S11) from the paper
def default_params():
    params = {
        'ak': 0.004,  # basal expression rate of ComK
        'bk': 0.07,   # saturating expression rate of ComK positive feedback
        'bs': 0.82,   # unrepressed expression rate of ComS
        'k0': 0.2,    # ComK concentration for half-maximal ComK activation
        'k1': 0.222,  # ComK concentration for half-maximal ComS repression
        'n': 2,       # Hill coefficient of ComK positive feedback
        'p': 5        # Hill coefficient of ComS repression by ComK
    }
    return params

def suel_params(scale_factor=1.0):
    """
    Parameters from Table S1 in Süel et al. supplementary information,
    with optional scaling of degradation rates.
    
    Args:
        scale_factor: Factor to scale degradation rates (>1 for shorter competence)
    
    Returns:
        dict: Parameter dictionary
    """
    params = {
        'ak': 0.00875,  # basal expression rate of ComK (molecules/s)
        'bk': 7.5,      # ComK saturating activation rate (molecules/s)
        'bs': 0.06,     # ComS expression rate (molecules/s)
        'k0': 5000,     # ComK half-maximal activation (molecules)
        'k1': 833,      # ComK concentration for half-maximal ComS repression (molecules)
        'n': 2,         # Hill coefficient of ComK positive feedback
        'p': 5,         # Hill coefficient of ComS repression by ComK
        # Additional parameters from the supplementary table
        'GammaK': 25000,  # ComK concentration for half-maximal degradation (molecules)
        'GammaS': 20,     # ComS concentration for half-maximal degradation (molecules)
        'lambdaK': 1e-4,  # Linear ComK degradation rate (s^-1)
        'lambdaS': 1e-4,  # Linear ComS degradation rate (s^-1)
        'deltaK': 0.001 * scale_factor,  # ComK max degradation rate (s^-1), scaled
        'deltaS': 0.001 * scale_factor   # ComS max degradation rate (s^-1), scaled
    }
    return params
    
def dimensionless_suel_params():
    """
    The Suel et al. parameters converted to dimensionless form,
    similar to what's used in the main model. This allows comparison
    with the default parameters.
    """
    params = {
        'ak': 0.00875/25,    # Normalized by deltaK*GammaK
        'bk': 7.5/25,        # Normalized by deltaK*GammaK
        'bs': 0.06/0.02,     # Normalized by deltaS*GammaS
        'k0': 5000/25000,    # Normalized by GammaK
        'k1': 833/25000,     # Normalized by GammaK
        'n': 2,              # Hill coefficient remains the same
        'p': 5               # Hill coefficient remains the same
    }
    return params

# ODE system for the model
def model_odes(t, y, params):
    K, S = y  # ComK, ComS
    
    # Unpack parameters
    ak = params['ak']
    bk = params['bk']
    bs = params['bs']
    k0 = params['k0']
    k1 = params['k1']
    n = params['n']
    p = params['p']
    
    # Check if we're using the Süel parameters (which include more parameters)
    if 'GammaK' in params and 'GammaS' in params:
        # Use the full form from the Süel et al. paper (equations S1-S2)
        GammaK = params['GammaK']
        GammaS = params['GammaS']
        lambdaK = params.get('lambdaK', 1e-4)
        lambdaS = params.get('lambdaS', 1e-4)
        deltaK = params.get('deltaK', 0.001)
        deltaS = params.get('deltaS', 0.001)
        
        # ComK dynamics
        dKdt = ak + (bk * K**n) / (k0**n + K**n) - (deltaK * K) / (1 + K/GammaK + S/GammaS) - lambdaK * K
        
        # ComS dynamics
        dSdt = bs / (1 + (K/k1)**p) - (deltaS * S) / (1 + K/GammaK + S/GammaS) - lambdaS * S
    else:
        # Use the dimensionless form (equations S10-S11)
        # Equation (S10) for ComK
        dKdt = ak + (bk * K**n) / (k0**n + K**n) - K / (1 + K + S)
        
        # Equation (S11) for ComS
        dSdt = bs / (1 + (K/k1)**p) - S / (1 + K + S)
    
    return [dKdt, dSdt]

# Calculate nullclines for the system
def nullclines(K_range, params):
    # Unpack parameters
    ak = params['ak']
    bk = params['bk']
    bs = params['bs']
    k0 = params['k0']
    k1 = params['k1']
    n = params['n']
    p = params['p']
    
    # ComK nullcline: dK/dt = 0 => S as a function of K
    # ak + (bk * K**n) / (k0**n + K**n) - K / (1 + K + S) = 0
    # Solving for S:
    S_from_K = K_range / (ak + (bk * K_range**n) / (k0**n + K_range**n)) - (1 + K_range)
    
    # ComS nullcline: dS/dt = 0 => S as a function of K
    # bs / (1 + (K/k1)**p) - S / (1 + K + S) = 0
    # Using approximation for numerical stability:
    S_from_S = bs * (1 + K_range) / (1 + (K_range/k1)**p - bs)
    
    return S_from_K, S_from_S

# Find fixed points of the system
def find_fixed_points(params, K_range=np.linspace(0.001, 2, 100), S_range=np.linspace(0.001, 2, 100)):
    # Define the equation system
    def equations(vars):
        K, S = vars
        dK, dS = model_odes(0, [K, S], params)
        return [dK, dS]
    
    fixed_points = []
    
    # Try different starting points in K-S space
    for K_start in np.linspace(0.01, 1.5, 10):
        for S_start in np.linspace(0.01, 1.5, 10):
            try:
                # Newton's method to find zeros
                result = root(equations, [K_start, S_start], method='hybr', tol=1e-8)
                
                if result.success:
                    K, S = result.x
                    
                    # Check for non-negative values
                    if K > 0 and S > 0:
                        # Check if this point is already in the list (with tolerance)
                        is_new = True
                        for existing_fp in fixed_points:
                            if np.linalg.norm(np.array([K, S]) - np.array(existing_fp)) < 1e-6:
                                is_new = False
                                break
                        
                        if is_new:
                            fixed_points.append((K, S))
            except:
                continue
    
    return fixed_points

# Calculate the Jacobian matrix at a point
def jacobian(K, S, params):
    # Unpack parameters
    ak = params['ak']
    bk = params['bk']
    bs = params['bs']
    k0 = params['k0']
    k1 = params['k1']
    n = params['n']
    p = params['p']
    
    # Derivative of dK/dt with respect to K
    dK_dK = bk * n * K**(n-1) * k0**n / (k0**n + K**n)**2 - 1/(1+K+S) + K/(1+K+S)**2
    
    # Derivative of dK/dt with respect to S
    dK_dS = K/(1+K+S)**2
    
    # Derivative of dS/dt with respect to K
    dS_dK = -bs * p * (K/k1)**(p-1) / (k1 * (1 + (K/k1)**p)**2) + S/(1+K+S)**2
    
    # Derivative of dS/dt with respect to S
    dS_dS = -1/(1+K+S) + S/(1+K+S)**2
    
    return np.array([[dK_dK, dK_dS], [dS_dK, dS_dS]])

# Classify fixed points based on the Jacobian matrix
def classify_fixed_point(K, S, params):
    J = jacobian(K, S, params)
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(J)
    
    # Calculate determinant and trace
    det = np.linalg.det(J)
    trace = np.trace(J)
    
    # Classification based on eigenvalues
    if det < 0:
        return "Sattelpunkt"
    elif det > 0:
        if trace < 0:
            # All eigenvalues have negative real part
            if np.all(np.imag(eigenvalues) == 0):
                return "Stabiler Knoten"
            else:
                return "Stabiler Fokus"
        elif trace > 0:
            # All eigenvalues have positive real part
            if np.all(np.imag(eigenvalues) == 0):
                return "Instabiler Knoten"
            else:
                return "Instabiler Fokus"
        else:  # trace == 0
            if np.all(np.imag(eigenvalues) != 0):
                return "Zentrum"
            else:
                return "Unklassifiziert"
    else:  # det == 0
        return "Nicht-hyperbolisch"

# Plot phase diagram
def plot_phase_diagram(params, K_range=np.linspace(0, 1, 100), S_range=np.linspace(0, 1, 100), 
                      title="Phasendiagramm", fixed_points=None, show_vector_field=True, trajectories=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Automatic adjustment of axis ranges based on fixed points
    if fixed_points:
        K_max = max([fp[0] for fp in fixed_points]) * 1.5  # 50% buffer
        S_max = max([fp[1] for fp in fixed_points]) * 1.2  # 20% buffer
        
        if max(K_range) < K_max:
            K_range = np.linspace(0, K_max, 200)
        if max(S_range) < S_max:
            S_range = np.linspace(0, S_max, 200)
    
    # Calculate nullclines
    S_from_K, S_from_S = nullclines(K_range, params)
    
    # Mask for valid values (non-negative and finite)
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    # Plot nullclines
    ax.plot(K_range[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullkline (dK/dt = 0)')
    ax.plot(K_range[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullkline (dS/dt = 0)')
    
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
    ax.set_xlabel('ComK Konzentration')
    ax.set_ylabel('ComS Konzentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig, ax

# Simulate the system over time
def simulate_system(params, initial_conditions=[0.01, 0.2], t_max=200, num_points=1000):
    t_span = [0, t_max]
    t_eval = np.linspace(0, t_max, num_points)
    
    # Integration of ODEs
    sol = solve_ivp(
        lambda t, y: model_odes(t, y, params),
        t_span,
        initial_conditions,
        method='RK45',
        t_eval=t_eval
    )
    
    return sol.t, sol.y[0], sol.y[1]  # Time, ComK, ComS

# Plot time series
def plot_time_series(t, K, S, threshold=0.5, title="Zeitreihe der Kompetenzdynamik"):
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
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label='Kompetenz-Schwelle')
    
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Konzentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True)
    
    # Display competence times in the plot
    if competence_periods:
        competence_durations = [end-start for start, end in competence_periods]
        avg_duration = np.mean(competence_durations)
        text_str = f"Anzahl Kompetenzereignisse: {len(competence_periods)}\nDurchschn. Dauer (Tc): {avg_duration:.2f}"
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5),
                verticalalignment='top')
    
    return fig, ax, competence_periods

# Find excitable configurations
def find_excitable_configurations(params_base, n_samples=10000):
    """
    Systematically search for parameter configurations that generate excitable systems.
    
    Args:
        params_base: Base parameters
        n_samples: Number of parameter combinations to test
    
    Returns:
        List of parameter tuples that generate excitable systems
    """
    print(f"Starting search for excitable configurations with {n_samples} samples...")
    
    # Parameter ranges based on the Schultz paper 
    # Note: The ranges are adjusted to avoid zero values and to ensure a more realistic search space
    # Schultz et al. ak: 0.0-0.2 und all others 0.0-1.0
    param_ranges = {
        'n': [2, 3, 4, 5],             # Hill coefficient for ComK (discrete values)
        'p': [2, 3, 4, 5],             # Hill coefficient for ComS (discrete values)
        'ak': np.linspace(0.001, 0.1, 20),  # basal ComK expression 
        'bk': np.linspace(0.04, 0.5, 20),    # ComK feedback
        'bs': np.linspace(0.1, 1.0, 20),     # ComS expression
        'k0': np.linspace(0.1, 1.0, 20),     # ComK activation
        'k1': np.linspace(0.1, 1.0, 20)      # ComS repression
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
        
        # Find and classify fixed points
        fps = find_fixed_points(params_test)
        
        if len(fps) == 3:  # Potentially excitable system with 3 fixed points
            types = [classify_fixed_point(fp[0], fp[1], params_test) for fp in fps]
            
            # Check for: 1 stable, 1 saddle, and 1 unstable fixed point
            has_stable = any('Stabil' in fp_type for fp_type in types)
            has_saddle = any('Sattel' in fp_type for fp_type in types)
            has_unstable = any('Instabil' in fp_type for fp_type in types)
            
            if has_stable and has_saddle and has_unstable:
                # Check if stable fixed point is at low ComK (vegetative state)
                stable_fps = [fp for fp, fp_type in zip(fps, types) if 'Stabil' in fp_type]
                
                if any(fp[0] < 0.2 for fp in stable_fps):  # Stable point with low ComK
                    # Save parameters for excitable system
                    excitable_configs.append((
                        params_test['n'], params_test['p'], 
                        params_test['ak'], params_test['bk'], 
                        params_test['bs'], params_test['k0'], 
                        params_test['k1']
                    ))
    
    print(f"Search completed: {len(excitable_configs)} excitable configurations found.")
    return excitable_configs

# Create histograms of parameters for excitable systems
def plot_excitable_parameter_histograms(excitable_configs, params_base):
    """
    Create histograms of parameters that lead to excitable systems,
    similar to Figure 8 in the Schultz paper.
    
    Args:
        excitable_configs: List of tuples with excitable parameter configurations
        params_base: Base parameter set
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
            axes[i].set_ylabel('Häufigkeit')
            axes[i].set_title(f'Verteilung von {param}')
            
            # Show standard value
            std_val = standard_params[param]
            axes[i].axvline(std_val, color='b', linestyle='-', 
                          label=f'Standard: {std_val:.3f}')
            
            # Show mean and median for parameters other than Hill coefficients
            if param not in ['n', 'p']:
                mean_val = np.mean(param_values[param])
                median_val = np.median(param_values[param])
                axes[i].axvline(mean_val, color='r', linestyle='--', 
                              label=f'Mittelwert: {mean_val:.3f}')
                axes[i].axvline(median_val, color='g', linestyle=':', 
                              label=f'Median: {median_val:.3f}')
            
            axes[i].legend(fontsize='small')
    
    # Add an example nullcline plot
    sample_config_idx = len(excitable_configs) // 2  # Take a middle parameter set
    sample_params = params_base.copy()
    
    for i, param_name in enumerate(param_names):
        sample_params[param_name] = excitable_configs[sample_config_idx][i]
    
    K_range = np.linspace(0, 1, 100)
    S_range = np.linspace(0, 10, 100)  # Larger range for ComS
    
    S_from_K, S_from_S = nullclines(K_range, sample_params)
    
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    axes[7].plot(K_range[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullkline')
    axes[7].plot(K_range[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullkline')
    axes[7].set_xlabel('ComK')
    axes[7].set_ylabel('ComS')
    axes[7].set_title('Beispiel Nullklinen')
    axes[7].legend()
    
    plt.tight_layout()
    
    return fig, axes, param_values

# Create an optimized nullcline diagram for an excitable system
def plot_example_nullclines(params, title="Beispiel Nullklinen"):
    """
    Draw an optimized nullcline diagram for an excitable system
    """
    # Adjusted axis ranges for better visualization
    K_range = np.linspace(0, 1.0, 200)
    S_range = np.linspace(0, 12.0, 200)  # Higher range for ComS
    
    # Calculate nullclines
    S_from_K, S_from_S = nullclines(K_range, params)
    
    # Find fixed points
    fps = find_fixed_points(params)
    fp_types = [classify_fixed_point(fp[0], fp[1], params) for fp in fps]
    
    # Determine axis limits
    valid_K = np.isfinite(S_from_K) & (S_from_K >= 0)
    valid_S = np.isfinite(S_from_S) & (S_from_S >= 0)
    
    # Determine max values for axis scaling
    if np.any(valid_K):
        s_max = np.max(S_from_K[valid_K]) * 1.2
    else:
        s_max = 10.0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot nullclines
    ax.plot(K_range[valid_K], S_from_K[valid_K], 'b-', label='ComK Nullkline', linewidth=2)
    ax.plot(K_range[valid_S], S_from_S[valid_S], 'g-', label='ComS Nullkline', linewidth=2)
    
    # Plot fixed points
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
    
    for i, (fp, fp_type) in enumerate(zip(fps, fp_types)):
        ax.plot(fp[0], fp[1], colors.get(fp_type, "ko"), markersize=10, 
                label=f'FP{i+1}: {fp_type} ({fp[0]:.3f}, {fp[1]:.3f})')
    
    # Adjust axis limits
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, s_max])
    
    # Labels
    ax.set_xlabel('ComK Konzentration')
    ax.set_ylabel('ComS Konzentration')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig, ax