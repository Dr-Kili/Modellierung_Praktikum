# B. subtilis Competence Circuit Analysis

This repository contains Python scripts for analyzing the competence circuit dynamics in B. subtilis bacteria. The code has been refactored into modular components for improved organization and reusability.

## Project Structure

The project is organized into the following modules:

- `competence_circuit_analysis.py` - Core model definitions and analysis functions
- `helpers.py` - Common utility functions
- `visualization.py` - Visualization functions for creating plots and diagrams
- `simulation.py` - Functions for deterministic and stochastic simulations
- `parameter_search.py` - Tools for searching parameter spaces for excitable configurations
- `run_analysis.py` - Main script that ties everything together and allows customization

## Getting Started

### Prerequisites

The code requires the following Python packages:

```
numpy
matplotlib
pandas
scipy
scikit-learn
```

You can install these with pip:

```bash
pip install numpy matplotlib pandas scipy scikit-learn
```

### Running the Analysis

You can run the analysis in several ways:

#### 1. Default Analysis

To run the full analysis with default settings, simply execute:

```bash
python run_analysis.py
```

#### 2. Using a Configuration File

You can customize the analysis by creating a JSON configuration file:

```bash
python run_analysis.py --config analysis_config.json
```

See the example `analysis_config.json` for available parameters.

#### 3. Command-line Options

You can run specific analyses and set parameters directly from the command line:

```bash
# Run only standard parameter analysis and stochastic analysis
python run_analysis.py --standard --stochastic

# Run Suel parameter search with a higher resolution
python run_analysis.py --suel --suel-points 150

# Run all analyses with customized stochastic simulation parameters
python run_analysis.py --all --stochastic-sims 50 --stochastic-time 500
```

#### Available Command-line Options

**Module Selection:**
- `--all`: Run all analyses
- `--standard`: Run standard parameter analysis
- `--excitable`: Run excitable configuration search
- `--hill`: Run Hill coefficient analysis
- `--suel`: Run Suel parameter space search
- `--stochastic`: Run stochastic analysis
- `--amplification`: Run amplification factor analysis

**Parameter Overrides:**
- `--output-prefix`: Prefix for output directory
- `--excitable-samples`: Number of samples for excitable search
- `--hill-grid-size`: Grid size for Hill coefficient search
- `--suel-points`: Number of points for Suel search
- `--stochastic-sims`: Number of stochastic simulations
- `--stochastic-time`: Maximum time for stochastic simulations
- `--amp-sims`: Number of amplification simulations
- `--amp-time`: Maximum time for amplification simulations

## Configuration Options

You can customize the analysis by modifying the following parameters:

### General Parameters
- `output_prefix`: Prefix for output directory name

### Module Enable/Disable
- `run_standard_analysis`: Whether to analyze standard parameters
- `run_excitable_search`: Whether to search for excitable configurations
- `run_hill_analysis`: Whether to analyze Hill coefficient effects
- `run_suel_search`: Whether to search the Suel parameter space
- `run_stochastic_analysis`: Whether to run stochastic simulations
- `run_amplification_analysis`: Whether to analyze amplification factors

### Analysis-specific Parameters
- `standard_t_max`: Maximum time for standard parameter simulations
- `excitable_n_samples`: Number of samples for excitable search
- `hill_grid_size`: Grid size for Hill coefficient search
- `hill_extended_search`: Whether to perform extended search for Hill coefficients
- `suel_n_points`: Number of points per dimension for Suel search
- `stochastic_n_simulations`: Number of stochastic simulations per parameter set
- `stochastic_t_max`: Maximum time for stochastic simulations
- `stochastic_amplification`: Noise amplification factor for stochastic simulations
- `amplification_factors`: List of amplification factors to analyze
- `amplification_n_simulations`: Number of simulations per amplification factor
- `amplification_t_max`: Maximum time for amplification factor simulations

## Example Configuration File

Here's an example of a configuration file:

```json
{
    "output_prefix": "competence_analysis_custom",
    
    "run_standard_analysis": true,
    "run_excitable_search": true,
    "run_hill_analysis": true,
    "run_suel_search": true,
    "run_stochastic_analysis": true,
    "run_amplification_analysis": true,
    
    "standard_t_max": 150,
    
    "excitable_n_samples": 25000,
    
    "hill_grid_size": 30,
    "hill_extended_search": true,
    
    "suel_n_points": 75,
    
    "stochastic_n_simulations": 30,
    "stochastic_t_max": 300,
    "stochastic_amplification": 10,
    
    "amplification_factors": [1, 2, 5, 10, 15],
    "amplification_n_simulations": 25,
    "amplification_t_max": 300
}
```

## Individual Analysis Components

You can also run specific parts of the analysis programmatically by importing and calling the functions from `run_analysis.py`:

```python
from run_analysis import analyze_standard_parameters, run_stochastic_analysis, load_parameters

# Load default parameters or from config
params = load_parameters('my_config.json')

# Create results directory
from helpers import create_results_directory
results_dir = create_results_directory(params["output_prefix"])

# Run only specific analyses
std_info = analyze_standard_parameters(results_dir, params)
stochastic_results = run_stochastic_analysis(results_dir, params)
```

## Modifying the Model

To modify the model, you'll primarily need to edit the `competence_circuit_analysis.py` file, which contains the core model definitions and ODEs.

## Visualizing Results

The `visualization.py` module provides various functions for creating plots and diagrams. You can use these functions directly to visualize your results:

```python
from visualization import plot_phase_diagram, plot_time_series
from competence_circuit_analysis import default_params, find_fixed_points, model_odes
from simulation import simulate_system
import matplotlib.pyplot as plt

# Load standard parameters and find fixed points
params = default_params()
fixed_points = find_fixed_points(params)

# Simulate the system
t, K, S = simulate_system(model_odes, params)

# Create and save plots
fig, ax = plot_phase_diagram(params, fixed_points=fixed_points)
plt.savefig('phase_diagram.png')
plt.close()

fig, ax, events = plot_time_series(t, K, S)
plt.savefig('time_series.png')
plt.close()
```

## Contributing

Feel free to fork this repository and make your own modifications. If you find any bugs or have suggestions for improvements, please open an issue.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This code was developed for analyzing the competence circuit dynamics in B. subtilis bacteria based on models described in the literature.