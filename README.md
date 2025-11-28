# ATM Location Optimizer

A comprehensive optimization toolkit for strategic ATM placement using real-world travel time data, multiple optimization algorithms (Annueling and Greedy algorithms), and advanced visualization capabilities.
The solution to classical optimal location problem is applied to ATM sites within an area in North Madrid. The data is collected from municipality and OSRM (Open Source Routing Machine). The back up for the API calls is provided so  it can be run off line.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Daniel%20Paz%20Martinez-blue)](https://www.linkedin.com/in/daniel-paz-martinez/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Data Requirements](#data-requirements)
- [Output Files](#output-files)
- [Examples](#examples)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project solves the **facility location problem** specifically for ATM placement by:

1. **Calculating real-world travel times** using the OSRM (Open Source Routing Machine) API
2. **Optimizing ATM locations** using Simulated Annealing and Greedy algorithms
3. **Comparing algorithms statistically** with comprehensive analysis
4. **Visualizing results** through interactive maps and animated visualizations
5. **Generating demand points** from population and building footprint data

### Use Case

Originally developed for optimizing ATM placement in Madrid's Fuencarral-El Pardo district, this tool is **adaptable to any geographical area** and can be used for various facility location problems (retail stores, emergency services, distribution centers, etc.). The first 21 locations coincide with existing ATMs and the rest up to 100 are randomly collected from the
existing demand points. The winners given by the algorithm could indicate locations that minimize driving distance and hence become candidates for ATM locations

## ✨ Features

### Core Optimization
- **Real travel time calculations** via OSRM routing API (not just Euclidean distance)
- **Two optimization algorithms**:
  - Simulated Annealing with adaptive reheating
  - Greedy algorithm for baseline comparison
- **Weighted demand points** to account for population density. The default is taking all the demand weights equally.
- **Result caching** for faster iterations and experimentation
- **Statistical comparison** with hypothesis testing and effect size analysis

### Visualization
- **Interactive HTML maps** using Folium
- **Static plots** for algorithm performance comparison
- **Animated GIFs** showing route development
- **Side-by-side algorithm comparisons** in animated format
- **Street-level basemaps** using Contextily

### Population Distribution (Optional)
- **Generate demand points** from OpenStreetMap building footprints
- **Separate residential and daytime populations**
- **Configurable distribution percentages**
- **Integration with geographical data** (shapefiles)

## 🏗️ Architecture & Design Decisions

### Modular Structure

The project is organized into specialized modules for maintainability and extensibility:

```
atm_optimizer/
├── __init__.py              # Package initialization
├── config.py                # Centralized configuration management
├── optimizer.py             # Core optimization algorithms
├── visualizer.py            # Map and plot generation
├── animator.py              # Animated visualization creation
├── population_distribution.py  # Demand point generation (optional)
├── utils.py                 # Helper functions and logging
└── cli.py                   # Command-line interface
```

**Why this structure?**

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Easy Testing**: Modules can be tested independently
3. **Extensibility**: New optimization algorithms or visualization methods can be added without modifying existing code
4. **Maintainability**: Changes in one module don't cascade to others
5. **Reusability**: Modules can be imported individually for custom workflows

### Configuration Management

All settings are centralized in `config.py`:
- **File paths** are relative and auto-configured
- **Algorithm parameters** can be tuned without code changes
- **Visualization settings** are easily customizable
- **Logging behavior** is configurable

This design decision eliminates hardcoded values and makes the tool more portable.

### Logging Strategy

Both file and console logging are implemented because:
- **Console logging**: Provides real-time feedback during optimization
- **File logging**: Enables post-mortem analysis and debugging
- **Configurable levels**: DEBUG mode for development, INFO for production

### Caching Mechanism

Travel time caching is crucial because:
- **API rate limits**: OSRM has usage limits
- **Reproducibility**: Same travel times across different runs
- **Speed**: Eliminates redundant API calls (saves minutes to hours)
- **Offline capability**: Once cached, no internet required

### Error Handling

Robust error handling includes:
- **Graceful fallbacks**: Uses cache if API fails
- **Data validation**: Checks file formats and coordinate validity
- **Informative errors**: Clear messages guide users to solutions
- **Retry logic**: Automatic retries for transient API failures

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for first run to fetch travel times). This is not necessary as a  cache file is given.

### Step 1: Clone the Repository

```bash
git clone https://github.com/pazdanielmartinez6/location-optimizer/
cd atm-location-optimizer
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import atm_optimizer; print(atm_optimizer.__version__)"
```

## 🚀 Quick Start

### Option 1: Using Provided Data (Recommended for First Run)

The repository includes backup data files for Madrid's Fuencarral-El Pardo district:

```bash
python -m atm_optimizer.cli
```

Follow the interactive prompts:
1. Choose "Use cached travel times" (faster, uses backup data)
2. Select "Single optimization run"
3. Enter number of ATMs (e.g., 5)
4. Choose optimization method (Simulated Annealing recommended)
5. Decide if you want animations

**Output**: Maps and results will be saved in `outputs/YYYYMMDD_HHMMSS_Natms/`

### Option 2: Generate Your Own Demand Points

If you have shapefiles and population data:

```bash
python -m atm_optimizer.population_distribution
```

This will generate `demand_points.csv` from building footprints.

## 📖 Detailed Usage

### 1. Single Optimization Run

Optimize ATM locations for a specific number of ATMs:

```python
from atm_optimizer import ATMLocationOptimizer
import pandas as pd

# Load data
demand_points = pd.read_csv('data/demand_points.csv')
candidates = pd.read_csv('data/atm_candidates.csv')

# Initialize optimizer
optimizer = ATMLocationOptimizer(
    demand_points=demand_points,
    candidate_locations=candidates,
    use_cache=True  # Use cached travel times
)

# Run optimization
solution = optimizer.optimize(
    num_atms=5,
    method='simulated_annealing'
)

# Evaluate solution
objective_value = optimizer.evaluate_solution(solution)
print(f"Objective value: {objective_value:.2f}")

# Visualize
map_viz = optimizer.visualize_solution(solution)
map_viz.save('atm_solution_map.html')
```

### 2. Statistical Algorithm Comparison

Compare Simulated Annealing vs Greedy across multiple runs:

```python
results = optimizer.run_algorithm_comparison(
    num_atms=5,
    num_iterations=30,  # Minimum 30 for statistical validity
    save_results=True
)

# Results include:
# - Descriptive statistics (mean, std, min, max)
# - Statistical tests (t-test or Mann-Whitney U)
# - Effect size (Cohen's d)
# - Performance improvement percentage
# - Execution time comparison
```

### 3. Create Animated Visualizations

Generate animated GIFs showing route development:

```python
from atm_optimizer import ATMAnimator

animator = ATMAnimator(optimizer)

# Single algorithm animation
animator.create_solution_animation(
    solution=solution,
    save_path='outputs/animated_solution.gif',
    add_streets=True,
    max_demand_points=200  # Sample for faster rendering
)

# Algorithm comparison animation
animator.create_comparison_animation(
    solution_sa=sa_solution,
    solution_greedy=greedy_solution,
    save_path='outputs/comparison.gif'
)
```

### 4. Generate Custom Demand Points

Create demand points from your own geographical data:

```python
from atm_optimizer import PopulationDistributor
import geopandas as gpd

# Load your shapefile
barrios = gpd.read_file('data/your_shapefile.shp')

# Configure population data
population_data = [
    {"code": "001", "name": "District 1", "population": 50000},
    {"code": "002", "name": "District 2", "population": 35000},
    # ...
]

# Generate demand points
distributor = PopulationDistributor(
    barrios_gdf=barrios,
    population_data=population_data
)

demand_points = distributor.generate_demand_points(
    residential_percentage=0.012,
    daytime_percentage=0.01
)

demand_points.to_csv('data/my_demand_points.csv', index=False)
```

## 📊 Data Requirements

### Required Files

#### 1. `demand_points.csv`
Represents locations where ATM services are needed.

**Columns:**
- `id` (int): Unique identifier
- `latitude` (float): Latitude in WGS84
- `longitude` (float): Longitude in WGS84
- `demand_type` (str, optional): 'residential' or 'daytime'

**Example:**
```csv
id,latitude,longitude,demand_type
1,40.5234,-3.6912,residential
2,40.5245,-3.6898,daytime
...
```

### Optional Files

#### 2. `atm_candidates.csv`
Pre-selected candidate locations for ATMs.

**Columns:**
- `id` (int): Unique identifier
- `latitude` (float): Latitude in WGS84
- `longitude` (float): Longitude in WGS84

If not provided, candidates are randomly sampled from demand points.

#### 3. `population_weights.csv`
Weights for demand points (higher weight = higher priority).

**Columns:**
- `id` (int): Matches demand point ID
- `weight` (float): Weight value

If not provided, all points have weight = 1.

#### 4. `travel_times_cache.pkl`
Pre-calculated travel times (binary file).

Automatically generated on first run and reused in subsequent runs.

### For Population Distribution

#### 5. `BARRIOS.shp` (+ associated files)
Shapefile defining geographical boundaries.

**Required for**: Generating demand points from building footprints

**Where to get it**: OpenStreetMap, municipal GIS portals, or government open data

## 📁 Output Files

Each optimization run creates a timestamped directory:

```
outputs/YYYYMMDD_HHMMSS_Natms_[experiment_name]/
├── metadata.json                          # Run parameters and results
├── atm_solution.csv                       # Selected ATM locations
├── atm_locations_map.html                 # Interactive map
├── solution_with_all_candidates.html      # Map showing all candidates
├── animated_atm_solution.gif              # Animated visualization (optional)
├── algorithm_comparison_Natms.csv         # Statistical results (if comparison run)
├── algorithm_comparison_plots.png         # Comparison plots
└── logs/atm_optimizer_YYYYMMDD_HHMMSS.log  # Detailed log file
```

### Output Descriptions

- **metadata.json**: Contains objective value, execution time, parameters
- **atm_solution.csv**: Coordinates of selected ATM locations
- **HTML maps**: Interactive Folium maps (open in browser)
- **GIF animations**: Route development visualization
- **CSV results**: Detailed statistical analysis data
- **PNG plots**: Box plots, histograms, convergence plots

## 💡 Examples

### Example 1: Basic Optimization

```bash
cd examples
python basic_optimization.py
```

### Example 2: Algorithm Comparison

```bash
python algorithm_comparison.py
```

### Example 3: Custom Visualization

```bash
python custom_visualization.py
```

### Example 4: Jupyter Notebook

```bash
jupyter notebook optimization_workflow.ipynb
```

## 📚 API Documentation

### ATMLocationOptimizer

Main class for ATM location optimization.

#### Methods

**`__init__(demand_points, candidate_locations=None, population_weights=None, use_cache=True)`**
- Initialize optimizer with data

**`optimize(num_atms, method='simulated_annealing', **kwargs)`**
- Run optimization
- Returns: List of selected ATM IDs

**`evaluate_solution(solution)`**
- Calculate objective value for a solution
- Returns: Float (lower is better)

**`run_algorithm_comparison(num_atms, num_iterations=30)`**
- Compare algorithms statistically
- Returns: Dict with statistical results

**`visualize_solution(solution)`**
- Create interactive map
- Returns: folium.Map object

**`save_travel_times(filename)`**
- Save calculated travel times to cache

**`load_from_cache(filename)` (classmethod)**
- Load optimizer with cached data

### Configuration

Update settings programmatically:

```python
from atm_optimizer import update_config

update_config(
    sa_iterations=15000,
    sa_initial_temp=500,
    animation_max_demand_points=150
)
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution

- Additional optimization algorithms (Genetic Algorithm, Tabu Search, etc.)
- Performance optimizations
- Additional visualization options
- Support for other routing APIs
- Unit tests
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Daniel Paz Martinez**

- LinkedIn: [https://www.linkedin.com/in/daniel-paz-martinez/](https://www.linkedin.com/in/daniel-paz-martinez/)
- GitHub: [@your-github-username](https://github.com/your-github-username)

---

## 🙏 Acknowledgments

- **OSRM Project** for providing the routing API
- **OpenStreetMap** contributors for geographical data
- **Madrid Open Data** for shapefiles and population data

## 📧 Contact

Questions or suggestions? Feel free to:
- Open an issue on GitHub
- Connect with me on [LinkedIn](https://www.linkedin.com/in/daniel-paz-martinez/)

---

**Made with ❤️ for optimizing urban services**
