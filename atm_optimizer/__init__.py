"""
ATM Location Optimizer

A comprehensive optimization toolkit for ATM placement using:
- Real-world travel time data (OSRM API)
- Simulated Annealing and Greedy algorithms
- Statistical analysis and comparison
- Interactive and animated visualizations

Author: Daniel Paz Martinez
GitHub: https://github.com/[your-username]/atm-location-optimizer
LinkedIn: https://www.linkedin.com/in/daniel-paz-martinez/
"""

__version__ = "1.0.0"
__author__ = "Daniel Paz Martinez"

from .optimizer import ATMLocationOptimizer
from .config import ProjectConfig, get_config, update_config
from .visualizer import ATMVisualizer
from .animator import ATMAnimator
from .population_distribution import PopulationDistributor
from .utils import setup_logging, validate_data_files

__all__ = [
    'ATMLocationOptimizer',
    'ProjectConfig',
    'get_config',
    'update_config',
    'ATMVisualizer',
    'ATMAnimator',
    'PopulationDistributor',
    'setup_logging',
    'validate_data_files',
]
