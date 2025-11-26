"""
Configuration module for ATM Location Optimizer.
Centralized management of all settings, paths, and parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataPaths:
    """Configuration for data file paths."""
    
    # Input files
    demand_points: Path = DATA_DIR / "demand_points.csv"
    atm_candidates: Path = DATA_DIR / "atm_candidates.csv"
    population_weights: Path = DATA_DIR / "population_weights.csv"
    barrios_shapefile: Path = DATA_DIR / "BARRIOS.shp"
    
    # Cache files
    travel_times_cache: Path = DATA_DIR / "travel_times_cache.pkl"
    
    # Output directory
    output_dir: Path = OUTPUT_DIR
    
    # Logs directory
    logs_dir: Path = LOGS_DIR
    
    def get_output_path(self, filename: str) -> Path:
        """Get path for an output file."""
        return self.output_dir / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get path for a log file."""
        return self.logs_dir / filename


@dataclass
class OSRMConfig:
    """Configuration for OSRM API settings."""
    
    base_url: str = "http://router.project-osrm.org/table/v1/driving/"
    batch_size: int = 50
    effective_batch_size: int = 25
    max_total_coords: int = 50
    request_timeout: int = 30
    request_delay: float = 0.1  # seconds between requests
    max_url_length: int = 8192
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    
    # Simulated Annealing parameters
    sa_initial_temp: float = 400.0
    sa_cooling_rate: float = 0.95
    sa_iterations: int = 10000
    sa_reheat_threshold: int = 10
    sa_beta: float = 0.5
    
    # Statistical comparison parameters
    stat_num_iterations: int = 30
    stat_min_iterations: int = 10
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    # Map settings
    default_zoom: int = 10
    
    # Animation settings
    animation_frames_per_route: int = 2
    animation_interval: int = 50
    animation_max_demand_points: int = 200
    animation_fps: int = 20
    
    # Street layer settings
    add_streets: bool = True
    street_zoom_level: int = 12
    street_alpha: float = 0.7
    
    # Figure settings
    figure_size: tuple = (15, 10)
    comparison_figure_size: tuple = (20, 10)
    dpi: int = 300


@dataclass
class PopulationDistributionConfig:
    """Configuration for population distribution."""
    
    # Distribution percentages
    residential_percentage: float = 0.012
    daytime_percentage: float = 0.01
    
    # Building settings
    min_people_per_building: int = 1
    
    # Building type filters
    residential_filters: List[str] = field(default_factory=lambda: [
        'residential', 'apartments', 'house', 'detached', 'terrace',
        'semidetached_house', 'farm', 'farmhouse', 'bungalow', 'cabin',
        'hut', 'villa', 'cottage', 'dwelling', 'static_caravan', 'chalet', 
        'flat', 'building'
    ])
    
    daytime_filters: List[str] = field(default_factory=lambda: [
        'commercial', 'office', 'industrial', 'retail', 'warehouse',
        'factory', 'government', 'public', 'hospital', 'university',
        'college', 'school', 'civic', 'institutional'
    ])
    
    # El Pardo bounding box (optional) - (min_x, min_y, max_x, max_y)
    el_pardo_box: Optional[tuple] = (430600, 4485800, 434000, 4488200)


@dataclass
class ProjectConfig:
    """Main configuration class that aggregates all settings."""
    
    paths: DataPaths = field(default_factory=DataPaths)
    osrm: OSRMConfig = field(default_factory=OSRMConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    population: PopulationDistributionConfig = field(default_factory=PopulationDistributionConfig)
    
    # Coordinate reference systems
    source_crs: Optional[str] = "EPSG:25830"  # Madrid CRS (ETRS89 / UTM zone 30N)
    target_crs: str = "EPSG:4326"  # WGS84 for OSRM
    is_projected: bool = True
    
    # Logging configuration
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_paths()
    
    def _validate_paths(self):
        """Check if required input files exist and log status."""
        required_files = {
            'Demand Points': self.paths.demand_points,
            'Travel Times Cache': self.paths.travel_times_cache,
        }
        
        optional_files = {
            'ATM Candidates': self.paths.atm_candidates,
            'Population Weights': self.paths.population_weights,
            'Barrios Shapefile': self.paths.barrios_shapefile,
        }
        
        print("\n" + "="*60)
        print("DATA FILES STATUS")
        print("="*60)
        
        print("\nRequired Files:")
        for name, path in required_files.items():
            if path.exists():
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name}: {path} (MISSING)")
        
        print("\nOptional Files:")
        for name, path in optional_files.items():
            if path.exists():
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  - {name}: {path} (not found)")
        
        print("="*60 + "\n")
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ProjectConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {
            'osrm': self.osrm.__dict__,
            'optimization': self.optimization.__dict__,
            'visualization': self.visualization.__dict__,
            'population': self.population.__dict__,
            'source_crs': self.source_crs,
            'is_projected': self.is_projected,
            'log_level': self.log_level,
        }


# Global configuration instance
_config = ProjectConfig()


def get_config() -> ProjectConfig:
    """Get the current global configuration."""
    return _config


def update_config(**kwargs):
    """
    Update global configuration with provided parameters.
    
    Example:
        update_config(log_level='DEBUG', sa_iterations=5000)
    """
    global _config
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
            logger.info(f"Updated config: {key} = {value}")
        else:
            logger.warning(f"Unknown configuration parameter: {key}")


def reset_config():
    """Reset configuration to defaults."""
    global _config
    _config = ProjectConfig()
    logger.info("Configuration reset to defaults")


def print_config():
    """Print current configuration to console."""
    config = get_config()
    
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    
    print("\n[Data Paths]")
    print(f"  Data Directory:   {DATA_DIR}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Logs Directory:   {LOGS_DIR}")
    
    print("\n[OSRM Settings]")
    print(f"  Batch Size:       {config.osrm.batch_size}")
    print(f"  Request Timeout:  {config.osrm.request_timeout}s")
    print(f"  Max Retries:      {config.osrm.max_retries}")
    
    print("\n[Optimization]")
    print(f"  SA Iterations:    {config.optimization.sa_iterations}")
    print(f"  Initial Temp:     {config.optimization.sa_initial_temp}")
    print(f"  Cooling Rate:     {config.optimization.sa_cooling_rate}")
    print(f"  Random Seed:      {config.optimization.random_seed}")
    
    print("\n[Visualization]")
    print(f"  Add Streets:      {config.visualization.add_streets}")
    print(f"  Max Demand Pts:   {config.visualization.animation_max_demand_points}")
    print(f"  DPI:              {config.visualization.dpi}")
    
    print("\n[Coordinate Systems]")
    print(f"  Source CRS:       {config.source_crs}")
    print(f"  Target CRS:       {config.target_crs}")
    print(f"  Is Projected:     {config.is_projected}")
    
    print("\n[Logging]")
    print(f"  Log Level:        {config.log_level}")
    print(f"  Log to File:      {config.log_to_file}")
    print(f"  Log to Console:   {config.log_to_console}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test configuration
    print_config()
