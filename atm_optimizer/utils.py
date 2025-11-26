"""
Utility functions for ATM Location Optimizer.
Includes logging setup, data validation, and helper functions.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .config import get_config


def setup_logging(log_level: str = "INFO", 
                  log_to_file: bool = True,
                  log_to_console: bool = True,
                  log_filename: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the application.
    
    Parameters:
    -----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_to_file : bool
        Whether to log to file
    log_to_console : bool
        Whether to log to console
    log_filename : str, optional
        Custom log filename (default: atm_optimizer_YYYYMMDD_HHMMSS.log)
    
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    config = get_config()
    
    # Create logger
    logger = logging.getLogger('atm_optimizer')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"atm_optimizer_{timestamp}.log"
        
        log_path = config.paths.get_log_path(log_filename)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def validate_data_files() -> Tuple[bool, list]:
    """
    Validate that required data files exist.
    
    Returns:
    --------
    Tuple[bool, list]
        (all_valid, missing_files)
    """
    config = get_config()
    logger = logging.getLogger('atm_optimizer')
    
    required_files = [
        ('Demand Points', config.paths.demand_points),
    ]
    
    missing_files = []
    
    for name, path in required_files:
        if not path.exists():
            missing_files.append((name, str(path)))
            logger.error(f"Required file missing: {name} at {path}")
    
    all_valid = len(missing_files) == 0
    
    if all_valid:
        logger.info("All required data files validated successfully")
    else:
        logger.error(f"Missing {len(missing_files)} required file(s)")
    
    return all_valid, missing_files


def validate_dataframe(df: pd.DataFrame, 
                       required_columns: list,
                       name: str = "DataFrame") -> bool:
    """
    Validate that a DataFrame has required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
    name : str
        Name of the DataFrame for error messages
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    logger = logging.getLogger('atm_optimizer')
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        logger.error(f"{name} missing required columns: {missing_cols}")
        return False
    
    logger.debug(f"{name} validation passed")
    return True


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within reasonable bounds.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return False
    
    if np.isnan(lat) or np.isnan(lon) or np.isinf(lat) or np.isinf(lon):
        return False
    
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False
    
    return True


def create_output_directory(num_atms: int, 
                            experiment_name: Optional[str] = None) -> Path:
    """
    Create an organized output directory for results.
    
    Directory structure: outputs/YYYYMMDD_HHMMSS_Natms_[experiment_name]/
    
    Parameters:
    -----------
    num_atms : int
        Number of ATMs in the solution
    experiment_name : str, optional
        Custom experiment name
    
    Returns:
    --------
    Path
        Path to created directory
    """
    config = get_config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        dir_name = f"{timestamp}_{num_atms}atms_{experiment_name}"
    else:
        dir_name = f"{timestamp}_{num_atms}atms"
    
    output_dir = config.paths.output_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('atm_optimizer')
    logger.info(f"Created output directory: {output_dir}")
    
    return output_dir


def load_data_file(file_path: Path, 
                   required_columns: list,
                   file_description: str = "file") -> Optional[pd.DataFrame]:
    """
    Load and validate a data file.
    
    Parameters:
    -----------
    file_path : Path
        Path to the data file
    required_columns : list
        Required columns in the file
    file_description : str
        Description for error messages
    
    Returns:
    --------
    pd.DataFrame or None
        Loaded DataFrame, or None if error
    """
    logger = logging.getLogger('atm_optimizer')
    
    try:
        if not file_path.exists():
            logger.error(f"{file_description} not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        
        if not validate_dataframe(df, required_columns, file_description):
            return None
        
        logger.info(f"Loaded {file_description}: {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error loading {file_description}: {e}")
        return None


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Parameters:
    -----------
    seconds : float
        Time in seconds
    
    Returns:
    --------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_section_header(title: str, width: int = 60):
    """
    Print a formatted section header.
    
    Parameters:
    -----------
    title : str
        Section title
    width : int
        Width of the header
    """
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def print_progress(current: int, total: int, prefix: str = "", width: int = 50):
    """
    Print a progress bar.
    
    Parameters:
    -----------
    current : int
        Current progress value
    total : int
        Total value
    prefix : str
        Prefix text
    width : int
        Width of progress bar
    """
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '-' * (width - filled)
    print(f'\r{prefix} |{bar}| {percent:.1%}', end='', flush=True)
    
    if current == total:
        print()  # New line when complete


def save_results_metadata(output_dir: Path, 
                         num_atms: int,
                         method: str,
                         objective_value: float,
                         execution_time: float,
                         **kwargs):
    """
    Save metadata about the optimization run.
    
    Parameters:
    -----------
    output_dir : Path
        Output directory
    num_atms : int
        Number of ATMs
    method : str
        Optimization method used
    objective_value : float
        Final objective value
    execution_time : float
        Execution time in seconds
    **kwargs : dict
        Additional metadata
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'num_atms': num_atms,
        'method': method,
        'objective_value': objective_value,
        'execution_time': execution_time,
        **kwargs
    }
    
    metadata_file = output_dir / 'metadata.json'
    
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger = logging.getLogger('atm_optimizer')
    logger.info(f"Saved metadata to {metadata_file}")


def check_cache_validity(cache_path: Path, 
                        demand_points: pd.DataFrame,
                        candidate_locations: pd.DataFrame) -> bool:
    """
    Check if cached travel times are valid for current data.
    
    Parameters:
    -----------
    cache_path : Path
        Path to cache file
    demand_points : pd.DataFrame
        Current demand points
    candidate_locations : pd.DataFrame
        Current candidate locations
    
    Returns:
    --------
    bool
        True if cache is valid, False otherwise
    """
    logger = logging.getLogger('atm_optimizer')
    
    if not cache_path.exists():
        logger.warning("Cache file does not exist")
        return False
    
    try:
        import pickle
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if required keys exist
        required_keys = ['travel_times', 'demand_points', 'candidate_locations']
        if not all(key in cache_data for key in required_keys):
            logger.warning("Cache file missing required keys")
            return False
        
        # Check if dimensions match
        cached_demand_ids = set(cache_data['demand_points']['id'])
        cached_candidate_ids = set(cache_data['candidate_locations']['id'])
        
        current_demand_ids = set(demand_points['id'])
        current_candidate_ids = set(candidate_locations['id'])
        
        if cached_demand_ids != current_demand_ids:
            logger.warning("Cached demand points don't match current data")
            return False
        
        if cached_candidate_ids != current_candidate_ids:
            logger.warning("Cached candidate locations don't match current data")
            return False
        
        logger.info("Cache validation passed")
        return True
    
    except Exception as e:
        logger.error(f"Error validating cache: {e}")
        return False


if __name__ == "__main__":
    # Test logging setup
    logger = setup_logging(log_level="DEBUG")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
