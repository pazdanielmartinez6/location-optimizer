"""
ATM Location Optimizer - Core Optimization Engine

This module contains the main ATMLocationOptimizer class that handles:
- Travel time calculation via OSRM API
- Simulated Annealing optimization
- Greedy optimization
- Statistical algorithm comparison
- Result caching

Author: Daniel Paz Martinez
"""

import numpy as np
import pandas as pd
import random
import requests
import pickle
import time
from scipy import stats
from pyproj import Transformer
from typing import Optional, Dict, List, Tuple, Union
import logging
from pathlib import Path

from .config import get_config
from .utils import validate_dataframe, validate_coordinates, check_cache_validity, format_time

logger = logging.getLogger('atm_optimizer')


class ATMLocationOptimizer:
    """
    Main optimizer class for ATM location optimization.
    
    This class handles the complete optimization workflow:
    1. Load and validate data
    2. Calculate travel times using OSRM API
    3. Run optimization algorithms
    4. Evaluate and compare solutions
    5. Cache results for reuse
    """
    
    def __init__(self, 
                 demand_points: pd.DataFrame,
                 candidate_locations: Optional[pd.DataFrame] = None,
                 population_weights: Optional[Union[pd.Series, Dict]] = None,
                 use_cache: bool = True):
        """
        Initialize the ATM location optimizer.
        
        Parameters:
        -----------
        demand_points : pd.DataFrame
            DataFrame with columns: id, latitude, longitude
        candidate_locations : pd.DataFrame, optional
            DataFrame with columns: id, latitude, longitude
            If None, uses 100 random demand points as candidates
        population_weights : pd.Series or dict, optional
            Weights for demand points (default: uniform weights of 1)
        use_cache : bool
            Whether to use cached travel times if available
        """
        self.config = get_config()
        logger.info("Initializing ATM Location Optimizer")
        
        # Validate demand points
        if not validate_dataframe(demand_points, ['id', 'latitude', 'longitude'], 'Demand Points'):
            raise ValueError("Invalid demand points data")
        
        self.demand_points = demand_points.copy()
        logger.info(f"Loaded {len(self.demand_points)} demand points")
        
        # Convert coordinates if necessary
        if self.config.is_projected and self.config.source_crs:
            logger.info(f"Converting coordinates from {self.config.source_crs} to {self.config.target_crs}")
            self._convert_demand_points_coordinates()
        
        # Setup candidate locations
        self._setup_candidate_locations(candidate_locations)
        
        # Setup population weights
        self._setup_population_weights(population_weights)
        
        # Calculate or load travel times
        cache_path = self.config.paths.travel_times_cache
        if use_cache and cache_path.exists():
            logger.info(f"Attempting to load travel times from cache: {cache_path}")
            if check_cache_validity(cache_path, self.demand_points, self.candidate_locations):
                self._load_travel_times_from_cache(cache_path)
            else:
                logger.warning("Cache invalid, recalculating travel times")
                self.travel_times = self._calculate_osrm_travel_times()
        else:
            logger.info("Calculating travel times using OSRM routing service")
            self.travel_times = self._calculate_osrm_travel_times()
        
        # Prepare distance matrix for optimization
        self._prepare_distance_matrix()
        logger.info("Optimizer initialization complete")
    
    def _convert_demand_points_coordinates(self):
        """Convert demand points coordinates from source CRS to WGS84 (EPSG:4326)."""
        try:
            transformer = Transformer.from_crs(
                self.config.source_crs, 
                self.config.target_crs, 
                always_xy=True
            )
            
            converted_count = 0
            for idx, row in self.demand_points.iterrows():
                lon, lat = transformer.transform(row['longitude'], row['latitude'])
                self.demand_points.at[idx, 'longitude'] = lon
                self.demand_points.at[idx, 'latitude'] = lat
                converted_count += 1
            
            logger.info(f"Converted {converted_count} coordinate pairs")
            logger.debug(f"Sample converted coordinates: {self.demand_points[['id', 'latitude', 'longitude']].head(3).to_dict('records')}")
        
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            raise
    
    def _setup_candidate_locations(self, candidate_locations: Optional[pd.DataFrame]):
        """
        Setup candidate ATM locations.
        
        Strategy: 21 from CSV + 79 random from demand points = 100 total candidates
        """
        seed = self.config.optimization.random_seed
        
        if candidate_locations is not None and len(candidate_locations) >= 21:
            logger.info("Using provided candidate locations")
            
            # Validate candidate locations
            if not validate_dataframe(candidate_locations, ['id', 'latitude', 'longitude'], 'Candidate Locations'):
                raise ValueError("Invalid candidate locations data")
            
            # Take first 21 candidates from CSV (assume already in WGS84)
            csv_candidates = candidate_locations.head(21).copy()
            logger.info(f"Using {len(csv_candidates)} candidates from CSV")
            
            # Get 79 random demand points (already converted to WGS84 if needed)
            available_demand = self.demand_points[
                ~self.demand_points['id'].isin(csv_candidates['id'])
            ].copy()
            
            num_random = min(79, len(available_demand))
            if num_random < 79:
                logger.warning(f"Only {num_random} demand points available for random candidates")
            
            # Sample random candidates
            np.random.seed(seed)
            random_demand = available_demand.sample(n=num_random, random_state=seed).copy()
            
            # Update IDs to continue from 22
            random_demand['id'] = list(range(22, 22 + len(random_demand)))
            logger.info(f"Sampled {len(random_demand)} random candidates from demand points")
            
            # Combine candidates
            self.candidate_locations = pd.concat([csv_candidates, random_demand], ignore_index=True)
            logger.info(f"Total candidates: {len(self.candidate_locations)} ({len(csv_candidates)} from CSV + {len(random_demand)} random)")
        
        else:
            # Fallback: use 100 random demand points
            logger.warning("No candidate locations provided or insufficient candidates")
            logger.info("Using 100 random demand points as candidates")
            
            num_candidates = min(100, len(self.demand_points))
            np.random.seed(seed)
            self.candidate_locations = self.demand_points.sample(n=num_candidates, random_state=seed).copy()
            
            # Update IDs to be sequential starting from 1
            self.candidate_locations['id'] = list(range(1, len(self.candidate_locations) + 1))
            logger.info(f"Created {len(self.candidate_locations)} candidate locations")
    
    def _setup_population_weights(self, population_weights: Optional[Union[pd.Series, Dict]]):
        """Setup population weights for demand points."""
        if population_weights is None:
            self.population_weights = {id: 1 for id in self.demand_points['id']}
            logger.info("Using uniform weights (all = 1) for demand points")
        elif isinstance(population_weights, pd.Series):
            self.population_weights = population_weights.to_dict()
            logger.info(f"Loaded {len(self.population_weights)} population weights from Series")
        else:
            self.population_weights = population_weights
            logger.info(f"Loaded {len(self.population_weights)} population weights from dict")
        
        # Log weight statistics
        weights = list(self.population_weights.values())
        logger.info(f"Weight statistics: min={min(weights):.2f}, max={max(weights):.2f}, mean={np.mean(weights):.2f}")
    
    def _load_travel_times_from_cache(self, cache_path: Path):
        """Load travel times from cache file."""
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.travel_times = cache_data['travel_times']
            logger.info(f"Loaded {len(self.travel_times)} travel time entries from cache")
            
            # Update demand points and candidate locations from cache if they match
            if 'population_weights' in cache_data:
                self.population_weights = cache_data['population_weights']
                logger.info("Loaded population weights from cache")
        
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            raise
    
    def _calculate_osrm_travel_times(self) -> Dict[Tuple[int, int], float]:
        """
        Calculate travel times using OSRM routing service.
        
        Returns:
        --------
        Dict[Tuple[int, int], float]
            Dictionary mapping (demand_id, candidate_id) to travel time in seconds
        """
        base_url = self.config.osrm.base_url
        travel_times = {}
        
        # Get configuration parameters
        effective_batch_size = min(
            self.config.osrm.batch_size, 
            self.config.osrm.effective_batch_size
        )
        
        # Prepare ID lists
        demand_ids = list(self.demand_points['id'])
        candidate_ids = list(self.candidate_locations['id'])
        
        # Create coordinate mappings
        demand_coords = {
            row['id']: (row['longitude'], row['latitude'])
            for _, row in self.demand_points.iterrows()
        }
        candidate_coords = {
            row['id']: (row['longitude'], row['latitude'])
            for _, row in self.candidate_locations.iterrows()
        }
        coords = {**demand_coords, **candidate_coords}
        
        # Validate coordinates
        def validate_coordinate(coord: Tuple[float, float]) -> Optional[str]:
            """Validate and format a coordinate pair."""
            lon, lat = coord
            
            if not validate_coordinates(lat, lon):
                return None
            
            # Format with limited precision (6 decimal places)
            return f"{lon:.6f},{lat:.6f}"
        
        # Log sample coordinates for debugging
        logger.debug("Sample coordinates for OSRM API:")
        for coord_id in list(coords.keys())[:3]:
            validated = validate_coordinate(coords[coord_id])
            logger.debug(f"  ID {coord_id}: {coords[coord_id]} -> {validated}")
        
        # Process in batches
        total_batches = (
            (len(demand_ids) // effective_batch_size + 1) * 
            (len(candidate_ids) // effective_batch_size + 1)
        )
        batch_count = 0
        
        logger.info(f"Processing {total_batches} batches (batch size: {effective_batch_size})")
        
        for i in range(0, len(demand_ids), effective_batch_size):
            batch_demand_ids = demand_ids[i:i+effective_batch_size]
            
            for j in range(0, len(candidate_ids), effective_batch_size):
                batch_candidate_ids = candidate_ids[j:j+effective_batch_size]
                batch_count += 1
                
                if batch_count % 10 == 0:
                    logger.info(f"Processing batch {batch_count}/{total_batches}...")
                
                # Process this batch with retry logic
                success = self._process_osrm_batch(
                    batch_demand_ids, 
                    batch_candidate_ids,
                    coords,
                    validate_coordinate,
                    travel_times,
                    base_url
                )
                
                if not success:
                    logger.warning(f"Batch {batch_count} failed after retries")
                
                # Delay between requests
                time.sleep(self.config.osrm.request_delay)
        
        logger.info(f"Travel time calculation complete: {len(travel_times)} pairs processed")
        
        # Fill missing travel times with penalty values
        missing_count = 0
        for demand_id in demand_ids:
            for candidate_id in candidate_ids:
                if (demand_id, candidate_id) not in travel_times:
                    travel_times[(demand_id, candidate_id)] = 999999
                    missing_count += 1
        
        if missing_count > 0:
            logger.warning(f"{missing_count} travel times missing, set to penalty value (999999)")
        
        return travel_times
    
    def _process_osrm_batch(self, 
                           batch_demand_ids: List[int],
                           batch_candidate_ids: List[int],
                           coords: Dict[int, Tuple[float, float]],
                           validate_coordinate,
                           travel_times: Dict,
                           base_url: str) -> bool:
        """Process a single batch of OSRM requests with retry logic."""
        
        for attempt in range(self.config.osrm.max_retries):
            try:
                # Validate and create coordinate strings for origins
                origin_coords = []
                valid_demand_ids = []
                
                for d_id in batch_demand_ids:
                    if d_id not in coords:
                        logger.warning(f"Demand ID {d_id} not found in coordinates")
                        continue
                    
                    validated_coord = validate_coordinate(coords[d_id])
                    if validated_coord:
                        origin_coords.append(validated_coord)
                        valid_demand_ids.append(d_id)
                    else:
                        logger.warning(f"Invalid coordinates for demand ID {d_id}: {coords[d_id]}")
                
                # Validate and create coordinate strings for destinations
                dest_coords = []
                valid_candidate_ids = []
                
                for c_id in batch_candidate_ids:
                    if c_id not in coords:
                        logger.warning(f"Candidate ID {c_id} not found in coordinates")
                        continue
                    
                    validated_coord = validate_coordinate(coords[c_id])
                    if validated_coord:
                        dest_coords.append(validated_coord)
                        valid_candidate_ids.append(c_id)
                    else:
                        logger.warning(f"Invalid coordinates for candidate ID {c_id}: {coords[c_id]}")
                
                # Skip if no valid coordinates
                if not origin_coords or not dest_coords:
                    logger.warning("No valid coordinates in batch, skipping")
                    return False
                
                # Limit total coordinates to avoid URL length issues
                max_total_coords = self.config.osrm.max_total_coords
                if len(origin_coords) + len(dest_coords) > max_total_coords:
                    max_origins = max_total_coords // 2
                    max_dests = max_total_coords - max_origins
                    
                    origin_coords = origin_coords[:max_origins]
                    valid_demand_ids = valid_demand_ids[:max_origins]
                    dest_coords = dest_coords[:max_dests]
                    valid_candidate_ids = valid_candidate_ids[:max_dests]
                    
                    logger.debug(f"Truncated batch to {len(origin_coords)} origins, {len(dest_coords)} destinations")
                
                # Build OSRM request URL
                all_coords_str = ';'.join(origin_coords + dest_coords)
                sources_str = ';'.join([str(i) for i in range(len(origin_coords))])
                destinations_str = ';'.join([str(len(origin_coords) + i) for i in range(len(dest_coords))])
                
                url = f"{base_url}{all_coords_str}?sources={sources_str}&destinations={destinations_str}"
                
                # Check URL length
                if len(url) > self.config.osrm.max_url_length:
                    logger.warning(f"URL too long ({len(url)} chars), skipping batch")
                    return False
                
                # Make request
                response = requests.get(url, timeout=self.config.osrm.request_timeout)
                
                if response.status_code != 200:
                    logger.error(f"HTTP Error {response.status_code}: {response.text}")
                    if attempt < self.config.osrm.max_retries - 1:
                        time.sleep(self.config.osrm.retry_delay)
                        continue
                    return False
                
                data = response.json()
                
                # Process results
                if data.get('code') == 'Ok':
                    durations = data.get('durations', [])
                    
                    if len(durations) != len(valid_demand_ids):
                        logger.warning(f"Duration matrix size mismatch")
                        if attempt < self.config.osrm.max_retries - 1:
                            continue
                        return False
                    
                    # Store travel times
                    for src_idx, src_id in enumerate(valid_demand_ids):
                        if src_idx >= len(durations):
                            continue
                        
                        duration_row = durations[src_idx]
                        if len(duration_row) != len(valid_candidate_ids):
                            logger.warning(f"Duration row size mismatch for demand {src_id}")
                            continue
                        
                        for dst_idx, dst_id in enumerate(valid_candidate_ids):
                            if dst_idx < len(duration_row):
                                travel_time = duration_row[dst_idx]
                                if travel_time is not None and not np.isnan(travel_time):
                                    travel_times[(src_id, dst_id)] = travel_time
                                else:
                                    travel_times[(src_id, dst_id)] = 999999
                    
                    return True
                
                else:
                    logger.error(f"OSRM API error: {data.get('message', 'Unknown error')}")
                    if attempt < self.config.osrm.max_retries - 1:
                        time.sleep(self.config.osrm.retry_delay)
                        continue
                    return False
            
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.config.osrm.max_retries})")
                if attempt < self.config.osrm.max_retries - 1:
                    time.sleep(self.config.osrm.retry_delay)
                    continue
                return False
            
            except Exception as e:
                logger.error(f"Exception during OSRM request: {e}")
                if attempt < self.config.osrm.max_retries - 1:
                    time.sleep(self.config.osrm.retry_delay)
                    continue
                return False
        
        return False
    
    def _prepare_distance_matrix(self):
        """Create a distance matrix from travel times for optimization."""
        demand_ids = list(self.demand_points['id'])
        candidate_ids = list(self.candidate_locations['id'])
        
        self.distances = np.zeros((len(demand_ids), len(candidate_ids)))
        for i, demand_id in enumerate(demand_ids):
            for j, candidate_id in enumerate(candidate_ids):
                self.distances[i, j] = self.travel_times.get((demand_id, candidate_id), float('inf'))
        
        # Create indices for lookup
        self.demand_indices = {id: i for i, id in enumerate(demand_ids)}
        self.candidate_indices = {id: i for i, id in enumerate(candidate_ids)}
        
        logger.info(f"Distance matrix prepared: {self.distances.shape}")
    
    def evaluate_solution(self, solution: List[int]) -> float:
        """
        Evaluate the objective function for a given solution.
        
        The objective is to minimize total weighted travel time from all demand
        points to their nearest ATM.
        
        Parameters:
        -----------
        solution : list
            List of candidate IDs representing ATM locations
        
        Returns:
        --------
        float
            Objective function value (lower is better)
        """
        # Get indices for the solution
        solution_indices = [self.candidate_indices[id] for id in solution]
        
        # Calculate total weighted distance
        total_weighted_distance = 0
        for demand_id, weight in self.population_weights.items():
            demand_idx = self.demand_indices[demand_id]
            min_distance = min(self.distances[demand_idx, j] for j in solution_indices)
            total_weighted_distance += weight * min_distance
        
        return total_weighted_distance
    
    def optimize(self, 
                 num_atms: int, 
                 method: str = 'simulated_annealing',
                 **kwargs) -> List[int]:
        """
        Optimize ATM locations using specified method.
        
        Parameters:
        -----------
        num_atms : int
            Number of ATMs to place
        method : str
            Optimization method: 'simulated_annealing' or 'greedy'
        **kwargs : dict
            Additional parameters for the optimization method
        
        Returns:
        --------
        list
            List of candidate IDs representing optimal ATM locations
        """
        logger.info(f"Starting optimization: {num_atms} ATMs using {method}")
        start_time = time.time()
        
        if method == 'simulated_annealing':
            solution = self._optimize_simulated_annealing(num_atms, **kwargs)
        elif method == 'greedy':
            solution = self._optimize_greedy(num_atms)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        elapsed = time.time() - start_time
        objective = self.evaluate_solution(solution)
        
        logger.info(f"Optimization complete in {format_time(elapsed)}")
        logger.info(f"Solution: {solution}")
        logger.info(f"Objective value: {objective:.2f}")
        
        return solution
    
    def _optimize_simulated_annealing(self, 
                                      num_atms: int,
                                      initial_temp: Optional[float] = None,
                                      cooling_rate: Optional[float] = None,
                                      iterations: Optional[int] = None,
                                      reheat_threshold: Optional[int] = None) -> List[int]:
        """
        Optimize using Simulated Annealing with adaptive reheating.
        
        Parameters:
        -----------
        num_atms : int
            Number of ATMs to place
        initial_temp : float, optional
            Initial temperature (uses config default if None)
        cooling_rate : float, optional
            Cooling rate (uses config default if None)
        iterations : int, optional
            Number of iterations (uses config default if None)
        reheat_threshold : int, optional
            Iterations without improvement before reheating (uses config default if None)
        
        Returns:
        --------
        list
            List of candidate IDs representing optimal ATM locations
        """
        # Use config defaults if not provided
        initial_temp = initial_temp or self.config.optimization.sa_initial_temp
        cooling_rate = cooling_rate or self.config.optimization.sa_cooling_rate
        iterations = iterations or self.config.optimization.sa_iterations
        reheat_threshold = reheat_threshold or self.config.optimization.sa_reheat_threshold
        
        logger.info(f"SA Parameters: T0={initial_temp}, alpha={cooling_rate}, iter={iterations}, reheat={reheat_threshold}")

        
        candidate_ids = list(self.candidate_locations['id'])
        
        # Generate initial random solution
        random.seed(self.config.optimization.random_seed)
        current_solution = random.sample(candidate_ids, num_atms)
        current_value = self.evaluate_solution(current_solution)
        best_solution = current_solution.copy()
        best_value = current_value
        
        temp = initial_temp
        no_improvement_count = 0
        beta = self.config.optimization.sa_beta
        
        logger.info(f"Initial solution: objective = {current_value:.2f}")
        
        for i in range(iterations):
            # Generate neighbor solution by replacing one ATM
            new_solution = current_solution.copy()
            idx_to_replace = random.randrange(num_atms)
            new_atm = random.choice([c for c in candidate_ids if c not in current_solution])
            new_solution[idx_to_replace] = new_atm
            
            # Evaluate new solution
            new_value = self.evaluate_solution(new_solution)
            
            # Decide whether to accept the new solution
            delta = new_value - current_value
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_solution = new_solution
                current_value = new_value
                
                # Update best solution if improved
                if current_value < best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value
                    no_improvement_count = 0
                    logger.debug(f"Iter {i}: New best = {best_value:.2f}")
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # Reheat if stuck
            if no_improvement_count >= reheat_threshold:
                beta += 0.5
                temp = temp * (3 ** beta)
                no_improvement_count = 0
                logger.debug(f"Iter {i}: Reheating to T={temp:.2f}")
            
            # Cool down
            temp *= cooling_rate
            
            # Log progress every 10%
            if (i + 1) % (iterations // 10) == 0:
                logger.info(f"Progress: {(i+1)/iterations*100:.0f}% - Best: {best_value:.2f}, Current: {current_value:.2f}, T: {temp:.2f}")
        
        logger.info(f"Final best solution: {best_value:.2f}")
        return best_solution
    
    def _optimize_greedy(self, num_atms: int) -> List[int]:
        """
        Optimize using a greedy approach.
        
        Iteratively adds the ATM that provides the maximum improvement
        to the objective function.
        
        Parameters:
        -----------
        num_atms : int
            Number of ATMs to place
        
        Returns:
        --------
        list
            List of candidate IDs representing ATM locations
        """
        logger.info("Starting greedy optimization")
        candidate_ids = list(self.candidate_locations['id'])
        solution = []
        
        for iteration in range(num_atms):
            best_value = float('inf')
            best_candidate = None
            
            # Try each candidate not yet in solution
            for candidate in candidate_ids:
                if candidate not in solution:
                    temp_solution = solution + [candidate]
                    value = self.evaluate_solution(temp_solution)
                    
                    if value < best_value:
                        best_value = value
                        best_candidate = candidate
            
            solution.append(best_candidate)
            logger.info(f"Greedy iter {iteration + 1}/{num_atms}: Added ATM {best_candidate}, objective = {best_value:.2f}")
        
        return solution
    
    def save_travel_times(self, filename: Optional[str] = None):
        """
        Save calculated travel times to a file for reuse.
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save cache (uses config default if None)
        """
        if filename is None:
            cache_path = self.config.paths.travel_times_cache
        else:
            cache_path = Path(filename)
        
        cache_data = {
            'travel_times': self.travel_times,
            'demand_points': self.demand_points,
            'candidate_locations': self.candidate_locations,
            'population_weights': self.population_weights
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Travel times cached to {cache_path}")
        
        # Also save as CSV for inspection
        csv_path = cache_path.with_suffix('.csv')
        travel_times_df = pd.DataFrame([
            {'demand_id': k[0], 'candidate_id': k[1], 'travel_time': v}
            for k, v in self.travel_times.items()
        ])
        travel_times_df.to_csv(csv_path, index=False)
        logger.info(f"Travel times also saved as CSV: {csv_path}")
    
    @classmethod
    def load_from_cache(cls, filename: Optional[str] = None) -> 'ATMLocationOptimizer':
        """
        Load optimizer with pre-calculated travel times from cache.
        
        Parameters:
        -----------
        filename : str, optional
            Filename of cache (uses config default if None)
        
        Returns:
        --------
        ATMLocationOptimizer
            Initialized optimizer with cached travel times
        """
        config = get_config()
        
        if filename is None:
            cache_path = config.paths.travel_times_cache
        else:
            cache_path = Path(filename)
        
        logger.info(f"Loading optimizer from cache: {cache_path}")
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Create instance without calling API
        instance = cls.__new__(cls)
        instance.config = config
        instance.demand_points = cache_data['demand_points']
        instance.candidate_locations = cache_data['candidate_locations']
        instance.population_weights = cache_data['population_weights']
        instance.travel_times = cache_data['travel_times']
        instance._prepare_distance_matrix()
        
        logger.info(f"Loaded optimizer with {len(instance.travel_times)} travel time entries")
        return instance
    
    def run_algorithm_comparison(self,
                                 num_atms: int,
                                 num_iterations: int = 30,
                                 sa_params: Optional[Dict] = None,
                                 save_results: bool = True,
                                 output_dir: Optional[Path] = None) -> Dict:
        """
        Compare Simulated Annealing vs Greedy algorithms with statistical analysis.
        
        Parameters:
        -----------
        num_atms : int
            Number of ATMs to place
        num_iterations : int
            Number of iterations for each algorithm (minimum 30 recommended)
        sa_params : dict, optional
            Parameters for simulated annealing
        save_results : bool
            Whether to save detailed results to CSV
        output_dir : Path, optional
            Directory to save results (uses timestamped dir if None)
        
        Returns:
        --------
        dict
            Statistical comparison results including:
            - Descriptive statistics for both algorithms
            - Statistical test results
            - Effect size
            - Performance comparison
        """
        from .utils import create_output_directory, save_results_metadata
        
        if num_iterations < self.config.optimization.stat_min_iterations:
            logger.warning(f"Recommended minimum iterations: {self.config.optimization.stat_min_iterations}")
        
        if sa_params is None:
            sa_params = {
                'initial_temp': self.config.optimization.sa_initial_temp,
                'cooling_rate': self.config.optimization.sa_cooling_rate,
                'iterations': 5000,  # Reduced for comparison speed
                'reheat_threshold': self.config.optimization.sa_reheat_threshold
            }
        
        logger.info(f"Starting algorithm comparison: {num_iterations} iterations each")
        
        # Storage for results
        sa_results = []
        greedy_results = []
        sa_times = []
        greedy_times = []
        sa_solutions = []
        greedy_solutions = []
        
        # Run Simulated Annealing iterations
        logger.info("Running Simulated Annealing iterations...")
        for i in range(num_iterations):
            start_time = time.time()
            solution = self._optimize_simulated_annealing(num_atms, **sa_params)
            end_time = time.time()
            
            objective_value = self.evaluate_solution(solution)
            sa_results.append(objective_value)
            sa_times.append(end_time - start_time)
            sa_solutions.append(solution)
            
            if (i + 1) % 5 == 0:
                logger.info(f"  SA iteration {i + 1}/{num_iterations} complete")
        
        # Run Greedy iterations
        logger.info("Running Greedy iterations...")
        for i in range(num_iterations):
            start_time = time.time()
            solution = self._optimize_greedy(num_atms)
            end_time = time.time()
            
            objective_value = self.evaluate_solution(solution)
            greedy_results.append(objective_value)
            greedy_times.append(end_time - start_time)
            greedy_solutions.append(solution)
            
            if (i + 1) % 5 == 0:
                logger.info(f"  Greedy iteration {i + 1}/{num_iterations} complete")
        
        # Perform statistical analysis
        results = self._perform_statistical_analysis(
            sa_results, greedy_results, sa_times, greedy_times
        )
        
        # Save detailed results if requested
        if save_results:
            if output_dir is None:
                output_dir = create_output_directory(num_atms, "comparison")
            
            self._save_comparison_results(
                sa_results, greedy_results, sa_times, greedy_times,
                sa_solutions, greedy_solutions, num_atms, output_dir
            )
            
            # Save metadata
            save_results_metadata(
                output_dir,
                num_atms=num_atms,
                method="comparison",
                objective_value=results['sa_stats']['mean'],
                execution_time=np.mean(sa_times),
                num_iterations=num_iterations,
                sa_params=sa_params
            )
        
        return results
    
    def _perform_statistical_analysis(self,
                                      sa_results: List[float],
                                      greedy_results: List[float],
                                      sa_times: List[float],
                                      greedy_times: List[float]) -> Dict:
        """Perform comprehensive statistical analysis of algorithm comparison."""
        
        logger.info("\n" + "="*60)
        logger.info("STATISTICAL ANALYSIS RESULTS")
        logger.info("="*60)
        
        # Convert to numpy arrays
        sa_obj = np.array(sa_results)
        greedy_obj = np.array(greedy_results)
        sa_time = np.array(sa_times)
        greedy_time = np.array(greedy_times)
        
        # Check if greedy is deterministic
        greedy_is_deterministic = len(np.unique(greedy_obj)) == 1
        
        if greedy_is_deterministic:
            logger.warning("Greedy algorithm produces identical results (deterministic)")
        
        # Descriptive statistics
        logger.info("\nDescriptive Statistics:")
        sa_stats = {
            'mean': np.mean(sa_obj),
            'std': np.std(sa_obj, ddof=1) if len(sa_obj) > 1 else 0,
            'min': np.min(sa_obj),
            'max': np.max(sa_obj),
            'median': np.median(sa_obj),
        }
        
        greedy_stats = {
            'mean': np.mean(greedy_obj),
            'std': np.std(greedy_obj, ddof=1) if len(greedy_obj) > 1 and not greedy_is_deterministic else 0,
            'min': np.min(greedy_obj),
            'max': np.max(greedy_obj),
            'median': np.median(greedy_obj),
        }
        
        logger.info(f"  Simulated Annealing: mean={sa_stats['mean']:.2f}, std={sa_stats['std']:.2f}")
        logger.info(f"  Greedy Algorithm: mean={greedy_stats['mean']:.2f}, std={greedy_stats['std']:.2f}")
        
        # Statistical tests
        logger.info("\nStatistical Tests:")
        
        if greedy_is_deterministic:
            # One-sample t-test
            if sa_stats['std'] > 0:
                t_stat, p_value = stats.ttest_1samp(sa_obj, greedy_stats['mean'])
                test_name = "One-sample t-test"
                logger.info(f"  {test_name}: t={t_stat:.4f}, p={p_value:.6f}")
                
                if sa_stats['std'] > 0:
                    cohens_d = (sa_stats['mean'] - greedy_stats['mean']) / sa_stats['std']
                else:
                    cohens_d = 0
            else:
                t_stat, p_value = 0, 1.0
                cohens_d = 0
                test_name = "N/A (no variation)"
        else:
            # Normality tests
            sa_normality = stats.shapiro(sa_obj) if len(sa_obj) <= 5000 else (None, 0.05)
            greedy_normality = stats.shapiro(greedy_obj) if len(greedy_obj) <= 5000 else (None, 0.05)
            
            both_normal = (sa_normality[1] > 0.05 if sa_normality[0] else True) and \
                         (greedy_normality[1] > 0.05 if greedy_normality[0] else True)
            
            if both_normal and sa_stats['std'] > 0 and greedy_stats['std'] > 0:
                t_stat, p_value = stats.ttest_ind(sa_obj, greedy_obj, equal_var=False)
                test_name = "Welch's t-test"
            else:
                u_stat, p_value = stats.mannwhitneyu(sa_obj, greedy_obj, alternative='two-sided')
                t_stat = u_stat
                test_name = "Mann-Whitney U test"
            
            logger.info(f"  {test_name}: statistic={t_stat:.4f}, p={p_value:.6f}")
            
            # Effect size
            if sa_stats['std'] > 0 or greedy_stats['std'] > 0:
                pooled_std = np.sqrt(((len(sa_obj) - 1) * sa_stats['std']**2 +
                                     (len(greedy_obj) - 1) * greedy_stats['std']**2) /
                                    (len(sa_obj) + len(greedy_obj) - 2))
                cohens_d = (sa_stats['mean'] - greedy_stats['mean']) / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = 0
        
        logger.info(f"  Cohen's d (effect size): {cohens_d:.4f}")
        
        # Performance improvement
        if greedy_stats['mean'] > 0:
            improvement = ((greedy_stats['mean'] - sa_stats['mean']) / greedy_stats['mean']) * 100
            logger.info(f"\nPerformance Comparison:")
            logger.info(f"  SA improvement over Greedy: {improvement:.2f}%")
        else:
            improvement = 0
        
        # Time analysis
        logger.info(f"\nTime Analysis:")
        logger.info(f"  SA avg time: {np.mean(sa_time):.4f}s")
        logger.info(f"  Greedy avg time: {np.mean(greedy_time):.4f}s")
        if np.mean(greedy_time) > 0:
            time_ratio = np.mean(sa_time) / np.mean(greedy_time)
            logger.info(f"  SA is {time_ratio:.1f}x slower than Greedy")
        
        logger.info("="*60 + "\n")
        
        return {
            'sa_stats': sa_stats,
            'greedy_stats': greedy_stats,
            'test_statistic': t_stat if 't_stat' in locals() else 0,
            'p_value': p_value if 'p_value' in locals() else 1.0,
            'cohens_d': cohens_d,
            'improvement_percent': improvement,
            'test_name': test_name if 'test_name' in locals() else "N/A",
            'significant': p_value < 0.05 if 'p_value' in locals() else False,
            'sa_time_avg': np.mean(sa_time),
            'greedy_time_avg': np.mean(greedy_time),
            'greedy_is_deterministic': greedy_is_deterministic,
        }
    
    def _save_comparison_results(self,
                                sa_results: List[float],
                                greedy_results: List[float],
                                sa_times: List[float],
                                greedy_times: List[float],
                                sa_solutions: List[List[int]],
                                greedy_solutions: List[List[int]],
                                num_atms: int,
                                output_dir: Path):
        """Save detailed comparison results to CSV files."""
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save objective values comparison
        results_df = pd.DataFrame({
            'iteration': range(1, len(sa_results) + 1),
            'sa_objective': sa_results,
            'greedy_objective': greedy_results,
            'sa_time': sa_times,
            'greedy_time': greedy_times
        })
        results_path = output_dir / f'algorithm_comparison_{num_atms}atms_{timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved comparison results: {results_path}")
        
        # Save solutions
        solutions_data = []
        for i, (sa_sol, greedy_sol) in enumerate(zip(sa_solutions, greedy_solutions)):
            for atm_id in sa_sol:
                solutions_data.append({
                    'iteration': i + 1,
                    'algorithm': 'SA',
                    'atm_id': atm_id,
                    'num_atms': num_atms
                })
            for atm_id in greedy_sol:
                solutions_data.append({
                    'iteration': i + 1,
                    'algorithm': 'Greedy',
                    'atm_id': atm_id,
                    'num_atms': num_atms
                })
        
        solutions_df = pd.DataFrame(solutions_data)
        solutions_path = output_dir / f'algorithm_solutions_{num_atms}atms_{timestamp}.csv'
        solutions_df.to_csv(solutions_path, index=False)
        logger.info(f"Saved solutions: {solutions_path}")

