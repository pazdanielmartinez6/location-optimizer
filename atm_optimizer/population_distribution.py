"""
Population Distribution Module

Generates demand points from OpenStreetMap building footprints and population data.
This is an OPTIONAL module for users who want to create their own demand points
from geographical data.

Author: Daniel Paz Martinez
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import random
from shapely.geometry import Point, Polygon, box
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
import logging

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False
    logging.warning("osmnx not available - population distribution features disabled")

from .config import get_config

logger = logging.getLogger('atm_optimizer')


class PopulationDistributor:
    """
    Generate demand points from building footprints and population data.
    
    This class extracts building footprints from OpenStreetMap and distributes
    population across them to create realistic demand points.
    """
    
    def __init__(self, barrios_gdf: gpd.GeoDataFrame, population_data: List[Dict]):
        """
        Initialize the population distributor.
        
        Parameters:
        -----------
        barrios_gdf : GeoDataFrame
            GeoDataFrame containing neighborhood boundaries
            Required columns: 'code', 'name', 'geometry'
        population_data : list of dict
            Population data for each neighborhood
            Format: [{'code': '001', 'name': 'District 1', 'population': 50000}, ...]
        """
        if not OSMNX_AVAILABLE:
            raise ImportError(
                "osmnx is required for population distribution. "
                "Install it with: pip install osmnx"
            )
        
        self.config = get_config()
        self.barrios_gdf = barrios_gdf.copy()
        self.population_df = pd.DataFrame(population_data)
        
        logger.info(f"Initialized PopulationDistributor with {len(barrios_gdf)} neighborhoods")
        logger.info(f"Total population: {self.population_df['population'].sum():,}")
    
    def get_building_footprints(self) -> gpd.GeoDataFrame:
        """
        Get building footprints from OpenStreetMap for all neighborhoods.
        
        Returns:
        --------
        GeoDataFrame
            Building footprints with classification (residential, daytime, unclassified)
        """
        logger.info("Fetching building footprints from OpenStreetMap...")
        
        # Convert to WGS84 for OSM queries
        barrios_wgs84 = self.barrios_gdf.to_crs("EPSG:4326")
        
        all_buildings = gpd.GeoDataFrame()
        
        # Building type filters from config
        residential_filters = self.config.population.residential_filters
        daytime_filters = self.config.population.daytime_filters
        
        for idx, row in tqdm(barrios_wgs84.iterrows(), total=len(barrios_wgs84), 
                            desc="Fetching buildings"):
            try:
                # Get the bounds of the neighborhood
                bounds_tuple = tuple(row.geometry.bounds)
                
                # Get buildings in the neighborhood
                tags = {'building': True}
                buildings = ox.features_from_bbox(bounds_tuple, tags)
                
                # Filter to only keep buildings that intersect with the neighborhood
                buildings = gpd.clip(buildings, row.geometry)
                
                # Add neighborhood info
                buildings['barrio_code'] = row['code']
                buildings['barrio_name'] = row['name']
                
                # Classify buildings
                buildings['building_category'] = 'unclassified'
                
                # Mark residential buildings
                residential_mask = buildings['building'].isin(residential_filters)
                buildings.loc[residential_mask, 'building_category'] = 'residential'
                
                # Mark daytime buildings
                daytime_mask = buildings['building'].isin(daytime_filters)
                buildings.loc[daytime_mask, 'building_category'] = 'daytime'
                
                # If no specific category matched, assume residential
                generic_mask = (buildings['building_category'] == 'unclassified') & buildings['building'].notna()
                buildings.loc[generic_mask, 'building_category'] = 'residential'
                
                if len(buildings) > 0:
                    all_buildings = pd.concat([all_buildings, buildings])
                    
            except Exception as e:
                logger.warning(f"Error processing {row['name']}: {e}")
                continue
        
        if len(all_buildings) > 0:
            # Convert back to original CRS
            all_buildings = all_buildings.to_crs(self.barrios_gdf.crs)
            
            # Calculate building area
            all_buildings['area'] = all_buildings.geometry.area
            
            logger.info(f"Retrieved {len(all_buildings)} buildings")
            logger.info(f"  Residential: {len(all_buildings[all_buildings['building_category']=='residential'])}")
            logger.info(f"  Daytime: {len(all_buildings[all_buildings['building_category']=='daytime'])}")
        
        return all_buildings
    
    def distribute_residential_population(self,
                                         buildings_gdf: gpd.GeoDataFrame,
                                         percentage: Optional[float] = None,
                                         min_people_per_building: int = None,
                                         el_pardo_box: Optional[Tuple] = None) -> gpd.GeoDataFrame:
        """
        Distribute residential population across buildings.
        
        Parameters:
        -----------
        buildings_gdf : GeoDataFrame
            Buildings with 'building_category' column
        percentage : float, optional
            Percentage of population to distribute (uses config default if None)
        min_people_per_building : int, optional
            Minimum people per building (uses config default if None)
        el_pardo_box : tuple, optional
            Bounding box to filter El Pardo area: (min_x, min_y, max_x, max_y)
        
        Returns:
        --------
        GeoDataFrame
            Population points (residential)
        """
        if percentage is None:
            percentage = self.config.population.residential_percentage
        if min_people_per_building is None:
            min_people_per_building = self.config.population.min_people_per_building
        if el_pardo_box is None:
            el_pardo_box = self.config.population.el_pardo_box
        
        logger.info(f"Distributing residential population ({percentage*100}% of total)")
        
        # Filter residential buildings
        residential = buildings_gdf[buildings_gdf['building_category'] == 'residential'].copy()
        
        # Apply El Pardo filter if specified
        if el_pardo_box is not None:
            el_pardo_polygon = box(*el_pardo_box)
            el_pardo_mask = residential['barrio_code'] == '081'
            el_pardo_buildings = residential[el_pardo_mask].copy()
            
            within_box = el_pardo_buildings[el_pardo_buildings.geometry.within(el_pardo_polygon)]
            residential = residential[~el_pardo_mask]
            residential = pd.concat([residential, within_box])
            
            logger.info(f"Filtered El Pardo to {len(within_box)} buildings within specified box")
        
        # Merge with population data
        residential = residential.merge(self.population_df, left_on='barrio_code', right_on='code')
        
        # Calculate total building area per neighborhood
        barrio_total_area = residential.groupby('barrio_code')['area'].sum().reset_index()
        barrio_total_area = barrio_total_area.rename(columns={'area': 'total_area'})
        residential = residential.merge(barrio_total_area, on='barrio_code')
        
        # Apply percentage and distribute
        residential['population_adjusted'] = residential['population'] * percentage
        residential['building_population'] = (
            (residential['area'] / residential['total_area']) * 
            residential['population_adjusted']
        ).round().astype(int)
        
        # Ensure minimum
        residential.loc[residential['building_population'] < min_people_per_building, 
                       'building_population'] = min_people_per_building
        
        # Rebalance to maintain totals
        for barrio in residential['barrio_code'].unique():
            mask = residential['barrio_code'] == barrio
            current_total = residential.loc[mask, 'building_population'].sum()
            target_total = residential.loc[mask, 'population_adjusted'].iloc[0]
            
            if current_total > target_total:
                scale_factor = target_total / current_total
                residential.loc[mask, 'building_population'] = (
                    residential.loc[mask, 'building_population'] * scale_factor
                ).round().astype(int)
                
                # Ensure minimum
                below_min = residential.loc[mask, 'building_population'] < min_people_per_building
                residential.loc[mask & below_min, 'building_population'] = min_people_per_building
        
        # Generate population points
        logger.info("Generating population points...")
        population_points = self._generate_points_in_buildings(
            residential, 
            'residential'
        )
        
        logger.info(f"Generated {len(population_points)} residential population points")
        
        return population_points
    
    def distribute_daytime_population(self,
                                     buildings_gdf: gpd.GeoDataFrame,
                                     percentage: Optional[float] = None) -> gpd.GeoDataFrame:
        """
        Distribute daytime population across non-residential buildings.
        
        Parameters:
        -----------
        buildings_gdf : GeoDataFrame
            Buildings with 'building_category' column
        percentage : float, optional
            Percentage of population to distribute (uses config default if None)
        
        Returns:
        --------
        GeoDataFrame
            Population points (daytime)
        """
        if percentage is None:
            percentage = self.config.population.daytime_percentage
        
        logger.info(f"Distributing daytime population ({percentage*100}% of total)")
        
        # Filter daytime buildings
        daytime = buildings_gdf[buildings_gdf['building_category'] == 'daytime'].copy()
        
        if len(daytime) == 0:
            logger.warning("No daytime buildings found")
            return gpd.GeoDataFrame()
        
        # Merge with population data
        daytime = daytime.merge(self.population_df, left_on='barrio_code', right_on='code')
        
        # Calculate daytime population
        daytime['daytime_population'] = daytime['population'] * percentage
        
        # Calculate total daytime building area per neighborhood
        barrio_total_area = daytime.groupby('barrio_code')['area'].sum().reset_index()
        barrio_total_area = barrio_total_area.rename(columns={'area': 'total_area'})
        daytime = daytime.merge(barrio_total_area, on='barrio_code')
        
        # Distribute proportionally
        daytime['building_population'] = (
            (daytime['area'] / daytime['total_area']) * 
            daytime['daytime_population']
        ).round().astype(int)
        
        # Ensure minimum
        daytime.loc[daytime['building_population'] < 1, 'building_population'] = 1
        
        # Generate population points
        logger.info("Generating daytime population points...")
        population_points = self._generate_points_in_buildings(
            daytime,
            'daytime'
        )
        
        logger.info(f"Generated {len(population_points)} daytime population points")
        
        return population_points
    
    def _generate_points_in_buildings(self,
                                     buildings_gdf: gpd.GeoDataFrame,
                                     population_type: str) -> gpd.GeoDataFrame:
        """
        Generate random points within building footprints.
        
        Parameters:
        -----------
        buildings_gdf : GeoDataFrame
            Buildings with 'building_population' column
        population_type : str
            'residential' or 'daytime'
        
        Returns:
        --------
        GeoDataFrame
            Population points
        """
        population_points = []
        point_attributes = []
        
        for idx, building in tqdm(buildings_gdf.iterrows(), 
                                 total=len(buildings_gdf),
                                 desc=f"Generating {population_type} points"):
            num_points = building['building_population']
            
            if num_points > 0:
                building_poly = building.geometry
                
                for _ in range(num_points):
                    # Generate point within building
                    min_x, min_y, max_x, max_y = building_poly.bounds
                    
                    # Try to generate point inside (with retries)
                    attempts = 0
                    point_inside = False
                    while not point_inside and attempts < 10:
                        x = random.uniform(min_x, max_x)
                        y = random.uniform(min_y, max_y)
                        point = Point(x, y)
                        point_inside = building_poly.contains(point)
                        attempts += 1
                    
                    # If failed, use centroid
                    if not point_inside:
                        point = building_poly.centroid
                    
                    population_points.append(point)
                    point_attributes.append({
                        'barrio_code': building['barrio_code'],
                        'barrio_name': building['barrio_name'],
                        'building_id': idx,
                        'building_type': building.get('building', 'unknown'),
                        'demand_type': population_type
                    })
        
        # Create GeoDataFrame
        population_gdf = gpd.GeoDataFrame(
            point_attributes,
            geometry=population_points,
            crs=buildings_gdf.crs
        )
        
        return population_gdf
    
    def generate_demand_points(self,
                              residential_percentage: Optional[float] = None,
                              daytime_percentage: Optional[float] = None,
                              save_to_csv: bool = True,
                              output_filename: str = "demand_points.csv") -> pd.DataFrame:
        """
        Complete workflow to generate demand points.
        
        Parameters:
        -----------
        residential_percentage : float, optional
            Percentage of residential population
        daytime_percentage : float, optional
            Percentage of daytime population
        save_to_csv : bool
            Whether to save to CSV
        output_filename : str
            Output filename
        
        Returns:
        --------
        DataFrame
            Combined demand points with id, latitude, longitude, demand_type
        """
        logger.info("Starting complete demand point generation workflow")
        
        # Step 1: Get buildings
        buildings = self.get_building_footprints()
        
        if len(buildings) == 0:
            raise ValueError("No buildings found - check your neighborhood boundaries")
        
        # Step 2: Distribute residential population
        residential_points = self.distribute_residential_population(
            buildings, 
            residential_percentage
        )
        
        # Step 3: Distribute daytime population
        daytime_points = self.distribute_daytime_population(
            buildings,
            daytime_percentage
        )
        
        # Step 4: Combine and prepare output
        combined = pd.concat([residential_points, daytime_points], ignore_index=True)
        
        # Add sequential IDs
        combined['id'] = range(1, len(combined) + 1)
        
        # Extract coordinates
        combined['latitude'] = combined.geometry.y
        combined['longitude'] = combined.geometry.x
        
        # Select output columns
        output_columns = ['id', 'latitude', 'longitude', 'demand_type', 
                         'barrio_code', 'barrio_name', 'building_type']
        output_df = combined[output_columns].copy()
        
        # Save to CSV
        if save_to_csv:
            config = get_config()
            output_path = config.paths.get_output_path(output_filename)
            output_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(output_df)} demand points to {output_path}")
        
        logger.info("Demand point generation complete")
        logger.info(f"  Total points: {len(output_df)}")
        logger.info(f"  Residential: {len(output_df[output_df['demand_type']=='residential'])}")
        logger.info(f"  Daytime: {len(output_df[output_df['demand_type']=='daytime'])}")
        
        return output_df


def load_shapefile_and_generate_demand_points(
    shapefile_path: str,
    population_data: List[Dict],
    output_filename: str = "demand_points.csv",
    residential_percentage: float = 0.012,
    daytime_percentage: float = 0.01
) -> pd.DataFrame:
    """
    Convenience function to load shapefile and generate demand points in one step.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to shapefile (.shp)
    population_data : list of dict
        Population data: [{'code': '001', 'name': 'Area 1', 'population': 50000}, ...]
    output_filename : str
        Output CSV filename
    residential_percentage : float
        Percentage of residential population to distribute
    daytime_percentage : float
        Percentage of daytime population to distribute
    
    Returns:
    --------
    DataFrame
        Generated demand points
    
    Example:
    --------
    >>> population_data = [
    ...     {'code': '001', 'name': 'District 1', 'population': 50000},
    ...     {'code': '002', 'name': 'District 2', 'population': 35000}
    ... ]
    >>> demand_points = load_shapefile_and_generate_demand_points(
    ...     'data/neighborhoods.shp',
    ...     population_data
    ... )
    """
    logger.info(f"Loading shapefile from {shapefile_path}")
    
    # Load shapefile
    barrios_gdf = gpd.read_file(shapefile_path)
    
    # Initialize distributor
    distributor = PopulationDistributor(barrios_gdf, population_data)
    
    # Generate demand points
    demand_points = distributor.generate_demand_points(
        residential_percentage=residential_percentage,
        daytime_percentage=daytime_percentage,
        save_to_csv=True,
        output_filename=output_filename
    )
    
    return demand_points


if __name__ == "__main__":
    # Example usage
    print("Population Distribution Module")
    print("This module is used to generate demand points from geographical data.")
    print("\nExample usage:")
    print("""
    from atm_optimizer import PopulationDistributor
    import geopandas as gpd
    
    # Load shapefile
    barrios = gpd.read_file('data/neighborhoods.shp')
    
    # Population data
    population_data = [
        {'code': '001', 'name': 'District 1', 'population': 50000},
        {'code': '002', 'name': 'District 2', 'population': 35000}
    ]
    
    # Generate demand points
    distributor = PopulationDistributor(barrios, population_data)
    demand_points = distributor.generate_demand_points()
    """)
