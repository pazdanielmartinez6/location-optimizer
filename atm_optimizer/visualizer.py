"""
Visualizer Module for ATM Location Optimizer

Handles creation of:
- Interactive Folium maps
- Static matplotlib plots
- Algorithm comparison visualizations

Author: Daniel Paz Martinez
"""

import folium
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Optional, Tuple
from pathlib import Path
import logging

from .config import get_config

logger = logging.getLogger('atm_optimizer')


class ATMVisualizer:
    """Handles all visualization tasks for the optimizer."""
    
    def __init__(self, optimizer):
        """
        Initialize visualizer with an optimizer instance.
        
        Parameters:
        -----------
        optimizer : ATMLocationOptimizer
            The optimizer instance to visualize
        """
        self.optimizer = optimizer
        self.config = get_config()
        logger.info("Visualizer initialized")
    
    def visualize_all_candidates(self,
                                 map_center: Optional[Tuple[float, float]] = None,
                                 zoom_start: int = None) -> folium.Map:
        """
        Visualize all candidate locations on a map with demand points.
        
        Parameters:
        -----------
        map_center : tuple, optional
            (latitude, longitude) for map center
        zoom_start : int, optional
            Initial zoom level
        
        Returns:
        --------
        folium.Map
            Interactive map showing all candidates and demand points
        """
        if zoom_start is None:
            zoom_start = self.config.visualization.default_zoom
        
        # Determine map center
        if map_center is None:
            map_center = (
                self.optimizer.demand_points['latitude'].mean(),
                self.optimizer.demand_points['longitude'].mean()
            )
        
        # Create map
        m = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Add demand points (blue circles)
        for _, row in self.optimizer.demand_points.iterrows():
            weight = self.optimizer.population_weights.get(row['id'], 1)
            radius = max(5, min(20, weight / 10))
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=radius,
                color='blue',
                fill=True,
                fillColor='blue',
                fill_opacity=0.6,
                popup=f"Demand point: {row['id']}, Weight: {weight}",
                tooltip=f"Demand: {row['id']}"
            ).add_to(m)
        
        # Add candidate locations (orange markers)
        for _, row in self.optimizer.candidate_locations.iterrows():
            folium.Marker(
                location=(row['latitude'], row['longitude']),
                icon=folium.Icon(color='orange', icon='home'),
                popup=f"Candidate ATM: {row['id']}",
                tooltip=f"Candidate: {row['id']}"
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 250px; height: 90px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
            <p><b>Legend</b></p>
            <p><i class="fa fa-circle" style="color:blue"></i> Demand Points</p>
            <p><i class="fa fa-map-marker" style="color:orange"></i> Candidate ATM Locations</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        logger.info("Created map with all candidates")
        return m
    
    def visualize_solution(self,
                          solution: List[int],
                          map_center: Optional[Tuple[float, float]] = None,
                          zoom_start: int = None) -> folium.Map:
        """
        Visualize the optimized solution on a map.
        
        Parameters:
        -----------
        solution : list
            List of candidate IDs representing ATM locations
        map_center : tuple, optional
            (latitude, longitude) for map center
        zoom_start : int, optional
            Initial zoom level
        
        Returns:
        --------
        folium.Map
            Interactive map showing solution with routes
        """
        if zoom_start is None:
            zoom_start = self.config.visualization.default_zoom
        
        if map_center is None:
            map_center = (
                self.optimizer.demand_points['latitude'].mean(),
                self.optimizer.demand_points['longitude'].mean()
            )
        
        m = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Add demand points
        for _, row in self.optimizer.demand_points.iterrows():
            weight = self.optimizer.population_weights.get(row['id'], 1)
            radius = max(5, min(20, weight / 10))
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=radius,
                color='blue',
                fill=True,
                fillColor='blue',
                fill_opacity=0.6,
                popup=f"Demand: {row['id']}, Weight: {weight}",
                tooltip=f"Demand: {row['id']}"
            ).add_to(m)
        
        # Add selected ATM locations
        for atm_id in solution:
            location_data = self.optimizer.candidate_locations.loc[
                self.optimizer.candidate_locations['id'] == atm_id
            ].iloc[0]
            
            folium.Marker(
                location=(location_data['latitude'], location_data['longitude']),
                icon=folium.Icon(color='red', icon='info-sign'),
                popup=f"Selected ATM: {atm_id}",
                tooltip=f"ATM: {atm_id}"
            ).add_to(m)
        
        # Add travel routes
        for atm_id in solution:
            atm_location = self.optimizer.candidate_locations.loc[
                self.optimizer.candidate_locations['id'] == atm_id
            ].iloc[0]
            atm_coords = (atm_location['latitude'], atm_location['longitude'])
            
            for _, demand_row in self.optimizer.demand_points.iterrows():
                demand_id = demand_row['id']
                demand_coords = (demand_row['latitude'], demand_row['longitude'])
                travel_time = self.optimizer.travel_times.get((demand_id, atm_id), float('inf'))
                
                # Only draw to closest ATM
                closest_atm = min(solution,
                                key=lambda x: self.optimizer.travel_times.get((demand_id, x), float('inf')))
                if atm_id == closest_atm:
                    folium.PolyLine(
                        locations=[demand_coords, atm_coords],
                        color='green',
                        weight=2,
                        opacity=0.7,
                        popup=f"Travel: {travel_time:.1f}s",
                        tooltip=f"{travel_time:.1f}s"
                    ).add_to(m)
        
        # Legend
        legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 200px; height: 90px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
            <p><b>Legend</b></p>
            <p><i class="fa fa-circle" style="color:blue"></i> Demand Points</p>
            <p><i class="fa fa-map-marker" style="color:red"></i> Selected ATMs</p>
            <p><i class="fa fa-minus" style="color:green"></i> Travel Routes</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        logger.info(f"Created solution map with {len(solution)} ATMs")
        return m
    
    def visualize_solution_with_candidates(self,
                                          solution: List[int],
                                          map_center: Optional[Tuple[float, float]] = None,
                                          zoom_start: int = None) -> folium.Map:
        """
        Visualize solution alongside all candidate locations.
        
        Parameters:
        -----------
        solution : list
            List of candidate IDs representing selected ATM locations
        map_center : tuple, optional
            (latitude, longitude) for map center
        zoom_start : int, optional
            Initial zoom level
        
        Returns:
        --------
        folium.Map
            Interactive map showing solution, candidates, and demand
        """
        if zoom_start is None:
            zoom_start = self.config.visualization.default_zoom
        
        if map_center is None:
            map_center = (
                self.optimizer.demand_points['latitude'].mean(),
                self.optimizer.demand_points['longitude'].mean()
            )
        
        m = folium.Map(location=map_center, zoom_start=zoom_start)
        
        # Add demand points
        for _, row in self.optimizer.demand_points.iterrows():
            weight = self.optimizer.population_weights.get(row['id'], 1)
            radius = max(5, min(20, weight / 10))
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=radius,
                color='blue',
                fill=True,
                fillColor='blue',
                fill_opacity=0.6,
                popup=f"Demand: {row['id']}, Weight: {weight}",
                tooltip=f"Demand: {row['id']}"
            ).add_to(m)
        
        # Add non-selected candidates (orange)
        for _, row in self.optimizer.candidate_locations.iterrows():
            if row['id'] not in solution:
                folium.Marker(
                    location=(row['latitude'], row['longitude']),
                    icon=folium.Icon(color='orange', icon='home'),
                    popup=f"Candidate: {row['id']}",
                    tooltip=f"Candidate: {row['id']}"
                ).add_to(m)
        
        # Add selected ATMs (red)
        for atm_id in solution:
            location_data = self.optimizer.candidate_locations.loc[
                self.optimizer.candidate_locations['id'] == atm_id
            ].iloc[0]
            
            folium.Marker(
                location=(location_data['latitude'], location_data['longitude']),
                icon=folium.Icon(color='red', icon='info-sign'),
                popup=f"Selected ATM: {atm_id}",
                tooltip=f"ATM: {atm_id}"
            ).add_to(m)
        
        # Add routes to selected ATMs
        for atm_id in solution:
            atm_location = self.optimizer.candidate_locations.loc[
                self.optimizer.candidate_locations['id'] == atm_id
            ].iloc[0]
            atm_coords = (atm_location['latitude'], atm_location['longitude'])
            
            for _, demand_row in self.optimizer.demand_points.iterrows():
                demand_id = demand_row['id']
                demand_coords = (demand_row['latitude'], demand_row['longitude'])
                travel_time = self.optimizer.travel_times.get((demand_id, atm_id), float('inf'))
                
                closest_atm = min(solution,
                                key=lambda x: self.optimizer.travel_times.get((demand_id, x), float('inf')))
                if atm_id == closest_atm:
                    folium.PolyLine(
                        locations=[demand_coords, atm_coords],
                        color='green',
                        weight=2,
                        opacity=0.7,
                        popup=f"Travel: {travel_time:.1f}s",
                        tooltip=f"{travel_time:.1f}s"
                    ).add_to(m)
        
        # Legend
        legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 250px; height: 120px; 
            background-color: white; border:2px solid grey; z-index:9999; 
            font-size:14px; padding: 10px">
            <p><b>Legend</b></p>
            <p><i class="fa fa-circle" style="color:blue"></i> Demand Points</p>
            <p><i class="fa fa-map-marker" style="color:red"></i> Selected ATMs</p>
            <p><i class="fa fa-map-marker" style="color:orange"></i> Candidate Locations</p>
            <p><i class="fa fa-minus" style="color:green"></i> Travel Routes</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        logger.info(f"Created comprehensive map with {len(solution)} selected ATMs")
        return m
    
    def plot_comparison_results(self,
                               sa_results: List[float],
                               greedy_results: List[float],
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive comparison plots for SA vs Greedy algorithms.
        
        Parameters:
        -----------
        sa_results : list
            Objective values from Simulated Annealing
        greedy_results : list
            Objective values from Greedy algorithm
        save_path : Path, optional
            Path to save the figure
        
        Returns:
        --------
        matplotlib.Figure
            Figure object with comparison plots
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Box plot
        ax1.boxplot([sa_results, greedy_results],
                   labels=['Simulated\nAnnealing', 'Greedy'])
        ax1.set_title('Algorithm Performance Comparison', fontsize=15)
        ax1.set_ylabel('Objective Value')
        ax1.grid(True, alpha=0.3)
        
        # Histograms
        ax2.hist(sa_results, alpha=0.7, label='Simulated Annealing',
                bins=15, color='skyblue')
        ax2.hist(greedy_results, alpha=0.7, label='Greedy',
                bins=15, color='lightcoral')
        ax2.set_title('Distribution of Results', fontsize=15)
        ax2.set_xlabel('Objective Value')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Convergence plot
        ax3.plot(range(1, len(sa_results) + 1), sa_results, 'o-',
                label='Simulated Annealing', alpha=0.7, color='skyblue')
        ax3.plot(range(1, len(greedy_results) + 1), greedy_results, 's-',
                label='Greedy', alpha=0.7, color='lightcoral')
        ax3.set_title('Results by Iteration', fontsize=15)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot for normality
        if len(np.unique(sa_results)) > 1:
            stats.probplot(sa_results, dist="norm", plot=ax4)
            ax4.set_title('Q-Q Plot: SA Results vs Normal', fontsize=15)
        else:
            ax4.text(0.5, 0.5, 'SA results identical\n(no variation)',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=15)
            ax4.set_title('SA Results Analysis')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.visualization.dpi, bbox_inches='tight')
            logger.info(f"Saved comparison plots to {save_path}")
        
        return fig
