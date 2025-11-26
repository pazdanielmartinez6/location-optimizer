"""
Animator Module for ATM Location Optimizer

Creates animated visualizations showing route development and algorithm comparisons.
Requires matplotlib and contextily for street-level basemaps.

Author: Daniel Paz Martinez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from typing import List, Optional, Dict
from pathlib import Path
import logging

try:
    import contextily as ctx
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False
    logging.warning("Contextily not available - street layers will be disabled")

from .config import get_config

logger = logging.getLogger('atm_optimizer')


class ATMAnimator:
    """Handles animated visualizations for the optimizer."""
    
    def __init__(self, optimizer):
        """
        Initialize animator with an optimizer instance.
        
        Parameters:
        -----------
        optimizer : ATMLocationOptimizer
            The optimizer instance to visualize
        """
        self.optimizer = optimizer
        self.config = get_config()
        logger.info("Animator initialized")
        
        if not CONTEXTILY_AVAILABLE:
            logger.warning("Street layer features disabled (contextily not installed)")
    
    def _add_basemap_to_axis(self, ax, bounds: tuple, zoom_level: int = 12):
        """
        Add a street map basemap to the given axis.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axis to add the basemap to
        bounds : tuple
            (min_lon, max_lon, min_lat, max_lat) bounds for the map
        zoom_level : int
            Zoom level for the basemap tiles
        """
        if not CONTEXTILY_AVAILABLE:
            logger.debug("Skipping basemap (contextily not available)")
            return
        
        try:
            west, east, south, north = bounds
            
            ctx.add_basemap(
                ax,
                crs='EPSG:4326',
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=zoom_level,
                alpha=self.config.visualization.street_alpha
            )
            
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            logger.debug("Added street basemap to plot")
        
        except Exception as e:
            logger.warning(f"Could not add basemap: {e}")
    
    def _prepare_routes_for_animation(self,
                                     solution: List[int],
                                     max_demand_points: Optional[int] = None,
                                     seed: Optional[int] = None) -> List[Dict]:
        """
        Prepare route data for animation by sampling demand points.
        
        Parameters:
        -----------
        solution : list
            List of selected ATM IDs
        max_demand_points : int, optional
            Maximum number of demand points to animate
        seed : int, optional
            Random seed for sampling
        
        Returns:
        --------
        list
            List of dictionaries containing route information
        """
        if max_demand_points is None:
            max_demand_points = self.config.visualization.animation_max_demand_points
        
        if seed is None:
            seed = self.config.optimization.random_seed
        
        routes = []
        
        # Sample demand points if needed
        if len(self.optimizer.demand_points) > max_demand_points:
            logger.info(f"Sampling {max_demand_points} demand points from {len(self.optimizer.demand_points)}")
            np.random.seed(seed)
            sampled_demand = self.optimizer.demand_points.sample(n=max_demand_points, random_state=seed)
        else:
            sampled_demand = self.optimizer.demand_points
        
        # Find closest ATM for each demand point
        for _, demand_row in sampled_demand.iterrows():
            demand_id = demand_row['id']
            
            # Find closest ATM
            closest_atm = None
            min_travel_time = float('inf')
            
            for atm_id in solution:
                travel_time = self.optimizer.travel_times.get((demand_id, atm_id), float('inf'))
                if travel_time < min_travel_time:
                    min_travel_time = travel_time
                    closest_atm = atm_id
            
            if closest_atm is not None and min_travel_time != float('inf'):
                # Get ATM coordinates
                atm_location = self.optimizer.candidate_locations.loc[
                    self.optimizer.candidate_locations['id'] == closest_atm
                ].iloc[0]
                
                route_info = {
                    'demand_id': demand_id,
                    'demand_lat': demand_row['latitude'],
                    'demand_lon': demand_row['longitude'],
                    'atm_id': closest_atm,
                    'atm_lat': atm_location['latitude'],
                    'atm_lon': atm_location['longitude'],
                    'travel_time': min_travel_time,
                    'weight': self.optimizer.population_weights.get(demand_id, 1)
                }
                routes.append(route_info)
        
        # Sort routes by travel time
        routes.sort(key=lambda x: x['travel_time'])
        
        logger.info(f"Prepared {len(routes)} routes for animation")
        return routes
    
    def animate_solution(self,
                        solution: List[int],
                        save_path: Optional[Path] = None,
                        frames_per_route: int = None,
                        interval: int = None,
                        figsize: tuple = None,
                        max_demand_points: int = None,
                        add_streets: bool = None,
                        zoom_level: int = None) -> None:
        """
        Create an animated visualization of the solution.
        
        Parameters:
        -----------
        solution : list
            List of selected ATM IDs
        save_path : Path, optional
            Path to save the animation (default: animated_solution.gif)
        frames_per_route : int, optional
            Frames per route revelation
        interval : int, optional
            Milliseconds between frames
        figsize : tuple, optional
            Figure size
        max_demand_points : int, optional
            Maximum demand points to animate
        add_streets : bool, optional
            Whether to add street layer
        zoom_level : int, optional
            Zoom level for street tiles
        """
        # Use config defaults if not provided
        if save_path is None:
            save_path = Path('animated_atm_solution.gif')
        if frames_per_route is None:
            frames_per_route = self.config.visualization.animation_frames_per_route
        if interval is None:
            interval = self.config.visualization.animation_interval
        if figsize is None:
            figsize = self.config.visualization.figure_size
        if add_streets is None:
            add_streets = self.config.visualization.add_streets and CONTEXTILY_AVAILABLE
        if zoom_level is None:
            zoom_level = self.config.visualization.street_zoom_level
        
        logger.info(f"Creating animated visualization for {len(solution)} ATMs")
        
        # Prepare routes
        routes_data = self._prepare_routes_for_animation(solution, max_demand_points)
        
        if not routes_data:
            logger.error("No routes to animate")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up plot bounds
        all_lats = list(self.optimizer.demand_points['latitude']) + \
                   list(self.optimizer.candidate_locations['latitude'])
        all_lons = list(self.optimizer.demand_points['longitude']) + \
                   list(self.optimizer.candidate_locations['longitude'])
        
        lat_margin = (max(all_lats) - min(all_lats)) * 0.1
        lon_margin = (max(all_lons) - min(all_lons)) * 0.1
        
        bounds = (
            min(all_lons) - lon_margin,
            max(all_lons) + lon_margin,
            min(all_lats) - lat_margin,
            max(all_lats) + lat_margin
        )
        
        def init():
            """Initialize animation."""
            ax.clear()
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            
            # Add street layer
            if add_streets:
                self._add_basemap_to_axis(ax, bounds, zoom_level)
            
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_title(f'ATM Location Optimization - Route Visualization\n' +
                        f'Showing {len(routes_data)} sampled routes', fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)
            
            # Plot non-selected candidates
            for _, row in self.optimizer.candidate_locations.iterrows():
                if row['id'] not in solution:
                    ax.scatter(row['longitude'], row['latitude'],
                             c='orange', marker='s', s=120, alpha=0.9,
                             zorder=5, edgecolors='black', linewidth=1)
            
            # Plot selected ATMs
            for atm_id in solution:
                location = self.optimizer.candidate_locations.loc[
                    self.optimizer.candidate_locations['id'] == atm_id
                ].iloc[0]
                ax.scatter(location['longitude'], location['latitude'],
                         c='red', marker='D', s=180, alpha=1.0,
                         zorder=6, edgecolors='black', linewidth=1)
            
            # Legend
            legend_elements = [
                mpatches.Patch(color='orange', alpha=0.9, label='Candidate Locations'),
                mpatches.Patch(color='red', alpha=1.0, label='Selected ATMs'),
                mpatches.Patch(color='green', alpha=0.8, label='Travel Routes')
            ]
            ax.legend(handles=legend_elements, loc='upper right',
                     fancybox=True, shadow=True, framealpha=0.9)
            
            return []
        
        def animate_frame(frame):
            """Animate each frame."""
            if frame % frames_per_route == 0:
                route_index = frame // frames_per_route
                if route_index < len(routes_data):
                    route = routes_data[route_index]
                    
                    # Draw route
                    ax.plot([route['demand_lon'], route['atm_lon']],
                           [route['demand_lat'], route['atm_lat']],
                           'g-', alpha=0.8, linewidth=2.5, zorder=4)
                    
                    # Update progress text
                    progress_text = (
                        f"Showing {route_index + 1} of {len(routes_data)} routes\n"
                        f"Current: Demand {route['demand_id']} → ATM {route['atm_id']}\n"
                        f"Travel time: {route['travel_time']:.1f} seconds"
                    )
                    
                    # Remove previous text
                    for txt in ax.texts:
                        if 'Showing' in txt.get_text():
                            txt.remove()
                    
                    ax.text(0.02, 0.02, progress_text,
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', alpha=0.95,
                                   edgecolor='black', boxstyle='round'),
                           verticalalignment='bottom', fontsize=10, zorder=10)
            
            return ax.get_children()
        
        # Create animation
        total_frames = len(routes_data) * frames_per_route
        anim = animation.FuncAnimation(
            fig, animate_frame, init_func=init,
            frames=total_frames, interval=interval,
            blit=False, repeat=True
        )
        
        # Save animation
        logger.info(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=self.config.visualization.animation_fps)
        plt.close()
        
        logger.info(f"Animation saved: {save_path}")
        logger.info(f"Total routes animated: {len(routes_data)}")
    
    def create_comparison_animation(self,
                                   solution_sa: List[int],
                                   solution_greedy: List[int],
                                   save_path: Optional[Path] = None,
                                   frames_per_route: int = None,
                                   interval: int = None,
                                   figsize: tuple = None,
                                   max_demand_points: int = None,
                                   add_streets: bool = None,
                                   zoom_level: int = None) -> None:
        """
        Create side-by-side animated comparison of two solutions.
        
        Parameters:
        -----------
        solution_sa : list
            Solution from Simulated Annealing
        solution_greedy : list
            Solution from Greedy algorithm
        save_path : Path, optional
            Path to save animation
        frames_per_route : int, optional
            Frames per route
        interval : int, optional
            Milliseconds between frames
        figsize : tuple, optional
            Figure size
        max_demand_points : int, optional
            Maximum demand points to animate
        add_streets : bool, optional
            Whether to add street layer
        zoom_level : int, optional
            Zoom level for streets
        """
        # Use defaults
        if save_path is None:
            save_path = Path('algorithm_comparison_animation.gif')
        if frames_per_route is None:
            frames_per_route = 3
        if interval is None:
            interval = self.config.visualization.animation_interval
        if figsize is None:
            figsize = self.config.visualization.comparison_figure_size
        if add_streets is None:
            add_streets = self.config.visualization.add_streets and CONTEXTILY_AVAILABLE
        if zoom_level is None:
            zoom_level = self.config.visualization.street_zoom_level
        
        logger.info("Creating comparison animation")
        
        # Prepare routes
        routes_sa = self._prepare_routes_for_animation(solution_sa, max_demand_points)
        routes_greedy = self._prepare_routes_for_animation(solution_greedy, max_demand_points)
        
        # Calculate objectives
        obj_sa = self.optimizer.evaluate_solution(solution_sa)
        obj_greedy = self.optimizer.evaluate_solution(solution_greedy)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Set up bounds (same for both)
        all_lats = list(self.optimizer.demand_points['latitude']) + \
                   list(self.optimizer.candidate_locations['latitude'])
        all_lons = list(self.optimizer.demand_points['longitude']) + \
                   list(self.optimizer.candidate_locations['longitude'])
        
        lat_margin = (max(all_lats) - min(all_lats)) * 0.1
        lon_margin = (max(all_lons) - min(all_lons)) * 0.1
        
        bounds = (
            min(all_lons) - lon_margin,
            max(all_lons) + lon_margin,
            min(all_lats) - lat_margin,
            max(all_lats) + lat_margin
        )
        
        def setup_subplot(ax, title, solution):
            """Setup a subplot."""
            ax.clear()
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            
            if add_streets:
                self._add_basemap_to_axis(ax, bounds, zoom_level)
            
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.set_title(title, fontsize=12, pad=15)
            ax.grid(True, alpha=0.3)
            
            # Plot non-selected candidates
            for _, row in self.optimizer.candidate_locations.iterrows():
                if row['id'] not in solution:
                    ax.scatter(row['longitude'], row['latitude'],
                             c='orange', marker='s', s=60, alpha=0.9,
                             zorder=5, edgecolors='black', linewidth=0.5)
            
            # Plot selected ATMs
            for atm_id in solution:
                location = self.optimizer.candidate_locations.loc[
                    self.optimizer.candidate_locations['id'] == atm_id
                ].iloc[0]
                ax.scatter(location['longitude'], location['latitude'],
                         c='red', marker='D', s=120, alpha=1.0,
                         zorder=6, edgecolors='black', linewidth=1)
        
        def init():
            """Initialize both subplots."""
            setup_subplot(ax1, f'Simulated Annealing\nObjective: {obj_sa:.0f}', solution_sa)
            setup_subplot(ax2, f'Greedy Algorithm\nObjective: {obj_greedy:.0f}', solution_greedy)
            
            fig.suptitle('Algorithm Comparison - Route Visualization',
                        fontsize=16, y=0.95)
            
            return []
        
        def animate_frame(frame):
            """Animate both subplots."""
            if frame % frames_per_route == 0:
                route_index = frame // frames_per_route
                
                # Animate SA
                if route_index < len(routes_sa):
                    route = routes_sa[route_index]
                    ax1.plot([route['demand_lon'], route['atm_lon']],
                            [route['demand_lat'], route['atm_lat']],
                            'g-', alpha=0.8, linewidth=2.5, zorder=4)
                
                # Animate Greedy
                if route_index < len(routes_greedy):
                    route = routes_greedy[route_index]
                    ax2.plot([route['demand_lon'], route['atm_lon']],
                            [route['demand_lat'], route['atm_lat']],
                            'g-', alpha=0.8, linewidth=2.5, zorder=4)
                
                # Update progress
                max_routes = max(len(routes_sa), len(routes_greedy))
                progress = f'Routes shown: {min(route_index + 1, max_routes)} / {max_routes}'
                
                # Remove previous progress
                for txt in fig.texts:
                    if 'routes shown:' in txt.get_text().lower():
                        txt.remove()
                
                fig.text(0.5, 0.02, progress, ha='center', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.95,
                                edgecolor='black', boxstyle='round'))
            
            return ax1.get_children() + ax2.get_children()
        
        # Create animation
        max_routes = max(len(routes_sa), len(routes_greedy))
        total_frames = max_routes * frames_per_route
        
        anim = animation.FuncAnimation(
            fig, animate_frame, init_func=init,
            frames=total_frames, interval=interval,
            blit=False, repeat=True
        )
        
        # Save
        logger.info(f"Saving comparison animation to {save_path}...")
        anim.save(save_path, writer='pillow', fps=self.config.visualization.animation_fps)
        plt.close()
        
        logger.info(f"Comparison animation saved: {save_path}")
        logger.info(f"SA routes: {len(routes_sa)}, Greedy routes: {len(routes_greedy)}")

