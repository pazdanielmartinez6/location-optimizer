"""
Command-Line Interface for ATM Location Optimizer

Provides an interactive menu system for users to:
- Run single optimizations
- Compare algorithms statistically
- Create visualizations and animations
- Generate demand points (optional)

Author: Daniel Paz Martinez
"""

import sys
from pathlib import Path
import pandas as pd
import logging
from typing import Optional

from .optimizer import ATMLocationOptimizer
from .visualizer import ATMVisualizer
from .animator import ATMAnimator
from .config import get_config, print_config
from .utils import (
    setup_logging, validate_data_files, load_data_file,
    create_output_directory, print_section_header, format_time
)

logger = logging.getLogger('atm_optimizer')


def main():
    """Main entry point for the CLI."""
    print_section_header("ATM LOCATION OPTIMIZER")
    print("A comprehensive optimization toolkit for strategic ATM placement")
    print("Author: Daniel Paz Martinez")
    print("GitHub: https://github.com/[your-username]/atm-location-optimizer")
    print("\n")
    
    # Setup logging
    config = get_config()
    setup_logging(
        log_level=config.log_level,
        log_to_file=config.log_to_file,
        log_to_console=config.log_to_console
    )
    
    logger.info("="*60)
    logger.info("ATM Location Optimizer - Session Started")
    logger.info("="*60)
    
    # Check if we have cached travel times
    cache_path = config.paths.travel_times_cache
    has_cache = cache_path.exists()
    
    if has_cache:
        print(f"✓ Found travel times cache: {cache_path}")
        use_cache = prompt_yes_no("Do you want to use cached travel times?", default=True)
    else:
        print(f"⚠ No cache file found: {cache_path}")
        print("  Travel times will be calculated using OSRM API (this may take time)")
        use_cache = False
    
    # Initialize optimizer
    try:
        optimizer = initialize_optimizer(use_cache)
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have the required data files:")
        print(f"  - {config.paths.demand_points}")
        print(f"  - {config.paths.travel_times_cache} (or internet connection for OSRM API)")
        return
    
    # Main menu loop
    while True:
        choice = show_main_menu()
        
        if choice == '1':
            run_single_optimization(optimizer)
        elif choice == '2':
            run_statistical_comparison(optimizer)
        elif choice == '3':
            run_both(optimizer)
        elif choice == '4':
            visualize_candidates(optimizer)
        elif choice == '5':
            show_configuration()
        elif choice == '6':
            print("\nThank you for using ATM Location Optimizer!")
            logger.info("Session ended by user")
            break
        else:
            print("\n❌ Invalid choice. Please try again.")


def show_main_menu() -> str:
    """Display main menu and get user choice."""
    print("\n" + "="*60)
    print("MAIN MENU")
    print("="*60)
    print("1. Single optimization run")
    print("2. Statistical comparison of algorithms")
    print("3. Both (optimization + comparison)")
    print("4. Visualize candidate locations")
    print("5. Show configuration")
    print("6. Exit")
    print("="*60)
    
    choice = input("\nEnter your choice (1-6): ").strip()
    return choice


def initialize_optimizer(use_cache: bool) -> ATMLocationOptimizer:
    """Initialize the optimizer with data."""
    config = get_config()
    
    print("\n" + "-"*60)
    print("INITIALIZING OPTIMIZER")
    print("-"*60)
    
    # Load demand points
    demand_points = load_data_file(
        config.paths.demand_points,
        ['id', 'latitude', 'longitude'],
        "Demand Points"
    )
    
    if demand_points is None:
        raise ValueError(f"Could not load demand points from {config.paths.demand_points}")
    
    # Load candidate locations (optional)
    candidate_locations = None
    if config.paths.atm_candidates.exists():
        candidate_locations = load_data_file(
            config.paths.atm_candidates,
            ['id', 'latitude', 'longitude'],
            "ATM Candidates"
        )
    
    # Load population weights (optional)
    population_weights = None
    if config.paths.population_weights.exists():
        weights_df = pd.read_csv(config.paths.population_weights)
        if 'id' in weights_df.columns and 'weight' in weights_df.columns:
            population_weights = weights_df.set_index('id')['weight']
    
    # Initialize optimizer
    print("\nInitializing optimizer...")
    if use_cache:
        print("Loading from cache...")
        optimizer = ATMLocationOptimizer.load_from_cache()
    else:
        print("Calculating travel times (this may take several minutes)...")
        optimizer = ATMLocationOptimizer(
            demand_points=demand_points,
            candidate_locations=candidate_locations,
            population_weights=population_weights,
            use_cache=False
        )
        # Save cache for future use
        optimizer.save_travel_times()
    
    print("✓ Optimizer initialized successfully")
    return optimizer


def run_single_optimization(optimizer: ATMLocationOptimizer):
    """Run a single optimization and visualization."""
    print_section_header("SINGLE OPTIMIZATION RUN")
    
    # Get parameters
    num_atms = prompt_integer("Enter number of ATMs to place", min_val=1, max_val=50, default=5)
    
    method = prompt_choice(
        "Choose optimization method",
        ["Simulated Annealing", "Greedy"],
        default=1
    )
    method_name = 'simulated_annealing' if method == 1 else 'greedy'
    
    create_animation = prompt_yes_no("Create animated visualization?", default=False)
    
    # Create output directory
    output_dir = create_output_directory(num_atms, "single_run")
    print(f"\n📁 Results will be saved to: {output_dir}")
    
    # Run optimization
    print(f"\n🔄 Running {method_name} optimization for {num_atms} ATMs...")
    import time
    start_time = time.time()
    
    solution = optimizer.optimize(num_atms=num_atms, method=method_name)
    
    elapsed = time.time() - start_time
    objective = optimizer.evaluate_solution(solution)
    
    print(f"\n✓ Optimization complete in {format_time(elapsed)}")
    print(f"  Solution: {solution}")
    print(f"  Objective value: {objective:.2f}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    visualizer = ATMVisualizer(optimizer)
    
    # All candidates map
    map_all = visualizer.visualize_all_candidates()
    map_all.save(output_dir / "all_candidate_locations.html")
    print(f"  ✓ Saved: all_candidate_locations.html")
    
    # Solution with candidates
    map_with_candidates = visualizer.visualize_solution_with_candidates(solution)
    map_with_candidates.save(output_dir / "solution_with_candidates.html")
    print(f"  ✓ Saved: solution_with_candidates.html")
    
    # Solution map
    map_solution = visualizer.visualize_solution(solution)
    map_solution.save(output_dir / "atm_solution_map.html")
    print(f"  ✓ Saved: atm_solution_map.html")
    
    # Save solution CSV
    solution_data = []
    for atm_id in solution:
        loc = optimizer.candidate_locations.loc[
            optimizer.candidate_locations['id'] == atm_id
        ].iloc[0]
        solution_data.append({
            'id': atm_id,
            'latitude': loc['latitude'],
            'longitude': loc['longitude']
        })
    
    solution_df = pd.DataFrame(solution_data)
    solution_df.to_csv(output_dir / "atm_solution.csv", index=False)
    print(f"  ✓ Saved: atm_solution.csv")
    
    # Animation
    if create_animation:
        print("\n🎬 Creating animation (this may take a minute)...")
        animator = ATMAnimator(optimizer)
        animator.animate_solution(
            solution,
            save_path=output_dir / "animated_solution.gif"
        )
        print(f"  ✓ Saved: animated_solution.gif")
    
    print(f"\n✅ All results saved to: {output_dir}")
    logger.info(f"Single optimization complete: {num_atms} ATMs, objective={objective:.2f}")


def run_statistical_comparison(optimizer: ATMLocationOptimizer):
    """Run statistical comparison between algorithms."""
    print_section_header("STATISTICAL ALGORITHM COMPARISON")
    
    # Get parameters
    num_atms = prompt_integer("Enter number of ATMs to place", min_val=1, max_val=50, default=5)
    num_iterations = prompt_integer(
        "Enter number of iterations for testing",
        min_val=10, max_val=100, default=30
    )
    
    if num_iterations < 30:
        print("\n⚠ Warning: Less than 30 iterations may not provide reliable statistical results.")
        if not prompt_yes_no("Continue anyway?", default=False):
            return
    
    create_animation = prompt_yes_no("Create comparison animation?", default=False)
    
    # Create output directory
    output_dir = create_output_directory(num_atms, "comparison")
    print(f"\n📁 Results will be saved to: {output_dir}")
    
    print(f"\n🔄 Running {num_iterations} iterations of each algorithm...")
    print("   This may take several minutes...")
    
    # Run comparison
    import time
    start_time = time.time()
    
    results = optimizer.run_algorithm_comparison(
        num_atms=num_atms,
        num_iterations=num_iterations,
        save_results=True,
        output_dir=output_dir
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Statistical comparison complete in {format_time(elapsed)}")
    
    # Display results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Simulated Annealing:")
    print(f"  Mean: {results['sa_stats']['mean']:.2f}")
    print(f"  Std:  {results['sa_stats']['std']:.2f}")
    print(f"  Best: {results['sa_stats']['min']:.2f}")
    print(f"\nGreedy Algorithm:")
    print(f"  Mean: {results['greedy_stats']['mean']:.2f}")
    print(f"  Std:  {results['greedy_stats']['std']:.2f}")
    print(f"  Best: {results['greedy_stats']['min']:.2f}")
    print(f"\nStatistical Test: {results['test_name']}")
    print(f"  p-value: {results['p_value']:.6f}")
    print(f"  Significant: {'Yes' if results['significant'] else 'No'}")
    print(f"  Effect size (Cohen's d): {results['cohens_d']:.4f}")
    print(f"\nPerformance:")
    print(f"  SA improvement: {results['improvement_percent']:.2f}%")
    print(f"  SA avg time: {results['sa_time_avg']:.4f}s")
    print(f"  Greedy avg time: {results['greedy_time_avg']:.4f}s")
    print("="*60)
    
    # Create comparison animation if requested
    if create_animation:
        print("\n🎬 Creating comparison animation...")
        # Get best solutions from each algorithm
        print("   Running one more iteration of each to get solutions for animation...")
        solution_sa = optimizer.optimize(num_atms, method='simulated_annealing')
        solution_greedy = optimizer.optimize(num_atms, method='greedy')
        
        animator = ATMAnimator(optimizer)
        animator.create_comparison_animation(
            solution_sa,
            solution_greedy,
            save_path=output_dir / "comparison_animation.gif"
        )
        print(f"  ✓ Saved: comparison_animation.gif")
    
    print(f"\n✅ All results saved to: {output_dir}")
    logger.info(f"Statistical comparison complete: {num_atms} ATMs, {num_iterations} iterations")


def run_both(optimizer: ATMLocationOptimizer):
    """Run both single optimization and statistical comparison."""
    print_section_header("COMPLETE ANALYSIS")
    print("This will run both a single optimization and statistical comparison.")
    
    num_atms = prompt_integer("Enter number of ATMs to place", min_val=1, max_val=50, default=5)
    
    print("\n" + "-"*60)
    print("PART 1: SINGLE OPTIMIZATION")
    print("-"*60)
    run_single_optimization(optimizer)
    
    print("\n" + "-"*60)
    print("PART 2: STATISTICAL COMPARISON")
    print("-"*60)
    run_statistical_comparison(optimizer)
    
    print("\n✅ Complete analysis finished!")


def visualize_candidates(optimizer: ATMLocationOptimizer):
    """Visualize candidate locations only."""
    print_section_header("VISUALIZE CANDIDATES")
    
    output_dir = create_output_directory(0, "visualization")
    
    print("Creating map with candidate locations...")
    visualizer = ATMVisualizer(optimizer)
    m = visualizer.visualize_all_candidates()
    
    output_path = output_dir / "candidates_and_demand.html"
    m.save(output_path)
    
    print(f"\n✓ Map saved to: {output_path}")
    print(f"  Total candidates: {len(optimizer.candidate_locations)}")
    print(f"  Total demand points: {len(optimizer.demand_points)}")
    
    logger.info("Candidate visualization complete")


def show_configuration():
    """Display current configuration."""
    print_config()
    
    input("\nPress Enter to continue...")


# Helper functions for user input

def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{question} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


def prompt_integer(question: str, min_val: int = 1, max_val: int = 100, default: int = 5) -> int:
    """Prompt user for integer input."""
    while True:
        response = input(f"{question} [{default}]: ").strip()
        
        if not response:
            return default
        
        try:
            value = int(response)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"❌ Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid number")


def prompt_choice(question: str, options: list, default: int = 1) -> int:
    """Prompt user to choose from a list of options."""
    print(f"\n{question}:")
    for i, option in enumerate(options, 1):
        marker = " (default)" if i == default else ""
        print(f"  {i}. {option}{marker}")
    
    while True:
        response = input(f"Enter choice [1-{len(options)}] [{default}]: ").strip()
        
        if not response:
            return default
        
        try:
            choice = int(response)
            if 1 <= choice <= len(options):
                return choice
            else:
                print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("❌ Please enter a valid number")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Check the log file for details.")
        sys.exit(1)
