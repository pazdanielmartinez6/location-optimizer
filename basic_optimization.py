"""
Basic Optimization Example

This script demonstrates how to:
1. Load data
2. Initialize the optimizer
3. Run an optimization
4. Visualize results
5. Save outputs

Author: Daniel Paz Martinez
"""

import pandas as pd
from pathlib import Path

# Import the optimizer package
from atm_optimizer import ATMLocationOptimizer, ATMVisualizer
from atm_optimizer.config import get_config

def main():
    """Run a basic optimization example."""
    
    print("="*60)
    print("ATM LOCATION OPTIMIZER - BASIC EXAMPLE")
    print("="*60)
    
    # Get configuration
    config = get_config()
    
    # Step 1: Load demand points
    print("\n1. Loading data...")
    demand_points_path = config.paths.demand_points
    
    if not demand_points_path.exists():
        print(f"Error: Data file not found: {demand_points_path}")
        print("Please ensure demand_points.csv is in the data/ directory")
        return
    
    demand_points = pd.read_csv(demand_points_path)
    print(f"   ✓ Loaded {len(demand_points)} demand points")
    
    # Step 2: Load candidate locations (optional)
    candidates_path = config.paths.atm_candidates
    candidate_locations = None
    
    if candidates_path.exists():
        candidate_locations = pd.read_csv(candidates_path)
        print(f"   ✓ Loaded {len(candidate_locations)} candidate locations")
    else:
        print("   ℹ No candidate locations file found - will use random demand points")
    
    # Step 3: Initialize optimizer
    print("\n2. Initializing optimizer...")
    print("   (Using cached travel times if available)")
    
    optimizer = ATMLocationOptimizer(
        demand_points=demand_points,
        candidate_locations=candidate_locations,
        use_cache=True  # Use cache if available
    )
    
    print("   ✓ Optimizer initialized")
    
    # Step 4: Run optimization
    print("\n3. Running optimization...")
    num_atms = 5
    print(f"   Finding optimal locations for {num_atms} ATMs")
    print("   Using Simulated Annealing algorithm")
    
    solution = optimizer.optimize(
        num_atms=num_atms,
        method='simulated_annealing'
    )
    
    # Evaluate solution
    objective_value = optimizer.evaluate_solution(solution)
    
    print(f"   ✓ Optimization complete!")
    print(f"   Selected ATM IDs: {solution}")
    print(f"   Objective value: {objective_value:.2f}")
    
    # Step 5: Create visualizations
    print("\n4. Creating visualizations...")
    
    visualizer = ATMVisualizer(optimizer)
    
    # Create output directory
    output_dir = Path("outputs") / "basic_example"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map with all candidates
    print("   Creating map with all candidates...")
    map_all = visualizer.visualize_all_candidates()
    map_all.save(output_dir / "all_candidates.html")
    print(f"   ✓ Saved: {output_dir / 'all_candidates.html'}")
    
    # Solution map
    print("   Creating solution map...")
    map_solution = visualizer.visualize_solution(solution)
    map_solution.save(output_dir / "solution_map.html")
    print(f"   ✓ Saved: {output_dir / 'solution_map.html'}")
    
    # Solution with candidates
    print("   Creating comprehensive map...")
    map_comprehensive = visualizer.visualize_solution_with_candidates(solution)
    map_comprehensive.save(output_dir / "solution_with_candidates.html")
    print(f"   ✓ Saved: {output_dir / 'solution_with_candidates.html'}")
    
    # Step 6: Save solution to CSV
    print("\n5. Saving solution...")
    
    solution_data = []
    for atm_id in solution:
        location = optimizer.candidate_locations.loc[
            optimizer.candidate_locations['id'] == atm_id
        ].iloc[0]
        
        solution_data.append({
            'atm_id': atm_id,
            'latitude': location['latitude'],
            'longitude': location['longitude']
        })
    
    solution_df = pd.DataFrame(solution_data)
    solution_df.to_csv(output_dir / "atm_solution.csv", index=False)
    print(f"   ✓ Saved: {output_dir / 'atm_solution.csv'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Number of ATMs: {num_atms}")
    print(f"Objective value: {objective_value:.2f}")
    print(f"Selected locations: {solution}")
    print(f"\nAll outputs saved to: {output_dir}")
    print("="*60)
    
    print("\n✅ Example complete! Open the HTML files in a browser to view the maps.")


if __name__ == "__main__":
    main()
