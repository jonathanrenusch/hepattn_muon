#!/usr/bin/env python3
"""
Example script showing how to run the ATLAS muon hit filter evaluation 
with baseline filtering functionality.

This demonstrates both all-tracks and baseline-filtered evaluations.
"""
import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def run_example_evaluation():
    """Run a small example evaluation to test the baseline filtering."""
    
    # Import the evaluator
    from evaluate_hit_filter_dataloader import AtlasMuonEvaluatorDataLoader
    
    # Example paths - you'll need to update these to your actual data paths
    eval_path = "/shared/tracking/data/1_5Mio_test_training/epoch=049-val_acc=0.99711_ml_test_data_150K_processed_eval.h5"
    data_dir = "/shared/tracking/data/ml_test_data_150K_processed"
    config_path = "./configs/NGT/atlas_muon_event_NGT_plotting.yaml"
    output_dir = "./evaluation_results_with_baseline"
    
    # Verify paths exist
    if not Path(eval_path).exists():
        print(f"❌ Evaluation file not found: {eval_path}")
        print("Please update the eval_path variable with the correct path to your evaluation HDF5 file")
        return False
        
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please update the data_dir variable with the correct path to your processed data")
        return False
        
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please update the config_path variable with the correct path to your config YAML")
        return False
    
    print("🚀 Starting ATLAS Muon Hit Filter Evaluation with Baseline Filtering")
    print("=" * 70)
    print(f"Evaluation file: {eval_path}")
    print(f"Data directory: {data_dir}")
    print(f"Config file: {config_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    try:
        # Create evaluator instance
        evaluator = AtlasMuonEvaluatorDataLoader(
            eval_path=eval_path,
            data_dir=data_dir,
            config_path=config_path,
            output_dir=output_dir,
            max_events=100  # Use small number for testing
        )
        
        # Run the full evaluation (this will create both all_tracks and baseline_filtered_tracks results)
        evaluator.run_full_evaluation(
            skip_individual_plots=True,     # Skip individual plots for speed
            skip_technology_plots=True,     # Skip technology plots for speed  
            skip_eta_phi_plots=True         # Skip eta/phi plots for speed
        )
        
        print("\\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {evaluator.output_dir}")
        print(f"  - All tracks results: {evaluator.all_tracks_dir}")
        print(f"  - Baseline filtered results: {evaluator.baseline_filtered_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_usage_instructions():
    """Print instructions for running the evaluation."""
    print("\\n" + "=" * 70)
    print("USAGE INSTRUCTIONS")
    print("=" * 70)
    print("To run the full evaluation with baseline filtering:")
    print("")
    print("1. Update the file paths in this script to match your data:")
    print("   - eval_path: path to your evaluation HDF5 file")
    print("   - data_dir: path to your processed test data directory")
    print("   - config_path: path to your config YAML file")
    print("")
    print("2. Run the evaluation using pixi from the hepattn_muon directory:")
    print("   cd /shared/tracking/hepattn_muon")
    print("   pixi run python src/hepattn/experiments/atlas_muon/dev/example_baseline_evaluation.py")
    print("")
    print("3. Or run the main evaluation script directly:")
    print("   pixi run python src/hepattn/experiments/atlas_muon/evaluate_hit_filter_dataloader.py \\\\")
    print("     --eval_path /path/to/your/eval.h5 \\\\")
    print("     --data_dir /path/to/your/data \\\\")
    print("     --config_path /path/to/your/config.yaml \\\\")
    print("     --output_dir ./results \\\\")
    print("     --max_events 1000")
    print("")
    print("The evaluation will create two subdirectories:")
    print("  - all_tracks/: Results using all available tracks")
    print("  - baseline_filtered_tracks/: Results using only tracks with ≥3 stations, ≥3 hits/station")

if __name__ == "__main__":
    print("ATLAS MUON HIT FILTER EVALUATION WITH BASELINE FILTERING")
    print("=" * 70)
    
    # Check if paths exist and run evaluation, otherwise show usage
    success = run_example_evaluation()
    
    if not success:
        print_usage_instructions()