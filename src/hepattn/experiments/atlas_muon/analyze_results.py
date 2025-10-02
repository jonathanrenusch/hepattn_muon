#!/usr/bin/env python3
"""
Script to analyze Optuna optimization results.

This script loads the optimization results from the SQLite database
and provides analysis of the hyperparameter search.
"""

import argparse
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna


def load_study_results(db_path: str, study_name: str) -> optuna.Study:
    """Load the Optuna study from the database."""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return study


def analyze_study(study: optuna.Study) -> None:
    """Analyze and display results from the study."""
    
    print(f"Study: {study.study_name}")
    print(f"Direction: {study.direction}")
    print(f"Number of trials: {len(study.trials)}")
    
    # Filter completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"Completed trials: {len(completed_trials)}")
    print(f"Failed trials: {len(failed_trials)}")
    print(f"Pruned trials: {len(pruned_trials)}")
    
    if not completed_trials:
        print("No completed trials found!")
        return
    
    # Best trial information
    print(f"\nBest trial:")
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Best value: {study.best_value:.6f}")
    print(f"  Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Statistics
    values = [t.value for t in completed_trials]
    print(f"\nObjective value statistics:")
    print(f"  Mean: {sum(values) / len(values):.6f}")
    print(f"  Std:  {pd.Series(values).std():.6f}")
    print(f"  Min:  {min(values):.6f}")
    print(f"  Max:  {max(values):.6f}")
    
    # Parameter importance
    print(f"\nParameter importance:")
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"  Could not compute parameter importance: {e}")


def create_visualizations(study: optuna.Study, output_dir: str = ".") -> None:
    """Create visualization plots for the study results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) < 2:
        print("Not enough completed trials for visualizations")
        return
    
    print(f"\nCreating visualizations in {output_path}...")
    
    try:
        # Optimization history
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("Optimization History")
        plt.savefig(output_path / "optimization_history.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Parameter importance
        if len(completed_trials) >= 5:
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title("Parameter Importance")
            plt.savefig(output_path / "parameter_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Parallel coordinate plot
        if len(completed_trials) >= 3:
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.title("Parallel Coordinate Plot")
            plt.savefig(output_path / "parallel_coordinate.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Slice plot for top parameters
        if len(completed_trials) >= 5:
            importance = optuna.importance.get_param_importances(study)
            top_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6]
            
            for param_name, _ in top_params:
                try:
                    fig = optuna.visualization.matplotlib.plot_slice(study, params=[param_name])
                    plt.title(f"Slice Plot: {param_name}")
                    plt.savefig(output_path / f"slice_{param_name}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    print(f"Could not create slice plot for {param_name}: {e}")
        
        print(f"Visualizations saved to {output_path}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")


def export_results(study: optuna.Study, output_path: str) -> None:
    """Export results to CSV file."""
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No completed trials to export")
        return
    
    # Create DataFrame with trial results
    data = []
    for trial in completed_trials:
        row = {"trial_number": trial.number, "objective_value": trial.value}
        row.update(trial.params)
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna optimization results")
    parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--study-name", type=str, required=True, help="Name of the Optuna study")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for plots")
    parser.add_argument("--export-csv", type=str, help="Export results to CSV file")
    parser.add_argument("--no-plots", action="store_true", help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"Database file not found: {args.db_path}")
        return 1
    
    try:
        # Load study
        print(f"Loading study from {args.db_path}...")
        study = load_study_results(args.db_path, args.study_name)
        
        # Analyze results
        analyze_study(study)
        
        # Create visualizations
        if not args.no_plots:
            create_visualizations(study, args.output_dir)
        
        # Export to CSV if requested
        if args.export_csv:
            export_results(study, args.export_csv)
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())