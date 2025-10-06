#!/usr/bin/env python3
"""
Script to generate ROOT file analysis plots for all configured ROOT files.
This script uses the RootAnalyzer to create hit distribution and track analysis plots
for all ROOT files defined in the configuration.
"""

import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import h5py
import numpy as np
from tqdm import tqdm
import torch

# Set matplotlib backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')

from .data_vis.h5_config import DEFAULT_TREE_NAME, H5_FILEPATH, HISTOGRAM_SETTINGS, HIT_EVAL_FILEPATH
from .data import AtlasMuonDataModule, AtlasMuonDataset

# Import from the utils module
from .data_vis.h5_analyzer import h5Analyzer
from .data_vis.track_visualizer_h5_MDTGeometry import h5TrackVisualizerMDTGeometry


def create_output_directory(base_output_dir: str) -> Path:
    """Create timestamped output directory for the plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"root_analysis_plots_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_inputs_targets_from_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config.get("data", {})
    inputs = data_cfg.get("inputs", {})
    targets = data_cfg.get("targets", {})
    return inputs, targets

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file system usage."""
    return "".join(c for c in filename if c.isalnum() or c in (" ", "-", "_")).rstrip()


def generate_plots_for_file(
    datamodule: AtlasMuonDataModule,
    config_key: str,
    output_dir: Path,
    num_events: int = 10,
    generate_histograms: bool = True,
    histogram_categories: List[str] = ["standard", "signal_background"],
    dataset: Optional[AtlasMuonDataset] = None,
) -> Optional[Dict[str, Union[List[str], int]]]:
    """Generate all plots for a single ROOT file."""
    print(f"\n{'='*80}")
    print(f"Processing: {config_key}")
    print(f"{'='*80}")

    # try:
        
    analyzer = h5Analyzer(datamodule, num_events)

    # Create subdirectory for this file
    file_output_dir = output_dir / sanitize_filename(config_key)
   # ...existing code...
    if HIT_EVAL_FILEPATH is not None:
        file_output_dir = file_output_dir.with_name(file_output_dir.name + "_FILTERED")
    # ...existing code...

    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    success = True

    # 1. Hits per event analysis
    print("\n--- Analyzing hits per event ---")
    hits_plot_path = file_output_dir / f"{config_key}_hits_distribution.png"
    hits_data = analyzer.analyze_hits_per_event(
        output_plot_path=str(hits_plot_path)
    )

    if hits_data is None:
        print("ERROR: Failed to analyze hits per event")
        success = False
    else:
        print(f"✓ Hits distribution plot saved to: {hits_plot_path}")

    # 2. Tracks and track lengths analysis
    print("\n--- Analyzing tracks and track lengths ---")
    tracks_plot_path = file_output_dir / f"{config_key}_tracks_analysis.png"
    tracks_data = analyzer.analyze_tracks_and_lengths(
        output_plot_path=str(tracks_plot_path)
    )

    if tracks_data is None:
        print("ERROR: Failed to analyze tracks and lengths")
        success = False
    else:
        print(f"✓ Tracks analysis plot saved to: {tracks_plot_path}")

    # 3. Generate branch histograms

    if generate_histograms: 
        for histogram_category in histogram_categories:
            print(f"\n--- Generating histograms (category: {histogram_category}) ---")
            analyzer.generate_feature_histograms_with_categories(
                output_dir=file_output_dir / f"histograms_{histogram_category}",
                histogram_settings=HISTOGRAM_SETTINGS,
                category=histogram_category
            )

    # 4. Plot number of true hits: 
    track_analyzer = h5TrackVisualizerMDTGeometry(
        dataset=dataset,
    )
    print("\n--- Plotting number of true hits ---")
    track_analyzer.plot_and_save_true_hits_histogram(
        dataloader=datamodule.test_dataloader(shuffle=True), 
        num_events=num_events,
        save_path=file_output_dir / f"{config_key}_true_hits_histogram.png",
    )

    # Generate 100 random event displays: 
    print("\n--- Generating random event displays ---")
    # picking 100 random numbers from the range of available events
    random_indices = np.random.choice(num_events, size=100, replace=False)
    # make new directory for random events
    (file_output_dir / "events").mkdir(parents=True, exist_ok=True)
    for idx in random_indices:
        print(f"Processing random event {idx + 1}/{num_events}")
        try:
            track_analyzer.plot_and_save_event(
                event_index=idx,
                save_path=file_output_dir / "events" / f"{config_key}_random_event_{idx}.png",
            )
            # Force matplotlib cleanup after each event
            import matplotlib.pyplot as plt
            plt.close('all')
            
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error processing event {idx}: {e}")
            continue



def main() -> None:
    """Main function to process all ROOT files."""
    parser = argparse.ArgumentParser(
        description="Generate ROOT analysis plots for all configured files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./root_analysis_output",
        help="Base directory for output plots (default: ./root_analysis_output)",
    )
    parser.add_argument(
        "--tree-name",
        type=str,
        default=DEFAULT_TREE_NAME,
        help=f"ROOT tree name to analyze (default: {DEFAULT_TREE_NAME})",
    )
    parser.add_argument(
        "--keys",
        type=str,
        nargs="+",
        help="Specific config keys to process (default: all keys)",
    )
    parser.add_argument(
        "--num-events",
        "-n",
        type=int,
        default=-1,
        # default=-1,
        help="number of events to use for statistics",
    )
    parser.add_argument(
        "--skip-histograms",
        action="store_true",
        help="Skip generation of branch histograms (default: False)",
    )
    parser.add_argument(
        "--histogram-categories",
        type=str,
        nargs="+",
        default=["standard", "signal_background"],
        choices=["standard", "signal_background"],
        help="Categories of histograms to generate (default: both 'standard' and 'signal_background')",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")

    config_path = "/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml"

    # Load inputs and targets fresh each time to avoid corruption
    inputs, targets = load_inputs_targets_from_config(config_path)

    # Doing quick metrics check: 
    if HIT_EVAL_FILEPATH is not None:
        # Create fresh copies of inputs and targets for the eval datamodule
        inputs_eval = {k: list(v) for k, v in inputs.items()}  # Deep copy
        targets_eval = {k: list(v) for k, v in targets.items()}  # Deep copy
        
        datamodule_eval = AtlasMuonDataModule(
        train_dir=H5_FILEPATH,
        val_dir=H5_FILEPATH,
        test_dir=H5_FILEPATH,
        num_workers=10,  # Use multiprocessing for better performance
        num_train=-1,
        num_val=-1,
        num_test=-1,
        batch_size=1,
        inputs=inputs_eval,
        targets=targets_eval,
    )
        datamodule_eval.setup("test")
        test_dataloader_eval = datamodule_eval.test_dataloader(shuffle=True)

        print("Running quick performance stats, since HIT_EVAL_FILEPATH is not None!")
        
        # ... your metrics collection code ...
        metrics_lists = {
            "nh_total_pre": [],
            "nh_total_post": [],
            "nh_pred_true": [],
            "nh_pred_false": [],
            "nh_valid_pre": [],
            "nh_valid_post": [],
            "nh_noise_pre": [],
            "nh_noise_post": [],
            "acc": [],
            "valid_recall": [],
            "valid_precision": [],
            "noise_recall": [],
            "noise_precision": [],
        }

        # Process metrics in smaller batches to avoid memory accumulation
        batch_size_metrics = 1000  # Process 1000 events at a time
        
        for batch_start in range(0, args.num_events, batch_size_metrics):
            batch_end = min(batch_start + batch_size_metrics, args.num_events)
            print(f"Processing metrics batch {batch_start}-{batch_end}/{args.num_events}")
            
            # Temporary storage for this batch
            batch_metrics = {
                "nh_total_pre": [],
                "nh_total_post": [],
                "nh_pred_true": [],
                "nh_pred_false": [],
                "nh_valid_pre": [],
                "nh_valid_post": [],
                "nh_noise_pre": [],
                "nh_noise_post": [],
                "acc": [],
                "valid_recall": [],
                "valid_precision": [],
                "noise_recall": [],
                "noise_precision": [],
            }

            # Pre-open the H5 file to avoid repeated opening/closing
            with h5py.File(HIT_EVAL_FILEPATH, "r") as hit_eval_file:
                for idx, batch in enumerate(tqdm(test_dataloader_eval, 
                                                total=batch_end - batch_start, 
                                                desc=f"Processing batch {batch_start//batch_size_metrics + 1}")):
                    
                    if idx < batch_start:
                        continue
                    if idx >= batch_end:
                        break
                        
                    inputs_batch, targets_batch = batch

                    pred = hit_eval_file[f"{idx}/preds/final/hit_filter/hit_on_valid_particle"][0]
                    pred = torch.from_numpy(pred).to(targets_batch["hit_on_valid_particle"].device)
                    if pred.dtype != torch.bool:
                        pred = pred.bool()

                    true = targets_batch["hit_on_valid_particle"][targets_batch["hit_valid"]]
                    tp = (pred & true).sum()
                    tn = ((~pred) & (~true)).sum()

                    event_metrics = {
                        "nh_total_pre": float(pred.numel()),
                        "nh_total_post": float(pred.sum().item()),
                        "nh_pred_true": pred.float().sum().item(),
                        "nh_pred_false": (~pred).float().sum().item(),
                        "nh_valid_pre": true.float().sum().item(),
                        "nh_valid_post": (pred & true).float().sum().item(),
                        "nh_noise_pre": (~true).float().sum().item(),
                        "nh_noise_post": (pred & ~true).float().sum().item(),
                        "acc": (pred == true).float().mean().item(),
                        "valid_recall": (tp / true.sum()).item() if true.sum() > 0 else float('nan'),
                        "valid_precision": (tp / pred.sum()).item() if pred.sum() > 0 else float('nan'),
                        "noise_recall": (tn / (~true).sum()).item() if (~true).sum() > 0 else float('nan'),
                        "noise_precision": (tn / (~pred).sum()).item() if (~pred).sum() > 0 else float('nan'),
                    }

                    for k, v in event_metrics.items():
                        batch_metrics[k].append(v)
            
            # Compute batch averages and add to global metrics
            for k, v in batch_metrics.items():
                if k not in metrics_lists:
                    metrics_lists[k] = []
                metrics_lists[k].extend(v)
            
            # Force garbage collection after each batch
            import gc
            gc.collect()

        averages = {k: float(np.nanmean(v)) for k, v in metrics_lists.items()}

        output_metrics_path = output_dir / "hit_filter_performance_metrics.txt"
        with open(output_metrics_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HIT FILTER PERFORMANCE METRICS\n")
            f.write("=" * 80 + "\n")
            for k, v in averages.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        print(f"Saved average metrics to {output_metrics_path}")

        # Calculate and save detector technology statistics for the filtered dataset
        print("\n--- Calculating detector technology statistics for filtered dataset ---")
        track_analyzer_eval = h5TrackVisualizerMDTGeometry(dataset=dataset)
        tech_stats = track_analyzer_eval.calculate_detector_technology_statistics(
            test_dataloader_eval, args.num_events
        )
        print("=" * 80)
        print(tech_stats)
        print("=" * 80) 
        if tech_stats:
            tech_stats_path = output_dir / "detector_technology_statistics_filtered.txt"
            with open(tech_stats_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("DETECTOR TECHNOLOGY STATISTICS (FILTERED DATASET)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Events processed: {tech_stats['overall']['events_processed']}\n")
                f.write(f"Total hits: {tech_stats['overall']['total_hits']:,}\n")
                f.write(f"Total true hits: {tech_stats['overall']['total_true_hits']:,}\n\n")
                
                f.write("TECHNOLOGY DISTRIBUTION (ABSOLUTE NUMBERS):\n")
                f.write("-" * 50 + "\n")
                for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
                    if tech_name in tech_stats:
                        stats = tech_stats[tech_name]
                        f.write(f"{tech_name}:\n")
                        f.write(f"  Total hits: {stats['total_hits']:,}\n")
                        f.write(f"  True hits: {stats['true_hits']:,}\n\n")
                
                f.write("TECHNOLOGY DISTRIBUTION (PERCENTAGES):\n")
                f.write("-" * 50 + "\n")
                f.write("Percentage of total hits by technology:\n")
                for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
                    if tech_name in tech_stats:
                        stats = tech_stats[tech_name]
                        f.write(f"  {tech_name}: {stats['total_hits_percentage']:.2f}%\n")
                
                f.write("\nPercentage of true hits by technology:\n")
                for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
                    if tech_name in tech_stats:
                        stats = tech_stats[tech_name]
                        f.write(f"  {tech_name}: {stats['true_hits_percentage']:.2f}%\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            print(f"Saved detector technology statistics to {tech_stats_path}")

        # IMPORTANT: Properly clean up the eval datamodule
        del test_dataloader_eval
        del datamodule_eval
        
        # Force garbage collection to ensure cleanup
        import gc
        gc.collect()

    # Load fresh inputs and targets again for the main datamodule
    inputs_main = {k: list(v) for k, v in load_inputs_targets_from_config(config_path)[0].items()}
    targets_main = {k: list(v) for k, v in load_inputs_targets_from_config(config_path)[1].items()}

    print("This is HIT_EVAL_FILEPATH", HIT_EVAL_FILEPATH)
    datamodule = AtlasMuonDataModule(
        train_dir=H5_FILEPATH,
        val_dir=H5_FILEPATH,
        test_dir=H5_FILEPATH,
        num_workers=10,  # Use multiprocessing for better performance
        num_train=-1,
        num_val=-1,
        num_test=-1,
        batch_size=1,
        inputs=inputs_main,
        targets=targets_main,
        hit_eval_train=HIT_EVAL_FILEPATH,
        hit_eval_val=HIT_EVAL_FILEPATH,
        hit_eval_test=HIT_EVAL_FILEPATH,
    )
    
    dataset = AtlasMuonDataset(
        dirpath=H5_FILEPATH,
        inputs=inputs_main,
        targets=targets_main,
        hit_eval_path=HIT_EVAL_FILEPATH,
    )
    
    datamodule.setup("test")

    file_info = generate_plots_for_file(
        datamodule,
        H5_FILEPATH.split("/")[-1],
        output_dir,
        args.num_events,
        not args.skip_histograms,
        args.histogram_categories,
        dataset=dataset,
    )

if __name__ == "__main__":
    main()