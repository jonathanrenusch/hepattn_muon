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

import numpy as np

from .data_vis.h5_config import DEFAULT_TREE_NAME, H5_FILEPATH, HISTOGRAM_SETTINGS
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

        analyzer.generate_feature_histograms(
            output_dir=file_output_dir / "histograms",
            histogram_settings=HISTOGRAM_SETTINGS
        )

    # 4. Plot number of true hits: 
    track_analyzer = h5TrackVisualizerMDTGeometry(
        dataset=dataset,
    )
    print("\n--- Plotting number of true hits ---")
    track_analyzer.plot_and_save_true_hits_histogram(
        dataloader=datamodule.test_dataloader(), 
        num_events=num_events,
        save_path=file_output_dir / f"{config_key}_true_hits_histogram.png",
    )

    # Generate 10 random event displays: 
    print("\n--- Generating random event displays ---")
    # picking 10 random numbers from the range of available events
    random_indices = np.random.choice(num_events, size=100, replace=False)
    # make new directory for random events
    (file_output_dir / "events").mkdir(parents=True, exist_ok=True)
    for idx in random_indices:
        print(f"Processing random event {idx + 1}/{num_events}")
        track_analyzer.plot_and_save_event(
            event_index=idx,
            save_path=file_output_dir / "events" / f"{config_key}_random_event_{idx}.png",
        )







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
        type=int,
        default=10000,
        help="Number of random events to display (default: 10)",
    )
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=1,
        help="Minimum number of tracks required per event (default: 1)",
    )
    parser.add_argument(
        "--skip-histograms",
        action="store_true",
        help="Skip generation of branch histograms (default: False)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")

    # Filter keys if specified
    import yaml



    # Usage example:
    config_path = "/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/atlas_muon_event_plotting.yaml"

    inputs, targets = load_inputs_targets_from_config(config_path)
    # print(f"Inputs: {inputs}")
    # print(f"Targets: {targets}")
    # print(H5_FILEPATH)
    # Now pass these to your DataModule
    datamodule = AtlasMuonDataModule(
        train_dir=H5_FILEPATH[0],
        val_dir=H5_FILEPATH[0],
        test_dir=H5_FILEPATH[0],
        num_workers=20,
        num_train=-1,
        num_val=-1,
        num_test=-1,
        batch_size=1,
        inputs=inputs,
        targets=targets,
        # ... any other kwargs
    )
    dataset = AtlasMuonDataset(
        dataset_dir=H5_FILEPATH[0],
        inputs=inputs,
        targets=targets,
    )
    datamodule.setup("test")
    # # Initialize datamodule
    # datamodule = AtlasMuonDataModule(
    #     train_dir=H5_FILEPATH,
    #     val_dir=H5_FILEPATH,
    #     num_workers=10,
    #     num_train=-1,
    #     num_val=-1,
    #     num_test=-1,
    #     batch_size=1,
    # )

    file_info = generate_plots_for_file(
        datamodule,
        H5_FILEPATH[0].split("/")[-1],
        output_dir,
        args.num_events,
        not args.skip_histograms,  # generate_histograms
        dataset=dataset,
    )

if __name__ == "__main__":
    main()
