#!/usr/bin/env python3
"""
Test script for detector technology statistics functionality.
"""

from pathlib import Path
from hepattn.experiments.atlas_muon.data_vis.track_visualizer_h5_MDTGeometry import h5TrackVisualizerMDTGeometry
from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule, AtlasMuonDataset
import yaml

def test_detector_technology_stats():
    # Load configuration
    config_path = './hepattn/experiments/atlas_muon/configs/atlas_muon_event_plotting.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_cfg = config.get('data', {})
    inputs = data_cfg.get('inputs', {})
    targets = data_cfg.get('targets', {})

    # Create dataset
    H5_FILEPATH = '/shared/tracking/data/ml_test_data_156000.hdf5'
    dataset = AtlasMuonDataset(
        dirpath=H5_FILEPATH,
        inputs=inputs,
        targets=targets,
    )

    # Create data module
    datamodule = AtlasMuonDataModule(
        train_dir=H5_FILEPATH,
        val_dir=H5_FILEPATH,
        test_dir=H5_FILEPATH,
        num_workers=1,
        num_train=10,
        num_val=10,
        num_test=10,
        batch_size=1,
        inputs=inputs,
        targets=targets,
    )
    datamodule.setup('test')

    # Create visualizer and calculate statistics
    visualizer = h5TrackVisualizerMDTGeometry(dataset=dataset)
    tech_stats = visualizer.calculate_detector_technology_statistics(
        datamodule.test_dataloader(), num_events=10
    )

    print('=' * 80)
    print('DETECTOR TECHNOLOGY STATISTICS TEST')
    print('=' * 80)
    print(f"Events processed: {tech_stats['overall']['events_processed']}")
    print(f"Total hits: {tech_stats['overall']['total_hits']:,}")
    print(f"Total true hits: {tech_stats['overall']['total_true_hits']:,}")
    print()
    
    print('TECHNOLOGY DISTRIBUTION:')
    print('-' * 50)
    for tech_name in ['MDT', 'RPC', 'TGC', 'STGC', 'MM']:
        if tech_name in tech_stats:
            stats = tech_stats[tech_name]
            print(f'{tech_name}:')
            print(f'  True hits: {stats["true_hits"]:,} ({stats["true_hits_percentage"]:.2f}%)')
            print(f'  Total hits: {stats["total_hits"]:,} ({stats["total_hits_percentage"]:.2f}%)')
            print()

    # Test the file saving
    output_path = Path('./test_tech_stats_output.txt')
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETECTOR TECHNOLOGY STATISTICS\n")
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
    
    print(f"Statistics saved to: {output_path}")
    return tech_stats

if __name__ == "__main__":
    test_detector_technology_stats()
