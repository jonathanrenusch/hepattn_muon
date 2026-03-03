#!/usr/bin/env python3
# ruff: noqa: TID252
"""Download ttbar pileup-zero dataset from ColliderML.

This module provides utilities to download the ttbar_pu0 sample from
the ColliderML Release 1 dataset on HuggingFace. It downloads three
configurations:
- ttbar_pu0_particles: Truth-level particle information
- ttbar_pu0_tracker_hits: Detector-level tracker measurements
- ttbar_pu0_tracks: ACTS reconstructed tracks

Dataset: https://huggingface.co/datasets/CERN/ColliderML-Release-1
Paper: https://arxiv.org/pdf/2512.15230
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# Default output directory for the dataset
DEFAULT_OUTPUT_DIR = Path("/eos/project/e/end-to-end-muon-tracking/tracking/colliderml/p0")

# Configuration names for ttbar pileup-zero sample
TTBAR_PU0_CONFIGS = [
    "ttbar_pu0_particles",
    "ttbar_pu0_tracker_hits",
    "ttbar_pu0_tracks",
]


def download_ttbar_pu0_dataset(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_events: int | None = None,
    force: bool = False,
    configs: list[str] | None = None,
) -> dict[str, Path]:
    """Download ttbar pileup-zero dataset from ColliderML.

    Parameters
    ----------
    output_dir : Path | str, optional
        Output directory for downloads. Default is the EOS project directory.
    max_events : int, optional
        Maximum number of events to download. If None, downloads all available.
    force : bool, optional
        Force re-download even if files exist locally.
    configs : list[str], optional
        Specific configurations to download. If None, downloads all three
        ttbar_pu0 configurations (particles, tracker_hits, tracks).

    Returns
    -------
    dict[str, Path]
        Dictionary mapping configuration names to their download paths.

    Raises
    ------
    RuntimeError
        If download fails for any configuration.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs_to_download = configs if configs else TTBAR_PU0_CONFIGS
    downloaded_paths = {}

    print(f"Downloading ColliderML ttbar_pu0 dataset to: {output_path}")
    print(f"Configurations: {configs_to_download}")

    for config in configs_to_download:
        print(f"\n{'='*60}")
        print(f"Downloading: {config}")
        print(f"{'='*60}")

        cmd = [
            "colliderml",
            "download",
            "--config", config,
            "--out", str(output_path),
        ]

        if max_events:
            cmd.extend(["--max-events", str(max_events)])

        if force:
            cmd.append("--force")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True,
            )
            downloaded_paths[config] = output_path / config
            print(f"Successfully downloaded: {config}")

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed to download {config}: {e}"
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")
    for config, path in downloaded_paths.items():
        print(f"  - {config}: {path}")

    return downloaded_paths


def download_with_huggingface(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    configs: list[str] | None = None,
    max_events: int | None = None,
) -> dict[str, Path]:
    """Alternative download using HuggingFace datasets library directly.

    This method downloads and saves the data as Parquet files directly
    using the HuggingFace datasets library, bypassing the colliderml CLI.

    Parameters
    ----------
    output_dir : Path | str, optional
        Output directory for downloads.
    configs : list[str], optional
        Specific configurations to download.
    max_events : int, optional
        Maximum number of events to download (limits the split).

    Returns
    -------
    dict[str, Path]
        Dictionary mapping configuration names to their download paths.
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs_to_download = configs if configs else TTBAR_PU0_CONFIGS
    downloaded_paths = {}

    print(f"Downloading ColliderML ttbar_pu0 dataset via HuggingFace to: {output_path}")

    for config in configs_to_download:
        print(f"\nDownloading: {config}")

        # Determine split based on max_events
        split = f"train[:{max_events}]" if max_events else "train"

        # Load dataset from HuggingFace
        dataset = load_dataset(
            "CERN/ColliderML-Release-1",
            config,
            split=split,
        )

        # Save to parquet
        config_path = output_path / config
        config_path.mkdir(parents=True, exist_ok=True)
        parquet_file = config_path / "data.parquet"

        dataset.to_parquet(str(parquet_file))
        downloaded_paths[config] = config_path

        print(f"  Saved {len(dataset)} events to {parquet_file}")

    print("\nDownload complete!")
    return downloaded_paths


def main():
    """Command-line interface for downloading ttbar_pu0 dataset."""
    parser = argparse.ArgumentParser(
        description="Download ttbar pileup-zero dataset from ColliderML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for downloads",
    )
    parser.add_argument(
        "--max-events", "-n",
        type=int,
        default=None,
        help="Maximum number of events to download (downloads all if not set)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--configs", "-c",
        type=str,
        default=None,
        help="Comma-separated list of configs to download (default: all three)",
    )
    parser.add_argument(
        "--method",
        choices=["cli", "huggingface"],
        default="cli",
        help="Download method: 'cli' uses colliderml CLI, 'huggingface' uses datasets library directly",
    )

    args = parser.parse_args()

    configs = args.configs.split(",") if args.configs else None

    if args.method == "cli":
        download_ttbar_pu0_dataset(
            output_dir=args.output_dir,
            max_events=args.max_events,
            force=args.force,
            configs=configs,
        )
    else:
        download_with_huggingface(
            output_dir=args.output_dir,
            configs=configs,
            max_events=args.max_events,
        )


if __name__ == "__main__":
    main()
