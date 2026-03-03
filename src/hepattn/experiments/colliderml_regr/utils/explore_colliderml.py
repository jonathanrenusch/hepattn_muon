#!/usr/bin/env python3
# ruff: noqa: TID252, PLR0912, PLR0915, C901
"""ColliderML Data Exploration Script.

This script provides comprehensive data exploration for the ColliderML ttbar_pu0 dataset,
generating feature distributions, statistics, event displays, and occupancy heatmaps.

Usage:
    python explore_colliderml.py --config exploration_config.yaml
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class ColliderMLExplorer:
    """Comprehensive data exploration for ColliderML dataset."""

    def __init__(self, config_path: str | Path) -> None:
        """Initialize explorer with configuration.

        Parameters
        ----------
        config_path : str | Path
            Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config["data"]["data_dir"])
        self.output_dir = Path(self.config["data"]["output_dir"])
        self.num_events = self.config["data"]["num_events"]
        self.random_seed = self.config["data"]["random_seed"]

        # Set random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Statistics accumulator
        self.statistics: dict[str, Any] = {}

        # Create output directories
        self._create_output_dirs()

    def _load_config(self, config_path: str | Path) -> dict:
        """Load YAML configuration file."""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _create_output_dirs(self) -> None:
        """Create output directory structure."""
        dirs = [
            self.output_dir,
            self.output_dir / "hit_features",
            self.output_dir / "particle_features",
            self.output_dir / "track_features",
            self.output_dir / "event_statistics",
            self.output_dir / "event_displays",
            self.output_dir / "occupancy_heatmaps",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load particles, tracker_hits, and tracks data.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            Particles, tracker_hits, and tracks DataFrames.
        """
        print("Loading ColliderML data...")

        # Load parquet files
        particles_dir = self.data_dir / "ttbar_pu0_particles"
        hits_dir = self.data_dir / "ttbar_pu0_tracker_hits"
        tracks_dir = self.data_dir / "ttbar_pu0_tracks"

        # Use lazy loading and scan all parquet files
        particles = pl.scan_parquet(particles_dir / "**/*.parquet")
        hits = pl.scan_parquet(hits_dir / "**/*.parquet")
        tracks = pl.scan_parquet(tracks_dir / "**/*.parquet")

        # Get unique event IDs and sample
        print("Sampling events...")
        all_event_ids = particles.select("event_id").unique().collect()["event_id"].to_list()
        print(f"Total events available: {len(all_event_ids)}")

        if self.num_events > 0 and self.num_events < len(all_event_ids):
            sampled_ids = random.sample(all_event_ids, self.num_events)
        else:
            sampled_ids = all_event_ids
            self.num_events = len(all_event_ids)

        print(f"Using {len(sampled_ids)} events for analysis")

        # Filter to sampled events and collect
        print("Collecting particles...")
        particles_df = particles.filter(pl.col("event_id").is_in(sampled_ids)).collect()
        print(f"  Loaded {len(particles_df)} particle rows")

        print("Collecting tracker hits...")
        hits_df = hits.filter(pl.col("event_id").is_in(sampled_ids)).collect()
        print(f"  Loaded {len(hits_df)} hit rows")

        print("Collecting tracks...")
        tracks_df = tracks.filter(pl.col("event_id").is_in(sampled_ids)).collect()
        print(f"  Loaded {len(tracks_df)} track rows")

        return particles_df, hits_df, tracks_df

    def _add_stats_to_plot(
        self,
        ax: plt.Axes,
        data: np.ndarray,
        position: str = "upper right",
    ) -> tuple[float, float]:
        """Add mean and std statistics to plot.

        Returns
        -------
        tuple[float, float]
            Mean and standard deviation.
        """
        mean = np.nanmean(data)
        std = np.nanstd(data)

        stats_text = f"μ = {mean:.4g}\nσ = {std:.4g}\nN = {len(data)}"

        # Position mapping
        pos_map = {
            "upper right": (0.95, 0.95),
            "upper left": (0.05, 0.95),
            "lower right": (0.95, 0.05),
            "lower left": (0.05, 0.05),
        }
        ha_map = {"upper right": "right", "upper left": "left", "lower right": "right", "lower left": "left"}
        va_map = {"upper right": "top", "upper left": "top", "lower right": "bottom", "lower left": "bottom"}

        x, y = pos_map.get(position, (0.95, 0.95))
        ha = ha_map.get(position, "right")
        va = va_map.get(position, "top")

        ax.text(
            x,
            y,
            stats_text,
            transform=ax.transAxes,
            fontsize=self.config["plot"]["stats_fontsize"],
            verticalalignment=va,
            horizontalalignment=ha,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        return mean, std

    def _get_bins(self, feature_config: dict, data: np.ndarray) -> int | np.ndarray:
        """Get appropriate bins for histogram."""
        bins_setting = feature_config.get("bins", self.config["histograms"]["default_bins"])

        if bins_setting == "integer":
            # Integer binning
            min_val = int(np.floor(np.nanmin(data)))
            max_val = int(np.ceil(np.nanmax(data)))
            return np.arange(min_val - 0.5, max_val + 1.5, 1)
        else:
            return bins_setting

    def _plot_histogram(
        self,
        data: np.ndarray,
        feature_name: str,
        feature_config: dict,
        save_path: Path,
        title: str | None = None,
        color: str | None = None,
    ) -> dict:
        """Plot a single histogram and return statistics."""
        fig, ax = plt.subplots(figsize=tuple(self.config["plot"]["figsize"]))

        # Filter out NaN values
        valid_data = data[~np.isnan(data)]

        if len(valid_data) == 0:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            fig.savefig(save_path, dpi=self.config["plot"]["dpi"], bbox_inches="tight")
            plt.close(fig)
            return {"mean": np.nan, "std": np.nan, "count": 0}

        # Get bins
        bins = self._get_bins(feature_config, valid_data)
        hist_range = feature_config.get("range")

        # Plot histogram
        if color is None:
            color = self.config["plot"]["true_hit_color"]

        ax.hist(
            valid_data,
            bins=bins,
            range=hist_range,
            alpha=self.config["histograms"]["fill_alpha"],
            color=color,
            edgecolor=self.config["histograms"]["edge_color"],
            linewidth=self.config["histograms"]["edge_linewidth"],
        )

        # Add statistics
        mean, std = self._add_stats_to_plot(ax, valid_data, self.config["plot"]["stats_position"])

        # Labels
        label = feature_config.get("label", feature_name)
        ax.set_xlabel(label, fontsize=self.config["plot"]["label_fontsize"])
        ax.set_ylabel("Count", fontsize=self.config["plot"]["label_fontsize"])
        if title:
            ax.set_title(title, fontsize=self.config["plot"]["title_fontsize"])
        else:
            ax.set_title(f"Distribution of {label}", fontsize=self.config["plot"]["title_fontsize"])

        ax.tick_params(labelsize=self.config["plot"]["tick_fontsize"])
        ax.grid(True, alpha=0.3)

        fig.savefig(save_path, dpi=self.config["plot"]["dpi"], bbox_inches="tight")
        plt.close(fig)

        return {"mean": mean, "std": std, "count": len(valid_data), "min": np.min(valid_data), "max": np.max(valid_data)}

    def _plot_side_by_side_histograms(
        self,
        data_left: np.ndarray,
        data_right: np.ndarray,
        feature_name: str,
        feature_config: dict,
        save_path: Path,
        title_left: str,
        title_right: str,
    ) -> tuple[dict, dict]:
        """Plot two histograms side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        label = feature_config.get("label", feature_name)

        stats_left = {"mean": np.nan, "std": np.nan, "count": 0}
        stats_right = {"mean": np.nan, "std": np.nan, "count": 0}

        # Left histogram (true hits)
        valid_left = data_left[~np.isnan(data_left)]
        if len(valid_left) > 0:
            bins = self._get_bins(feature_config, valid_left)
            hist_range = feature_config.get("range")

            ax1.hist(
                valid_left,
                bins=bins,
                range=hist_range,
                alpha=self.config["histograms"]["fill_alpha"],
                color=self.config["plot"]["true_hit_color"],
                edgecolor=self.config["histograms"]["edge_color"],
                linewidth=self.config["histograms"]["edge_linewidth"],
            )
            mean, std = self._add_stats_to_plot(ax1, valid_left, self.config["plot"]["stats_position"])
            stats_left = {"mean": mean, "std": std, "count": len(valid_left)}
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax1.transAxes)

        ax1.set_xlabel(label, fontsize=self.config["plot"]["label_fontsize"])
        ax1.set_ylabel("Count", fontsize=self.config["plot"]["label_fontsize"])
        ax1.set_title(title_left, fontsize=self.config["plot"]["title_fontsize"])
        ax1.grid(True, alpha=0.3)

        # Right histogram (noise hits)
        valid_right = data_right[~np.isnan(data_right)]
        if len(valid_right) > 0:
            # Use same bins as left for comparison
            if len(valid_left) > 0:
                bins = self._get_bins(feature_config, valid_left)
            else:
                bins = self._get_bins(feature_config, valid_right)
            hist_range = feature_config.get("range")

            ax2.hist(
                valid_right,
                bins=bins,
                range=hist_range,
                alpha=self.config["histograms"]["fill_alpha"],
                color=self.config["plot"]["noise_hit_color"],
                edgecolor=self.config["histograms"]["edge_color"],
                linewidth=self.config["histograms"]["edge_linewidth"],
            )
            mean, std = self._add_stats_to_plot(ax2, valid_right, self.config["plot"]["stats_position"])
            stats_right = {"mean": mean, "std": std, "count": len(valid_right)}
        else:
            ax2.text(0.5, 0.5, "No noise hits detected", ha="center", va="center", transform=ax2.transAxes, fontsize=12)

        ax2.set_xlabel(label, fontsize=self.config["plot"]["label_fontsize"])
        ax2.set_ylabel("Count", fontsize=self.config["plot"]["label_fontsize"])
        ax2.set_title(title_right, fontsize=self.config["plot"]["title_fontsize"])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=self.config["plot"]["dpi"], bbox_inches="tight")
        plt.close(fig)

        return stats_left, stats_right

    def explore_hit_features(self, hits_df: pl.DataFrame, particles_df: pl.DataFrame) -> dict:
        """Explore and plot hit feature distributions."""
        print("\n" + "=" * 60)
        print("Exploring Hit Features")
        print("=" * 60)

        stats = {}
        hit_config = self.config["histograms"]["hit_features"]

        # Explode list columns to get individual hits
        print("Exploding hit data...")

        # Get available columns
        available_cols = hits_df.columns
        print(f"Available hit columns: {available_cols}")

        # Identify true vs noise hits
        # A hit is "true" if its particle_id matches a valid particle
        # First, get all valid particle IDs
        print("Identifying true vs noise hits...")

        # Explode hits to individual rows
        list_cols = [c for c in available_cols if c != "event_id" and hits_df[c].dtype == pl.List]
        if list_cols:
            exploded_hits = hits_df.explode(list_cols)
        else:
            exploded_hits = hits_df

        # Get particle IDs from particles
        particle_list_cols = [c for c in particles_df.columns if c != "event_id" and particles_df[c].dtype == pl.List]
        if particle_list_cols:
            exploded_particles = particles_df.explode(particle_list_cols)
        else:
            exploded_particles = particles_df

        # Get valid particle IDs (primary particles)
        if "primary" in exploded_particles.columns:
            valid_particle_ids = set(
                exploded_particles.filter(pl.col("primary") == True).select("particle_id")["particle_id"].to_list()  # noqa: E712
            )
        elif "particle_id" in exploded_particles.columns:
            valid_particle_ids = set(exploded_particles.select("particle_id")["particle_id"].to_list())
        else:
            valid_particle_ids = set()

        print(f"Found {len(valid_particle_ids)} valid particles")

        # Separate true and noise hits
        if "particle_id" in exploded_hits.columns:
            true_hits = exploded_hits.filter(pl.col("particle_id").is_in(list(valid_particle_ids)))
            noise_hits = exploded_hits.filter(~pl.col("particle_id").is_in(list(valid_particle_ids)))
        else:
            true_hits = exploded_hits
            noise_hits = pl.DataFrame()

        print(f"True hits: {len(true_hits)}, Noise hits: {len(noise_hits)}")

        # Plot each feature
        for feature_name, feature_cfg in tqdm(hit_config.items(), desc="Plotting hit features"):
            if feature_name not in exploded_hits.columns:
                print(f"  Skipping {feature_name} - not in data")
                continue

            # Get data
            true_data = true_hits[feature_name].to_numpy().astype(float)
            noise_data = noise_hits[feature_name].to_numpy().astype(float) if len(noise_hits) > 0 else np.array([])

            # Plot side by side
            save_path = self.output_dir / "hit_features" / f"{feature_name}_true_vs_noise.png"
            true_stats, noise_stats = self._plot_side_by_side_histograms(
                true_data,
                noise_data,
                feature_name,
                feature_cfg,
                save_path,
                f"True Hits: {feature_name}",
                f"Noise Hits: {feature_name}",
            )

            stats[feature_name] = {"true": true_stats, "noise": noise_stats}

        return stats

    def explore_particle_features(self, particles_df: pl.DataFrame) -> dict:
        """Explore and plot particle feature distributions."""
        print("\n" + "=" * 60)
        print("Exploring Particle Features")
        print("=" * 60)

        stats = {}
        particle_config = self.config["histograms"]["particle_features"]

        # Explode list columns
        list_cols = [c for c in particles_df.columns if c != "event_id" and particles_df[c].dtype == pl.List]
        if list_cols:
            exploded = particles_df.explode(list_cols)
        else:
            exploded = particles_df

        print(f"Available particle columns: {exploded.columns}")
        print(f"Total particles: {len(exploded)}")

        # Calculate derived quantities
        print("Calculating derived quantities...")
        if all(c in exploded.columns for c in ["px", "py"]):
            exploded = exploded.with_columns((pl.col("px") ** 2 + pl.col("py") ** 2).sqrt().alias("pt"))

        if all(c in exploded.columns for c in ["px", "py", "pz"]):
            # eta = -ln(tan(theta/2)), theta = arctan2(pt, pz)
            exploded = exploded.with_columns(
                (
                    -((((pl.col("px") ** 2 + pl.col("py") ** 2).sqrt() / pl.col("pz")).arctan() / 2).tan().log())
                ).alias("eta_calc")
            )
            exploded = exploded.with_columns((pl.col("py").arctan2(pl.col("px"))).alias("phi_calc"))

        if all(c in exploded.columns for c in ["charge", "pt"]):
            exploded = exploded.with_columns((pl.col("charge") / pl.col("pt")).alias("q_over_pt"))

        # Plot each feature
        for feature_name, feature_cfg in tqdm(particle_config.items(), desc="Plotting particle features"):
            # Handle calculated features
            actual_col = feature_name
            if feature_name == "eta" and "eta_calc" in exploded.columns:
                actual_col = "eta_calc"
            elif feature_name == "phi" and "phi_calc" in exploded.columns:
                actual_col = "phi_calc"

            if actual_col not in exploded.columns:
                print(f"  Skipping {feature_name} - not in data")
                continue

            data = exploded[actual_col].to_numpy().astype(float)
            save_path = self.output_dir / "particle_features" / f"{feature_name}_distribution.png"
            feature_stats = self._plot_histogram(data, feature_name, feature_cfg, save_path)
            stats[feature_name] = feature_stats

        # PDG ID distribution (special handling)
        if "pdg_id" in exploded.columns:
            self._plot_pdg_distribution(exploded, stats)

        return stats

    def _plot_pdg_distribution(self, particles: pl.DataFrame, stats: dict) -> None:
        """Plot PDG ID distribution with particle names."""
        pdg_ids = particles["pdg_id"].to_numpy()
        pdg_names = self.config.get("pdg_names", {})

        # Count occurrences
        unique, counts = np.unique(pdg_ids, return_counts=True)

        # Sort by count
        sort_idx = np.argsort(counts)[::-1]
        unique = unique[sort_idx]
        counts = counts[sort_idx]

        # Take top 20 for readability
        top_n = 20
        unique = unique[:top_n]
        counts = counts[:top_n]

        # Create labels
        labels = [pdg_names.get(int(pid), str(int(pid))) for pid in unique]

        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.bar(range(len(unique)), counts, color=self.config["plot"]["true_hit_color"])
        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Particle Type", fontsize=self.config["plot"]["label_fontsize"])
        ax.set_ylabel("Count", fontsize=self.config["plot"]["label_fontsize"])
        ax.set_title(f"Particle Type Distribution (Top {top_n})", fontsize=self.config["plot"]["title_fontsize"])
        ax.grid(True, alpha=0.3, axis="y")

        # Add count labels on bars
        for bar, count in zip(bars, counts, strict=False):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "particle_features" / "pdg_distribution.png",
            dpi=self.config["plot"]["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)

        stats["pdg_distribution"] = {"top_particles": list(zip(labels, counts.tolist(), strict=False))}

    def explore_track_features(self, tracks_df: pl.DataFrame) -> dict:
        """Explore and plot track feature distributions."""
        print("\n" + "=" * 60)
        print("Exploring Track Features")
        print("=" * 60)

        stats = {}
        track_config = self.config["histograms"]["track_features"]

        # Explode list columns
        list_cols = [c for c in tracks_df.columns if c != "event_id" and tracks_df[c].dtype == pl.List]
        if list_cols:
            exploded = tracks_df.explode(list_cols)
        else:
            exploded = tracks_df

        print(f"Available track columns: {exploded.columns}")
        print(f"Total tracks: {len(exploded)}")

        # Calculate derived quantities
        if "qop" in exploded.columns and "theta" in exploded.columns:
            # pt = |1/qop| * sin(theta)
            exploded = exploded.with_columns(((1 / pl.col("qop")).abs() * pl.col("theta").sin()).alias("pt_calc"))
            # eta = -ln(tan(theta/2))
            exploded = exploded.with_columns((-(pl.col("theta") / 2).tan().log()).alias("eta_calc"))

        # Plot each feature
        for feature_name, feature_cfg in tqdm(track_config.items(), desc="Plotting track features"):
            actual_col = feature_name
            if feature_name == "pt" and "pt_calc" in exploded.columns:
                actual_col = "pt_calc"
            elif feature_name == "eta" and "eta_calc" in exploded.columns:
                actual_col = "eta_calc"

            if actual_col not in exploded.columns:
                print(f"  Skipping {feature_name} - not in data")
                continue

            data = exploded[actual_col].to_numpy().astype(float)
            save_path = self.output_dir / "track_features" / f"{feature_name}_distribution.png"
            feature_stats = self._plot_histogram(data, feature_name, feature_cfg, save_path)
            stats[feature_name] = feature_stats

        return stats

    def explore_event_statistics(
        self, particles_df: pl.DataFrame, hits_df: pl.DataFrame, tracks_df: pl.DataFrame
    ) -> dict:
        """Plot event-level statistics."""
        print("\n" + "=" * 60)
        print("Exploring Event Statistics")
        print("=" * 60)

        stats = {}

        # Tracks per event
        print("Counting tracks per event...")
        if "track_id" in tracks_df.columns:
            # Get list lengths
            tracks_per_event = tracks_df.with_columns(pl.col("track_id").list.len().alias("n_tracks"))["n_tracks"].to_numpy()
        else:
            tracks_per_event = np.array([len(tracks_df.filter(pl.col("event_id") == eid)) for eid in tracks_df["event_id"].unique().to_list()])

        save_path = self.output_dir / "event_statistics" / "tracks_per_event.png"
        stats["tracks_per_event"] = self._plot_histogram(
            tracks_per_event,
            "tracks_per_event",
            {"bins": "integer", "label": "Number of Tracks"},
            save_path,
            title="Number of Reconstructed Tracks per Event",
        )

        # Hits per event
        print("Counting hits per event...")
        if "x" in hits_df.columns and hits_df["x"].dtype == pl.List:
            hits_per_event = hits_df.with_columns(pl.col("x").list.len().alias("n_hits"))["n_hits"].to_numpy()
        else:
            hits_per_event = np.array([len(hits_df.filter(pl.col("event_id") == eid)) for eid in hits_df["event_id"].unique().to_list()])

        save_path = self.output_dir / "event_statistics" / "hits_per_event.png"
        stats["hits_per_event"] = self._plot_histogram(
            hits_per_event,
            "hits_per_event",
            {"bins": 100, "label": "Number of Hits"},
            save_path,
            title="Number of Tracker Hits per Event",
        )

        # Hits per track
        print("Counting hits per track...")
        if "hit_ids" in tracks_df.columns:
            # Explode tracks first
            exploded_tracks = tracks_df.explode([c for c in tracks_df.columns if c != "event_id" and tracks_df[c].dtype == pl.List])
            if "hit_ids" in exploded_tracks.columns and exploded_tracks["hit_ids"].dtype == pl.List:
                hits_per_track = exploded_tracks.with_columns(pl.col("hit_ids").list.len().alias("n_hits"))["n_hits"].to_numpy()
            else:
                hits_per_track = np.array([])
        else:
            hits_per_track = np.array([])

        if len(hits_per_track) > 0:
            save_path = self.output_dir / "event_statistics" / "hits_per_track.png"
            stats["hits_per_track"] = self._plot_histogram(
                hits_per_track,
                "hits_per_track",
                {"bins": "integer", "label": "Number of Hits per Track"},
                save_path,
                title="Number of Hits per Reconstructed Track",
            )

        # Pixel vs Strip detector hits
        print("Counting pixel vs strip hits...")
        self._plot_detector_type_distribution(hits_df, stats)

        return stats

    def _plot_detector_type_distribution(self, hits_df: pl.DataFrame, stats: dict) -> None:
        """Plot pixel vs strip detector hit distribution."""
        if "detector" not in hits_df.columns:
            print("  Skipping detector type distribution - 'detector' column not found")
            return

        # Explode to get individual hits
        list_cols = [c for c in hits_df.columns if c != "event_id" and hits_df[c].dtype == pl.List]
        if list_cols:
            exploded = hits_df.explode(list_cols)
        else:
            exploded = hits_df

        detector_codes = self.config.get("detector_codes", {"pixel": [0, 1, 2, 3], "strip": [4, 5, 6, 7, 8, 9]})

        detector_vals = exploded["detector"].to_numpy()

        pixel_mask = np.isin(detector_vals, detector_codes["pixel"])
        strip_mask = np.isin(detector_vals, detector_codes["strip"])

        pixel_count = np.sum(pixel_mask)
        strip_count = np.sum(strip_mask)
        other_count = len(detector_vals) - pixel_count - strip_count

        print(f"  Pixel hits: {pixel_count}, Strip hits: {strip_count}, Other: {other_count}")

        # Create side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pixel hits by layer
        pixel_layers = detector_vals[pixel_mask]
        if len(pixel_layers) > 0:
            unique_layers, layer_counts = np.unique(pixel_layers, return_counts=True)
            ax1.bar(unique_layers, layer_counts, color=self.config["plot"]["true_hit_color"])
            ax1.set_xlabel("Detector Layer", fontsize=self.config["plot"]["label_fontsize"])
            ax1.set_ylabel("Count", fontsize=self.config["plot"]["label_fontsize"])
            ax1.text(
                0.95,
                0.95,
                f"Total: {pixel_count:,}",
                transform=ax1.transAxes,
                fontsize=self.config["plot"]["stats_fontsize"],
                va="top",
                ha="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        else:
            ax1.text(0.5, 0.5, "No pixel hits", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Pixel Detector Hits by Layer", fontsize=self.config["plot"]["title_fontsize"])
        ax1.grid(True, alpha=0.3)

        # Strip hits by layer
        strip_layers = detector_vals[strip_mask]
        if len(strip_layers) > 0:
            unique_layers, layer_counts = np.unique(strip_layers, return_counts=True)
            ax2.bar(unique_layers, layer_counts, color=self.config["plot"]["noise_hit_color"])
            ax2.set_xlabel("Detector Layer", fontsize=self.config["plot"]["label_fontsize"])
            ax2.set_ylabel("Count", fontsize=self.config["plot"]["label_fontsize"])
            ax2.text(
                0.95,
                0.95,
                f"Total: {strip_count:,}",
                transform=ax2.transAxes,
                fontsize=self.config["plot"]["stats_fontsize"],
                va="top",
                ha="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        else:
            ax2.text(0.5, 0.5, "No strip hits", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Strip Detector Hits by Layer", fontsize=self.config["plot"]["title_fontsize"])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "event_statistics" / "pixel_vs_strip_hits.png",
            dpi=self.config["plot"]["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)

        stats["detector_types"] = {"pixel": int(pixel_count), "strip": int(strip_count), "other": int(other_count)}

    def create_event_displays(
        self, hits_df: pl.DataFrame, particles_df: pl.DataFrame, tracks_df: pl.DataFrame
    ) -> None:
        """Create event display visualizations."""
        print("\n" + "=" * 60)
        print("Creating Event Displays")
        print("=" * 60)

        num_displays = self.config["event_display"]["num_displays"]
        event_ids = hits_df["event_id"].unique().to_list()

        if len(event_ids) < num_displays:
            num_displays = len(event_ids)

        selected_events = random.sample(event_ids, num_displays)

        for i, event_id in enumerate(tqdm(selected_events, desc="Creating event displays")):
            self._plot_single_event_display(event_id, hits_df, particles_df, tracks_df, i)

    def _plot_single_event_display(
        self,
        event_id: int,
        hits_df: pl.DataFrame,
        particles_df: pl.DataFrame,
        tracks_df: pl.DataFrame,
        display_idx: int,
    ) -> None:
        """Create event display for a single event."""
        cfg = self.config["event_display"]

        # Get hits for this event
        event_hits = hits_df.filter(pl.col("event_id") == event_id)

        # Explode to individual hits
        list_cols = [c for c in event_hits.columns if c != "event_id" and event_hits[c].dtype == pl.List]
        if list_cols:
            hits = event_hits.explode(list_cols)
        else:
            hits = event_hits

        if len(hits) == 0:
            return

        # Get coordinates
        x = hits["x"].to_numpy() if "x" in hits.columns else np.array([])
        y = hits["y"].to_numpy() if "y" in hits.columns else np.array([])
        z = hits["z"].to_numpy() if "z" in hits.columns else np.array([])

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            return

        # Get particle association
        if "particle_id" in hits.columns:
            hit_particle_ids = hits["particle_id"].to_numpy()
        else:
            hit_particle_ids = np.zeros(len(x))

        # Get valid particle IDs
        event_particles = particles_df.filter(pl.col("event_id") == event_id)
        if len(event_particles) > 0:
            particle_list_cols = [c for c in event_particles.columns if c != "event_id" and event_particles[c].dtype == pl.List]
            if particle_list_cols:
                particles = event_particles.explode(particle_list_cols)
            else:
                particles = event_particles

            if "particle_id" in particles.columns:
                valid_particle_ids = set(particles["particle_id"].to_list())
            else:
                valid_particle_ids = set()
        else:
            valid_particle_ids = set()

        # Classify hits
        is_true_hit = np.isin(hit_particle_ids, list(valid_particle_ids))

        # Create figure with pyramid layout
        fig = plt.figure(figsize=tuple(cfg["figsize"]))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1], hspace=0.25, wspace=0.35)

        ax_xy = fig.add_subplot(gs[0, 1:3])
        ax_zy = fig.add_subplot(gs[1, 0:2])
        ax_zx = fig.add_subplot(gs[1, 2:4])

        # Plot background (noise) hits
        bg_mask = ~is_true_hit
        if np.sum(bg_mask) > 0:
            ax_xy.scatter(
                x[bg_mask],
                y[bg_mask],
                s=cfg["marker_size"],
                c=self.config["plot"]["background_color"],
                alpha=self.config["plot"]["background_alpha"],
                label="Background",
            )
            ax_zy.scatter(
                z[bg_mask],
                y[bg_mask],
                s=cfg["marker_size"],
                c=self.config["plot"]["background_color"],
                alpha=self.config["plot"]["background_alpha"],
            )
            ax_zx.scatter(
                z[bg_mask],
                x[bg_mask],
                s=cfg["marker_size"],
                c=self.config["plot"]["background_color"],
                alpha=self.config["plot"]["background_alpha"],
            )

        # Plot true hits colored by particle
        unique_particles = np.unique(hit_particle_ids[is_true_hit])
        track_colors = self.config["plot"]["track_colors"]

        for i, pid in enumerate(unique_particles):
            mask = (hit_particle_ids == pid) & is_true_hit
            color = track_colors[i % len(track_colors)]
            ax_xy.scatter(x[mask], y[mask], s=cfg["marker_size"] * 2, c=color, alpha=0.8, label=f"Particle {int(pid)}")
            ax_zy.scatter(z[mask], y[mask], s=cfg["marker_size"] * 2, c=color, alpha=0.8)
            ax_zx.scatter(z[mask], x[mask], s=cfg["marker_size"] * 2, c=color, alpha=0.8)

        # Format axes
        ax_xy.set_xlabel("X [mm]")
        ax_xy.set_ylabel("Y [mm]")
        ax_xy.set_title(f"X-Y Plane (Event {event_id})")
        ax_xy.grid(True, alpha=0.3)
        ax_xy.set_aspect("equal", adjustable="box")

        ax_zy.set_xlabel("Z [mm]")
        ax_zy.set_ylabel("Y [mm]")
        ax_zy.set_title(f"Z-Y Plane (Event {event_id})")
        ax_zy.grid(True, alpha=0.3)

        ax_zx.set_xlabel("Z [mm]")
        ax_zx.set_ylabel("X [mm]")
        ax_zx.set_title(f"Z-X Plane (Event {event_id})")
        ax_zx.grid(True, alpha=0.3)

        # Add legend
        handles, labels = ax_xy.get_legend_handles_labels()
        if len(handles) > 0 and len(handles) <= 10:
            fig.legend(
                handles[:10],
                labels[:10],
                loc="upper right",
                bbox_to_anchor=(0.98, 0.98),
                fontsize=8,
            )

        # Stats
        fig.text(
            0.02,
            0.02,
            f"Total hits: {len(x)} | True: {np.sum(is_true_hit)} | Background: {np.sum(bg_mask)} | Particles: {len(unique_particles)}",
            fontsize=10,
        )

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "event_displays" / f"event_display_{display_idx:03d}_id{event_id}.png",
            dpi=self.config["plot"]["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)

    def create_occupancy_heatmaps(self, hits_df: pl.DataFrame) -> None:
        """Create detector occupancy heatmaps."""
        print("\n" + "=" * 60)
        print("Creating Occupancy Heatmaps")
        print("=" * 60)

        cfg = self.config["occupancy"]

        # Explode to individual hits
        list_cols = [c for c in hits_df.columns if c != "event_id" and hits_df[c].dtype == pl.List]
        if list_cols:
            hits = hits_df.explode(list_cols)
        else:
            hits = hits_df

        x = hits["x"].to_numpy() if "x" in hits.columns else np.array([])
        y = hits["y"].to_numpy() if "y" in hits.columns else np.array([])
        z = hits["z"].to_numpy() if "z" in hits.columns else np.array([])

        if len(x) == 0:
            print("No hit data available for occupancy plots")
            return

        fig, axes = plt.subplots(1, 3, figsize=tuple(cfg["figsize"]))

        norm = LogNorm() if cfg["log_scale"] else None

        # X-Y plane
        h1, xedges1, yedges1 = np.histogram2d(x, y, bins=[cfg["x_bins"], cfg["y_bins"]])
        im1 = axes[0].imshow(
            h1.T,
            origin="lower",
            extent=[xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]],
            aspect="auto",
            cmap=cfg["colormap"],
            norm=norm,
        )
        axes[0].set_xlabel("X [mm]")
        axes[0].set_ylabel("Y [mm]")
        axes[0].set_title("Occupancy: X-Y Plane")
        plt.colorbar(im1, ax=axes[0], label="Hits")

        # Z-Y plane
        h2, xedges2, yedges2 = np.histogram2d(z, y, bins=[cfg["z_bins"], cfg["y_bins"]])
        im2 = axes[1].imshow(
            h2.T,
            origin="lower",
            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]],
            aspect="auto",
            cmap=cfg["colormap"],
            norm=norm,
        )
        axes[1].set_xlabel("Z [mm]")
        axes[1].set_ylabel("Y [mm]")
        axes[1].set_title("Occupancy: Z-Y Plane")
        plt.colorbar(im2, ax=axes[1], label="Hits")

        # Z-X plane
        h3, xedges3, yedges3 = np.histogram2d(z, x, bins=[cfg["z_bins"], cfg["x_bins"]])
        im3 = axes[2].imshow(
            h3.T,
            origin="lower",
            extent=[xedges3[0], xedges3[-1], yedges3[0], yedges3[-1]],
            aspect="auto",
            cmap=cfg["colormap"],
            norm=norm,
        )
        axes[2].set_xlabel("Z [mm]")
        axes[2].set_ylabel("X [mm]")
        axes[2].set_title("Occupancy: Z-X Plane")
        plt.colorbar(im3, ax=axes[2], label="Hits")

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "occupancy_heatmaps" / "detector_occupancy.png",
            dpi=self.config["plot"]["dpi"],
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Created occupancy heatmaps with {len(x):,} hits")

    def save_statistics_summary(self) -> None:
        """Save all statistics to a summary text file."""
        print("\n" + "=" * 60)
        print("Saving Statistics Summary")
        print("=" * 60)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.output_dir / "statistics_summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ColliderML Data Exploration Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Number of events analyzed: {self.num_events}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")

            for category, cat_stats in self.statistics.items():
                f.write("-" * 80 + "\n")
                f.write(f"{category.upper()}\n")
                f.write("-" * 80 + "\n\n")

                if isinstance(cat_stats, dict):
                    for feature, feat_stats in cat_stats.items():
                        f.write(f"  {feature}:\n")
                        if isinstance(feat_stats, dict):
                            for key, value in feat_stats.items():
                                if isinstance(value, float):
                                    f.write(f"    {key}: {value:.6g}\n")
                                elif isinstance(value, dict):
                                    f.write(f"    {key}:\n")
                                    for k2, v2 in value.items():
                                        if isinstance(v2, float):
                                            f.write(f"      {k2}: {v2:.6g}\n")
                                        else:
                                            f.write(f"      {k2}: {v2}\n")
                                else:
                                    f.write(f"    {key}: {value}\n")
                        else:
                            f.write(f"    {feat_stats}\n")
                        f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"Statistics saved to: {self.output_dir / 'statistics_summary.txt'}")

    def run(self) -> None:
        """Run the complete exploration pipeline."""
        print("\n" + "=" * 80)
        print("ColliderML Data Exploration")
        print("=" * 80)

        # Load data
        particles_df, hits_df, tracks_df = self.load_data()

        # Explore features
        self.statistics["hit_features"] = self.explore_hit_features(hits_df, particles_df)
        self.statistics["particle_features"] = self.explore_particle_features(particles_df)
        self.statistics["track_features"] = self.explore_track_features(tracks_df)
        self.statistics["event_statistics"] = self.explore_event_statistics(particles_df, hits_df, tracks_df)

        # Create visualizations
        self.create_event_displays(hits_df, particles_df, tracks_df)
        self.create_occupancy_heatmaps(hits_df)

        # Save summary
        self.save_statistics_summary()

        print("\n" + "=" * 80)
        print("Exploration Complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ColliderML Data Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path(__file__).parent / "exploration_config.yaml",
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    explorer = ColliderMLExplorer(args.config)
    explorer.run()

    return 0


if __name__ == "__main__":
    exit(main())
