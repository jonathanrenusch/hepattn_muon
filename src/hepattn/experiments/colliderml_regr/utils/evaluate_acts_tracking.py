#!/usr/bin/env python3
"""
ACTS Track Fit Evaluation Script

Evaluates the goodness of ACTS track reconstruction on ColliderML data.

Metrics computed:
1. Residuals and Resolution (vs η):
   - d0, z0, phi, theta, qop residuals
   - pT resolution (recovered from qop and theta)
   - Binned standard deviation (precision)

2. Pull distributions:
   - Pseudo-pulls normalized by residual std dev
   - Should be approximately N(0,1) shaped

3. Reconstruction Efficiency and Fake Rate (vs η):
   - Efficiency = matched tracks / reconstructible particles
   - Fake rate = fake tracks / total tracks
   - Unweighted efficiency within |η| < 3

Reference: https://acts.readthedocs.io/en/latest/tracking.html
"""

from __future__ import annotations

import glob
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


class ACTSTrackingEvaluator:
    """Evaluator for ACTS track reconstruction quality."""

    def __init__(self, config_path: str | Path):
        """Initialize evaluator with configuration."""
        self.config = load_config(config_path)
        self.output_dir = Path(self.config["output"]["directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["residuals", "pulls", "efficiency", "resolution", "precision",
                      "resolution_double_matched", "precision_double_matched",
                      "hit_assignment"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)

        # Set random seed for reproducibility
        random.seed(self.config["data"].get("random_seed", 42))
        np.random.seed(self.config["data"].get("random_seed", 42))

    @staticmethod
    def _find_parquets(base_dir: Path, config_name: str) -> list[str]:
        """Find parquet files for a given config, supporting both flat and nested layouts.

        Nested (HuggingFace CLI): {base_dir}/{config}/data/{config}/train-*.parquet
        Flat (datasets library):  {base_dir}/{config}/train-*.parquet
        Direct (old eos layout):  {base_dir}/{config}/*.parquet  (no 'train-' prefix)
        """
        # Try nested first
        nested = sorted(glob.glob(str(base_dir / config_name / "data" / config_name / "train-*.parquet")))
        if nested:
            return nested
        # Try flat with train- prefix
        flat = sorted(glob.glob(str(base_dir / config_name / "train-*.parquet")))
        if flat:
            return flat
        # Try any parquet in the directory
        any_pq = sorted(glob.glob(str(base_dir / config_name / "*.parquet")))
        return any_pq

    def load_data(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Load particle, hit, and track data from ColliderML parquet files."""
        data_dir = Path(self.config["data"]["directory"])
        n_events = self.config["data"]["n_events"]
        prefix = self.config["data"].get("dataset_prefix", "ttbar_pu0")

        # Find all parquet files (supports both flat and nested layouts)
        particle_files = self._find_parquets(data_dir, f"{prefix}_particles")
        hit_files = self._find_parquets(data_dir, f"{prefix}_tracker_hits")
        track_files = self._find_parquets(data_dir, f"{prefix}_tracks")

        print(f"Found {len(particle_files)} particle files, {len(hit_files)} hit files, {len(track_files)} track files")

        # Count total events and sample (fast - only reads parquet footer metadata)
        total_events = 0
        event_counts = []
        for f in tqdm(particle_files, desc="Counting events"):
            # Use pyarrow to read only metadata (extremely fast)
            count = pq.read_metadata(f).num_rows
            event_counts.append((f, total_events, total_events + count))
            total_events += count

        print(f"Total events available: {total_events:,}")
        n_events = min(n_events, total_events)
        print(f"Using {n_events} events for evaluation")

        # Random sample of event indices
        sampled_indices = sorted(random.sample(range(total_events), n_events))

        # Determine which files contain sampled events
        files_to_load = set()
        for idx in sampled_indices:
            for f, start, end in event_counts:
                if start <= idx < end:
                    files_to_load.add(f)
                    break

        # Get file index mapping
        file_indices = {f: i for i, (f, _, _) in enumerate(event_counts)}
        indices_to_load = sorted([file_indices[f] for f in files_to_load])

        print(f"Events are in {len(indices_to_load)} file(s)")

        # Load data from selected files
        particles_list = []
        hits_list = []
        tracks_list = []

        for idx in tqdm(indices_to_load, desc="Loading particles"):
            particles_list.append(pl.read_parquet(particle_files[idx]))
        for idx in tqdm(indices_to_load, desc="Loading hits"):
            hits_list.append(pl.read_parquet(hit_files[idx]))
        for idx in tqdm(indices_to_load, desc="Loading tracks"):
            tracks_list.append(pl.read_parquet(track_files[idx]))

        particles_df = pl.concat(particles_list)
        hits_df = pl.concat(hits_list)
        tracks_df = pl.concat(tracks_list)

        # Filter to sampled events
        local_indices = []
        event_id = 0
        for f, start, end in event_counts:
            if f in files_to_load:
                for idx in sampled_indices:
                    if start <= idx < end:
                        local_indices.append(event_id + (idx - start))
                event_id += end - start
            else:
                pass

        # Use first n_events from loaded data
        unique_events = particles_df["event_id"].unique().sort()[:n_events].to_list()
        particles_df = particles_df.filter(pl.col("event_id").is_in(unique_events))
        hits_df = hits_df.filter(pl.col("event_id").is_in(unique_events))
        tracks_df = tracks_df.filter(pl.col("event_id").is_in(unique_events))

        print(f"  Loaded {len(particles_df)} event rows for particles")
        print(f"  Loaded {len(hits_df)} event rows for hits")
        print(f"  Loaded {len(tracks_df)} event rows for tracks")

        return particles_df, hits_df, tracks_df

    def _explode_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Explode list columns in a dataframe."""
        list_cols = [c for c in df.columns if c != "event_id" and df[c].dtype == pl.List]
        if list_cols:
            return df.explode(list_cols)
        return df

    def match_tracks_to_particles(
        self, tracks_df: pl.DataFrame, particles_df: pl.DataFrame, hits_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Match reconstructed tracks to true particles using majority_particle_id.

        Returns a DataFrame with matched tracks containing both reco and truth info.
        """
        # tracks_df is already passed as unique tracks (not exploded by hit_ids)
        tracks = tracks_df

        # Explode particles
        particles = self._explode_dataframe(particles_df)

        # Filter for primary particles only
        if "primary" in particles.columns:
            particles = particles.filter(pl.col("primary") == True)  # noqa: E712

        # Add truth particle columns
        particle_cols = ["particle_id", "px", "py", "pz", "charge", "perigee_d0", "perigee_z0", "pdg_id"]
        existing_cols = [c for c in particle_cols if c in particles.columns]

        # Merge tracks with particles using majority_particle_id
        if "majority_particle_id" in tracks.columns:
            matched = tracks.join(
                particles.select(["event_id"] + existing_cols),
                left_on=["event_id", "majority_particle_id"],
                right_on=["event_id", "particle_id"],
                how="left",
                suffix="_truth",
            )
        else:
            print("Warning: majority_particle_id not in tracks, cannot match to truth")
            return tracks

        # Calculate truth track parameters
        # pt_truth from momentum
        if "px" in matched.columns and "py" in matched.columns:
            matched = matched.with_columns((pl.col("px") ** 2 + pl.col("py") ** 2).sqrt().alias("pt_truth"))

        # eta_truth from momentum
        if "px" in matched.columns and "py" in matched.columns and "pz" in matched.columns:
            matched = matched.with_columns(
                (pl.col("pz") / (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt()).arcsinh().alias("eta_truth")
            )

        # phi_truth from momentum
        if "px" in matched.columns and "py" in matched.columns:
            matched = matched.with_columns(pl.arctan2(pl.col("py"), pl.col("px")).alias("phi_truth"))

        # theta_truth from eta
        if "eta_truth" in matched.columns:
            matched = matched.with_columns((2.0 * (-pl.col("eta_truth")).exp().arctan()).alias("theta_truth"))

        # qop_truth = q / p where p = sqrt(px^2 + py^2 + pz^2)
        if all(c in matched.columns for c in ["px", "py", "pz", "charge"]):
            matched = matched.with_columns(
                (
                    pl.col("charge") / (pl.col("px") ** 2 + pl.col("py") ** 2 + pl.col("pz") ** 2).sqrt()
                ).alias("qop_truth")
            )

        # Rename truth perigee parameters
        if "perigee_d0" in matched.columns:
            matched = matched.rename({"perigee_d0": "d0_truth"})
        if "perigee_z0" in matched.columns:
            matched = matched.rename({"perigee_z0": "z0_truth"})

        # Mark matched tracks (those with valid truth info)
        matched = matched.with_columns(pl.col("pt_truth").is_not_null().alias("is_matched"))

        return matched

    def compute_residuals(self, matched_tracks: pl.DataFrame) -> dict:
        """Compute residuals for all track parameters.

        Residual = reconstructed - truth
        """
        residuals = {}

        # Track parameters to evaluate
        params = {
            "d0": ("d0", "d0_truth"),
            "z0": ("z0", "z0_truth"),
            "phi": ("phi", "phi_truth"),
            "theta": ("theta", "theta_truth"),
            "qop": ("qop", "qop_truth"),
        }

        # Filter to matched tracks only
        matched = matched_tracks.filter(pl.col("is_matched"))

        for param_name, (reco_col, truth_col) in params.items():
            if reco_col in matched.columns and truth_col in matched.columns:
                reco = matched[reco_col].to_numpy()
                truth = matched[truth_col].to_numpy()

                # Handle phi wrapping
                if param_name == "phi":
                    res = reco - truth
                    res = np.where(res > np.pi, res - 2 * np.pi, res)
                    res = np.where(res < -np.pi, res + 2 * np.pi, res)
                else:
                    res = reco - truth

                # Filter infinities and NaN
                valid = np.isfinite(res)
                residuals[param_name] = res[valid]
                residuals[f"{param_name}_truth"] = truth[valid]

        # Compute pt residuals
        # pt_reco = sin(theta) / |qop|  (derived from track perigee parameters)
        # pt_truth = sqrt(px^2 + py^2)  (from truth particle momentum)
        if "pt" in matched.columns and "pt_truth" in matched.columns:
            pt_reco = matched["pt"].to_numpy()
            pt_truth = matched["pt_truth"].to_numpy()
            pt_abs = pt_reco - pt_truth                    # absolute residual [GeV]
            pt_res = (pt_reco - pt_truth) / pt_truth       # relative residual
            valid = np.isfinite(pt_res) & np.isfinite(pt_abs)
            residuals["pt_rel"] = pt_res[valid]
            residuals["pt_abs"] = pt_abs[valid]            # absolute pT residual [GeV]
            residuals["pt_truth"] = pt_truth[valid]
            residuals["pt_reco"] = pt_reco[valid]

        # Store eta for binning
        if "eta_truth" in matched.columns:
            residuals["eta"] = matched.filter(pl.col("is_matched"))["eta_truth"].to_numpy()

        return residuals

    def compute_double_matching(
        self,
        tracks_df: pl.DataFrame,
        hits_df: pl.DataFrame,
        purity_threshold: float = 0.5,
        efficiency_threshold: float = 0.5,
    ) -> dict:
        """Compute per-track hit purity and hit efficiency for double matching.

        For each reconstructed track:
          - **Purity** = (# hits from majority particle) / (# total hits on track)
          - **Hit efficiency** = (# hits from majority particle on track) /
                                 (# total hits from that particle in the event)
          - **Double-matched** = purity > threshold AND hit efficiency > threshold

        This is the standard matching quality criterion used in ATLAS/CMS
        tracking performance studies. It replaces sigma-clipping by
        identifying tracks whose truth assignment is reliable on the basis
        of the hit content, rather than the residuals themselves.

        Parameters
        ----------
        tracks_df : pl.DataFrame
            Raw event-level tracks (one row per event, list columns).
        hits_df : pl.DataFrame
            Raw event-level hits (one row per event, list columns).
        purity_threshold : float
            Minimum fraction of track hits from the majority particle.
        efficiency_threshold : float
            Minimum fraction of the particle's total hits recovered by the track.

        Returns
        -------
        dict with arrays indexed per exploded track:
            purity, hit_efficiency, is_double_matched, majority_particle_id,
            event_id, track_id, and _matching_info summary.
        """
        purities = []
        hit_effs = []
        dm_flags = []
        event_ids = []
        track_ids_out = []

        for row_idx in range(len(tracks_df)):
            eid = tracks_df["event_id"][row_idx]
            hit_ids_per_track = tracks_df["hit_ids"][row_idx]       # list of lists
            maj_pids = tracks_df["majority_particle_id"][row_idx]   # list
            t_ids = tracks_df["track_id"][row_idx]                  # list
            hit_pids = hits_df.filter(pl.col("event_id") == eid)["particle_id"][0]  # list

            # Count total hits per particle in this event (for efficiency denominator)
            from collections import Counter
            total_hits_per_particle = Counter(hit_pids.to_list())

            n_tracks_evt = len(maj_pids)
            for t_idx in range(n_tracks_evt):
                track_hit_indices = hit_ids_per_track[t_idx].to_list()
                maj_pid = maj_pids[t_idx]
                n_track_hits = len(track_hit_indices)

                # Count how many track hits belong to majority particle
                n_majority = sum(1 for h_idx in track_hit_indices if hit_pids[h_idx] == maj_pid)

                purity = n_majority / n_track_hits if n_track_hits > 0 else 0.0
                hit_eff = n_majority / total_hits_per_particle[maj_pid] if total_hits_per_particle[maj_pid] > 0 else 0.0
                is_dm = (purity > purity_threshold) and (hit_eff > efficiency_threshold)

                purities.append(purity)
                hit_effs.append(hit_eff)
                dm_flags.append(is_dm)
                event_ids.append(eid)
                track_ids_out.append(t_ids[t_idx])

        purities = np.array(purities)
        hit_effs = np.array(hit_effs)
        dm_flags = np.array(dm_flags, dtype=bool)

        n_total = len(dm_flags)
        n_dm = int(np.sum(dm_flags))
        n_pure_only = int(np.sum(purities > purity_threshold))
        n_eff_only = int(np.sum(hit_effs > efficiency_threshold))

        return {
            "purity": purities,
            "hit_efficiency": hit_effs,
            "is_double_matched": dm_flags,
            "event_id": np.array(event_ids),
            "track_id": np.array(track_ids_out),
            "_matching_info": {
                "n_total": n_total,
                "n_double_matched": n_dm,
                "n_purity_pass": n_pure_only,
                "n_efficiency_pass": n_eff_only,
                "fraction_double_matched": n_dm / n_total if n_total > 0 else 0.0,
                "purity_threshold": purity_threshold,
                "efficiency_threshold": efficiency_threshold,
                "mean_purity": float(np.mean(purities)),
                "mean_hit_efficiency": float(np.mean(hit_effs)),
            },
        }

    @staticmethod
    def apply_double_matching(
        residuals: dict,
        dm_mask: np.ndarray,
    ) -> dict:
        """Apply a double-matching boolean mask to residual arrays.

        Returns a new residuals dict containing only the double-matched entries,
        plus a ``_dm_info`` summary key.
        """
        n_tracks = len(dm_mask)
        n_dm = int(np.sum(dm_mask))

        dm_residuals = {}
        for key, arr in residuals.items():
            if isinstance(arr, np.ndarray) and len(arr) == n_tracks:
                dm_residuals[key] = arr[dm_mask]
            elif isinstance(arr, np.ndarray) and len(arr) < n_tracks:
                # Shorter arrays (e.g. d0/z0 with null perigee values)
                # Use a length-matched sub-mask
                sub_mask = dm_mask[:len(arr)]
                dm_residuals[key] = arr[sub_mask]
            else:
                dm_residuals[key] = arr

        dm_residuals["_dm_info"] = {
            "n_total": n_tracks,
            "n_double_matched": n_dm,
            "fraction": n_dm / n_tracks if n_tracks > 0 else 0.0,
        }
        return dm_residuals

    def compute_pulls(self, matched_tracks: pl.DataFrame) -> dict:
        """Compute pull distributions.

        Pull = residual / uncertainty
        For well-calibrated uncertainties, pulls should be N(0, 1).

        Note: ColliderML may not have per-track uncertainties.
        In that case, we estimate from residual distributions.
        """
        pulls = {}

        # Filter matched tracks
        matched = matched_tracks.filter(pl.col("is_matched"))

        # Check if we have uncertainty columns (typically named like d0_cov, etc.)
        # ColliderML tracks may not have these, so we'll use residual-based estimates

        # For now, return residuals - pulls require covariance information
        # which we need to check in the data

        return pulls

    def compute_efficiency_vs_eta(
        self, matched_tracks: pl.DataFrame, particles_df: pl.DataFrame, hits_df: pl.DataFrame
    ) -> dict:
        """Compute reconstruction efficiency and fake rate vs pseudorapidity.

        Efficiency = matched tracks / reconstructible particles
        Fake rate = unmatched tracks / total tracks

        A particle is reconstructible if it:
        - Is primary
        - Has sufficient hits (N >= min_hits)
        """
        min_hits = self.config["efficiency"].get("min_hits", 7)
        min_energy = self.config["efficiency"].get("min_energy", 1.0)
        eta_range = self.config["binning"]["eta_range"]
        n_eta_bins = self.config["binning"]["n_eta_bins"]
        eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)

        # Get particles
        particles = self._explode_dataframe(particles_df)
        if "primary" in particles.columns:
            particles = particles.filter(pl.col("primary") == True)  # noqa: E712
        
        # Filter by minimum energy
        if "energy" in particles.columns:
            particles = particles.filter(pl.col("energy") >= min_energy)

        # Get hits
        hits = self._explode_dataframe(hits_df)

        # Count hits per particle (needs event_id + particle_id join)
        hits_per_particle = (
            hits.group_by(["event_id", "particle_id"])
            .agg(pl.len().alias("n_hits"))
        )

        # Join with particles
        particles_with_hits = particles.join(
            hits_per_particle,
            on=["event_id", "particle_id"],
            how="left",
        ).with_columns(pl.col("n_hits").fill_null(0))

        # Reconstructible = primary with enough hits and energy
        reconstructible = particles_with_hits.filter(pl.col("n_hits") >= min_hits)
        
        print(f"  Primary particles with E >= {min_energy} GeV: {len(particles)}")
        print(f"  Primary particles with >= {min_hits} hits (reconstructible): {len(reconstructible)}")

        # Calculate eta for reconstructible particles
        if all(c in reconstructible.columns for c in ["px", "py", "pz"]):
            reconstructible = reconstructible.with_columns(
                (pl.col("pz") / (pl.col("px") ** 2 + pl.col("py") ** 2).sqrt()).arcsinh().alias("eta")
            )

        # Get matched track majority_particle_ids
        matched_particle_ids = matched_tracks.filter(pl.col("is_matched")).select(
            ["event_id", "majority_particle_id"]
        ).unique().with_columns(pl.lit(True).alias("_matched"))

        # Mark which reconstructible particles were found
        reconstructible_matched = reconstructible.join(
            matched_particle_ids.rename({"majority_particle_id": "particle_id"}),
            on=["event_id", "particle_id"],
            how="left",
        ).with_columns(
            pl.col("_matched").fill_null(False).alias("is_found")
        )
        
        n_found = reconstructible_matched.filter(pl.col("is_found")).height
        print(f"  Unique matched particle IDs from tracks: {len(matched_particle_ids)}")
        print(f"  Reconstructible particles found by tracks: {n_found}")
        print(f"  => Reconstruction efficiency: {n_found}/{len(reconstructible)} = {n_found/len(reconstructible):.1%}")

        # Compute unweighted efficiency within |eta| < 3
        eta_values = reconstructible_matched["eta"].to_numpy()
        is_found = reconstructible_matched["is_found"].to_numpy()
        
        eta3_mask = np.abs(eta_values) < 3.0
        n_reconstructible_eta3 = np.sum(eta3_mask)
        n_found_eta3 = np.sum(is_found[eta3_mask])
        unweighted_efficiency_eta3 = n_found_eta3 / n_reconstructible_eta3 if n_reconstructible_eta3 > 0 else 0.0
        print(f"  Unweighted efficiency (|eta| < 3): {n_found_eta3}/{n_reconstructible_eta3} = {unweighted_efficiency_eta3:.1%}")

        # Bin by eta 
        efficiency_data = {
            "eta_bins": eta_bins,
            "eta_centers": [], 
            "efficiency": [], 
            "efficiency_err": [], 
            "n_total": [], 
            "n_matched": [],
            "unweighted_efficiency_eta3": unweighted_efficiency_eta3,
            "n_reconstructible_eta3": n_reconstructible_eta3,
            "n_found_eta3": n_found_eta3,
            "n_reconstructible_total": len(reconstructible),
            "n_primary_particles_with_hits": len(particles_with_hits),
        }

        for i in range(len(eta_bins) - 1):
            eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
            mask = (eta_values >= eta_low) & (eta_values < eta_high)
            n_total = np.sum(mask)
            n_matched = np.sum(is_found[mask])

            eff = n_matched / n_total if n_total > 0 else 0.0
            # Binomial error
            eff_err = np.sqrt(eff * (1 - eff) / n_total) if n_total > 0 else 0.0

            efficiency_data["eta_centers"].append((eta_low + eta_high) / 2)
            efficiency_data["efficiency"].append(eff)
            efficiency_data["efficiency_err"].append(eff_err)
            efficiency_data["n_total"].append(n_total)
            efficiency_data["n_matched"].append(n_matched)

        efficiency_data["eta_bins"] = eta_bins

        # Compute fake rate vs eta
        track_eta = matched_tracks["eta"].to_numpy() if "eta" in matched_tracks.columns else None
        if track_eta is None and "theta" in matched_tracks.columns:
            theta = matched_tracks["theta"].to_numpy()
            track_eta = -np.log(np.tan(theta / 2))

        is_fake = ~matched_tracks["is_matched"].to_numpy()

        fake_data = {"eta_centers": [], "fake_rate": [], "fake_rate_err": [], "n_total": [], "n_fake": []}

        if track_eta is not None:
            for i in range(len(eta_bins) - 1):
                eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
                mask = (track_eta >= eta_low) & (track_eta < eta_high)
                n_total = np.sum(mask)
                n_fake = np.sum(is_fake[mask])

                rate = n_fake / n_total if n_total > 0 else 0.0
                rate_err = np.sqrt(rate * (1 - rate) / n_total) if n_total > 0 else 0.0

                fake_data["eta_centers"].append((eta_low + eta_high) / 2)
                fake_data["fake_rate"].append(rate)
                fake_data["fake_rate_err"].append(rate_err)
                fake_data["n_total"].append(n_total)
                fake_data["n_fake"].append(n_fake)
        
        fake_data["eta_bins"] = eta_bins

        return {"efficiency": efficiency_data, "fake_rate": fake_data}

    def compute_resolution_vs_eta(self, residuals: dict) -> dict:
        """Compute binned resolution (mean relative residual) vs pseudorapidity.

        For each parameter, computes:
        - mean: mean of absolute residual per eta bin (bias)
        - std: std of absolute residual per eta bin (precision)
        - rel_mean: mean of (reco - truth) / |truth| per eta bin (resolution)
        - unbinned_rel_mean: overall unbinned average of relative residual
        """
        eta_range = self.config["binning"]["eta_range"]
        n_eta_bins = self.config["binning"]["n_eta_bins"]
        eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)

        if "eta" not in residuals:
            print("Warning: eta not in residuals, cannot compute resolution vs eta")
            return {}

        eta = residuals["eta"]
        resolution_data = {}

        # Only process proper residual parameters (skip auxiliary arrays like pt_abs, pt_reco)
        residual_params = {"d0", "z0", "phi", "theta", "qop", "pt_rel"}
        for param_name, res_values in residuals.items():
            if param_name not in residual_params:
                continue

            # Ensure same length
            min_len = min(len(eta), len(res_values))
            param_eta = eta[:min_len]
            param_res = res_values[:min_len]

            # Compute relative residuals: (reco - truth) / |truth|
            if param_name == "pt_rel":
                # pt_rel is already (pt_reco - pt_truth) / pt_truth
                param_rel = param_res
            else:
                truth_key = f"{param_name}_truth"
                if truth_key in residuals:
                    param_truth = residuals[truth_key][:min_len]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        param_rel = param_res / np.abs(param_truth)
                else:
                    param_rel = param_res  # fallback to absolute

            # Unbinned average of relative residual (finite values only)
            unbinned_finite = param_rel[np.isfinite(param_rel)]
            unbinned_rel_mean = float(np.mean(unbinned_finite)) if len(unbinned_finite) > 0 else 0.0
            # Unbinned std of absolute residual
            unbinned_std = float(np.std(param_res)) if len(param_res) > 0 else 0.0

            centers = []
            means = []
            stds = []
            rel_means = []
            rel_stds = []
            counts = []

            for i in range(len(eta_bins) - 1):
                eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
                mask = (param_eta >= eta_low) & (param_eta < eta_high)

                if np.sum(mask) > 2:
                    bin_res = param_res[mask]
                    centers.append((eta_low + eta_high) / 2)
                    means.append(np.mean(bin_res))
                    stds.append(np.std(bin_res))
                    counts.append(len(bin_res))

                    # Mean relative residual for this bin
                    bin_rel = param_rel[mask]
                    bin_rel_finite = bin_rel[np.isfinite(bin_rel)]
                    rel_means.append(float(np.mean(bin_rel_finite)) if len(bin_rel_finite) > 0 else 0.0)
                    rel_stds.append(float(np.std(bin_rel_finite)) if len(bin_rel_finite) > 0 else 0.0)

            counts_arr = np.array(counts, dtype=float)
            stds_arr = np.array(stds)
            rel_stds_arr = np.array(rel_stds)

            resolution_data[param_name] = {
                "eta_centers": np.array(centers),
                "mean": np.array(means),
                "std": stds_arr,
                "rel_mean": np.array(rel_means),
                "rel_std": rel_stds_arr,
                "count": counts_arr,
                # Uncertainties: SEM for means, σ/√(2N) for std
                "mean_err": stds_arr / np.sqrt(counts_arr),
                "std_err": stds_arr / np.sqrt(2 * counts_arr),
                "rel_mean_err": rel_stds_arr / np.sqrt(counts_arr),
                "unbinned_rel_mean": unbinned_rel_mean,
                "unbinned_std": unbinned_std,
            }

        return resolution_data

    # Unit scaling for human-readable plots
    UNIT_SCALE = {
        "d0": 1.0,       # already mm
        "z0": 1.0,       # already mm
        "phi": 1e3,      # rad -> mrad
        "theta": 1e3,    # rad -> mrad
        "qop": 1.0,      # 1/GeV
        "pt_rel": 1.0,   # dimensionless
    }

    @staticmethod
    def _build_step_arrays(bin_edges: np.ndarray, values: np.ndarray) -> tuple:
        """Build x/y arrays for a continuous step histogram from bin edges and values.

        Returns (x, y) suitable for ax.step(..., where='post').
        bin_edges has length N+1, values has length N.
        """
        x = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
        y = np.concatenate([values, [values[-1]]])
        return x, y

    @staticmethod
    def _step_fill_between(ax, bin_edges: np.ndarray, y_lo: np.ndarray, y_hi: np.ndarray,
                           color: str = "steelblue", alpha: float = 0.25, label: str = "") -> None:
        """Draw a step-style fill_between band from bin edges and lo/hi arrays."""
        # Build staircase coordinates
        n = len(y_lo)
        xs = np.empty(2 * n)
        lo = np.empty(2 * n)
        hi = np.empty(2 * n)
        for i in range(n):
            xs[2 * i] = bin_edges[i]
            xs[2 * i + 1] = bin_edges[i + 1]
            lo[2 * i] = y_lo[i]
            lo[2 * i + 1] = y_lo[i]
            hi[2 * i] = y_hi[i]
            hi[2 * i + 1] = y_hi[i]
        ax.fill_between(xs, lo, hi, color=color, alpha=alpha, label=label, step=None)

    def _get_bin_edges_for_centers(self, eta_centers: np.ndarray, eta_bins: np.ndarray) -> np.ndarray:
        """Map eta_centers back to bin edges from the eta_bins array.

        Returns an array of length len(eta_centers)+1 with the edges of the
        bins that actually contain data.
        """
        # For each center, find the bin it falls in
        bin_indices = np.searchsorted(eta_bins, eta_centers, side="right") - 1
        bin_indices = np.clip(bin_indices, 0, len(eta_bins) - 2)
        # Left edge of each occupied bin, plus right edge of last
        left_edges = eta_bins[bin_indices]
        right_edge = eta_bins[bin_indices[-1] + 1]
        return np.append(left_edges, right_edge)

    def plot_residuals(self, residuals: dict) -> None:
        """Plot residual distributions."""
        param_labels = {
            "d0": r"$d_0$ residual [mm]",
            "z0": r"$z_0$ residual [mm]",
            "phi": r"$\phi$ residual [mrad]",
            "theta": r"$\theta$ residual [mrad]",
            "qop": r"$q/p$ residual [1/GeV]",
            "pt_rel": r"$(p_T^{reco} - p_T^{truth}) / p_T^{truth}$",
        }

        for param_name, res_values in residuals.items():
            if param_name == "eta" or param_name.endswith("_truth"):
                continue

            fig, ax = plt.subplots(figsize=(8, 6))

            # Apply unit scaling
            scale = self.UNIT_SCALE.get(param_name, 1.0)
            res_scaled = res_values * scale

            mean = np.mean(res_scaled)
            std = np.std(res_scaled)

            # Use mean ± 3sigma range, clip outliers into edge bins
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            n_bins = 80
            bins = np.linspace(lower_bound, upper_bound, n_bins + 1)
            clipped = np.clip(res_scaled, lower_bound, upper_bound)
            counts, bin_edges = np.histogram(clipped, bins=bins, density=False)

            # Step histogram
            x_step = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
            y_step = np.concatenate([counts, [counts[-1] if len(counts) > 0 else 0]])
            ax.step(x_step, y_step, where="post", linewidth=1.5, color="steelblue",
                   label=f"{param_name.upper()} Residual")

            # Vertical lines for mean and ±1σ
            ax.axvline(mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean:.4f}")
            ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.5, alpha=0.8,
                      label=f"±1σ: {std:.4f}")
            ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.5, alpha=0.8)
            ax.axvline(0, color="black", linestyle="-", alpha=0.5, linewidth=1)

            ax.set_xlabel(param_labels.get(param_name, f"{param_name} residual"), fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"{param_name.upper()} Residual Distribution", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Stats text box
            n_total = len(res_scaled)
            n_clipped = np.sum((res_scaled < lower_bound) | (res_scaled > upper_bound))
            stats_text = f"Mean: {mean:.4f}\nSTD: {std:.4f}\nN: {n_total}\nClipped: {n_clipped}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}, fontsize=10)

            fig.savefig(
                self.output_dir / "residuals" / f"{param_name}_residual.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plot_resolution_vs_eta(self, resolution_data: dict, eta_bins: np.ndarray,
                              subdir: str = "resolution", title_suffix: str = "") -> None:
        """Plot resolution (mean relative residual) vs pseudorapidity using step style.

        Top panel: mean of (reco - truth) / |truth| per eta bin
        Bottom panel: mean of absolute residual (bias) per eta bin
        """
        resolution_labels = {
            "d0": r"$\langle \Delta d_0 / |d_0| \rangle$ [%]",
            "z0": r"$\langle \Delta z_0 / |z_0| \rangle$ [%]",
            "phi": r"$\langle \Delta\phi / |\phi| \rangle$ [%]",
            "theta": r"$\langle \Delta\theta / \theta \rangle$ [%]",
            "qop": r"$\langle \Delta(q/p) / |q/p| \rangle$ [%]",
            "pt_rel": r"$\langle \Delta p_T / p_T \rangle$ [%]",
        }
        bias_labels = {
            "d0": "Mean (bias) [mm]",
            "z0": "Mean (bias) [mm]",
            "phi": "Mean (bias) [mrad]",
            "theta": "Mean (bias) [mrad]",
            "qop": "Mean (bias) [1/GeV]",
            "pt_rel": "Mean (bias)",
        }

        for param_name, data in resolution_data.items():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

            eta_centers = data["eta_centers"]
            mean = data["mean"]
            rel_mean = data["rel_mean"] * 100  # convert to percentage
            rel_mean_err = data["rel_mean_err"] * 100
            mean_err = data["mean_err"]
            unbinned_avg = data.get("unbinned_rel_mean", 0.0) * 100

            # Apply unit scaling for bias panel only
            scale = self.UNIT_SCALE.get(param_name, 1.0)

            # Build contiguous bin edges
            edges = self._get_bin_edges_for_centers(eta_centers, eta_bins)

            # Top panel: mean relative residual per bin (in %)
            x, y = self._build_step_arrays(edges, rel_mean)
            ax1.step(x, y, where="post", color="steelblue", linewidth=2.5,
                     label=f"Resolution (avg: {unbinned_avg:.2f}%)")
            self._step_fill_between(ax1, edges, rel_mean - rel_mean_err,
                                    rel_mean + rel_mean_err, color="steelblue",
                                    alpha=0.25, label="SEM")

            ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax1.set_ylabel(resolution_labels.get(param_name, f"$\\langle \\Delta {param_name} / |{param_name}| \\rangle$"), fontsize=12)
            ax1.set_title(f"{param_name.upper()} Resolution vs $\\eta${title_suffix}", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Bottom panel: mean (bias) vs eta
            ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
            x2, y2 = self._build_step_arrays(edges, mean * scale)
            ax2.step(x2, y2, where="post", color="darkorange", linewidth=2.5,
                     label="Mean (bias)")
            self._step_fill_between(ax2, edges, (mean - mean_err) * scale,
                                    (mean + mean_err) * scale, color="darkorange",
                                    alpha=0.25, label="SEM")

            ax2.set_xlabel(r"$\eta_{truth}$", fontsize=12)
            ax2.set_ylabel(bias_labels.get(param_name, "Mean (bias)"), fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            fig.savefig(
                self.output_dir / subdir / f"{param_name}_resolution_vs_eta.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

    def plot_precision_vs_eta(self, resolution_data: dict, eta_bins: np.ndarray,
                             subdir: str = "precision", title_suffix: str = "") -> None:
        """Plot precision (std dev only) vs pseudorapidity as standalone single-panel plots."""
        param_labels = {
            "d0": r"$\sigma(d_0)$ [mm]",
            "z0": r"$\sigma(z_0)$ [mm]",
            "phi": r"$\sigma(\phi)$ [mrad]",
            "theta": r"$\sigma(\theta)$ [mrad]",
            "qop": r"$\sigma(q/p)$ [1/GeV]",
            "pt_rel": r"$\sigma(p_T) / p_T$",
        }

        for param_name, data in resolution_data.items():
            fig, ax = plt.subplots(figsize=(10, 6))

            eta_centers = data["eta_centers"]
            std = data["std"]
            std_err = data["std_err"]
            scale = self.UNIT_SCALE.get(param_name, 1.0)
            unbinned_std = data.get("unbinned_std", 0.0) * scale

            edges = self._get_bin_edges_for_centers(eta_centers, eta_bins)
            x, y = self._build_step_arrays(edges, std * scale)
            ax.step(x, y, where="post", color="steelblue", linewidth=2.5,
                    label=f"Precision (avg σ: {unbinned_std:.4f})")
            self._step_fill_between(ax, edges, (std - std_err) * scale,
                                    (std + std_err) * scale, color="steelblue",
                                    alpha=0.25, label="Uncertainty")

            ax.set_xlabel(r"$\eta_{truth}$", fontsize=12)
            ax.set_ylabel(param_labels.get(param_name, f"$\\sigma$({param_name})"), fontsize=12)
            ax.set_title(f"{param_name.upper()} Precision vs $\\eta${title_suffix}", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.legend()

            plt.tight_layout()
            fig.savefig(
                self.output_dir / subdir / f"{param_name}_precision_vs_eta.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Summary precision plot
        self._plot_summary_precision(resolution_data, eta_bins, subdir=subdir, title_suffix=title_suffix)

    def _plot_summary_precision(self, resolution_data: dict, eta_bins: np.ndarray,
                               subdir: str = "precision", title_suffix: str = "") -> None:
        """Create summary plot with all precision vs eta curves."""
        params_to_plot = ["d0", "z0", "phi", "theta", "qop", "pt_rel"]
        params_available = [p for p in params_to_plot if p in resolution_data]

        if not params_available:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        param_labels = {
            "d0": r"$\sigma(d_0)$ [mm]",
            "z0": r"$\sigma(z_0)$ [mm]",
            "phi": r"$\sigma(\phi)$ [mrad]",
            "theta": r"$\sigma(\theta)$ [mrad]",
            "qop": r"$\sigma(q/p)$ [1/GeV]",
            "pt_rel": r"$\sigma(p_T)/p_T$",
        }

        for i, param in enumerate(params_available):
            ax = axes[i]
            data = resolution_data[param]
            eta_centers = data["eta_centers"]
            std_vals = data["std"]
            std_err = data["std_err"]
            scale = self.UNIT_SCALE.get(param, 1.0)
            unbinned_std = data.get("unbinned_std", 0.0) * scale

            edges = self._get_bin_edges_for_centers(eta_centers, eta_bins)
            x, y = self._build_step_arrays(edges, std_vals * scale)
            ax.step(x, y, where="post", color="steelblue", linewidth=2,
                    label=f"Avg \u03c3: {unbinned_std:.4f}")
            self._step_fill_between(ax, edges, (std_vals - std_err) * scale,
                                    (std_vals + std_err) * scale, color="steelblue",
                                    alpha=0.25)

            ax.set_xlabel(r"$\eta$", fontsize=11)
            ax.set_ylabel(param_labels.get(param, f"$\\sigma$({param})"), fontsize=11)
            ax.set_title(f"{param.upper()} Precision", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=9)

        for j in range(len(params_available), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Track Parameter Precision vs $\\eta${title_suffix}", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(
            self.output_dir / subdir / "summary_precision_vs_eta.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_efficiency_and_fake_rate(self, eff_data: dict) -> None:
        """Plot reconstruction efficiency and fake rate vs eta using step style."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Efficiency with step style
        eff = eff_data["efficiency"]
        eta_bins = eff.get("eta_bins", None)
        unbinned_eff = eff.get("unweighted_efficiency_eta3", 0.0)

        if eta_bins is not None:
            eff_vals = np.array(eff["efficiency"])
            eff_errs = np.array(eff["efficiency_err"])
            x, y = self._build_step_arrays(eta_bins, eff_vals)
            ax1.step(x, y, where="post", color="green", linewidth=2.5,
                     label=f"Efficiency (avg: {unbinned_eff:.3f})")
            self._step_fill_between(ax1, eta_bins,
                                    np.clip(eff_vals - eff_errs, 0, 1),
                                    np.clip(eff_vals + eff_errs, 0, 1),
                                    color="green", alpha=0.25, label="Binomial error")

        ax1.set_xlabel(r"$\eta_{truth}$", fontsize=12)
        ax1.set_ylabel("Efficiency", fontsize=12)
        ax1.set_title("Track Reconstruction Efficiency vs $\\eta$", fontsize=14)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Fake rate with step style
        fake = eff_data["fake_rate"]
        fake_eta_bins = fake.get("eta_bins", None)

        if fake_eta_bins is not None and fake["eta_centers"]:
            fake_vals = np.array(fake["fake_rate"])
            fake_errs = np.array(fake["fake_rate_err"])
            unbinned_fake = float(np.mean(fake_vals)) if len(fake_vals) > 0 else 0.0
            x2, y2 = self._build_step_arrays(fake_eta_bins, fake_vals)
            ax2.step(x2, y2, where="post", color="red", linewidth=2.5,
                     label=f"Fake Rate (avg: {unbinned_fake:.3f})")
            self._step_fill_between(ax2, fake_eta_bins,
                                    np.clip(fake_vals - fake_errs, 0, 1),
                                    np.clip(fake_vals + fake_errs, 0, 1),
                                    color="red", alpha=0.25, label="Binomial error")

        ax2.set_xlabel(r"$\eta_{reco}$", fontsize=12)
        ax2.set_ylabel("Fake Rate", fontsize=12)
        ax2.set_title("Track Fake Rate vs $\\eta$", fontsize=14)
        ax2.set_ylim(0, max(0.5, max(fake["fake_rate"]) * 1.2) if fake["fake_rate"] else 0.5)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "efficiency" / "efficiency_fake_rate_vs_eta.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_summary_residuals(self, residuals: dict) -> None:
        """Create summary plot with all residual distributions."""
        params_to_plot = ["d0", "z0", "phi", "theta", "qop", "pt_rel"]
        params_available = [p for p in params_to_plot if p in residuals]

        n_params = len(params_available)
        if n_params == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        param_labels = {
            "d0": r"$d_0$ residual [mm]",
            "z0": r"$z_0$ residual [mm]",
            "phi": r"$\phi$ residual [mrad]",
            "theta": r"$\theta$ residual [mrad]",
            "qop": r"$q/p$ residual [1/GeV]",
            "pt_rel": r"$\Delta p_T / p_T$",
        }
        unit_suffix = {"d0": " mm", "z0": " mm", "phi": " mrad", "theta": " mrad", "qop": " 1/GeV", "pt_rel": ""}

        for i, param in enumerate(params_available):
            ax = axes[i]
            res = residuals[param]

            # Apply unit scaling
            scale = self.UNIT_SCALE.get(param, 1.0)
            res_scaled = res * scale

            mean, std = np.mean(res_scaled), np.std(res_scaled)

            # Step histogram in mean ± 3sigma range
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            bins = np.linspace(lower_bound, upper_bound, 61)
            clipped = np.clip(res_scaled, lower_bound, upper_bound)
            counts, bin_edges = np.histogram(clipped, bins=bins, density=False)
            x_step = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
            y_step = np.concatenate([counts, [counts[-1] if len(counts) > 0 else 0]])
            ax.step(x_step, y_step, where="post", linewidth=1.5, color="steelblue")

            ax.axvline(mean, color="red", linestyle="--", lw=1.5, label=f"$\\mu$={mean:.4f}")
            ax.axvline(mean - std, color="orange", linestyle=":", lw=1.5, alpha=0.8)
            ax.axvline(mean + std, color="orange", linestyle=":", lw=1.5, alpha=0.8,
                      label=f"$\\sigma$={std:.4f}")
            ax.axvline(0, color="black", linestyle="-", alpha=0.4, lw=1)

            ax.set_xlabel(param_labels.get(param, param), fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"{param.upper()}: $\\sigma$={std:.4f}{unit_suffix.get(param, '')}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Track Parameter Residual Distributions", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(
            self.output_dir / "residuals" / "summary_residuals.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_summary_resolution(self, resolution_data: dict, eta_bins: np.ndarray) -> None:
        """Create summary plot with all resolution (mean relative residual) vs eta curves."""
        params_to_plot = ["d0", "z0", "phi", "theta", "qop", "pt_rel"]
        params_available = [p for p in params_to_plot if p in resolution_data]

        if not params_available:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        param_labels = {
            "d0": r"$\langle \Delta d_0 / |d_0| \rangle$ [%]",
            "z0": r"$\langle \Delta z_0 / |z_0| \rangle$ [%]",
            "phi": r"$\langle \Delta\phi / |\phi| \rangle$ [%]",
            "theta": r"$\langle \Delta\theta / \theta \rangle$ [%]",
            "qop": r"$\langle \Delta(q/p) / |q/p| \rangle$ [%]",
            "pt_rel": r"$\langle \Delta p_T / p_T \rangle$ [%]",
        }

        for i, param in enumerate(params_available):
            ax = axes[i]
            data = resolution_data[param]

            eta_centers = data["eta_centers"]
            rel_mean_vals = data["rel_mean"] * 100  # convert to percentage
            rel_mean_err = data["rel_mean_err"] * 100
            unbinned_avg = data.get("unbinned_rel_mean", 0.0) * 100

            edges = self._get_bin_edges_for_centers(eta_centers, eta_bins)
            x, y = self._build_step_arrays(edges, rel_mean_vals)
            ax.step(x, y, where="post", color="steelblue", linewidth=2,
                    label=f"Avg: {unbinned_avg:.2f}%")
            self._step_fill_between(ax, edges, rel_mean_vals - rel_mean_err,
                                    rel_mean_vals + rel_mean_err, color="steelblue",
                                    alpha=0.25)

            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel(r"$\eta$", fontsize=11)
            ax.set_ylabel(param_labels.get(param, f"Resolution ({param})"), fontsize=11)
            ax.set_title(f"{param.upper()} Resolution", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        for j in range(len(params_available), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Track Parameter Resolution vs $\\eta$", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(
            self.output_dir / "resolution" / "summary_resolution_vs_eta.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_pulls(self, residuals: dict, eta_bins: np.ndarray) -> None:
        """Plot pull distributions using step histogram style.
        
        Since ColliderML tracks don't have per-track covariance, we compute
        'pseudo-pulls' by normalizing residuals by the overall std dev.
        This shows the normalized residual distribution shape.
        """
        param_labels = {
            "d0": r"$d_0$ pull",
            "z0": r"$z_0$ pull",
            "phi": r"$\phi$ pull",
            "theta": r"$\theta$ pull",
            "qop": r"$q/p$ pull",
            "pt_rel": r"$p_T$ relative pull",
        }
        
        params_to_plot = ["d0", "z0", "phi", "theta", "qop", "pt_rel"]
        params_available = [p for p in params_to_plot if p in residuals]
        
        if not params_available:
            print("No parameters available for pull plots")
            return
        
        for param_name in params_available:
            res_values = residuals[param_name]
            
            if len(res_values) == 0:
                continue
            
            # Compute pseudo-pulls: residual / std(residuals)
            # This normalizes to unit variance
            std_dev = np.std(res_values)
            if std_dev == 0:
                continue
            pulls = res_values / std_dev
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate mean and std for display
            mean_pull = np.mean(pulls)
            std_pull = np.std(pulls)
            
            # Define range as mean ± 4*sigma for pulls (should be ~N(0,1))
            lower_bound = -4.0
            upper_bound = 4.0
            
            # Create bins
            n_bins = 50
            bins = np.linspace(lower_bound, upper_bound, n_bins + 1)
            
            # Clip outliers into edge bins
            clipped_pulls = np.clip(pulls, lower_bound, upper_bound)
            
            # Calculate histogram
            counts, bin_edges = np.histogram(clipped_pulls, bins=bins, density=False)
            
            # Create step histogram 
            x_step = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
            y_step = np.concatenate([counts, [counts[-1] if len(counts) > 0 else 0]])
            
            ax.step(x_step, y_step, where="post", linewidth=1.5, color="blue", 
                   label=f"{param_name.capitalize()} Pull (pseudo)")
            
            # Add vertical lines for statistics
            ax.axvline(mean_pull, color="red", linestyle="--", linewidth=2, 
                      label=f"Mean: {mean_pull:.3f}")
            ax.axvline(0, color="black", linestyle="-", alpha=0.7, linewidth=1, 
                      label="Expected (0)")
            ax.axvline(-1, color="orange", linestyle=":", alpha=0.7, 
                      label=f"±1σ (std={std_pull:.3f})")
            ax.axvline(1, color="orange", linestyle=":", alpha=0.7)
            ax.axvline(-2, color="yellow", linestyle=":", alpha=0.5)
            ax.axvline(2, color="yellow", linestyle=":", alpha=0.5, label="±2σ")
            
            ax.set_xlabel(param_labels.get(param_name, f"{param_name} pull"), fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"{param_name.upper()} Pull Distribution\n(Pseudo-pull: residual / σ(residuals))", fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Add text box with statistics
            n_total = len(pulls)
            n_clipped = np.sum((pulls < lower_bound) | (pulls > upper_bound))
            clipped_pct = 100 * n_clipped / n_total if n_total > 0 else 0
            
            stats_text = f"Mean: {mean_pull:.4f}\nSTD: {std_pull:.4f}\nN: {n_total}\nClipped: {n_clipped} ({clipped_pct:.1f}%)"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment="top",
                   bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}, fontsize=10)
            
            plt.tight_layout()
            fig.savefig(
                self.output_dir / "pulls" / f"{param_name}_pull.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
        
        # Summary pull plot
        self.plot_summary_pulls(residuals)
    
    def plot_summary_pulls(self, residuals: dict) -> None:
        """Create summary plot with all pull distributions."""
        params_to_plot = ["d0", "z0", "phi", "theta", "qop", "pt_rel"]
        params_available = [p for p in params_to_plot if p in residuals]
        
        n_params = len(params_available)
        if n_params == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        param_labels = {
            "d0": r"$d_0$ pull",
            "z0": r"$z_0$ pull",
            "phi": r"$\phi$ pull",
            "theta": r"$\theta$ pull",
            "qop": r"$q/p$ pull",
            "pt_rel": r"$\Delta p_T / p_T$ pull",
        }
        
        for i, param in enumerate(params_available):
            ax = axes[i]
            res = residuals[param]
            
            if len(res) == 0:
                continue
            
            # Compute pseudo-pulls
            std_dev = np.std(res)
            if std_dev == 0:
                continue
            pulls = res / std_dev
            
            # Use ±4σ range
            lower_bound, upper_bound = -4.0, 4.0
            n_bins = 40
            bins = np.linspace(lower_bound, upper_bound, n_bins + 1)
            
            clipped_pulls = np.clip(pulls, lower_bound, upper_bound)
            counts, bin_edges = np.histogram(clipped_pulls, bins=bins, density=False)
            
            x_step = np.concatenate([bin_edges[:-1], [bin_edges[-1]]])
            y_step = np.concatenate([counts, [counts[-1] if len(counts) > 0 else 0]])
            
            ax.step(x_step, y_step, where="post", linewidth=1.5, color="steelblue")
            
            mean_pull = np.mean(pulls)
            std_pull = np.std(pulls)
            ax.axvline(mean_pull, color="red", linestyle="--", lw=1.5, label=f"μ={mean_pull:.3f}")
            ax.axvline(0, color="black", linestyle="-", alpha=0.5, lw=1)
            
            ax.set_xlabel(param_labels.get(param, param), fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(f"{param.upper()}: σ={std_pull:.3f}", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle("Track Parameter Pull Distributions (Pseudo-pulls)", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(
            self.output_dir / "pulls" / "summary_pulls.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_pulls_vs_eta(self, residuals: dict, eta_bins: np.ndarray) -> None:
        """Plot pull mean and width as a function of pseudorapidity.

        For each parameter, computes pseudo-pulls (residual / overall σ)
        then bins by η to show:
          - Top panel: mean of pull per η bin (bias; ideal = 0)
          - Bottom panel: std of pull per η bin (width; ideal = 1)
        """
        params_to_plot = ["d0", "z0", "phi", "theta", "qop", "pt_rel"]
        params_available = [p for p in params_to_plot if p in residuals]

        if not params_available or "eta" not in residuals:
            return

        eta = residuals["eta"]

        param_labels = {
            "d0": r"$d_0$",
            "z0": r"$z_0$",
            "phi": r"$\phi$",
            "theta": r"$\theta$",
            "qop": r"$q/p$",
            "pt_rel": r"$p_T$",
        }

        for param_name in params_available:
            res_values = residuals[param_name]
            if len(res_values) == 0:
                continue

            std_dev = np.std(res_values)
            if std_dev == 0:
                continue

            pulls = res_values / std_dev
            min_len = min(len(pulls), len(eta))
            pulls, param_eta = pulls[:min_len], eta[:min_len]

            centers, means, stds, counts = [], [], [], []
            for i in range(len(eta_bins) - 1):
                mask = (param_eta >= eta_bins[i]) & (param_eta < eta_bins[i + 1])
                if np.sum(mask) > 2:
                    bin_pulls = pulls[mask]
                    centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)
                    means.append(np.mean(bin_pulls))
                    stds.append(np.std(bin_pulls))
                    counts.append(np.sum(mask))

            if not centers:
                continue

            centers = np.array(centers)
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            counts_arr = np.array(counts, dtype=float)
            mean_err = stds_arr / np.sqrt(counts_arr)
            std_err = stds_arr / np.sqrt(2 * counts_arr)

            edges = self._get_bin_edges_for_centers(centers, eta_bins)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                           gridspec_kw={"height_ratios": [1, 1]})

            # Top: mean pull vs eta
            x1, y1 = self._build_step_arrays(edges, means_arr)
            ax1.step(x1, y1, where="post", color="steelblue", linewidth=2.5,
                     label=f"Mean pull (avg: {np.mean(pulls):.3f})")
            self._step_fill_between(ax1, edges, means_arr - mean_err,
                                    means_arr + mean_err, color="steelblue",
                                    alpha=0.25, label="SEM")
            ax1.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Ideal (0)")
            ax1.set_ylabel("Mean (pull)", fontsize=12)
            ax1.set_title(f"{param_name.upper()} Pull vs $\\eta$", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)

            # Bottom: std pull vs eta
            x2, y2 = self._build_step_arrays(edges, stds_arr)
            ax2.step(x2, y2, where="post", color="darkorange", linewidth=2.5,
                     label=f"Pull width (avg: {np.std(pulls):.3f})")
            self._step_fill_between(ax2, edges, stds_arr - std_err,
                                    stds_arr + std_err, color="darkorange",
                                    alpha=0.25, label="Uncertainty")
            ax2.axhline(1, color="gray", linestyle="--", alpha=0.7, label="Ideal (1)")
            ax2.set_xlabel(r"$\eta_{truth}$", fontsize=12)
            ax2.set_ylabel("Std (pull)", fontsize=12)
            ax2.set_ylim(bottom=0)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=9)

            plt.tight_layout()
            fig.savefig(
                self.output_dir / "pulls" / f"{param_name}_pull_vs_eta.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Summary: all params on one figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, param_name in enumerate(params_available):
            ax = axes[i]
            res_values = residuals[param_name]
            std_dev = np.std(res_values)
            if std_dev == 0 or len(res_values) == 0:
                continue

            pulls = res_values / std_dev
            min_len = min(len(pulls), len(eta))
            pulls_b, eta_b = pulls[:min_len], eta[:min_len]

            centers, stds_b, counts_b = [], [], []
            for j in range(len(eta_bins) - 1):
                mask = (eta_b >= eta_bins[j]) & (eta_b < eta_bins[j + 1])
                if np.sum(mask) > 2:
                    centers.append((eta_bins[j] + eta_bins[j + 1]) / 2)
                    stds_b.append(np.std(pulls_b[mask]))
                    counts_b.append(np.sum(mask))

            if not centers:
                continue

            centers_arr = np.array(centers)
            stds_b_arr = np.array(stds_b)
            counts_b_arr = np.array(counts_b, dtype=float)
            std_err_b = stds_b_arr / np.sqrt(2 * counts_b_arr)

            edges = self._get_bin_edges_for_centers(centers_arr, eta_bins)
            x, y = self._build_step_arrays(edges, stds_b_arr)
            ax.step(x, y, where="post", color="darkorange", linewidth=2,
                    label=f"Width (avg: {np.std(pulls):.3f})")
            self._step_fill_between(ax, edges, stds_b_arr - std_err_b,
                                    stds_b_arr + std_err_b, color="darkorange",
                                    alpha=0.25)
            ax.axhline(1, color="gray", linestyle="--", alpha=0.7)
            ax.set_xlabel(r"$\eta$", fontsize=11)
            ax.set_ylabel("Pull width", fontsize=11)
            label = param_labels.get(param_name, param_name)
            ax.set_title(f"{label} pull width vs $\\eta$", fontsize=12)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        for j in range(len(params_available), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Pull Width vs $\\eta$", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(
            self.output_dir / "pulls" / "summary_pull_vs_eta.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def save_statistics(self, stats: dict) -> None:
        """Save statistics summary to text file."""
        output_path = self.output_dir / "evaluation_summary.txt"

        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("ACTS Track Reconstruction Evaluation Summary\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Data directory: {self.config['data']['directory']}\n")
            f.write(f"Number of events: {stats.get('n_events', 'N/A')}\n\n")

            f.write("-" * 80 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            f.write(f"  Total tracks: {stats.get('n_tracks', 'N/A')}\n")
            f.write(f"  Matched tracks: {stats.get('n_matched', 'N/A')}\n")
            f.write(f"  Fake tracks: {stats.get('n_fake', 'N/A')}\n")
            f.write(f"  Primary particles with hits: {stats.get('n_primary_particles_with_hits', 'N/A')}\n")
            f.write(f"  Reconstructible particles: {stats.get('n_reconstructible', 'N/A')}\n")
            f.write(f"  Overall efficiency: {stats.get('unweighted_efficiency_eta3', 'N/A'):.3f} (unweighted, |eta| < 3)\n")
            f.write(f"  Reconstructible particles (|eta| < 3): {stats.get('n_reconstructible_eta3', 'N/A')}\n")
            f.write(f"  Found particles (|eta| < 3): {stats.get('n_found_eta3', 'N/A')}\n")
            f.write(f"  Overall fake rate: {stats.get('overall_fake_rate', 'N/A'):.3f}\n\n")

            f.write("-" * 80 + "\n")
            f.write("RESIDUAL STATISTICS\n")
            f.write("-" * 80 + "\n\n")

            unit_labels = {"d0": "mm", "z0": "mm", "phi": "mrad", "theta": "mrad", "qop": "1/GeV", "pt_rel": ""}
            for param, res_stats in stats.get("residuals", {}).items():
                scale = self.UNIT_SCALE.get(param, 1.0)
                unit = unit_labels.get(param, "")
                unit_str = f" {unit}" if unit else ""
                f.write(f"  {param}:\n")
                f.write(f"    mean: {res_stats['mean'] * scale:.6f}{unit_str}\n")
                f.write(f"    std: {res_stats['std'] * scale:.6f}{unit_str}\n")
                f.write(f"    count: {res_stats['count']}\n\n")

            # Resolution statistics (unbinned relative residual averages)
            resolution_data = stats.get("resolution_data", {})
            if resolution_data:
                f.write("-" * 80 + "\n")
                f.write("RESOLUTION STATISTICS (unbinned mean relative residual)\n")
                f.write("-" * 80 + "\n\n")

                for param in ["d0", "z0", "phi", "theta", "qop", "pt_rel"]:
                    if param in resolution_data:
                        data = resolution_data[param]
                        unbinned_rel = data.get("unbinned_rel_mean", 0.0)
                        unbinned_std = data.get("unbinned_std", 0.0)
                        scale = self.UNIT_SCALE.get(param, 1.0)
                        unit = unit_labels.get(param, "")
                        unit_str = f" {unit}" if unit else ""
                        f.write(f"  {param}:\n")
                        f.write(f"    mean relative residual: {unbinned_rel * 100:.4f}%\n")
                        f.write(f"    precision (std):        {unbinned_std * scale:.6f}{unit_str}\n\n")

            # Double-matched track statistics
            matching_info = stats.get("matching_info", {})
            dm_residuals_stats = stats.get("dm_residuals", {})
            dm_res_data = stats.get("dm_resolution_data", {})
            if matching_info and dm_residuals_stats:
                f.write("-" * 80 + "\n")
                f.write("DOUBLE-MATCHED TRACK STATISTICS\n")
                f.write("-" * 80 + "\n\n")

                f.write("  Method: Double matching using hit content\n")
                f.write(f"    Purity threshold:     > {matching_info.get('purity_threshold', 0):.0%}\n")
                f.write(f"    Hit eff. threshold:   > {matching_info.get('efficiency_threshold', 0):.0%}\n")
                f.write(f"    Pass purity:          {matching_info.get('n_purity_pass', 'N/A')} / {matching_info.get('n_total', 'N/A')}\n")
                f.write(f"    Pass hit efficiency:  {matching_info.get('n_efficiency_pass', 'N/A')} / {matching_info.get('n_total', 'N/A')}\n")
                f.write(f"    Double-matched:       {matching_info.get('n_double_matched', 'N/A')} / "
                        f"{matching_info.get('n_total', 'N/A')} "
                        f"({matching_info.get('fraction_double_matched', 0)*100:.1f}%)\n")
                f.write(f"    Mean purity:          {matching_info.get('mean_purity', 0):.3f}\n")
                f.write(f"    Mean hit efficiency:  {matching_info.get('mean_hit_efficiency', 0):.3f}\n\n")

                f.write("  Definition:\n")
                f.write("    Purity = (# track hits from majority particle) / (# total track hits)\n")
                f.write("    Hit eff = (# track hits from majority particle) / (# total particle hits in event)\n")
                f.write("    A track is double-matched if both purity AND hit efficiency exceed\n")
                f.write("    their respective thresholds. This ensures the truth assignment is\n")
                f.write("    reliable: the track is dominated by one particle (purity) and\n")
                f.write("    recovers a significant fraction of that particle's hits (efficiency).\n\n")

                f.write("  Double-matched residuals:\n\n")
                for param in ["d0", "z0", "phi", "theta", "qop", "pt_rel", "pt_abs"]:
                    if param in dm_residuals_stats:
                        cs = dm_residuals_stats[param]
                        scale = self.UNIT_SCALE.get(param, 1.0)
                        unit = unit_labels.get(param, "")
                        if param == "pt_abs":
                            unit = "GeV"
                        unit_str = f" {unit}" if unit else ""
                        f.write(f"    {param}:\n")
                        f.write(f"      mean: {cs['mean'] * scale:.6f}{unit_str}\n")
                        f.write(f"      std:  {cs['std'] * scale:.6f}{unit_str}\n")
                        f.write(f"      count: {cs['count']}\n\n")

                if dm_res_data:
                    f.write("  Double-matched resolution (unbinned mean relative residual):\n\n")
                    for param in ["d0", "z0", "phi", "theta", "qop", "pt_rel"]:
                        if param in dm_res_data:
                            data = dm_res_data[param]
                            unbinned_rel = data.get("unbinned_rel_mean", 0.0)
                            unbinned_std = data.get("unbinned_std", 0.0)
                            scale = self.UNIT_SCALE.get(param, 1.0)
                            unit = unit_labels.get(param, "")
                            unit_str = f" {unit}" if unit else ""
                            f.write(f"    {param}:\n")
                            f.write(f"      mean relative residual: {unbinned_rel * 100:.4f}%\n")
                            f.write(f"      precision (std):        {unbinned_std * scale:.6f}{unit_str}\n\n")

            f.write("=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")

        print(f"Statistics saved to: {output_path}")

    def plot_pt_abs_precision_vs_eta(self, residuals: dict, eta_bins: np.ndarray,
                                    subdir: str = "precision", title_suffix: str = "") -> None:
        """Plot absolute pT precision (std of pt_reco - pt_truth) [GeV] vs eta."""
        if "pt_abs" not in residuals or "eta" not in residuals:
            print("Warning: pt_abs or eta not in residuals, skipping pt abs precision vs eta")
            return

        pt_abs = residuals["pt_abs"]
        eta = residuals["eta"]
        min_len = min(len(pt_abs), len(eta))
        pt_abs, eta = pt_abs[:min_len], eta[:min_len]

        centers, stds, counts = [], [], []
        for i in range(len(eta_bins) - 1):
            mask = (eta >= eta_bins[i]) & (eta < eta_bins[i + 1])
            if np.sum(mask) > 2:
                bin_vals = pt_abs[mask]
                centers.append((eta_bins[i] + eta_bins[i + 1]) / 2)
                stds.append(np.std(bin_vals))
                counts.append(np.sum(mask))

        if not centers:
            return

        centers = np.array(centers)
        stds = np.array(stds)
        counts_arr = np.array(counts, dtype=float)
        std_err = stds / np.sqrt(2 * counts_arr)

        unbinned_std = float(np.std(pt_abs))

        edges = self._get_bin_edges_for_centers(centers, eta_bins)
        x, y = self._build_step_arrays(edges, stds)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(x, y, where="post", color="steelblue", linewidth=2.5,
                label=f"$\\sigma(\\Delta p_T)$ (avg: {unbinned_std:.4f} GeV)")
        self._step_fill_between(ax, edges, stds - std_err, stds + std_err,
                                color="steelblue", alpha=0.25, label="Uncertainty")

        ax.set_xlabel(r"$\eta_{truth}$", fontsize=12)
        ax.set_ylabel(r"$\sigma(p_T^{reco} - p_T^{truth})$ [GeV]", fontsize=12)
        ax.set_title(f"Absolute $p_T$ Precision vs $\\eta${title_suffix}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend()

        plt.tight_layout()
        fig.savefig(
            self.output_dir / subdir / "pt_abs_precision_vs_eta.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_pt_abs_precision_vs_pt(self, residuals: dict,
                                   subdir: str = "precision", title_suffix: str = "") -> None:
        """Plot absolute pT precision (std of pt_reco - pt_truth) [GeV] vs pt_truth."""
        if "pt_abs" not in residuals or "pt_truth" not in residuals:
            print("Warning: pt_abs or pt_truth not in residuals, skipping pt abs precision vs pt")
            return

        pt_abs = residuals["pt_abs"]
        pt_truth = residuals["pt_truth"]
        min_len = min(len(pt_abs), len(pt_truth))
        pt_abs, pt_truth = pt_abs[:min_len], pt_truth[:min_len]

        pt_min = max(1.0, np.min(pt_truth))
        pt_max = np.max(pt_truth)
        n_pt_bins = 25
        pt_bins = np.linspace(pt_min, pt_max, n_pt_bins + 1)

        centers, stds, counts = [], [], []
        for i in range(len(pt_bins) - 1):
            mask = (pt_truth >= pt_bins[i]) & (pt_truth < pt_bins[i + 1])
            if np.sum(mask) > 2:
                bin_vals = pt_abs[mask]
                centers.append((pt_bins[i] + pt_bins[i + 1]) / 2)
                stds.append(np.std(bin_vals))
                counts.append(np.sum(mask))

        if not centers:
            return

        centers = np.array(centers)
        stds = np.array(stds)
        counts_arr = np.array(counts, dtype=float)
        std_err = stds / np.sqrt(2 * counts_arr)

        unbinned_std = float(np.std(pt_abs))

        # Build contiguous bin edges for plotted bins
        pt_edges = []
        for c in centers:
            idx = np.argmin(np.abs((pt_bins[:-1] + pt_bins[1:]) / 2 - c))
            if not pt_edges or pt_edges[-1] != pt_bins[idx]:
                pt_edges.append(pt_bins[idx])
        last_idx = np.argmin(np.abs((pt_bins[:-1] + pt_bins[1:]) / 2 - centers[-1]))
        pt_edges.append(pt_bins[last_idx + 1])
        pt_edges = np.array(pt_edges)

        x, y = self._build_step_arrays(pt_edges, stds)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.step(x, y, where="post", color="steelblue", linewidth=2.5,
                label=f"$\\sigma(\\Delta p_T)$ (avg: {unbinned_std:.4f} GeV)")
        self._step_fill_between(ax, pt_edges, stds - std_err, stds + std_err,
                                color="steelblue", alpha=0.25, label="Uncertainty")

        ax.set_xlabel(r"$p_T^{truth}$ [GeV]", fontsize=12)
        ax.set_ylabel(r"$\sigma(p_T^{reco} - p_T^{truth})$ [GeV]", fontsize=12)
        ax.set_title(f"Absolute $p_T$ Precision vs $p_T^{{truth}}${title_suffix}", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend()

        plt.tight_layout()
        fig.savefig(
            self.output_dir / subdir / "pt_abs_precision_vs_pt.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_pt_resolution_vs_pt(self, residuals: dict) -> None:
        """Plot pt resolution (mean relative residual) as a function of truth pt."""
        if "pt_rel" not in residuals or "pt_truth" not in residuals:
            print("Warning: pt_rel or pt_truth not in residuals, skipping pt resolution vs pt")
            return

        pt_rel = residuals["pt_rel"]
        pt_truth = residuals["pt_truth"]

        min_len = min(len(pt_rel), len(pt_truth))
        pt_rel = pt_rel[:min_len]
        pt_truth = pt_truth[:min_len]

        # Use linear bins in pt from 1 GeV to max pt
        pt_min = max(1.0, np.min(pt_truth))
        pt_max = np.max(pt_truth)
        n_pt_bins = 25
        pt_bins = np.linspace(pt_min, pt_max, n_pt_bins + 1)

        centers = []
        rel_means = []
        stds = []
        counts = []

        for i in range(len(pt_bins) - 1):
            mask = (pt_truth >= pt_bins[i]) & (pt_truth < pt_bins[i + 1])
            if np.sum(mask) > 2:
                bin_rel = pt_rel[mask]
                centers.append((pt_bins[i] + pt_bins[i + 1]) / 2)  # arithmetic center
                rel_means.append(np.mean(bin_rel))
                stds.append(np.std(bin_rel))
                counts.append(np.sum(mask))

        if not centers:
            return

        centers = np.array(centers)
        rel_means = np.array(rel_means)
        stds = np.array(stds)
        counts_arr = np.array(counts, dtype=float)

        # Uncertainties
        rel_mean_err = stds / np.sqrt(counts_arr)  # SEM
        std_err = stds / np.sqrt(2 * counts_arr)    # uncertainty on std

        unbinned_avg = float(np.mean(pt_rel)) * 100  # convert to percentage
        unbinned_std = float(np.std(pt_rel)) * 100

        # Build contiguous bin edges for plotted bins
        pt_edges = []
        for c in centers:
            idx = np.argmin(np.abs((pt_bins[:-1] + pt_bins[1:]) / 2 - c))
            if not pt_edges or pt_edges[-1] != pt_bins[idx]:
                pt_edges.append(pt_bins[idx])
        last_idx = np.argmin(np.abs((pt_bins[:-1] + pt_bins[1:]) / 2 - centers[-1]))
        pt_edges.append(pt_bins[last_idx + 1])
        pt_edges = np.array(pt_edges)

        # --- Resolution (mean relative residual) vs pt ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        x1, y1 = self._build_step_arrays(pt_edges, rel_means * 100)
        ax1.step(x1, y1, where="post", color="steelblue", linewidth=2.5,
                 label=f"Resolution (avg: {unbinned_avg:.2f}%)")
        self._step_fill_between(ax1, pt_edges,
                                (rel_means - rel_mean_err) * 100,
                                (rel_means + rel_mean_err) * 100,
                                color="steelblue", alpha=0.25, label="SEM")

        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel(r"$\langle \Delta p_T / p_T \rangle$ [%]", fontsize=12)
        ax1.set_title(r"$p_T$ Resolution vs $p_T^{truth}$", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Bottom panel: std (precision) vs pt
        x2, y2 = self._build_step_arrays(pt_edges, stds * 100)
        ax2.step(x2, y2, where="post", color="darkorange", linewidth=2.5,
                 label=f"Precision (avg \u03c3: {unbinned_std:.2f}%)")
        self._step_fill_between(ax2, pt_edges,
                                (stds - std_err) * 100,
                                (stds + std_err) * 100,
                                color="darkorange", alpha=0.25, label="Uncertainty")

        ax2.set_xlabel(r"$p_T^{truth}$ [GeV]", fontsize=12)
        ax2.set_ylabel(r"$\sigma(\Delta p_T / p_T)$ [%]", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        ax2.legend()

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "resolution" / "pt_resolution_vs_pt_truth.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_matching_quality_vs_eta(
        self,
        dm_data: dict,
        eta_bins: np.ndarray,
        matched_tracks: pl.DataFrame,
    ) -> None:
        """Plot hit purity, hit efficiency, and double-match fraction vs eta.

        Produces three step-style plots in the ``hit_assignment/`` subdirectory.
        Only matched tracks (not fakes) are included.
        """
        # Get per-track eta from matched_tracks (aligned with dm_data arrays)
        if "eta" in matched_tracks.columns:
            mt_eta = matched_tracks["eta"].to_numpy()
        elif "theta" in matched_tracks.columns:
            theta = matched_tracks["theta"].to_numpy()
            mt_eta = -np.log(np.tan(theta / 2))
        else:
            print("  Warning: no eta/theta in matched_tracks, skipping matching quality plots")
            return

        # Build a lookup: (event_id, track_id) -> index in matched_tracks
        mt_event_ids = matched_tracks["event_id"].to_numpy()
        mt_track_ids = matched_tracks["track_id"].to_numpy()
        mt_is_matched = matched_tracks["is_matched"].to_numpy()

        dm_event_ids = dm_data["event_id"]
        dm_track_ids = dm_data["track_id"]
        purities = dm_data["purity"]
        hit_effs = dm_data["hit_efficiency"]
        dm_flags = dm_data["is_double_matched"]

        # Map dm_data entries to matched_tracks indices to retrieve eta
        # Both are ordered per-event, so build a dict for unambiguous lookup
        mt_lookup = {}
        for i in range(len(mt_event_ids)):
            mt_lookup[(int(mt_event_ids[i]), int(mt_track_ids[i]))] = i

        # Align: get eta for each dm_data entry (only matched tracks)
        eta_aligned = np.full(len(dm_event_ids), np.nan)
        is_matched_aligned = np.zeros(len(dm_event_ids), dtype=bool)
        for j in range(len(dm_event_ids)):
            key = (int(dm_event_ids[j]), int(dm_track_ids[j]))
            if key in mt_lookup:
                idx = mt_lookup[key]
                eta_aligned[j] = mt_eta[idx]
                is_matched_aligned[j] = mt_is_matched[idx]

        # Only keep matched (non-fake) tracks
        valid = is_matched_aligned & ~np.isnan(eta_aligned)
        eta_valid = eta_aligned[valid]
        pur_valid = purities[valid]
        eff_valid = hit_effs[valid]
        dm_valid = dm_flags[valid]

        # Compute binned quantities
        n_bins = len(eta_bins) - 1
        mean_purity = np.zeros(n_bins)
        mean_hiteff = np.zeros(n_bins)
        dm_fraction = np.zeros(n_bins)
        purity_err = np.zeros(n_bins)
        hiteff_err = np.zeros(n_bins)
        dm_frac_err = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (eta_valid >= eta_bins[i]) & (eta_valid < eta_bins[i + 1])
            n = np.sum(mask)
            if n > 0:
                mean_purity[i] = np.mean(pur_valid[mask])
                mean_hiteff[i] = np.mean(eff_valid[mask])
                dm_fraction[i] = np.mean(dm_valid[mask])
                purity_err[i] = np.std(pur_valid[mask]) / np.sqrt(n)
                hiteff_err[i] = np.std(eff_valid[mask]) / np.sqrt(n)
                # Binomial error for fraction
                p = dm_fraction[i]
                dm_frac_err[i] = np.sqrt(p * (1 - p) / n)

        # Global averages for legend
        avg_purity = float(np.mean(pur_valid)) if len(pur_valid) > 0 else 0.0
        avg_hiteff = float(np.mean(eff_valid)) if len(eff_valid) > 0 else 0.0
        avg_dm = float(np.mean(dm_valid)) if len(dm_valid) > 0 else 0.0

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

        # --- Mean Purity vs eta ---
        x1, y1 = self._build_step_arrays(eta_bins, mean_purity)
        ax1.step(x1, y1, where="post", color="steelblue", linewidth=2.5,
                 label=f"Mean purity (avg: {avg_purity:.3f})")
        self._step_fill_between(ax1, eta_bins,
                                np.clip(mean_purity - purity_err, 0, 1),
                                np.clip(mean_purity + purity_err, 0, 1),
                                color="steelblue", alpha=0.25, label="Std error")
        ax1.axhline(0.5, color="red", linestyle="--", lw=1, alpha=0.6, label="50% threshold")
        ax1.set_xlabel(r"$\eta$", fontsize=12)
        ax1.set_ylabel("Mean Hit Purity", fontsize=12)
        ax1.set_title("Track Hit Purity vs $\\eta$", fontsize=14)
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- Mean Hit Efficiency vs eta ---
        x2, y2 = self._build_step_arrays(eta_bins, mean_hiteff)
        ax2.step(x2, y2, where="post", color="darkorange", linewidth=2.5,
                 label=f"Mean hit eff. (avg: {avg_hiteff:.3f})")
        self._step_fill_between(ax2, eta_bins,
                                np.clip(mean_hiteff - hiteff_err, 0, 1),
                                np.clip(mean_hiteff + hiteff_err, 0, 1),
                                color="darkorange", alpha=0.25, label="Std error")
        ax2.axhline(0.5, color="red", linestyle="--", lw=1, alpha=0.6, label="50% threshold")
        ax2.set_xlabel(r"$\eta$", fontsize=12)
        ax2.set_ylabel("Mean Hit Efficiency", fontsize=12)
        ax2.set_title("Track Hit Efficiency vs $\\eta$", fontsize=14)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # --- Double-match fraction vs eta ---
        x3, y3 = self._build_step_arrays(eta_bins, dm_fraction)
        ax3.step(x3, y3, where="post", color="green", linewidth=2.5,
                 label=f"DM fraction (avg: {avg_dm:.3f})")
        self._step_fill_between(ax3, eta_bins,
                                np.clip(dm_fraction - dm_frac_err, 0, 1),
                                np.clip(dm_fraction + dm_frac_err, 0, 1),
                                color="green", alpha=0.25, label="Binomial error")
        ax3.set_xlabel(r"$\eta$", fontsize=12)
        ax3.set_ylabel("Double-matched Fraction", fontsize=12)
        ax3.set_title("Double-match Fraction vs $\\eta$", fontsize=14)
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        fig.savefig(
            self.output_dir / "hit_assignment" / "matching_quality_vs_eta.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def run(self) -> None:
        """Run the full evaluation pipeline."""
        print("=" * 80)
        print("ACTS Track Reconstruction Evaluation")
        print("=" * 80)

        # Step 1: Load data
        print("\n[1/8] Loading data...")
        particles_df, hits_df, tracks_df = self.load_data()

        # Explode tracks for track-level analysis (NOT hit_ids - we want one row per track)
        # Only explode scalar list columns, not hit_ids
        tracks_exploded = self._explode_dataframe(tracks_df)
        
        # If hit_ids was exploded, we need to get unique tracks back
        # by selecting unique (event_id, track_id) combinations
        if "track_id" in tracks_exploded.columns:
            # Get unique tracks (first row per track)
            track_cols = [c for c in tracks_exploded.columns if c != "hit_ids"]
            tracks_unique = tracks_exploded.group_by(["event_id", "track_id"]).first()
        else:
            tracks_unique = tracks_exploded
        
        # Calculate pt for tracks (from theta and qop)
        if "theta" in tracks_unique.columns and "qop" in tracks_unique.columns:
            # pt = |q| / (|qop| * sin(theta)) = sin(theta) / |qop|
            tracks_unique = tracks_unique.with_columns(
                (pl.col("theta").sin() / pl.col("qop").abs()).alias("pt")
            )
            # eta from theta
            tracks_unique = tracks_unique.with_columns(
                (-((pl.col("theta") / 2).tan()).log()).alias("eta")
            )

        # Step 2: Match tracks to particles
        print("\n[2/8] Matching tracks to particles...")
        matched_tracks = self.match_tracks_to_particles(tracks_unique, particles_df, hits_df)
        n_matched = matched_tracks.filter(pl.col("is_matched")).height
        n_fake = matched_tracks.filter(~pl.col("is_matched")).height
        n_total = len(matched_tracks)
        print(f"  Total unique tracks: {n_total}")
        print(f"  Matched tracks: {n_matched} ({100*n_matched/n_total:.1f}%)")
        print(f"  Fake tracks: {n_fake} ({100*n_fake/n_total:.1f}%)")

        # Step 3: Compute residuals
        print("\n[3/8] Computing residuals...")
        print("  pT calculation:")
        print("    pt_reco  = sin(theta) / |qop|         (from track perigee parameters)")
        print("    pt_truth = sqrt(px^2 + py^2)           (from truth particle momentum)")
        residuals = self.compute_residuals(matched_tracks)
        for param, res in residuals.items():
            if param != "eta" and not param.endswith("_truth") and not param.endswith("_reco"):
                print(f"  {param}: mean={np.mean(res):.6f}, std={np.std(res):.6f}")

        # Step 4: Compute double matching (hit purity + hit efficiency)
        print("\n[4/8] Computing double matching (purity > 50% AND hit efficiency > 50%)...")
        dm_data = self.compute_double_matching(tracks_df, hits_df)
        mi = dm_data["_matching_info"]
        print(f"  Total tracks: {mi['n_total']}")
        print(f"  Pass purity > {mi['purity_threshold']:.0%}: {mi['n_purity_pass']} ({mi['n_purity_pass']/mi['n_total']*100:.1f}%)")
        print(f"  Pass hit eff > {mi['efficiency_threshold']:.0%}: {mi['n_efficiency_pass']} ({mi['n_efficiency_pass']/mi['n_total']*100:.1f}%)")
        print(f"  Double-matched: {mi['n_double_matched']} / {mi['n_total']} ({mi['fraction_double_matched']*100:.1f}%)")
        print(f"  Mean purity: {mi['mean_purity']:.3f}")
        print(f"  Mean hit efficiency: {mi['mean_hit_efficiency']:.3f}")

        # Build a mask aligned with the matched_tracks rows (which were exploded)
        # dm_data has one entry per (event_id, track_id) — join on those keys
        dm_lookup = {}
        for i in range(len(dm_data["event_id"])):
            dm_lookup[(int(dm_data["event_id"][i]), int(dm_data["track_id"][i]))] = dm_data["is_double_matched"][i]

        # Build mask aligned with the residuals (which come from matched_tracks filtered to is_matched)
        matched_only = matched_tracks.filter(pl.col("is_matched"))
        dm_mask_residuals = np.array([
            dm_lookup.get((int(row["event_id"]), int(row["track_id"])), False)
            for row in matched_only.select(["event_id", "track_id"]).iter_rows(named=True)
        ], dtype=bool)

        # Apply mask to residuals
        dm_residuals = self.apply_double_matching(residuals, dm_mask_residuals)
        dmi = dm_residuals["_dm_info"]
        print(f"\n  Double-matched residuals: {dmi['n_double_matched']} / {dmi['n_total']} tracks")
        for param in ["d0", "z0", "phi", "theta", "qop", "pt_rel"]:
            if param in dm_residuals and isinstance(dm_residuals[param], np.ndarray):
                res = dm_residuals[param]
                print(f"  {param} (double-matched): mean={np.mean(res):.6f}, std={np.std(res):.6f}")

        # Step 5: Compute resolution vs eta (full + double-matched)
        print("\n[5/8] Computing resolution vs eta...")
        resolution_data = self.compute_resolution_vs_eta(residuals)
        dm_resolution_data = self.compute_resolution_vs_eta(dm_residuals)

        # Step 6: Compute efficiency and fake rate
        print("\n[6/8] Computing efficiency and fake rate...")
        eff_data = self.compute_efficiency_vs_eta(matched_tracks, particles_df, hits_df)

        # Get eta bins for plotting
        eta_range = self.config["binning"]["eta_range"]
        n_eta_bins = self.config["binning"]["n_eta_bins"]
        eta_bins = np.linspace(eta_range[0], eta_range[1], n_eta_bins + 1)

        # Step 7: Create plots (full)
        print("\n[7/8] Creating plots...")
        self.plot_residuals(residuals)
        self.plot_resolution_vs_eta(resolution_data, eta_bins)
        self.plot_precision_vs_eta(resolution_data, eta_bins)
        self.plot_efficiency_and_fake_rate(eff_data)
        self.plot_summary_residuals(residuals)
        self.plot_summary_resolution(resolution_data, eta_bins)
        self.plot_pt_resolution_vs_pt(residuals)
        self.plot_pt_abs_precision_vs_eta(residuals, eta_bins)
        self.plot_pt_abs_precision_vs_pt(residuals)
        self.plot_pulls(residuals, eta_bins)
        self.plot_pulls_vs_eta(residuals, eta_bins)

        # Step 8: Create double-matched plots + hit assignment quality
        print("\n[8/8] Creating double-matched and hit assignment plots...")
        self.plot_resolution_vs_eta(dm_resolution_data, eta_bins, subdir="resolution_double_matched", title_suffix=" (Double-matched)")
        self.plot_precision_vs_eta(dm_resolution_data, eta_bins, subdir="precision_double_matched", title_suffix=" (Double-matched)")
        self.plot_pt_abs_precision_vs_eta(dm_residuals, eta_bins, subdir="precision_double_matched", title_suffix=" (Double-matched)")
        self.plot_pt_abs_precision_vs_pt(dm_residuals, subdir="precision_double_matched", title_suffix=" (Double-matched)")
        self.plot_matching_quality_vs_eta(dm_data, eta_bins, matched_tracks)

        # Compute overall statistics
        stats = {
            "n_events": self.config["data"]["n_events"],
            "n_tracks": len(matched_tracks),
            "n_matched": n_matched,
            "n_fake": n_fake,
            "n_primary_particles_with_hits": eff_data["efficiency"].get("n_primary_particles_with_hits", "N/A"),
            "n_reconstructible": eff_data["efficiency"].get("n_reconstructible_total", "N/A"),
            "overall_efficiency": np.mean(eff_data["efficiency"]["efficiency"]),
            "overall_fake_rate": np.mean(eff_data["fake_rate"]["fake_rate"]) if eff_data["fake_rate"]["fake_rate"] else 0.0,
            "unweighted_efficiency_eta3": eff_data["efficiency"].get("unweighted_efficiency_eta3", 0.0),
            "n_reconstructible_eta3": eff_data["efficiency"].get("n_reconstructible_eta3", "N/A"),
            "n_found_eta3": eff_data["efficiency"].get("n_found_eta3", "N/A"),
            "residuals": {
                param: {"mean": np.mean(res), "std": np.std(res), "count": len(res)}
                for param, res in residuals.items()
                if param != "eta" and not param.endswith("_truth") and not param.endswith("_reco")
            },
            "resolution_data": resolution_data,
            "matching_info": dm_data["_matching_info"],
            "dm_residuals": {
                param: {"mean": float(np.mean(dm_residuals[param])),
                        "std": float(np.std(dm_residuals[param])),
                        "count": len(dm_residuals[param])}
                for param in ["d0", "z0", "phi", "theta", "qop", "pt_rel", "pt_abs"]
                if param in dm_residuals and isinstance(dm_residuals[param], np.ndarray)
            },
            "dm_resolution_data": dm_resolution_data,
        }

        self.save_statistics(stats)

        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ACTS track reconstruction on ColliderML data")
    parser.add_argument(
        "--config",
        type=str,
        default=Path(__file__).parent / "evaluation_config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    evaluator = ACTSTrackingEvaluator(args.config)
    evaluator.run()


if __name__ == "__main__":
    main()
