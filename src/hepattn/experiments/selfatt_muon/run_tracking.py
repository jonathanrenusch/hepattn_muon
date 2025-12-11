"""Training script for self-attention based particle tracking on ATLAS muon data.

This script trains an encoder-only transformer to learn hit-to-hit correlations
by predicting a self-similarity matrix that indicates which hits belong to the
same particle.
"""

import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

# Set float32 matmul precision for better performance on Tensor Core GPUs
torch.set_float32_matmul_precision('high')

from hepattn.experiments.selfatt_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class SelfAttentionTracker(ModelWrapper):
    """Lightning wrapper for the self-attention tracking model.
    
    This wrapper handles training, validation, and custom metric logging
    for the hit correlation prediction task.
    """
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        min_hits_per_track: int = 3,
    ):
        """Initialize the tracker wrapper.
        
        Args:
            name: Name of the experiment.
            model: The tracking model (HitFilter or SelfAttentionTracking).
            lrs_config: Learning rate scheduler configuration.
            optimizer: Optimizer to use ("AdamW" or "Lion").
            min_hits_per_track: Minimum hits in a row to count as a track candidate.
        """
        super().__init__(name, model, lrs_config, optimizer)
        self.min_hits_per_track = min_hits_per_track
    
    def log_custom_metrics(self, preds, targets, stage):
        """Log custom metrics for the correlation prediction task.
        
        Metrics computed (fully vectorized for performance):
        - n_pred_tracks: Number of predicted tracks (rows with >= min_hits_per_track hits)
        - n_true_tracks: Number of true tracks in the event
        - fake_rate: Percentage of predicted tracks that are fake
        - efficiency: Percentage of true tracks that were correctly predicted
        - avg_track_efficiency: Average fraction of true hits correctly assigned per track
        - avg_track_purity: Average fraction of assigned hits that are correct per track
        - double_match_rate: Percentage of tracks with both purity >= 50% and efficiency >= 50%
        
        Args:
            preds: Dictionary containing model predictions.
            targets: Dictionary containing ground truth targets.
            stage: Current stage ("train", "val", or "test").
        """
        # Get the task to access threshold
        task = self.model.tasks[0]
        threshold = task.threshold
        
        # Get predictions and targets
        pred_probs = preds["final"]["hit_correlation"][f"hit_particle_hit_corr_prob"]
        pred_matrix = pred_probs >= threshold  # [B, N, N]
        target_corr = targets["hit_particle_hit_corr"]  # [B, N, N]
        valid_hits = targets["hit_valid"]  # [B, N]
        
        batch_size = pred_matrix.shape[0]
        
        # Create valid mask for matrix operations [B, N, N]
        valid_2d = valid_hits.unsqueeze(-1) & valid_hits.unsqueeze(-2)
        
        # Mask predictions and targets
        pred_masked = pred_matrix & valid_2d
        target_masked = target_corr & valid_2d
        
        # Count hits per row [B, N]
        pred_hits_per_row = pred_masked.sum(dim=-1)
        target_hits_per_row = target_masked.sum(dim=-1)
        
        # Identify track rows (rows with >= min_hits hits) [B, N]
        pred_track_rows = (pred_hits_per_row >= self.min_hits_per_track) & valid_hits
        true_track_rows = (target_hits_per_row >= self.min_hits_per_track) & valid_hits
        
        # Count tracks
        n_pred_tracks = pred_track_rows.sum().item()
        n_true_tracks = true_track_rows.sum().item()
        
        # For efficiency/purity, we need intersection between pred and target
        # intersection[b, i, j] = pred[b, i, j] & target[b, i, j]
        intersection = pred_masked & target_masked
        
        # Per-row metrics [B, N]
        intersection_per_row = intersection.sum(dim=-1).float()
        pred_per_row = pred_masked.sum(dim=-1).float()
        target_per_row = target_masked.sum(dim=-1).float()
        
        # Track efficiency: intersection / target (for true track rows)
        # Track purity: intersection / pred (for pred track rows that are also true)
        
        # Rows that are both predicted and true tracks
        both_track_rows = pred_track_rows & true_track_rows
        
        # Compute metrics only for valid track rows
        eps = 1e-6
        track_efficiency = intersection_per_row / (target_per_row + eps)
        track_purity = intersection_per_row / (pred_per_row + eps)
        
        # Fake tracks: predicted but not true track rows
        fake_track_rows = pred_track_rows & ~true_track_rows
        n_fake_tracks = fake_track_rows.sum().item()
        
        # For matched tracks (both pred and true), compute averages
        n_both = both_track_rows.sum().item()
        
        if n_both > 0:
            # Per-track hit assignment metrics (averaged over matched tracks)
            avg_hit_eff = track_efficiency[both_track_rows].mean().item()
            avg_hit_pur = track_purity[both_track_rows].mean().item()
            
            # Double match: hit_eff >= 0.5 AND hit_purity >= 0.5
            double_matched = (track_efficiency >= 0.5) & (track_purity >= 0.5) & both_track_rows
            n_double_matched = double_matched.sum().item()
        else:
            avg_hit_eff = 0.0
            avg_hit_pur = 0.0
            n_double_matched = 0
        
        # Compute aggregate metrics
        # Naming convention:
        # - track_matching_eff: Fraction of TRUE tracks that were correctly reconstructed (recall)
        # - track_matching_pur: Fraction of PREDICTED tracks that match a true track (precision)
        # - avg_hit_eff: Average fraction of true hits correctly assigned per matched track
        # - avg_hit_purity: Average fraction of assigned hits that are correct per matched track
        metrics = {
            "n_pred_tracks": float(n_pred_tracks),
            "n_true_tracks": float(n_true_tracks),
            "fake_rate": float(n_fake_tracks) / max(n_pred_tracks, 1),
            "track_matching_eff": float(n_double_matched) / max(n_true_tracks, 1),
            "track_matching_pur": float(n_double_matched) / max(n_pred_tracks, 1),
            "avg_hit_eff": avg_hit_eff,
            "avg_hit_purity": avg_hit_pur,
        }
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}/{metric_name}", metric_value, sync_dist=True, batch_size=batch_size)


def main(args: ArgsType = None) -> None:
    """Main entry point for training."""
    CLI(
        model_class=SelfAttentionTracker,
        datamodule_class=AtlasMuonDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
