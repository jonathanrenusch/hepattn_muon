"""Training script for outward graph-based particle tracking.

This script trains an encoder-only transformer to predict directed edges
between hits on the same track, pointing outward from the interaction point.
Track extraction uses connected components instead of Hungarian matching.
"""

import torch
from lightning.pytorch.cli import ArgsType
from torch import nn
import numpy as np

# Set float32 matmul precision for better performance on Tensor Core GPUs
torch.set_float32_matmul_precision('high')

from hepattn.experiments.outward_tracking.data import OutwardTrackingDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class OutwardTracker(ModelWrapper):
    """Lightning wrapper for the outward edge tracking model.
    
    This wrapper handles training, validation, and connected components
    based track extraction and evaluation.
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
            model: The tracking model (encoder with OutwardEdgeTask).
            lrs_config: Learning rate scheduler configuration.
            optimizer: Optimizer to use ("AdamW" or "Lion").
            min_hits_per_track: Minimum hits required to count as a valid track.
        """
        super().__init__(name, model, lrs_config, optimizer)
        self.min_hits_per_track = min_hits_per_track
    
    def log_custom_metrics(self, preds, targets, stage):
        """Log custom metrics for the outward edge prediction task.
        
        Uses connected components on predicted edges to extract tracks,
        then evaluates against ground truth tracks.
        
        Metrics computed:
        - edge_precision: Fraction of predicted edges that are correct
        - edge_recall: Fraction of true edges that were predicted
        - n_pred_tracks: Number of extracted track candidates
        - n_true_tracks: Number of true tracks in the event  
        - track_efficiency: Fraction of true tracks correctly reconstructed
        - fake_rate: Fraction of predicted tracks that are fake
        - avg_track_purity: Average purity of reconstructed tracks
        - avg_track_eff: Average efficiency of reconstructed tracks
        
        Args:
            preds: Dictionary containing model predictions.
            targets: Dictionary containing ground truth targets.
            stage: Current stage ("train", "val", or "test").
        """
        # Get the task to access threshold
        task = self.model.tasks[0]
        threshold = task.threshold
        
        # Get predictions and targets
        pred_probs = preds["final"]["hit_edge"]["hit_outward_edge_prob"]
        pred_edges = pred_probs >= threshold  # [B, N, N]
        target_edges = targets["outward_adjacency"]  # [B, N, N]
        valid_hits = targets["hit_valid"]  # [B, N]
        
        # Get full adjacency for track assignment evaluation
        full_adj = targets["full_adjacency"]  # [B, N, N]
        
        batch_size = pred_edges.shape[0]
        
        # Create valid mask for matrix operations [B, N, N]
        valid_2d = valid_hits.unsqueeze(-1) & valid_hits.unsqueeze(-2)
        
        # Mask predictions and targets
        pred_masked = pred_edges & valid_2d
        target_masked = target_edges & valid_2d
        
        # ===== EDGE-LEVEL METRICS =====
        intersection = pred_masked & target_masked
        n_intersection = intersection.sum().item()
        n_pred_edges = pred_masked.sum().item()
        n_true_edges = target_masked.sum().item()
        
        edge_precision = n_intersection / max(n_pred_edges, 1)
        edge_recall = n_intersection / max(n_true_edges, 1)
        edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-6)
        
        # ===== TRACK-LEVEL METRICS (using connected components via task.extract_tracks) =====
        track_metrics = self._compute_track_metrics(
            pred_probs, valid_hits, full_adj, targets, threshold
        )
        
        # Combine all metrics
        metrics = {
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
            **track_metrics,
        }
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}/{metric_name}", metric_value, sync_dist=True, batch_size=batch_size)
    
    def _compute_track_metrics(self, pred_probs, valid_hits, full_adj, targets, threshold):
        """Extract tracks using connected components and compute track-level metrics.
        
        Uses the task's extract_tracks method for connected components.
        
        Args:
            pred_probs: Predicted edge probabilities [B, N, N]
            valid_hits: Valid hits mask [B, N]
            full_adj: Full adjacency matrix (true track assignments) [B, N, N]
            targets: Full targets dictionary
            threshold: Edge prediction threshold
        
        Returns:
            Dictionary of track-level metrics.
        """
        task = self.model.tasks[0]
        
        # Extract predicted tracks using task method
        pred_tracks_per_event = task.extract_tracks(
            pred_probs, valid_hits, min_hits=self.min_hits_per_track
        )
        
        # Extract true tracks from full adjacency (treat as edge probs of 1.0)
        true_tracks_per_event = task.extract_tracks(
            full_adj.float(), valid_hits, min_hits=self.min_hits_per_track
        )
        
        batch_size = pred_probs.shape[0]
        
        total_pred_tracks = 0
        total_true_tracks = 0
        total_matched_tracks = 0
        total_fake_tracks = 0
        all_purities = []
        all_efficiencies = []
        
        for b in range(batch_size):
            pred_tracks = pred_tracks_per_event[b]  # List of tracks, each track is list of hit indices
            true_tracks = true_tracks_per_event[b]
            
            n_pred = len(pred_tracks)
            n_true = len(true_tracks)
            
            total_pred_tracks += n_pred
            total_true_tracks += n_true
            
            # Match predicted tracks to true tracks
            for pred_track in pred_tracks:
                pred_hits = set(pred_track)
                
                # Find best matching true track
                best_purity = 0.0
                best_eff = 0.0
                is_matched = False
                
                for true_track in true_tracks:
                    true_hits = set(true_track)
                    
                    intersection = len(pred_hits & true_hits)
                    purity = intersection / len(pred_hits) if len(pred_hits) > 0 else 0
                    eff = intersection / len(true_hits) if len(true_hits) > 0 else 0
                    
                    if purity > best_purity:
                        best_purity = purity
                        best_eff = eff
                    
                    # Track is matched if purity >= 50% and efficiency >= 50%
                    if purity >= 0.5 and eff >= 0.5:
                        is_matched = True
                
                all_purities.append(best_purity)
                all_efficiencies.append(best_eff)
                
                if is_matched:
                    total_matched_tracks += 1
                else:
                    total_fake_tracks += 1
        
        # Compute aggregate metrics
        # Naming convention:
        # - track_matching_eff: Fraction of TRUE tracks that were correctly reconstructed (recall)
        # - track_matching_pur: Fraction of PREDICTED tracks that match a true track (precision)  
        # - avg_hit_eff: Average fraction of true hits correctly assigned per matched track
        # - avg_hit_purity: Average fraction of assigned hits that are correct per matched track
        metrics = {
            "n_pred_tracks": float(total_pred_tracks),
            "n_true_tracks": float(total_true_tracks),
            "fake_rate": float(total_fake_tracks) / max(total_pred_tracks, 1),
            "track_matching_eff": float(total_matched_tracks) / max(total_true_tracks, 1),
            "track_matching_pur": float(total_matched_tracks) / max(total_pred_tracks, 1),
            "avg_hit_eff": float(np.mean(all_efficiencies)) if all_efficiencies else 0.0,
            "avg_hit_purity": float(np.mean(all_purities)) if all_purities else 0.0,
        }
        
        return metrics


def main(args: ArgsType = None) -> None:
    """Main entry point for training."""
    CLI(
        model_class=OutwardTracker,
        datamodule_class=OutwardTrackingDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
