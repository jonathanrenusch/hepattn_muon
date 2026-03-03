"""YOLO Stage 2: Regression Head Training with Frozen Encoder.

This module trains only the regression heads while keeping the encoder and
classification heads frozen. The regression heads learn to predict per-bin
offsets that refine the classification predictions.

Key features:
- Multi-bin loss: Train all bins within a window around the predicted bin
- Softmax weighting: Use classification probabilities to weight bin losses
- Normalized offsets: Offsets are normalized by bin width
- Phi wrapping: Proper circular handling for phi predictions
- Clamping: Predicted bins clamped to be within k bins of truth

Training pipeline:
1. Load Stage 1 checkpoint (classification pretrained)
2. Freeze encoder + classification heads
3. Train regression heads with multi-bin weighted loss
4. Log physics metrics using combined cls + regression predictions
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning import LightningModule
from torchmetrics import Accuracy, MeanAbsoluteError
from torchmetrics.classification import BinaryAUROC as AUROC

from hepattn.models.yolo_regressor import YOLORegressor


class YOLORegressionTraining(LightningModule):
    """Stage 2 training: Regression heads with frozen encoder.
    
    Trains regression heads to predict per-bin offsets while keeping the
    encoder and classification heads frozen from Stage 1.
    
    Parameters
    ----------
    model : YOLORegressor
        Model with regression heads enabled.
    stage1_checkpoint : str
        Path to Stage 1 checkpoint to load encoder/classification weights.
    bin_dir : str
        Directory containing bin edge and center arrays.
    eta_bins, phi_bins, pt_bins : int
        Number of bins for each parameter.
    pt_log_bins : bool
        Whether to use log-spaced pt bins.
    
    Clamping parameters (predicted bin clamped to be within k of truth):
    eta_clamp_bins, phi_clamp_bins, pt_clamp_bins : int
        Maximum distance from truth bin for clamping.
    
    Multi-bin loss window (centered on clamped predicted bin):
    eta_window_bins, phi_window_bins, pt_window_bins : int
        Half-width of window for multi-bin loss.
    
    Loss parameters:
    smooth_l1_beta : float
        Beta parameter for Smooth L1 loss.
    eta_loss_weight, phi_loss_weight, pt_loss_weight : float
        Weights for each parameter's loss.
    """
    
    def __init__(
        self,
        model: dict | YOLORegressor,
        name: str = "YOLO-Stage2-Regression",
        optimizer: str = "AdamW",
        lrs_config: dict | None = None,
        # Checkpoint
        stage1_checkpoint: str = "",
        # Bin configuration
        bin_dir: str = "/scratch/ml_training_data_2694000_regression_true_hits_only/bins",
        eta_bins: int = 1001,
        phi_bins: int = 500,
        pt_bins: int = 100,
        pt_log_bins: bool = True,
        # Clipping ranges
        pt_min: float = 5.0,
        pt_max: float = 200.0,
        eta_min: float = -2.7,
        eta_max: float = 2.7,
        eta_gap_min: float = -0.1,
        eta_gap_max: float = 0.1,
        # Clamping: max bins from truth for predicted bin
        eta_clamp_bins: int = 15,
        phi_clamp_bins: int = 10,
        pt_clamp_bins: int = 5,
        # Multi-bin loss window half-width
        eta_window_bins: int = 50,
        phi_window_bins: int = 25,
        pt_window_bins: int = 10,
        # Loss configuration
        smooth_l1_beta: float = 0.1,
        eta_loss_weight: float = 1.0,
        phi_loss_weight: float = 1.0,
        pt_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.name = name
        self.stage1_checkpoint = stage1_checkpoint
        
        # Model - handle Lightning CLI's class_path/init_args structure
        if isinstance(model, dict):
            if 'class_path' in model and 'init_args' in model:
                self.model = YOLORegressor(**model['init_args'])
            else:
                self.model = YOLORegressor(**model)
        else:
            self.model = model
        
        # Verify regression heads are enabled
        if not self.model.enable_regression_heads:
            raise ValueError(
                "Model must have enable_regression_heads=True for Stage 2 training"
            )
        
        # Store config
        self.optimizer_name = optimizer
        self.lrs_config = lrs_config or {
            'initial': 1e-5,
            'max': 1e-4,
            'end': 1e-6,
            'pct_start': 0.05,
            'weight_decay': 1e-5,
        }
        
        # Bin configuration
        self.eta_bins = eta_bins
        self.phi_bins = phi_bins
        self.pt_bins = pt_bins
        self.pt_log_bins = pt_log_bins
        
        # Clipping ranges
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_gap_min = eta_gap_min
        self.eta_gap_max = eta_gap_max
        
        # Clamping configuration
        self.eta_clamp_bins = eta_clamp_bins
        self.phi_clamp_bins = phi_clamp_bins
        self.pt_clamp_bins = pt_clamp_bins
        
        # Multi-bin loss window
        self.eta_window_bins = eta_window_bins
        self.phi_window_bins = phi_window_bins
        self.pt_window_bins = pt_window_bins
        
        # Loss configuration
        self.smooth_l1_beta = smooth_l1_beta
        self.eta_loss_weight = eta_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.pt_loss_weight = pt_loss_weight
        
        # Load bin arrays
        bin_dir = Path(bin_dir)
        self._load_bin_arrays(bin_dir)
        
        # Loss function
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=smooth_l1_beta)
        
        # Accumulators for validation physics metrics
        self._reset_val_accumulators()
        
    def _reset_val_accumulators(self):
        """Reset validation metric accumulators."""
        self._val_eta_preds = []
        self._val_eta_targets = []
        self._val_phi_preds = []
        self._val_phi_targets = []
        self._val_pt_preds = []
        self._val_pt_targets = []
        
    def _load_bin_arrays(self, bin_dir: Path):
        """Load bin edge and center arrays."""
        # Map actual bin counts to file naming convention
        eta_actual_to_file = {51: 50, 101: 100, 1001: 1000, 2001: 2000, 6001: 6000, 12001: 12000}
        eta_file_bins = eta_actual_to_file.get(self.eta_bins, self.eta_bins)
        phi_file_bins = self.phi_bins
        pt_file_bins = self.pt_bins
        
        pt_suffix = '_log' if self.pt_log_bins else ''
        
        # Load arrays
        eta_edges = np.load(bin_dir / f'eta_bins_{eta_file_bins}.npy')
        phi_edges = np.load(bin_dir / f'phi_bins_{phi_file_bins}.npy')
        pt_edges = np.load(bin_dir / f'pt_bins_{pt_file_bins}{pt_suffix}.npy')
        
        eta_centers = np.load(bin_dir / f'eta_bin_centers_{eta_file_bins}.npy')
        phi_centers = np.load(bin_dir / f'phi_bin_centers_{phi_file_bins}.npy')
        pt_centers = np.load(bin_dir / f'pt_bin_centers_{pt_file_bins}{pt_suffix}.npy')
        
        # Compute bin widths
        eta_widths = np.diff(eta_edges)
        phi_widths = np.diff(phi_edges)
        pt_widths = np.diff(pt_edges)
        
        # Register as buffers
        self.register_buffer('eta_bin_edges', torch.from_numpy(eta_edges).float())
        self.register_buffer('phi_bin_edges', torch.from_numpy(phi_edges).float())
        self.register_buffer('pt_bin_edges', torch.from_numpy(pt_edges).float())
        self.register_buffer('eta_bin_centers', torch.from_numpy(eta_centers).float())
        self.register_buffer('phi_bin_centers', torch.from_numpy(phi_centers).float())
        self.register_buffer('pt_bin_centers', torch.from_numpy(pt_centers).float())
        self.register_buffer('eta_bin_widths', torch.from_numpy(eta_widths).float())
        self.register_buffer('phi_bin_widths', torch.from_numpy(phi_widths).float())
        self.register_buffer('pt_bin_widths', torch.from_numpy(pt_widths).float())
        
        pt_type = 'logarithmic' if self.pt_log_bins else 'quantile'
        print(f"Loaded bin arrays from {bin_dir}")
        print(f"  Eta: {len(eta_centers)} bins")
        print(f"  Phi: {len(phi_centers)} bins")
        print(f"  Pt:  {len(pt_centers)} bins ({pt_type})")
        
    def setup(self, stage: str):
        """Load Stage 1 checkpoint and freeze encoder/classification heads."""
        if stage == "fit" and self.stage1_checkpoint:
            print(f"Loading Stage 1 checkpoint: {self.stage1_checkpoint}")
            
            # Load checkpoint
            checkpoint = torch.load(self.stage1_checkpoint, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Filter to model weights only (remove 'model.' prefix from training module)
            model_state = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    model_state[k[6:]] = v  # Remove 'model.' prefix
            
            # Load into model (strict=False allows missing regression head weights)
            missing, unexpected = self.model.load_state_dict(model_state, strict=False)
            
            print(f"  Loaded {len(model_state)} weights")
            if missing:
                print(f"  Missing keys (expected for regression heads): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
        
        # Freeze encoder and classification heads
        print("Freezing encoder and classification heads...")
        self.model.freeze_encoder()
        self.model.freeze_classification_heads()
        
        # Verify only regression heads are trainable
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def _value_to_bin_index(self, values: Tensor, bin_edges: Tensor) -> Tensor:
        """Convert continuous values to bin indices."""
        indices = torch.searchsorted(bin_edges, values.contiguous()) - 1
        indices = indices.clamp(0, len(bin_edges) - 2)
        return indices
    
    def _wrap_bin_index_phi(self, indices: Tensor) -> Tensor:
        """Wrap phi bin indices to valid range [0, phi_bins-1]."""
        return indices % self.phi_bins
    
    def _circular_bin_distance(self, idx1: Tensor, idx2: Tensor, num_bins: int) -> Tensor:
        """Compute signed circular distance between bin indices.
        
        Returns value in range [-num_bins//2, num_bins//2].
        """
        diff = idx1 - idx2
        # Wrap to [-num_bins//2, num_bins//2]
        diff = ((diff + num_bins // 2) % num_bins) - num_bins // 2
        return diff
    
    def _clamp_predicted_bin(
        self, 
        pred_bins: Tensor, 
        true_bins: Tensor, 
        max_distance: int,
        circular: bool = False,
        num_bins: int = None,
    ) -> Tensor:
        """Clamp predicted bins to be within max_distance of true bins.
        
        Parameters
        ----------
        pred_bins : Tensor
            Predicted bin indices, shape (B,).
        true_bins : Tensor
            True bin indices, shape (B,).
        max_distance : int
            Maximum allowed distance from true bin.
        circular : bool
            If True, use circular distance (for phi).
        num_bins : int
            Number of bins (required if circular=True).
            
        Returns
        -------
        Tensor
            Clamped bin indices, shape (B,).
        """
        if circular:
            # Circular distance
            distance = self._circular_bin_distance(pred_bins, true_bins, num_bins)
            clamped_distance = distance.clamp(-max_distance, max_distance)
            clamped_bins = self._wrap_bin_index_phi(true_bins + clamped_distance)
        else:
            # Linear distance
            distance = pred_bins - true_bins
            clamped_distance = distance.clamp(-max_distance, max_distance)
            clamped_bins = true_bins + clamped_distance
            # Clamp to valid range
            clamped_bins = clamped_bins.clamp(0, num_bins - 1)
        
        return clamped_bins
    
    def _compute_target_offsets(
        self,
        true_values: Tensor,
        bin_centers: Tensor,
        bin_widths: Tensor,
        circular: bool = False,
    ) -> Tensor:
        """Compute normalized target offsets for all bins.
        
        Parameters
        ----------
        true_values : Tensor
            True parameter values, shape (B,).
        bin_centers : Tensor
            Bin centers, shape (num_bins,).
        bin_widths : Tensor
            Bin widths, shape (num_bins,).
        circular : bool
            If True, use circular difference (for phi).
            
        Returns
        -------
        Tensor
            Normalized target offsets for all bins, shape (B, num_bins).
        """
        # true_values: (B,) -> (B, 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        diff = true_values.unsqueeze(1) - bin_centers.unsqueeze(0)  # (B, num_bins)
        
        if circular:
            # Wrap to [-π, π]
            diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        # Normalize by bin width
        normalized_offsets = diff / bin_widths.unsqueeze(0)
        
        return normalized_offsets
    
    def _compute_softmax_weights(
        self,
        logits: Tensor,
        center_bins: Tensor,
        window_half_width: int,
        num_bins: int,
        circular: bool = False,
    ) -> Tensor:
        """Compute softmax weights with window masking.
        
        Parameters
        ----------
        logits : Tensor
            Classification logits, shape (B, num_bins).
        center_bins : Tensor
            Center bins for window (clamped predicted), shape (B,).
        window_half_width : int
            Half-width of window in bins.
        num_bins : int
            Total number of bins.
        circular : bool
            If True, window wraps around (for phi).
            
        Returns
        -------
        Tensor
            Softmax weights masked to window, shape (B, num_bins).
        """
        B = logits.shape[0]
        device = logits.device
        
        # Create bin indices: (num_bins,)
        bin_indices = torch.arange(num_bins, device=device)
        
        # Compute distance from center for each bin: (B, num_bins)
        # center_bins: (B,) -> (B, 1)
        if circular:
            distance = self._circular_bin_distance(
                bin_indices.unsqueeze(0).expand(B, -1),
                center_bins.unsqueeze(1).expand(-1, num_bins),
                num_bins
            )
        else:
            distance = bin_indices.unsqueeze(0) - center_bins.unsqueeze(1)
        
        # Create window mask: (B, num_bins)
        in_window = distance.abs() <= window_half_width
        
        # Masked softmax: set out-of-window logits to -inf
        masked_logits = logits.clone()
        masked_logits[~in_window] = float('-inf')
        
        # Softmax over masked logits
        weights = F.softmax(masked_logits, dim=-1)
        
        # Zero out any NaN weights (from all -inf rows, shouldn't happen)
        weights = torch.nan_to_num(weights, nan=0.0)
        
        return weights
    
    def _compute_regression_loss(
        self,
        pred_offsets: Tensor,
        target_offsets: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Compute weighted Smooth L1 loss.
        
        Parameters
        ----------
        pred_offsets : Tensor
            Predicted offsets, shape (B, num_bins).
        target_offsets : Tensor
            Target offsets, shape (B, num_bins).
        weights : Tensor
            Softmax weights, shape (B, num_bins).
            
        Returns
        -------
        Tensor
            Scalar loss.
        """
        # Smooth L1 loss for each bin: (B, num_bins)
        loss_per_bin = self.smooth_l1(pred_offsets, target_offsets)
        
        # Weighted sum: (B, num_bins) * (B, num_bins) -> (B,)
        weighted_loss = (weights * loss_per_bin).sum(dim=-1)
        
        # Mean over batch
        return weighted_loss.mean()
    
    def forward(self, inputs: dict) -> dict:
        """Forward pass through the model."""
        return self.model(
            inputs,
            run_classification=True,
            run_regression=True,
        )
    
    def _compute_classification_targets(self, targets: dict) -> dict:
        """Compute bin indices for targets."""
        eta = targets['eta']
        phi = targets['phi']
        pt = targets['pt'].clamp(self.pt_min, self.pt_max)
        
        eta_bins = self._value_to_bin_index(eta, self.eta_bin_edges)
        phi_bins = self._value_to_bin_index(phi, self.phi_bin_edges)
        pt_bins = self._value_to_bin_index(pt, self.pt_bin_edges)
        
        return {
            'eta_bin': eta_bins,
            'phi_bin': phi_bins,
            'pt_bin': pt_bins,
            'eta': eta,
            'phi': phi,
            'pt': pt,
        }
    
    def _compute_loss(self, outputs: dict, cls_targets: dict) -> dict:
        """Compute multi-bin weighted regression loss.
        
        Parameters
        ----------
        outputs : dict
            Model outputs with logits and offsets.
        cls_targets : dict
            Classification targets with bin indices and raw values.
            
        Returns
        -------
        dict
            Dictionary with individual and total losses.
        """
        # Get predicted bins from classification logits
        eta_pred_bins = outputs['eta_logits'].argmax(dim=-1)
        phi_pred_bins = outputs['phi_logits'].argmax(dim=-1)
        pt_pred_bins = outputs['pt_logits'].argmax(dim=-1)
        
        # Clamp predicted bins to be within max distance of truth
        eta_clamped = self._clamp_predicted_bin(
            eta_pred_bins, cls_targets['eta_bin'], 
            self.eta_clamp_bins, circular=False, num_bins=self.eta_bins
        )
        phi_clamped = self._clamp_predicted_bin(
            phi_pred_bins, cls_targets['phi_bin'],
            self.phi_clamp_bins, circular=True, num_bins=self.phi_bins
        )
        pt_clamped = self._clamp_predicted_bin(
            pt_pred_bins, cls_targets['pt_bin'],
            self.pt_clamp_bins, circular=False, num_bins=self.pt_bins
        )
        
        # Compute target offsets for all bins (normalized by bin width)
        eta_target_offsets = self._compute_target_offsets(
            cls_targets['eta'], self.eta_bin_centers, self.eta_bin_widths, circular=False
        )
        phi_target_offsets = self._compute_target_offsets(
            cls_targets['phi'], self.phi_bin_centers, self.phi_bin_widths, circular=True
        )
        pt_target_offsets = self._compute_target_offsets(
            cls_targets['pt'], self.pt_bin_centers, self.pt_bin_widths, circular=False
        )
        
        # Compute softmax weights with window around clamped predicted bin
        eta_weights = self._compute_softmax_weights(
            outputs['eta_logits'], eta_clamped, self.eta_window_bins,
            self.eta_bins, circular=False
        )
        phi_weights = self._compute_softmax_weights(
            outputs['phi_logits'], phi_clamped, self.phi_window_bins,
            self.phi_bins, circular=True
        )
        pt_weights = self._compute_softmax_weights(
            outputs['pt_logits'], pt_clamped, self.pt_window_bins,
            self.pt_bins, circular=False
        )
        
        # Compute weighted regression losses
        eta_loss = self._compute_regression_loss(
            outputs['eta_offsets'], eta_target_offsets, eta_weights
        )
        phi_loss = self._compute_regression_loss(
            outputs['phi_offsets'], phi_target_offsets, phi_weights
        )
        pt_loss = self._compute_regression_loss(
            outputs['pt_offsets'], pt_target_offsets, pt_weights
        )
        
        # Total loss
        total_loss = (
            self.eta_loss_weight * eta_loss +
            self.phi_loss_weight * phi_loss +
            self.pt_loss_weight * pt_loss
        )
        
        return {
            'loss': total_loss,
            'eta_loss': eta_loss,
            'phi_loss': phi_loss,
            'pt_loss': pt_loss,
        }
    
    def _combine_cls_and_regression(self, outputs: dict) -> dict:
        """Combine classification and regression predictions.
        
        Final prediction = bin_center[argmax(cls)] + offset[argmax(cls)] * bin_width[argmax(cls)]
        
        Returns continuous predictions for eta, phi, pt.
        """
        # Get predicted bins
        eta_pred_bins = outputs['eta_logits'].argmax(dim=-1)
        phi_pred_bins = outputs['phi_logits'].argmax(dim=-1)
        pt_pred_bins = outputs['pt_logits'].argmax(dim=-1)
        
        # Get bin centers for predicted bins
        eta_centers = self.eta_bin_centers[eta_pred_bins]
        phi_centers = self.phi_bin_centers[phi_pred_bins]
        pt_centers = self.pt_bin_centers[pt_pred_bins]
        
        # Get bin widths for predicted bins
        eta_widths = self.eta_bin_widths[eta_pred_bins]
        phi_widths = self.phi_bin_widths[phi_pred_bins]
        pt_widths = self.pt_bin_widths[pt_pred_bins]
        
        # Get offsets for predicted bins
        B = eta_pred_bins.shape[0]
        batch_idx = torch.arange(B, device=eta_pred_bins.device)
        
        eta_offsets = outputs['eta_offsets'][batch_idx, eta_pred_bins]
        phi_offsets = outputs['phi_offsets'][batch_idx, phi_pred_bins]
        pt_offsets = outputs['pt_offsets'][batch_idx, pt_pred_bins]
        
        # Combined prediction: center + offset * width
        eta_pred = eta_centers + eta_offsets * eta_widths
        phi_pred = phi_centers + phi_offsets * phi_widths
        pt_pred = pt_centers + pt_offsets * pt_widths
        
        # Wrap phi to [-π, π]
        phi_pred = torch.atan2(torch.sin(phi_pred), torch.cos(phi_pred))
        
        # Clamp pt to valid range
        pt_pred = pt_pred.clamp(self.pt_min, self.pt_max)
        
        return {
            'eta': eta_pred,
            'phi': phi_pred,
            'pt': pt_pred,
        }
    
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Compute classification targets
        cls_targets = self._compute_classification_targets(targets)
        
        # Compute loss
        losses = self._compute_loss(outputs, cls_targets)
        
        # Log losses
        self.log('train/loss', losses['loss'], prog_bar=True)
        self.log('train/eta_loss', losses['eta_loss'])
        self.log('train/phi_loss', losses['phi_loss'])
        self.log('train/pt_loss', losses['pt_loss'])
        
        return losses['loss']
    
    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        inputs, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Compute classification targets
        cls_targets = self._compute_classification_targets(targets)
        
        # Compute loss
        losses = self._compute_loss(outputs, cls_targets)
        
        # Get combined predictions for physics metrics
        combined_preds = self._combine_cls_and_regression(outputs)
        
        # Accumulate for epoch-level metrics
        self._val_eta_preds.append(combined_preds['eta'].detach().cpu())
        self._val_eta_targets.append(targets['eta'].cpu())
        self._val_phi_preds.append(combined_preds['phi'].detach().cpu())
        self._val_phi_targets.append(targets['phi'].cpu())
        self._val_pt_preds.append(combined_preds['pt'].detach().cpu())
        self._val_pt_targets.append(targets['pt'].cpu())
        
        # Log losses
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val/eta_loss', losses['eta_loss'], sync_dist=True)
        self.log('val/phi_loss', losses['phi_loss'], sync_dist=True)
        self.log('val/pt_loss', losses['pt_loss'], sync_dist=True)
        
        return losses
    
    def on_validation_epoch_end(self):
        """Compute epoch-level physics metrics."""
        # Concatenate accumulated predictions
        eta_preds = torch.cat(self._val_eta_preds)
        eta_targets = torch.cat(self._val_eta_targets)
        phi_preds = torch.cat(self._val_phi_preds)
        phi_targets = torch.cat(self._val_phi_targets)
        pt_preds = torch.cat(self._val_pt_preds)
        pt_targets = torch.cat(self._val_pt_targets)
        
        # Eta metrics
        eta_residuals = eta_preds - eta_targets
        eta_mae = eta_residuals.abs().mean()
        eta_std = eta_residuals.std()
        
        # Phi metrics (with periodicity)
        phi_residuals = torch.atan2(
            torch.sin(phi_preds - phi_targets),
            torch.cos(phi_preds - phi_targets)
        )
        phi_mae = phi_residuals.abs().mean()
        phi_std = phi_residuals.std()
        
        # Pt metrics
        pt_residuals = pt_preds - pt_targets
        pt_mae = pt_residuals.abs().mean()
        pt_std = pt_residuals.std()
        
        # Relative pt metrics
        pt_relative_residuals = (pt_preds - pt_targets) / pt_targets.clamp(min=1.0)
        pt_relative_mae = pt_relative_residuals.abs().mean()
        pt_relative_std = pt_relative_residuals.std()
        
        # Log physics metrics
        self.log('val/eta_mae_mrad', eta_mae * 1000, sync_dist=True)
        self.log('val/eta_std_mrad', eta_std * 1000, sync_dist=True)
        self.log('val/phi_mae_mrad', phi_mae * 1000, sync_dist=True)
        self.log('val/phi_std_mrad', phi_std * 1000, sync_dist=True)
        self.log('val/pt_mae_GeV', pt_mae, sync_dist=True)
        self.log('val/pt_std_GeV', pt_std, sync_dist=True)
        self.log('val/pt_relative_mae_pct', pt_relative_mae * 100, sync_dist=True)
        self.log('val/pt_relative_std_pct', pt_relative_std * 100, sync_dist=True)
        
        # Reset accumulators
        self._reset_val_accumulators()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Only optimize regression heads (others are frozen)
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,
                lr=self.lrs_config['initial'],
                weight_decay=self.lrs_config.get('weight_decay', 1e-5),
            )
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                params,
                lr=self.lrs_config['initial'],
                weight_decay=self.lrs_config.get('weight_decay', 0),
            )
        elif self.optimizer_name == 'Lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    params,
                    lr=self.lrs_config['initial'],
                    weight_decay=self.lrs_config.get('weight_decay', 1e-5),
                )
            except ImportError:
                print("Lion not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    params,
                    lr=self.lrs_config['initial'],
                    weight_decay=self.lrs_config.get('weight_decay', 1e-5),
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # OneCycleLR scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lrs_config['max'],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.lrs_config['pct_start'],
            final_div_factor=self.lrs_config['max'] / self.lrs_config['end'],
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
