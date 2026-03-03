#!/usr/bin/env python
"""YOLO Stage 2: Simplified Regression Training with Truth-Anchored Offsets.

This module implements Stage 2 of the YOLO 3-stage training pipeline:
1. Loads a trained Stage 1 checkpoint (encoder + classification heads)
2. Freezes encoder and classification heads
3. Trains ONLY the regression heads to predict sub-bin offsets

Key design: TRUTH-ANCHORED OFFSETS
==================================
- Regression heads learn to predict: (truth_value - truth_bin_center) / bin_width
- This is ALWAYS in [-0.5, +0.5] regardless of classifier accuracy
- At inference: reconstructed = predicted_bin_center + offset * bin_width

Why this approach?
- The regression head learns ONE well-defined quantity: sub-bin position of truth
- No impossible task of "fixing" large classification errors
- Better training signal (bounded, consistent targets)
- Stage 3 (end-to-end) can later coordinate classifier + regression

Loss function:
- Simple Smooth L1 loss on the offset predictions
- No multi-bin complexity needed
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lightning import LightningModule
from torchmetrics import Accuracy, MeanAbsoluteError, AUROC

from hepattn.models.yolo_regressor import YOLORegressor


class YOLORegressionSimple(LightningModule):
    """Stage 2: Train regression heads with frozen encoder using truth-anchored offsets.
    
    This simplified approach:
    - Trains regression heads to predict (truth - truth_bin_center) / bin_width
    - Uses Smooth L1 loss directly on offsets
    - No multi-bin loss complexity
    - Validation metrics show both classification-only and cls+regression accuracy
    
    Parameters
    ----------
    model : YOLORegressor
        Model with regression heads enabled.
    stage1_checkpoint : str
        Path to Stage 1 checkpoint to load encoder/classification weights.
    bin_dir : str
        Directory containing bin edge files.
    eta_bins, phi_bins, pt_bins : int
        Number of bins for each parameter.
    pt_log_bins : bool
        Whether to use log-spaced pt bins.
    smooth_l1_beta : float
        Beta parameter for Smooth L1 loss.
    eta_loss_weight, phi_loss_weight, pt_loss_weight : float
        Weights for each parameter's loss.
    """
    
    def __init__(
        self,
        model: dict | YOLORegressor,
        name: str = "YOLO-Stage2-Simple",
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
        # Loss configuration
        smooth_l1_beta: float = 0.1,
        eta_loss_weight: float = 1.0,
        phi_loss_weight: float = 1.0,
        pt_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        # Focal loss for classification logging
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.name = name
        self.optimizer_name = optimizer
        self.lrs_config = lrs_config or {
            'initial': 1e-5,
            'max': 1e-4,
            'end': 1e-6,
            'pct_start': 0.05,
            'weight_decay': 1e-5,
        }
        
        # Checkpoint path
        self.stage1_checkpoint = stage1_checkpoint
        
        # Bin configuration
        self.bin_dir = Path(bin_dir)
        self.n_eta_bins = eta_bins
        self.n_phi_bins = phi_bins
        self.n_pt_bins = pt_bins
        self.pt_log_bins = pt_log_bins
        
        # Clipping ranges
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_gap_min = eta_gap_min
        self.eta_gap_max = eta_gap_max
        
        # Loss configuration
        self.smooth_l1_beta = smooth_l1_beta
        self.eta_loss_weight = eta_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.pt_loss_weight = pt_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.focal_gamma = focal_gamma
        
        # Build or use provided model
        if isinstance(model, dict):
            model_class = model.get('class_path', 'hepattn.models.yolo_regressor.YOLORegressor')
            model_args = model.get('init_args', {})
            if 'YOLORegressor' in model_class:
                self.model = YOLORegressor(**model_args)
            else:
                raise ValueError(f"Unknown model class: {model_class}")
        else:
            self.model = model
        
        # Load bin edges
        self._load_bins()
        
        # Metrics
        self._setup_metrics()
        
        # Validation accumulators for physics metrics
        self._reset_val_accumulators()
    
    def _load_bins(self):
        """Load bin edges and compute bin centers/widths."""
        # Load eta bins
        eta_bins_file = self.bin_dir / f"eta_bins_{self.n_eta_bins}.npy"
        eta_centers_file = self.bin_dir / f"eta_bin_centers_{self.n_eta_bins}.npy"
        
        if eta_bins_file.exists():
            self.register_buffer('eta_bin_edges', torch.from_numpy(np.load(eta_bins_file)).float())
            self.register_buffer('eta_bin_centers', torch.from_numpy(np.load(eta_centers_file)).float())
        else:
            # Create uniform bins if file doesn't exist
            edges = torch.linspace(self.eta_min, self.eta_max, self.n_eta_bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            self.register_buffer('eta_bin_edges', edges)
            self.register_buffer('eta_bin_centers', centers)
        
        # Compute eta bin widths
        eta_widths = self.eta_bin_edges[1:] - self.eta_bin_edges[:-1]
        self.register_buffer('eta_bin_widths', eta_widths)
        
        # Load phi bins
        phi_bins_file = self.bin_dir / f"phi_bins_{self.n_phi_bins}.npy"
        phi_centers_file = self.bin_dir / f"phi_bin_centers_{self.n_phi_bins}.npy"
        
        if phi_bins_file.exists():
            self.register_buffer('phi_bin_edges', torch.from_numpy(np.load(phi_bins_file)).float())
            self.register_buffer('phi_bin_centers', torch.from_numpy(np.load(phi_centers_file)).float())
        else:
            edges = torch.linspace(-math.pi, math.pi, self.n_phi_bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2
            self.register_buffer('phi_bin_edges', edges)
            self.register_buffer('phi_bin_centers', centers)
        
        # Compute phi bin widths
        phi_widths = self.phi_bin_edges[1:] - self.phi_bin_edges[:-1]
        self.register_buffer('phi_bin_widths', phi_widths)
        
        # Load pt bins (log or linear)
        suffix = "_log" if self.pt_log_bins else ""
        pt_bins_file = self.bin_dir / f"pt_bins_{self.n_pt_bins}{suffix}.npy"
        pt_centers_file = self.bin_dir / f"pt_bin_centers_{self.n_pt_bins}{suffix}.npy"
        
        if pt_bins_file.exists():
            self.register_buffer('pt_bin_edges', torch.from_numpy(np.load(pt_bins_file)).float())
            self.register_buffer('pt_bin_centers', torch.from_numpy(np.load(pt_centers_file)).float())
        else:
            if self.pt_log_bins:
                edges = torch.logspace(
                    math.log10(self.pt_min), math.log10(self.pt_max), self.n_pt_bins + 1
                )
                centers = torch.sqrt(edges[:-1] * edges[1:])  # Geometric mean
            else:
                edges = torch.linspace(self.pt_min, self.pt_max, self.n_pt_bins + 1)
                centers = (edges[:-1] + edges[1:]) / 2
            self.register_buffer('pt_bin_edges', edges)
            self.register_buffer('pt_bin_centers', centers)
        
        # Compute pt bin widths
        pt_widths = self.pt_bin_edges[1:] - self.pt_bin_edges[:-1]
        self.register_buffer('pt_bin_widths', pt_widths)
    
    def _setup_metrics(self):
        """Setup torchmetrics for tracking."""
        # Classification metrics (for logging classifier performance)
        self.train_eta_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins)
        self.train_phi_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins)
        self.train_pt_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins)
        self.train_charge_auroc = AUROC(task="binary")
        
        self.val_eta_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins)
        self.val_phi_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins)
        self.val_pt_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins)
        self.val_charge_auroc = AUROC(task="binary")
        
        # Top-k accuracy metrics
        self.val_eta_top5_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins, top_k=5)
        self.val_phi_top5_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins, top_k=5)
        self.val_pt_top5_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins, top_k=5)
        
        self.val_eta_top10_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins, top_k=10)
        self.val_phi_top10_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins, top_k=10)
        self.val_pt_top10_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins, top_k=10)
        
        # Regression MAE metrics (for offset learning)
        self.train_eta_offset_mae = MeanAbsoluteError()
        self.train_phi_offset_mae = MeanAbsoluteError()
        self.train_pt_offset_mae = MeanAbsoluteError()
        
        self.val_eta_offset_mae = MeanAbsoluteError()
        self.val_phi_offset_mae = MeanAbsoluteError()
        self.val_pt_offset_mae = MeanAbsoluteError()
    
    def _reset_val_accumulators(self):
        """Reset validation accumulators for physics metrics."""
        self.val_eta_errors = []
        self.val_phi_errors = []
        self.val_pt_errors = []
        self.val_pt_true = []
    
    def setup(self, stage: str):
        """Load Stage 1 checkpoint and freeze appropriate components."""
        if stage == "fit" and self.stage1_checkpoint:
            self._load_stage1_checkpoint()
    
    def _load_stage1_checkpoint(self):
        """Load encoder and classification weights from Stage 1, freeze them."""
        if not Path(self.stage1_checkpoint).exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {self.stage1_checkpoint}")
        
        print(f"Loading Stage 1 checkpoint: {self.stage1_checkpoint}")
        checkpoint = torch.load(self.stage1_checkpoint, map_location='cpu', weights_only=False)
        
        # Get state dict
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Filter to model weights only (remove 'model.' prefix if present)
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state[k[6:]] = v  # Remove 'model.' prefix
            else:
                model_state[k] = v
        
        # Load weights (strict=False to allow missing regression head weights)
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        
        print(f"Loaded Stage 1 weights:")
        print(f"  Missing keys (expected - regression heads): {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        # Freeze encoder and classification heads
        self.model.freeze_encoder()
        self.model.freeze_classification_heads()
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters (regression heads): {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def _get_true_bins(self, eta: Tensor, phi: Tensor, pt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Convert true values to bin indices."""
        # Eta bins
        eta_bins = torch.searchsorted(self.eta_bin_edges[1:-1], eta)
        eta_bins = eta_bins.clamp(0, self.n_eta_bins - 1)
        
        # Phi bins
        phi_bins = torch.searchsorted(self.phi_bin_edges[1:-1], phi)
        phi_bins = phi_bins.clamp(0, self.n_phi_bins - 1)
        
        # Pt bins (clamp values first)
        pt_clamped = pt.clamp(self.pt_min, self.pt_max)
        pt_bins = torch.searchsorted(self.pt_bin_edges[1:-1], pt_clamped)
        pt_bins = pt_bins.clamp(0, self.n_pt_bins - 1)
        
        return eta_bins, phi_bins, pt_bins
    
    def _compute_target_offsets(
        self,
        true_eta: Tensor,
        true_phi: Tensor,
        true_pt: Tensor,
        true_eta_bins: Tensor,
        true_phi_bins: Tensor,
        true_pt_bins: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute truth-anchored offsets (always in [-0.5, +0.5]).
        
        For each parameter:
            offset = (true_value - true_bin_center) / bin_width
        
        For phi, we use proper circular handling.
        """
        batch_size = true_eta.shape[0]
        device = true_eta.device
        
        # Eta offset
        eta_centers = self.eta_bin_centers[true_eta_bins]
        eta_widths = self.eta_bin_widths[true_eta_bins]
        eta_offset = (true_eta - eta_centers) / eta_widths
        
        # Phi offset (with circular wrapping)
        phi_centers = self.phi_bin_centers[true_phi_bins]
        phi_widths = self.phi_bin_widths[true_phi_bins]
        # Use atan2 for proper circular difference
        phi_diff = torch.atan2(
            torch.sin(true_phi - phi_centers),
            torch.cos(true_phi - phi_centers)
        )
        phi_offset = phi_diff / phi_widths
        
        # Pt offset
        pt_centers = self.pt_bin_centers[true_pt_bins]
        pt_widths = self.pt_bin_widths[true_pt_bins]
        pt_offset = (true_pt - pt_centers) / pt_widths
        
        return eta_offset, phi_offset, pt_offset
    
    def _compute_regression_loss(
        self,
        pred_eta_offsets: Tensor,  # (B, n_eta_bins)
        pred_phi_offsets: Tensor,  # (B, n_phi_bins)
        pred_pt_offsets: Tensor,   # (B, n_pt_bins)
        true_eta_bins: Tensor,
        true_phi_bins: Tensor,
        true_pt_bins: Tensor,
        target_eta_offset: Tensor,
        target_phi_offset: Tensor,
        target_pt_offset: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute Smooth L1 loss on the offsets at the TRUE bin.
        
        We only train the offset at the true bin, since that's where the
        ground truth offset is well-defined.
        """
        batch_size = pred_eta_offsets.shape[0]
        device = pred_eta_offsets.device
        
        # Gather predicted offsets at true bins
        # pred_*_offsets: (B, n_bins) -> extract offset at true bin -> (B,)
        pred_eta_at_true = pred_eta_offsets.gather(1, true_eta_bins.unsqueeze(1)).squeeze(1)
        pred_phi_at_true = pred_phi_offsets.gather(1, true_phi_bins.unsqueeze(1)).squeeze(1)
        pred_pt_at_true = pred_pt_offsets.gather(1, true_pt_bins.unsqueeze(1)).squeeze(1)
        
        # Smooth L1 loss
        eta_loss = F.smooth_l1_loss(pred_eta_at_true, target_eta_offset, beta=self.smooth_l1_beta)
        phi_loss = F.smooth_l1_loss(pred_phi_at_true, target_phi_offset, beta=self.smooth_l1_beta)
        pt_loss = F.smooth_l1_loss(pred_pt_at_true, target_pt_offset, beta=self.smooth_l1_beta)
        
        # Weighted total
        total_loss = (
            self.eta_loss_weight * eta_loss +
            self.phi_loss_weight * phi_loss +
            self.pt_loss_weight * pt_loss
        )
        
        return total_loss, eta_loss, phi_loss, pt_loss
    
    def _combine_cls_and_regression(
        self,
        cls_eta_logits: Tensor,
        cls_phi_logits: Tensor,
        cls_pt_logits: Tensor,
        reg_eta_offsets: Tensor,
        reg_phi_offsets: Tensor,
        reg_pt_offsets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Combine classification and regression for final predictions.
        
        Uses argmax of classification logits to select bin, then adds
        the corresponding offset to get the final value.
        """
        # Get predicted bins from classifier
        pred_eta_bins = cls_eta_logits.argmax(dim=1)
        pred_phi_bins = cls_phi_logits.argmax(dim=1)
        pred_pt_bins = cls_pt_logits.argmax(dim=1)
        
        # Get bin centers and widths
        eta_centers = self.eta_bin_centers[pred_eta_bins]
        phi_centers = self.phi_bin_centers[pred_phi_bins]
        pt_centers = self.pt_bin_centers[pred_pt_bins]
        
        eta_widths = self.eta_bin_widths[pred_eta_bins]
        phi_widths = self.phi_bin_widths[pred_phi_bins]
        pt_widths = self.pt_bin_widths[pred_pt_bins]
        
        # Get offsets at predicted bins
        pred_eta_offset = reg_eta_offsets.gather(1, pred_eta_bins.unsqueeze(1)).squeeze(1)
        pred_phi_offset = reg_phi_offsets.gather(1, pred_phi_bins.unsqueeze(1)).squeeze(1)
        pred_pt_offset = reg_pt_offsets.gather(1, pred_pt_bins.unsqueeze(1)).squeeze(1)
        
        # Reconstruct values
        pred_eta = eta_centers + pred_eta_offset * eta_widths
        pred_pt = pt_centers + pred_pt_offset * pt_widths
        
        # Phi needs circular handling
        pred_phi_raw = phi_centers + pred_phi_offset * phi_widths
        pred_phi = torch.atan2(torch.sin(pred_phi_raw), torch.cos(pred_phi_raw))
        
        return pred_eta, pred_phi, pred_pt
    
    def forward(self, inputs: dict) -> dict:
        """Forward pass through the model."""
        return self.model(
            inputs,
            run_classification=True,
            run_regression=True,
        )
    
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Training step - compute regression loss only."""
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Get true values
        true_eta = targets['eta']
        true_phi = targets['phi']
        true_pt = targets['pt']
        true_charge = targets['charge']
        
        # Get true bins
        true_eta_bins, true_phi_bins, true_pt_bins = self._get_true_bins(
            true_eta, true_phi, true_pt
        )
        
        # Compute target offsets (truth-anchored)
        target_eta_offset, target_phi_offset, target_pt_offset = self._compute_target_offsets(
            true_eta, true_phi, true_pt,
            true_eta_bins, true_phi_bins, true_pt_bins,
        )
        
        # Get regression outputs
        eta_offsets = outputs['eta_offsets']  # (B, n_eta_bins)
        phi_offsets = outputs['phi_offsets']  # (B, n_phi_bins)
        pt_offsets = outputs['pt_offsets']    # (B, n_pt_bins)
        
        # Compute regression loss
        reg_loss, eta_loss, phi_loss, pt_loss = self._compute_regression_loss(
            eta_offsets, phi_offsets, pt_offsets,
            true_eta_bins, true_phi_bins, true_pt_bins,
            target_eta_offset, target_phi_offset, target_pt_offset,
        )
        
        # Logging
        self.log('train/loss', reg_loss, prog_bar=True)
        self.log('train/eta_loss', eta_loss)
        self.log('train/phi_loss', phi_loss)
        self.log('train/pt_loss', pt_loss)
        
        # Log offset prediction quality (gather at true bins)
        pred_eta_at_true = eta_offsets.gather(1, true_eta_bins.unsqueeze(1)).squeeze(1)
        pred_phi_at_true = phi_offsets.gather(1, true_phi_bins.unsqueeze(1)).squeeze(1)
        pred_pt_at_true = pt_offsets.gather(1, true_pt_bins.unsqueeze(1)).squeeze(1)
        
        self.train_eta_offset_mae(pred_eta_at_true, target_eta_offset)
        self.train_phi_offset_mae(pred_phi_at_true, target_phi_offset)
        self.train_pt_offset_mae(pred_pt_at_true, target_pt_offset)
        
        self.log('train/eta_offset_mae', self.train_eta_offset_mae, on_step=False, on_epoch=True)
        self.log('train/phi_offset_mae', self.train_phi_offset_mae, on_step=False, on_epoch=True)
        self.log('train/pt_offset_mae', self.train_pt_offset_mae, on_step=False, on_epoch=True)
        
        # Also log classification accuracy (frozen, but useful for monitoring)
        eta_logits = outputs['eta_logits']
        phi_logits = outputs['phi_logits']
        pt_logits = outputs['pt_logits']
        
        self.train_eta_acc(eta_logits, true_eta_bins)
        self.train_phi_acc(phi_logits, true_phi_bins)
        self.train_pt_acc(pt_logits, true_pt_bins)
        
        self.log('train/cls_eta_acc', self.train_eta_acc, on_step=False, on_epoch=True)
        self.log('train/cls_phi_acc', self.train_phi_acc, on_step=False, on_epoch=True)
        self.log('train/cls_pt_acc', self.train_pt_acc, on_step=False, on_epoch=True)
        
        return reg_loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Validation step - compute metrics for both classification and combined."""
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Get true values
        true_eta = targets['eta']
        true_phi = targets['phi']
        true_pt = targets['pt']
        true_charge = targets['charge']
        
        # Get true bins
        true_eta_bins, true_phi_bins, true_pt_bins = self._get_true_bins(
            true_eta, true_phi, true_pt
        )
        
        # Compute target offsets
        target_eta_offset, target_phi_offset, target_pt_offset = self._compute_target_offsets(
            true_eta, true_phi, true_pt,
            true_eta_bins, true_phi_bins, true_pt_bins,
        )
        
        # Get outputs
        eta_logits = outputs['eta_logits']
        phi_logits = outputs['phi_logits']
        pt_logits = outputs['pt_logits']
        eta_offsets = outputs['eta_offsets']
        phi_offsets = outputs['phi_offsets']
        pt_offsets = outputs['pt_offsets']
        
        # Compute regression loss
        reg_loss, eta_loss, phi_loss, pt_loss = self._compute_regression_loss(
            eta_offsets, phi_offsets, pt_offsets,
            true_eta_bins, true_phi_bins, true_pt_bins,
            target_eta_offset, target_phi_offset, target_pt_offset,
        )
        
        # Logging
        self.log('val/loss', reg_loss, prog_bar=True, sync_dist=True)
        self.log('val/eta_loss', eta_loss, sync_dist=True)
        self.log('val/phi_loss', phi_loss, sync_dist=True)
        self.log('val/pt_loss', pt_loss, sync_dist=True)
        
        # Classification accuracy (frozen classifier)
        self.val_eta_acc(eta_logits, true_eta_bins)
        self.val_phi_acc(phi_logits, true_phi_bins)
        self.val_pt_acc(pt_logits, true_pt_bins)
        
        self.log('val/cls_eta_acc', self.val_eta_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_phi_acc', self.val_phi_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_pt_acc', self.val_pt_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        # Top-k accuracy
        self.val_eta_top5_acc(eta_logits, true_eta_bins)
        self.val_phi_top5_acc(phi_logits, true_phi_bins)
        self.val_pt_top5_acc(pt_logits, true_pt_bins)
        self.val_eta_top10_acc(eta_logits, true_eta_bins)
        self.val_phi_top10_acc(phi_logits, true_phi_bins)
        self.val_pt_top10_acc(pt_logits, true_pt_bins)
        
        self.log('val/cls_eta_top5_acc', self.val_eta_top5_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_phi_top5_acc', self.val_phi_top5_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_pt_top5_acc', self.val_pt_top5_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_eta_top10_acc', self.val_eta_top10_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_phi_top10_acc', self.val_phi_top10_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_pt_top10_acc', self.val_pt_top10_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        # Offset MAE
        pred_eta_at_true = eta_offsets.gather(1, true_eta_bins.unsqueeze(1)).squeeze(1)
        pred_phi_at_true = phi_offsets.gather(1, true_phi_bins.unsqueeze(1)).squeeze(1)
        pred_pt_at_true = pt_offsets.gather(1, true_pt_bins.unsqueeze(1)).squeeze(1)
        
        self.val_eta_offset_mae(pred_eta_at_true, target_eta_offset)
        self.val_phi_offset_mae(pred_phi_at_true, target_phi_offset)
        self.val_pt_offset_mae(pred_pt_at_true, target_pt_offset)
        
        self.log('val/eta_offset_mae', self.val_eta_offset_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/phi_offset_mae', self.val_phi_offset_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/pt_offset_mae', self.val_pt_offset_mae, on_step=False, on_epoch=True, sync_dist=True)
        
        # Compute combined predictions (cls + regression)
        pred_eta, pred_phi, pred_pt = self._combine_cls_and_regression(
            eta_logits, phi_logits, pt_logits,
            eta_offsets, phi_offsets, pt_offsets,
        )
        
        # Accumulate physics errors
        eta_error = pred_eta - true_eta
        phi_error = torch.atan2(torch.sin(pred_phi - true_phi), torch.cos(pred_phi - true_phi))
        pt_error = pred_pt - true_pt
        
        self.val_eta_errors.append(eta_error.detach().cpu())
        self.val_phi_errors.append(phi_error.detach().cpu())
        self.val_pt_errors.append(pt_error.detach().cpu())
        self.val_pt_true.append(true_pt.detach().cpu())
        
        return reg_loss
    
    def on_validation_epoch_end(self):
        """Compute physics metrics at end of validation epoch."""
        if not self.val_eta_errors:
            return
        
        # Concatenate all errors
        eta_errors = torch.cat(self.val_eta_errors)
        phi_errors = torch.cat(self.val_phi_errors)
        pt_errors = torch.cat(self.val_pt_errors)
        pt_true = torch.cat(self.val_pt_true)
        
        # Compute MAE and std
        eta_mae = eta_errors.abs().mean()
        eta_std = eta_errors.std()
        phi_mae = phi_errors.abs().mean()
        phi_std = phi_errors.std()
        pt_mae = pt_errors.abs().mean()
        pt_std = pt_errors.std()
        
        # Relative pt metrics
        pt_relative_error = pt_errors / pt_true.clamp(min=1.0)
        pt_relative_mae = pt_relative_error.abs().mean()
        pt_relative_std = pt_relative_error.std()
        
        # Log physics metrics
        self.log('val/combined_eta_mae', eta_mae, sync_dist=True)
        self.log('val/combined_eta_std', eta_std, sync_dist=True)
        self.log('val/combined_phi_mae_rad', phi_mae, sync_dist=True)
        self.log('val/combined_phi_std_rad', phi_std, sync_dist=True)
        self.log('val/combined_phi_mae_mrad', phi_mae * 1000, sync_dist=True)
        self.log('val/combined_phi_std_mrad', phi_std * 1000, sync_dist=True)
        self.log('val/combined_pt_mae_GeV', pt_mae, sync_dist=True)
        self.log('val/combined_pt_std_GeV', pt_std, sync_dist=True)
        self.log('val/combined_pt_relative_mae_pct', pt_relative_mae * 100, sync_dist=True)
        self.log('val/combined_pt_relative_std_pct', pt_relative_std * 100, sync_dist=True)
        
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
