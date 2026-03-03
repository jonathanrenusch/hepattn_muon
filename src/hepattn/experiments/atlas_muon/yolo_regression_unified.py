"""Unified Stage 2 regression: single scalar offsets from bin centers.

Stage 2 of the 3-stage YOLO training pipeline (unified version):
  - Loads frozen encoder + classification heads from Stage 1 checkpoint
  - Adds lightweight regression heads (external to the model)
  - Each head outputs ONE scalar offset per variable (not per-bin)
  - Targets: truth_value - bin_center[true_bin]
  - Eta: linear offset (absolute units)
  - Phi: wrapped offset via atan2 (absolute radians)
  - Pt: log-space offset: log(truth_pt) - log(bin_center_pt)
  - Loss: Smooth L1 on all three offsets
  - Encoder and classification heads are FROZEN (Stage 3 unfreezes them)

Key design differences from yolo_regression_simple.py:
  - yolo_regression_simple: per-bin offset heads (N outputs per var), gather at true bin
  - yolo_regression_unified: single scalar offset heads (1 output per var), always trains full head

Inference reconstruction:
  eta_pred = eta_bin_center[argmax(eta_logits)] + eta_offset
  phi_pred = wrap(phi_bin_center[argmax(phi_logits)] + phi_offset)
  pt_pred  = exp(log(pt_bin_center[argmax(pt_logits)]) + pt_offset)
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from lightning import LightningModule
from torchmetrics import Accuracy, MeanAbsoluteError, AUROC

from hepattn.models.yolo_regressor import YOLORegressor, FocalLoss


class YOLORegressionUnified(LightningModule):
    """Lightning module for unified regression training (Stage 2).
    
    Regression heads are external to YOLORegressor and added by this module.
    They take the cls_embedding from the frozen model and predict a single
    scalar offset per variable.
    
    Parameters
    ----------
    model : dict or YOLORegressor
        The YOLO model (encoder + classification heads).
    name : str
        Experiment name.
    optimizer : str
        Optimizer name ('Adam', 'AdamW', 'Lion').
    lrs_config : dict
        Learning rate scheduler config.
    stage1_checkpoint : str
        Path to Stage 1 classification checkpoint.
    bin_dir : str
        Directory with bin arrays.
    eta_bins : int or list[int]
        Number of eta bins. If list (hierarchical), finest level is used for regression.
    phi_bins : int or list[int]
        Number of phi bins.
    pt_bins : int or list[int]
        Number of pt bins.
    use_inv_pt : bool
        Whether Stage 1 used 1/pt binning. If True, bin centers are in 1/pt space
        and we convert to pt for offset computation.
    reg_hidden_dim : int
        Hidden dimension for regression heads.
    reg_num_layers : int
        Number of layers in regression heads (1, 2, or 3).
    reg_dropout : float
        Dropout in regression heads.
    smooth_l1_beta : float
        Beta parameter for Smooth L1 loss.
    eta_loss_weight : float
        Weight for eta regression loss.
    phi_loss_weight : float
        Weight for phi regression loss.
    pt_loss_weight : float
        Weight for pt regression loss.
    freeze_backbone : bool
        If True, freeze encoder + cls heads (default for Stage 2).
    """
    
    def __init__(
        self,
        model: dict | YOLORegressor,
        name: str = "YOLO-Stage2-Unified",
        optimizer: str = "AdamW",
        lrs_config: dict | None = None,
        stage1_checkpoint: str = "",
        stage2_checkpoint: str = "",
        bin_dir: str = "/scratch/ml_training_data_2694000_regression_true_hits_only/bins",
        eta_bins: int | list[int] = 501,
        phi_bins: int | list[int] = 500,
        pt_bins: int | list[int] = 50,
        use_inv_pt: bool = False,
        pt_log_bins: bool = False,
        # Regression head config
        reg_hidden_dim: int = 256,
        reg_num_layers: int = 2,
        reg_dropout: float = 0.15,
        # Loss config
        smooth_l1_beta: float = 0.1,
        eta_loss_weight: float = 1.0,
        phi_loss_weight: float = 1.0,
        pt_loss_weight: float = 1.0,
        # Clipping ranges
        pt_min: float = 5.0,
        pt_max: float = 200.0,
        eta_min: float = -2.7,
        eta_max: float = 2.7,
        eta_gap_min: float = -0.1,
        eta_gap_max: float = 0.1,
        # Freezing control
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.name = name
        self.stage1_checkpoint = stage1_checkpoint
        self.stage2_checkpoint = stage2_checkpoint
        self.bin_dir = Path(bin_dir)
        self.use_inv_pt = use_inv_pt
        self.pt_log_bins = pt_log_bins
        self.freeze_backbone = freeze_backbone
        
        # Optimizer config
        self.optimizer_name = optimizer
        self.lrs_config = lrs_config or {
            'initial': 1e-5,
            'max': 1e-4,
            'end': 1e-6,
            'pct_start': 0.05,
            'weight_decay': 1e-5,
        }
        
        # Normalize bins to lists and use finest level
        self.eta_bins_list = [eta_bins] if isinstance(eta_bins, int) else list(eta_bins)
        self.phi_bins_list = [phi_bins] if isinstance(phi_bins, int) else list(phi_bins)
        self.pt_bins_list = [pt_bins] if isinstance(pt_bins, int) else list(pt_bins)
        self.n_eta_bins = self.eta_bins_list[-1]  # finest level
        self.n_phi_bins = self.phi_bins_list[-1]
        self.n_pt_bins = self.pt_bins_list[-1]
        
        # Clipping ranges
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_gap_min = eta_gap_min
        self.eta_gap_max = eta_gap_max
        
        # Loss config
        self.smooth_l1_beta = smooth_l1_beta
        self.eta_loss_weight = eta_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.pt_loss_weight = pt_loss_weight
        
        # Build model
        if isinstance(model, dict):
            if 'class_path' in model and 'init_args' in model:
                self.model = YOLORegressor(**model['init_args'])
            elif 'class_path' in model:
                self.model = YOLORegressor(**model.get('init_args', {}))
            else:
                self.model = YOLORegressor(**model)
        else:
            self.model = model
        
        # Get model embedding dim
        dim = self.model.dim
        
        # ===== External regression heads (NOT inside YOLORegressor) =====
        # Each outputs a single scalar offset
        self.eta_reg_head = self._make_regression_head(
            dim, reg_hidden_dim, reg_num_layers, reg_dropout
        )
        self.phi_reg_head = self._make_regression_head(
            dim, reg_hidden_dim, reg_num_layers, reg_dropout
        )
        self.pt_reg_head = self._make_regression_head(
            dim, reg_hidden_dim, reg_num_layers, reg_dropout
        )
        
        # Load bin arrays (finest level only, for offset targets)
        self._load_bins()
        
        # Metrics
        self._setup_metrics()
        
        # Validation accumulators
        self._reset_val_accumulators()
    
    def _make_regression_head(
        self, dim: int, hidden_dim: int, num_layers: int, dropout: float
    ) -> nn.Module:
        """Create a regression head that outputs a single scalar.
        
        Parameters
        ----------
        dim : int
            Input dimension (model embedding dim).
        hidden_dim : int
            Hidden layer dimension.
        num_layers : int
            Number of layers (1, 2, or 3).
        dropout : float
            Dropout rate.
            
        Returns
        -------
        nn.Module
            Head mapping (B, dim) → (B, 1).
        """
        if num_layers == 1:
            head = nn.Linear(dim, 1)
        elif num_layers == 2:
            head = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:  # num_layers >= 3
            head = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        
        # Initialize
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        return head
    
    def _load_bins(self):
        """Load bin edges and centers for the finest classification level.
        
        These are used to:
        1. Compute offset targets: truth - bin_center[true_bin]
        2. Reconstruct final predictions: bin_center[pred_bin] + offset
        """
        # Eta bin mapping: actual bins → file name
        eta_actual_to_file = {
            51: 50, 101: 100, 501: 500, 1001: 1000,
            2001: 2000, 6001: 6000, 12001: 12000,
        }
        
        # Eta
        eta_file = eta_actual_to_file.get(self.n_eta_bins, self.n_eta_bins)
        eta_edges = np.load(self.bin_dir / f'eta_bins_{eta_file}.npy')
        eta_centers = np.load(self.bin_dir / f'eta_bin_centers_{eta_file}.npy')
        assert len(eta_centers) == self.n_eta_bins, (
            f"Eta bins mismatch: config={self.n_eta_bins}, file={len(eta_centers)}"
        )
        self.register_buffer('eta_bin_edges', torch.from_numpy(eta_edges).float())
        self.register_buffer('eta_bin_centers', torch.from_numpy(eta_centers).float())
        
        # Phi
        phi_edges = np.load(self.bin_dir / f'phi_bins_{self.n_phi_bins}.npy')
        phi_centers = np.load(self.bin_dir / f'phi_bin_centers_{self.n_phi_bins}.npy')
        assert len(phi_centers) == self.n_phi_bins, (
            f"Phi bins mismatch: config={self.n_phi_bins}, file={len(phi_centers)}"
        )
        self.register_buffer('phi_bin_edges', torch.from_numpy(phi_edges).float())
        self.register_buffer('phi_bin_centers', torch.from_numpy(phi_centers).float())
        
        # Pt (or 1/pt)
        if self.use_inv_pt:
            pt_edges = np.load(self.bin_dir / f'inv_pt_bins_{self.n_pt_bins}.npy')
            pt_centers = np.load(self.bin_dir / f'inv_pt_bin_centers_{self.n_pt_bins}.npy')
        else:
            suffix = '_log' if self.pt_log_bins else ''
            pt_edges = np.load(self.bin_dir / f'pt_bins_{self.n_pt_bins}{suffix}.npy')
            pt_centers = np.load(self.bin_dir / f'pt_bin_centers_{self.n_pt_bins}{suffix}.npy')
        assert len(pt_centers) == self.n_pt_bins, (
            f"Pt bins mismatch: config={self.n_pt_bins}, file={len(pt_centers)}"
        )
        self.register_buffer('pt_bin_edges', torch.from_numpy(pt_edges).float())
        self.register_buffer('pt_bin_centers', torch.from_numpy(pt_centers).float())
        
        # If using inv_pt, also store pt-space bin centers (inverted)
        if self.use_inv_pt:
            pt_space_centers = 1.0 / torch.from_numpy(pt_centers).float().clamp(min=1e-6)
            self.register_buffer('pt_space_bin_centers', pt_space_centers)
        
        print(f"Loaded FINEST-LEVEL bin arrays for regression:")
        print(f"  Eta: {self.n_eta_bins} bins, range [{eta_edges[0]:.3f}, {eta_edges[-1]:.3f}]")
        print(f"  Phi: {self.n_phi_bins} bins, range [{phi_edges[0]:.3f}, {phi_edges[-1]:.3f}]")
        pt_label = '1/pT' if self.use_inv_pt else 'pT'
        print(f"  Pt:  {self.n_pt_bins} bins ({pt_label}), range [{pt_edges[0]:.4f}, {pt_edges[-1]:.4f}]")
    
    def _setup_metrics(self):
        """Set up torchmetrics."""
        # Classification accuracy (frozen, for monitoring)
        self.train_eta_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins)
        self.train_phi_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins)
        self.train_pt_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins)
        
        self.val_eta_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins)
        self.val_phi_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins)
        self.val_pt_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins)
        
        # Regression offset MAE
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
        """Load checkpoint and configure freezing.
        
        Priority: stage2_checkpoint > stage1_checkpoint.
        Stage 2 checkpoint loads full state (model + regression heads).
        Stage 1 checkpoint loads only model weights (regression heads are new).
        """
        if stage == "fit":
            if self.stage2_checkpoint:
                self._load_stage2_checkpoint()
            elif self.stage1_checkpoint:
                self._load_stage1_checkpoint()
    
    def _load_stage1_checkpoint(self):
        """Load encoder + classification weights, freeze them."""
        ckpt_path = Path(self.stage1_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {self.stage1_checkpoint}")
        
        print(f"\n{'='*60}")
        print(f"Loading Stage 1 checkpoint: {self.stage1_checkpoint}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Filter to model weights (remove 'model.' prefix if present)
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state[k[6:]] = v
            else:
                model_state[k] = v
        
        # Load into self.model (strict=False: regression heads are new)
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        
        print(f"  Missing keys (expected - no regression in checkpoint): {len(missing)}")
        if missing:
            for k in missing[:5]:
                print(f"    {k}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        print(f"  Unexpected keys: {len(unexpected)}")
        
        # Freeze backbone
        if self.freeze_backbone:
            self.model.freeze_encoder()
            self.model.freeze_classification_heads()
        
        self._print_param_summary()
    
    def _load_stage2_checkpoint(self):
        """Load full state from Stage 2 checkpoint (model + regression heads).
        
        Used for Stage 3 joint finetuning: loads everything, then
        applies freeze_backbone setting (typically False for Stage 3).
        """
        ckpt_path = Path(self.stage2_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage 2 checkpoint not found: {self.stage2_checkpoint}")
        
        print(f"\n{'='*60}")
        print(f"Loading Stage 2 checkpoint: {self.stage2_checkpoint}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Load full state into this LightningModule (model + reg heads)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        
        print(f"  Missing keys: {len(missing)}")
        if missing:
            for k in missing[:10]:
                print(f"    {k}")
            if len(missing) > 10:
                print(f"    ... and {len(missing) - 10} more")
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            for k in unexpected[:5]:
                print(f"    {k}")
        
        # Apply freezing policy
        if self.freeze_backbone:
            self.model.freeze_encoder()
            self.model.freeze_classification_heads()
        else:
            # Stage 3: unfreeze everything
            self.model.unfreeze_encoder()
            self.model.unfreeze_classification_heads()
        
        self._print_param_summary()
    
    def _print_param_summary(self):
        """Print parameter count summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n  Total parameters:     {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Frozen parameters:    {frozen:,}")
        print(f"{'='*60}\n")
    
    def _get_true_bins(
        self, eta: Tensor, phi: Tensor, pt: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert true continuous values to finest-level bin indices."""
        # Eta
        eta_bins = torch.searchsorted(self.eta_bin_edges[1:-1], eta)
        eta_bins = eta_bins.clamp(0, self.n_eta_bins - 1)
        
        # Phi
        phi_bins = torch.searchsorted(self.phi_bin_edges[1:-1], phi)
        phi_bins = phi_bins.clamp(0, self.n_phi_bins - 1)
        
        # Pt (need to handle inv_pt: bin edges are in 1/pt space)
        if self.use_inv_pt:
            pt_clamped = pt.clamp(self.pt_min, self.pt_max)
            pt_for_binning = 1.0 / pt_clamped
            pt_bins = torch.searchsorted(self.pt_bin_edges[1:-1], pt_for_binning)
        else:
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
        """Compute scalar offset targets from bin centers.
        
        Returns
        -------
        eta_offset : Tensor
            (B,) eta offset in absolute units (truth - center).
        phi_offset : Tensor
            (B,) phi offset in radians, wrapped via atan2.
        pt_offset : Tensor
            (B,) pt offset in log space: log(truth_pt) - log(center_pt).
        """
        # Eta: simple linear offset
        eta_centers = self.eta_bin_centers[true_eta_bins]
        eta_offset = true_eta - eta_centers
        
        # Phi: wrapped circular offset
        phi_centers = self.phi_bin_centers[true_phi_bins]
        phi_offset = torch.atan2(
            torch.sin(true_phi - phi_centers),
            torch.cos(true_phi - phi_centers)
        )
        
        # Pt: log-space offset
        # If using inv_pt bins, bin_centers are in 1/pt space → convert to pt
        if self.use_inv_pt:
            pt_centers = self.pt_space_bin_centers[true_pt_bins]
        else:
            pt_centers = self.pt_bin_centers[true_pt_bins]
        
        # log(true_pt) - log(center_pt) = log(true_pt / center_pt)
        pt_clamped = true_pt.clamp(min=self.pt_min)
        pt_centers_clamped = pt_centers.clamp(min=1.0)
        pt_offset = torch.log(pt_clamped) - torch.log(pt_centers_clamped)
        
        return eta_offset, phi_offset, pt_offset
    
    def _reconstruct_predictions(
        self,
        eta_offset: Tensor,
        phi_offset: Tensor,
        pt_offset: Tensor,
        cls_eta_logits: Tensor,
        cls_phi_logits: Tensor,
        cls_pt_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct continuous predictions from classification + offset.
        
        eta_pred = bin_center[argmax(eta_logits)] + eta_offset
        phi_pred = wrap(bin_center[argmax(phi_logits)] + phi_offset)
        pt_pred  = exp(log(bin_center[argmax(pt_logits)]) + pt_offset)
        """
        # Get predicted bins from frozen classifier
        pred_eta_bins = cls_eta_logits.argmax(dim=-1)
        pred_phi_bins = cls_phi_logits.argmax(dim=-1)
        pred_pt_bins = cls_pt_logits.argmax(dim=-1)
        
        # Eta reconstruction
        eta_centers = self.eta_bin_centers[pred_eta_bins]
        pred_eta = eta_centers + eta_offset
        
        # Phi reconstruction (with wrapping)
        phi_centers = self.phi_bin_centers[pred_phi_bins]
        pred_phi_raw = phi_centers + phi_offset
        pred_phi = torch.atan2(torch.sin(pred_phi_raw), torch.cos(pred_phi_raw))
        
        # Pt reconstruction (from log-space offset)
        if self.use_inv_pt:
            pt_centers = self.pt_space_bin_centers[pred_pt_bins]
        else:
            pt_centers = self.pt_bin_centers[pred_pt_bins]
        pred_pt = pt_centers * torch.exp(pt_offset)
        
        return pred_eta, pred_phi, pred_pt
    
    def forward(self, inputs: dict) -> dict:
        """Forward pass: frozen model + regression heads.
        
        Returns
        -------
        dict with:
            - cls_embedding: (B, dim)
            - eta_logits, phi_logits, pt_logits: from frozen classifier
            - charge_logit: from frozen classifier
            - eta_offset, phi_offset, pt_offset: (B,) scalar offsets
        """
        # Run frozen model (classification only)
        model_outputs = self.model(
            inputs,
            run_classification=True,
            run_regression=False,
        )
        
        cls_embedding = model_outputs['cls_embedding']
        
        # Run external regression heads
        eta_offset = self.eta_reg_head(cls_embedding).squeeze(-1)  # (B,)
        phi_offset = self.phi_reg_head(cls_embedding).squeeze(-1)  # (B,)
        pt_offset = self.pt_reg_head(cls_embedding).squeeze(-1)    # (B,)
        
        model_outputs['eta_offset'] = eta_offset
        model_outputs['phi_offset'] = phi_offset
        model_outputs['pt_offset'] = pt_offset
        
        return model_outputs
    
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Training step: compute regression loss on scalar offsets."""
        inputs, targets = batch
        outputs = self(inputs)
        
        # True values
        true_eta = targets['eta']
        true_phi = targets['phi']
        true_pt = targets['pt']
        
        # True bins (finest level)
        true_eta_bins, true_phi_bins, true_pt_bins = self._get_true_bins(
            true_eta, true_phi, true_pt
        )
        
        # Target offsets
        target_eta, target_phi, target_pt = self._compute_target_offsets(
            true_eta, true_phi, true_pt,
            true_eta_bins, true_phi_bins, true_pt_bins,
        )
        
        # Predicted offsets
        pred_eta = outputs['eta_offset']
        pred_phi = outputs['phi_offset']
        pred_pt = outputs['pt_offset']
        
        # Smooth L1 losses
        eta_loss = F.smooth_l1_loss(pred_eta, target_eta, beta=self.smooth_l1_beta)
        phi_loss = F.smooth_l1_loss(pred_phi, target_phi, beta=self.smooth_l1_beta)
        pt_loss = F.smooth_l1_loss(pred_pt, target_pt, beta=self.smooth_l1_beta)
        
        total_loss = (
            self.eta_loss_weight * eta_loss +
            self.phi_loss_weight * phi_loss +
            self.pt_loss_weight * pt_loss
        )
        
        # Logging
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/eta_loss', eta_loss)
        self.log('train/phi_loss', phi_loss)
        self.log('train/pt_loss', pt_loss)
        
        # Offset MAE
        self.train_eta_offset_mae(pred_eta, target_eta)
        self.train_phi_offset_mae(pred_phi, target_phi)
        self.train_pt_offset_mae(pred_pt, target_pt)
        self.log('train/eta_offset_mae', self.train_eta_offset_mae, on_step=False, on_epoch=True)
        self.log('train/phi_offset_mae', self.train_phi_offset_mae, on_step=False, on_epoch=True)
        self.log('train/pt_offset_mae', self.train_pt_offset_mae, on_step=False, on_epoch=True)
        
        # Frozen classifier accuracy (monitoring)
        eta_logits = outputs['eta_logits']
        phi_logits = outputs['phi_logits']
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        pt_logits = outputs[f'{pt_prefix}_logits']
        
        self.train_eta_acc(eta_logits, true_eta_bins)
        self.train_phi_acc(phi_logits, true_phi_bins)
        self.train_pt_acc(pt_logits, true_pt_bins)
        self.log('train/cls_eta_acc', self.train_eta_acc, on_step=False, on_epoch=True)
        self.log('train/cls_phi_acc', self.train_phi_acc, on_step=False, on_epoch=True)
        self.log('train/cls_pt_acc', self.train_pt_acc, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        """Validation step: compute loss + physics metrics."""
        inputs, targets = batch
        outputs = self(inputs)
        
        true_eta = targets['eta']
        true_phi = targets['phi']
        true_pt = targets['pt']
        
        true_eta_bins, true_phi_bins, true_pt_bins = self._get_true_bins(
            true_eta, true_phi, true_pt
        )
        
        target_eta, target_phi, target_pt = self._compute_target_offsets(
            true_eta, true_phi, true_pt,
            true_eta_bins, true_phi_bins, true_pt_bins,
        )
        
        pred_eta_off = outputs['eta_offset']
        pred_phi_off = outputs['phi_offset']
        pred_pt_off = outputs['pt_offset']
        
        # Regression loss
        eta_loss = F.smooth_l1_loss(pred_eta_off, target_eta, beta=self.smooth_l1_beta)
        phi_loss = F.smooth_l1_loss(pred_phi_off, target_phi, beta=self.smooth_l1_beta)
        pt_loss = F.smooth_l1_loss(pred_pt_off, target_pt, beta=self.smooth_l1_beta)
        
        total_loss = (
            self.eta_loss_weight * eta_loss +
            self.phi_loss_weight * phi_loss +
            self.pt_loss_weight * pt_loss
        )
        
        self.log('val/loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('val/eta_loss', eta_loss, sync_dist=True)
        self.log('val/phi_loss', phi_loss, sync_dist=True)
        self.log('val/pt_loss', pt_loss, sync_dist=True)
        
        # Offset MAE
        self.val_eta_offset_mae(pred_eta_off, target_eta)
        self.val_phi_offset_mae(pred_phi_off, target_phi)
        self.val_pt_offset_mae(pred_pt_off, target_pt)
        self.log('val/eta_offset_mae', self.val_eta_offset_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/phi_offset_mae', self.val_phi_offset_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/pt_offset_mae', self.val_pt_offset_mae, on_step=False, on_epoch=True, sync_dist=True)
        
        # Frozen classifier accuracy
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        eta_logits = outputs['eta_logits']
        phi_logits = outputs['phi_logits']
        pt_logits = outputs[f'{pt_prefix}_logits']
        
        self.val_eta_acc(eta_logits, true_eta_bins)
        self.val_phi_acc(phi_logits, true_phi_bins)
        self.val_pt_acc(pt_logits, true_pt_bins)
        self.log('val/cls_eta_acc', self.val_eta_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_phi_acc', self.val_phi_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_pt_acc', self.val_pt_acc, on_step=False, on_epoch=True, sync_dist=True)
        
        # Reconstruct final continuous predictions (cls + regression)
        pred_eta, pred_phi, pred_pt = self._reconstruct_predictions(
            pred_eta_off, pred_phi_off, pred_pt_off,
            eta_logits, phi_logits, pt_logits,
        )
        
        # Accumulate physics residuals
        eta_error = pred_eta - true_eta
        phi_error = torch.atan2(
            torch.sin(pred_phi - true_phi),
            torch.cos(pred_phi - true_phi)
        )
        pt_error = pred_pt - true_pt
        
        self.val_eta_errors.append(eta_error.detach().cpu())
        self.val_phi_errors.append(phi_error.detach().cpu())
        self.val_pt_errors.append(pt_error.detach().cpu())
        self.val_pt_true.append(true_pt.detach().cpu())
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level physics metrics."""
        if not self.val_eta_errors:
            return
        
        eta_errors = torch.cat(self.val_eta_errors)
        phi_errors = torch.cat(self.val_phi_errors)
        pt_errors = torch.cat(self.val_pt_errors)
        pt_true = torch.cat(self.val_pt_true)
        
        # Absolute metrics
        eta_mae = eta_errors.abs().mean()
        eta_std = eta_errors.std()
        phi_mae = phi_errors.abs().mean()
        phi_std = phi_errors.std()
        pt_mae = pt_errors.abs().mean()
        pt_std = pt_errors.std()
        
        # Relative pt metrics
        pt_rel_error = pt_errors / pt_true.clamp(min=1.0)
        pt_rel_mae = pt_rel_error.abs().mean()
        pt_rel_std = pt_rel_error.std()
        
        # Log physics metrics
        self.log('val/combined_eta_mae', eta_mae, sync_dist=True)
        self.log('val/combined_eta_std', eta_std, sync_dist=True)
        self.log('val/combined_eta_mae_mrad', eta_mae * 1000, sync_dist=True)
        self.log('val/combined_eta_std_mrad', eta_std * 1000, sync_dist=True)
        self.log('val/combined_phi_mae_rad', phi_mae, sync_dist=True)
        self.log('val/combined_phi_std_rad', phi_std, sync_dist=True)
        self.log('val/combined_phi_mae_mrad', phi_mae * 1000, sync_dist=True)
        self.log('val/combined_phi_std_mrad', phi_std * 1000, sync_dist=True)
        self.log('val/combined_pt_mae_GeV', pt_mae, sync_dist=True)
        self.log('val/combined_pt_std_GeV', pt_std, sync_dist=True)
        self.log('val/combined_pt_relative_mae_pct', pt_rel_mae * 100, sync_dist=True)
        self.log('val/combined_pt_relative_std_pct', pt_rel_std * 100, sync_dist=True)
        
        self._reset_val_accumulators()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler for regression heads only."""
        # Only trainable params (regression heads)
        params = [p for p in self.parameters() if p.requires_grad]
        
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
