"""Dual-backbone regression: separate trainable encoder for offset prediction.

Architecture:
  - FROZEN classification backbone (YOLORegressor from Stage 1):
    Input → InputProjection → CLS + PosEmbed → BidirectionalMamba → ClassificationHeads
    Produces bin predictions (argmax of logits gives bin index).
    All parameters frozen.

  - TRAINABLE regression backbone (fresh, same architecture):
    Input → InputProjection → CLS + PosEmbed → BidirectionalMamba → RegressionHeads
    Processes the SAME hit sequence independently.
    Learns its own representation specialized for high-resolution offset regression.
    Regression heads output 1 scalar per variable (absolute offset from bin center).

Offset targets:
  eta: truth_eta - bin_center[true_bin]  (linear)
  phi: atan2(sin(truth_phi - center), cos(truth_phi - center))  (wrapped)
  pt:  log(truth_pt) - log(bin_center_pt)  (log-space)

Loss: Smooth L1 on scalar offsets.

Inference reconstruction:
  eta_pred = eta_center[argmax(cls_eta)] + reg_eta_offset
  phi_pred = wrap(phi_center[argmax(cls_phi)] + reg_phi_offset)
  pt_pred  = exp(log(pt_center[argmax(cls_pt)]) + reg_pt_offset)

The regression backbone has the same model dimension, number of layers,
and SSM parameters as the classification backbone, so it has equal
capacity to learn a rich sequence representation—but one that is
optimized specifically for predicting within-bin offsets rather than
bin classification.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from lightning import LightningModule
from torchmetrics import Accuracy, MeanAbsoluteError

from hepattn.models.yolo_regressor import YOLORegressor
from hepattn.models.mamba import BidirectionalMambaEncoder


class YOLORegressionDualBackbone(LightningModule):
    """Lightning module for dual-backbone regression training.

    Two independent backbones process the same hit sequence:
      1. Frozen classification backbone → bin predictions
      2. Trainable regression backbone → scalar offsets from bin centers

    Parameters
    ----------
    cls_model : dict or YOLORegressor
        Frozen classification model configuration (must match Stage 1 checkpoint).
    stage1_checkpoint : str
        Path to Stage 1 classification checkpoint (loads into cls_model, frozen).
    bin_dir : str
        Directory with bin arrays.
    eta_bins, phi_bins, pt_bins : int or list[int]
        Bin configuration (finest level used for regression).
    use_inv_pt : bool
        Whether Stage 1 used 1/pt binning.

    Regression backbone (trainable, same architecture):
    reg_dim : int
        Embedding dimension for regression backbone (default: same as cls).
    reg_num_layers : int
        Number of Mamba layers (default: same as cls).
    reg_d_state, reg_d_conv, reg_expand : int
        SSM parameters (default: same as cls).
    reg_head_hidden_dim : int
        Hidden dim for regression heads (default: same as cls head_hidden_dim).
    reg_head_dropout : float
        Dropout for regression heads.
    """

    def __init__(
        self,
        cls_model: dict | YOLORegressor,
        name: str = "YOLO-DualBackbone-Regression",
        optimizer: str = "AdamW",
        lrs_config: dict | None = None,
        stage1_checkpoint: str = "",
        bin_dir: str = "/scratch/ml_training_data_2694000_regression_true_hits_only/bins",
        eta_bins: int | list[int] = 501,
        phi_bins: int | list[int] = 100,
        pt_bins: int | list[int] = 50,
        use_inv_pt: bool = False,
        pt_log_bins: bool = False,
        # Regression backbone architecture (mirrors cls backbone)
        input_dim: int = 27,
        reg_dim: int = 128,
        reg_num_layers: int = 2,
        reg_d_state: int = 16,
        reg_d_conv: int = 4,
        reg_expand: int = 2,
        reg_use_mamba2: bool = True,
        reg_headdim: int = 32,
        reg_norm: str = "RMSNorm",
        reg_dropout: float = 0.0,
        # Regression heads (same size as classification heads)
        reg_head_hidden_dim: int = 512,
        reg_head_dropout: float = 0.15,
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['cls_model'])

        self.name = name
        self.stage1_checkpoint = stage1_checkpoint
        self.bin_dir = Path(bin_dir)
        self.use_inv_pt = use_inv_pt
        self.pt_log_bins = pt_log_bins

        # Optimizer config
        self.optimizer_name = optimizer
        self.lrs_config = lrs_config or {
            'initial': 1e-5,
            'max': 1e-4,
            'end': 1e-6,
            'pct_start': 0.05,
            'weight_decay': 1e-5,
        }

        # Bins (finest level)
        self.eta_bins_list = [eta_bins] if isinstance(eta_bins, int) else list(eta_bins)
        self.phi_bins_list = [phi_bins] if isinstance(phi_bins, int) else list(phi_bins)
        self.pt_bins_list = [pt_bins] if isinstance(pt_bins, int) else list(pt_bins)
        self.n_eta_bins = self.eta_bins_list[-1]
        self.n_phi_bins = self.phi_bins_list[-1]
        self.n_pt_bins = self.pt_bins_list[-1]

        # Clipping
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_min = eta_min
        self.eta_max = eta_max

        # Loss
        self.smooth_l1_beta = smooth_l1_beta
        self.eta_loss_weight = eta_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.pt_loss_weight = pt_loss_weight

        # ===== 1. Classification backbone (frozen) =====
        if isinstance(cls_model, dict):
            if 'class_path' in cls_model and 'init_args' in cls_model:
                self.cls_model = YOLORegressor(**cls_model['init_args'])
            elif 'class_path' in cls_model:
                self.cls_model = YOLORegressor(**cls_model.get('init_args', {}))
            else:
                self.cls_model = YOLORegressor(**cls_model)
        else:
            self.cls_model = cls_model

        # ===== 2. Regression backbone (trainable, trained from scratch) =====
        self.reg_dim = reg_dim
        self.input_dim = input_dim

        # Input projection (independent from classification)
        self.reg_input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, reg_dim),
        )

        # CLS token (independent)
        self.reg_cls_token = nn.Parameter(torch.zeros(1, 1, reg_dim))
        nn.init.trunc_normal_(self.reg_cls_token, std=0.02)

        # Positional embedding (independent)
        self.max_seq_len = 256
        self.reg_pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, reg_dim))
        nn.init.trunc_normal_(self.reg_pos_embedding, std=0.02)

        # Bidirectional Mamba encoder (independent, same architecture)
        self.reg_encoder = BidirectionalMambaEncoder(
            num_layers=reg_num_layers,
            dim=reg_dim,
            d_state=reg_d_state,
            d_conv=reg_d_conv,
            expand=reg_expand,
            use_mamba2=reg_use_mamba2,
            headdim=reg_headdim,
            norm=reg_norm,
            dropout=reg_dropout,
        )

        # ===== 3. Regression heads (output 1 scalar each) =====
        self.eta_reg_head = nn.Sequential(
            nn.Linear(reg_dim, reg_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(reg_head_dropout),
            nn.Linear(reg_head_hidden_dim, 1),
        )
        self.phi_reg_head = nn.Sequential(
            nn.Linear(reg_dim, reg_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(reg_head_dropout),
            nn.Linear(reg_head_hidden_dim, 1),
        )
        self.pt_reg_head = nn.Sequential(
            nn.Linear(reg_dim, reg_head_hidden_dim),
            nn.GELU(),
            nn.Dropout(reg_head_dropout),
            nn.Linear(reg_head_hidden_dim, 1),
        )

        # Initialize regression components
        self._init_regression_weights()

        # Load bins
        self._load_bins()

        # Metrics
        self._setup_metrics()

        # Validation accumulators
        self._reset_val_accumulators()

    def _init_regression_weights(self):
        """Initialize all regression backbone and head weights."""
        for module in [self.reg_input_projection, self.eta_reg_head,
                       self.phi_reg_head, self.pt_reg_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _load_bins(self):
        """Load bin edges and centers for finest classification level."""
        eta_actual_to_file = {
            51: 50, 101: 100, 501: 500, 1001: 1000,
            2001: 2000, 6001: 6000, 12001: 12000,
        }

        # Eta
        eta_file = eta_actual_to_file.get(self.n_eta_bins, self.n_eta_bins)
        eta_edges = np.load(self.bin_dir / f'eta_bins_{eta_file}.npy')
        eta_centers = np.load(self.bin_dir / f'eta_bin_centers_{eta_file}.npy')
        assert len(eta_centers) == self.n_eta_bins
        self.register_buffer('eta_bin_edges', torch.from_numpy(eta_edges).float())
        self.register_buffer('eta_bin_centers', torch.from_numpy(eta_centers).float())

        # Phi
        phi_edges = np.load(self.bin_dir / f'phi_bins_{self.n_phi_bins}.npy')
        phi_centers = np.load(self.bin_dir / f'phi_bin_centers_{self.n_phi_bins}.npy')
        assert len(phi_centers) == self.n_phi_bins
        self.register_buffer('phi_bin_edges', torch.from_numpy(phi_edges).float())
        self.register_buffer('phi_bin_centers', torch.from_numpy(phi_centers).float())

        # Pt / inv_pt
        if self.use_inv_pt:
            pt_edges = np.load(self.bin_dir / f'inv_pt_bins_{self.n_pt_bins}.npy')
            pt_centers = np.load(self.bin_dir / f'inv_pt_bin_centers_{self.n_pt_bins}.npy')
        else:
            suffix = '_log' if self.pt_log_bins else ''
            pt_edges = np.load(self.bin_dir / f'pt_bins_{self.n_pt_bins}{suffix}.npy')
            pt_centers = np.load(self.bin_dir / f'pt_bin_centers_{self.n_pt_bins}{suffix}.npy')
        assert len(pt_centers) == self.n_pt_bins
        self.register_buffer('pt_bin_edges', torch.from_numpy(pt_edges).float())
        self.register_buffer('pt_bin_centers', torch.from_numpy(pt_centers).float())

        if self.use_inv_pt:
            pt_space = 1.0 / torch.from_numpy(pt_centers).float().clamp(min=1e-6)
            self.register_buffer('pt_space_bin_centers', pt_space)

        print(f"Loaded bins: Eta={self.n_eta_bins}, Phi={self.n_phi_bins}, "
              f"Pt={self.n_pt_bins} ({'1/pT' if self.use_inv_pt else 'pT'})")

    def _setup_metrics(self):
        """Set up torchmetrics."""
        self.train_eta_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins)
        self.train_phi_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins)
        self.train_pt_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins)

        self.val_eta_acc = Accuracy(task='multiclass', num_classes=self.n_eta_bins)
        self.val_phi_acc = Accuracy(task='multiclass', num_classes=self.n_phi_bins)
        self.val_pt_acc = Accuracy(task='multiclass', num_classes=self.n_pt_bins)

        self.train_eta_offset_mae = MeanAbsoluteError()
        self.train_phi_offset_mae = MeanAbsoluteError()
        self.train_pt_offset_mae = MeanAbsoluteError()

        self.val_eta_offset_mae = MeanAbsoluteError()
        self.val_phi_offset_mae = MeanAbsoluteError()
        self.val_pt_offset_mae = MeanAbsoluteError()

    def _reset_val_accumulators(self):
        self.val_eta_errors = []
        self.val_phi_errors = []
        self.val_pt_errors = []
        self.val_pt_true = []

    # ================================================================
    # Checkpoint loading
    # ================================================================

    def setup(self, stage: str):
        """Load Stage 1 checkpoint into classification backbone, freeze it."""
        if stage == "fit" and self.stage1_checkpoint:
            self._load_stage1_checkpoint()

    def _load_stage1_checkpoint(self):
        """Load checkpoint into cls_model and freeze it entirely."""
        ckpt_path = Path(self.stage1_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {self.stage1_checkpoint}"
            )

        print(f"\n{'='*60}")
        print(f"Loading Stage 1 → cls_model (frozen): {self.stage1_checkpoint}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Strip 'model.' prefix (from ClassificationTrainingModule)
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state[k[6:]] = v
            else:
                model_state[k] = v

        missing, unexpected = self.cls_model.load_state_dict(model_state, strict=False)
        print(f"  Missing keys: {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

        # Freeze everything in cls_model
        self.cls_model.freeze_encoder()
        self.cls_model.freeze_classification_heads()

        # Count parameters
        cls_params = sum(p.numel() for p in self.cls_model.parameters())
        reg_params = (
            sum(p.numel() for p in self.reg_encoder.parameters())
            + sum(p.numel() for p in self.reg_input_projection.parameters())
            + self.reg_cls_token.numel()
            + self.reg_pos_embedding.numel()
            + sum(p.numel() for p in self.eta_reg_head.parameters())
            + sum(p.numel() for p in self.phi_reg_head.parameters())
            + sum(p.numel() for p in self.pt_reg_head.parameters())
        )
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n  Classification backbone: {cls_params:,} params (FROZEN)")
        print(f"  Regression backbone+heads: {reg_params:,} params (TRAINABLE)")
        print(f"  Total: {total:,}  |  Trainable: {trainable:,}")
        print(f"{'='*60}\n")

    # ================================================================
    # Target / offset computation (same as unified)
    # ================================================================

    def _get_true_bins(
        self, eta: Tensor, phi: Tensor, pt: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        eta_bins = torch.searchsorted(self.eta_bin_edges[1:-1], eta)
        eta_bins = eta_bins.clamp(0, self.n_eta_bins - 1)

        phi_bins = torch.searchsorted(self.phi_bin_edges[1:-1], phi)
        phi_bins = phi_bins.clamp(0, self.n_phi_bins - 1)

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
        true_eta: Tensor, true_phi: Tensor, true_pt: Tensor,
        true_eta_bins: Tensor, true_phi_bins: Tensor, true_pt_bins: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute scalar offset targets from bin centers."""
        # Eta
        eta_centers = self.eta_bin_centers[true_eta_bins]
        eta_offset = true_eta - eta_centers

        # Phi (wrapped)
        phi_centers = self.phi_bin_centers[true_phi_bins]
        phi_offset = torch.atan2(
            torch.sin(true_phi - phi_centers),
            torch.cos(true_phi - phi_centers),
        )

        # Pt (log-space)
        if self.use_inv_pt:
            pt_centers = self.pt_space_bin_centers[true_pt_bins]
        else:
            pt_centers = self.pt_bin_centers[true_pt_bins]
        pt_clamped = true_pt.clamp(min=self.pt_min)
        pt_centers_clamped = pt_centers.clamp(min=1.0)
        pt_offset = torch.log(pt_clamped) - torch.log(pt_centers_clamped)

        return eta_offset, phi_offset, pt_offset

    def _reconstruct_predictions(
        self,
        eta_offset: Tensor, phi_offset: Tensor, pt_offset: Tensor,
        cls_eta_logits: Tensor, cls_phi_logits: Tensor, cls_pt_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Combine frozen classification bins + regression offsets."""
        pred_eta_bins = cls_eta_logits.argmax(dim=-1)
        pred_phi_bins = cls_phi_logits.argmax(dim=-1)
        pred_pt_bins = cls_pt_logits.argmax(dim=-1)

        # Eta
        pred_eta = self.eta_bin_centers[pred_eta_bins] + eta_offset

        # Phi (wrap)
        phi_raw = self.phi_bin_centers[pred_phi_bins] + phi_offset
        pred_phi = torch.atan2(torch.sin(phi_raw), torch.cos(phi_raw))

        # Pt (log-space → exp)
        if self.use_inv_pt:
            pt_c = self.pt_space_bin_centers[pred_pt_bins]
        else:
            pt_c = self.pt_bin_centers[pred_pt_bins]
        pred_pt = pt_c * torch.exp(pt_offset)

        return pred_eta, pred_phi, pred_pt

    # ================================================================
    # Forward
    # ================================================================

    def _forward_regression_backbone(self, inputs: dict) -> Tensor:
        """Run the trainable regression backbone on the hit sequence.

        Returns the regression CLS embedding (B, reg_dim).
        """
        hit_features = inputs['hit_features']
        B, seq_len, _ = hit_features.shape

        x = self.reg_input_projection(hit_features)

        cls_tokens = self.reg_cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x[:, 1:, :]], dim=1)
        x = x + self.reg_pos_embedding[:, :seq_len, :]

        x = self.reg_encoder(x)
        reg_embedding = x[:, 0, :]  # CLS token output

        return reg_embedding

    def forward(self, inputs: dict) -> dict:
        """Forward pass through both backbones.

        Returns dict with:
          Classification (frozen):
            eta_logits, phi_logits, pt_logits, charge_logit, cls_embedding
          Regression (trainable):
            reg_embedding, eta_offset, phi_offset, pt_offset
        """
        # 1. Frozen classification backbone
        with torch.no_grad():
            cls_outputs = self.cls_model(
                inputs, run_classification=True, run_regression=False,
            )

        # 2. Trainable regression backbone
        reg_embedding = self._forward_regression_backbone(inputs)

        # 3. Regression heads → scalar offsets
        eta_offset = self.eta_reg_head(reg_embedding).squeeze(-1)
        phi_offset = self.phi_reg_head(reg_embedding).squeeze(-1)
        pt_offset = self.pt_reg_head(reg_embedding).squeeze(-1)

        # Merge outputs
        outputs = cls_outputs
        outputs['reg_embedding'] = reg_embedding
        outputs['eta_offset'] = eta_offset
        outputs['phi_offset'] = phi_offset
        outputs['pt_offset'] = pt_offset

        return outputs

    # ================================================================
    # Training / validation
    # ================================================================

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        true_eta, true_phi, true_pt = targets['eta'], targets['phi'], targets['pt']
        true_eta_bins, true_phi_bins, true_pt_bins = self._get_true_bins(
            true_eta, true_phi, true_pt,
        )

        target_eta, target_phi, target_pt = self._compute_target_offsets(
            true_eta, true_phi, true_pt,
            true_eta_bins, true_phi_bins, true_pt_bins,
        )

        pred_eta = outputs['eta_offset']
        pred_phi = outputs['phi_offset']
        pred_pt = outputs['pt_offset']

        eta_loss = F.smooth_l1_loss(pred_eta, target_eta, beta=self.smooth_l1_beta)
        phi_loss = F.smooth_l1_loss(pred_phi, target_phi, beta=self.smooth_l1_beta)
        pt_loss = F.smooth_l1_loss(pred_pt, target_pt, beta=self.smooth_l1_beta)

        total_loss = (
            self.eta_loss_weight * eta_loss
            + self.phi_loss_weight * phi_loss
            + self.pt_loss_weight * pt_loss
        )

        # Logging
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/eta_loss', eta_loss)
        self.log('train/phi_loss', phi_loss)
        self.log('train/pt_loss', pt_loss)

        self.train_eta_offset_mae(pred_eta, target_eta)
        self.train_phi_offset_mae(pred_phi, target_phi)
        self.train_pt_offset_mae(pred_pt, target_pt)
        self.log('train/eta_offset_mae', self.train_eta_offset_mae,
                 on_step=False, on_epoch=True)
        self.log('train/phi_offset_mae', self.train_phi_offset_mae,
                 on_step=False, on_epoch=True)
        self.log('train/pt_offset_mae', self.train_pt_offset_mae,
                 on_step=False, on_epoch=True)

        # Frozen classifier accuracy (monitoring)
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        self.train_eta_acc(outputs['eta_logits'], true_eta_bins)
        self.train_phi_acc(outputs['phi_logits'], true_phi_bins)
        self.train_pt_acc(outputs[f'{pt_prefix}_logits'], true_pt_bins)
        self.log('train/cls_eta_acc', self.train_eta_acc,
                 on_step=False, on_epoch=True)
        self.log('train/cls_phi_acc', self.train_phi_acc,
                 on_step=False, on_epoch=True)
        self.log('train/cls_pt_acc', self.train_pt_acc,
                 on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)

        true_eta, true_phi, true_pt = targets['eta'], targets['phi'], targets['pt']
        true_eta_bins, true_phi_bins, true_pt_bins = self._get_true_bins(
            true_eta, true_phi, true_pt,
        )

        target_eta, target_phi, target_pt = self._compute_target_offsets(
            true_eta, true_phi, true_pt,
            true_eta_bins, true_phi_bins, true_pt_bins,
        )

        pred_eta_off = outputs['eta_offset']
        pred_phi_off = outputs['phi_offset']
        pred_pt_off = outputs['pt_offset']

        eta_loss = F.smooth_l1_loss(pred_eta_off, target_eta, beta=self.smooth_l1_beta)
        phi_loss = F.smooth_l1_loss(pred_phi_off, target_phi, beta=self.smooth_l1_beta)
        pt_loss = F.smooth_l1_loss(pred_pt_off, target_pt, beta=self.smooth_l1_beta)

        total_loss = (
            self.eta_loss_weight * eta_loss
            + self.phi_loss_weight * phi_loss
            + self.pt_loss_weight * pt_loss
        )

        self.log('val/loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('val/eta_loss', eta_loss, sync_dist=True)
        self.log('val/phi_loss', phi_loss, sync_dist=True)
        self.log('val/pt_loss', pt_loss, sync_dist=True)

        self.val_eta_offset_mae(pred_eta_off, target_eta)
        self.val_phi_offset_mae(pred_phi_off, target_phi)
        self.val_pt_offset_mae(pred_pt_off, target_pt)
        self.log('val/eta_offset_mae', self.val_eta_offset_mae,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/phi_offset_mae', self.val_phi_offset_mae,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/pt_offset_mae', self.val_pt_offset_mae,
                 on_step=False, on_epoch=True, sync_dist=True)

        # Frozen classifier accuracy
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        self.val_eta_acc(outputs['eta_logits'], true_eta_bins)
        self.val_phi_acc(outputs['phi_logits'], true_phi_bins)
        self.val_pt_acc(outputs[f'{pt_prefix}_logits'], true_pt_bins)
        self.log('val/cls_eta_acc', self.val_eta_acc,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_phi_acc', self.val_phi_acc,
                 on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_pt_acc', self.val_pt_acc,
                 on_step=False, on_epoch=True, sync_dist=True)

        # Combined physics predictions
        pred_eta, pred_phi, pred_pt = self._reconstruct_predictions(
            pred_eta_off, pred_phi_off, pred_pt_off,
            outputs['eta_logits'], outputs['phi_logits'],
            outputs[f'{pt_prefix}_logits'],
        )

        eta_error = pred_eta - true_eta
        phi_error = torch.atan2(
            torch.sin(pred_phi - true_phi),
            torch.cos(pred_phi - true_phi),
        )
        pt_error = pred_pt - true_pt

        self.val_eta_errors.append(eta_error.detach().cpu())
        self.val_phi_errors.append(phi_error.detach().cpu())
        self.val_pt_errors.append(pt_error.detach().cpu())
        self.val_pt_true.append(true_pt.detach().cpu())

        return total_loss

    def on_validation_epoch_end(self):
        if not self.val_eta_errors:
            return

        eta_errors = torch.cat(self.val_eta_errors)
        phi_errors = torch.cat(self.val_phi_errors)
        pt_errors = torch.cat(self.val_pt_errors)
        pt_true = torch.cat(self.val_pt_true)

        eta_mae = eta_errors.abs().mean()
        eta_std = eta_errors.std()
        phi_mae = phi_errors.abs().mean()
        phi_std = phi_errors.std()
        pt_mae = pt_errors.abs().mean()
        pt_std = pt_errors.std()

        pt_rel = pt_errors / pt_true.clamp(min=1.0)
        pt_rel_mae = pt_rel.abs().mean()
        pt_rel_std = pt_rel.std()

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

    # ================================================================
    # Optimizer
    # ================================================================

    def configure_optimizers(self):
        """Only optimize regression backbone + heads (cls is frozen)."""
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
            },
        }
