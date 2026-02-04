"""Training module for YOLO-style classification pretraining.

Stage 1 of the 3-stage training pipeline:
- Classification heads for eta, phi, pt (binned) and charge (binary)
- Focal loss for pT to handle class imbalance
- Unbinned physics metrics: MAE and std of residuals (unweighted over batch)

The model learns to classify track parameters into discrete bins,
which will later be refined in Stage 2 (regression warmup) and
Stage 3 (joint finetuning).
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric, AUROC

from hepattn.models.yolo_regressor import YOLORegressor, FocalLoss


class ClassificationTrainingModule(LightningModule):
    """Lightning module for classification pretraining (Stage 1).
    
    Handles:
    - Loading bin arrays from disk
    - Computing classification targets (bin indices)
    - Combined loss: CE(eta) + CE(phi) + Focal(pt) + BCE(charge)
    - Physics metrics: unbinned MAE and residual std for eta, phi, pt
      (computed unweighted across the full validation set)
    
    Parameters
    ----------
    model : YOLORegressor
        The YOLO regression model.
    name : str
        Name for the experiment (used by CLI for logging).
    optimizer : str
        Optimizer name ('Adam', 'AdamW', 'Lion').
    lrs_config : dict
        Learning rate scheduler config with keys:
        initial, max, end, pct_start, weight_decay.
    bin_dir : str
        Directory containing bin arrays (.npy files).
    eta_bins : int
        Number of eta bins (must match model).
    phi_bins : int
        Number of phi bins (must match model).
    pt_bins : int
        Number of pt bins (must match model).
    eta_loss_weight : float
        Weight for eta classification loss.
    phi_loss_weight : float
        Weight for phi classification loss.
    pt_loss_weight : float
        Weight for pt classification loss.
    charge_loss_weight : float
        Weight for charge classification loss.
    focal_gamma : float
        Gamma parameter for focal loss on pt.
    """
    
    def __init__(
        self,
        model: dict | YOLORegressor,
        name: str = "YOLO-Stage1",
        optimizer: str = "AdamW",
        lrs_config: dict | None = None,
        bin_dir: str = "/scratch/ml_training_data_2694000_regression_true_hits_only/bins",
        eta_bins: int = 100,
        phi_bins: int = 100,
        pt_bins: int = 50,
        eta_loss_weight: float = 1.0,
        phi_loss_weight: float = 1.0,
        pt_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        focal_gamma: float = 2.0,
        # Clipping ranges (must match bin computation)
        pt_min: float = 5.0,
        pt_max: float = 200.0,
        eta_min: float = -2.7,
        eta_max: float = 2.7,
        eta_gap_min: float = -0.1,
        eta_gap_max: float = 0.1,
        # Seed-biased softmax parameters
        use_seed_bias_eta: bool = False,
        use_seed_bias_phi: bool = False,
        eta_seed_sigma: float = 0.05,  # ~50 mrad from innermost hit std
        phi_seed_sigma: float = 0.05,  # ~50 mrad from innermost hit std
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Store name for CLI
        self.name = name
        
        # Model - handle Lightning CLI's class_path/init_args structure
        if isinstance(model, dict):
            if 'class_path' in model and 'init_args' in model:
                # Lightning CLI structure - model will be instantiated by CLI
                # This shouldn't happen if CLI is properly configured
                self.model = YOLORegressor(**model['init_args'])
            else:
                # Plain dict of arguments
                self.model = YOLORegressor(**model)
        else:
            # Already instantiated model
            self.model = model
            
        # Store config
        self.optimizer_name = optimizer
        self.lrs_config = lrs_config or {
            'initial': 1e-5,
            'max': 1e-4,
            'end': 1e-6,
            'pct_start': 0.05,
            'weight_decay': 1e-5,
        }
        
        self.eta_bins = eta_bins
        self.phi_bins = phi_bins
        self.pt_bins = pt_bins
        
        # Loss weights
        self.eta_loss_weight = eta_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.pt_loss_weight = pt_loss_weight
        self.charge_loss_weight = charge_loss_weight
        
        # Clipping ranges
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_gap_min = eta_gap_min
        self.eta_gap_max = eta_gap_max
        
        # Seed-biased softmax parameters
        self.use_seed_bias_eta = use_seed_bias_eta
        self.use_seed_bias_phi = use_seed_bias_phi
        self.eta_seed_sigma = eta_seed_sigma
        self.phi_seed_sigma = phi_seed_sigma
        
        # Load bin arrays (sets self.actual_eta_bins, etc.)
        bin_dir = Path(bin_dir)
        self._load_bin_arrays(bin_dir)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Metrics for accuracy - eta_bins/phi_bins/pt_bins are now actual counts
        self.train_eta_acc = Accuracy(task='multiclass', num_classes=self.eta_bins)
        self.train_phi_acc = Accuracy(task='multiclass', num_classes=self.phi_bins)
        self.train_pt_acc = Accuracy(task='multiclass', num_classes=self.pt_bins)
        self.train_charge_acc = Accuracy(task='binary')
        
        self.val_eta_acc = Accuracy(task='multiclass', num_classes=self.eta_bins)
        self.val_phi_acc = Accuracy(task='multiclass', num_classes=self.phi_bins)
        self.val_pt_acc = Accuracy(task='multiclass', num_classes=self.pt_bins)
        self.val_charge_acc = Accuracy(task='binary')
        
        # AUC for charge classification
        self.val_charge_auc = AUROC(task='binary')
        
        # Running accumulators for unbinned physics metrics (epoch-level computation)
        # We accumulate predictions and targets across batches, then compute stats at epoch end
        self._reset_physics_accumulators()
        
    def _reset_physics_accumulators(self):
        """Reset accumulators for physics metrics."""
        self._val_eta_preds = []
        self._val_eta_targets = []
        self._val_phi_preds = []
        self._val_phi_targets = []
        self._val_pt_preds = []
        self._val_pt_targets = []
        
    def _load_bin_arrays(self, bin_dir: Path):
        """Load bin edge and center arrays.
        
        Note: File naming convention uses nominal counts (50, 100, 2000, 6000, 12000 for eta)
        but actual counts differ due to gap handling (+1 for eta).
        
        Config should specify ACTUAL bin counts (51, 101, 2001, 6001, 12001 for eta).
        This method maps actual -> file naming for loading.
        """
        # Map actual bin counts to file naming convention
        # eta: 51->50, 101->100, 2001->2000, 6001->6000, 12001->12000
        # phi/pt: counts match file names directly
        eta_actual_to_file = {51: 50, 101: 100, 2001: 2000, 6001: 6000, 12001: 12000}
        eta_file_bins = eta_actual_to_file.get(self.eta_bins, self.eta_bins)
        phi_file_bins = self.phi_bins
        pt_file_bins = self.pt_bins
        
        # Load bin edges
        eta_edges = np.load(bin_dir / f'eta_bins_{eta_file_bins}.npy')
        phi_edges = np.load(bin_dir / f'phi_bins_{phi_file_bins}.npy')
        pt_edges = np.load(bin_dir / f'pt_bins_{pt_file_bins}.npy')
        
        # Load bin centers
        eta_centers = np.load(bin_dir / f'eta_bin_centers_{eta_file_bins}.npy')
        phi_centers = np.load(bin_dir / f'phi_bin_centers_{phi_file_bins}.npy')
        pt_centers = np.load(bin_dir / f'pt_bin_centers_{pt_file_bins}.npy')
        
        # Verify actual bin counts match expectation
        actual_eta = len(eta_centers)
        actual_phi = len(phi_centers)
        actual_pt = len(pt_centers)
        
        if actual_eta != self.eta_bins:
            raise ValueError(f"Config eta_bins={self.eta_bins} but file has {actual_eta} bins")
        if actual_phi != self.phi_bins:
            raise ValueError(f"Config phi_bins={self.phi_bins} but file has {actual_phi} bins")
        if actual_pt != self.pt_bins:
            raise ValueError(f"Config pt_bins={self.pt_bins} but file has {actual_pt} bins")
        
        # Register as buffers (non-trainable, move with model)
        self.register_buffer('eta_bin_edges', torch.from_numpy(eta_edges).float())
        self.register_buffer('phi_bin_edges', torch.from_numpy(phi_edges).float())
        self.register_buffer('pt_bin_edges', torch.from_numpy(pt_edges).float())
        self.register_buffer('eta_bin_centers', torch.from_numpy(eta_centers).float())
        self.register_buffer('phi_bin_centers', torch.from_numpy(phi_centers).float())
        self.register_buffer('pt_bin_centers', torch.from_numpy(pt_centers).float())
        
        print(f"Loaded bin arrays from {bin_dir}")
        print(f"  Eta: {actual_eta} bins, range [{eta_edges[0]:.3f}, {eta_edges[-1]:.3f}]")
        print(f"  Phi: {actual_phi} bins, range [{phi_edges[0]:.3f}, {phi_edges[-1]:.3f}]")
        print(f"  Pt:  {actual_pt} bins, range [{pt_edges[0]:.1f}, {pt_edges[-1]:.1f}] GeV")
        
    def _value_to_bin_index(self, values: Tensor, bin_edges: Tensor) -> Tensor:
        """Convert continuous values to bin indices.
        
        Uses searchsorted to find which bin each value falls into.
        Values outside range are clamped to first/last bin.
        
        Parameters
        ----------
        values : Tensor
            Continuous values, shape (B,).
        bin_edges : Tensor
            Bin edges, shape (num_bins + 1,).
            
        Returns
        -------
        Tensor
            Bin indices, shape (B,), values in [0, num_bins - 1].
        """
        # searchsorted returns index where value would be inserted to maintain order
        # We subtract 1 and clamp to get bin index
        indices = torch.searchsorted(bin_edges, values, right=True) - 1
        indices = indices.clamp(0, len(bin_edges) - 2)  # num_bins - 1
        return indices.long()
    
    def _bin_index_to_value(self, indices: Tensor, bin_centers: Tensor) -> Tensor:
        """Convert bin indices back to continuous values using bin centers.
        
        Parameters
        ----------
        indices : Tensor
            Bin indices, shape (B,).
        bin_centers : Tensor
            Bin centers, shape (num_bins,).
            
        Returns
        -------
        Tensor
            Predicted continuous values, shape (B,).
        """
        return bin_centers[indices]
    
    def _angular_difference(self, phi_pred: Tensor, phi_true: Tensor) -> Tensor:
        """Compute angular difference handling periodicity."""
        diff = phi_pred - phi_true
        return torch.atan2(torch.sin(diff), torch.cos(diff))
    
    def _compute_seed_bias_eta(self, seed_eta: Tensor, bin_centers: Tensor, sigma: float) -> Tensor:
        """Compute Gaussian bias for eta bins based on seed value.
        
        For each sample, computes a log-Gaussian weight for each bin center:
        log_weight[i] = -0.5 * ((center[i] - seed) / sigma)^2
        
        This adds a prior toward bins near the seed value, which is the innermost hit.
        
        Parameters
        ----------
        seed_eta : Tensor
            Seed eta values from innermost hit, shape (B,).
        bin_centers : Tensor
            Eta bin centers, shape (num_bins,).
        sigma : float
            Width of Gaussian in radians.
            
        Returns
        -------
        Tensor
            Log-Gaussian bias to add to logits, shape (B, num_bins).
        """
        # seed_eta: (B,) -> (B, 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        diff = bin_centers.unsqueeze(0) - seed_eta.unsqueeze(1)  # (B, num_bins)
        log_weights = -0.5 * (diff / sigma) ** 2
        return log_weights
    
    def _compute_seed_bias_phi(self, seed_phi: Tensor, bin_centers: Tensor, sigma: float) -> Tensor:
        """Compute Gaussian bias for phi bins with proper circular wrapping.
        
        For phi, we must use angular difference to handle the wrap-around at ±π.
        A seed at +3.0 rad should be close to a bin center at -3.0 rad.
        
        Uses: diff = atan2(sin(center - seed), cos(center - seed))
        This gives the shortest angular path in [-π, π].
        
        Parameters
        ----------
        seed_phi : Tensor
            Seed phi values from innermost hit, shape (B,).
        bin_centers : Tensor
            Phi bin centers, shape (num_bins,).
        sigma : float
            Width of Gaussian in radians.
            
        Returns
        -------
        Tensor
            Log-Gaussian bias to add to logits, shape (B, num_bins).
        """
        # seed_phi: (B,) -> (B, 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        raw_diff = bin_centers.unsqueeze(0) - seed_phi.unsqueeze(1)  # (B, num_bins)
        
        # Wrap difference to [-π, π] using atan2
        wrapped_diff = torch.atan2(torch.sin(raw_diff), torch.cos(raw_diff))
        
        log_weights = -0.5 * (wrapped_diff / sigma) ** 2
        return log_weights
    
    def _compute_classification_targets(self, targets: dict) -> dict:
        """Compute bin indices for each target.
        
        Also handles:
        - PT clipping to [pt_min, pt_max]
        - Eta gap handling (samples in gap get special treatment)
        """
        eta = targets['eta']
        phi = targets['phi']
        pt = targets['pt']
        
        # Clip PT to valid range
        pt_clipped = pt.clamp(self.pt_min, self.pt_max)
        
        # Compute bin indices
        eta_bins = self._value_to_bin_index(eta, self.eta_bin_edges)
        phi_bins = self._value_to_bin_index(phi, self.phi_bin_edges)
        pt_bins = self._value_to_bin_index(pt_clipped, self.pt_bin_edges)
        
        return {
            'eta_bin': eta_bins,
            'phi_bin': phi_bins,
            'pt_bin': pt_bins,
            'charge': targets['charge'],  # Already 0/1 for BCE
        }
    
    def forward(self, inputs: dict) -> dict:
        """Forward pass through the model (classification only for Stage 1)."""
        return self.model(
            inputs,
            run_classification=True,
            run_regression=False,
        )
    
    def _extract_seed_values(self, inputs: dict) -> dict:
        """Extract seed eta/phi from innermost hit (position 1).
        
        The hit features have eta at index 26 (last) and phi at index 25 (second to last).
        Position 0 is reserved for CLS token placeholder, position 1 is the first real hit.
        
        Returns
        -------
        dict with:
            - seed_eta: (B,) - eta of innermost hit
            - seed_phi: (B,) - phi of innermost hit  
        """
        hit_features = inputs['hit_features']
        # Position 1 is the first real hit (position 0 is CLS placeholder)
        # eta is at index 26 (last), phi at index 25 (second to last)
        seed_eta = hit_features[:, 1, 26]  # (B,)
        seed_phi = hit_features[:, 1, 25]  # (B,)
        return {'seed_eta': seed_eta, 'seed_phi': seed_phi}
    
    def _compute_loss(self, outputs: dict, classification_targets: dict, 
                      seed_values: dict = None) -> dict:
        """Compute combined loss.
        
        Parameters
        ----------
        outputs : dict
            Model outputs with logits.
        classification_targets : dict
            Target bin indices.
        seed_values : dict, optional
            Seed eta/phi values for biased softmax. If None, no bias is applied.
        
        Returns dict with individual and total losses.
        """
        eta_logits = outputs['eta_logits']
        phi_logits = outputs['phi_logits']
        
        # Apply seed bias if enabled
        if self.use_seed_bias_eta and seed_values is not None:
            eta_bias = self._compute_seed_bias_eta(
                seed_values['seed_eta'], 
                self.eta_bin_centers, 
                self.eta_seed_sigma
            )
            eta_logits = eta_logits + eta_bias
            
        if self.use_seed_bias_phi and seed_values is not None:
            phi_bias = self._compute_seed_bias_phi(
                seed_values['seed_phi'], 
                self.phi_bin_centers, 
                self.phi_seed_sigma
            )
            phi_logits = phi_logits + phi_bias
        
        # Eta: cross-entropy (with potentially biased logits)
        eta_loss = self.ce_loss(eta_logits, classification_targets['eta_bin'])
        
        # Phi: cross-entropy (with potentially biased logits)
        phi_loss = self.ce_loss(phi_logits, classification_targets['phi_bin'])
        
        # Pt: focal loss (handles class imbalance from quantile binning)
        pt_loss = self.focal_loss(outputs['pt_logits'], classification_targets['pt_bin'])
        
        # Charge: binary cross-entropy
        charge_loss = self.bce_loss(
            outputs['charge_logit'].squeeze(-1),
            classification_targets['charge']
        )
        
        # Combined loss
        total_loss = (
            self.eta_loss_weight * eta_loss +
            self.phi_loss_weight * phi_loss +
            self.pt_loss_weight * pt_loss +
            self.charge_loss_weight * charge_loss
        )
        
        return {
            'loss': total_loss,
            'eta_loss': eta_loss,
            'phi_loss': phi_loss,
            'pt_loss': pt_loss,
            'charge_loss': charge_loss,
        }
    
    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        inputs, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Compute classification targets
        classification_targets = self._compute_classification_targets(targets)
        
        # Extract seed values if using seed bias
        seed_values = None
        if self.use_seed_bias_eta or self.use_seed_bias_phi:
            seed_values = self._extract_seed_values(inputs)
        
        # Compute loss (with optional seed bias)
        losses = self._compute_loss(outputs, classification_targets, seed_values)
        
        # Update accuracy metrics
        eta_preds = outputs['eta_logits'].argmax(dim=-1)
        phi_preds = outputs['phi_logits'].argmax(dim=-1)
        pt_preds = outputs['pt_logits'].argmax(dim=-1)
        charge_preds = (outputs['charge_logit'].squeeze(-1) > 0).long()
        
        self.train_eta_acc(eta_preds, classification_targets['eta_bin'])
        self.train_phi_acc(phi_preds, classification_targets['phi_bin'])
        self.train_pt_acc(pt_preds, classification_targets['pt_bin'])
        self.train_charge_acc(charge_preds, classification_targets['charge'].long())
        
        # Log losses
        self.log('train/loss', losses['loss'], prog_bar=True)
        self.log('train/eta_loss', losses['eta_loss'])
        self.log('train/phi_loss', losses['phi_loss'])
        self.log('train/pt_loss', losses['pt_loss'])
        self.log('train/charge_loss', losses['charge_loss'])
        
        # Log accuracies
        self.log('train/eta_acc', self.train_eta_acc, on_step=False, on_epoch=True)
        self.log('train/phi_acc', self.train_phi_acc, on_step=False, on_epoch=True)
        self.log('train/pt_acc', self.train_pt_acc, on_step=False, on_epoch=True)
        self.log('train/charge_acc', self.train_charge_acc, on_step=False, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        inputs, targets = batch
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Compute classification targets
        classification_targets = self._compute_classification_targets(targets)
        
        # Extract seed values if using seed bias
        seed_values = None
        if self.use_seed_bias_eta or self.use_seed_bias_phi:
            seed_values = self._extract_seed_values(inputs)
        
        # Compute loss (with optional seed bias)
        losses = self._compute_loss(outputs, classification_targets, seed_values)
        
        # Get predicted bin indices
        eta_pred_bins = outputs['eta_logits'].argmax(dim=-1)
        phi_pred_bins = outputs['phi_logits'].argmax(dim=-1)
        pt_pred_bins = outputs['pt_logits'].argmax(dim=-1)
        charge_preds = (outputs['charge_logit'].squeeze(-1) > 0).long()
        charge_probs = torch.sigmoid(outputs['charge_logit'].squeeze(-1))
        
        # Update accuracy metrics
        self.val_eta_acc(eta_pred_bins, classification_targets['eta_bin'])
        self.val_phi_acc(phi_pred_bins, classification_targets['phi_bin'])
        self.val_pt_acc(pt_pred_bins, classification_targets['pt_bin'])
        self.val_charge_acc(charge_preds, classification_targets['charge'].long())
        
        # Update AUC metric for charge
        self.val_charge_auc(charge_probs, classification_targets['charge'].long())
        
        # Convert bin predictions to continuous values for physics metrics
        eta_preds_cont = self._bin_index_to_value(eta_pred_bins, self.eta_bin_centers)
        phi_preds_cont = self._bin_index_to_value(phi_pred_bins, self.phi_bin_centers)
        pt_preds_cont = self._bin_index_to_value(pt_pred_bins, self.pt_bin_centers)
        
        # Accumulate for epoch-level physics metrics
        self._val_eta_preds.append(eta_preds_cont.detach().cpu())
        self._val_eta_targets.append(targets['eta'].cpu())
        self._val_phi_preds.append(phi_preds_cont.detach().cpu())
        self._val_phi_targets.append(targets['phi'].cpu())
        self._val_pt_preds.append(pt_preds_cont.detach().cpu())
        self._val_pt_targets.append(targets['pt'].cpu())
        
        # Log losses
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val/eta_loss', losses['eta_loss'], sync_dist=True)
        self.log('val/phi_loss', losses['phi_loss'], sync_dist=True)
        self.log('val/pt_loss', losses['pt_loss'], sync_dist=True)
        self.log('val/charge_loss', losses['charge_loss'], sync_dist=True)
        
        # Log accuracies and AUC
        self.log('val/eta_acc', self.val_eta_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/phi_acc', self.val_phi_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/pt_acc', self.val_pt_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/charge_acc', self.val_charge_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/charge_auc', self.val_charge_auc, on_step=False, on_epoch=True, sync_dist=True)
        
        return losses
    
    def on_validation_epoch_end(self):
        """Compute epoch-level physics metrics."""
        # Concatenate accumulated predictions and targets
        eta_preds = torch.cat(self._val_eta_preds)
        eta_targets = torch.cat(self._val_eta_targets)
        phi_preds = torch.cat(self._val_phi_preds)
        phi_targets = torch.cat(self._val_phi_targets)
        pt_preds = torch.cat(self._val_pt_preds)
        pt_targets = torch.cat(self._val_pt_targets)
        
        # Eta: MAE and residual std
        eta_residuals = eta_preds - eta_targets
        eta_mae = eta_residuals.abs().mean()
        eta_std = eta_residuals.std()
        
        # Phi: MAE and residual std (with periodicity handling)
        phi_residuals = torch.atan2(
            torch.sin(phi_preds - phi_targets),
            torch.cos(phi_preds - phi_targets)
        )
        phi_mae = phi_residuals.abs().mean()
        phi_std = phi_residuals.std()
        
        # Pt: MAE and residual std (relative for pt)
        pt_residuals = pt_preds - pt_targets
        pt_mae = pt_residuals.abs().mean()
        pt_std = pt_residuals.std()
        
        # Also compute relative pt metrics
        pt_relative_residuals = (pt_preds - pt_targets) / pt_targets.clamp(min=1.0)
        pt_relative_mae = pt_relative_residuals.abs().mean()
        pt_relative_std = pt_relative_residuals.std()
        
        # Log physics metrics (in mrad for angular, GeV for pt)
        self.log('val/eta_mae_mrad', eta_mae * 1000, sync_dist=True)
        self.log('val/eta_std_mrad', eta_std * 1000, sync_dist=True)
        self.log('val/phi_mae_mrad', phi_mae * 1000, sync_dist=True)
        self.log('val/phi_std_mrad', phi_std * 1000, sync_dist=True)
        self.log('val/pt_mae_GeV', pt_mae, sync_dist=True)
        self.log('val/pt_std_GeV', pt_std, sync_dist=True)
        self.log('val/pt_relative_mae_pct', pt_relative_mae * 100, sync_dist=True)
        self.log('val/pt_relative_std_pct', pt_relative_std * 100, sync_dist=True)
        
        # Reset accumulators
        self._reset_physics_accumulators()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get optimizer
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lrs_config['initial'],
                weight_decay=self.lrs_config.get('weight_decay', 0),
            )
        elif self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lrs_config['initial'],
                weight_decay=self.lrs_config.get('weight_decay', 1e-5),
            )
        elif self.optimizer_name == 'Lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(
                    self.parameters(),
                    lr=self.lrs_config['initial'],
                    weight_decay=self.lrs_config.get('weight_decay', 1e-5),
                )
            except ImportError:
                print("Lion not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
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
