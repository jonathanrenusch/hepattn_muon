"""Training module for YOLO-style classification pretraining.

Stage 1 of the 3-stage training pipeline:
- Classification heads for eta, phi, pt/inv_pt (binned) and charge (binary)
- Hierarchical multi-resolution classification (up to 3 levels per variable)
- 1/pt binning support (detector measures curvature directly)
- Focal loss for pT to handle class imbalance
- Unbinned physics metrics: MAE and std of residuals (unweighted over batch)

The model learns to classify track parameters into discrete bins,
which will later be refined in Stage 2 (regression warmup) and
Stage 3 (joint finetuning).

Hierarchical classification:
    Multiple resolution levels per variable (e.g., eta_bins=[50, 200, 501]).
    Each level has its own loss with configurable weights.
    Coarser levels provide "easy" gradient signal, finer levels for precision.

1/pt classification:
    Set use_inv_pt=True to classify in 1/pt space (curvature).
    Accuracy measured on 1/pt bins, but physics metrics reported in pt space.
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
    - Loading bin arrays from disk (all hierarchical levels)
    - Computing classification targets (bin indices at each level)
    - Combined loss: sum over levels of CE(eta) + CE(phi) + Focal(pt) + BCE(charge)
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
    eta_bins : int or list[int]
        Number of eta bins per level (must match model).
    phi_bins : int or list[int]
        Number of phi bins per level (must match model).
    pt_bins : int or list[int]
        Number of pt bins per level (must match model).
    use_inv_pt : bool
        If True, classify in 1/pt space. Accuracy on 1/pt bins, metrics in pt space.
    hierarchy_loss_weights : list[float] or None
        Per-level loss weight multipliers. Default: equal weights.
        E.g., [0.3, 0.5, 1.0] weights finest level highest.
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
        eta_bins: int | list[int] = 100,
        phi_bins: int | list[int] = 100,
        pt_bins: int | list[int] = 50,
        use_inv_pt: bool = False,
        hierarchy_loss_weights: list[float] | None = None,
        pt_log_bins: bool = False,  # Use logarithmic pt binning files
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
        # Hierarchy weight scheduling
        # When enabled, loss weights per level change over training:
        #   coarse (L0): start_weight → end_weight  (e.g. 1.0 → 0.1)
        #   fine (last): start_weight → end_weight   (e.g. 0.1 → 1.0)
        #   middle levels: constant weight
        schedule_hierarchy_weights: bool = False,
        hierarchy_weight_schedule: dict | None = None,
        # Seed-biased softmax parameters
        use_seed_bias_eta: bool = False,
        use_seed_bias_phi: bool = False,
        eta_seed_sigma: float = 0.05,
        phi_seed_sigma: float = 0.05,
        # Gaussian label smoothing parameters
        use_gaussian_label_smoothing: bool = False,
        eta_label_sigma: float = 0.01,
        phi_label_sigma: float = 0.01,
        pt_label_sigma_frac: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        # Store name for CLI
        self.name = name
        
        # Model - handle Lightning CLI's class_path/init_args structure
        if isinstance(model, dict):
            if 'class_path' in model and 'init_args' in model:
                self.model = YOLORegressor(**model['init_args'])
            else:
                self.model = YOLORegressor(**model)
        else:
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
        
        # Normalize bins to lists for hierarchical support
        self.eta_bins_list = [eta_bins] if isinstance(eta_bins, int) else list(eta_bins)
        self.phi_bins_list = [phi_bins] if isinstance(phi_bins, int) else list(phi_bins)
        self.pt_bins_list = [pt_bins] if isinstance(pt_bins, int) else list(pt_bins)
        
        # Backward-compatible scalar properties: finest level
        self.eta_bins = self.eta_bins_list[-1]
        self.phi_bins = self.phi_bins_list[-1]
        self.pt_bins = self.pt_bins_list[-1]
        
        self.num_eta_levels = len(self.eta_bins_list)
        self.num_phi_levels = len(self.phi_bins_list)
        self.num_pt_levels = len(self.pt_bins_list)
        
        self.use_inv_pt = use_inv_pt
        self.pt_log_bins = pt_log_bins
        
        # Hierarchy level loss weights
        max_levels = max(self.num_eta_levels, self.num_phi_levels, self.num_pt_levels)
        if hierarchy_loss_weights is not None:
            self.hierarchy_loss_weights = list(hierarchy_loss_weights)
            # Pad with 1.0 if not enough weights provided
            while len(self.hierarchy_loss_weights) < max_levels:
                self.hierarchy_loss_weights.append(1.0)
        else:
            # Default: equal weights
            self.hierarchy_loss_weights = [1.0] * max_levels
        
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
        
        # Hierarchy weight scheduling
        self.schedule_hierarchy_weights = schedule_hierarchy_weights
        self.hierarchy_weight_schedule = hierarchy_weight_schedule or {
            'coarse_start': 1.0,
            'coarse_end': 0.1,
            'fine_start': 0.1,
            'fine_end': 1.0,
            'schedule_type': 'cosine',  # 'cosine' or 'linear'
        }
        
        # Gaussian label smoothing parameters
        self.use_gaussian_label_smoothing = use_gaussian_label_smoothing
        self.eta_label_sigma = eta_label_sigma
        self.phi_label_sigma = phi_label_sigma
        self.pt_label_sigma_frac = pt_label_sigma_frac
        
        # Load bin arrays (sets buffers for all levels)
        bin_dir = Path(bin_dir)
        self._load_bin_arrays(bin_dir)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Metrics — finest level only for accuracy
        self.train_eta_acc = Accuracy(task='multiclass', num_classes=self.eta_bins)
        self.train_phi_acc = Accuracy(task='multiclass', num_classes=self.phi_bins)
        self.train_pt_acc = Accuracy(task='multiclass', num_classes=self.pt_bins)
        self.train_charge_acc = Accuracy(task='binary')
        
        self.val_eta_acc = Accuracy(task='multiclass', num_classes=self.eta_bins)
        self.val_phi_acc = Accuracy(task='multiclass', num_classes=self.phi_bins)
        self.val_pt_acc = Accuracy(task='multiclass', num_classes=self.pt_bins)
        self.val_charge_acc = Accuracy(task='binary')
        
        # Per-level accuracy for hierarchical (coarser levels)
        self.val_eta_level_accs = nn.ModuleList([
            Accuracy(task='multiclass', num_classes=n) for n in self.eta_bins_list
        ])
        self.val_phi_level_accs = nn.ModuleList([
            Accuracy(task='multiclass', num_classes=n) for n in self.phi_bins_list
        ])
        self.val_pt_level_accs = nn.ModuleList([
            Accuracy(task='multiclass', num_classes=n) for n in self.pt_bins_list
        ])
        
        # AUC for charge classification
        self.val_charge_auc = AUROC(task='binary')
        
        # Top-k physics-informed accuracy accumulators
        self._val_eta_pred_bins = []
        self._val_eta_true_bins = []
        self._val_phi_pred_bins = []
        self._val_phi_true_bins = []
        self._val_pt_pred_bins = []
        self._val_pt_true_bins = []
        
        # Running accumulators for unbinned physics metrics
        self._reset_physics_accumulators()
        
    def _reset_physics_accumulators(self):
        """Reset accumulators for physics metrics."""
        self._val_eta_preds = []
        self._val_eta_targets = []
        self._val_phi_preds = []
        self._val_phi_targets = []
        self._val_pt_preds = []
        self._val_pt_targets = []
        # Top-k bin accumulators
        self._val_eta_pred_bins = []
        self._val_eta_true_bins = []
        self._val_phi_pred_bins = []
        self._val_phi_true_bins = []
        self._val_pt_pred_bins = []
        self._val_pt_true_bins = []
        
    def _load_bin_arrays(self, bin_dir: Path):
        """Load bin edge and center arrays for all hierarchical levels.
        
        For hierarchical configs (eta_bins=[50, 200, 501]),
        loads bin files for each level. Finest level bins are available
        as self.eta_bin_edges / self.eta_bin_centers (backward compatible).
        All levels available as self.eta_bin_edges_levels[i].
        
        For use_inv_pt=True, loads inv_pt bin files instead of pt files.
        """
        # Map actual bin counts to file naming convention
        # eta: 51->50, 101->100, 501->500, 1001->1000, 2001->2000, ...
        eta_actual_to_file = {51: 50, 101: 100, 501: 500, 1001: 1000, 2001: 2000, 6001: 6000, 12001: 12000}
        
        # Helper to load a single set of bins
        def _load_single_eta(n_bins):
            eta_file = eta_actual_to_file.get(n_bins, n_bins)
            edges = np.load(bin_dir / f'eta_bins_{eta_file}.npy')
            centers = np.load(bin_dir / f'eta_bin_centers_{eta_file}.npy')
            if len(centers) != n_bins:
                raise ValueError(f"Config eta_bins={n_bins} but file has {len(centers)} bins")
            return edges, centers
        
        def _load_single_phi(n_bins):
            edges = np.load(bin_dir / f'phi_bins_{n_bins}.npy')
            centers = np.load(bin_dir / f'phi_bin_centers_{n_bins}.npy')
            if len(centers) != n_bins:
                raise ValueError(f"Config phi_bins={n_bins} but file has {len(centers)} bins")
            return edges, centers
        
        def _load_single_pt(n_bins):
            """Load pt or inv_pt bins depending on use_inv_pt."""
            if self.use_inv_pt:
                edges = np.load(bin_dir / f'inv_pt_bins_{n_bins}.npy')
                centers = np.load(bin_dir / f'inv_pt_bin_centers_{n_bins}.npy')
            else:
                pt_suffix = '_log' if self.pt_log_bins else ''
                edges = np.load(bin_dir / f'pt_bins_{n_bins}{pt_suffix}.npy')
                centers = np.load(bin_dir / f'pt_bin_centers_{n_bins}{pt_suffix}.npy')
            if len(centers) != n_bins:
                raise ValueError(f"Config pt_bins={n_bins} but file has {len(centers)} bins")
            return edges, centers
        
        # Load all eta levels
        self.eta_bin_edges_levels = []
        self.eta_bin_centers_levels = []
        for i, n_bins in enumerate(self.eta_bins_list):
            edges, centers = _load_single_eta(n_bins)
            self.register_buffer(f'eta_bin_edges_L{i}', torch.from_numpy(edges).float())
            self.register_buffer(f'eta_bin_centers_L{i}', torch.from_numpy(centers).float())
            self.eta_bin_edges_levels.append(f'eta_bin_edges_L{i}')
            self.eta_bin_centers_levels.append(f'eta_bin_centers_L{i}')
        
        # Load all phi levels
        self.phi_bin_edges_levels = []
        self.phi_bin_centers_levels = []
        for i, n_bins in enumerate(self.phi_bins_list):
            edges, centers = _load_single_phi(n_bins)
            self.register_buffer(f'phi_bin_edges_L{i}', torch.from_numpy(edges).float())
            self.register_buffer(f'phi_bin_centers_L{i}', torch.from_numpy(centers).float())
            self.phi_bin_edges_levels.append(f'phi_bin_edges_L{i}')
            self.phi_bin_centers_levels.append(f'phi_bin_centers_L{i}')
        
        # Load all pt/inv_pt levels
        self.pt_bin_edges_levels = []
        self.pt_bin_centers_levels = []
        for i, n_bins in enumerate(self.pt_bins_list):
            edges, centers = _load_single_pt(n_bins)
            self.register_buffer(f'pt_bin_edges_L{i}', torch.from_numpy(edges).float())
            self.register_buffer(f'pt_bin_centers_L{i}', torch.from_numpy(centers).float())
            self.pt_bin_edges_levels.append(f'pt_bin_edges_L{i}')
            self.pt_bin_centers_levels.append(f'pt_bin_centers_L{i}')
        
        # Backward-compatible aliases: finest level
        finest_eta = self.num_eta_levels - 1
        finest_phi = self.num_phi_levels - 1
        finest_pt = self.num_pt_levels - 1
        self.register_buffer('eta_bin_edges', getattr(self, f'eta_bin_edges_L{finest_eta}').clone())
        self.register_buffer('eta_bin_centers', getattr(self, f'eta_bin_centers_L{finest_eta}').clone())
        self.register_buffer('phi_bin_edges', getattr(self, f'phi_bin_edges_L{finest_phi}').clone())
        self.register_buffer('phi_bin_centers', getattr(self, f'phi_bin_centers_L{finest_phi}').clone())
        self.register_buffer('pt_bin_edges', getattr(self, f'pt_bin_edges_L{finest_pt}').clone())
        self.register_buffer('pt_bin_centers', getattr(self, f'pt_bin_centers_L{finest_pt}').clone())
        
        # Print summary
        bin_type = '1/pT (curvature)' if self.use_inv_pt else ('log pT' if self.pt_log_bins else 'quantile pT')
        print(f"Loaded bin arrays from {bin_dir}")
        for i, n_bins in enumerate(self.eta_bins_list):
            edges = getattr(self, f'eta_bin_edges_L{i}')
            print(f"  Eta L{i}: {n_bins} bins, range [{edges[0]:.3f}, {edges[-1]:.3f}]")
        for i, n_bins in enumerate(self.phi_bins_list):
            edges = getattr(self, f'phi_bin_edges_L{i}')
            print(f"  Phi L{i}: {n_bins} bins, range [{edges[0]:.3f}, {edges[-1]:.3f}]")
        for i, n_bins in enumerate(self.pt_bins_list):
            edges = getattr(self, f'pt_bin_edges_L{i}')
            unit = '1/GeV' if self.use_inv_pt else 'GeV'
            print(f"  Pt  L{i}: {n_bins} bins ({bin_type}), range [{edges[0]:.4f}, {edges[-1]:.4f}] {unit}")
        
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
    
    def _compute_gaussian_soft_labels_eta(self, target_values: Tensor, bin_centers: Tensor, sigma: float) -> Tensor:
        """Compute Gaussian soft labels for eta classification.
        
        Instead of hard one-hot labels, creates soft labels with Gaussian shape
        centered at the true value. This smooths gradients and provides a physics-
        informed training signal (nearby bins should have similar probability).
        
        Parameters
        ----------
        target_values : Tensor
            True eta values, shape (B,).
        bin_centers : Tensor
            Eta bin centers, shape (num_bins,).
        sigma : float
            Width of Gaussian in radians.
            
        Returns
        -------
        Tensor
            Soft label distribution, shape (B, num_bins), normalized to sum to 1.
        """
        # target_values: (B,) -> (B, 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        diff = bin_centers.unsqueeze(0) - target_values.unsqueeze(1)  # (B, num_bins)
        log_probs = -0.5 * (diff / sigma) ** 2
        # Normalize to probability distribution
        soft_labels = torch.softmax(log_probs, dim=-1)
        return soft_labels
    
    def _compute_gaussian_soft_labels_phi(self, target_values: Tensor, bin_centers: Tensor, sigma: float) -> Tensor:
        """Compute Gaussian soft labels for phi classification with circular wrapping.
        
        For phi, uses circular distance to properly handle wrap-around at ±π.
        
        Parameters
        ----------
        target_values : Tensor
            True phi values, shape (B,).
        bin_centers : Tensor
            Phi bin centers, shape (num_bins,).
        sigma : float
            Width of Gaussian in radians.
            
        Returns
        -------
        Tensor
            Soft label distribution, shape (B, num_bins), normalized to sum to 1.
        """
        # target_values: (B,) -> (B, 1)
        # bin_centers: (num_bins,) -> (1, num_bins)
        raw_diff = bin_centers.unsqueeze(0) - target_values.unsqueeze(1)  # (B, num_bins)
        
        # Wrap difference to [-π, π] using atan2 for proper circular handling
        wrapped_diff = torch.atan2(torch.sin(raw_diff), torch.cos(raw_diff))
        
        log_probs = -0.5 * (wrapped_diff / sigma) ** 2
        # Normalize to probability distribution
        soft_labels = torch.softmax(log_probs, dim=-1)
        return soft_labels
    
    def _compute_gaussian_soft_labels_pt(self, target_values: Tensor, bin_centers: Tensor, sigma_frac: float) -> Tensor:
        """Compute Gaussian soft labels for pt classification in log-space.
        
        For pt, uses log-space distance since pt resolution scales with pt value.
        sigma_frac is the fractional width (e.g., 0.05 = 5% of pt value).
        
        Parameters
        ----------
        target_values : Tensor
            True pt values in GeV, shape (B,).
        bin_centers : Tensor
            Pt bin centers in GeV, shape (num_bins,).
        sigma_frac : float
            Gaussian width as fraction of pt value (in log-space).
            
        Returns
        -------
        Tensor
            Soft label distribution, shape (B, num_bins), normalized to sum to 1.
        """
        # Work in log-space for pt
        log_target = torch.log(target_values.clamp(min=1e-6)).unsqueeze(1)  # (B, 1)
        log_centers = torch.log(bin_centers.clamp(min=1e-6)).unsqueeze(0)  # (1, num_bins)
        
        # sigma in log-space: log(1 + sigma_frac) ≈ sigma_frac for small values
        log_sigma = torch.log(torch.tensor(1.0 + sigma_frac, device=target_values.device))
        
        log_diff = log_centers - log_target  # (B, num_bins)
        log_probs = -0.5 * (log_diff / log_sigma) ** 2
        
        # Normalize to probability distribution
        soft_labels = torch.softmax(log_probs, dim=-1)
        return soft_labels
    
    def _compute_classification_targets(self, targets: dict) -> dict:
        """Compute bin indices for each target at all hierarchical levels.
        
        For use_inv_pt=True, converts pt to 1/pt before binning.
        Returns targets for all levels plus finest-level aliases.
        """
        eta = targets['eta']
        phi = targets['phi']
        pt = targets['pt']
        
        # Clip PT to valid range
        pt_clipped = pt.clamp(self.pt_min, self.pt_max)
        
        # For 1/pt: convert pt to 1/pt for binning
        if self.use_inv_pt:
            pt_for_binning = 1.0 / pt_clipped
        else:
            pt_for_binning = pt_clipped
        
        result = {
            'charge': targets['charge'],
        }
        
        # Eta bins at all levels
        for i in range(self.num_eta_levels):
            edges = getattr(self, f'eta_bin_edges_L{i}')
            key = f'eta_bin_L{i}' if self.num_eta_levels > 1 else 'eta_bin'
            result[key] = self._value_to_bin_index(eta, edges)
        
        # Phi bins at all levels
        for i in range(self.num_phi_levels):
            edges = getattr(self, f'phi_bin_edges_L{i}')
            key = f'phi_bin_L{i}' if self.num_phi_levels > 1 else 'phi_bin'
            result[key] = self._value_to_bin_index(phi, edges)
        
        # PT/inv_pt bins at all levels
        for i in range(self.num_pt_levels):
            edges = getattr(self, f'pt_bin_edges_L{i}')
            key = f'pt_bin_L{i}' if self.num_pt_levels > 1 else 'pt_bin'
            result[key] = self._value_to_bin_index(pt_for_binning, edges)
        
        # Backward-compatible finest-level aliases
        if self.num_eta_levels > 1:
            result['eta_bin'] = result[f'eta_bin_L{self.num_eta_levels - 1}']
        if self.num_phi_levels > 1:
            result['phi_bin'] = result[f'phi_bin_L{self.num_phi_levels - 1}']
        if self.num_pt_levels > 1:
            result['pt_bin'] = result[f'pt_bin_L{self.num_pt_levels - 1}']
        
        return result
    
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
    
    def _get_hierarchy_weights(self) -> list[float]:
        """Get hierarchy loss weights, optionally scheduled over training.
        
        When schedule_hierarchy_weights=True:
          - Coarsest level (L0): interpolates from coarse_start → coarse_end
          - Finest level (last): interpolates from fine_start → fine_end
          - Middle levels: constant at their initial hierarchy_loss_weights value
        
        Scheduling uses cosine or linear interpolation based on
        current_epoch / max_epochs progress.
        
        Returns
        -------
        list[float]
            Per-level loss weight multipliers for the current epoch.
        """
        if not self.schedule_hierarchy_weights or not self.training:
            return self.hierarchy_loss_weights
        
        # Compute progress fraction [0, 1]
        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs <= 1:
            return self.hierarchy_loss_weights
        progress = self.trainer.current_epoch / (max_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        
        sched = self.hierarchy_weight_schedule
        schedule_type = sched.get('schedule_type', 'cosine')
        
        if schedule_type == 'cosine':
            # Smooth cosine interpolation: 0.5 * (1 - cos(pi * t))
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
        else:
            # Linear interpolation
            alpha = progress
        
        # Build scheduled weights
        weights = list(self.hierarchy_loss_weights)
        max_levels = len(weights)
        
        if max_levels >= 2:
            # Coarse level (index 0): coarse_start → coarse_end
            coarse_start = sched.get('coarse_start', 1.0)
            coarse_end = sched.get('coarse_end', 0.1)
            weights[0] = coarse_start + alpha * (coarse_end - coarse_start)
            
            # Finest level (last index): fine_start → fine_end
            fine_start = sched.get('fine_start', 0.1)
            fine_end = sched.get('fine_end', 1.0)
            weights[-1] = fine_start + alpha * (fine_end - fine_start)
            
            # Middle levels: keep their original constant values
        
        return weights
    
    def _compute_loss(self, outputs: dict, classification_targets: dict, 
                      seed_values: dict = None, raw_targets: dict = None) -> dict:
        """Compute combined loss over all hierarchical levels.
        
        For hierarchical models, sums losses from all levels weighted by
        hierarchy_loss_weights. Finest level loss is reported separately.
        
        For 1/pt: loss is computed on 1/pt bins, but pt_for_binning
        is already 1/pt in classification_targets.
        """
        losses = {}
        
        # Determine pt key prefix for model outputs  
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        
        # Get (possibly scheduled) hierarchy weights
        current_weights = self._get_hierarchy_weights()
        
        # ===== ETA losses (all levels) =====
        eta_total_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_eta_levels):
            logit_key = f'eta_logits_L{i}' if self.num_eta_levels > 1 else 'eta_logits'
            target_key = f'eta_bin_L{i}' if self.num_eta_levels > 1 else 'eta_bin'
            logits = outputs[logit_key]
            
            # Apply seed bias at finest level only
            if i == self.num_eta_levels - 1 and self.use_seed_bias_eta and seed_values is not None:
                centers = getattr(self, f'eta_bin_centers_L{i}')
                eta_bias = self._compute_seed_bias_eta(seed_values['seed_eta'], centers, self.eta_seed_sigma)
                logits = logits + eta_bias
            
            if self.use_gaussian_label_smoothing and raw_targets is not None:
                centers = getattr(self, f'eta_bin_centers_L{i}')
                soft_labels = self._compute_gaussian_soft_labels_eta(
                    raw_targets['eta'], centers, self.eta_label_sigma)
                log_probs = torch.log_softmax(logits, dim=-1)
                level_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            else:
                level_loss = self.ce_loss(logits, classification_targets[target_key])
            
            w = current_weights[i] if i < len(current_weights) else 1.0
            eta_total_loss = eta_total_loss + w * level_loss
            losses[f'eta_loss_L{i}'] = level_loss
        
        # ===== PHI losses (all levels) =====
        phi_total_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_phi_levels):
            logit_key = f'phi_logits_L{i}' if self.num_phi_levels > 1 else 'phi_logits'
            target_key = f'phi_bin_L{i}' if self.num_phi_levels > 1 else 'phi_bin'
            logits = outputs[logit_key]
            
            if i == self.num_phi_levels - 1 and self.use_seed_bias_phi and seed_values is not None:
                centers = getattr(self, f'phi_bin_centers_L{i}')
                phi_bias = self._compute_seed_bias_phi(seed_values['seed_phi'], centers, self.phi_seed_sigma)
                logits = logits + phi_bias
            
            if self.use_gaussian_label_smoothing and raw_targets is not None:
                centers = getattr(self, f'phi_bin_centers_L{i}')
                soft_labels = self._compute_gaussian_soft_labels_phi(
                    raw_targets['phi'], centers, self.phi_label_sigma)
                log_probs = torch.log_softmax(logits, dim=-1)
                level_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            else:
                level_loss = self.ce_loss(logits, classification_targets[target_key])
            
            w = current_weights[i] if i < len(current_weights) else 1.0
            phi_total_loss = phi_total_loss + w * level_loss
            losses[f'phi_loss_L{i}'] = level_loss
        
        # ===== PT/inv_pt losses (all levels) =====
        pt_total_loss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_pt_levels):
            logit_key = f'{pt_prefix}_logits_L{i}' if self.num_pt_levels > 1 else f'{pt_prefix}_logits'
            target_key = f'pt_bin_L{i}' if self.num_pt_levels > 1 else 'pt_bin'
            logits = outputs[logit_key]
            
            if self.use_gaussian_label_smoothing and raw_targets is not None:
                centers = getattr(self, f'pt_bin_centers_L{i}')
                if self.use_inv_pt:
                    # For inv_pt, sigma_frac is in inv_pt space
                    soft_labels = self._compute_gaussian_soft_labels_pt(
                        raw_targets['pt_for_binning'], centers, self.pt_label_sigma_frac)
                else:
                    soft_labels = self._compute_gaussian_soft_labels_pt(
                        raw_targets['pt'], centers, self.pt_label_sigma_frac)
                log_probs = torch.log_softmax(logits, dim=-1)
                level_loss = -(soft_labels * log_probs).sum(dim=-1).mean()
            else:
                level_loss = self.focal_loss(logits, classification_targets[target_key])
            
            w = current_weights[i] if i < len(current_weights) else 1.0
            pt_total_loss = pt_total_loss + w * level_loss
            losses[f'pt_loss_L{i}'] = level_loss
        
        # Charge: binary cross-entropy (always single level)
        charge_loss = self.bce_loss(
            outputs['charge_logit'].squeeze(-1),
            classification_targets['charge']
        )
        
        # Combined loss
        total_loss = (
            self.eta_loss_weight * eta_total_loss +
            self.phi_loss_weight * phi_total_loss +
            self.pt_loss_weight * pt_total_loss +
            self.charge_loss_weight * charge_loss
        )
        
        # Store finest-level losses for backward compatibility
        losses['eta_loss'] = losses[f'eta_loss_L{self.num_eta_levels - 1}']
        losses['phi_loss'] = losses[f'phi_loss_L{self.num_phi_levels - 1}']
        losses['pt_loss'] = losses[f'pt_loss_L{self.num_pt_levels - 1}']
        losses['charge_loss'] = charge_loss
        losses['loss'] = total_loss
        
        return losses
    
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
        
        # Prepare raw targets for Gaussian label smoothing
        raw_targets = None
        if self.use_gaussian_label_smoothing:
            pt_clipped = targets['pt'].clamp(self.pt_min, self.pt_max)
            raw_targets = {
                'eta': targets['eta'],
                'phi': targets['phi'],
                'pt': pt_clipped,
                'pt_for_binning': (1.0 / pt_clipped) if self.use_inv_pt else pt_clipped,
            }
        
        # Compute loss (with optional seed bias and label smoothing)
        losses = self._compute_loss(outputs, classification_targets, seed_values, raw_targets)
        
        # Update accuracy metrics (finest level only)
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
        
        # Log per-level losses for hierarchical
        for i in range(self.num_eta_levels):
            if f'eta_loss_L{i}' in losses:
                self.log(f'train/eta_loss_L{i}', losses[f'eta_loss_L{i}'])
        for i in range(self.num_phi_levels):
            if f'phi_loss_L{i}' in losses:
                self.log(f'train/phi_loss_L{i}', losses[f'phi_loss_L{i}'])
        for i in range(self.num_pt_levels):
            if f'pt_loss_L{i}' in losses:
                self.log(f'train/pt_loss_L{i}', losses[f'pt_loss_L{i}'])
        
        # Log scheduled hierarchy weights if scheduling is active
        if self.schedule_hierarchy_weights:
            current_weights = self._get_hierarchy_weights()
            for i, w in enumerate(current_weights):
                self.log(f'train/hierarchy_weight_L{i}', w)
        
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
        
        # Prepare raw targets for Gaussian label smoothing
        raw_targets = None
        if self.use_gaussian_label_smoothing:
            pt_clipped = targets['pt'].clamp(self.pt_min, self.pt_max)
            raw_targets = {
                'eta': targets['eta'],
                'phi': targets['phi'],
                'pt': pt_clipped,
                'pt_for_binning': (1.0 / pt_clipped) if self.use_inv_pt else pt_clipped,
            }
        
        # Compute loss (with optional seed bias and label smoothing)
        losses = self._compute_loss(outputs, classification_targets, seed_values, raw_targets)
        
        # Get predicted bin indices (finest level)
        eta_pred_bins = outputs['eta_logits'].argmax(dim=-1)
        phi_pred_bins = outputs['phi_logits'].argmax(dim=-1)
        pt_pred_bins = outputs['pt_logits'].argmax(dim=-1)
        charge_preds = (outputs['charge_logit'].squeeze(-1) > 0).long()
        charge_probs = torch.sigmoid(outputs['charge_logit'].squeeze(-1))
        
        # Update finest-level accuracy metrics
        self.val_eta_acc(eta_pred_bins, classification_targets['eta_bin'])
        self.val_phi_acc(phi_pred_bins, classification_targets['phi_bin'])
        self.val_pt_acc(pt_pred_bins, classification_targets['pt_bin'])
        self.val_charge_acc(charge_preds, classification_targets['charge'].long())
        
        # Update per-level accuracy metrics for hierarchical
        pt_prefix = 'inv_pt' if self.use_inv_pt else 'pt'
        for i in range(self.num_eta_levels):
            logit_key = f'eta_logits_L{i}' if self.num_eta_levels > 1 else 'eta_logits'
            target_key = f'eta_bin_L{i}' if self.num_eta_levels > 1 else 'eta_bin'
            preds_i = outputs[logit_key].argmax(dim=-1)
            self.val_eta_level_accs[i](preds_i, classification_targets[target_key])
        
        for i in range(self.num_phi_levels):
            logit_key = f'phi_logits_L{i}' if self.num_phi_levels > 1 else 'phi_logits'
            target_key = f'phi_bin_L{i}' if self.num_phi_levels > 1 else 'phi_bin'
            preds_i = outputs[logit_key].argmax(dim=-1)
            self.val_phi_level_accs[i](preds_i, classification_targets[target_key])
        
        for i in range(self.num_pt_levels):
            logit_key = f'{pt_prefix}_logits_L{i}' if self.num_pt_levels > 1 else f'{pt_prefix}_logits'
            target_key = f'pt_bin_L{i}' if self.num_pt_levels > 1 else 'pt_bin'
            preds_i = outputs[logit_key].argmax(dim=-1)
            self.val_pt_level_accs[i](preds_i, classification_targets[target_key])
        
        # Update AUC metric for charge
        self.val_charge_auc(charge_probs, classification_targets['charge'].long())
        
        # Convert bin predictions to continuous values for physics metrics
        # For eta/phi: use finest level bin centers
        eta_preds_cont = self._bin_index_to_value(eta_pred_bins, self.eta_bin_centers)
        phi_preds_cont = self._bin_index_to_value(phi_pred_bins, self.phi_bin_centers)
        
        # For pt: if using 1/pt, convert predicted 1/pt bin center back to pt
        pt_preds_cont = self._bin_index_to_value(pt_pred_bins, self.pt_bin_centers)
        if self.use_inv_pt:
            # pt_bin_centers are in 1/pt space -> invert to get pt
            pt_preds_cont = 1.0 / pt_preds_cont.clamp(min=1e-6)
        
        # Accumulate for epoch-level physics metrics
        self._val_eta_preds.append(eta_preds_cont.detach().cpu())
        self._val_eta_targets.append(targets['eta'].cpu())
        self._val_phi_preds.append(phi_preds_cont.detach().cpu())
        self._val_phi_targets.append(targets['phi'].cpu())
        self._val_pt_preds.append(pt_preds_cont.detach().cpu())
        self._val_pt_targets.append(targets['pt'].cpu())
        
        # Accumulate bin predictions for top-k accuracy (finest level)
        self._val_eta_pred_bins.append(eta_pred_bins.detach().cpu())
        self._val_eta_true_bins.append(classification_targets['eta_bin'].cpu())
        self._val_phi_pred_bins.append(phi_pred_bins.detach().cpu())
        self._val_phi_true_bins.append(classification_targets['phi_bin'].cpu())
        self._val_pt_pred_bins.append(pt_pred_bins.detach().cpu())
        self._val_pt_true_bins.append(classification_targets['pt_bin'].cpu())
        
        # Log losses
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val/eta_loss', losses['eta_loss'], sync_dist=True)
        self.log('val/phi_loss', losses['phi_loss'], sync_dist=True)
        self.log('val/pt_loss', losses['pt_loss'], sync_dist=True)
        self.log('val/charge_loss', losses['charge_loss'], sync_dist=True)
        
        # Log per-level losses
        for i in range(self.num_eta_levels):
            if f'eta_loss_L{i}' in losses:
                self.log(f'val/eta_loss_L{i}', losses[f'eta_loss_L{i}'], sync_dist=True)
        for i in range(self.num_phi_levels):
            if f'phi_loss_L{i}' in losses:
                self.log(f'val/phi_loss_L{i}', losses[f'phi_loss_L{i}'], sync_dist=True)
        for i in range(self.num_pt_levels):
            if f'pt_loss_L{i}' in losses:
                self.log(f'val/pt_loss_L{i}', losses[f'pt_loss_L{i}'], sync_dist=True)
        
        # Log accuracies and AUC
        self.log('val/eta_acc', self.val_eta_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/phi_acc', self.val_phi_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/pt_acc', self.val_pt_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/charge_acc', self.val_charge_acc, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/charge_auc', self.val_charge_auc, on_step=False, on_epoch=True, sync_dist=True)
        
        # Log per-level accuracies
        for i in range(self.num_eta_levels):
            self.log(f'val/eta_acc_L{i}', self.val_eta_level_accs[i], on_step=False, on_epoch=True, sync_dist=True)
        for i in range(self.num_phi_levels):
            self.log(f'val/phi_acc_L{i}', self.val_phi_level_accs[i], on_step=False, on_epoch=True, sync_dist=True)
        for i in range(self.num_pt_levels):
            self.log(f'val/pt_acc_L{i}', self.val_pt_level_accs[i], on_step=False, on_epoch=True, sync_dist=True)
        
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
        
        # Compute top-k physics-informed accuracy (adjacent bins)
        # For eta and pt: bins are adjacent if |pred - true| <= k
        # For phi: bins wrap around (bin 0 adjacent to bin N-1)
        eta_pred_bins = torch.cat(self._val_eta_pred_bins)
        eta_true_bins = torch.cat(self._val_eta_true_bins)
        phi_pred_bins = torch.cat(self._val_phi_pred_bins)
        phi_true_bins = torch.cat(self._val_phi_true_bins)
        pt_pred_bins = torch.cat(self._val_pt_pred_bins)
        pt_true_bins = torch.cat(self._val_pt_true_bins)
        
        for k in [5, 10]:
            # Eta: simple adjacent (no wrapping)
            eta_diff = (eta_pred_bins - eta_true_bins).abs()
            eta_topk_acc = (eta_diff <= k).float().mean()
            
            # Phi: circular adjacent (wrap around)
            phi_diff = (phi_pred_bins - phi_true_bins).abs()
            # Also check wrapped distance: N - diff
            phi_wrapped_diff = self.phi_bins - phi_diff
            phi_min_diff = torch.minimum(phi_diff, phi_wrapped_diff)
            phi_topk_acc = (phi_min_diff <= k).float().mean()
            
            # Pt: simple adjacent (no wrapping)
            pt_diff = (pt_pred_bins - pt_true_bins).abs()
            pt_topk_acc = (pt_diff <= k).float().mean()
            
            self.log(f'val/eta_top{k}_acc', eta_topk_acc, sync_dist=True)
            self.log(f'val/phi_top{k}_acc', phi_topk_acc, sync_dist=True)
            self.log(f'val/pt_top{k}_acc', pt_topk_acc, sync_dist=True)
        
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
