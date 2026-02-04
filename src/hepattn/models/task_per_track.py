"""Task classes for Mamba-based per-track regression.

This module provides the loss computation and metrics logging for
the Mamba track parameter regression model.

Physics-informed design (standard mode):
- eta: Direct regression (centered around 0, range ~[-3, 3])
- phi: Regress sin(φ) and cos(φ), normalize to unit circle for loss
- pt: Regress log(pt) to handle long-tail distribution (5-200 GeV)
- charge: Binary classification (-1/+1)

Delta prediction mode:
- eta: Predict delta from innermost hit reference
- phi: Predict delta from innermost hit reference (simpler than sin/cos)
- pt: Still regress log(pt)
- charge: Binary classification (-1/+1)
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import math

try:
    from torchmetrics.functional import auroc
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False


def angular_difference(phi_pred: Tensor, phi_true: Tensor) -> Tensor:
    """Compute angular difference handling periodicity.
    
    Returns the shortest angular distance between two angles,
    properly handling the wrap-around at ±π.
    
    Result is in range [0, π].
    """
    diff = torch.abs(phi_pred - phi_true)
    return torch.minimum(diff, 2 * math.pi - diff)


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angle to [-π, π] range."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class MambaRegressionTask(nn.Module):
    """Task for Mamba-based track parameter regression and charge classification.
    
    Physics-informed loss computation:
    - eta: Smooth L1 loss (direct or delta regression)
    - phi: Smooth L1 on normalized (sin, cos) predictions vs true sin/cos, or delta
    - pt: Smooth L1 on log(pt) scale
    - charge: BCE loss
    
    Supports three loss weighting modes:
    1. Fixed weights (default): Manual per-target weights via loss_weight_* params
    2. Inverse magnitude scaling: Auto-scale based on typical loss magnitudes
    3. Learned weighting: Uncertainty-based learned weights (Kendall et al. 2018)
    
    Single-task mode:
    - When single_task is set ('eta', 'phi', 'pt', or 'charge'), only compute loss
      for that task. This eliminates gradient interference between tasks.
    
    Metrics are computed on recovered physics values:
    - MAE and std for eta, phi (with periodic handling), pt
    - Relative resolution (|pred - true| / |true|) for eta and pt
    
    Parameters
    ----------
    regression_fields : list[str]
        Names of regression target fields (user-facing: eta, phi, pt).
    regression_weight : float
        Weight for regression loss in composite loss.
    classification_weight : float
        Weight for classification loss in composite loss.
    loss_weight_eta : float
        Per-target weight for eta loss (default: 1.0).
    loss_weight_phi : float
        Per-target weight for phi loss (default: 1.0).
    loss_weight_pt : float
        Per-target weight for pT loss (default: 1.0).
    use_learned_weights : bool
        If True, use uncertainty-based learned loss weighting (ignores manual weights).
    use_inverse_scaling : bool
        If True, use inverse magnitude scaling (requires loss_scale_* params).
    loss_scale_eta : float
        Typical magnitude of eta loss for inverse scaling.
    loss_scale_phi : float
        Typical magnitude of phi loss for inverse scaling.
    loss_scale_pt : float
        Typical magnitude of pT loss for inverse scaling.
    single_task : str or None
        If set to 'eta', 'phi', 'pt', or 'charge', only train that single task.
        All other tasks will have zero loss contribution.
        Default: None (train all tasks).
    """
    
    def __init__(
        self,
        regression_fields: list[str] | None = None,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        target_stds: dict[str, float] | None = None,  # Kept for backward compatibility, ignored
        # Per-target loss weights (Option 2: fixed weights)
        loss_weight_eta: float = 1.0,
        loss_weight_phi: float = 1.0,
        loss_weight_pt: float = 1.0,
        # Option 3: Learned weighting (Kendall et al. 2018)
        use_learned_weights: bool = False,
        # Option 2 alternative: Inverse magnitude scaling
        use_inverse_scaling: bool = False,
        loss_scale_eta: float = 0.0001,  # Typical eta loss magnitude
        loss_scale_phi: float = 0.003,   # Typical phi loss magnitude
        loss_scale_pt: float = 0.03,     # Typical pT loss magnitude
        # Single-task mode
        single_task: str | None = None,
    ):
        super().__init__()
        
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        # Single-task mode
        self.single_task = single_task
        if single_task is not None:
            valid_tasks = ['eta', 'phi', 'pt', 'charge']
            if single_task not in valid_tasks:
                raise ValueError(f"single_task must be one of {valid_tasks}, got '{single_task}'")
        
        # Per-target weights
        self.loss_weight_eta = loss_weight_eta
        self.loss_weight_phi = loss_weight_phi
        self.loss_weight_pt = loss_weight_pt
        
        # Learned weighting mode
        self.use_learned_weights = use_learned_weights
        if use_learned_weights:
            # Log-variance parameters (Kendall et al. 2018)
            # Initialize to 0 -> initial precision = 1
            self.log_var_eta = nn.Parameter(torch.zeros(1))
            self.log_var_phi = nn.Parameter(torch.zeros(1))
            self.log_var_pt = nn.Parameter(torch.zeros(1))
        
        # Inverse magnitude scaling
        self.use_inverse_scaling = use_inverse_scaling
        if use_inverse_scaling:
            # Compute normalized inverse weights
            total_inv = (1.0/loss_scale_eta + 1.0/loss_scale_phi + 1.0/loss_scale_pt)
            self.inv_weight_eta = (1.0/loss_scale_eta) / total_inv * 3.0
            self.inv_weight_phi = (1.0/loss_scale_phi) / total_inv * 3.0
            self.inv_weight_pt = (1.0/loss_scale_pt) / total_inv * 3.0
    
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute physics-informed losses.
        
        Automatically detects prediction mode by outputs:
        - 'ref_eta' present + 4 regression outputs -> hybrid mode (delta_eta + sin/cos phi)
        - 'ref_eta' present + 3 regression outputs -> full delta mode
        - 'ref_eta' absent -> standard mode (absolute eta + sin/cos phi)
        
        Parameters
        ----------
        outputs : dict
            Model outputs with 'regression' tensor and optionally 'ref_eta', 'ref_phi'.
        targets : dict
            Target values with 'eta', 'phi', 'pt', 'charge' keys.
            Note: charge should be in 0/1 format for BCE.
            
        Returns
        -------
        dict with loss values.
        """
        regression_pred = outputs['regression']
        charge_logit = outputs['charge_logit'].view(-1)  # (B, 1) -> (B,) safely
        
        # Detect prediction mode
        use_delta_eta = 'ref_eta' in outputs
        num_outputs = regression_pred.shape[1]
        
        # Mode detection:
        # - 4 outputs + ref_eta -> hybrid mode (delta_eta + sin/cos phi)
        # - 3 outputs + ref_eta -> full delta mode
        # - 4 outputs, no ref_eta -> standard mode
        use_delta_phi = use_delta_eta and num_outputs == 3
        use_sincos_phi = num_outputs == 4
        
        # Get targets
        eta_target = targets['eta']
        phi_target = targets['phi']
        pt_target = targets['pt']
        charge_target = targets['charge']  # Already in 0/1 format
        
        # === ETA LOSS ===
        if use_delta_eta:
            # Delta eta mode
            delta_eta_pred = regression_pred[:, 0]
            ref_eta = outputs['ref_eta']
            delta_eta_target = eta_target - ref_eta
            loss_eta = F.smooth_l1_loss(delta_eta_pred, delta_eta_target)
        else:
            # Standard absolute eta
            eta_pred = regression_pred[:, 0]
            loss_eta = F.smooth_l1_loss(eta_pred, eta_target)
        
        # === PHI LOSS ===
        if use_delta_phi:
            # Full delta mode: delta_phi is at index 1
            delta_phi_pred = regression_pred[:, 1]
            ref_phi = outputs['ref_phi']
            delta_phi_target = wrap_angle(phi_target - ref_phi)
            loss_phi = F.smooth_l1_loss(delta_phi_pred, delta_phi_target)
            # pT is at index 2
            log_pt_pred = regression_pred[:, 2]
        else:
            # Sin/cos phi mode (standard or hybrid)
            sin_phi_pred = regression_pred[:, 1]
            cos_phi_pred = regression_pred[:, 2]
            
            # Normalize predicted (sin, cos) to unit circle
            pred_magnitude = torch.sqrt(sin_phi_pred**2 + cos_phi_pred**2 + 1e-8)
            sin_phi_norm = sin_phi_pred / pred_magnitude
            cos_phi_norm = cos_phi_pred / pred_magnitude
            
            # True sin/cos from target phi
            sin_phi_target = torch.sin(phi_target)
            cos_phi_target = torch.cos(phi_target)
            
            # Smooth L1 on normalized predictions
            loss_sin_phi = F.smooth_l1_loss(sin_phi_norm, sin_phi_target)
            loss_cos_phi = F.smooth_l1_loss(cos_phi_norm, cos_phi_target)
            loss_phi = loss_sin_phi + loss_cos_phi
            # pT is at index 3
            log_pt_pred = regression_pred[:, 3]
        
        # === PT LOSS: Smooth L1 on log scale ===
        log_pt_target = torch.log(pt_target + 1e-8)
        loss_pt = F.smooth_l1_loss(log_pt_pred, log_pt_target)
        
        # === CHARGE LOSS: BCE ===
        loss_charge = F.binary_cross_entropy_with_logits(
            charge_logit,
            charge_target.float()
        )
        
        # === SINGLE-TASK MODE: Override loss computation ===
        # Check if single_task is set in outputs (from model) or in self (from config)
        single_task = outputs.get('single_task', self.single_task)
        
        if single_task is not None:
            # Only compute loss for the specified task
            if single_task == 'eta':
                total_loss = loss_eta
                loss_regression = loss_eta
                # Zero out other losses for logging (they're computed but not used)
                loss_phi = loss_phi.detach() * 0 + loss_phi.detach()  # Keep value for logging
                loss_pt = loss_pt.detach() * 0 + loss_pt.detach()
                loss_charge = loss_charge.detach() * 0 + loss_charge.detach()
            elif single_task == 'phi':
                total_loss = loss_phi
                loss_regression = loss_phi
                loss_eta = loss_eta.detach() * 0 + loss_eta.detach()
                loss_pt = loss_pt.detach() * 0 + loss_pt.detach()
                loss_charge = loss_charge.detach() * 0 + loss_charge.detach()
            elif single_task == 'pt':
                total_loss = loss_pt
                loss_regression = loss_pt
                loss_eta = loss_eta.detach() * 0 + loss_eta.detach()
                loss_phi = loss_phi.detach() * 0 + loss_phi.detach()
                loss_charge = loss_charge.detach() * 0 + loss_charge.detach()
            elif single_task == 'charge':
                total_loss = loss_charge
                loss_regression = torch.zeros_like(loss_eta)
                loss_eta = loss_eta.detach() * 0 + loss_eta.detach()
                loss_phi = loss_phi.detach() * 0 + loss_phi.detach()
                loss_pt = loss_pt.detach() * 0 + loss_pt.detach()
            
            # Build return dict for single-task mode
            result = {
                'loss': total_loss,
                'loss_regression': loss_regression,
                'loss_charge': loss_charge if single_task != 'charge' else loss_charge,
                'loss_eta': loss_eta,
                'loss_phi': loss_phi,
                'loss_pt': loss_pt,
                'single_task': single_task,
            }
            return result
        
        # === APPLY LOSS WEIGHTING (multi-task mode) ===
        if self.use_learned_weights:
            # Learned uncertainty weighting (Kendall et al. 2018)
            precision_eta = torch.exp(-self.log_var_eta)
            precision_phi = torch.exp(-self.log_var_phi)
            precision_pt = torch.exp(-self.log_var_pt)
            
            weighted_loss_eta = precision_eta * loss_eta + 0.5 * self.log_var_eta
            weighted_loss_phi = precision_phi * loss_phi + 0.5 * self.log_var_phi
            weighted_loss_pt = precision_pt * loss_pt + 0.5 * self.log_var_pt
            
            loss_regression = weighted_loss_eta + weighted_loss_phi + weighted_loss_pt
        elif self.use_inverse_scaling:
            # Inverse magnitude scaling
            loss_regression = (
                self.inv_weight_eta * loss_eta +
                self.inv_weight_phi * loss_phi +
                self.inv_weight_pt * loss_pt
            )
        else:
            # Fixed per-target weights
            loss_regression = (
                self.loss_weight_eta * loss_eta +
                self.loss_weight_phi * loss_phi +
                self.loss_weight_pt * loss_pt
            )
        
        # Composite loss
        total_loss = (
            self.regression_weight * loss_regression +
            self.classification_weight * loss_charge
        )
        
        # Build return dict
        result = {
            'loss': total_loss,
            'loss_regression': loss_regression,
            'loss_charge': loss_charge,
            'loss_eta': loss_eta,
            'loss_phi': loss_phi,
            'loss_pt': loss_pt,
        }
        
        # Add learned weight info if using uncertainty weighting
        if self.use_learned_weights:
            result['weight_eta'] = precision_eta.squeeze()
            result['weight_phi'] = precision_phi.squeeze()
            result['weight_pt'] = precision_pt.squeeze()
        
        return result
    
    def metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute metrics on recovered physics values.
        
        In all modes, metrics are computed on the final
        recovered values (eta, phi, pt) for easy comparison between runs.
        
        Parameters
        ----------
        outputs : dict
            Model outputs.
        targets : dict
            Target values.
            
        Returns
        -------
        dict with metrics:
            - mae_eta, std_eta, std_mae_eta: mean/std absolute error for eta
            - rel_res_eta, rel_res_std_eta: relative resolution for eta
            - mae_phi, std_phi, std_mae_phi: mean/std absolute error for phi
            - mae_pt, std_pt, std_mae_pt: mean/std absolute error for pt
            - rel_res_pt, rel_res_std_pt: relative resolution (sigma_pt/pt) for pt
            - charge_accuracy: fraction correctly classified
            - charge_auc: area under ROC curve
        """
        regression_pred = outputs['regression']
        charge_logit = outputs['charge_logit'].view(-1)  # (B, 1) -> (B,) safely
        
        # Detect prediction mode
        use_delta_eta = 'ref_eta' in outputs
        num_outputs = regression_pred.shape[1]
        use_delta_phi = use_delta_eta and num_outputs == 3
        
        # === RECOVER ETA ===
        if use_delta_eta:
            delta_eta = regression_pred[:, 0]
            ref_eta = outputs['ref_eta']
            eta_pred = ref_eta + delta_eta
        else:
            eta_pred = regression_pred[:, 0]
        
        # === RECOVER PHI ===
        if use_delta_phi:
            # Full delta mode
            delta_phi = regression_pred[:, 1]
            ref_phi = outputs['ref_phi']
            phi_pred_raw = ref_phi + delta_phi
            phi_pred = wrap_angle(phi_pred_raw)
            log_pt_pred = regression_pred[:, 2]
        else:
            # Sin/cos mode (standard or hybrid)
            sin_phi_pred = regression_pred[:, 1]
            cos_phi_pred = regression_pred[:, 2]
            phi_pred = torch.atan2(sin_phi_pred, cos_phi_pred)
            log_pt_pred = regression_pred[:, 3]
        
        # === RECOVER PT ===
        pt_pred = torch.exp(log_pt_pred)
        
        # Get targets
        eta_target = targets['eta']
        phi_target = targets['phi']
        pt_target = targets['pt']
        charge_target = targets['charge']
        
        metrics = {}
        
        # === ETA METRICS (computed on recovered values) ===
        eta_residuals = eta_pred - eta_target
        eta_abs_residuals = eta_residuals.abs()
        metrics['mae_eta'] = eta_abs_residuals.mean()
        metrics['std_eta'] = eta_residuals.std()
        metrics['std_mae_eta'] = eta_abs_residuals.std()  # Std of absolute errors
        # Relative resolution for eta (where |eta| > 0.1 to avoid division issues)
        eta_mask = eta_target.abs() > 0.1
        if eta_mask.sum() > 0:
            rel_res_eta = (eta_abs_residuals[eta_mask] / eta_target[eta_mask].abs())
            metrics['rel_res_eta'] = rel_res_eta.mean()
            metrics['rel_res_std_eta'] = rel_res_eta.std()
        else:
            # Fallback if all eta values are near zero
            metrics['rel_res_eta'] = torch.tensor(0.0, device=eta_pred.device)
            metrics['rel_res_std_eta'] = torch.tensor(0.0, device=eta_pred.device)
        
        # === PHI METRICS (with periodic handling, computed on recovered values) ===
        phi_residuals = angular_difference(phi_pred, phi_target)
        metrics['mae_phi'] = phi_residuals.mean()  # Already absolute
        metrics['std_phi'] = phi_residuals.std()
        metrics['std_mae_phi'] = phi_residuals.std()  # Same as std for angular diff
        
        # === PT METRICS (computed on recovered values) ===
        pt_residuals = pt_pred - pt_target
        pt_abs_residuals = pt_residuals.abs()
        metrics['mae_pt'] = pt_abs_residuals.mean()
        metrics['std_pt'] = pt_residuals.std()
        metrics['std_mae_pt'] = pt_abs_residuals.std()  # Std of absolute errors
        # Relative resolution for pt: sigma(pT)/pT = |pred - true| / true
        rel_res_pt = pt_abs_residuals / (pt_target.abs() + 1e-8)
        metrics['rel_res_pt'] = rel_res_pt.mean()
        metrics['rel_res_std_pt'] = rel_res_pt.std()
        
        # === CHARGE METRICS ===
        charge_prob = torch.sigmoid(charge_logit)
        charge_pred = (charge_prob > 0.5).float()
        
        metrics['charge_accuracy'] = (charge_pred == charge_target).float().mean()
        
        # Always compute AUC if torchmetrics is available
        if TORCHMETRICS_AVAILABLE:
            # Check if we have both classes present
            if len(charge_target.unique()) > 1:
                try:
                    metrics['charge_auc'] = auroc(
                        charge_prob,
                        charge_target.long(),
                        task='binary'
                    )
                except Exception:
                    # Fallback to 0.5 (random) if computation fails
                    metrics['charge_auc'] = torch.tensor(0.5, device=charge_prob.device)
            else:
                # Single class in batch - AUC undefined, use accuracy as proxy
                metrics['charge_auc'] = metrics['charge_accuracy']
        
        return metrics


class MambaRegressionLoss(nn.Module):
    """Standalone loss module for the Mamba regression task.
    
    This is a simpler version that can be used directly without the full task.
    """
    
    def __init__(
        self,
        regression_weight: float = 1.0,
        classification_weight: float = 0.1,
    ):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
    
    def forward(
        self,
        regression_pred: Tensor,
        regression_target: Tensor,
        charge_logit: Tensor,
        charge_target: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute composite loss.
        
        Parameters
        ----------
        regression_pred : Tensor
            Regression predictions of shape (B, 4): eta, sin_phi, cos_phi, log_pt.
        regression_target : Tensor
            Regression targets of shape (B, 3): eta, phi, pt.
        charge_logit : Tensor
            Charge logits of shape (B,).
        charge_target : Tensor
            Charge targets of shape (B,) in 0/1 format.
            
        Returns
        -------
        total_loss : Tensor
            Scalar loss value.
        loss_dict : dict
            Individual loss components.
        """
        # This is a simplified version - for full physics-informed loss use MambaRegressionTask
        loss_regression = F.smooth_l1_loss(regression_pred, regression_target)
        loss_charge = F.binary_cross_entropy_with_logits(charge_logit, charge_target)
        
        total_loss = (
            self.regression_weight * loss_regression +
            self.classification_weight * loss_charge
        )
        
        return total_loss, {
            'loss_regression': loss_regression,
            'loss_charge': loss_charge,
        }
