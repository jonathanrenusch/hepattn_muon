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
    - eta: Smooth L1 loss (direct regression)
    - phi: Smooth L1 on normalized (sin, cos) predictions vs true sin/cos
    - pt: Smooth L1 on log(pt) scale
    - charge: BCE loss
    
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
    """
    
    def __init__(
        self,
        regression_fields: list[str] | None = None,
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        target_stds: dict[str, float] | None = None,  # Kept for backward compatibility, ignored
    ):
        super().__init__()
        
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
    
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute physics-informed losses.
        
        Automatically detects delta prediction mode by presence of 'ref_eta' in outputs.
        
        Parameters
        ----------
        outputs : dict
            Model outputs. Standard mode: 'regression' (eta, sin_phi, cos_phi, log_pt).
            Delta mode: 'regression' (delta_eta, delta_phi, log_pt) + 'ref_eta', 'ref_phi'.
        targets : dict
            Target values with 'eta', 'phi', 'pt', 'charge' keys.
            Note: charge should be in 0/1 format for BCE.
            
        Returns
        -------
        dict with loss values.
        """
        regression_pred = outputs['regression']
        charge_logit = outputs['charge_logit']
        
        # Detect delta prediction mode
        use_delta = 'ref_eta' in outputs
        
        # Get targets
        eta_target = targets['eta']
        phi_target = targets['phi']
        pt_target = targets['pt']
        charge_target = targets['charge']  # Already in 0/1 format
        
        if use_delta:
            # === DELTA PREDICTION MODE ===
            # regression is [delta_eta, delta_phi, log_pt]
            delta_eta_pred = regression_pred[:, 0]
            delta_phi_pred = regression_pred[:, 1]
            log_pt_pred = regression_pred[:, 2]
            
            ref_eta = outputs['ref_eta']
            ref_phi = outputs['ref_phi']
            
            # Compute target deltas
            delta_eta_target = eta_target - ref_eta
            
            # For phi, compute the shortest angular delta and wrap to [-π, π]
            # This handles periodicity at the boundaries
            delta_phi_target = wrap_angle(phi_target - ref_phi)
            
            # === ETA LOSS: Smooth L1 on delta ===
            loss_eta = F.smooth_l1_loss(delta_eta_pred, delta_eta_target)
            
            # === PHI LOSS: Smooth L1 on delta (much simpler than sin/cos!) ===
            loss_phi = F.smooth_l1_loss(delta_phi_pred, delta_phi_target)
            
            # === PT LOSS: Smooth L1 on log scale ===
            log_pt_target = torch.log(pt_target + 1e-8)
            loss_pt = F.smooth_l1_loss(log_pt_pred, log_pt_target)
            
        else:
            # === STANDARD MODE ===
            # regression is [eta, sin_phi, cos_phi, log_pt]
            eta_pred = regression_pred[:, 0]
            sin_phi_pred = regression_pred[:, 1]
            cos_phi_pred = regression_pred[:, 2]
            log_pt_pred = regression_pred[:, 3]
            
            # === ETA LOSS: Direct Smooth L1 ===
            loss_eta = F.smooth_l1_loss(eta_pred, eta_target)
            
            # === PHI LOSS: Normalized unit circle Smooth L1 ===
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
            
            # === PT LOSS: Smooth L1 on log scale ===
            log_pt_target = torch.log(pt_target + 1e-8)
            loss_pt = F.smooth_l1_loss(log_pt_pred, log_pt_target)
        
        # === CHARGE LOSS: BCE ===
        loss_charge = F.binary_cross_entropy_with_logits(
            charge_logit,
            charge_target.float()
        )
        
        # Total regression loss (sum of individual losses)
        loss_regression = loss_eta + loss_phi + loss_pt
        
        # Composite loss
        total_loss = (
            self.regression_weight * loss_regression +
            self.classification_weight * loss_charge
        )
        
        return {
            'loss': total_loss,
            'loss_regression': loss_regression,
            'loss_charge': loss_charge,
            'loss_eta': loss_eta,
            'loss_phi': loss_phi,
            'loss_pt': loss_pt,
        }
    
    def metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute metrics on recovered physics values.
        
        In both standard and delta modes, metrics are computed on the final
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
            - mae_eta, mae_phi, mae_pt: mean absolute error
            - std_eta, std_phi, std_pt: standard deviation of residuals
            - rel_res_eta, rel_res_pt: mean relative resolution |pred-true|/|true|
            - rel_res_std_eta, rel_res_std_pt: std of relative resolution
            - charge_accuracy, charge_auc
        """
        regression_pred = outputs['regression']
        charge_logit = outputs['charge_logit']
        
        # Detect delta prediction mode
        use_delta = 'ref_eta' in outputs
        
        if use_delta:
            # Delta mode: recover final values
            delta_eta = regression_pred[:, 0]
            delta_phi = regression_pred[:, 1]
            log_pt_pred = regression_pred[:, 2]
            
            ref_eta = outputs['ref_eta']
            ref_phi = outputs['ref_phi']
            
            # Recover final predictions
            eta_pred = ref_eta + delta_eta
            phi_pred_raw = ref_phi + delta_phi
            phi_pred = wrap_angle(phi_pred_raw)  # Wrap to [-π, π]
            pt_pred = torch.exp(log_pt_pred)
        else:
            # Standard mode
            eta_pred = regression_pred[:, 0]
            sin_phi_pred = regression_pred[:, 1]
            cos_phi_pred = regression_pred[:, 2]
            log_pt_pred = regression_pred[:, 3]
            
            phi_pred = torch.atan2(sin_phi_pred, cos_phi_pred)
            pt_pred = torch.exp(log_pt_pred)
        
        # Get targets
        eta_target = targets['eta']
        phi_target = targets['phi']
        pt_target = targets['pt']
        charge_target = targets['charge']
        
        metrics = {}
        
        # === ETA METRICS (computed on recovered values) ===
        eta_residuals = eta_pred - eta_target
        metrics['mae_eta'] = eta_residuals.abs().mean()
        metrics['std_eta'] = eta_residuals.std()
        # Relative resolution for eta (where |eta| > 0.1 to avoid division issues)
        eta_mask = eta_target.abs() > 0.1
        if eta_mask.sum() > 0:
            rel_res_eta = (eta_residuals[eta_mask].abs() / eta_target[eta_mask].abs())
            metrics['rel_res_eta'] = rel_res_eta.mean()
            metrics['rel_res_std_eta'] = rel_res_eta.std()
        
        # === PHI METRICS (with periodic handling, computed on recovered values) ===
        phi_residuals = angular_difference(phi_pred, phi_target)
        metrics['mae_phi'] = phi_residuals.mean()  # Already absolute
        metrics['std_phi'] = phi_residuals.std()
        
        # === PT METRICS (computed on recovered values) ===
        pt_residuals = pt_pred - pt_target
        metrics['mae_pt'] = pt_residuals.abs().mean()
        metrics['std_pt'] = pt_residuals.std()
        # Relative resolution for pt
        rel_res_pt = pt_residuals.abs() / (pt_target.abs() + 1e-8)
        metrics['rel_res_pt'] = rel_res_pt.mean()
        metrics['rel_res_std_pt'] = rel_res_pt.std()
        
        # === CHARGE METRICS ===
        charge_prob = torch.sigmoid(charge_logit)
        charge_pred = (charge_prob > 0.5).float()
        
        metrics['charge_accuracy'] = (charge_pred == charge_target).float().mean()
        
        if TORCHMETRICS_AVAILABLE and len(charge_target.unique()) > 1:
            try:
                metrics['charge_auc'] = auroc(
                    charge_prob,
                    charge_target.long(),
                    task='binary'
                )
            except Exception:
                pass
        
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
