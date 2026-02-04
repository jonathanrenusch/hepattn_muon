"""Single-task loss and metrics for isolated track parameter training.

Each task class handles loss computation and metrics for ONE parameter only.
This eliminates gradient interference between tasks.
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
    """Compute angular difference handling periodicity."""
    diff = torch.abs(phi_pred - phi_true)
    return torch.minimum(diff, 2 * math.pi - diff)


class EtaRegressionTask(nn.Module):
    """Task for eta regression only.
    
    Loss: Smooth L1 on delta_eta (predicted - reference)
    """
    
    def __init__(self):
        super().__init__()
    
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute eta loss."""
        delta_eta_pred = outputs['delta_eta']
        ref_eta = outputs['ref_eta']
        eta_target = targets['eta']
        
        # Target is delta from reference
        delta_eta_target = eta_target - ref_eta
        
        loss_eta = F.smooth_l1_loss(delta_eta_pred, delta_eta_target)
        
        return {
            'loss': loss_eta,
            'loss_eta': loss_eta,
        }
    
    def metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute eta metrics on recovered values."""
        delta_eta_pred = outputs['delta_eta']
        ref_eta = outputs['ref_eta']
        eta_target = targets['eta']
        
        # Recover absolute eta
        eta_pred = ref_eta + delta_eta_pred
        
        # Residuals
        eta_residuals = eta_pred - eta_target
        eta_abs_residuals = eta_residuals.abs()
        
        metrics = {
            'mae_eta': eta_abs_residuals.mean(),
            'std_eta': eta_residuals.std(),
            'std_mae_eta': eta_abs_residuals.std(),
        }
        
        # Relative resolution (where |eta| > 0.1)
        eta_mask = eta_target.abs() > 0.1
        if eta_mask.sum() > 0:
            rel_res = eta_abs_residuals[eta_mask] / eta_target[eta_mask].abs()
            metrics['rel_res_eta'] = rel_res.mean()
            metrics['rel_res_std_eta'] = rel_res.std()
        else:
            metrics['rel_res_eta'] = torch.tensor(0.0, device=eta_pred.device)
            metrics['rel_res_std_eta'] = torch.tensor(0.0, device=eta_pred.device)
        
        return metrics


class PhiRegressionTask(nn.Module):
    """Task for phi regression only.
    
    Loss: Smooth L1 on normalized (sin_phi, cos_phi)
    """
    
    def __init__(self):
        super().__init__()
    
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute phi loss."""
        sin_phi_pred = outputs['sin_phi']
        cos_phi_pred = outputs['cos_phi']
        phi_target = targets['phi']
        
        # Normalize predictions to unit circle
        pred_magnitude = torch.sqrt(sin_phi_pred**2 + cos_phi_pred**2 + 1e-8)
        sin_phi_norm = sin_phi_pred / pred_magnitude
        cos_phi_norm = cos_phi_pred / pred_magnitude
        
        # True sin/cos from target
        sin_phi_target = torch.sin(phi_target)
        cos_phi_target = torch.cos(phi_target)
        
        # Smooth L1 on normalized predictions
        loss_sin = F.smooth_l1_loss(sin_phi_norm, sin_phi_target)
        loss_cos = F.smooth_l1_loss(cos_phi_norm, cos_phi_target)
        loss_phi = loss_sin + loss_cos
        
        return {
            'loss': loss_phi,
            'loss_phi': loss_phi,
        }
    
    def metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute phi metrics on recovered values."""
        sin_phi_pred = outputs['sin_phi']
        cos_phi_pred = outputs['cos_phi']
        phi_target = targets['phi']
        
        # Recover phi from sin/cos
        phi_pred = torch.atan2(sin_phi_pred, cos_phi_pred)
        
        # Angular residuals (handle periodicity)
        phi_residuals = angular_difference(phi_pred, phi_target)
        
        return {
            'mae_phi': phi_residuals.mean(),
            'std_phi': phi_residuals.std(),
            'std_mae_phi': phi_residuals.std(),  # Same as std for angular diff
        }


class PtRegressionTask(nn.Module):
    """Task for pT regression only.
    
    Loss: Smooth L1 on log(pT)
    """
    
    def __init__(self):
        super().__init__()
    
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute pT loss."""
        log_pt_pred = outputs['log_pt']
        pt_target = targets['pt']
        
        log_pt_target = torch.log(pt_target + 1e-8)
        loss_pt = F.smooth_l1_loss(log_pt_pred, log_pt_target)
        
        return {
            'loss': loss_pt,
            'loss_pt': loss_pt,
        }
    
    def metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute pT metrics on recovered values."""
        log_pt_pred = outputs['log_pt']
        pt_target = targets['pt']
        
        # Recover pT
        pt_pred = torch.exp(log_pt_pred)
        
        # Residuals
        pt_residuals = pt_pred - pt_target
        pt_abs_residuals = pt_residuals.abs()
        
        # Relative resolution: sigma(pT)/pT
        rel_res = pt_abs_residuals / (pt_target.abs() + 1e-8)
        
        return {
            'mae_pt': pt_abs_residuals.mean(),
            'std_pt': pt_residuals.std(),
            'std_mae_pt': pt_abs_residuals.std(),
            'rel_res_pt': rel_res.mean(),
            'rel_res_std_pt': rel_res.std(),
        }


class ChargeClassificationTask(nn.Module):
    """Task for charge classification only.
    
    Loss: Binary cross-entropy
    """
    
    def __init__(self):
        super().__init__()
    
    def loss(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute charge loss."""
        charge_logit = outputs['charge_logit']
        charge_target = targets['charge']  # 0/1 format
        
        loss_charge = F.binary_cross_entropy_with_logits(
            charge_logit,
            charge_target.float()
        )
        
        return {
            'loss': loss_charge,
            'loss_charge': loss_charge,
        }
    
    def metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute charge metrics."""
        charge_logit = outputs['charge_logit']
        charge_target = targets['charge']
        
        charge_prob = torch.sigmoid(charge_logit)
        charge_pred = (charge_prob > 0.5).float()
        
        metrics = {
            'charge_accuracy': (charge_pred == charge_target).float().mean(),
        }
        
        # AUC
        if TORCHMETRICS_AVAILABLE:
            if len(charge_target.unique()) > 1:
                try:
                    metrics['charge_auc'] = auroc(
                        charge_prob,
                        charge_target.long(),
                        task='binary'
                    )
                except Exception:
                    metrics['charge_auc'] = torch.tensor(0.5, device=charge_prob.device)
            else:
                metrics['charge_auc'] = metrics['charge_accuracy']
        
        return metrics
