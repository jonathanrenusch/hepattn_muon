"""Training script for Mamba-based track parameter regression.

This script trains a bidirectional Mamba model for track parameter estimation
using ground truth hit-to-track assignments. It serves as a proof-of-concept
for the Mamba architecture before integration with the full tracking pipeline.

Usage:
    python run_mamba_regression.py fit --config configs/mamba_track_regression.yaml
"""

import comet_ml  # noqa: F401
import math
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn, Tensor

from hepattn.experiments.atlas_muon.data_per_track import PerTrackDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.models.mamba_regressor import MambaTrackRegressor
from hepattn.models.task_per_track import MambaRegressionTask
from hepattn.utils.cli import CLI

try:
    from torchmetrics.functional import auroc
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False


def angular_difference_abs(phi_pred: Tensor, phi_true: Tensor) -> Tensor:
    """Compute absolute angular difference handling periodicity."""
    diff = torch.abs(phi_pred - phi_true)
    return torch.minimum(diff, 2 * math.pi - diff)


class MambaRegressionWrapper(ModelWrapper):
    """Lightning wrapper for Mamba track regression training.
    
    Handles training loop, loss computation, and metrics logging for
    the MambaTrackRegressor model.
    
    Parameters
    ----------
    name : str
        Experiment name.
    model : nn.Module
        The MambaTrackRegressor model.
    lrs_config : dict
        Learning rate scheduler configuration.
    optimizer : str
        Optimizer type ('AdamW' or 'Lion').
    regression_weight : float
        Weight for regression loss.
    classification_weight : float
        Weight for classification loss.
    regression_fields : list[str]
        Names of regression target fields.
    loss_weight_eta : float
        Per-target weight for eta loss.
    loss_weight_phi : float
        Per-target weight for phi loss.
    loss_weight_pt : float
        Per-target weight for pT loss.
    use_learned_weights : bool
        If True, use uncertainty-based learned loss weighting.
    use_inverse_scaling : bool
        If True, use inverse magnitude scaling.
    loss_scale_eta : float
        Typical magnitude of eta loss for inverse scaling.
    loss_scale_phi : float
        Typical magnitude of phi loss for inverse scaling.
    loss_scale_pt : float
        Typical magnitude of pT loss for inverse scaling.
    single_task : str or None
        If set to 'eta', 'phi', 'pt', or 'charge', only train that single task.
        Default: None (train all tasks).
    """
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "Lion",
        regression_weight: float = 1.0,
        classification_weight: float = 1.0,
        regression_fields: list[str] | None = None,
        target_stds: dict[str, float] | None = None,
        mtl: bool = False,
        # Per-target loss weights
        loss_weight_eta: float = 1.0,
        loss_weight_phi: float = 1.0,
        loss_weight_pt: float = 1.0,
        # Learned weighting
        use_learned_weights: bool = False,
        # Inverse magnitude scaling
        use_inverse_scaling: bool = False,
        loss_scale_eta: float = 0.0001,
        loss_scale_phi: float = 0.003,
        loss_scale_pt: float = 0.03,
        # Single-task mode
        single_task: str | None = None,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        self.single_task = single_task
        self.task = MambaRegressionTask(
            regression_fields=self.regression_fields,
            regression_weight=regression_weight,
            classification_weight=classification_weight,
            target_stds=target_stds,
            loss_weight_eta=loss_weight_eta,
            loss_weight_phi=loss_weight_phi,
            loss_weight_pt=loss_weight_pt,
            use_learned_weights=use_learned_weights,
            use_inverse_scaling=use_inverse_scaling,
            loss_scale_eta=loss_scale_eta,
            loss_scale_phi=loss_scale_phi,
            loss_scale_pt=loss_scale_pt,
            single_task=single_task,
        )
        
        # Accumulators for epoch-level metrics (unweighted)
        self._val_eta_residuals = []
        self._val_phi_residuals = []
        self._val_pt_residuals = []
        self._val_pt_targets = []
        self._val_charge_preds = []
        self._val_charge_targets = []
    
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass through the model."""
        return self.model(inputs)
    
    def predict(self, outputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert model outputs to predictions."""
        return self.model.predict(outputs)
    
    def _compute_and_log_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        stage: str,
    ) -> Tensor:
        """Compute losses and log them."""
        losses = self.task.loss(outputs, targets)
        
        # Log all losses (skip non-tensor values like 'single_task' string)
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, Tensor):
                self.log(f"{stage}/{loss_name}", loss_value, sync_dist=True, prog_bar=(loss_name == 'loss'))
        
        return losses['loss']
    
    def _compute_and_log_metrics(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        stage: str,
    ):
        """Compute and log metrics."""
        metrics = self.task.metrics(outputs, targets)
        
        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}/{metric_name}", metric_value, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute and log losses
        loss = self._compute_and_log_losses(outputs, targets, "train")
        
        # Log metrics periodically
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self._compute_and_log_metrics(outputs, targets, "train")
        
        return loss
    
    def on_validation_epoch_start(self):
        """Reset accumulators at start of validation epoch."""
        self._val_eta_residuals = []
        self._val_phi_residuals = []
        self._val_pt_residuals = []
        self._val_pt_targets = []
        self._val_charge_preds = []
        self._val_charge_targets = []
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute and log losses (per-batch)
        loss = self._compute_and_log_losses(outputs, targets, "val")
        
        # Accumulate residuals for epoch-level unweighted metrics
        with torch.no_grad():
            # Eta residuals
            if 'delta_eta' in outputs:
                delta_eta_pred = outputs['delta_eta']
                ref_eta = outputs['ref_eta']
                eta_pred = ref_eta + delta_eta_pred
                eta_target = targets['eta']
                eta_residuals = eta_pred - eta_target
                self._val_eta_residuals.append(eta_residuals.detach().cpu())
            
            # Phi residuals (absolute angular difference)
            if 'sin_phi' in outputs:
                sin_phi_pred = outputs['sin_phi']
                cos_phi_pred = outputs['cos_phi']
                phi_pred = torch.atan2(sin_phi_pred, cos_phi_pred)
                phi_target = targets['phi']
                phi_residuals = angular_difference_abs(phi_pred, phi_target)
                self._val_phi_residuals.append(phi_residuals.detach().cpu())
            
            # Pt residuals
            if 'log_pt' in outputs:
                log_pt_pred = outputs['log_pt']
                pt_pred = torch.exp(log_pt_pred)
                pt_target = targets['pt']
                pt_residuals = pt_pred - pt_target
                self._val_pt_residuals.append(pt_residuals.detach().cpu())
                self._val_pt_targets.append(pt_target.detach().cpu())
            
            # Charge predictions
            if 'charge_logit' in outputs:
                charge_logit = outputs['charge_logit']
                charge_prob = torch.sigmoid(charge_logit)
                charge_pred = (charge_prob > 0.5).float()
                charge_target = targets['charge']
                self._val_charge_preds.append(charge_pred.detach().cpu())
                self._val_charge_targets.append(charge_target.detach().cpu())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-level unweighted metrics."""
        # Eta metrics
        if len(self._val_eta_residuals) > 0:
            all_eta_res = torch.cat(self._val_eta_residuals, dim=0)
            self.log("val/epoch_std_eta", all_eta_res.std(), sync_dist=True, prog_bar=True)
            self.log("val/epoch_mae_eta", all_eta_res.abs().mean(), sync_dist=True)
        
        # Phi metrics
        if len(self._val_phi_residuals) > 0:
            all_phi_res = torch.cat(self._val_phi_residuals, dim=0)
            self.log("val/epoch_std_phi", all_phi_res.std(), sync_dist=True, prog_bar=True)
            self.log("val/epoch_mae_phi", all_phi_res.mean(), sync_dist=True)  # Already absolute
        
        # Pt metrics
        if len(self._val_pt_residuals) > 0:
            all_pt_res = torch.cat(self._val_pt_residuals, dim=0)
            all_pt_targets = torch.cat(self._val_pt_targets, dim=0)
            self.log("val/epoch_std_pt", all_pt_res.std(), sync_dist=True, prog_bar=True)
            self.log("val/epoch_mae_pt", all_pt_res.abs().mean(), sync_dist=True)
            rel_res = all_pt_res.abs() / (all_pt_targets.abs() + 1e-8)
            self.log("val/epoch_rel_res_pt", rel_res.mean(), sync_dist=True)
        
        # Charge metrics
        if len(self._val_charge_preds) > 0:
            all_preds = torch.cat(self._val_charge_preds, dim=0)
            all_targets = torch.cat(self._val_charge_targets, dim=0)
            accuracy = (all_preds == all_targets).float().mean()
            self.log("val/epoch_charge_accuracy", accuracy, sync_dist=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Get predictions
        preds = self.predict(outputs)
        
        # Compute losses
        losses = self.task.loss(outputs, targets)
        
        # Wrap in layer/task structure for PredictionWriter compatibility
        # PredictionWriter expects: {'layer_name': {'task_name': {'field': tensor}}}
        wrapped_outputs = {'final': {'mamba_regression': outputs}}
        wrapped_preds = {'final': {'mamba_regression': preds}}
        wrapped_losses = {'final': {'mamba_regression': losses}}
        
        return wrapped_outputs, wrapped_preds, wrapped_losses
    
    def log_custom_metrics(self, preds, targets, stage, outputs=None):
        """Log additional custom metrics.
        
        This is called by the parent class for compatibility,
        but we handle metrics in _compute_and_log_metrics.
        """
        pass


def cli_main(args: ArgsType = None):
    """Main CLI entry point."""
    CLI(
        model_class=MambaRegressionWrapper,
        datamodule_class=PerTrackDataModule,
        args=args,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
