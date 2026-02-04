"""Training script for single-task Mamba track parameter models.

Each model trains on ONE parameter only:
- eta: MambaEtaRegressor + EtaRegressionTask
- phi: MambaPhiRegressor + PhiRegressionTask  
- pt: MambaPtRegressor + PtRegressionTask
- charge: MambaChargeClassifier + ChargeClassificationTask

Usage:
    python run_single_task.py fit --config configs/exp_proxy/single_task_eta.yaml
"""

import comet_ml  # noqa: F401
import math
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn, Tensor

from hepattn.experiments.atlas_muon.data_per_track import PerTrackDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.models.mamba_single_task import (
    MambaEtaRegressor,
    MambaPhiRegressor,
    MambaPtRegressor,
    MambaChargeClassifier,
)
from hepattn.models.single_task_losses import (
    EtaRegressionTask,
    PhiRegressionTask,
    PtRegressionTask,
    ChargeClassificationTask,
)
from hepattn.utils.cli import CLI


def angular_difference_abs(phi_pred: Tensor, phi_true: Tensor) -> Tensor:
    """Compute absolute angular difference handling periodicity."""
    diff = torch.abs(phi_pred - phi_true)
    return torch.minimum(diff, 2 * math.pi - diff)


class SingleTaskWrapper(ModelWrapper):
    """Lightning wrapper for single-task Mamba training.
    
    Each wrapper instance trains ONE task only.
    
    Parameters
    ----------
    name : str
        Experiment name.
    model : nn.Module
        One of: MambaEtaRegressor, MambaPhiRegressor, MambaPtRegressor, MambaChargeClassifier
    task : str
        Which task: 'eta', 'phi', 'pt', or 'charge'
    lrs_config : dict
        Learning rate scheduler configuration.
    optimizer : str
        Optimizer type ('AdamW' or 'Lion').
    """
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        task: str,
        lrs_config: dict,
        optimizer: str = "Lion",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        
        self.task_name = task
        
        # Create appropriate task loss/metrics
        if task == 'eta':
            self.task = EtaRegressionTask()
        elif task == 'phi':
            self.task = PhiRegressionTask()
        elif task == 'pt':
            self.task = PtRegressionTask()
        elif task == 'charge':
            self.task = ChargeClassificationTask()
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'eta', 'phi', 'pt', or 'charge'")
        
        # Accumulators for epoch-level metrics (unweighted)
        self._val_residuals = []
        self._val_targets = []
        self._val_preds = []
    
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
        
        # Log all losses
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
            if isinstance(metric_value, Tensor):
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
        self._val_residuals = []
        self._val_targets = []
        self._val_preds = []
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute and log losses (per-batch, for loss tracking)
        loss = self._compute_and_log_losses(outputs, targets, "val")
        
        # Accumulate residuals for epoch-level metrics
        with torch.no_grad():
            if self.task_name == 'eta':
                delta_eta_pred = outputs['delta_eta']
                ref_eta = outputs['ref_eta']
                eta_pred = ref_eta + delta_eta_pred
                eta_target = targets['eta']
                residuals = eta_pred - eta_target
                self._val_residuals.append(residuals.detach().cpu())
                
            elif self.task_name == 'phi':
                sin_phi_pred = outputs['sin_phi']
                cos_phi_pred = outputs['cos_phi']
                phi_pred = torch.atan2(sin_phi_pred, cos_phi_pred)
                phi_target = targets['phi']
                # Use absolute angular difference (matches training metric)
                residuals = angular_difference_abs(phi_pred, phi_target)
                self._val_residuals.append(residuals.detach().cpu())
                
            elif self.task_name == 'pt':
                log_pt_pred = outputs['log_pt']
                pt_pred = torch.exp(log_pt_pred)
                pt_target = targets['pt']
                residuals = pt_pred - pt_target
                self._val_residuals.append(residuals.detach().cpu())
                self._val_targets.append(pt_target.detach().cpu())
                
            elif self.task_name == 'charge':
                charge_logit = outputs['charge_logit']
                charge_prob = torch.sigmoid(charge_logit)
                charge_pred = (charge_prob > 0.5).float()
                charge_target = targets['charge']
                self._val_preds.append(charge_pred.detach().cpu())
                self._val_targets.append(charge_target.detach().cpu())
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-level unweighted metrics."""
        if self.task_name == 'charge':
            if len(self._val_preds) > 0:
                all_preds = torch.cat(self._val_preds, dim=0)
                all_targets = torch.cat(self._val_targets, dim=0)
                accuracy = (all_preds == all_targets).float().mean()
                self.log("val/epoch_charge_accuracy", accuracy, sync_dist=True, prog_bar=True)
        else:
            if len(self._val_residuals) > 0:
                all_residuals = torch.cat(self._val_residuals, dim=0)
                epoch_std = all_residuals.std()
                epoch_mae = all_residuals.abs().mean()
                
                self.log(f"val/epoch_std_{self.task_name}", epoch_std, sync_dist=True, prog_bar=True)
                self.log(f"val/epoch_mae_{self.task_name}", epoch_mae, sync_dist=True)
                
                # For pt, also log relative resolution
                if self.task_name == 'pt' and len(self._val_targets) > 0:
                    all_targets = torch.cat(self._val_targets, dim=0)
                    rel_res = all_residuals.abs() / (all_targets.abs() + 1e-8)
                    self.log("val/epoch_rel_res_pt", rel_res.mean(), sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Get predictions
        preds = self.predict(outputs)
        
        # Compute losses
        losses = self.task.loss(outputs, targets)
        
        # Wrap for PredictionWriter compatibility
        wrapped_outputs = {'final': {f'mamba_{self.task_name}': outputs}}
        wrapped_preds = {'final': {f'mamba_{self.task_name}': preds}}
        wrapped_losses = {'final': {f'mamba_{self.task_name}': losses}}
        
        return wrapped_outputs, wrapped_preds, wrapped_losses


def cli_main(args: ArgsType = None):
    """Main CLI entry point."""
    CLI(
        model_class=SingleTaskWrapper,
        datamodule_class=PerTrackDataModule,
        args=args,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
