"""Training script for Mamba-based track parameter regression.

This script trains a bidirectional Mamba model for track parameter estimation
using ground truth hit-to-track assignments. It serves as a proof-of-concept
for the Mamba architecture before integration with the full tracking pipeline.

Usage:
    python run_mamba_regression.py fit --config configs/mamba_track_regression.yaml
"""

import comet_ml  # noqa: F401
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
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        
        self.regression_fields = regression_fields or ['eta', 'phi', 'pt']
        self.task = MambaRegressionTask(
            regression_fields=self.regression_fields,
            regression_weight=regression_weight,
            classification_weight=classification_weight,
            target_stds=target_stds,
        )
    
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
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward pass
        outputs = self(inputs)
        
        # Compute and log losses
        loss = self._compute_and_log_losses(outputs, targets, "val")
        
        # Always log metrics for validation
        self._compute_and_log_metrics(outputs, targets, "val")
        
        return loss
    
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
