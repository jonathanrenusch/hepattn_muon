#!/usr/bin/env python3
"""
Hyperparameter optimization script for ATLAS muon tracking using Optuna.

This script allows running multiple optimization trials across different GPUs,
all connected to a shared SQLite database. Each run can be started independently
by specifying a GPU ID.

Usage:
    python optuna_tune.py --gpu 0 --n-trials 100 --study-name atlas_muon_study
    python optuna_tune.py --gpu 1 --n-trials 100 --study-name atlas_muon_study
    ...
"""

import argparse
import copy
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Any
import yaml

import optuna
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class OptimizedWrapperModule(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)
        # Store raw losses for objective function
        self.validation_raw_losses = []

    def log_custom_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        # First log metrics that depend on outputs from multiple tasks
        pred_valid = preds["track_valid"]["track_valid"]
        true_valid = targets["particle_valid"]
        
        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

        # Calculate the true/false positive rates between the predicted and true masks
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
        hit_p = pred_hit_masks.sum(-1)
        hit_t = true_hit_masks.sum(-1)

        # Calculate the efficiency and purity at different matching working points
        both_valid = true_valid & pred_valid
        for wp in [0.25, 0.5, 0.75, 1.0]:
            effs = (hit_tp / hit_t >= wp) & both_valid
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        # Calculate track-level efficiency and fake rate for track_valid prediction
        track_tp = (pred_valid & true_valid).float()
        track_fp = (pred_valid & ~true_valid).float()
        track_fn = (~pred_valid & true_valid).float()
        track_tn = (~pred_valid & ~true_valid).float()
        
        # Calculate per-batch efficiency and fake rate, then average
        batch_track_effs = []
        batch_track_fake_rates = []
        
        for batch_idx in range(true_valid.shape[0]):
            tp_batch = track_tp[batch_idx].sum()
            fp_batch = track_fp[batch_idx].sum()
            fn_batch = track_fn[batch_idx].sum()
            tn_batch = track_tn[batch_idx].sum()
            
            # Track-level efficiency: TP / (TP + FN)
            if (tp_batch + fn_batch) > 0:
                track_eff = tp_batch / (tp_batch + fn_batch)
                batch_track_effs.append(track_eff)
            
            # Track-level fake rate: FP / (FP + TN)
            if (fp_batch + tn_batch) > 0:
                track_fake_rate = fp_batch / (fp_batch + tn_batch)
                if fp_batch.sum() >= 3:
                    batch_track_fake_rates.append(track_fake_rate)
        
        # Log averaged metrics
        if batch_track_effs:
            mean_track_eff = torch.stack(batch_track_effs).mean()
            self.log(f"{stage}/track_efficiency", mean_track_eff, sync_dist=True)
        
        if batch_track_fake_rates:
            mean_track_fake_rate = torch.stack(batch_track_fake_rates).mean()
            self.log(f"{stage}/track_fake_rate", mean_track_fake_rate, sync_dist=True)

        true_num = true_valid.sum(-1)
        pred_num = pred_valid.sum(-1)
        
        # Hit-level metrics
        true_pos_hits = (true_hit_masks & pred_hit_masks).sum() / torch.sum(true_hit_masks)
        false_pos_hits = (pred_hit_masks.sum() - (true_hit_masks & pred_hit_masks).sum()) / torch.sum(~true_hit_masks)

        self.log(f"{stage}/true_pos_hits", true_pos_hits, sync_dist=True)
        self.log(f"{stage}/false_pos_hits", false_pos_hits, sync_dist=True)

        nh_per_true = true_hit_masks.sum(-1).float()[true_valid].mean()
        nh_per_pred = pred_hit_masks.sum(-1).float()[pred_valid].mean()

        self.log(f"{stage}/nh_per_particle", torch.mean(nh_per_true.float()), sync_dist=True)
        self.log(f"{stage}/nh_per_track", torch.mean(nh_per_pred.float()), sync_dist=True)

        self.log(f"{stage}/num_tracks", torch.mean(pred_num.float()), sync_dist=True)
        self.log(f"{stage}/num_particles", torch.mean(true_num.float()), sync_dist=True)

    def compute_raw_losses(self, outputs, targets):
        """Compute raw losses from all tasks without weighting."""
        raw_losses = {}
        
        # Get losses from the model
        losses, _ = self.model.loss(outputs, targets)
        
        # Extract raw losses from each task and layer
        for layer_name, layer_losses in losses.items():
            for task_name, task_losses in layer_losses.items():
                for loss_name, loss_value in task_losses.items():
                    # Store loss without the weight multiplier
                    # Get the task to access its loss_weight
                    task = next(task for task in self.model.tasks if task.name == task_name)
                    if hasattr(task, 'loss_weight'):
                        # Remove the weight by dividing
                        raw_loss = loss_value / task.loss_weight if task.loss_weight != 0 else loss_value
                    else:
                        raw_loss = loss_value
                    
                    raw_losses[f"{layer_name}_{task_name}_{loss_name}"] = raw_loss
        
        return raw_losses

    def validation_step(self, batch):
        inputs, targets = batch

        # Get the raw model outputs
        outputs = self.model(inputs)

        # Compute and log losses
        losses, targets = self.model.loss(outputs, targets)
        total_loss = self.log_losses(losses, "val")

        # Compute raw losses for objective function
        raw_losses = self.compute_raw_losses(outputs, targets)
        
        # Store raw losses for later use in objective function
        self.validation_raw_losses.append(raw_losses)

        # Get the predictions from the model
        preds = self.model.predict(outputs)
        self.log_metrics(preds, targets, "val")

        return {"loss": total_loss, **outputs}

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        super().on_validation_epoch_end()
        # Clear the raw losses for next epoch
        self.validation_raw_losses = []


def suggest_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters for the trial."""
    
    # Model architecture hyperparameters
    num_encoder_layers = trial.suggest_categorical("num_encoder_layers", [1, 2, 3, 4])
    num_decoder_layers = trial.suggest_categorical("num_decoder_layers", [1, 2, 3, 4])
    dim = trial.suggest_categorical("dim", [16, 32, 64])
    
    # Task hyperparameters
    # Cost weights for all tasks
    track_valid_cost_weight = trial.suggest_categorical("track_valid_cost_weight", [0.1, 1.0, 10.0])
    track_hit_valid_cost_weight = trial.suggest_categorical("track_hit_valid_cost_weight", [0.1, 1.0, 10.0])
    parameter_regression_cost_weight = trial.suggest_categorical("parameter_regression_cost_weight", [0.1, 1.0, 10.0])
    charge_classification_cost_weight = trial.suggest_categorical("charge_classification_cost_weight", [0.1, 1.0, 10.0])
    
    # Loss weights for all tasks
    track_valid_loss_weight = trial.suggest_categorical("track_valid_loss_weight", [0.1, 1.0, 10.0])
    track_hit_valid_loss_weight = trial.suggest_categorical("track_hit_valid_loss_weight", [0.1, 1.0, 10.0])
    parameter_regression_loss_weight = trial.suggest_categorical("parameter_regression_loss_weight", [0.1, 1.0, 10.0])
    charge_classification_loss_weight = trial.suggest_categorical("charge_classification_loss_weight", [0.1, 1.0, 10.0])
    
    # Dense network hidden layer dimensions for all tasks
    track_valid_hidden_dim = trial.suggest_categorical("track_valid_hidden_dim", [64, 128, 256, 512])
    track_hit_valid_hidden_dim = trial.suggest_categorical("track_hit_valid_hidden_dim", [64, 128, 256, 512])
    parameter_regression_hidden_dim = trial.suggest_categorical("parameter_regression_hidden_dim", [64, 128, 256, 512])
    charge_classification_hidden_dim = trial.suggest_categorical("charge_classification_hidden_dim", [64, 128, 256, 512])
    input_net_hidden_dim = trial.suggest_categorical("input_net_hidden_dim", [64, 128, 256, 512])
    
    return {
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers,
        "dim": dim,
        "track_valid_cost_weight": track_valid_cost_weight,
        "track_hit_valid_cost_weight": track_hit_valid_cost_weight,
        "parameter_regression_cost_weight": parameter_regression_cost_weight,
        "charge_classification_cost_weight": charge_classification_cost_weight,
        "track_valid_loss_weight": track_valid_loss_weight,
        "track_hit_valid_loss_weight": track_hit_valid_loss_weight,
        "parameter_regression_loss_weight": parameter_regression_loss_weight,
        "charge_classification_loss_weight": charge_classification_loss_weight,
        "track_valid_hidden_dim": track_valid_hidden_dim,
        "track_hit_valid_hidden_dim": track_hit_valid_hidden_dim,
        "parameter_regression_hidden_dim": parameter_regression_hidden_dim,
        "charge_classification_hidden_dim": charge_classification_hidden_dim,
        "input_net_hidden_dim": input_net_hidden_dim,
    }


def create_config_from_trial(base_config: Dict[str, Any], trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """Create a config with hyperparameters from the trial."""
    config = copy.deepcopy(base_config)
    
    # Update model architecture
    config["model"]["model"]["init_args"]["dim"] = trial_params["dim"]
    config["model"]["model"]["init_args"]["encoder"]["init_args"]["num_layers"] = trial_params["num_encoder_layers"]
    config["model"]["model"]["init_args"]["encoder"]["init_args"]["dim"] = trial_params["dim"]
    config["model"]["model"]["init_args"]["decoder"]["num_decoder_layers"] = trial_params["num_decoder_layers"]
    config["model"]["model"]["init_args"]["decoder"]["decoder_layer_config"]["dim"] = trial_params["dim"]
    config["model"]["model"]["init_args"]["decoder"]["num_queries"] = 2  # Keep fixed
    
    # Update input net
    config["model"]["model"]["init_args"]["input_nets"]["init_args"]["modules"][0]["init_args"]["net"]["init_args"]["output_size"] = trial_params["dim"]
    config["model"]["model"]["init_args"]["input_nets"]["init_args"]["modules"][0]["init_args"]["net"]["init_args"]["hidden_layers"] = [trial_params["input_net_hidden_dim"]]
    config["model"]["model"]["init_args"]["input_nets"]["init_args"]["modules"][0]["init_args"]["posenc"]["init_args"]["dim"] = trial_params["dim"]
    
    # Update task configurations
    tasks = config["model"]["model"]["init_args"]["tasks"]["init_args"]["modules"]
    
    # Track valid task
    tasks[0]["init_args"]["dim"] = trial_params["dim"]
    tasks[0]["init_args"]["losses"] = {"object_bce": trial_params["track_valid_loss_weight"]}
    tasks[0]["init_args"]["costs"] = {"object_bce": trial_params["track_valid_cost_weight"]}
    tasks[0]["init_args"]["dense_kwargs"]["hidden_layers"] = [trial_params["track_valid_hidden_dim"]]
    
    # Track hit valid task
    tasks[1]["init_args"]["dim"] = trial_params["dim"]
    tasks[1]["init_args"]["losses"] = {"mask_bce": trial_params["track_hit_valid_loss_weight"]}
    tasks[1]["init_args"]["costs"] = {"mask_bce": trial_params["track_hit_valid_cost_weight"]}
    tasks[1]["init_args"]["dense_kwargs"]["hidden_layers"] = [trial_params["track_hit_valid_hidden_dim"]]
    
    # Parameter regression task
    tasks[2]["init_args"]["loss_weight"] = trial_params["parameter_regression_loss_weight"]
    tasks[2]["init_args"]["cost_weight"] = trial_params["parameter_regression_cost_weight"]
    tasks[2]["init_args"]["dim"] = trial_params["dim"]
    tasks[2]["init_args"]["dense_kwargs"]["hidden_layers"] = [trial_params["parameter_regression_hidden_dim"]]
    
    # Charge classification task
    tasks[3]["init_args"]["loss_weight"] = trial_params["charge_classification_loss_weight"]
    tasks[3]["init_args"]["cost_weight"] = trial_params["charge_classification_cost_weight"]
    tasks[3]["init_args"]["dim"] = trial_params["dim"]
    tasks[3]["init_args"]["dense_kwargs"]["hidden_layers"] = [trial_params["charge_classification_hidden_dim"]]
    
    return config


def objective(trial: optuna.Trial, base_config: Dict[str, Any], gpu_id: int) -> float:
    """Objective function for Optuna optimization."""
    
    # Get hyperparameters for this trial
    trial_params = suggest_hyperparameters(trial)
    
    # Create config with trial hyperparameters
    config = create_config_from_trial(base_config, trial_params)
    
    # Set GPU device
    config["trainer"]["devices"] = [gpu_id]
    
    # Update experiment name to include trial number
    trial_name = f"optuna_trial_{trial.number}_gpu_{gpu_id}"
    config["name"] = trial_name
    
    # Set up logger with trial-specific name
    if "logger" in config["trainer"]:
        config["trainer"]["logger"]["init_args"]["experiment_name"] = trial_name
    
    # Add learning rate configuration if not present
    if "lrs_config" not in config["model"]:
        config["model"]["lrs_config"] = {
            "initial": 1e-5,
            "max": 5e-5,
            "end": 1e-5,
            "pct_start": 0.05,
            "skip_scheduler": False,
            "weight_decay": 1e-5
        }
    
    # Save config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        config_path = f.name
    
    try:
        # Create CLI with the temporary config
        cli = CLI(
            model_class=OptimizedWrapperModule,
            datamodule_class=AtlasMuonDataModule,
            args=["--config", config_path],
            parser_kwargs={"default_env": True},
            save_config_kwargs={"overwrite": True},
        )
        
        # Run training
        cli.trainer.fit(cli.model, cli.datamodule)
        
        # Get the best validation loss
        best_val_loss = cli.trainer.callback_metrics.get("val/loss", float('inf'))
        
        # Also compute the sum of raw losses from the final validation epoch
        # This provides an unweighted objective for optimization
        if hasattr(cli.model, 'validation_raw_losses') and cli.model.validation_raw_losses:
            # Average the raw losses across all validation batches
            raw_loss_sums = {}
            for batch_losses in cli.model.validation_raw_losses:
                for loss_name, loss_value in batch_losses.items():
                    if loss_name not in raw_loss_sums:
                        raw_loss_sums[loss_name] = []
                    raw_loss_sums[loss_name].append(loss_value.item())
            
            # Sum all the averaged raw losses
            total_raw_loss = sum(
                sum(losses) / len(losses) for losses in raw_loss_sums.values()
            )
            
            # Use raw loss sum as objective (lower is better)
            objective_value = total_raw_loss
        else:
            # Fallback to validation loss if raw losses are not available
            objective_value = float(best_val_loss)
        
        # Report intermediate values for pruning
        if hasattr(cli.trainer, 'current_epoch'):
            trial.report(objective_value, cli.trainer.current_epoch)
        
        return objective_value
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        # Return a high value for failed trials
        return float('inf')
        
    finally:
        # Clean up temporary config file
        if os.path.exists(config_path):
            os.unlink(config_path)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for ATLAS muon tracking")
    parser.add_argument("--gpu", type=int, required=True, help="GPU ID to use for this optimization run")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--study-name", type=str, default="atlas_muon_optuna_study", help="Name of the Optuna study")
    parser.add_argument("--db-path", type=str, default="./optuna_study.db", help="Path to SQLite database")
    parser.add_argument("--base-config", type=str, 
                        default="/shared/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/smallCuts/optuna_base_config.yaml",
                        help="Path to base configuration file")
    parser.add_argument("--n-warmup", type=int, default=50, help="Number of warmup trials for TPE sampler")
    parser.add_argument("--pruner", action="store_true", help="Enable pruning of unpromising trials")
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.base_config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create SQLite storage URL
    storage_url = f"sqlite:///{args.db_path}"
    
    # Create or load study
    if args.pruner:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    else:
        pruner = optuna.pruners.NopPruner()
    
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_warmup)
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",  # We want to minimize the loss
        sampler=sampler,
        pruner=pruner
    )
    
    print(f"Starting optimization on GPU {args.gpu}")
    print(f"Study name: {args.study_name}")
    print(f"Database: {args.db_path}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Number of warmup trials: {args.n_warmup}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, args.gpu),
        n_trials=args.n_trials,
        catch=(Exception,)  # Continue even if some trials fail
    )
    
    print(f"Optimization completed on GPU {args.gpu}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best parameters: {study.best_trial.params}")


if __name__ == "__main__":
    main()