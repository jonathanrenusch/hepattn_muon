from lightning.pytorch.cli import ArgsType
from torch import nn
import torch
from torchmetrics.functional import auroc

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class AtlasMuonFilter(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "Lion",
    ):
        super().__init__(name, model, lrs_config, optimizer)

    def log_custom_metrics(self, preds, targets, stage, outputs=None):
        batch_size = targets["hit_on_valid_particle"].shape[0]
        # print("This is batch_size,", batch_size)
        pred = preds["final"]["hit_filter"]["hit_on_valid_particle"][targets["hit_valid"]]
        # print("pred shape:", pred.shape)
        # print("target shape:", targets["hit_on_valid_particle"].shape)
        # print("predictions", pred)
        true = targets["hit_on_valid_particle"][targets["hit_valid"]]
        # true = targets["hit_valid"]
        # print("num of true particles:", true.sum())
        # print("num of false particles:", (~true).sum())
        # print("num of predicted predictions:", pred.sum())
        # print("target keys:", targets.keys())

        tp = (pred * true).sum()
        tn = ((~pred) * (~true)).sum()

        # Calculate AUC if we have access to outputs (raw logits)
        auc = None
        if outputs is not None:
            # Get the raw logits from outputs
            pred_logits = outputs["final"]["hit_filter"]["hit_logit"][targets["hit_valid"]]
            # Convert logits to probabilities
            pred_probs = torch.sigmoid(pred_logits)
            # Calculate AUC
            auc = auroc(pred_probs.flatten(), true.flatten().long(), task="binary")

        # Calculate reconstructable particles metric
        # Get the particle-hit mask and filter it to only valid hits
        particle_hit_mask = targets["particle_hit_valid"][0]  # Remove batch dimension: (num_particles, num_hits)
        particle_hit_mask_valid = particle_hit_mask[:, targets["hit_valid"][0]]  # Only consider valid hits
        
        # Count hits per particle before filtering (ground truth)
        hits_per_particle_true = particle_hit_mask_valid.sum(dim=1)  # Sum over hits for each particle
        
        # Count hits per particle after filtering (prediction)
        # pred is already filtered to valid hits, so we need to map it back to the particle-hit mask
        pred_expanded = torch.zeros_like(targets["hit_valid"][0], dtype=torch.bool)
        pred_expanded[targets["hit_valid"][0]] = pred
        particle_hit_mask_pred = particle_hit_mask & pred_expanded.unsqueeze(0)
        hits_per_particle_pred = particle_hit_mask_pred.sum(dim=1)  # Sum over hits for each particle
        
        # Find particles with >3 true hits (reconstructable particles)
        reconstructable_particles = hits_per_particle_true > 3
        
        # Among reconstructable particles, find those that retain >=3 hits after filtering
        reconstructable_and_filtered = reconstructable_particles & (hits_per_particle_pred >= 3)
        
        # Calculate the percentage
        num_reconstructable = reconstructable_particles.sum().float()
        num_reconstructable_retained = reconstructable_and_filtered.sum().float()
        
        reconstructable_retention = num_reconstructable_retained / num_reconstructable if num_reconstructable > 0 else torch.tensor(0.0)

        metrics = {
            # Log quantities based on the number of hits
            "nh_total_pre": float(pred.numel()),
            "nh_total_post": float(pred.sum()),
            "nh_pred_true": pred.float().sum(),
            "nh_pred_false": (~pred).float().sum(),
            "nh_valid_pre": true.float().sum(),
            "nh_valid_post": (pred & true).float().sum(),
            "nh_noise_pre": (~true).float().sum(),
            "nh_noise_post": (pred & ~true).float().sum(),
            # Standard binary classification metrics
            "acc": (pred == true).half().mean(),
            "valid_recall": tp / true.sum(),
            "valid_precision": tp / pred.sum(),
            "noise_recall": tn / (~true).sum(),
            "noise_precision": tn / (~pred).sum(),
            # Reconstructable particles metric
            "reconstructable_particle_retention": reconstructable_retention,
            "num_reconstructable_particles": num_reconstructable,
            "num_reconstructable_retained": num_reconstructable_retained,
        }
        
        # Add AUC if calculated
        if auc is not None:
            metrics["auc"] = auc
        
        # print("Batch size:", pred.shape[0])
        # Now actually log the metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"{stage}/{metric_name}", 
                     metric_value, 
                     sync_dist=True, 
                     batch_size=batch_size, 
                     on_step=False, 
                     on_epoch=True)


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=AtlasMuonFilter,
        datamodule_class=AtlasMuonDataModule,
        args=args,
        parser_kwargs={"default_env": True},
        save_config_callback=None
    )


if __name__ == "__main__":
    main()