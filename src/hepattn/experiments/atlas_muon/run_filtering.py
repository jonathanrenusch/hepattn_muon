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

    def log_custom_metrics(self, preds, targets, stage):
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
        # print("num of predicted particles:", pred.sum())
        # print("target keys:", targets.keys())

        # # Get the raw probabilities for AUC calculation
        # pred_probs = preds["final"]["hit_filter"]["hit_on_valid_particle_logits"]
        
        tp = (pred * true).sum()
        tn = ((~pred) * (~true)).sum()

        # # Calculate AUC
        # auc = auroc(pred_probs.flatten(), true.flatten().long(), task="binary")
        # print("pred.shape", pred.shape)
        # print("true.shape", true.shape)

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
            # "auc": auc,
        }
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