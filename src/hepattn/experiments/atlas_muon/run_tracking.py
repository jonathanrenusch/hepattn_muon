import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class WrapperModule(ModelWrapper):
    def __init__(
        self,
        name: str,
        model: nn.Module,
        lrs_config: dict,
        optimizer: str = "AdamW",
        mtl: bool = False,
    ):
        super().__init__(name, model, lrs_config, optimizer, mtl)

    def log_custom_metrics(self, preds, targets, stage):
        # Just log predictions from the final layer
        preds = preds["final"]

        # Task names are defined by the model config; read them from the preds dict
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

        batch_track_effs = []
        batch_track_fake_rates = []

        for batch_idx in range(true_valid.shape[0]):
            tp_batch = track_tp[batch_idx].sum()
            fp_batch = track_fp[batch_idx].sum()
            fn_batch = track_fn[batch_idx].sum()
            tn_batch = track_tn[batch_idx].sum()

            if (tp_batch + fn_batch) > 0:
                track_eff = tp_batch / (tp_batch + fn_batch)
                batch_track_effs.append(track_eff)

            if (fp_batch + tn_batch) > 0:
                track_fake_rate = fp_batch / (fp_batch + tn_batch)
                if fp_batch.sum() >= 3:
                    batch_track_fake_rates.append(track_fake_rate)

        if batch_track_effs:
            mean_track_eff = torch.stack(batch_track_effs).mean()
            self.log(f"{stage}/track_efficiency", mean_track_eff, sync_dist=True)

        if batch_track_fake_rates:
            mean_track_fake_rate = torch.stack(batch_track_fake_rates).mean()
            self.log(f"{stage}/track_fake_rate", mean_track_fake_rate, sync_dist=True)

        true_num = true_valid.sum(-1)
        pred_num = pred_valid.sum(-1)
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


def main(args: ArgsType = None) -> None:
    CLI(
        model_class=WrapperModule,
        datamodule_class=AtlasMuonDataModule,
        args=args,
        parser_kwargs={"default_env": True},
        save_config_kwargs={"overwrite": True},  # Allow overwriting config to avoid conflicts
    )


if __name__ == "__main__":
    main()
