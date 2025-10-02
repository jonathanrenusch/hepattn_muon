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
        # print(preds.keys())
        # print(preds)

        # First log metrics that depend on outputs from multiple tasks
        # TODO: Make the task names configurable or match task names automatically
        pred_valid = preds["track_valid"]["track_valid"]
        # print("track_valid.shape", pred_valid.shape)
        true_valid = targets["particle_valid"]
        # print("particle_valid.shape", true_valid.shape)
        # print("particle_valid.shape", true_valid.shape)
        # Set the masks of any track slots that are not used as null
        pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
        # print("track_hit_valid", preds["track_hit_valid"]["track_hit_valid"].shape)
        # print("track_hit_valid.shape", pred_hit_masks.shape)
        true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

        # print("true_hit_masks.shape", true_hit_masks.shape)
        # Calculate the true/false positive rates between the predicted and true masks
        # Number of hits that were correctly assigned to the track
        hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
        # print("hit_tp.shape", hit_tp.shape)
        # print("hit_tp", hit_tp)

        # Number of predicted hits on the track
        hit_p = pred_hit_masks.sum(-1)
        # print("hit_p.shape", hit_p.shape)
        # print("hit_p", hit_p)

        # True number of hits on the track
        hit_t = true_hit_masks.sum(-1)
        # print("hit_t.shape", hit_t.shape)
        # print("hit_t", hit_t)

        # Calculate the efficiency and purity at differnt matching working points
        both_valid = true_valid & pred_valid
        for wp in [0.25, 0.5, 0.75, 1.0]:
            # print("both_valid.shape", both_valid.shape)
            # print("both_valid", both_valid)
            effs = (hit_tp / hit_t >= wp) & both_valid
            # print(effs.shape)
            # print("effs", effs) 
            purs = (hit_tp / hit_p >= wp) & both_valid

            roi_effs = effs.float().sum(-1) / true_valid.float().sum(-1)
            # print("roi_effs.shape", roi_effs.shape)
            # print("roi_effs", roi_effs)
            roi_purs = purs.float().sum(-1) / pred_valid.float().sum(-1)

            mean_eff = roi_effs.nanmean()
            mean_pur = roi_purs.nanmean()

            self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

        # Calculate track-level efficiency and fake rate for track_valid prediction
        # True Positives: predicted valid AND actually valid
        track_tp = (pred_valid & true_valid).float()
        # False Positives: predicted valid BUT actually invalid  
        track_fp = (pred_valid & ~true_valid).float()
        # False Negatives: predicted invalid BUT actually valid
        track_fn = (~pred_valid & true_valid).float()
        # True Negatives: predicted invalid AND actually invalid
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
                if fp_batch.sum() >=3:
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
        # number true predicted hits: 
        true_pos_hits = (true_hit_masks & pred_hit_masks).sum() / torch.sum(true_hit_masks)
        false_pos_hits = (pred_hit_masks.sum() - (true_hit_masks & pred_hit_masks).sum()) / torch.sum(~true_hit_masks)
        # true_neg_hits = (~pred_hit_masks.sum() - (~true_hit_masks & ~pred_hit_masks).sum()) / torch.sum(~true_hit_masks)
        # false_neg_hits = (true_hit_masks & ~pred_hit_masks).sum() / torch.sum(true_hit_masks)

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
    )


if __name__ == "__main__":
    main()
