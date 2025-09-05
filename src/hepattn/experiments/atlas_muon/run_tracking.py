import comet_ml  # noqa: F401
import torch
from lightning.pytorch.cli import ArgsType
from torch import nn

from hepattn.experiments.atlas_muon.data import AtlasMuonDataModule
from hepattn.models.wrapper import ModelWrapper
from hepattn.utils.cli import CLI


class TrackMLTracker(ModelWrapper):
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

    # def log_custom_metrics(self, preds, targets, stage):
    #     # Just log predictions from the final layer
    #     preds = preds["final"]

    #     # First log metrics that depend on outputs from multiple tasks
    #     # TODO: Make the task names configurable or match task names automatically
    #     pred_valid = preds["track_valid"]["track_valid"]
    #     # print("track_valid.shape", pred_valid.shape)
    #     true_valid = targets["particle_valid"]
    #     # print("particle_valid.shape", true_valid.shape)

    #     # Set the masks of any track slots that are not used as null
    #     pred_hit_masks = preds["track_hit_valid"]["track_hit_valid"] & pred_valid.unsqueeze(-1)
    #     # print("track_hit_valid", preds["track_hit_valid"]["track_hit_valid"].shape)
    #     # print("track_hit_valid.shape", pred_hit_masks.shape)
    #     true_hit_masks = targets["particle_hit_valid"] & true_valid.unsqueeze(-1)

    #     # Calculate the true/false positive rates between the predicted and true masks
    #     # Number of hits that were correctly assigned to the track
    #     hit_tp = (pred_hit_masks & true_hit_masks).sum(-1)
    #     # self.log(f"{stage}/true_pos_hits", hit_tp.sum(), sync_dist=True)

    #     # Number of predicted hits on the track
    #     hit_p = pred_hit_masks.sum(-1)
    #     # self.log(f"{stage}/pred_hits_on_track", hit_p.sum(), sync_dist=True)

    #     # True number of hits on the track
    #     hit_t = true_hit_masks.sum(-1)
    #     # self.log(f"{stage}/true_hits_on_particle", hit_t.sum(), sync_dist=True)

    #     # Calculate the efficiency and purity at different matching working points
    #     for wp in [0.5, 0.75, 1.0]:
    #         both_valid = true_valid & pred_valid

    #         # Avoid division by zero by adding small epsilon where denominator is 0
    #         hit_t_safe = torch.where(hit_t > 0, hit_t, torch.ones_like(hit_t))
    #         hit_p_safe = torch.where(hit_p > 0, hit_p, torch.ones_like(hit_p))
            
    #         effs = (hit_tp / hit_t_safe >= wp) & both_valid & (hit_t > 0)
    #         purs = (hit_tp / hit_p_safe >= wp) & both_valid & (hit_p > 0)

    #         # Calculate efficiency and purity per batch item, then average
    #         batch_effs = []
    #         batch_purs = []
            
    #         for batch_idx in range(true_valid.shape[0]):
    #             true_valid_batch = true_valid[batch_idx]
    #             pred_valid_batch = pred_valid[batch_idx]
    #             effs_batch = effs[batch_idx]
    #             purs_batch = purs[batch_idx]
                
    #             if true_valid_batch.sum() > 0:
    #                 roi_eff = effs_batch.float().sum() / true_valid_batch.float().sum()
    #                 batch_effs.append(roi_eff)
                
    #             if pred_valid_batch.sum() > 0:
    #                 roi_pur = purs_batch.float().sum() / pred_valid_batch.float().sum()
    #                 batch_purs.append(roi_pur)
            
    #         # Average across batch items
    #         if batch_effs:
    #             mean_eff = torch.stack(batch_effs).mean()
    #             self.log(f"{stage}/p{wp}_eff", mean_eff, sync_dist=True)
            
    #         if batch_purs:
    #             mean_pur = torch.stack(batch_purs).mean()
    #             self.log(f"{stage}/p{wp}_pur", mean_pur, sync_dist=True)

    #     # Calculate number statistics
    #     true_num = true_valid.sum(-1)  # [batch_size]
    #     pred_num = pred_valid.sum(-1)  # [batch_size]

    #     # Calculate hits per particle/track, handling batch dimension properly
    #     batch_nh_per_true = []
    #     batch_nh_per_pred = []
        
    #     for batch_idx in range(true_valid.shape[0]):
    #         true_valid_batch = true_valid[batch_idx]
    #         pred_valid_batch = pred_valid[batch_idx]
            
    #         if true_valid_batch.sum() > 0:
    #             nh_per_true = true_hit_masks[batch_idx].sum(-1).float()[true_valid_batch].mean()
    #             batch_nh_per_true.append(nh_per_true)
            
    #         if pred_valid_batch.sum() > 0:
    #             nh_per_pred = pred_hit_masks[batch_idx].sum(-1).float()[pred_valid_batch].mean()
    #             batch_nh_per_pred.append(nh_per_pred)

    #     if batch_nh_per_true:
    #         self.log(f"{stage}/nh_per_particle", torch.stack(batch_nh_per_true).mean(), sync_dist=True)
        
    #     if batch_nh_per_pred:
    #         self.log(f"{stage}/nh_per_track", torch.stack(batch_nh_per_pred).mean(), sync_dist=True)

    #     # Average number of tracks/particles across batch
    #     self.log(f"{stage}/num_tracks", pred_num.float().mean(), sync_dist=True)
    #     self.log(f"{stage}/num_particles", true_num.float().mean(), sync_dist=True)




def main(args: ArgsType = None) -> None:
    CLI(
        model_class=TrackMLTracker,
        datamodule_class=AtlasMuonDataModule,
        args=args,
        parser_kwargs={"default_env": True},
    )


if __name__ == "__main__":
    main()
