"""Test the validation step directly to debug the hang."""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parents[3]))

print("=" * 60)
print("VALIDATION STEP DEBUG TEST")
print("=" * 60)


def main():
    """Test the full validation step."""
    from hepattn.models import HitFilter, Encoder, InputNet, Dense
    from hepattn.models.task import SelfAttentionCorrelationTask
    from hepattn.experiments.selfatt_muon.data import AtlasMuonDataset, AtlasMuonCollator
    from hepattn.experiments.selfatt_muon.run_tracking import SelfAttentionTracker
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")

    dim = 64

    # Create model
    print("\n1. Creating model...")
    t0 = time.time()
    
    input_net = InputNet(
        input_name="hit",
        fields=[
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "spacePoint_globEdgeLowZ",
            "r",
            "phi",
            "eta",
        ],
        net=Dense(input_size=6, hidden_layers=[64], output_size=dim),
    )

    encoder = Encoder(
        num_layers=4,
        dim=dim,
        attn_type="flash",
        window_size=None,
        window_wrap=False,
        hybrid_norm=True,
        norm="RMSNorm",
    )

    task = SelfAttentionCorrelationTask(
        name="hit_correlation",
        input_object="hit",
        target_field="particle_hit_corr",
        dim=dim,
        threshold=0.5,
        loss_fn="bce",
        has_intermediate_loss=False,
    )

    model = HitFilter(
        input_nets=nn.ModuleList([input_net]),
        encoder=encoder,
        tasks=nn.ModuleList([task]),
        input_sort_field=None,
    )
    
    model = model.to(device).to(dtype)
    print(f"   Model created in {time.time() - t0:.2f}s")

    # Load data
    print("\n2. Loading data...")
    t0 = time.time()
    
    data_dir = "/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    inputs_config = {
        "hit": [
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "spacePoint_globEdgeLowZ",
            "r",
            "phi",
            "eta",
        ]
    }
    targets_config = {"particle": [], "hit": []}

    dataset = AtlasMuonDataset(
        dirpath=data_dir,
        inputs=inputs_config,
        targets=targets_config,
        num_events=32,
        event_max_num_particles=2,
    )

    collator = AtlasMuonCollator(
        dataset_inputs=inputs_config,
        dataset_targets=targets_config,
        max_num_obj=2,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        collate_fn=collator,
        num_workers=0,
    )
    print(f"   Data loaded in {time.time() - t0:.2f}s")

    # Get batch
    print("\n3. Getting batch...")
    t0 = time.time()
    batch_inputs, batch_targets = next(iter(dataloader))
    print(f"   Batch retrieved in {time.time() - t0:.2f}s")

    # Move to device
    print("\n4. Moving to device...")
    t0 = time.time()
    
    def move_to_device(d, dev, dt):
        result = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.to(dev)
                if v.dtype in (torch.float32, torch.float64):
                    v = v.to(dt)
            result[k] = v
        return result

    batch_inputs = move_to_device(batch_inputs, device, dtype)
    batch_targets = move_to_device(batch_targets, device, dtype)
    print(f"   Moved in {time.time() - t0:.2f}s")

    # Forward pass
    print("\n5. Forward pass...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(batch_inputs)
    print(f"   Forward in {time.time() - t0:.2f}s")
    print(f"   Output shape: {outputs['final']['hit_correlation']['hit_particle_hit_corr_logit'].shape}")

    # Loss computation
    print("\n6. Loss computation...")
    t0 = time.time()
    with torch.no_grad():
        losses, targets_with_matching = model.loss(outputs, batch_targets)
    print(f"   Loss in {time.time() - t0:.2f}s")
    total_loss = sum(lv for ll in losses.values() for tl in ll.values() for lv in tl.values())
    print(f"   Total loss: {total_loss.item():.6f}")

    # Predictions
    print("\n7. Predict...")
    t0 = time.time()
    with torch.no_grad():
        preds = model.predict(outputs)
    print(f"   Predict in {time.time() - t0:.2f}s")
    print(f"   Pred shape: {preds['final']['hit_correlation']['hit_particle_hit_corr'].shape}")

    # Now test the log_custom_metrics function
    print("\n8. Testing log_custom_metrics (without logging)...")
    t0 = time.time()
    
    # Create a minimal tracker instance without trainer
    # We'll just call the metric computation logic directly
    
    threshold = 0.5
    min_hits_per_track = 3
    
    pred_probs = preds["final"]["hit_correlation"]["hit_particle_hit_corr_prob"]
    pred_matrix = pred_probs >= threshold
    target_corr = targets_with_matching["hit_particle_hit_corr"]
    valid_hits = targets_with_matching["hit_valid"]
    
    batch_size = pred_matrix.shape[0]
    
    print(f"   pred_matrix shape: {pred_matrix.shape}")
    print(f"   target_corr shape: {target_corr.shape}")
    print(f"   valid_hits shape: {valid_hits.shape}")
    
    total_pred_tracks = 0
    total_true_tracks = 0
    total_fake_tracks = 0
    total_matched_true_tracks = 0
    track_efficiencies = []
    track_purities = []
    double_matched_count = 0
    
    for b in range(batch_size):
        print(f"\n   Processing batch {b}...")
        valid_mask = valid_hits[b]
        pred_b = pred_matrix[b]
        target_b = target_corr[b]
        
        pred_hits_per_row = (pred_b & valid_mask.unsqueeze(0)).sum(dim=-1)
        pred_track_rows = (pred_hits_per_row >= min_hits_per_track) & valid_mask
        n_pred_tracks_b = pred_track_rows.sum().item()
        
        target_hits_per_row = (target_b & valid_mask.unsqueeze(0)).sum(dim=-1)
        true_track_rows = (target_hits_per_row >= min_hits_per_track) & valid_mask
        n_true_tracks_b = true_track_rows.sum().item()
        
        print(f"      Pred tracks: {n_pred_tracks_b}, True tracks: {n_true_tracks_b}")
        
        total_pred_tracks += n_pred_tracks_b
        total_true_tracks += n_true_tracks_b
        
        pred_row_indices = torch.where(pred_track_rows)[0]
        print(f"      Processing {len(pred_row_indices)} pred track rows...")
        
        for row_idx in pred_row_indices:
            pred_hits = pred_b[row_idx] & valid_mask
            n_pred_hits = pred_hits.sum().item()
            
            if n_pred_hits == 0:
                continue
            
            is_true_track_row = target_hits_per_row[row_idx] >= min_hits_per_track
            
            if not is_true_track_row:
                total_fake_tracks += 1
                continue
            
            true_hits = target_b[row_idx] & valid_mask
            n_true_hits = true_hits.sum().item()
            
            if n_true_hits == 0:
                total_fake_tracks += 1
                continue
            
            correct_hits = (pred_hits & true_hits).sum().item()
            track_eff = correct_hits / n_true_hits if n_true_hits > 0 else 0.0
            track_pur = correct_hits / n_pred_hits if n_pred_hits > 0 else 0.0
            
            track_efficiencies.append(track_eff)
            track_purities.append(track_pur)
            
            if track_eff >= 0.5 and track_pur >= 0.5:
                double_matched_count += 1
                total_matched_true_tracks += 1
    
    print(f"\n   Metrics computed in {time.time() - t0:.2f}s")
    print(f"   Total pred tracks: {total_pred_tracks}")
    print(f"   Total true tracks: {total_true_tracks}")
    print(f"   Fake tracks: {total_fake_tracks}")
    print(f"   Matched true tracks: {total_matched_true_tracks}")
    print(f"   Double matched: {double_matched_count}")

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
