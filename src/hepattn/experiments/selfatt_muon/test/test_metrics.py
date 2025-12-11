"""
Test script for the custom metrics logging in run_tracking.py.

This script tests:
1. Metric computation logic
2. Edge cases (no predictions, no true tracks, etc.)
3. Correct handling of valid/invalid hits
"""

import torch
import sys
sys.path.insert(0, '/shared/tracking/hepattn_muon/src')


def compute_metrics(pred_matrix, target_corr, valid_hits, min_hits_per_track=3, threshold=0.5):
    """
    Standalone implementation of the metric computation for testing.
    
    Args:
        pred_matrix: [B, N, N] binary predicted correlation matrix
        target_corr: [B, N, N] ground truth correlation matrix
        valid_hits: [B, N] mask for valid (non-padding) hits
        min_hits_per_track: minimum hits to count as a track
        threshold: not used here since pred_matrix is already binary
    
    Returns:
        Dictionary of computed metrics
    """
    batch_size = pred_matrix.shape[0]
    num_hits = pred_matrix.shape[1]
    
    # Metrics accumulators
    total_pred_tracks = 0
    total_true_tracks = 0
    total_fake_tracks = 0
    total_matched_true_tracks = 0
    
    track_efficiencies = []
    track_purities = []
    double_matched_count = 0
    
    for b in range(batch_size):
        valid_mask = valid_hits[b]  # [N]
        pred_b = pred_matrix[b]  # [N, N]
        target_b = target_corr[b]  # [N, N]
        
        print(f"\n--- Batch {b} ---")
        print(f"Valid hits: {valid_mask.sum().item()} / {num_hits}")
        
        # Count hits per row in predictions (only for valid hits)
        pred_hits_per_row = (pred_b & valid_mask.unsqueeze(0)).sum(dim=-1)  # [N]
        pred_track_rows = (pred_hits_per_row >= min_hits_per_track) & valid_mask  # [N]
        n_pred_tracks_b = pred_track_rows.sum().item()
        
        print(f"Predicted track rows: {torch.where(pred_track_rows)[0].tolist()}")
        print(f"Number of predicted tracks: {n_pred_tracks_b}")
        
        # Count true tracks
        target_hits_per_row = (target_b & valid_mask.unsqueeze(0)).sum(dim=-1)  # [N]
        true_track_rows = (target_hits_per_row >= min_hits_per_track) & valid_mask  # [N]
        n_true_tracks_b = true_track_rows.sum().item()
        
        print(f"True track rows: {torch.where(true_track_rows)[0].tolist()}")
        print(f"Number of true tracks: {n_true_tracks_b}")
        
        total_pred_tracks += n_pred_tracks_b
        total_true_tracks += n_true_tracks_b
        
        # For each predicted track
        pred_row_indices = torch.where(pred_track_rows)[0]
        
        for row_idx in pred_row_indices:
            pred_hits = pred_b[row_idx] & valid_mask
            n_pred_hits = pred_hits.sum().item()
            
            if n_pred_hits == 0:
                continue
            
            is_true_track_row = target_hits_per_row[row_idx] >= min_hits_per_track
            
            if not is_true_track_row:
                total_fake_tracks += 1
                print(f"  Row {row_idx.item()}: FAKE (not a true innermost hit)")
                continue
            
            true_hits = target_b[row_idx] & valid_mask
            n_true_hits = true_hits.sum().item()
            
            if n_true_hits == 0:
                total_fake_tracks += 1
                print(f"  Row {row_idx.item()}: FAKE (no true hits)")
                continue
            
            correct_hits = (pred_hits & true_hits).sum().item()
            track_eff = correct_hits / n_true_hits if n_true_hits > 0 else 0.0
            track_pur = correct_hits / n_pred_hits if n_pred_hits > 0 else 0.0
            
            track_efficiencies.append(track_eff)
            track_purities.append(track_pur)
            
            print(f"  Row {row_idx.item()}: pred={n_pred_hits}, true={n_true_hits}, correct={correct_hits}, eff={track_eff:.2f}, pur={track_pur:.2f}")
            
            if track_eff >= 0.5 and track_pur >= 0.5:
                double_matched_count += 1
                total_matched_true_tracks += 1
    
    # Compute aggregate metrics
    metrics = {
        "n_pred_tracks": float(total_pred_tracks),
        "n_true_tracks": float(total_true_tracks),
    }
    
    if total_pred_tracks > 0:
        metrics["fake_rate"] = float(total_fake_tracks) / total_pred_tracks
    else:
        metrics["fake_rate"] = 0.0
    
    if total_true_tracks > 0:
        metrics["efficiency"] = float(total_matched_true_tracks) / total_true_tracks
    else:
        metrics["efficiency"] = 0.0
    
    if len(track_efficiencies) > 0:
        metrics["avg_track_efficiency"] = sum(track_efficiencies) / len(track_efficiencies)
    else:
        metrics["avg_track_efficiency"] = 0.0
    
    if len(track_purities) > 0:
        metrics["avg_track_purity"] = sum(track_purities) / len(track_purities)
    else:
        metrics["avg_track_purity"] = 0.0
    
    if total_pred_tracks > 0:
        metrics["double_match_rate"] = float(double_matched_count) / total_pred_tracks
    else:
        metrics["double_match_rate"] = 0.0
    
    return metrics


def test_perfect_prediction():
    """Test metrics when prediction perfectly matches target."""
    print("\n" + "=" * 60)
    print("TEST: Perfect Prediction")
    print("=" * 60)
    
    batch_size = 2
    num_hits = 15
    
    # Create target
    target = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    valid = torch.ones(batch_size, num_hits, dtype=torch.bool)
    
    # Event 0: 2 tracks
    target[0, 0, 0:5] = True   # Track 0: innermost at 0, hits 0-4
    target[0, 5, 5:10] = True  # Track 1: innermost at 5, hits 5-9
    valid[0, 10:] = False      # Padding
    
    # Event 1: 1 track
    target[1, 0, 0:8] = True   # Track 0: innermost at 0, hits 0-7
    valid[1, 8:] = False       # Padding
    
    # Perfect prediction
    pred = target.clone()
    
    metrics = compute_metrics(pred, target, valid)
    
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Verify
    assert metrics["n_pred_tracks"] == 3, f"Expected 3 pred tracks, got {metrics['n_pred_tracks']}"
    assert metrics["n_true_tracks"] == 3, f"Expected 3 true tracks, got {metrics['n_true_tracks']}"
    assert metrics["fake_rate"] == 0.0, f"Expected 0% fake rate, got {metrics['fake_rate']}"
    assert metrics["efficiency"] == 1.0, f"Expected 100% efficiency, got {metrics['efficiency']}"
    assert metrics["avg_track_efficiency"] == 1.0, f"Expected 100% avg track eff, got {metrics['avg_track_efficiency']}"
    assert metrics["avg_track_purity"] == 1.0, f"Expected 100% avg track pur, got {metrics['avg_track_purity']}"
    
    print("✓ Perfect prediction test passed\n")


def test_no_predictions():
    """Test metrics when no tracks are predicted."""
    print("\n" + "=" * 60)
    print("TEST: No Predictions")
    print("=" * 60)
    
    batch_size = 1
    num_hits = 10
    
    target = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    valid = torch.ones(batch_size, num_hits, dtype=torch.bool)
    
    # True track exists
    target[0, 0, 0:5] = True
    
    # No predictions (all zeros)
    pred = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    
    metrics = compute_metrics(pred, target, valid)
    
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    assert metrics["n_pred_tracks"] == 0, f"Expected 0 pred tracks"
    assert metrics["n_true_tracks"] == 1, f"Expected 1 true track"
    assert metrics["efficiency"] == 0.0, f"Expected 0% efficiency"
    
    print("✓ No predictions test passed\n")


def test_fake_predictions():
    """Test metrics with fake tracks."""
    print("\n" + "=" * 60)
    print("TEST: Fake Predictions")
    print("=" * 60)
    
    batch_size = 1
    num_hits = 10
    
    target = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    valid = torch.ones(batch_size, num_hits, dtype=torch.bool)
    
    # True track at row 0
    target[0, 0, 0:5] = True
    
    # Predict from a non-innermost row (fake track)
    pred = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    pred[0, 3, 0:5] = True  # Row 3 is not an innermost hit
    
    metrics = compute_metrics(pred, target, valid)
    
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    assert metrics["n_pred_tracks"] == 1, f"Expected 1 pred track"
    assert metrics["fake_rate"] == 1.0, f"Expected 100% fake rate, got {metrics['fake_rate']}"
    
    print("✓ Fake predictions test passed\n")


def test_partial_match():
    """Test metrics with partial matches."""
    print("\n" + "=" * 60)
    print("TEST: Partial Match")
    print("=" * 60)
    
    batch_size = 1
    num_hits = 10
    
    target = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    valid = torch.ones(batch_size, num_hits, dtype=torch.bool)
    
    # True track: hits 0-5 (6 hits)
    target[0, 0, 0:6] = True
    
    # Predict: hits 0-3 correct + 2 wrong (4 correct, 2 wrong)
    pred = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    pred[0, 0, 0:4] = True   # Correct hits
    pred[0, 0, 7:9] = True   # Wrong hits
    
    metrics = compute_metrics(pred, target, valid)
    
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Efficiency: 4/6 = 0.667
    # Purity: 4/6 = 0.667
    expected_eff = 4/6
    expected_pur = 4/6
    
    assert abs(metrics["avg_track_efficiency"] - expected_eff) < 0.01, \
        f"Expected eff {expected_eff:.3f}, got {metrics['avg_track_efficiency']:.3f}"
    assert abs(metrics["avg_track_purity"] - expected_pur) < 0.01, \
        f"Expected pur {expected_pur:.3f}, got {metrics['avg_track_purity']:.3f}"
    
    print("✓ Partial match test passed\n")


def test_padding_handling():
    """Test that padding is correctly handled."""
    print("\n" + "=" * 60)
    print("TEST: Padding Handling")
    print("=" * 60)
    
    batch_size = 1
    num_hits = 20
    
    target = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    valid = torch.zeros(batch_size, num_hits, dtype=torch.bool)
    
    # Only first 8 hits are valid
    valid[0, 0:8] = True
    
    # True track in valid region
    target[0, 0, 0:5] = True
    
    # Prediction in valid region (correct)
    pred = torch.zeros(batch_size, num_hits, num_hits, dtype=torch.bool)
    pred[0, 0, 0:5] = True
    
    # Also predict in padding region (should be ignored)
    pred[0, 10, 10:15] = True
    
    metrics = compute_metrics(pred, target, valid)
    
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Should only count 1 pred track (padding prediction ignored)
    assert metrics["n_pred_tracks"] == 1, f"Expected 1 pred track (padding ignored), got {metrics['n_pred_tracks']}"
    
    print("✓ Padding handling test passed\n")


def main():
    print("\n" + "=" * 60)
    print("METRICS COMPUTATION TEST SUITE")
    print("=" * 60 + "\n")
    
    test_perfect_prediction()
    test_no_predictions()
    test_fake_predictions()
    test_partial_match()
    test_padding_handling()
    
    print("=" * 60)
    print("ALL METRICS TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
