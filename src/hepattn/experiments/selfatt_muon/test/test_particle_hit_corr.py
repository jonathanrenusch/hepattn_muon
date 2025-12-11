"""
Test script to verify the particle_hit_corr matrix building logic.

This script simulates mock hit data with up to 2 tracks and up to 30 hits per event,
then builds the correlation matrix that marks which hits belong to the same particle
based on the innermost hit (min_r) for each particle.
"""

import numpy as np
import torch

np.random.seed(42)


def generate_mock_event(event_id, max_hits=30, max_tracks=2):
    """
    Generate mock hit data for a single event.
    
    Returns:
        hits: dict with 'spacePoint_truthLink', 'r' (radial position)
        num_hits: number of hits
        num_tracks: number of particles/tracks
    """
    # Randomly decide number of tracks (1 or 2)
    num_tracks = np.random.randint(1, max_tracks + 1)
    
    # Randomly decide hits per track (between 3 and 15 hits per track)
    hits_per_track = [np.random.randint(3, 16) for _ in range(num_tracks)]
    
    # Also add some noise hits (not belonging to any valid particle)
    num_noise_hits = np.random.randint(0, 6)
    
    total_hits = sum(hits_per_track) + num_noise_hits
    if total_hits > max_hits:
        # Scale down if we exceed max
        scale = max_hits / total_hits
        hits_per_track = [max(2, int(h * scale)) for h in hits_per_track]
        num_noise_hits = max(0, max_hits - sum(hits_per_track))
        total_hits = sum(hits_per_track) + num_noise_hits
    
    num_hits = total_hits
    
    # Build hit arrays
    truth_links = []
    r_values = []
    
    # Particle IDs will be 0, 1, ... for valid particles
    for track_id in range(num_tracks):
        n_hits_this_track = hits_per_track[track_id]
        # Generate random radial positions for this track's hits
        # Radial values typically increase as particle moves outward
        r_for_track = np.sort(np.random.uniform(0.1, 10.0, n_hits_this_track))
        
        truth_links.extend([track_id] * n_hits_this_track)
        r_values.extend(r_for_track)
    
    # Add noise hits with truth_link = -1
    for _ in range(num_noise_hits):
        truth_links.append(-1)
        r_values.append(np.random.uniform(0.1, 10.0))
    
    # Shuffle the hits to simulate random ordering
    indices = np.random.permutation(num_hits)
    truth_links = np.array(truth_links)[indices]
    r_values = np.array(r_values)[indices]
    
    hits = {
        'spacePoint_truthLink': truth_links,
        'r': r_values,
        'on_valid_particle': truth_links >= 0,
    }
    
    # Get unique particle IDs (excluding noise)
    particle_ids = np.unique(truth_links[truth_links >= 0])
    
    return hits, num_hits, num_tracks, particle_ids


def build_particle_hit_corr_matrix(hits, particle_ids, num_hits, event_max_num_particles=6):
    """
    Build the particle_hit_corr matrix following the logic from data.py.
    
    This is the extracted logic from __getitem__ that we want to test.
    """
    # Convert to torch tensors as in original code
    particle_ids_tensor = torch.from_numpy(particle_ids)
    
    # Fill in empty slots with -999s and get the IDs of the particle on each hit
    particle_ids_padded = torch.cat([
        particle_ids_tensor, 
        -999 * torch.ones(event_max_num_particles - len(particle_ids_tensor))
    ]).type(torch.int32)
    
    hit_particle_ids = torch.from_numpy(hits["spacePoint_truthLink"])

    # Finding innermost hit indices for each particle
    # NOTE: This is the logic from the original code that has bugs
    min_r_indices = []
    for idx in particle_ids_padded:
        # Original buggy logic:
        # if hit_particle_ids[idx] < 0:
        #     continue
        # mask = hits["spacePoint_truthLink"] == idx
        # min_r_indices.append(np.argmin(hits["hit_r"][mask]))
        
        # Corrected logic - idx is a particle ID, not an index
        idx_val = idx.item()
        if idx_val < 0:  # Skip padding slots
            continue
        mask = hits["spacePoint_truthLink"] == idx_val
        if mask.sum() == 0:  # No hits for this particle
            continue
        # Get indices where mask is True
        hit_indices = np.where(mask)[0]
        # Find which of these has minimum r
        r_vals = hits["r"][mask]
        min_local_idx = np.argmin(r_vals)
        min_r_indices.append(hit_indices[min_local_idx])

    min_r_indices = torch.tensor(min_r_indices).type(torch.int64)
    
    print(f"  Particle IDs: {particle_ids}")
    print(f"  Min-r hit indices: {min_r_indices.tolist()}")
    print(f"  R values at min-r indices: {[hits['r'][i] for i in min_r_indices]}")
    
    # Build the targets for whether a particle slot is used or not
    # Original code has shape issues - let's also fix that
    targets_particle_valid = torch.full((1, num_hits,), False)
    targets_particle_valid[0, min_r_indices] = True
    
    # Build adjacency targets between hits and particles
    # This is the correlation matrix we want to test
    particle_hit_corr = torch.full((1, num_hits, num_hits), False)
    
    # CORRECT LOGIC: For each particle's innermost hit (row), mark only the columns 
    # that correspond to hits belonging to the SAME particle
    for i, min_r_idx in enumerate(min_r_indices):
        if i < len(particle_ids):
            pid = particle_ids[i]
            # Create mask for hits belonging to this particle
            same_particle_mask = hits["spacePoint_truthLink"] == pid
            particle_hit_corr[0, min_r_idx, :] = torch.from_numpy(same_particle_mask)
    
    return particle_hit_corr, min_r_indices, targets_particle_valid


def visualize_matrix(matrix, hits, num_hits, particle_ids):
    """Print a nice visualization of the correlation matrix."""
    # Get the 2D matrix (remove batch dim)
    mat = matrix[0].numpy()
    
    print("\n  Correlation Matrix (rows=innermost hits, cols=all hits):")
    print("  " + "=" * (num_hits * 2 + 10))
    
    # Header with hit indices
    header = "       "
    for j in range(num_hits):
        header += f"{j:2d}"
    print(header)
    
    # Show which particle each hit belongs to
    truth_row = "  PID: "
    for j in range(num_hits):
        pid = hits["spacePoint_truthLink"][j]
        if pid >= 0:
            truth_row += f"{pid:2d}"
        else:
            truth_row += " ."
    print(truth_row)
    print("       " + "--" * num_hits)
    
    # Print the matrix
    for i in range(num_hits):
        row_str = f"  {i:2d} | "
        for j in range(num_hits):
            if mat[i, j]:
                row_str += " X"
            else:
                row_str += " ."
        # Mark if this row is an innermost hit
        if mat[i, :].any():
            row_str += f"  <- innermost for particle"
        print(row_str)
    
    print()


def main():
    print("=" * 70)
    print("Testing particle_hit_corr Matrix Building Logic")
    print("=" * 70)
    print()
    
    for event_id in range(10):
        print(f"\n{'='*70}")
        print(f"EVENT {event_id}")
        print(f"{'='*70}")
        
        # Generate mock data
        hits, num_hits, num_tracks, particle_ids = generate_mock_event(event_id)
        
        print(f"\n  Number of hits: {num_hits}")
        print(f"  Number of tracks: {num_tracks}")
        print(f"  Truth links: {hits['spacePoint_truthLink']}")
        print(f"  R values: {np.round(hits['r'], 3)}")
        
        # Build the correlation matrix
        particle_hit_corr, min_r_indices, particle_valid = build_particle_hit_corr_matrix(
            hits, particle_ids, num_hits
        )
        
        # Visualize
        visualize_matrix(particle_hit_corr, hits, num_hits, particle_ids)
        
        # Some basic sanity checks
        print("  Sanity Checks:")
        print(f"    - Number of True rows in corr matrix: {particle_hit_corr[0].any(dim=1).sum().item()}")
        print(f"    - Should equal num_tracks: {num_tracks}")
        print(f"    - Matrix shape: {particle_hit_corr.shape}")
        
        # Verify the innermost hits are correctly identified
        for i, pid in enumerate(particle_ids):
            mask = hits["spacePoint_truthLink"] == pid
            r_vals = hits["r"][mask]
            min_r = r_vals.min()
            expected_idx = np.where((hits["spacePoint_truthLink"] == pid) & (hits["r"] == min_r))[0][0]
            actual_idx = min_r_indices[i].item()
            match = "✓" if expected_idx == actual_idx else "✗"
            print(f"    - Particle {pid}: min_r={min_r:.3f}, expected_idx={expected_idx}, actual_idx={actual_idx} {match}")


if __name__ == "__main__":
    main()
