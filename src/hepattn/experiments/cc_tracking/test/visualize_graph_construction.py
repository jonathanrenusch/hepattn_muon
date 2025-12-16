#!/usr/bin/env python
"""
Visualization script for outward graph construction in particle tracking.

This script demonstrates and validates the graph construction approach where:
1. Hits are sorted by radial distance (r) from the interaction point
2. Each hit on a track is connected to the NEXT hit (outward) on the same track
3. This creates a directed graph where edges point outward from the interaction point

The resulting graph can be used for:
- Training: Predict next-hit connections (edge classification)
- Inference: Use connected components to extract tracks

Key insight: No Hungarian matching needed! We use connected components to find tracks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import yaml
import h5py
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import argparse


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib visualization."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def load_event(data_dir: str, event_idx: int = 0):
    """Load a single event from the HDF5 dataset."""
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / 'metadata.yaml', 'r') as f:
        metadata = yaml.safe_load(f)
    
    hit_features = metadata['hit_features']
    
    # Load index arrays
    file_indices = np.load(data_dir / 'event_file_indices.npy')
    row_indices = np.load(data_dir / 'event_row_indices.npy')
    
    # Get file and row info
    file_idx = file_indices[event_idx]
    row_idx = row_indices[event_idx]
    
    # Get chunk info
    chunk = metadata['event_mapping']['chunk_summary'][file_idx]
    h5_file_path = data_dir / chunk['h5_file']
    
    with h5py.File(h5_file_path, 'r') as f:
        num_hits = f['num_hits'][row_idx]
        num_tracks = f['num_tracks'][row_idx]
        hits_array = f['hits'][row_idx, :num_hits]
    
    # Convert to dictionary
    hits = {}
    for i, feature_name in enumerate(hit_features):
        hits[feature_name] = hits_array[:, i]
    
    # Scale coordinates (matching data.py)
    hits['x'] = hits['spacePoint_globEdgeLowX'] * 0.001
    hits['y'] = hits['spacePoint_globEdgeLowY'] * 0.001
    hits['z'] = hits['spacePoint_globEdgeLowZ'] * 0.001
    
    # Compute r (radial distance from interaction point)
    hits['r'] = np.sqrt(hits['x']**2 + hits['y']**2)
    
    # Get particle IDs
    hits['particle_id'] = hits['spacePoint_truthLink']
    
    return hits, num_hits, num_tracks


def build_outward_edges(hits: dict, num_hits: int):
    """
    Build outward-directed edges for track reconstruction.
    
    For each track:
    1. Sort hits by r (radial distance)
    2. Connect hit[i] -> hit[i+1] (pointing outward)
    
    Returns:
        edge_index: (2, E) array of (source, target) pairs
        edge_particle_ids: (E,) array of particle IDs for each edge
        adjacency_matrix: (N, N) sparse adjacency matrix
    """
    # Get unique particle IDs (excluding noise = -1)
    particle_ids = hits['particle_id']
    unique_pids = np.unique(particle_ids[particle_ids >= 0])
    
    source_nodes = []
    target_nodes = []
    edge_pids = []
    
    for pid in unique_pids:
        # Get hits for this particle
        mask = particle_ids == pid
        hit_indices = np.where(mask)[0]
        
        if len(hit_indices) < 2:
            continue  # Need at least 2 hits to form an edge
        
        # Sort by r (radial distance)
        r_values = hits['r'][hit_indices]
        sorted_order = np.argsort(r_values)
        sorted_indices = hit_indices[sorted_order]
        
        # Create edges: hit[i] -> hit[i+1]
        for i in range(len(sorted_indices) - 1):
            source_nodes.append(sorted_indices[i])
            target_nodes.append(sorted_indices[i + 1])
            edge_pids.append(pid)
    
    edge_index = np.array([source_nodes, target_nodes])
    edge_particle_ids = np.array(edge_pids)
    
    # Build sparse adjacency matrix
    if len(source_nodes) > 0:
        data = np.ones(len(source_nodes), dtype=np.int8)
        adjacency = csr_matrix((data, (source_nodes, target_nodes)), shape=(num_hits, num_hits))
    else:
        adjacency = csr_matrix((num_hits, num_hits), dtype=np.int8)
    
    return edge_index, edge_particle_ids, adjacency


def build_full_adjacency(hits: dict, num_hits: int):
    """
    Build full adjacency matrix (symmetric) for correlation task.
    
    hit[i] is connected to hit[j] if they belong to the same particle.
    This is the target for the auxiliary SelfAttentionCorrelationTask.
    
    Returns:
        adjacency_matrix: (N, N) sparse adjacency matrix
    """
    particle_ids = hits['particle_id']
    unique_pids = np.unique(particle_ids[particle_ids >= 0])
    
    rows = []
    cols = []
    
    for pid in unique_pids:
        mask = particle_ids == pid
        hit_indices = np.where(mask)[0]
        
        # Create all pairs within this particle's hits
        for i in range(len(hit_indices)):
            for j in range(len(hit_indices)):
                if i != j:  # No self-loops
                    rows.append(hit_indices[i])
                    cols.append(hit_indices[j])
    
    if len(rows) > 0:
        data = np.ones(len(rows), dtype=np.int8)
        adjacency = csr_matrix((data, (rows, cols)), shape=(num_hits, num_hits))
    else:
        adjacency = csr_matrix((num_hits, num_hits), dtype=np.int8)
    
    return adjacency


def extract_tracks_from_adjacency(adjacency: csr_matrix):
    """
    Extract tracks using connected components.
    
    This is the key insight: No Hungarian matching needed!
    We simply find connected components in the graph.
    
    Args:
        adjacency: Sparse adjacency matrix (can be directed or undirected)
    
    Returns:
        n_tracks: Number of tracks found
        labels: (N,) array of track labels for each hit
    """
    # Make symmetric for undirected connected components
    symmetric = adjacency + adjacency.T
    symmetric.data = np.ones_like(symmetric.data)  # Binary
    
    n_components, labels = connected_components(
        csgraph=symmetric, 
        directed=False, 
        return_labels=True
    )
    
    return n_components, labels


def visualize_event(hits: dict, edge_index: np.ndarray, edge_particle_ids: np.ndarray,
                    full_adjacency: csr_matrix, num_hits: int, save_path: str = None):
    """Create visualization of the graph construction."""
    
    fig = plt.figure(figsize=(20, 8))
    
    # Get unique particles for coloring
    particle_ids = hits['particle_id']
    unique_pids = np.unique(particle_ids[particle_ids >= 0])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_pids), 1)))
    pid_to_color = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_pids)}
    
    # ===== Plot 1: R-Z view with outward edges =====
    ax1 = fig.add_subplot(131)
    
    # Plot noise hits
    noise_mask = particle_ids < 0
    ax1.scatter(hits['z'][noise_mask], hits['r'][noise_mask], 
                c='lightgray', s=10, alpha=0.5, label='Noise')
    
    # Plot track hits with colors
    for pid in unique_pids:
        mask = particle_ids == pid
        ax1.scatter(hits['z'][mask], hits['r'][mask], 
                    c=[pid_to_color[pid]], s=30, alpha=0.8, label=f'Track {int(pid)}')
    
    # Draw outward edges
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i], edge_index[1, i]
        pid = edge_particle_ids[i]
        ax1.annotate('', 
                     xy=(hits['z'][tgt], hits['r'][tgt]),
                     xytext=(hits['z'][src], hits['r'][src]),
                     arrowprops=dict(arrowstyle='->', color=pid_to_color[pid], lw=1.5))
    
    ax1.set_xlabel('Z (m)')
    ax1.set_ylabel('R (m)')
    ax1.set_title('R-Z View: Outward Edges (Training Target)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: X-Y view (transverse) =====
    ax2 = fig.add_subplot(132)
    
    ax2.scatter(hits['x'][noise_mask], hits['y'][noise_mask], 
                c='lightgray', s=10, alpha=0.5, label='Noise')
    
    for pid in unique_pids:
        mask = particle_ids == pid
        ax2.scatter(hits['x'][mask], hits['y'][mask], 
                    c=[pid_to_color[pid]], s=30, alpha=0.8)
    
    # Draw outward edges
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i], edge_index[1, i]
        pid = edge_particle_ids[i]
        ax2.annotate('', 
                     xy=(hits['x'][tgt], hits['y'][tgt]),
                     xytext=(hits['x'][src], hits['y'][src]),
                     arrowprops=dict(arrowstyle='->', color=pid_to_color[pid], lw=1.5))
    
    # Mark interaction point
    ax2.scatter([0], [0], c='red', s=100, marker='x', linewidths=3, label='IP')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y View: Tracks Propagating Outward')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: Adjacency matrices =====
    ax3 = fig.add_subplot(133)
    
    # Show full adjacency matrix
    full_adj_dense = full_adjacency.toarray()
    ax3.imshow(full_adj_dense, cmap='Blues', aspect='auto')
    ax3.set_xlabel('Hit Index')
    ax3.set_ylabel('Hit Index')
    ax3.set_title(f'Full Adjacency Matrix\n({full_adjacency.nnz} non-zero entries)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def print_statistics(hits: dict, edge_index: np.ndarray, adjacency: csr_matrix,
                     full_adjacency: csr_matrix, num_hits: int, num_tracks: int):
    """Print statistics about the graph construction."""
    
    particle_ids = hits['particle_id']
    noise_hits = (particle_ids < 0).sum()
    track_hits = (particle_ids >= 0).sum()
    
    # Extract tracks from outward adjacency
    n_extracted_outward, labels_outward = extract_tracks_from_adjacency(adjacency)
    
    # Extract tracks from full adjacency
    n_extracted_full, labels_full = extract_tracks_from_adjacency(full_adjacency)
    
    print("=" * 60)
    print("OUTWARD GRAPH CONSTRUCTION STATISTICS")
    print("=" * 60)
    print(f"\n📊 Event Statistics:")
    print(f"   Total hits: {num_hits}")
    print(f"   Track hits: {track_hits} ({100*track_hits/num_hits:.1f}%)")
    print(f"   Noise hits: {noise_hits} ({100*noise_hits/num_hits:.1f}%)")
    print(f"   True tracks: {num_tracks}")
    
    print(f"\n🔗 Outward Edge Graph (Training Target):")
    print(f"   Number of edges: {edge_index.shape[1]}")
    print(f"   Edges per track: {edge_index.shape[1] / max(num_tracks, 1):.1f}")
    print(f"   Adjacency matrix size: {adjacency.shape}")
    print(f"   Sparsity: {100 * (1 - adjacency.nnz / (num_hits**2)):.4f}%")
    
    print(f"\n🔗 Full Adjacency Graph (Correlation Target):")
    print(f"   Number of edges: {full_adjacency.nnz}")
    print(f"   Sparsity: {100 * (1 - full_adjacency.nnz / (num_hits**2)):.4f}%")
    
    print(f"\n🎯 Connected Components Extraction:")
    print(f"   From outward edges: {n_extracted_outward} components")
    print(f"   From full adjacency: {n_extracted_full} components")
    print(f"   True tracks: {num_tracks}")
    
    # Check if extraction is correct
    if n_extracted_full == num_tracks + noise_hits:
        print(f"   ✅ Full adjacency correctly identifies {num_tracks} tracks + {noise_hits} isolated noise hits")
    else:
        print(f"   ⚠️  Mismatch: expected {num_tracks + noise_hits}, got {n_extracted_full}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: No Hungarian matching needed!")
    print("Connected components directly give us tracks.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize outward graph construction for tracking')
    parser.add_argument('--data_dir', type=str, 
                        default='/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit600',
                        help='Path to HDF5 dataset directory')
    parser.add_argument('--event_idx', type=int, default=0,
                        help='Event index to visualize')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the visualization')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting, only print statistics')
    args = parser.parse_args()
    
    print(f"Loading event {args.event_idx} from {args.data_dir}...")
    hits, num_hits, num_tracks = load_event(args.data_dir, args.event_idx)
    
    print("Building outward edge graph...")
    edge_index, edge_particle_ids, outward_adjacency = build_outward_edges(hits, num_hits)
    
    print("Building full adjacency matrix...")
    full_adjacency = build_full_adjacency(hits, num_hits)
    
    # Print statistics
    print_statistics(hits, edge_index, outward_adjacency, full_adjacency, num_hits, num_tracks)
    
    if not args.no_plot:
        print("\nCreating visualization...")
        visualize_event(hits, edge_index, edge_particle_ids, full_adjacency, 
                        num_hits, save_path=args.save_path)


if __name__ == '__main__':
    main()
