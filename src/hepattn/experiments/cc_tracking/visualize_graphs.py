"""
Visualization script to verify CC tracking graph building logic.

This script loads events from the validation dataset and plots the adjacency
matrices as graphs using networkx to visually verify the graph construction.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
from pathlib import Path

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hepattn.experiments.cc_tracking.data import CCTrackingDataset


def plot_adjacency_graph(
    adjacency_matrix: torch.Tensor,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    r_values: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    save_path: str,
):
    """
    Plot a graph from an adjacency matrix using x,y coordinates as node positions.
    
    Args:
        adjacency_matrix: [N, N] boolean tensor indicating edges
        x_coords: [N] array of x coordinates (already scaled back to mm)
        y_coords: [N] array of y coordinates (already scaled back to mm)
        r_values: [N] array of radial distances to display on nodes
        valid_mask: [N] boolean array indicating valid hits
        title: Plot title
        save_path: Path to save the figure
    """
    # Convert to numpy
    adj = adjacency_matrix.numpy() if isinstance(adjacency_matrix, torch.Tensor) else adjacency_matrix
    
    # Filter to valid hits only
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        print(f"No valid hits for {title}, skipping...")
        return
    
    # Create subgraph for valid hits
    adj_valid = adj[np.ix_(valid_indices, valid_indices)]
    x_valid = x_coords[valid_indices]
    y_valid = y_coords[valid_indices]
    r_valid = r_values[valid_indices]
    
    # Create graph
    G = nx.DiGraph()  # Directed graph to show edge direction
    
    # Add nodes
    for i in range(n_valid):
        G.add_node(i)
    
    # Add edges from adjacency matrix
    edges = np.where(adj_valid)
    for src, dst in zip(edges[0], edges[1]):
        if src != dst:  # Skip self-loops for visualization clarity
            G.add_edge(src, dst)
    
    # Node positions from x, y coordinates
    pos = {i: (x_valid[i], y_valid[i]) for i in range(n_valid)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw graph
    # Color nodes by their r value (fixed scale 0-15000)
    r_normalized = np.clip(r_valid / 15000, 0, 1)
    node_colors = plt.cm.viridis(r_normalized)
    
    # Draw edges with arrows (slim for better resolution when zooming)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=8,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
        width=0.5,
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=30,
        edgecolors='black',
        linewidths=1,
    )
    
    # Set axis limits and labels
    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_xlabel('X [mm]', fontsize=12)
    ax.set_ylabel('Y [mm]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for r values (fixed scale 0-15000)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=15000))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Radial distance r [mm]', fontsize=10)
    
    # Add info text
    n_edges = G.number_of_edges()
    info_text = f"Nodes: {n_valid}, Edges: {n_edges}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    # Configuration
    val_dir = "/scratch/ml_validation_data_144000_hdf5_filtered_wp0990_maxtrk2_maxhit600"
    output_dir = Path("/shared/tracking/logs/verifying")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_events = 10
    np.random.seed(42)
    
    # Input features needed (minimal set for visualization)
    inputs = {
        "hit": [
            "spacePoint_globEdgeLowX",
            "spacePoint_globEdgeLowY",
            "r",
        ]
    }
    targets = {"particle": [], "hit": []}
    
    # Adjacency types to test - include knn_outward with different k values and station_outward
    adjacency_configs = [
        {"adjacency_type": "outward", "knn_k": None, "name": "outward"},
        {"adjacency_type": "bidirectional", "knn_k": None, "name": "bidirectional"},
        {"adjacency_type": "full", "knn_k": None, "name": "full"},
        {"adjacency_type": "knn_outward", "knn_k": 3, "name": "knn_outward_k3"},
        {"adjacency_type": "knn_outward", "knn_k": 5, "name": "knn_outward_k5"},
        {"adjacency_type": "knn_outward", "knn_k": 8, "name": "knn_outward_k8"},
        {"adjacency_type": "station_outward", "knn_k": None, "name": "station_outward"},
    ]
    
    # Load one dataset to get total number of events for random sampling
    temp_dataset = CCTrackingDataset(
        dirpath=val_dir,
        inputs=inputs,
        targets=targets,
        num_events=100,  # Just need a small sample to get indices
        adjacency_type="outward",
        self_connections=False,
    )
    
    # Get random event indices (use same indices for all adjacency types)
    max_events = temp_dataset.num_events
    random_indices = np.random.choice(max_events, size=num_events, replace=False)
    print(f"\nSelected event indices: {random_indices}")
    
    del temp_dataset
    
    # Process each adjacency configuration
    for config in adjacency_configs:
        adj_type = config["adjacency_type"]
        knn_k = config["knn_k"]
        folder_name = config["name"]
        
        print(f"\n{'='*60}")
        print(f"Processing: {folder_name}")
        print(f"{'='*60}")
        
        # Create dataset with this adjacency type
        dataset_kwargs = {
            "dirpath": val_dir,
            "inputs": inputs,
            "targets": targets,
            "num_events": -1,  # Load all to access any index
            "adjacency_type": adj_type,
            "self_connections": False,
        }
        if knn_k is not None:
            dataset_kwargs["knn_k"] = knn_k
            
        dataset = CCTrackingDataset(**dataset_kwargs)
        
        # Create subfolder for this adjacency type
        adj_output_dir = output_dir / folder_name
        adj_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot each selected event
        for plot_idx, event_idx in enumerate(random_indices):
            print(f"\nProcessing event {event_idx} ({plot_idx + 1}/{num_events})...")
            
            # Get the event
            inputs_dict, targets_dict = dataset[event_idx]
            
            # Extract coordinates (reverse the 0.001 scaling)
            x_coords = inputs_dict["hit_spacePoint_globEdgeLowX"].squeeze(0).numpy() * 1000
            y_coords = inputs_dict["hit_spacePoint_globEdgeLowY"].squeeze(0).numpy() * 1000
            r_values = inputs_dict["hit_r"].squeeze(0).numpy() * 1000  # Also scale r back
            
            # Get validity mask
            valid_mask = targets_dict["hit_valid"].squeeze(0).numpy()
            
            # Get the adjacency matrix (cc_adjacency is the selected type)
            adjacency = targets_dict["cc_adjacency"].squeeze(0)
            
            # Create title and filename
            display_name = folder_name.replace("_", " ").title()
            title = f"{display_name} Adjacency - Event {event_idx}"
            save_path = adj_output_dir / f"event_{event_idx:05d}.pdf"
            
            # Plot
            plot_adjacency_graph(
                adjacency_matrix=adjacency,
                x_coords=x_coords,
                y_coords=y_coords,
                r_values=r_values,
                valid_mask=valid_mask,
                title=title,
                save_path=str(save_path),
            )
        
        del dataset
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
