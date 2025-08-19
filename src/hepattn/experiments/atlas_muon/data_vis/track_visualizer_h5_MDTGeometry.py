"""
Track visualization utilities for muon tracking data.
"""
import typing
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm 
import h5py


from ..data import AtlasMuonDataset, AtlasMuonDataModule

BACKGROUND_LINEWIDTH = 0.3  # Width of background lines
TRACK_LINEWIDTH = 0.5


class h5TrackVisualizerMDTGeometry:
    """
    A class for visualizing muon tracks from ROOT files.
    """

    def __init__(self, dataset: AtlasMuonDataset) -> None:
        """
        Initialize the visualizer with a ROOT file.

        Parameters:
        -----------
        root_file_path : str
            Path to the ROOT file
        tree_name : str
            Name of the tree in the ROOT file
        """
        self.dataset = dataset

        # Define saturated colors for up to 8 tracks
        self.track_colors = [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FF00FF",
            "#00FFFF",
            "#FFFF00",
            "#FF8000",
            "#8000FF",
        ]

    def plot_muon_tracks(
        self, event_index: int, show_plot: bool = True, just_background: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot ATLAS Muon spectrometer tracks in three 2D projections (X-Y, Z-Y, Z-X).

        Parameters:
        -----------
        event_index : int
            Event index to visualize (0-based, corresponds to tree entry)
        show_plot : bool
            Whether to display the plot (default: True)

        Returns:
        --------
        matplotlib.figure.Figure : The figure object containing the plots
        """
        # try:
            # Check if event index is valid
        if event_index < 0 or event_index >= len(self.dataset):
            print(
                f"Event index {event_index} is out of range. Available entries: 0 to {len(self.dataset) - 1}"
            )
            return None

        # # Load event data using the dataset's __getitem__ method
        # hits, particles, num_hits, num_tracks = self.dataset.load_event(event_index)

        # # Extract the data we need for visualization
        # # Convert back to mm for display (dataset converts to meters)
        # all_high_x = hits["spacePoint_globEdgeHighX"] * 1000  # Convert back to mm
        # all_high_y = hits["spacePoint_globEdgeHighY"] * 1000
        # all_high_z = hits["spacePoint_globEdgeHighZ"] * 1000
        # all_low_x = hits["spacePoint_globEdgeLowX"] * 1000
        # all_low_y = hits["spacePoint_globEdgeLowY"] * 1000
        # all_low_z = hits["spacePoint_globEdgeLowZ"] * 1000

        # all_truth = hits["spacePoint_truthLink"]

        # # Get truth muon parameters
        # truthMuon_phi = particles["truthMuon_phi"]
        # truthMuon_eta = particles["truthMuon_eta"]

        # # Use event_index as event_number since we don't have eventNumber in the dataset
        # event_number = event_index

        # Load the event data using the dataset's __getitem__ method
        inputs, targets = self.dataset.__getitem__(event_index)
        # print(targets.keys())
        # print(targets["particle_valid"])
        num_particles = np.sum(targets["particle_valid"].numpy())
        # print("targets phi", targets["particle_truthMuon_phi"])
        # print("targets eta", targets["particle_truthMuon_eta"])
        all_high_x = inputs["hit_spacePoint_globEdgeHighX"][0].numpy() * 1000  # Convert back to mm
        all_high_y = inputs["hit_spacePoint_globEdgeHighY"][0].numpy() * 1000
        all_high_z = inputs["hit_spacePoint_globEdgeHighZ"][0].numpy() * 1000
        all_low_x = inputs["hit_spacePoint_globEdgeLowX"][0].numpy() * 1000
        all_low_y = inputs["hit_spacePoint_globEdgeLowY"][0].numpy() * 1000
        all_low_z = inputs["hit_spacePoint_globEdgeLowZ"][0].numpy() * 1000

        print(targets["particle_hit_valid"].shape)
        truth_links = targets["particle_hit_valid"][0][:num_particles, :].numpy()
        print(truth_links.shape)
        print(truth_links)
        all_truth = np.full(len(all_high_x), -1, dtype=int)  # Default to -1 for background

        for id, truth_link in enumerate(truth_links):
            indices = np.where(truth_link)[0]
            print()
            all_truth[indices] = id  # Assign the track ID to the corresponding hits

        
        truthMuon_phi = targets["particle_truthMuon_phi"][:num_particles].numpy()
        truthMuon_eta = targets["particle_truthMuon_eta"][:num_particles].numpy()

        print(
            f"Event {event_index} (index {event_index}): Found {len(all_high_x)} space points"
        )
        print(f"X range: [{np.min(all_high_x):.1f}, {np.max(all_high_x):.1f}] mm")
        print(f"Y range: [{np.min(all_high_y):.1f}, {np.max(all_high_y):.1f}] mm")
        print(f"Z range: [{np.min(all_high_z):.1f}, {np.max(all_high_z):.1f}] mm")

        # Create the visualization
        
        # Create the visualization
        fig = self._create_track_plots(
            all_high_x,
            all_high_y,
            all_high_z,
            all_low_x,
            all_low_y,
            all_low_z,
            all_truth,
            truthMuon_phi,
            truthMuon_eta,
            event_index,
            show_plot,
            just_background=just_background,
        )

            # create corresponding background plot

        return fig

        # except Exception as e:
        #     print(f"Error plotting tracks: {e}")
        #     return None



    def _create_track_plots(
        self,
        all_high_x: np.ndarray,
        all_high_y: np.ndarray,
        all_high_z: np.ndarray,
        all_low_x: np.ndarray,
        all_low_y: np.ndarray,
        all_low_z: np.ndarray,
        all_truth: np.ndarray,
        truthMuon_phi: np.ndarray,
        truthMuon_eta: np.ndarray,
        event_number: int,
        show_plot: bool = True,
        just_background: bool = False,
    ) -> plt.Figure:
        """Create the three 2D projection plots."""
        # Separate background and track hits
        if just_background:
            background_mask = np.ones(len(all_truth), dtype=bool)
        else: 
            background_mask = all_truth == -1
            
        track_mask = all_truth >= 0

        print(f"Background hits: {np.sum(background_mask)}")
        print(f"Track hits: {np.sum(track_mask)}")

        # Debug: Show all unique truth values
        unique_truth_values = np.unique(all_truth)
        print(f"All truth values: {unique_truth_values}")

        # Create figure with pyramid layout - X-Y centered on top, Z-Y and Z-X equal width on bottom
        fig = plt.figure(figsize=(16, 10))

        # Create subplots in pyramid style: X-Y centered on top, equal Z projections on bottom
        # Reduced hspace for more compact layout, increased wspace to prevent label overlap
        gs = fig.add_gridspec(
            2,
            4,
            height_ratios=[1, 1],
            width_ratios=[1, 1, 1, 1],
            hspace=0.25,
            wspace=0.35,
        )

        # X-Y plot centered on top row (takes middle 2 columns)
        ax_xy = fig.add_subplot(gs[0, 1:3])

        # Z-Y plot on bottom left (takes 2 columns)
        ax_zy = fig.add_subplot(gs[1, 0:2])

        # Z-X plot on bottom right (takes 2 columns)
        ax_zx = fig.add_subplot(gs[1, 2:4])

        axes = [ax_xy, ax_zy, ax_zx]

        # Plot background hits first (without legend)
        if np.sum(background_mask) > 0:
            self._plot_background_hits(axes, 
                                       all_high_x=all_high_x, 
                                       all_high_y=all_high_y, 
                                       all_high_z=all_high_z, 
                                       all_low_x=all_low_x, 
                                       all_low_y=all_low_y, 
                                       all_low_z=all_low_z, background_mask=background_mask)

        # Plot each track in a different color (only add legend to first subplot)

        if np.sum(track_mask) > 0 and not just_background:
            self._plot_track_hits(
                axes,
                all_high_x,
                all_high_y,
                all_high_z,
                all_low_x,
                all_low_y,
                all_low_z,
                all_truth,
                truthMuon_phi,
                truthMuon_eta,
                track_mask,
            )

        # Set labels and formatting
        self._format_subplots(axes, event_number)

        # Create a single legend positioned outside the plots
        if not just_background:
            self._create_single_legend(fig, axes, all_truth, track_mask, background_mask)

        plt.tight_layout()

        if show_plot:
            plt.show()

        return fig

    def _plot_background_hits(
        self,
        axes: typing.List[plt.Axes],
        all_high_x: np.ndarray,
        all_high_y: np.ndarray,
        all_high_z: np.ndarray,
        all_low_x: np.ndarray,
        all_low_y: np.ndarray,
        all_low_z: np.ndarray,
        background_mask: np.ndarray,
    ) -> None:

        """Plot background hits on all three projections without legend."""
        for low_x, low_y, low_z, high_x, high_y, high_z in zip(
            all_low_x[background_mask],
            all_low_y[background_mask],
            all_low_z[background_mask],
            all_high_x[background_mask],
            all_high_y[background_mask],
            all_high_z[background_mask],
        ):
            axes[0].plot(
                [low_x, high_x], [low_y, high_y], color="gray", alpha=0.3, linewidth=BACKGROUND_LINEWIDTH   
            )
            axes[1].plot(
                [low_z, high_z], [low_y, high_y], color="gray", alpha=0.3, linewidth=BACKGROUND_LINEWIDTH
            )
            axes[2].plot(
                [low_z, high_z], [low_x, high_x], color="gray", alpha=0.3, linewidth=BACKGROUND_LINEWIDTH
            )

    

    def _plot_track_hits(
        self,
        axes: typing.List[plt.Axes],
        all_high_x: np.ndarray,
        all_high_y: np.ndarray,
        all_high_z: np.ndarray,
        all_low_x: np.ndarray,
        all_low_y: np.ndarray,
        all_low_z: np.ndarray,
        all_truth: np.ndarray,
        truthMuon_phi: np.ndarray,
        truthMuon_eta: np.ndarray,
        track_mask: np.ndarray,
        line_length: float = 14000,
        line_length_z: float = 25000,
    ) -> None:
        """Plot detector elements with different colors for each track without individual legends."""
        unique_tracks = np.unique(all_truth[track_mask])

        print(f"Unique track IDs: {unique_tracks}")

        for i, track_id in enumerate(unique_tracks):
            track_color = self.track_colors[i % len(self.track_colors)]
            track_points = all_truth == track_id

            # Count hits for this track
            num_hits: int = np.sum(track_points)
            print(f"  Track {track_id}: {num_hits} hits, color: {track_color}")

            # Plot on all three projections without individual legends
            for x_low, y_low, z_low, x_high, y_high, z_high in zip(
                all_low_x[track_points],
                all_low_y[track_points],
                all_low_z[track_points],
                all_high_x[track_points],
                all_high_y[track_points],
                all_high_z[track_points],
            ):
                axes[0].plot([x_low, x_high], [y_low, y_high], color=track_color, alpha=0.9, linewidth=TRACK_LINEWIDTH)
                axes[1].plot([z_low, z_high], [y_low, y_high], color=track_color, alpha=0.9, linewidth=TRACK_LINEWIDTH)
                axes[2].plot([z_low, z_high], [x_low, x_high], color=track_color, alpha=0.9, linewidth=TRACK_LINEWIDTH)

            # plotting the line at a fixed angle
            phi = truthMuon_phi[0][i]
            # Calculate end point
            x1 = line_length * np.cos(phi)
            y1 = line_length * np.sin(phi)
            
            # Calculate the Z-X and Z-Y projections
            eta_value = truthMuon_eta[0][i]
            theta = 2 * np.arctan(np.exp(-eta_value))
            z1 = line_length_z * np.cos(theta)
            x1_zplane = line_length * np.sin(theta) * np.cos(phi)
            y1_zplane = line_length * np.sin(theta) * np.sin(phi)
            # ...existing code...

            # ...existing code...
            # plotting the line at a fixed angle
            axes[0].plot([0, x1], [0, y1], color=track_color, linewidth=1, alpha=0.5)
            axes[1].plot([0, z1], [0, y1_zplane], color=track_color, linewidth=1, alpha=0.5)
            axes[2].plot([0, z1], [0, x1_zplane], color=track_color, linewidth=1, alpha=0.5)


        # Print track statistics
        self._print_track_stats(unique_tracks, all_truth)


    def _print_track_stats(
        self, unique_tracks: np.ndarray, all_truth: np.ndarray
    ) -> None:
        """Print statistics about the tracks."""
        print("\nTrack statistics:")
        print(f"Number of unique tracks: {len(unique_tracks)}")
        for i, track_id in enumerate(unique_tracks):
            track_hits: int = np.sum(all_truth == track_id)
            color_used = self.track_colors[i % len(self.track_colors)]
            print(f"  Track {track_id}: {track_hits} hits (color: {color_used})")

        if len(unique_tracks) > len(self.track_colors):
            print(
                f"  Warning: {len(unique_tracks)} tracks found, but only {len(self.track_colors)} colors available."
            )
            print(
                f"  Colors will repeat for tracks beyond the first {len(self.track_colors)}."
            )

    def _format_subplots(self, axes: typing.List[plt.Axes], event_number: int) -> None:
        """Format the three subplots with proper labels and styling for pyramid layout."""
        # X-Y plane (top plot) - ensure it's perfectly square with fixed axis ranges
        axes[0].set_xlabel("X [mm]")
        axes[0].set_ylabel("Y [mm]")
        axes[0].set_title(f"X-Y Plane (Event {event_number})")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect("equal", adjustable="box")  # Force square aspect ratio
        axes[0].set_xlim([-14000, 14000])
        axes[0].set_ylim([-14000, 14000])

        # Z-Y plane (bottom left) with fixed axis ranges
        axes[1].set_xlabel("Z [mm]")
        axes[1].set_ylabel("Y [mm]")
        axes[1].set_title(f"Z-Y Plane (Event {event_number})")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([-25000, 25000])
        axes[1].set_ylim([-14000, 14000])

        # Z-X plane (bottom right) with fixed axis ranges
        axes[2].set_xlabel("Z [mm]")
        axes[2].set_ylabel("X [mm]")
        axes[2].set_title(f"Z-X Plane (Event {event_number})")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([-25000, 25000])
        axes[2].set_ylim([-14000, 14000])

    def _create_single_legend(
        self,
        fig: plt.Figure,
        axes: typing.List[plt.Axes],
        all_truth: np.ndarray,
        track_mask: np.ndarray,
        background_mask: np.ndarray,
    ) -> None:
        """Create a single legend positioned outside the plots."""
        legend_elements = []

        # Add background entry if background hits exist
        if np.sum(background_mask) > 0:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markersize=8,
                    alpha=0.6,
                    label="Background",
                )
            )

        # Add track entries
        if np.sum(track_mask) > 0:
            unique_tracks = np.unique(all_truth[track_mask])
            for i, track_id in enumerate(unique_tracks):
                track_color = self.track_colors[i % len(self.track_colors)]
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="x",
                        color="w",
                        markerfacecolor=track_color,
                        markeredgecolor=track_color,
                        markersize=10,
                        markeredgewidth=2,
                        label=f"Track {track_id}",
                    )
                )

        # Position the legend close to the centered upper X-Y plot
        fig.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.85, 0.92),
            fontsize=11,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        # Adjust subplot parameters for more compact spacing
        fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08)

    def plot_and_save_event(
        self, event_index: int, save_path: Optional[str] = None, just_background: bool = False
    ) -> Optional[plt.Figure]:
        """
        Plot tracks for a specific event and optionally save the figure.

        Parameters:
        -----------
        event_index : int
            Event index to plot (0-based)
        save_path : str, optional
            Path to save the figure (if None, only displays)

        Returns:
        --------
        matplotlib.figure.Figure : The figure object
        """
        # ...existing code...
        fig = self.plot_muon_tracks(event_index)

        if fig is not None and save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")
        # Background plots
        fig_background = self.plot_muon_tracks(event_index, just_background=True)
        # Save background plot if requested
        # if fig_background is not None and save_path is not None:
        #     print("This is save path!", save_path)
        #     save_path = save_path.replace(".png", "_background.png")
        #     fig_background.savefig(save_path, dpi=300, bbox_inches="tight")
        #     # Modify save path for background plot
        #     print(f"Figure saved to: {save_path}")
        if fig_background is not None and save_path is not None:
            save_path_bg = str(save_path).replace(".png", "_background.png")
            fig_background.savefig(save_path_bg, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path_bg}")


        return fig

    # ...existing code...

    def plot_and_save_true_hits_histogram(
        self,
        dataloader: DataLoader,
        max_events: Optional[int] = None,
        num_bins: int = 100,
        num_events: int = 1000,
        show_plot: bool = True,
        save_path: Optional[str] = None,  # Change from Optional[Path] to Optional[str]
    ) -> Optional[plt.Figure]:
        # ...existing code...
        """
        Create a histogram showing the distribution of true hits per event.

        Parameters:
        -----------
        max_events : int, optional
            Maximum number of events to process. If None, process all events.
        num_bins : int, optional
            Number of bins for the histogram
        show_plot : bool, optional
            Whether to display the plot
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        matplotlib.figure.Figure : The figure object containing the histogram
        """
        try:
            true_hits_counts = []
            n_0_hits = 0
            for i, batch in tqdm(enumerate(dataloader), desc="Processing events", total=num_events):
                if i >= num_events:
                    break
                inputs, targets = batch
                # Extract truth links from the hits data
                true_hits_counts.append(np.sum(targets["hit_on_valid_particle"].numpy()))
                if np.sum(targets["hit_on_valid_particle"].numpy()) == 0:
                    n_0_hits += 1 

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            bins = np.arange(
                np.min(true_hits_counts) - 0.5,
                max(true_hits_counts) + 1.5,
                1,
            )
            # Create histogram
            n, bins, patches = ax.hist(
                true_hits_counts,
                bins=bins,
                alpha=0.7,
                color="blue",
                edgecolor="black",
            )

            # Add labels and title
            ax.set_xlabel("Number of True Hits per Event")
            ax.set_ylabel("Frequency")
            ax.set_title(
                f"Distribution of True Hits per Event\n(Analysis of {num_events} events)"
            )

            # Add grid and statistics
            ax.grid(alpha=0.3)

            # Add text with statistics
            mean_hits: float = float(np.mean(true_hits_counts))
            median_hits: float = float(np.median(true_hits_counts))
            max_hits: int = np.max(true_hits_counts)
            min_hits: int = np.min(true_hits_counts)

            stats_text = (
                f"Statistics:\n"
                f"Mean: {mean_hits:.1f}\n"
                f"Median: {median_hits:.1f}\n"
                f"Min: {min_hits}\n"
                f"Max: {max_hits}"
                f"\nTotal events: {num_events}"
                f"\nEvents with 0 hits: {n_0_hits} ({n_0_hits / num_events * 100:.1f}%)"
            )

            # Position text in the upper right
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                # bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

            # Save figure if path is provided
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Histogram saved to: {save_path}")

            # Show plot if requested
            if show_plot:
                plt.show()

            return fig

        except Exception as e:
            print(f"Error plotting true hits histogram: {e}")
            import traceback

            traceback.print_exc()
            return None
