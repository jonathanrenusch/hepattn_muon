"""
ROOT file analysis utilities for muon tracking data.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import uproot

from .config import HISTOGRAM_SETTINGS


class RootAnalyzer:
    """
    A class for analyzing ROOT files containing muon tracking data.
    """

    def __init__(self, root_file_path: str, tree_name: str = "MuonHitDump") -> None:
        """
        Initialize the analyzer with a ROOT file.

        Parameters:
        -----------
        root_file_path : str
            Path to the ROOT file
        tree_name : str
            Name of the tree in the ROOT file
        """
        self.root_file_path = root_file_path
        self.tree_name = tree_name
        self._tree = None
        self._file_info: Optional[Dict[str, Union[List[str], int]]] = None

    def _load_tree(self) -> None:
        """Load the ROOT tree if not already loaded."""
        if self._tree is None:
            with uproot.open(self.root_file_path) as file:
                if self.tree_name not in file:
                    available_trees = [key for key in file.keys() if ";" in key]
                    raise ValueError(
                        f"Tree '{self.tree_name}' not found. Available: {available_trees}"
                    )
                self._tree = file[self.tree_name]

    def get_file_info(self) -> Dict[str, Union[List[str], int]]:
        """Get information about the ROOT file structure."""
        if self._file_info is None:
            with uproot.open(self.root_file_path) as file:
                self._file_info = {
                    "keys": list(file.keys()),
                    "num_entries": file[self.tree_name].num_entries
                    if self.tree_name in file
                    else 0,
                    "branches": list(file[self.tree_name].keys())
                    if self.tree_name in file
                    else [],
                }
        return self._file_info

    def analyze_hits_per_event(
        self, output_plot_path: Optional[str] = None
    ) -> Optional[Dict[int, int]]:
        """
        Analyze the distribution of hits per event.

        Parameters:
        -----------
        output_plot_path : str, optional
            Path to save the plot

        Returns:
        --------
        dict : Dictionary with event numbers as keys and hit counts as values
        """
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]

                # Read required branches
                event_numbers = tree["eventNumber"].array(library="np")
                space_points_x = tree["spacePoint_PositionX"].array(library="np")

                print(f"Found {len(event_numbers)} entries")
                print(
                    f"Event number range: {np.min(event_numbers)} to {np.max(event_numbers)}"
                )

                # Count hits per event
                hits_per_event = {}

                if hasattr(space_points_x, "__len__") and len(space_points_x) > 0:
                    if hasattr(space_points_x[0], "__len__"):
                        # Jagged array case
                        for event_num, x_positions in zip(
                            event_numbers, space_points_x
                        ):
                            if event_num not in hits_per_event:
                                hits_per_event[event_num] = 0
                            hits_per_event[event_num] += len(x_positions)
                    else:
                        # Flat array case
                        hit_counter = Counter(event_numbers)
                        hits_per_event = dict(hit_counter)

                # Create visualization
                self._plot_hits_distribution(hits_per_event, output_plot_path)

                return hits_per_event

        except Exception as e:
            print(f"Error analyzing hits per event: {e}")
            return None

    def _plot_hits_distribution(
        self, hits_per_event: Dict[int, int], output_plot_path: Optional[str] = None
    ) -> None:
        """Create histogram of hits per event distribution."""
        hit_counts = list(hits_per_event.values())
        # bins = np.arange(
        #     np.min(hit_counts) - 0.5,
        #     max(hit_counts) + 1.5,
        #     1,
        # )
        plt.figure(figsize=(10, 6))
        plt.hist(hit_counts, bins=100, alpha=0.7, edgecolor="black")
        plt.xlabel("Number of Hits per Event")
        plt.ylabel("Number of Events")
        plt.title("Distribution of Hits per Event")
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_hits = np.mean(hit_counts)
        std_hits = np.std(hit_counts)
        plt.axvline(
            mean_hits,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_hits:.1f}",
        )
        plt.axvline(
            mean_hits + std_hits,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Mean + σ: {mean_hits + std_hits:.1f}",
        )
        plt.axvline(
            mean_hits - std_hits,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Mean - σ: {mean_hits - std_hits:.1f}",
        )
        plt.legend()

        # Print statistics
        print("\nHits per Event Statistics:")
        print(f"Total events: {len(hits_per_event)}")
        print(f"Mean hits per event: {mean_hits:.2f}")
        print(f"Standard deviation: {std_hits:.2f}")
        print(f"Min hits per event: {np.min(hit_counts)}")
        print(f"Max hits per event: {np.max(hit_counts)}")

        if output_plot_path:
            plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_plot_path}")

        plt.show()

    def analyze_tracks_and_lengths(
        self, output_plot_path: Optional[str] = None
    ) -> Optional[Dict[str, Union[Dict[int, int], List[int]]]]:
        """
        Analyze tracks per event and track lengths.

        Parameters:
        -----------
        output_plot_path : str, optional
            Path to save the plot

        Returns:
        --------
        dict : Dictionary containing analysis results
        """
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]

                # Read required branches
                event_numbers = tree["eventNumber"].array(library="np")
                truth_links = tree["spacePoint_truthLink"].array(library="np")

                print(f"Found {len(event_numbers)} entries")

                # Analyze tracks per event
                tracks_per_event: Dict[int, set] = {}
                for event_num, links in zip(event_numbers, truth_links):
                    if event_num not in tracks_per_event:
                        tracks_per_event[event_num] = set()

                    if hasattr(links, "__len__") and len(links) > 0:
                        valid_links = links[(links >= 0) & (links < 1e6)]
                        unique_track_ids = np.unique(valid_links)
                        for track_id in unique_track_ids:
                            tracks_per_event[event_num].add(track_id)

                tracks_per_event_counts = {
                    event: len(tracks) for event, tracks in tracks_per_event.items()
                }

                # Analyze track lengths
                all_event_track_lengths = []
                for links in truth_links:
                    if hasattr(links, "__len__") and len(links) > 0:
                        valid_links = links[(links >= 0) & (links < 1e6)]
                        if len(valid_links) > 0:
                            unique_tracks, track_hit_counts = np.unique(
                                valid_links, return_counts=True
                            )
                            all_event_track_lengths.extend(track_hit_counts)

                # Create visualization
                self._plot_tracks_analysis(
                    tracks_per_event_counts, all_event_track_lengths, output_plot_path
                )

                return {
                    "tracks_per_event": tracks_per_event_counts,
                    "track_lengths": all_event_track_lengths,
                    "track_counts": list(tracks_per_event_counts.values()),
                }

        except Exception as e:
            print(f"Error in tracks analysis: {e}")
            return None

    def _plot_tracks_analysis(
        self,
        tracks_per_event_counts: Dict[int, int],
        all_event_track_lengths: List[int],
        output_plot_path: Optional[str] = None,
    ) -> None:
        """Create side-by-side plots for tracks analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot tracks per event
        track_counts = list(tracks_per_event_counts.values())

        # Create bins that are properly aligned with integer values
        if len(track_counts) > 0:
            min_tracks = min(track_counts)
            max_tracks = max(track_counts)
            # Create bins from min-0.5 to max+0.5 with step 1 to center bars on integers
            bins_tracks = np.arange(min_tracks - 0.5, max_tracks + 1.5, 1)
        else:
            bins_tracks = np.arange(-0.5, 1.5, 1)

        ax1.hist(
            track_counts,
            bins=bins_tracks,
            alpha=0.7,
            edgecolor="black",
            color="lightgreen",
        )
        ax1.set_xlabel("Number of Tracks per Event")
        ax1.set_ylabel("Number of Events")
        ax1.set_title("Distribution of Tracks per Event")
        ax1.grid(True, alpha=0.3)

        # Set x-axis to show integer ticks
        if len(track_counts) > 0:
            ax1.set_xticks(range(min(track_counts), max(track_counts) + 1))

        mean_tracks = np.mean(track_counts)
        ax1.axvline(
            mean_tracks,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_tracks:.1f}",
        )
        ax1.legend()

        # Plot track lengths
        if len(all_event_track_lengths) > 0:
            min_length = min(all_event_track_lengths)
            max_length = max(all_event_track_lengths)
            # Create bins that are properly aligned with integer values
            bins_lengths = np.arange(min_length - 0.5, max_length + 1.5, 1)

            ax2.hist(
                all_event_track_lengths,
                bins=bins_lengths,
                alpha=0.7,
                edgecolor="black",
                color="orange",
            )
            ax2.set_xlabel("Track Length (Hits per Track)")
            ax2.set_ylabel("Number of Track Segments")
            ax2.set_title("Distribution of Track Lengths\n(Within Individual Events)")
            ax2.grid(True, alpha=0.3)

            # Set x-axis to show integer ticks (but limit to reasonable range)
            if max_length - min_length <= 20:
                ax2.set_xticks(range(min_length, max_length + 1))
            else:
                # For large ranges, use automatic ticking but ensure integers
                ax2.set_xticks(
                    range(
                        min_length,
                        max_length + 1,
                        max(1, (max_length - min_length) // 10),
                    )
                )

            mean_length = np.mean(all_event_track_lengths)
            ax2.axvline(
                mean_length,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_length:.1f}",
            )
            ax2.legend()

        plt.tight_layout()

        if output_plot_path:
            plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_plot_path}")

        plt.show()

    def generate_branch_histograms(
        self, output_dir: Union[str, Path], 
        histogram_settings: Optional[Dict] = None
    ) -> Dict[str, bool]:
        """
        Generate HEP ROOT style histograms for all branches defined in HISTOGRAM_SETTINGS.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save histogram plots
        histogram_settings : dict, optional
            Dictionary defining histogram settings. If None, uses HISTOGRAM_SETTINGS from config.
            
        Returns:
        --------
        dict : Dictionary with branch names as keys and success status as values
        """
        if histogram_settings is None:
            histogram_settings = HISTOGRAM_SETTINGS
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        try:
            with uproot.open(self.root_file_path) as file:
                tree = file[self.tree_name]
                available_branches = list(tree.keys())
                
                print(f"Available branches: {len(available_branches)}")
                print(f"Branches to plot: {len(histogram_settings)}")
                
                for branch_name, settings in histogram_settings.items():
                    try:
                        if branch_name not in available_branches:
                            print(f"WARNING: Branch '{branch_name}' not found in tree")
                            results[branch_name] = False
                            continue
                            
                        # Read branch data
                        print(f"Processing branch: {branch_name}")
                        branch_data = tree[branch_name].array(library="np")
                        
                        # Flatten if jagged array
                        if hasattr(branch_data, 'tolist') and any(hasattr(item, '__len__') and not isinstance(item, str) for item in branch_data[:10]):
                            # This is a jagged array, flatten it
                            flattened_data = []
                            for item in branch_data:
                                if hasattr(item, '__len__') and not isinstance(item, str):
                                    flattened_data.extend(item)
                                else:
                                    flattened_data.append(item)
                            data_to_plot = np.array(flattened_data)
                        else:
                            data_to_plot = np.array(branch_data)
                        
                        # Remove invalid values (NaN, inf)
                        data_to_plot = data_to_plot[np.isfinite(data_to_plot)]
                        
                        if len(data_to_plot) == 0:
                            print(f"WARNING: No valid data for branch '{branch_name}'")
                            results[branch_name] = False
                            continue
                            
                        # Generate histogram
                        success = self._create_hep_style_histogram(
                            data_to_plot, 
                            branch_name, 
                            settings, 
                            output_path
                        )
                        results[branch_name] = success
                        
                    except Exception as e:
                        print(f"ERROR processing branch '{branch_name}': {str(e)}")
                        results[branch_name] = False
                        
        except Exception as e:
            print(f"ERROR reading ROOT file: {str(e)}")
            return {branch: False for branch in histogram_settings.keys()}
            
        return results
    
    def _create_hep_style_histogram(
        self, 
        data: np.ndarray, 
        branch_name: str, 
        settings: Dict, 
        output_dir: Path
    ) -> bool:
        """
        Create a HEP ROOT style histogram for a given branch.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to plot
        branch_name : str
            Name of the branch (used for labeling)
        settings : dict
            Dictionary containing 'bins' and 'range' settings
        output_dir : Path
            Directory to save the plot
            
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        try:
            # Set HEP/ROOT style
            plt.style.use('default')  # Reset to default style first
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get settings
            bins = settings.get('bins', 50)
            data_range = settings.get('range', (np.min(data), np.max(data)))
            
            # Apply range filter and calculate exclusions
            mask = (data >= data_range[0]) & (data <= data_range[1])
            filtered_data = data[mask]
            
            # Calculate exclusion counts
            total_entries = len(data)
            entries_in_range = len(filtered_data)
            excluded_below = len(data[data < data_range[0]])
            excluded_above = len(data[data > data_range[1]])
            excluded_total = excluded_below + excluded_above
            
            if len(filtered_data) == 0:
                print(f"WARNING: No data in range {data_range} for branch '{branch_name}'")
                return False
            
            # Determine bins
            if bins is None:
                # Integer binning for discrete variables
                min_val = int(np.floor(np.min(filtered_data)))
                max_val = int(np.ceil(np.max(filtered_data)))
                bins_array = np.arange(min_val - 0.5, max_val + 1.5, 1)
            else:
                # Use specified number of bins
                bins_array = np.linspace(data_range[0], data_range[1], bins + 1)
            
            # Create histogram with HEP ROOT style
            n, bins_edges, patches = ax.hist(
                filtered_data,
                bins=bins_array,
                histtype='step',  # ROOT style outline
                linewidth=0.5,
                color='black',
                alpha=0.8
            )
            
            # Fill histogram with light color
            ax.hist(
                filtered_data,
                bins=bins_array,
                alpha=0.3,
                color='lightblue',
                edgecolor='black'
            )
            
            # HEP ROOT style formatting
            ax.set_xlabel(branch_name, fontsize=14, fontweight='bold')
            ax.set_ylabel('Entries', fontsize=14, fontweight='bold')
            ax.set_title(f'Distribution of {branch_name}', fontsize=16, fontweight='bold', pad=20)
            
            # Add statistics box (HEP ROOT style)
            entries = len(filtered_data)
            mean = np.mean(filtered_data)
            std = np.std(filtered_data)
            
            # Create comprehensive statistics text
            stats_text = f'Entries: {entries}\nMean: {mean:.3g}\nStd: {std:.3g}'
            
            # Add exclusion information 
            stats_text += f'\n\nExcluded: {excluded_total}'
            if excluded_below > 0:
                stats_text += f'\n  < {data_range[0]:.3g}: {excluded_below}'
            if excluded_above > 0:
                stats_text += f'\n  > {data_range[1]:.3g}: {excluded_above}'

            
            # Position stats box in upper right (adjust position based on content size)
            stats_y_pos = 0.98 if excluded_total == 0 else 0.95  # Move down slightly if exclusions shown
            ax.text(0.72, stats_y_pos, stats_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5),
                   fontsize=11, fontfamily='monospace')
            
            # Grid (ROOT style)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Apply log scale for Y-axis (default True, can be overridden in settings)
            log_y = settings.get('log_y', True)  # Default to log scale for Y-axis
            
            if log_y:
                ax.set_yscale('log')
                # For log scale, ensure we have positive values
                if len(n[n > 0]) > 0:
                    ax.set_ylim(bottom=max(0.1, np.min(n[n > 0]) * 0.5))
                else:
                    ax.set_ylim(bottom=0.1)
                print(f"  Applied log Y-scale for '{branch_name}'")
            else:
                print(f"  Using linear Y-scale for '{branch_name}'")
            
            # Set axis style
            ax.tick_params(labelsize=12)
            ax.tick_params(direction='in', length=6, width=1)
            
            # Add minor ticks
            ax.minorticks_on()
            ax.tick_params(which='minor', direction='in', length=3, width=0.5)
            
            # Set tight layout
            plt.tight_layout()
            
            # Save plot
            output_file = output_dir / f"{branch_name}_histogram.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print summary with exclusion information
            if excluded_total > 0:
                exclusion_pct = (excluded_total / total_entries) * 100
                print(f"✓ Histogram saved: {output_file}")
                print(f"  Entries: {entries_in_range}/{total_entries} ({exclusion_pct:.1f}% excluded)")
                if excluded_below > 0:
                    print(f"    Below range: {excluded_below}")
                if excluded_above > 0:
                    print(f"    Above range: {excluded_above}")
            else:
                print(f"✓ Histogram saved: {output_file}")
                print(f"  Entries: {entries_in_range} (no exclusions)")
            
            return True
            
        except Exception as e:
            print(f"ERROR creating histogram for '{branch_name}': {str(e)}")
            return False
