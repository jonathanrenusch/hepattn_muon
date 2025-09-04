"""
Lightning-based data analysis utilities for muon tracking data.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ..data import AtlasMuonDataModule


from .config import HISTOGRAM_SETTINGS


class h5Analyzer:
    """
    A class for analyzing muon tracking data using the AtlasMuonDataModule.
    """

    def __init__(self, data_module: AtlasMuonDataModule, max_events: int = 1000) -> None:
        """
        Initialize the analyzer with an AtlasMuonDataModule.

        Parameters:
        -----------
        data_module : AtlasMuonDataModule
            The Lightning data module containing the dataset
        """
        self.data_module = data_module
        self.max_events = max_events

    def analyze_hits_per_event(
        self, output_plot_path: Optional[str] = None, 
    ) -> Optional[Dict[int, int]]:
        """
        Analyze the distribution of hits per event using the testing dataset.

        Parameters:
        -----------
        output_plot_path : str, optional
            Path to save the plot
        max_events : int
            Maximum number of events to analyze

        Returns:
        --------
        dict : Dictionary with event numbers as keys and hit counts as values
        """
        # try:
        test_dataloader = self.data_module.test_dataloader()
        
        hits_per_event = {}
        event_count = 0

        for batch in tqdm(test_dataloader, desc="Analyzing hits per event", total=self.max_events):
            inputs, targets = batch
            hit_valid = inputs["hit_valid"]
            num_hits = torch.sum(hit_valid).item()
            hits_per_event[event_count] = num_hits
            event_count += 1
            if event_count >= self.max_events:
                break

        self._plot_hits_distribution(hits_per_event, output_plot_path)
        return hits_per_event

        # except Exception as e:
        #     print(f"Error analyzing hits per event: {e}")
        #     return None

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
        
        # Close the figure to free memory
        plt.close()

        # plt.show()

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
        # try:
        test_dataloader = self.data_module.test_dataloader()
        
        tracks_per_event = {}
        all_event_track_lengths = []
        event_count = 0
        
        print(f"Analyzing tracks and lengths from validation dataset...")
        for batch in tqdm(test_dataloader, total=self.max_events, desc="Analyzing tracks"):
            inputs, targets = batch
            # print("inputs", inputs.keys())
            # print("target_keys", targets.keys())
            # print("Hit on valid particle:", targets["hit_on_valid_particle"].shape)
            # print("hit_valid:", inputs["hit_valid"].shape)
            # print("particle_valid:", targets["particle_valid"].shape)
            # print("particle_hit_valid:", targets["particle_hit_valid"].shape)
            # print("hit_on_valid_particle:", targets["hit_on_valid_particle"].shape)
            # # print("sample_id:", inputs["sample_id"].shape)
            # # print("particle_hit_valid:", torch.sum(targets["particle_hit_valid"]))
            # print("number of true hits via the particle_hit_valid:", torch.sum(targets["particle_hit_valid"], dim=2))
            # print("number of true hits via the hit_on_particle:", torch.sum(targets["hit_on_valid_particle"]).item())
            # print("number of particles via the particle_valid:", torch.sum(targets["particle_valid"]).item())
            # print("inputs", inputs)
            # print("targets", targets["particle_valid"])

            # Count tracks per event
            # checking for number of unique tracks

            tracks_per_event[event_count] = torch.sum(targets["particle_valid"]).item()
            
            all_event_track_lengths.extend((torch.sum(targets["particle_hit_valid"], dim=2)[targets["particle_valid"]]).tolist())
            if torch.sum(targets["particle_hit_valid"]).item() != torch.sum(targets["hit_on_valid_particle"]).item():
                print(
                    f"Warning: Mismatch in hit counts for event {event_count}: "
                    f"particle_hit_valid={torch.sum(targets['particle_hit_valid'])}, "
                    f"Number of particles={torch.sum(targets['particle_valid'])},"
                    f"hit_on_valid_particle={torch.sum(targets['hit_on_valid_particle'])}, "
                    f"hit_valid={torch.sum(inputs['hit_valid'])}"
                )
                break
            event_count += 1
            if event_count >= self.max_events:
                break
            # Create visualization
        
        self._plot_tracks_analysis(
            tracks_per_event, all_event_track_lengths, output_plot_path
        )

        return {
            "tracks_per_event": tracks_per_event,
            "track_lengths": all_event_track_lengths,
            "track_counts": list(tracks_per_event.values()),
        }

        # except Exception as e:
        #     print(f"Error in tracks analysis: {e}")
        #     return None

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

        # Save track statistics to file
        self._save_track_statistics(track_counts, all_event_track_lengths, output_plot_path)
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
            # Note: Track length statistics are already saved in _save_track_statistics method
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
        
        # Close the figure to free memory
        plt.close()

        # plt.show()

    def _save_track_statistics(self, track_counts: List[int], all_event_track_lengths: List[int], output_plot_path: Optional[str] = None) -> None:
        """
        Save comprehensive track statistics to a text file.
        
        Parameters:
        -----------
        track_counts : List[int]
            Number of tracks per event
        all_event_track_lengths : List[int]
            Length of each individual track
        output_plot_path : str, optional
            Path where the plot is saved, used to determine where to save the statistics file
        """
        from datetime import datetime
        
        # Determine output directory and filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_plot_path:
            # If plot path is provided, save the txt file in the same directory
            plot_path = Path(output_plot_path)
            output_dir = plot_path.parent
            # Create filename based on plot name but with .txt extension and timestamp
            base_name = plot_path.stem
            filename = output_dir / f"{base_name}_statistics_{timestamp}.txt"
        else:
            # Default to current directory if no plot path provided
            filename = f"track_statistics_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRACK ANALYSIS STATISTICS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Max events analyzed: {self.max_events}\n\n")
            
            # Event statistics
            total_events = len(track_counts)
            f.write("EVENT STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total events analyzed: {total_events}\n\n")
            
            # Tracks per event statistics
            f.write("TRACKS PER EVENT DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            track_counter = Counter(track_counts)
            
            # Calculate percentages and write detailed breakdown
            for num_tracks in sorted(track_counter.keys()):
                count = track_counter[num_tracks]
                percentage = (count / total_events) * 100
                f.write(f"{num_tracks} tracks: {count} events ({percentage:.2f}%)\n")
            
            f.write(f"\nMean tracks per event: {np.mean(track_counts):.2f}\n")
            f.write(f"Standard deviation: {np.std(track_counts):.2f}\n")
            f.write(f"Min tracks per event: {min(track_counts)}\n")
            f.write(f"Max tracks per event: {max(track_counts)}\n\n")
            
            # Track length statistics
            f.write("TRACK LENGTH STATISTICS:\n")
            f.write("-" * 35 + "\n")
            total_tracks = len(all_event_track_lengths)
            tracks_short = np.sum(np.array(all_event_track_lengths) < 3)
            tracks_short_percentage = (tracks_short / total_tracks) * 100
            
            f.write(f"Total number of tracks: {total_tracks}\n")
            f.write(f"Tracks with length < 3 hits: {tracks_short}\n")
            f.write(f"Percentage of tracks with < 3 hits: {tracks_short_percentage:.2f}%\n")
            f.write(f"Tracks with length > 3 hits: {total_tracks - tracks_short}\n")
            f.write(f"Percentage of tracks with > 3 hits: {100 - tracks_short_percentage:.2f}%\n\n")
            
            f.write(f"Mean track length: {np.mean(all_event_track_lengths):.2f} hits\n")
            f.write(f"Standard deviation: {np.std(all_event_track_lengths):.2f} hits\n")
            f.write(f"Min track length: {min(all_event_track_lengths)} hits\n")
            f.write(f"Max track length: {max(all_event_track_lengths)} hits\n\n")
            
            # Track length distribution
            f.write("TRACK LENGTH DISTRIBUTION:\n")
            f.write("-" * 35 + "\n")
            length_counter = Counter(all_event_track_lengths)
            for length in sorted(length_counter.keys()):
                count = length_counter[length]
                percentage = (count / total_tracks) * 100
                f.write(f"{length} hits: {count} tracks ({percentage:.2f}%)\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"\nTrack statistics saved to: {filename}")
        print(f"Summary:")
        print(f"  - Total events: {total_events}")
        print(f"  - Total tracks: {total_tracks}")
        print(f"  - Tracks with ≤3 hits: {tracks_short} ({tracks_short_percentage:.1f}%)")
        print(f"  - Events with each track count: {dict(track_counter)}")

    def generate_feature_histograms(
        self, 
        output_dir: Union[str, Path], 
        histogram_settings: Optional[Dict] = None,
        features_to_plot: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Generate HEP ROOT style histograms for input features using the test dataloader.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save histogram plots
        histogram_settings : dict, optional
            Dictionary defining histogram settings for specific features
        features_to_plot : list, optional
            List of feature names to plot. If None, plots all available input features.
            
        Returns:
        --------
        dict : Dictionary with feature names as keys and success status as values
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Get test dataloader
        test_dataloader = self.data_module.test_dataloader()
        # print(histogram_settings)
        # Collect data from all batches
        collected_data = {}
        event_count = 0
        
        for batch in tqdm(test_dataloader, desc="Collecting feature data", total=self.max_events):
            inputs, targets = batch
            
            # Get histogram settings from config
            from .h5_config import HISTOGRAM_SETTINGS
            
            # Process hits (inputs)
            for key, value in inputs.items():
                if key.endswith('_valid') or key not in HISTOGRAM_SETTINGS["hits"]:
                    continue
                    
                if features_to_plot is not None and key not in features_to_plot:
                    continue
                    
                if key not in collected_data:
                    collected_data[key] = []
                
                # Use hit_valid mask for all input features
                if 'hit_valid' in inputs:
                    valid_mask = inputs['hit_valid']
                    valid_values = value[valid_mask]
                    if len(valid_values) > 0:
                        collected_data[key].extend(valid_values.cpu().numpy().flatten())

            # Process targets (fix the typo "tragets" -> "targets")
            for key, value in targets.items():
                if key.endswith('_valid') or key not in HISTOGRAM_SETTINGS["tragets"]:
                    continue
                    
                if features_to_plot is not None and key not in features_to_plot:
                    continue
                    
                if key not in collected_data:
                    collected_data[key] = []
                
                # Use particle_valid mask for particle features
                if 'particle_valid' in targets:
                    valid_mask = targets['particle_valid']
                    valid_values = value[valid_mask]
                    if len(valid_values) > 0:
                        collected_data[key].extend(valid_values.cpu().numpy().flatten())
            
            event_count += 1
            if event_count >= self.max_events:
                break
        # Generate histograms for collected data
        for feature_name, data_list in collected_data.items():
            # try:
            if len(data_list) == 0:
                print(f"WARNING: No data collected for feature '{feature_name}'")
                results[feature_name] = False
                continue
            
            data_array = np.array(data_list)
            
            # Remove invalid values (NaN, inf)
            data_array = data_array[np.isfinite(data_array)]
            
            if len(data_array) == 0:
                print(f"WARNING: No valid data for feature '{feature_name}'")
                results[feature_name] = False
                continue
            
            # Get histogram settings for this feature
            all_settings = HISTOGRAM_SETTINGS["hits"].copy()
            all_settings.update(HISTOGRAM_SETTINGS["tragets"])  # Fix typo if needed

            if feature_name in all_settings:
                settings = all_settings[feature_name]
            else:
                # Default settings
                settings = {
                    'bins': 50,
                    'range': (np.min(data_array), np.max(data_array)),
                    'log_y': True
                }
            # print(histogram_settings)
            # Create histogram using the same method
            success = self._create_hep_style_histogram(
                data_array, 
                feature_name, 
                settings, 
                output_path
            )
            results[feature_name] = success
                
            # except Exception as e:
            #     print(f"ERROR processing feature '{feature_name}': {str(e)}")
            #     results[feature_name] = False

        # if "particle_truthMuon_pt" in collected_data and "particle_truthMuon_q" in collected_data:
        #     feature_name = "q over p_t"
        #     pt_values = np.array(collected_data["particle_truthMuon_pt"])
        #     q_values = np.array(collected_data["particle_truthMuon_q"])
            # data_array = q_values / pt_values
            # print(data_array)
            # print(data_array.shape)
            # print(np.unique(data_array))
            # success = self._create_hep_style_histogram(
            #     data_array,
            #     feature_name,
            #     {"bins": 50,
            #     "range": (np.min(data_array), np.max(data_array)),
            #     "log_y": True},
            #     output_path
            # )
            # # Process pt and q values together
            # pass

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
            data = data / settings.get('scale_factor', 1.0)  # Apply scale factor if provided
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
