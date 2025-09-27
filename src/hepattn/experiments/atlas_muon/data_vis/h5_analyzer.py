"""
Lightning-based data analysis utilities for muon tracking data.
"""
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
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

    def _calculate_detector_technology_statistics(self) -> dict:
        """
        Calculate detector technology statistics from the dataset.
        
        Returns:
        --------
        dict: Statistics for each detector technology including counts and percentages
        """
        technology_mapping = {"MDT": 0, "RPC": 2, "TGC": 3, "STGC": 4, "MM": 5}

        # Initialize counters
        tech_true_hits = {tech: 0 for tech in technology_mapping.keys()}
        tech_total_hits = {tech: 0 for tech in technology_mapping.keys()}
        total_true_hits = 0
        total_hits = 0
        
        try:
            test_dataloader = self.data_module.test_dataloader()
            
            for i, batch in tqdm(enumerate(test_dataloader), desc="Processing technology statistics", total=self.max_events):
                if i >= self.max_events:
                    break
                    
                inputs, targets = batch
                
                # Get hit technology information
                hit_technologies = inputs["hit_spacePoint_technology"][0].numpy()
                hit_valid = targets["hit_valid"][0].numpy()
                hit_on_valid_particle = targets["hit_on_valid_particle"][0].numpy()
                
                # Only consider valid hits
                valid_hit_mask = hit_valid.astype(bool)
                hit_technologies = hit_technologies[valid_hit_mask]
                hit_on_valid_particle = hit_on_valid_particle[valid_hit_mask]
                
                total_hits += len(hit_technologies)
                total_true_hits += np.sum(hit_on_valid_particle)
                
                # Count hits per technology
                for tech_name, tech_value in technology_mapping.items():
                    tech_mask = hit_technologies == tech_value
                    tech_total_hits[tech_name] += np.sum(tech_mask)
                    tech_true_hits[tech_name] += np.sum(hit_on_valid_particle & tech_mask)
            
            # Calculate statistics
            stats = {}
            for tech_name in technology_mapping.keys():
                true_count = tech_true_hits[tech_name]
                total_count = tech_total_hits[tech_name]
                
                # Percentage of total true hits
                true_percentage = (true_count / total_true_hits * 100) if total_true_hits > 0 else 0.0
                # Percentage of total hits
                total_percentage = (total_count / total_hits * 100) if total_hits > 0 else 0.0
                
                stats[tech_name] = {
                    'true_hits': int(true_count),
                    'total_hits': int(total_count),
                    'true_hits_percentage': true_percentage,
                    'total_hits_percentage': total_percentage
                }
            
            # Add overall statistics
            stats['overall'] = {
                'total_true_hits': int(total_true_hits),
                'total_hits': int(total_hits),
                'events_processed': min(self.max_events, i + 1)
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating detector technology statistics: {e}")
            import traceback
            traceback.print_exc()
            return {}

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
            
            # Add detector technology statistics
            print("Calculating detector technology statistics...")
            tech_stats = self._calculate_detector_technology_statistics()
            
            if tech_stats and 'overall' in tech_stats:
                f.write("DETECTOR TECHNOLOGY STATISTICS\n")
                f.write("=" * 60 + "\n")
                f.write(f"Events processed: {tech_stats['overall']['events_processed']}\n")
                f.write(f"Total hits: {tech_stats['overall']['total_hits']:,}\n")
                f.write(f"Total true hits: {tech_stats['overall']['total_true_hits']:,}\n\n")
                
                f.write("TECHNOLOGY DISTRIBUTION (ABSOLUTE NUMBERS):\n")
                f.write("-" * 50 + "\n")
                for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
                    if tech_name in tech_stats:
                        stats = tech_stats[tech_name]
                        f.write(f"{tech_name}:\n")
                        f.write(f"  Total hits: {stats['total_hits']:,}\n")
                        f.write(f"  True hits: {stats['true_hits']:,}\n\n")
                
                f.write("TECHNOLOGY DISTRIBUTION (PERCENTAGES):\n")
                f.write("-" * 50 + "\n")
                f.write("Percentage of total hits by technology:\n")
                for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
                    if tech_name in tech_stats:
                        stats = tech_stats[tech_name]
                        f.write(f"  {tech_name}: {stats['total_hits_percentage']:.2f}%\n")
                
                f.write("\nPercentage of true hits by technology:\n")
                for tech_name in ["MDT", "RPC", "TGC", "STGC", "MM"]:
                    if tech_name in tech_stats:
                        stats = tech_stats[tech_name]
                        f.write(f"  {tech_name}: {stats['true_hits_percentage']:.2f}%\n")
                
                f.write("\n" + "=" * 60 + "\n")
            else:
                f.write("DETECTOR TECHNOLOGY STATISTICS: Failed to calculate\n")
                f.write("=" * 60 + "\n")
        
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
    
    def generate_feature_histograms_with_categories(
        self, 
        output_dir: Union[str, Path], 
        histogram_settings: Optional[Dict] = None,
        features_to_plot: Optional[List[str]] = None,
        category: str = "standard"
    ) -> Dict[str, bool]:
        """
        Generate HEP ROOT style histograms with different categories.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save histogram plots
        histogram_settings : dict, optional
            Dictionary defining histogram settings for specific features
        features_to_plot : list, optional
            List of feature names to plot. If None, plots all available input features.
        category : str
            Category of histograms to generate:
            - "standard": Original implementation (unchanged)
            - "signal_background": Side-by-side plots of signal vs background
            
        Returns:
        --------
        dict : Dictionary with feature names as keys and success status as values
        """
        if category == "standard":
            # Use the original implementation unchanged
            return self.generate_feature_histograms(output_dir, histogram_settings, features_to_plot)
        elif category == "signal_background":
            return self._generate_signal_background_histograms(output_dir, histogram_settings, features_to_plot)
        else:
            raise ValueError(f"Unknown category: {category}. Must be 'standard' or 'signal_background'")
    
    def _generate_signal_background_histograms(
        self, 
        output_dir: Union[str, Path], 
        histogram_settings: Optional[Dict] = None,
        features_to_plot: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Generate side-by-side histograms comparing signal vs background for input features.
        
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
        
        # Collect data from all batches with signal/background separation
        collected_signal_data = {}
        collected_background_data = {}
        event_count = 0
        
        for batch in tqdm(test_dataloader, desc="Collecting signal/background feature data", total=self.max_events):
            inputs, targets = batch
            
            # Get histogram settings from config
            from .h5_config import HISTOGRAM_SETTINGS
            

            num_particles = np.sum(targets["particle_valid"].numpy())
            # Get number of valid particles and hits
            truth_links = targets["particle_hit_valid"][0][:num_particles, :].numpy()
            # print(truth_links.shape)
            # print(truth_links)
            all_truth = np.full(len(inputs["hit_valid"][0]), -1, dtype=int)  # Default to -1 for background

            for id, truth_link in enumerate(truth_links):
                indices = np.where(truth_link)[0]
                # print()
                all_truth[indices] = id  # Assign the track ID to the corresponding hits
            
            # Create signal and background masks
            signal_mask = all_truth != -1
            background_mask = all_truth == -1
            
            # Process hits (inputs) - only consider features in HISTOGRAM_SETTINGS
            for key, value in inputs.items():
                if key.endswith('_valid') or key not in HISTOGRAM_SETTINGS["hits"]:
                    continue
                    
                if features_to_plot is not None and key not in features_to_plot:
                    continue
                
                # Initialize collections if needed
                if key not in collected_signal_data:
                    collected_signal_data[key] = []
                    collected_background_data[key] = []
                
                # Get valid hits and apply signal/background separation
                if 'hit_valid' in inputs:
                    valid_mask = inputs['hit_valid'][0].cpu().numpy().astype(bool)
                    valid_values = value[0][valid_mask].cpu().numpy()
                    
                    # Separate signal and background
                    signal_values = valid_values[signal_mask]
                    background_values = valid_values[background_mask]
                    
                    if len(signal_values) > 0:
                        collected_signal_data[key].extend(signal_values.flatten())
                    if len(background_values) > 0:
                        collected_background_data[key].extend(background_values.flatten())
            
            event_count += 1
            if event_count >= self.max_events:
                break
        
        # Generate side-by-side histograms for collected data
        for feature_name in collected_signal_data.keys():
            if feature_name not in collected_background_data:
                continue
                
            signal_data = collected_signal_data[feature_name]
            background_data = collected_background_data[feature_name]
            
            if len(signal_data) == 0 and len(background_data) == 0:
                print(f"WARNING: No data collected for feature '{feature_name}'")
                results[feature_name] = False
                continue
            
            # Get histogram settings for this feature
            all_settings = HISTOGRAM_SETTINGS["hits"].copy()
            all_settings.update(HISTOGRAM_SETTINGS["tragets"])  # Fix typo if needed

            if feature_name in all_settings:
                settings = all_settings[feature_name]
            else:
                # Default settings - combine both datasets to determine range
                combined_data = np.concatenate([signal_data, background_data]) if len(signal_data) > 0 and len(background_data) > 0 else (signal_data if len(signal_data) > 0 else background_data)
                settings = {
                    'bins': 50,
                    'range': (np.min(combined_data), np.max(combined_data)),
                    'log_y': True
                }
            
            # Create side-by-side histogram
            success = self._create_signal_background_histogram(
                np.array(signal_data) if len(signal_data) > 0 else np.array([]),
                np.array(background_data) if len(background_data) > 0 else np.array([]),
                feature_name, 
                settings, 
                output_path
            )
            results[feature_name] = success
                
        return results
    
    def _create_signal_background_histogram(
        self, 
        signal_data: np.ndarray, 
        background_data: np.ndarray,
        branch_name: str, 
        settings: Dict, 
        output_dir: Path
    ) -> bool:
        """
        Create side-by-side histograms comparing signal vs background for a given feature.
        
        Parameters:
        -----------
        signal_data : np.ndarray
            Data for signal hits (truth_links != -1)
        background_data : np.ndarray
            Data for background hits (truth_links == -1)
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
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Get settings
            bins = settings.get('bins', 50)
            scale_factor = settings.get('scale_factor', 1.0)
            data_range = settings.get('range', None)
            
            # Apply scale factor
            signal_data_scaled = signal_data / settings.get('scale_factor', 1.0) 
            background_data_scaled = background_data / settings.get('scale_factor', 1.0)
            data_range = settings.get('range', (np.min(np.concatenate([signal_data_scaled, background_data_scaled])), np.max(np.concatenate([signal_data_scaled, background_data_scaled]))))

            # Determine bins
            if bins is None:
                # Integer binning for discrete variables
                all_data = np.concatenate([signal_data_scaled, background_data_scaled])
                min_val = int(np.floor(np.min(all_data)))
                max_val = int(np.ceil(np.max(all_data)))
                bins_array = np.arange(min_val - 0.5, max_val + 1.5, 1)
            else:
                # Use specified number of bins
                bins_array = np.linspace(data_range[0], data_range[1], bins + 1)
            
            # Plot signal histogram
            self._plot_single_histogram(ax1, signal_data_scaled, bins_array, data_range, 
                                       f"Signal - {branch_name}", "lightgreen", settings, "Signal")
            
            # Plot background histogram  
            self._plot_single_histogram(ax2, background_data_scaled, bins_array, data_range,
                                       f"Background - {branch_name}", "lightcoral", settings, "Background")
            
            # Set overall title
            fig.suptitle(f'Signal vs Background Comparison: {branch_name}', fontsize=18, fontweight='bold')
            
            # Set tight layout
            plt.tight_layout()
            
            # Save plot
            output_file = output_dir / f"{branch_name}_signal_background_comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Signal/Background comparison saved: {output_file}")
            print(f"  Signal entries: {len(signal_data_scaled)}")
            print(f"  Background entries: {len(background_data_scaled)}")
            
            return True
            
        except Exception as e:
            print(f"ERROR creating signal/background histogram for '{branch_name}': {str(e)}")
            return False
    
    def _plot_single_histogram(self, ax, data, bins_array, data_range, title, color, settings, data_type):
        """Helper function to plot a single histogram on given axes."""
        # Apply range filter
        if len(data) > 0:
            mask = (data >= data_range[0]) & (data <= data_range[1])
            filtered_data = data[mask]
        else:
            filtered_data = data
        
        # Calculate exclusion counts
        total_entries = len(data)
        entries_in_range = len(filtered_data)
        excluded_below = len(data[data < data_range[0]]) if len(data) > 0 else 0
        excluded_above = len(data[data > data_range[1]]) if len(data) > 0 else 0
        excluded_total = excluded_below + excluded_above
        
        if len(filtered_data) > 0:
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
                color=color,
                edgecolor='black'
            )
            
            # Add statistics
            mean = np.mean(filtered_data)
            std = np.std(filtered_data)
        else:
            # Empty histogram
            n = np.zeros(len(bins_array) - 1)
            mean = 0
            std = 0
        
        # HEP ROOT style formatting
        ax.set_xlabel(title.split(' - ')[1], fontsize=14, fontweight='bold')
        ax.set_ylabel('Entries', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add statistics box (HEP ROOT style)
        stats_text = f'{data_type} Entries: {entries_in_range}\nMean: {mean:.3g}\nStd: {std:.3g}'
        
        # Add exclusion information 
        if excluded_total > 0:
            stats_text += f'\n\nExcluded: {excluded_total}'
            if excluded_below > 0:
                stats_text += f'\n  < {data_range[0]:.3g}: {excluded_below}'
            if excluded_above > 0:
                stats_text += f'\n  > {data_range[1]:.3g}: {excluded_above}'
        
        # Position stats box in upper right
        stats_y_pos = 0.98 if excluded_total == 0 else 0.95
        ax.text(0.72, stats_y_pos, stats_text, 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5),
               fontsize=11, fontfamily='monospace')
        
        # Grid (ROOT style)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Apply log scale for Y-axis
        log_y = settings.get('log_y', True)
        if log_y and len(filtered_data) > 0:
            ax.set_yscale('log')
            if len(n[n > 0]) > 0:
                ax.set_ylim(bottom=max(0.1, np.min(n[n > 0]) * 0.5))
            else:
                ax.set_ylim(bottom=0.1)
        
        # Set axis style
        ax.tick_params(labelsize=12)
        ax.tick_params(direction='in', length=6, width=1)
        
        # Add minor ticks
        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=3, width=0.5)
    
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
