"""
Class for handling the windowed data format of the SMU-Textron Cognitive Load dataset.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

from data.data_loader import BaseDataset, list_json_files, load_json_file
from utils.visualization import save_or_show_plot


class WindowedDataset(BaseDataset):
    """Class to handle loading and processing of the Windowed Samples format."""
    
    def __init__(self, folder_path: str = '.'):
        """Initialize and load the dataset.
        
        Args:
            folder_path: Path to the folder containing the JSON files
        """
        self.folder_path = folder_path
        self.file_list = list_json_files(folder_path)
        self.pilot_data = {}
        self.all_trials = []
        
        print(f"Found {len(self.file_list)} pilot JSON files.")
    
    def load_data(self) -> None:
        """Load all pilot data."""
        self.load_all_pilots()
    
    def load_pilot_data(self, pilot_file: str) -> List[Dict[str, Any]]:
        """Load data for a specific pilot.
        
        Args:
            pilot_file: Filename of the pilot JSON file
            
        Returns:
            List of trials for the pilot
        """
        file_path = os.path.join(self.folder_path, pilot_file)
        return load_json_file(file_path)
    
    def load_all_pilots(self, max_pilots: Optional[int] = None) -> None:
        """Load data for all pilots.
        
        Args:
            max_pilots: Maximum number of pilots to load
        """
        files_to_load = self.file_list[:max_pilots] if max_pilots else self.file_list
        
        for pilot_file in tqdm(files_to_load, desc="Loading pilot data"):
            pilot_id = pilot_file.split('.')[0]
            pilot_data = self.load_pilot_data(pilot_file)
            self.pilot_data[pilot_id] = pilot_data
            self.all_trials.extend([(pilot_id, trial_idx, trial) 
                                  for trial_idx, trial in enumerate(pilot_data)])
        
        print(f"Loaded {len(self.pilot_data)} pilots with {len(self.all_trials)} total trials.")
    
    def explore_data(self) -> None:
        """Print basic statistics and information about the dataset."""
        if not self.pilot_data:
            print("No data loaded. Call load_all_pilots() first.")
            return
        
        total_windows = 0
        tlx_values = []
        window_counts = []
        
        for pilot_id, pilot_data in self.pilot_data.items():
            for trial in pilot_data:
                windows = trial.get('windowed_features', [])
                total_windows += len(windows)
                window_counts.append(len(windows))
                tlx_values.append(trial.get('label', None))
        
        print("\nWindowed Dataset Overview:")
        print(f"Total number of pilots: {len(self.pilot_data)}")
        print(f"Total number of trials: {len(self.all_trials)}")
        print(f"Total number of windows: {total_windows}")
        print(f"Average windows per trial: {np.mean(window_counts):.2f}")
        print(f"TLX range: {min(tlx_values):.2f} - {max(tlx_values):.2f}")
        print(f"Average TLX: {np.mean(tlx_values):.2f}")
    
    def visualize_window_features(
        self, 
        pilot_id: Optional[str] = None,
        trial_idx: int = 0,
        window_idx: int = 0,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize features from a specific window.
        
        Args:
            pilot_id: Pilot ID to visualize
            trial_idx: Trial index to visualize
            window_idx: Window index to visualize
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if not self.pilot_data:
            print("No data loaded. Call load_all_pilots() first.")
            return None
            
        if not pilot_id:
            # Get first pilot ID if none specified
            pilot_id = list(self.pilot_data.keys())[0]
        
        # Get the trial and window
        trial = self.pilot_data[pilot_id][trial_idx]
        window = trial['windowed_features'][window_idx]
        
        # Create figure with subplots
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        
        # Plot PPG data
        ppg_data = window['ppg_input'][0]
        axes[0].plot(ppg_data)
        axes[0].set_title('PPG Signal')
        axes[0].set_xlabel('Samples (64Hz)')
        
        # Plot EDA data
        eda_data = window['eda_input'][0]
        axes[1].plot(eda_data)
        axes[1].set_title('EDA Signal')
        axes[1].set_xlabel('Samples (4Hz)')
        
        # Plot tonic component
        tonic_data = window['tonic_input'][0]
        axes[2].plot(tonic_data)
        axes[2].set_title('Tonic Component')
        axes[2].set_xlabel('Samples (4Hz)')
        
        # Plot acceleration
        accel_data = window['accel_input'][0]
        axes[3].plot(accel_data)
        axes[3].set_title('Acceleration')
        axes[3].set_xlabel('Samples (32Hz)')
        
        # Plot temperature
        temp_data = window['temp_input'][0]
        axes[4].plot(temp_data)
        axes[4].set_title('Temperature')
        axes[4].set_xlabel('Samples (4Hz)')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        save_or_show_plot(save_path, "Window visualization saved")
        
        # Print some statistics
        print(f"Window timestamp: {window['timestamp']} seconds")
        print(f"Window duration: {len(ppg_data)/64:.2f} seconds")
        print(f"Trial TLX: {trial['label']}")
        
        return fig
    
    def get_windows_for_trial(self, pilot_id: str, trial_idx: int) -> List[Dict[str, Any]]:
        """Get all windows for a specific trial.
        
        Args:
            pilot_id: Pilot ID
            trial_idx: Trial index
            
        Returns:
            List of window data dictionaries
        """
        try:
            return self.pilot_data[pilot_id][trial_idx]['windowed_features']
        except (KeyError, IndexError):
            print(f"Trial not found for pilot {pilot_id}, trial index {trial_idx}")
            return []
    
    def extract_window_features(self, window: Dict[str, Any], feature_type: str = 'eng_features') -> np.ndarray:
        """Extract features from a window.
        
        Args:
            window: Window data dictionary
            feature_type: Type of features to extract ('eng_features', 'raw_eng_features', 'ppg', etc.)
            
        Returns:
            Feature vector as numpy array
        """
        feature_map = {
            'eng_features': 'eng_features_input',
            'raw_eng_features': 'raw_eng_features_input',
            'ppg': 'ppg_input',
            'eda': 'eda_input',
            'tonic': 'tonic_input',
            'accel': 'accel_input',
            'temp': 'temp_input'
        }
        
        feature_key = feature_map.get(feature_type, 'eng_features_input')
        
        if feature_type in ['eng_features', 'raw_eng_features']:
            # For engineered features, take only the first 22 values
            return np.array(window[feature_key][:22])
        else:
            # For raw signals, take the entire sequence
            return np.array(window[feature_key][0])


"""
TODO Improvements:
1. Implement signal preprocessing (filtering, normalization, etc.)
2. Add signal quality assessment metrics
3. Implement feature extraction from raw signals
4. Add support for different window sizes and overlap percentages
5. Implement signal visualization with annotations for events
6. Add spectral analysis for physiological signals
7. Implement heart rate variability analysis from PPG signal
8. Add support for synchronizing signals with different sampling rates
9. Implement artifact detection and removal for physiological signals
10. Add support for merging windows into continuous time series
11. Implement batch processing for large datasets
12. Add support for real-time signal processing
13. Implement signal segmentation based on events
14. Add support for multimodal signal analysis
15. Implement signal complexity measures (entropy, fractal dimension, etc.)
16. Add pilot-specific signal normalization
17. Implement adaptive window sizing based on signal characteristics
18. Add missing data handling for signal gaps
"""
