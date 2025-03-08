import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm
import time
import cProfile
import pstats
from functools import wraps
import io
import multiprocessing as mp
from functools import partial
from pathlib import Path
import re
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import LogNorm, Normalize
import traceback


def organize_experiment_files(experiment_number, original_data_file, original_json_file, fitted_params_file, base_output_dir):
    """
    Organize experiment files by copying them to a dedicated experiment directory.
    """
    # Create experiment-specific directory
    experiment_dir = os.path.join(base_output_dir, f"experiment_{experiment_number}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Define new file paths - now directly in the experiment directory
    new_data_file = os.path.join(experiment_dir, os.path.basename(original_data_file))
    new_json_file = os.path.join(experiment_dir, os.path.basename(original_json_file))
    new_params_file = os.path.join(experiment_dir, os.path.basename(fitted_params_file))
    
    # Copy files to new location
    import shutil
    shutil.copy2(original_data_file, new_data_file)
    shutil.copy2(original_json_file, new_json_file)
    
    # For the fitted params, we might need to move rather than copy if it's temporary
    if os.path.exists(fitted_params_file):
        shutil.move(fitted_params_file, new_params_file)
    
    print(f"\nOrganized experiment files in: {experiment_dir}")
    print(f"Copied:")
    print(f"  - Original data file")
    print(f"  - JSON parameters file")
    print(f"  - Fitted parameters file")
    
    return new_data_file, new_json_file, new_params_file

def get_experiment_number(filename):
    """Extract experiment number from filename with format '2D_ODMR_scan_{number}.npy'"""
    pattern = r'2D_ODMR_scan_(\d+)\.npy$'
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None

def process_directory(directory, method='trf', output_dir='./fitted_parameters', use_multi_dip=False, n_pairs=1):
    """
    Process all ODMR data files in a directory
    
    Args:
        directory: Directory containing ODMR data (.npy and .json pairs)
        method: Curve fitting method ('trf' or 'lm')
        output_dir: Output directory for fitted parameters
        use_multi_dip: Whether to use multi-dip fitting
        n_pairs: Number of dip pairs to fit if using multi-dip fitting
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .npy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy') and 'ODMR_scan' in f]
    
    if not npy_files:
        print(f"No ODMR scan files found in {directory}")
        return
    
    print(f"Found {len(npy_files)} potential ODMR data files.")
    
    # Process each file if corresponding JSON exists
    success_count = 0
    for npy_file in npy_files:
        base_name = os.path.splitext(npy_file)[0]
        json_file = os.path.join(directory, base_name + '.json')
        npy_path = os.path.join(directory, npy_file)
        
        if not os.path.exists(json_file):
            print(f"Skipping {npy_file} - no matching JSON file found.")
            continue
        
        print(f"\nProcessing {npy_file}")
        try:
            # Initialize analyzer for this file
            analyzer = ODMRAnalyzer(npy_path, json_file, enable_profiling=False)
            
            # Create experiment-specific subdirectory
            exp_output_dir = os.path.join(output_dir, f"experiment_{base_name}")
            os.makedirs(exp_output_dir, exist_ok=True)

            print(f"Using multi-dip fitting with {n_pairs} pair(s) ({n_pairs*2} dips)")
            output_dir, fitted_params_file = analyzer.fit_multi_lorentzian_with_fixed_center(
                n_pairs=n_pairs,
                method=method,
                output_dir=exp_output_dir
            )

            
            if fitted_params_file:
                success_count += 1
                print(f"Successfully processed {npy_file}")
            else:
                print(f"Processing {npy_file} completed but no output file was generated.")
                
        except Exception as e:
            print(f"Error processing {npy_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\nBatch processing complete. Successfully processed {success_count}/{len(npy_files)} files.")
    print(f"Results saved in {output_dir}")

def timing_decorator(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

def calculate_fit_quality(original_data, fitted_curve):
    """
    Calculate a single quality metric for ODMR fit assessment.
    Returns a value between 0 (poor fit) and 1 (perfect fit).
    
    The metric is: 1 - NRMSE (Normalized Root Mean Square Error)
    This gives us an intuitive score where:
    - 1.0 means perfect fit
    - 0.0 means the fit is as bad as a flat line at the mean
    - Values around 0.9 or higher typically indicate good fits
    """
    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((original_data - fitted_curve) ** 2))
    
    # Normalize by the range of the data
    data_range = np.ptp(original_data)
    if data_range == 0:
        return 0  # If data is flat, fit quality is 0
        
    nrmse = rmse / data_range
    
    # Convert to a score between 0 and 1
    quality_score = 1 - nrmse
    
    return max(0, quality_score)  # Ensure non-negative score

def print_performance_metrics(total_processed, start_time, speed_samples=None):
    """Print performance metrics based on collected data."""
    end_time = time.time()
    total_time = end_time - start_time
    average_speed = total_processed / total_time if total_time > 0 else 0
    
    # Calculate max speed if samples are provided
    if speed_samples and len(speed_samples) > 0:
        max_speed = max(speed_samples)
    else:
        max_speed = average_speed  # Fall back to average if no samples
    
    # Print the metrics in a clear format
    print("\n" + "="*50)
    print("ODMR PERFORMANCE METRICS")
    print("="*50)
    print(f"Average speed: {average_speed:.2f} pixels/second")
    print(f"Maximum speed: {max_speed:.2f} pixels/second")
    print(f"Total computation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*50)
    
    return {
        "average_speed": average_speed,
        "max_speed": max_speed,
        "total_computation_time": total_time
    }

def process_pixel_row_with_asymmetric_pairs(args):
    """
    Process a row of pixels with asymmetric pairs approach.
    
    Args:
        args: Tuple containing (row_idx, data, freq_axis, n_pairs, default_values, method)
    
    Returns:
        tuple: (row_idx, row_results)
    """
    m, data, freq_axis, n_pairs, default_values, method = args
    M, N, F = data.shape
    
    # Create result structure with all potential parameters
    row_results = {
        "I0": np.zeros(N),
        "f_center": np.zeros(N),
        "quality_score": np.zeros(N)
    }
    
    # Add parameters for each pair
    for i in range(1, n_pairs + 1):
        row_results[f"A_{i}_1"] = np.zeros(N)
        row_results[f"A_{i}_2"] = np.zeros(N)
        row_results[f"width_{i}_1"] = np.zeros(N)
        row_results[f"width_{i}_2"] = np.zeros(N)
        row_results[f"f_d_{i}_1"] = np.zeros(N)
        row_results[f"f_d_{i}_2"] = np.zeros(N)
    
    # Add compatibility parameters
    row_results["A"] = np.zeros(N)
    row_results["width"] = np.zeros(N)
    row_results["f_delta"] = np.zeros(N)
    
    # Process each pixel in the row
    for n in range(N):
        pixel_data = data[m, n, :]
        
        try:
            result = ODMRAnalyzer.fit_pixel_with_asymmetric_pairs(
                pixel_data, freq_axis, n_pairs=n_pairs, 
                method=method
            )
            
            # Copy results to row arrays
            for key, value in result.items():
                if key in row_results:
                    row_results[key][n] = value
        except Exception as e:
            print(f"Error processing pixel ({m}, {n}): {str(e)}")
            # Apply default values for this pixel
            if default_values:
                for key, value in default_values.items():
                    if key in row_results:
                        row_results[key][n] = value
    
    return m, row_results

class ODMRFitChecker:
    def __init__(self, fitted_params_file, original_data_file, json_params_file):
        """Initialize with quality assessment capabilities"""
        # Load the fitted parameters
        self.fitted_params = np.load(fitted_params_file)
        
        # Load original data
        self.original_data = np.load(original_data_file)
        
        # Load frequency parameters
        with open(json_params_file, 'r') as f:
            self.params = json.load(f)
                
        # Create frequency axis
        self.freq_axis = np.linspace(
            self.params['min_freq'] / 1e9,
            self.params['max_freq'] / 1e9,
            self.params['num_measurements']
        )
        
        # Create spatial axes
        self.x_axis = np.linspace(
            self.params['x1'],
            self.params['x2'],
            self.params['x_steps']
        )
        self.y_axis = np.linspace(
            self.params['y1'],
            self.params['y2'],
            self.params['y_steps']
        )

        # Get data dimensions
        self.M, self.N = self.fitted_params.shape[:2]
        
        # Determine number of dip pairs based on parameter count
        param_count = self.fitted_params.shape[2]
        # Determine number of dip pairs based on parameter count
        param_count = self.fitted_params.shape[2]
        self.n_pairs = (param_count - 3) // 6  # I0, f_center, quality_score, and 6 params per pair

        print(f"Detected {self.n_pairs} dip pairs from parameter count ({param_count})")

        # Define parameter indices for easier reference
        self.param_indices = {
            'I0': 0,
            'f_center': 1
        }

        # Add indices for pair-specific parameters
        for i in range(1, self.n_pairs + 1):
            base_idx = 2 + (i-1) * 6
            self.param_indices[f'A_{i}_1'] = base_idx
            self.param_indices[f'A_{i}_2'] = base_idx + 1
            self.param_indices[f'width_{i}_1'] = base_idx + 2
            self.param_indices[f'width_{i}_2'] = base_idx + 3
            self.param_indices[f'f_d_{i}_1'] = base_idx + 4
            self.param_indices[f'f_d_{i}_2'] = base_idx + 5
        
        print(f"Detected {self.n_pairs} dip pairs from parameter count ({param_count})")
        
        # Extract quality scores (last parameter in the stack)
        self.quality_scores = self.fitted_params[:, :, -1]
        
        # Calculate success statistics
        self.success_rate = np.mean(self.quality_scores >= 0.9) * 100
        self.mean_quality = np.mean(self.quality_scores)
        
        self.pl_map = np.mean(self.original_data, axis=2)
        # Calculate baseline (maximum value for each pixel)
        baseline = np.max(self.original_data, axis=2)
        # Calculate minimum value for each pixel
        min_val = np.min(self.original_data, axis=2)
        # Calculate contrast as percentage: (max-min)/max * 100
        self.raw_contrast = (baseline - min_val) / baseline * 100

        # Store the fitting parameters and quality scores separately
        # Last column is quality score
        self.fitting_params = self.fitted_params[:, :, :-1]  # Extract fitting parameters
        self.quality_scores = self.fitted_params[:, :, -1]   # Extract quality scores

        # Add new parameters for neighborhood averaging
        self.enable_averaging = False  # Toggle for neighborhood averaging
        self.quality_threshold = 0.80  # Quality score threshold
        
        # Create cached averaged data dictionary
        self.averaged_data = {}
        
        # Define parameter indices for easier reference
        self.param_indices = {
            'I0': 0,
            'f_center': 1
        }
        
        # Add indices for pair-specific parameters
        for i in range(1, self.n_pairs + 1):
            base_idx = 2 + (i-1) * 5
            self.param_indices[f'A_{i}_1'] = base_idx
            self.param_indices[f'A_{i}_2'] = base_idx + 1
            self.param_indices[f'width_{i}_1'] = base_idx + 2
            self.param_indices[f'width_{i}_2'] = base_idx + 3
            self.param_indices[f'f_delta_{i}'] = base_idx + 4
        
        # Define visualization options
        self.viz_options = {
            'Fit Quality': {
                'data': self.quality_scores,
                'cmap': 'RdYlGn',
                'label': 'Fit Quality Score (0-1)'
            },
            'PL Map': {
                'data': self.pl_map,
                'cmap': 'viridis',
                'label': 'PL Intensity (a.u.)'
            },
            'Raw Contrast': {
                'data': self.raw_contrast,
                'cmap': 'viridis',
                'label': 'Raw Contrast (%)'
            },
            'Fit Contrast': {
                'data': self.calculate_total_contrast(),
                'cmap': 'viridis',
                'label': 'Fitted Contrast (%)'
            },
            'Peak Splitting': {
                'data': self.get_outermost_splitting() * 1000,  # Convert from GHz to MHz
                'cmap': 'magma',
                'label': 'Peak Splitting (MHz)'  # MHz units
            },
            'Frequency Shift': {
                'data': self.fitting_params[:, :, self.param_indices['f_center']],
                'cmap': 'magma',
                'label': 'Center Frequency (GHz)'
            },
            'Baseline (I0)': {
                'data': self.fitting_params[:, :, self.param_indices['I0']],
                'cmap': 'viridis',
                'label': 'Baseline Intensity (a.u.)'
            }
        }
        
        # Add amplitude and width visualizations for each pair
        for i in range(1, self.n_pairs + 1):
            self.viz_options[f'Amplitude Pair {i}'] = {
                'data': (self.fitting_params[:, :, self.param_indices[f'A_{i}_1']] + 
                        self.fitting_params[:, :, self.param_indices[f'A_{i}_2']]) / 2,
                'cmap': 'viridis',
                'label': f'Dip Amplitude Pair {i} (a.u.)'
            }
            self.viz_options[f'Width Pair {i}'] = {
                'data': (self.fitting_params[:, :, self.param_indices[f'width_{i}_1']] + 
                        self.fitting_params[:, :, self.param_indices[f'width_{i}_2']]) / 2,
                'cmap': 'plasma',
                'label': f'Dip Width Pair {i} (GHz)'
            }
            self.viz_options[f'Splitting Pair {i}'] = {
                'data': self.fitting_params[:, :, self.param_indices[f'f_delta_{i}']] * 1000,  # MHz
                'cmap': 'magma',
                'label': f'Splitting Pair {i} (MHz)'
            }

        # Then initialize log_scale_states
        self.log_scale_states = {key: False for key in self.viz_options.keys()}

        self._update_timer = None  # For debouncing
        self.averaged_data = {}    # For caching averaged data

    def calculate_total_contrast(self):
        """Calculate total contrast from all dip pairs"""
        total_amplitude = np.zeros((self.M, self.N))
        baseline = self.fitting_params[:, :, self.param_indices['I0']]
        
        # Sum the average amplitude from each pair
        for i in range(1, self.n_pairs + 1):
            amp_1 = self.fitting_params[:, :, self.param_indices[f'A_{i}_1']]
            amp_2 = self.fitting_params[:, :, self.param_indices[f'A_{i}_2']]
            avg_amp = (amp_1 + amp_2) / 2
            total_amplitude += avg_amp
        
        # Calculate contrast percentage
        total_contrast = total_amplitude / baseline * 100
        return total_contrast
    
    def get_outermost_splitting(self):
        """Get the splitting of the outermost dip pair"""
        if self.n_pairs == 1:
            return self.fitting_params[:, :, self.param_indices['f_delta_1']]
        
        # For multiple pairs, find the largest splitting for each pixel
        max_splitting = np.zeros((self.M, self.N))
        
        for m in range(self.M):
            for n in range(self.N):
                # Get all splitting values for this pixel
                splittings = [self.fitting_params[m, n, self.param_indices[f'f_delta_{i}']] 
                            for i in range(1, self.n_pairs + 1)]
                # Use the maximum splitting
                max_splitting[m, n] = max(splittings)
        
        return max_splitting

    def multi_lorentzian(self, f, params):
        """
        Calculate multi-Lorentzian function based on the parameter structure in fitted_params.
        This method creates a model based on the independent dips parameters but organizes
        them as symmetric pairs around the global center frequency.
        
        Args:
            f: Frequency values
            params: Array of parameters as stored in fitted_params
            
        Returns:
            Combined signal with all Lorentzian dips
        """
        # Extract baseline intensity
        I0 = params[self.param_indices['I0']]
        # Start with baseline intensity
        result = I0
        
        # Get center frequency
        f_center = params[self.param_indices['f_center']]
        
        # Process each pair
        for i in range(1, self.n_pairs + 1):
            try:
                # Get parameters for this pair
                A_i_1 = params[self.param_indices[f'A_{i}_1']]
                A_i_2 = params[self.param_indices[f'A_{i}_2']]
                width_i_1 = params[self.param_indices[f'width_{i}_1']]
                width_i_2 = params[self.param_indices[f'width_{i}_2']]
                f_delta_i = params[self.param_indices[f'f_delta_{i}']]
                
                # Calculate positions of dips
                f_1 = f_center - f_delta_i/2
                f_2 = f_center + f_delta_i/2
                
                # Calculate and subtract dips
                dip_1 = A_i_1 / (1 + ((f_1 - f) / width_i_1) ** 2)
                dip_2 = A_i_2 / (1 + ((f_2 - f) / width_i_2) ** 2)
                
                # Subtract dips from baseline
                result -= (dip_1 + dip_2)
                
            except Exception as e:
                # Log error but continue with the remaining pairs
                print(f"Error processing pair {i} in multi_lorentzian: {str(e)}")
                continue
        
        return result
    
    def calculate_neighborhood_average(self, original_data, x, y):
        """
        Calculate neighborhood average using a more robust approach.
        When threshold is high, we use the best available neighbors instead of all neighbors.
        """
        height, width = original_data.shape
        neighbors = []
        
        # Collect all valid neighbors and their quality scores
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width:
                    neighbors.append({
                        'value': original_data[nx, ny],
                        'quality': self.quality_scores[nx, ny]
                    })
        
        if not neighbors:
            return original_data[x, y]
            
        # Sort neighbors by quality score
        neighbors.sort(key=lambda x: x['quality'], reverse=True)
        
        # If we have any neighbors above threshold, use only those
        good_neighbors = [n['value'] for n in neighbors if n['quality'] >= self.quality_threshold]
        if good_neighbors:
            return np.mean(good_neighbors)
        
        # If threshold is high and no good neighbors, use top 3 best quality neighbors
        if self.quality_threshold > 0.95:
            top_neighbors = [n['value'] for n in neighbors[:3]]
            return np.mean(top_neighbors)
        
        # Otherwise, return original value
        return original_data[x, y]

    def get_averaged_data(self, data_key):
        """
        Apply averaging to pixels with low quality scores.
        Never average the quality scores themselves.
        """
        # Return original data if averaging is off or if showing quality scores
        if not self.enable_averaging or data_key == 'Fit Quality':
            return self.viz_options[data_key]['data']
        
        # Get the data we want to average
        original_data = self.viz_options[data_key]['data']
        result = original_data.copy()
        
        # Find all pixels with low quality scores
        height, width = original_data.shape
        for x in range(height):
            for y in range(width):
                if self.quality_scores[x, y] < self.quality_threshold:
                    # Replace low quality pixels with neighborhood average
                    result[x, y] = self.calculate_neighborhood_average(original_data, x, y)
        
        return result

    def update_data_ranges(self):
        """Calculate and store data ranges for plotting"""
        # Initialize data ranges and margins
        self.y_min = np.min(self.original_data)
        self.y_max = np.max(self.original_data)
        self.y_range = self.y_max - self.y_min
        self.y_margin = self.y_range * 0.3
        
        # Store global ranges for consistent scaling
        self.global_y_min = np.min(self.original_data)
        self.global_y_max = np.max(self.original_data)
        self.global_y_range = self.global_y_max - self.global_y_min
        self.y_margin_factor = 0.05  # Factor for dynamic margin calculation

    def initialize_plots(self):
        """Initialize all plot elements"""
        # Create spectrum plot
        y_data = self.original_data[self.x_idx, self.y_idx]
        self.spectrum_line, = self.ax_data.plot(self.freq_axis, y_data, 'b.', label='Data')
        
        # Create fit line
        params = self.fitting_params[self.x_idx, self.y_idx]
        fitted_curve = self.multi_lorentzian(self.freq_axis, params)
        self.fit_line, = self.ax_data.plot(self.freq_axis, fitted_curve, 'r-', label='Fit')
        
        # Set initial axis limits
        self.ax_data.set_xlim(self.freq_axis[0], self.freq_axis[-1])
        self.ax_data.set_ylim(self.y_min - self.y_margin, self.y_max + self.y_margin)
        
        # Create map visualization
        viz_data = self.get_averaged_data(self.current_viz)
        self.map_img = self.ax_map.imshow(
            viz_data.T,
            origin='lower',
            extent=[
                self.params['x1'],  # Left edge
                self.params['x2'],  # Right edge
                self.params['y1'],  # Bottom edge
                self.params['y2']   # Top edge
            ],
            cmap=self.viz_options[self.current_viz]['cmap'],
            aspect='equal'
        )
        x_pos = self.x_axis[self.x_idx]
        y_pos = self.y_axis[self.y_idx]
        self.pixel_marker, = self.ax_map.plot(x_pos, y_pos, 'rx')
                
        # Add colorbar
        self.colorbar = plt.colorbar(self.map_img, ax=self.ax_map)
        self.colorbar.set_label(self.viz_options[self.current_viz]['label'])

    def create_interactive_viewer(self):
        """Create interactive viewer with improved update handling"""
        # Set up the figure with tight_layout for better spacing
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.set_tight_layout(False)
        plt.subplots_adjust(bottom=0.25, left=0.2)
        
        # Create grid and subplots
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        self.ax_data = self.fig.add_subplot(gs[0])
        self.ax_map = self.fig.add_subplot(gs[1])
        
        # Initialize state variables
        self.x_idx, self.y_idx = 0, 0
        self.full_range = True
        self.current_viz = 'Fit Quality'
        self.local_scaling = True
        
        # Calculate initial ranges and create plots
        self.update_data_ranges()
        self.initialize_plots()
        
        # Set labels and titles
        self.ax_data.set_xlabel('Frequency (GHz)')
        self.ax_data.set_ylabel('ODMR Signal (a.u.)')
        self.ax_data.legend()
        self.ax_map.set_xlabel('X Position')
        self.ax_map.set_ylabel('Y Position')
        
        # Create UI elements with improved positioning
        self._create_sliders()
        self._create_radio_buttons()
        self._create_control_buttons()
        
        # Connect mouse click event for map interaction
        self.fig.canvas.mpl_connect('button_press_event', self._on_map_click)
        
        # Enable keyboard navigation
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Connect figure close event
        self.fig.canvas.mpl_connect('close_event', self.handle_close)
        
        # Initial update
        self.force_update()
        
        # Show the plot
        plt.show()
    
    def update_axis_limits(self, y_data, fitted_curve, params):
        """Update axis limits based on current settings"""
        # Y-axis limits
        local_min = min(np.min(y_data), np.min(fitted_curve))
        local_max = max(np.max(y_data), np.max(fitted_curve))
        local_range = local_max - local_min
        y_center = (local_max + local_min) / 2
        
        if self.local_scaling:
            y_margin = local_range * self.y_margin_factor
            self.ax_data.set_ylim(
                y_center - local_range/2 - y_margin,
                y_center + local_range/2 + y_margin
            )
        else:
            y_margin = self.global_y_range * self.y_margin_factor
            self.ax_data.set_ylim(
                self.global_y_min - y_margin,
                self.global_y_max + y_margin
            )
        
        # X-axis limits
        if self.full_range:
            self.ax_data.set_xlim(self.freq_axis[0], self.freq_axis[-1])
        else:
            f_center = params[self.param_indices['f_center']]
            
            # Calculate the maximum splitting from all pairs
            max_f_delta = 0
            for i in range(1, self.n_pairs + 1):
                f_delta = params[self.param_indices[f'f_delta_{i}']]
                max_f_delta = max(max_f_delta, f_delta)
                
            # Get the maximum width as well
            max_width = 0
            for i in range(1, self.n_pairs + 1):
                width_1 = params[self.param_indices[f'width_{i}_1']]
                width_2 = params[self.param_indices[f'width_{i}_2']]
                max_width = max(max_width, width_1, width_2)
            
            # Use the larger of width*4 or max_delta*1.5 for margin
            x_margin = max(max_width * 4, max_f_delta * 1.5)
            self.ax_data.set_xlim(f_center - x_margin, f_center + x_margin)

    def _create_sliders(self):
        """Create sliders with improved update handling"""
        ax_x = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_y = plt.axes([0.2, 0.05, 0.6, 0.03])
        ax_threshold = plt.axes([0.2, 0.15, 0.6, 0.03])
        
        self.x_slider = Slider(ax_x, 'X', 0, self.M-1, valinit=0, valstep=1)
        self.y_slider = Slider(ax_y, 'Y', 0, self.N-1, valinit=0, valstep=1)
        self.threshold_slider = Slider(
            ax_threshold, 'Quality Threshold', 
            0.0, 1.0, valinit=self.quality_threshold,
            valstep=0.001
        )
        
        # Connect slider events with immediate updates
        self.x_slider.on_changed(self._on_slider_change)
        self.y_slider.on_changed(self._on_slider_change)
        self.threshold_slider.on_changed(self._on_threshold_change)

    def _create_radio_buttons(self):
        """Create radio buttons with improved layout"""
        ax_radio = plt.axes([0.02, 0.25, 0.12, 0.6])  # Slightly wider for more text
        self.viz_radio = RadioButtons(ax_radio, list(self.viz_options.keys()), 
                                    active=list(self.viz_options.keys()).index(self.current_viz))
        
        # Improve radio button appearance
        for label in self.viz_radio.labels:
            label.set_fontsize(7)
        
        self.viz_radio.on_clicked(self._on_viz_change)

    def _create_control_buttons(self):
        """Create control buttons with consistent layout"""
        button_width = 0.12
        button_height = 0.04
        button_left = 0.85
        
        buttons_config = [
                ('log_button', 'Toggle Log Scale', 0.16),  
                ('avg_button', 'Toggle Averaging', 0.11),  
                ('range_button', 'Toggle Range', 0.06),    
                ('scale_button', 'Toggle Scale', 0.01)     
            ]
        
        for attr_name, label, position in buttons_config:
            ax = plt.axes([button_left, position, button_width, button_height])
            button = Button(ax, label)
            setattr(self, attr_name, button)
            # Create a separate function to properly capture the button name
            def make_callback(name):
                return lambda event: self._on_button_click(event, name)
            button.on_clicked(make_callback(attr_name))

    def _on_slider_change(self, val):
        """Handle slider changes with immediate update"""
        self.x_idx = int(self.x_slider.val)
        self.y_idx = int(self.y_slider.val)
        self.force_update()

    def _on_threshold_change(self, val):
        """Handle threshold changes with immediate update"""
        self.quality_threshold = val
        self.averaged_data.clear()
        self.force_update()

    def _on_viz_change(self, label):
        """Handle visualization changes with immediate update"""
        self.current_viz = label
        self.force_update()

    def _on_button_click(self, event, button_name):
        """Handle button clicks with immediate update"""
        if button_name == 'log_button':
            self.log_scale_states[self.current_viz] = not self.log_scale_states[self.current_viz]
        elif button_name == 'avg_button':
            self.enable_averaging = not self.enable_averaging
            self.averaged_data.clear()
        elif button_name == 'range_button':
            self.full_range = not self.full_range
        elif button_name == 'scale_button':
            self.local_scaling = not self.local_scaling
        
        self.force_update()

    def _on_map_click(self, event):
        """Handle mouse clicks on the map with proper coordinate conversion"""
        if event.inaxes == self.ax_map:
            # Convert clicked coordinates to nearest pixel indices
            x_idx = np.argmin(np.abs(self.x_axis - event.xdata))
            y_idx = np.argmin(np.abs(self.y_axis - event.ydata))
            
            if 0 <= x_idx < self.M and 0 <= y_idx < self.N:
                self.x_idx = x_idx
                self.y_idx = y_idx
                self.x_slider.set_val(x_idx)
                self.y_slider.set_val(y_idx)
                self.force_update()

    def update_colorbar(self):
        """Update the colorbar for the current visualization"""
        viz_data = self.get_averaged_data(self.current_viz)
        
        if self.log_scale_states[self.current_viz]:
            # Handle log scale visualization
            from matplotlib.colors import LogNorm
            min_positive = np.min(viz_data[viz_data > 0]) if np.any(viz_data > 0) else 1e-10
            scaled_data = np.maximum(viz_data, min_positive)
            self.map_img.norm = LogNorm(vmin=min_positive, vmax=np.max(scaled_data))
            self.colorbar.set_label(f"{self.viz_options[self.current_viz]['label']} (log scale)")
        else:
            # Handle linear scale visualization
            from matplotlib.colors import Normalize
            self.map_img.norm = Normalize(vmin=np.min(viz_data), vmax=np.max(viz_data))
            self.colorbar.set_label(self.viz_options[self.current_viz]['label'])
        
        # Set the colormap and update the colorbar
        self.map_img.set_cmap(self.viz_options[self.current_viz]['cmap'])
        self.colorbar.update_normal(self.map_img)

    def _on_key_press(self, event):
        """Handle keyboard navigation"""
        if event.key in ['left', 'right', 'up', 'down']:
            x, y = self.x_idx, self.y_idx
            
            if event.key == 'left':
                x = max(0, x - 1)
            elif event.key == 'right':
                x = min(self.M - 1, x + 1)
            elif event.key == 'up':
                y = min(self.N - 1, y + 1)
            elif event.key == 'down':
                y = max(0, y - 1)
            
            self.x_idx, self.y_idx = x, y
            self.x_slider.set_val(x)
            self.y_slider.set_val(y)
            self.force_update()

    def format_parameter_text(self, params, quality_score):
        """Format the parameter text for display, supporting multiple dip pairs"""
        # Always include I0 and f_center
        param_str = f"I0={params[self.param_indices['I0']]:.3f}, "
        param_str += f"f_c={params[self.param_indices['f_center']]:.3f}\n"
        
        # Add dip pairs, one per line
        for i in range(1, self.n_pairs + 1):
            param_str += f"Pair {i}: A1={params[self.param_indices[f'A_{i}_1']]:.3e}, "
            param_str += f"A2={params[self.param_indices[f'A_{i}_2']]:.3e}, "
            param_str += f"w1={params[self.param_indices[f'width_{i}_1']]:.3f}, "
            param_str += f"w2={params[self.param_indices[f'width_{i}_2']]:.3f}, "
            param_str += f"fd={params[self.param_indices[f'f_delta_{i}']]:.3f}\n"
        
        # Add quality score
        param_str += f"quality_score={quality_score:.3f}"
        
        return param_str

    def force_update(self):
        """Force a complete update of all plot elements"""
        try:
            # Update data plot
            y_data = self.original_data[self.x_idx, self.y_idx]
            params = self.fitting_params[self.x_idx, self.y_idx]
            fitted_curve = self.multi_lorentzian(self.freq_axis, params)
            
            self.spectrum_line.set_ydata(y_data)
            self.fit_line.set_ydata(fitted_curve)
            
            # Update map with proper coordinates
            viz_data = self.get_averaged_data(self.current_viz)
            self.map_img.set_data(viz_data.T)

            # Update marker position with real coordinates
            x_pos = self.x_axis[self.x_idx]
            y_pos = self.y_axis[self.y_idx]
            self.pixel_marker.set_data([x_pos], [y_pos])
            
            # Update titles and labels with proper coordinates
            self.ax_data.set_title(
                f'Position: ({x_pos:.3f}, {y_pos:.3f})\n'
                f'Quality Score: {self.quality_scores[self.x_idx, self.y_idx]:.3f}'
            )

            self.ax_map.set_xlabel('X Position (mm)')
            self.ax_map.set_ylabel('Y Position (mm)')
                        
            # Update colorbar
            self.update_colorbar()
            
            # Update parameter display
            quality_score = self.quality_scores[self.x_idx, self.y_idx]
            if quality_score < self.quality_threshold and self.enable_averaging:
                param_str = "Using neighborhood average (low quality fit)"
                # Create a list of averaged parameters
                avg_params = np.zeros_like(params)
                for i, key in enumerate(self.param_indices.keys()):
                    avg_params[i] = self.calculate_neighborhood_average(
                        self.fitting_params[:, :, i], 
                        self.x_idx, 
                        self.y_idx,
                    )
                param_str += "\n" + self.format_parameter_text(avg_params, quality_score)
            else:
                param_str = self.format_parameter_text(params, quality_score)
            
            self.ax_data.set_title(f'Pixel ({self.x_idx}, {self.y_idx})\n{param_str}')
            
            # Update axis limits
            self.update_axis_limits(y_data, fitted_curve, params)
            
            # Update map title
            self.ax_map.set_title(
                f'Success Rate: {self.success_rate:.1f}%\n'
                f'Mean Quality: {self.mean_quality:.3f}'
            )
            
            # Force a complete redraw
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"Error during update: {str(e)}")

    def handle_close(self, event):
        """Clean up resources when the figure is closed"""
        plt.close(self.fig)

class ODMRAnalyzer:
    def __init__(self, data_file, json_file, enable_profiling=False):
        """
        Initialize ODMR analyzer with data and parameters
        
        Args:
            data_file (str): Path to numpy data file
            json_file (str): Path to JSON parameter file
            enable_profiling (bool): Whether to enable profiling functionality
        """
        self.profiling_enabled = enable_profiling
        
        # Store file paths
        self.data_file = data_file
        self.json_file = json_file
        
        # Only create profiler if enabled
        if self.profiling_enabled:
            self.profiler = cProfile.Profile()
        else:
            self.profiler = None

        # Load data directly on initialization
        self.data = np.load(data_file)
        with open(json_file, 'r') as f:
            self.params = json.load(f)

        self.freq_axis = np.linspace(
            self.params['min_freq'] / 1e9,  # Convert to GHz
            self.params['max_freq'] / 1e9,
            self.params['num_measurements']
        )
        self.mean_spectrum = np.mean(self.data, axis=(0, 1))

    def check_fits(self, fitted_params_file):
        """
        Launch the interactive fit checker for the current experiment
        
        Args:
            fitted_params_file (str): Path to the saved fitted parameters
        """
        checker = ODMRFitChecker(fitted_params_file, self.data_file, self.json_file)
        checker.create_interactive_viewer()

    #separate method for loading the data to see how much time is spent doing that    
    @timing_decorator
    def load_data(self, data_file, json_file):
        """Load data and parameters with timing measurement"""
        self.data = np.load(data_file)
        with open(json_file, 'r') as f:
            self.params = json.load(f)
        
        self.freq_axis = np.linspace(
            self.params['min_freq'] / 1e9,
            self.params['max_freq'] / 1e9,
            self.params['num_measurements']
        )
        self.mean_spectrum = np.mean(self.data, axis=(0, 1))

    #methods for profiling
    def start_profiling(self):
        """Start profiling if enabled"""
        if self.profiling_enabled and self.profiler:
            self.profiler.enable()

    def stop_profiling(self, verbose=False):
        """Stop profiling and optionally print results if enabled"""
        if self.profiling_enabled and self.profiler:
            self.profiler.disable()
            if verbose:
                s = io.StringIO()
                stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
                stats.print_stats()
                print(s.getvalue())

    @staticmethod
    def find_outermost_dips(freq_axis, pixel_data_norm, prominence_threshold=0.005, min_distance_factor=0.02):
        """
        Find the outermost significant dips in the spectrum to determine center frequency.
        
        Args:
            freq_axis: Frequency axis values
            pixel_data_norm: Normalized pixel data
            prominence_threshold: Minimum prominence for a dip to be considered
            min_distance_factor: Minimum distance between dips as a fraction of frequency range
            
        Returns:
            tuple: (left_dip_idx, right_dip_idx, center_freq) or (None, None, mean_freq) if not found
        """
        inverted = -pixel_data_norm
        freq_range = freq_axis[-1] - freq_axis[0]
        min_distance = int(min_distance_factor * len(freq_axis))
        
        # Try multiple prominence levels to ensure we find enough dips
        prominence_levels = [
            prominence_threshold, 
            prominence_threshold * 0.7, 
            prominence_threshold * 0.5, 
            prominence_threshold * 0.3
        ]
        
        # Find all potential dips
        all_dips = []
        
        for prominence in prominence_levels:
            peaks, properties = find_peaks(
                inverted, 
                prominence=prominence, 
                width=1,
                distance=min_distance
            )
            
            if len(peaks) > 0:
                # Store peak indices with their prominences
                for i, peak_idx in enumerate(peaks):
                    all_dips.append({
                        'index': peak_idx,
                        'frequency': freq_axis[peak_idx],
                        'prominence': properties['prominences'][i]
                    })
        
        if len(all_dips) < 2:
            # Not enough dips found, return the center of the frequency range
            mean_freq = np.mean(freq_axis)
            return None, None, mean_freq
        
        # Sort dips by frequency
        all_dips.sort(key=lambda x: x['frequency'])
        
        # Get prominent dips from left and right sides
        # We'll define the first third of the spectrum as 'left' and last third as 'right'
        left_boundary = freq_axis[0] + freq_range / 3
        right_boundary = freq_axis[-1] - freq_range / 3
        
        left_dips = [dip for dip in all_dips if dip['frequency'] < left_boundary]
        right_dips = [dip for dip in all_dips if dip['frequency'] > right_boundary]
        
        # Prefer the most prominent dips
        left_dips.sort(key=lambda x: -x['prominence'])
        right_dips.sort(key=lambda x: -x['prominence'])
        
        if left_dips and right_dips:
            # Use the most prominent dips from each side
            left_dip = left_dips[0]
            right_dip = right_dips[0]
            
            # Calculate center frequency as the midpoint between these dips
            center_freq = (left_dip['frequency'] + right_dip['frequency']) / 2
            
            return left_dip['index'], right_dip['index'], center_freq
        elif len(all_dips) >= 2:
            # If we don't have clear left/right dips, take the leftmost and rightmost
            leftmost_dip = all_dips[0]
            rightmost_dip = all_dips[-1]
            
            # Calculate center frequency
            center_freq = (leftmost_dip['frequency'] + rightmost_dip['frequency']) / 2
            
            return leftmost_dip['index'], rightmost_dip['index'], center_freq
        else:
            # Fallback to mean frequency if can't determine reliably
            mean_freq = np.mean(freq_axis)
            return None, None, mean_freq
    
    @staticmethod
    def double_dip_func(f, I0, A, width, f_center, f_delta):
        """
        Static method version of double Lorentzian dip function.
        Must be static for multiprocessing to work.
        """
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)

    def multi_lorentzian(self, f, params):
        """
        Calculate multi-Lorentzian function with asymmetric pairs
        """
        # Extract baseline intensity
        I0 = params[self.param_indices['I0']]
        result = I0
        
        # Get center frequency
        f_center = params[self.param_indices['f_center']]
        
        # Process each pair
        for i in range(1, self.n_pairs + 1):
            try:
                # Get parameters for this pair
                A_i_1 = params[self.param_indices[f'A_{i}_1']]
                A_i_2 = params[self.param_indices[f'A_{i}_2']]
                width_i_1 = params[self.param_indices[f'width_{i}_1']]
                width_i_2 = params[self.param_indices[f'width_{i}_2']]
                f_d_i_1 = params[self.param_indices[f'f_d_{i}_1']]
                f_d_i_2 = params[self.param_indices[f'f_d_{i}_2']]
                
                # Calculate positions of dips
                f_1 = f_center - f_d_i_1
                f_2 = f_center + f_d_i_2
                
                # Calculate and subtract dips
                dip_1 = A_i_1 / (1 + ((f_1 - f) / width_i_1) ** 2)
                dip_2 = A_i_2 / (1 + ((f_2 - f) / width_i_2) ** 2)
                
                # Subtract dips from baseline
                result -= (dip_1 + dip_2)
                
            except Exception as e:
                # Log error but continue with the remaining pairs
                print(f"Error processing pair {i} in multi_lorentzian: {str(e)}")
                continue
        
        return result

    @staticmethod
    def multi_lorentzian_asymmetric_pairs(f, I0, f_center, *params):
        """
        Multi-Lorentzian function with asymmetric dip pairs around a common center frequency.
        
        Args:
            f: Frequency values
            I0: Baseline intensity
            f_center: Common center frequency for all pairs
            *params: List of parameters for each pair in format:
                    [A_1_1, A_1_2, w_1_1, w_1_2, f_d_1_1, f_d_1_2, 
                    A_2_1, A_2_2, w_2_1, w_2_2, f_d_2_1, f_d_2_2, ...]
                    
                    Where for each pair i:
                    - A_i_1, A_i_2 are amplitudes
                    - w_i_1, w_i_2 are widths
                    - f_d_i_1, f_d_i_2 are distance from center
        
        Returns:
            Combined signal with all Lorentzian dips
        """
        # Start with baseline intensity
        result = I0
        
        # Each pair has 6 parameters
        n_pairs = len(params) // 6
        
        for i in range(n_pairs):
            # Index for this pair's parameters
            base_idx = i * 6
            
            # Extract parameters for this pair
            A_i_1 = params[base_idx]
            A_i_2 = params[base_idx + 1]
            w_i_1 = params[base_idx + 2]
            w_i_2 = params[base_idx + 3]
            f_d_i_1 = params[base_idx + 4]
            f_d_i_2 = params[base_idx + 5]
            
            # Calculate dip positions (left and right of center)
            f_left = f_center - f_d_i_1
            f_right = f_center + f_d_i_2
            
            # Calculate dips
            dip_1 = A_i_1 / (1 + ((f_left - f) / w_i_1) ** 2)
            dip_2 = A_i_2 / (1 + ((f_right - f) / w_i_2) ** 2)
            
            # Subtract dips from baseline
            result -= (dip_1 + dip_2)
        
        return result
    
    @staticmethod
    def find_dips_around_center(freq_axis, pixel_data_norm, center_freq, max_dips=6, prominence_levels=None):
        """
        Find individual dips in the spectrum, with center frequency used for organization only.
        
        Args:
            freq_axis: Frequency axis values
            pixel_data_norm: Normalized pixel data
            center_freq: Central frequency (used for organization but not for constraining positions)
            max_dips: Maximum number of dips to find
            prominence_levels: Prominence levels to try for peak finding
        
        Returns:
            List of dip indices and dip info
        """
        inverted = -pixel_data_norm
        
        # Try progressively lower prominence values
        if prominence_levels is None:
            prominence_levels = [0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]
        
        # Find all dips at any prominence level
        all_dips = []
        
        for prominence in prominence_levels:
            try:
                peaks, properties = find_peaks(
                    inverted, 
                    prominence=prominence, 
                    width=1
                )
                
                if len(peaks) > 0:
                    # Store dips with their prominences
                    for i, peak_idx in enumerate(peaks):
                        dip_info = {
                            'index': peak_idx,
                            'frequency': freq_axis[peak_idx],
                            'value': pixel_data_norm[peak_idx],
                            'prominence': properties['prominences'][i] if 'prominences' in properties else prominence,
                            'distance_from_center': abs(freq_axis[peak_idx] - center_freq)
                        }
                        all_dips.append(dip_info)
            except Exception as e:
                print(f"Error finding peaks with prominence {prominence}: {e}")
                continue
        
        # If no dips found, return empty lists
        if not all_dips:
            print("No dips found at any prominence level")
            return [], None
        
        # Remove duplicate dips (same or very close indices)
        filtered_dips = []
        for dip in all_dips:
            # Check if this dip is too close to any already filtered dip
            is_duplicate = False
            for existing_dip in filtered_dips:
                if abs(dip['index'] - existing_dip['index']) <= 1:  # Adjacent points
                    is_duplicate = True
                    # Keep the more prominent dip
                    if dip['prominence'] > existing_dip['prominence']:
                        filtered_dips.remove(existing_dip)
                        filtered_dips.append(dip)
                    break
            
            if not is_duplicate:
                filtered_dips.append(dip)
        
        # Sort dips by prominence (highest first)
        filtered_dips.sort(key=lambda x: -x['prominence'])
        
        # Take the most prominent dips (up to max_dips)
        best_dips = filtered_dips[:max_dips]
        
        # Sort by frequency (left to right)
        best_dips.sort(key=lambda x: x['frequency'])
        
        # Extract dip indices
        dip_indices = [dip['index'] for dip in best_dips]
        
        return dip_indices, best_dips
   
    @staticmethod
    def estimate_multi_dip_independent(freq_axis, pixel_data_norm, dip_indices, center_freq, filtered_dips=None):
        """
        Estimate parameters for multiple independent dips.
        Center frequency is only used for reference.
        
        Args:
            freq_axis: Frequency axis values
            pixel_data_norm: Normalized pixel data
            dip_indices: List of dip indices (may be empty)
            center_freq: Reference center frequency
            filtered_dips: Optional list of dip properties for better initialization
        
        Returns:
            Tuple of (I0_est, params_list)
        """
        # Physics-based constraints
        TYPICAL_WIDTH = 0.006
        
        # Calculate baseline intensity using high percentile
        I0_est = np.percentile(pixel_data_norm, 98)
        
        # Apply Savitzky-Golay filter for smoother data
        window_length = min(15, len(pixel_data_norm)//2*2+1)
        smoothed_data = savgol_filter(pixel_data_norm, window_length=window_length, polyorder=3)
        
        # Parameter list for all dips
        params = []
        
        # Function to estimate width from data
        def estimate_width(idx, default_width=TYPICAL_WIDTH):
            try:
                # Look for nearest points at half-prominence height
                half_height = (I0_est + smoothed_data[idx]) / 2
                left_idx, right_idx = idx, idx
                
                # Search left
                while left_idx > 0 and smoothed_data[left_idx] < half_height:
                    left_idx -= 1
                    
                # Search right
                while right_idx < len(smoothed_data)-1 and smoothed_data[right_idx] < half_height:
                    right_idx += 1
                    
                # Calculate width in frequency units
                if right_idx > left_idx:
                    width = (freq_axis[right_idx] - freq_axis[left_idx]) / 2
                    # Constrain width to reasonable values
                    width = max(width, 0.002)  # Minimum width of 0.002 GHz
                    width = min(width, 0.05)   # Maximum width of 0.05 GHz
                    return width
                return default_width
            except Exception:
                return default_width
        
        # If no dips were found, create synthetic ones based on the center frequency
        if not dip_indices:
            print("No real dips found, creating synthetic ones")
            # Create parameter estimates for default dips around the center
            freq_range = freq_axis[-1] - freq_axis[0]
            
            # Create two synthetic dips around center
            f_pos_1 = center_freq - 0.05  # 0.05 GHz to the left
            f_pos_2 = center_freq + 0.05  # 0.05 GHz to the right
            
            A_est = 0.1 * np.ptp(pixel_data_norm)
            w_est = TYPICAL_WIDTH
            
            # Add params for first dip: amplitude, width, position
            params.extend([A_est, w_est, f_pos_1])
            # Add params for second dip: amplitude, width, position
            params.extend([A_est, w_est, f_pos_2])
            
            return I0_est, params
        
        # Process each dip individually
        for dip_idx in dip_indices:
            # Get position directly from frequency axis
            f_pos = freq_axis[dip_idx]
            
            # Estimate amplitude
            A_est = max(I0_est - smoothed_data[dip_idx], 0.01 * I0_est) * 2.0
            
            # Estimate width
            w_est = estimate_width(dip_idx)
            
            # Add parameters for this dip
            params.extend([A_est, w_est, f_pos])
        
        return I0_est, params

    @staticmethod
    def fit_pixel_with_asymmetric_pairs(pixel_data, freq_axis, n_pairs=2, default_values=None, method='trf'):
        """
        Fit ODMR data with asymmetric Lorentzian dip pairs.
        
        Args:
            pixel_data: Raw pixel data
            freq_axis: Frequency axis values
            n_pairs: Number of dip pairs to fit
            method: Fitting method ('trf' or 'lm')
        
        Returns:
            dict: Fitted parameters
        """
        # Normalize data
        scale_factor = np.max(np.abs(pixel_data))
        if scale_factor == 0:
            scale_factor = 1
        pixel_data_norm = pixel_data / scale_factor
        
        # Smooth data for analysis
        window_length = min(11, len(pixel_data_norm)//2*2+1)
        smoothed_data = savgol_filter(pixel_data_norm, window_length=window_length, polyorder=3)
        
        try:
            # Estimate center frequency
            # Estimate center frequency
            left_idx, right_idx, center_freq = ODMRAnalyzer.find_outermost_dips(freq_axis, smoothed_data)
            if center_freq is None or np.isnan(center_freq):
                center_freq = np.mean(freq_axis)
            
            # Estimate baseline intensity (I0)
            I0_est = np.percentile(smoothed_data, 98)
            
            # Initial parameters for all pairs
            p0 = [I0_est, center_freq]

            # Find initial dips to estimate pair parameters
            dip_indices, filtered_dips = ODMRAnalyzer.find_dips_around_center(
                freq_axis, smoothed_data, center_freq, max_dips=n_pairs*2
            )
            
            # Sort dips by frequency
            if filtered_dips:
                filtered_dips.sort(key=lambda x: x['frequency'])
            
            # For each pair, estimate parameters
            for i in range(n_pairs):
                # Default parameters for this pair
                A_i_1_est = 0.1
                A_i_2_est = 0.1
                w_i_1_est = 0.005
                w_i_2_est = 0.005
                f_d_i_1_est = 0.02 + 0.02 * i  # Increase spacing for higher pairs
                f_d_i_2_est = 0.02 + 0.02 * i
                
                # If we have detected dips, use their properties
                if filtered_dips and len(filtered_dips) >= 2*i+2:
                    left_dip = filtered_dips[2*i]
                    right_dip = filtered_dips[2*i+1]
                    
                    # Calculate distance from center for each dip
                    f_d_i_1_est = abs(center_freq - left_dip['frequency'])
                    f_d_i_2_est = abs(right_dip['frequency'] - center_freq)
                    
                    # Estimate amplitude from dip depth
                    A_i_1_est = max(0.01, I0_est - left_dip['value']) * 1.5
                    A_i_2_est = max(0.01, I0_est - right_dip['value']) * 1.5
                    
                    # Width is harder to estimate - use typical values
                    w_i_1_est = 0.005
                    w_i_2_est = 0.005
                
                # Add parameters for this pair
                p0.extend([A_i_1_est, A_i_2_est, w_i_1_est, w_i_2_est, f_d_i_1_est, f_d_i_2_est])
            
            # Set bounds for TRF method
            if method == 'trf':
                # Define bounds for general parameters
                I0_min = max(0.01, I0_est * 0.5)
                I0_max = I0_est * 1.5
                f_center_min = center_freq - 0.05
                f_center_max = center_freq + 0.05
                
                lower_bounds = [I0_min, f_center_min]
                upper_bounds = [I0_max, f_center_max]
                
                # Define bounds for each pair
                for i in range(n_pairs):
                    # Amplitude bounds
                    A_min = 1e-6
                    A_max = 2.0
                    
                    # Width bounds
                    w_min = 0.001
                    w_max = 0.02
                    
                    # Frequency offset bounds
                    f_d_min = 0.001
                    f_d_max = 0.1
                    
                    # Add bounds for this pair
                    pair_lower = [A_min, A_min, w_min, w_min, f_d_min, f_d_min]
                    pair_upper = [A_max, A_max, w_max, w_max, f_d_max, f_d_max]
                    
                    lower_bounds.extend(pair_lower)
                    upper_bounds.extend(pair_upper)
                
                bounds = (lower_bounds, upper_bounds)
            else:
                bounds = (-np.inf, np.inf)
            
            # Weights to emphasize dip regions
            weights = np.ones_like(freq_axis)
            if filtered_dips:
                for dip in filtered_dips:
                    idx = dip['index']
                    width = 5  # Points on each side
                    left = max(0, idx - width)
                    right = min(len(weights) - 1, idx + width)
                    weights[left:right+1] = 3.0  # 3x weight
            
            # Perform fitting
            popt, pcov = curve_fit(
                ODMRAnalyzer.multi_lorentzian_asymmetric_pairs,
                freq_axis,
                pixel_data_norm,
                p0=p0,
                bounds=bounds if method == 'trf' else (-np.inf, np.inf),
                method=method,
                sigma=1/weights,
                maxfev=5000,
                ftol=1e-5,
                xtol=1e-5
            )
            
            # Process results into dictionary
            result = {}
            
            # Extract baseline and center frequency
            result["I0"] = popt[0] * scale_factor
            result["f_center"] = popt[1]
            
            # Extract parameters for each pair
            for i in range(n_pairs):
                base_idx = 2 + i * 6  # Start after I0, f_center
                
                result[f"A_{i+1}_1"] = popt[base_idx] * scale_factor
                result[f"A_{i+1}_2"] = popt[base_idx + 1] * scale_factor
                result[f"width_{i+1}_1"] = popt[base_idx + 2]
                result[f"width_{i+1}_2"] = popt[base_idx + 3]
                result[f"f_d_{i+1}_1"] = popt[base_idx + 4]
                result[f"f_d_{i+1}_2"] = popt[base_idx + 5]
                
                # For compatibility with existing code
                if i == 0:
                    result["A"] = (result[f"A_{i+1}_1"] + result[f"A_{i+1}_2"]) / 2
                    result["width"] = (result[f"width_{i+1}_1"] + result[f"width_{i+1}_2"]) / 2
                    result["f_delta"] = result[f"f_d_{i+1}_1"] + result[f"f_d_{i+1}_2"]
            
            # Calculate fit quality
            fitted_curve = ODMRAnalyzer.multi_lorentzian_asymmetric_pairs(freq_axis, *popt)
            quality_score = calculate_fit_quality(pixel_data_norm, fitted_curve)
            result["quality_score"] = quality_score
            
            return result
            
        except Exception as e:
            print(f"Asymmetric pair fitting failed: {str(e)}")
            
            # Fall back to default values
            default_values = {
                "I0": np.mean(pixel_data),
                "f_center": np.mean(freq_axis)
            }
            
            # Add pair-specific defaults
            for i in range(n_pairs):
                default_values[f"A_{i+1}_1"] = np.ptp(pixel_data) * 0.1
                default_values[f"A_{i+1}_2"] = np.ptp(pixel_data) * 0.1
                default_values[f"width_{i+1}_1"] = 0.006
                default_values[f"width_{i+1}_2"] = 0.006
                default_values[f"f_d_{i+1}_1"] = 0.015 + 0.01 * i
                default_values[f"f_d_{i+1}_2"] = 0.015 + 0.01 * i
            
            # Add compatibility defaults
            default_values["A"] = np.ptp(pixel_data) * 0.1
            default_values["width"] = 0.006
            default_values["f_delta"] = 0.03
            default_values["quality_score"] = 0.0
            
            return default_values
    
    def fit_multi_lorentzian_with_asymmetric_pairs(self, n_pairs=2, method='trf', output_dir=None):
        """
        Parallel version of multi-pair Lorentzian fitting with asymmetric pairs.
        
        Args:
            n_pairs: Number of dip pairs to fit
            method: Fitting method ('trf', 'lm', etc.)
            output_dir: Directory to save results
            
        Returns:
            tuple: (output_directory, fitted_parameters_file)
        """
        self.start_profiling()
        
        M, N, F = self.data.shape
        
        # Initialize parameter storage
        # Core parameters: I0, f_center
        fitted_params = {
            "I0": np.zeros((M, N)),
            "f_center": np.zeros((M, N)),
        }
        
        # Add parameters for each pair
        for i in range(1, n_pairs + 1):
            fitted_params[f"A_{i}_1"] = np.zeros((M, N))
            fitted_params[f"A_{i}_2"] = np.zeros((M, N))
            fitted_params[f"width_{i}_1"] = np.zeros((M, N))
            fitted_params[f"width_{i}_2"] = np.zeros((M, N))
            fitted_params[f"f_d_{i}_1"] = np.zeros((M, N))
            fitted_params[f"f_d_{i}_2"] = np.zeros((M, N))
        
        # Add compatibility parameters
        fitted_params["A"] = np.zeros((M, N))
        fitted_params["width"] = np.zeros((M, N))
        fitted_params["f_delta"] = np.zeros((M, N))
        fitted_params["quality_score"] = np.zeros((M, N))

        #-------

        # Default values
        default_values = {
            "I0": 1.0,
            "f_center": np.mean(self.freq_axis)
        }

        # Add defaults for each pair
        for i in range(1, n_pairs + 1):
            default_values[f"A_{i}_1"] = 0.1
            default_values[f"A_{i}_2"] = 0.1
            default_values[f"width_{i}_1"] = 0.006
            default_values[f"width_{i}_2"] = 0.006
            default_values[f"f_d_{i}_1"] = 0.015 + 0.01 * (i-1)
            default_values[f"f_d_{i}_2"] = 0.015 + 0.01 * (i-1)

        # Add compatibility defaults
        default_values["A"] = 0.1
        default_values["width"] = 0.006
        default_values["f_delta"] = 0.030
        default_values["quality_score"] = 0.0

        print(f"Processing {M*N} pixels using multiprocessing...")
        print(f"Fitting model: {n_pairs} asymmetric pairs")

        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores")

        # Process rows in parallel
        row_args = [(m, self.data, self.freq_axis, n_pairs, default_values, method) 
                    for m in range(M)]
        
        total_processed = 0
        start_time = time.time()
        speed_samples = []
        last_sample_time = start_time
        last_pixel_count = 0

        with mp.Pool(processes=num_cores) as pool:
            for m, row_results in tqdm(pool.imap(process_pixel_row_with_asymmetric_pairs, row_args), 
                                    total=M, 
                                    desc="Processing rows"):
                for key in fitted_params:
                    fitted_params[key][m] = row_results[key]
                
                total_processed += N
                current_time = time.time()
                
                # Sample speed every few seconds
                elapsed_since_last = current_time - last_sample_time
                if elapsed_since_last >= 5:
                    pixels_since_last = total_processed - last_pixel_count
                    sample_speed = pixels_since_last / elapsed_since_last
                    speed_samples.append(sample_speed)
                    
                    last_sample_time = current_time
                    last_pixel_count = total_processed
                
                # Print progress update periodically
                if total_processed % (N * 2) == 0:
                    elapsed_time = current_time - start_time
                    pixels_per_second = total_processed / elapsed_time
                    remaining_pixels = (M * N) - total_processed
                    remaining_time = remaining_pixels / pixels_per_second
                    
                    print(f"\nProcessing speed: {pixels_per_second:.2f} pixels/second")
                    print(f"Estimated time remaining: {remaining_time/60:.2f} minutes")

        # Print final performance metrics
        performance_metrics = print_performance_metrics(total_processed, start_time, speed_samples)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.data_file))[0]
            fitted_params_file = os.path.join(output_dir, f"{base_name}_asymmetric_pairs_params.npy")
            quality_stats_file = os.path.join(output_dir, f"{base_name}_asymmetric_pairs_quality_stats.txt")
            
            # Save all parameters
            param_order = ['I0', 'f_center']
            
            # Add parameters for each pair
            for i in range(1, n_pairs + 1):
                param_order.extend([
                    f"A_{i}_1", f"A_{i}_2", 
                    f"width_{i}_1", f"width_{i}_2", 
                    f"f_d_{i}_1", f"f_d_{i}_2"
                ])

            # Add compatibility parameters
            param_order.extend(['A', 'width', 'f_delta', 'quality_score'])
            
            stacked_params = np.stack([fitted_params[param] for param in param_order], axis=-1)
            np.save(fitted_params_file, stacked_params)
            
            # Also save a compatibility version with traditional parameter names
            compat_params_file = os.path.join(output_dir, f"{base_name}_compat_params.npy")
            compat_param_order = ['I0', 'A', 'width', 'f_center', 'f_delta', 'quality_score']
            compat_stacked = np.stack([fitted_params[param] for param in compat_param_order], axis=-1)
            np.save(compat_params_file, compat_stacked)
            
            # Calculate and save quality statistics
            quality_scores = fitted_params['quality_score']
            success_rate = np.mean(quality_scores >= 0.9) * 100
            mean_quality = np.mean(quality_scores)
            median_quality = np.median(quality_scores)
            
            with open(quality_stats_file, 'w') as f:
                f.write(f"Asymmetric Pairs Fitting Quality Statistics:\n")
                f.write(f"Success Rate (score >= 0.9): {success_rate:.1f}%\n")
                f.write(f"Mean Quality Score: {mean_quality:.3f}\n")
                f.write(f"Median Quality Score: {median_quality:.3f}\n")
                
                # Add performance metrics to the same file
                f.write(f"\nODMR Asymmetric Pairs Fitting Performance Statistics:\n")
                f.write(f"Average speed: {performance_metrics['average_speed']:.2f} pixels/second\n")
                f.write(f"Maximum speed: {performance_metrics['max_speed']:.2f} pixels/second\n")
                f.write(f"Total computation time: {performance_metrics['total_computation_time']:.2f} seconds ({performance_metrics['total_computation_time']/60:.2f} minutes)\n")
            
            # Copy the original files
            new_data_file = os.path.join(output_dir, os.path.basename(self.data_file))
            new_json_file = os.path.join(output_dir, os.path.basename(self.json_file))
            import shutil
            shutil.copy2(self.data_file, new_data_file)
            shutil.copy2(self.json_file, new_json_file)
            
            print(f"\nSaved files in: {output_dir}")
            print(f"Saved:")
            print(f"  - Asymmetric pairs parameters: {os.path.basename(fitted_params_file)}")
            print(f"  - Compatible parameters: {os.path.basename(compat_params_file)}")
            print(f"  - Quality statistics: {os.path.basename(quality_stats_file)}")
            print(f"  - Original data: {os.path.basename(new_data_file)}")
            print(f"  - JSON parameters: {os.path.basename(new_json_file)}")
            
            return output_dir, fitted_params_file

        self.stop_profiling()
        return None, None

def main():
    data_file = r"C:\Users\Diederik\Documents\BEP\PB_diamond_data\2D_ODMR_scan_1733764788.npy"
    json_file = r"C:\Users\Diederik\Documents\BEP\PB_diamond_data\2D_ODMR_scan_1733764788.json"

    analyzer = None
    if os.path.exists(data_file) and os.path.exists(json_file):
        analyzer = ODMRAnalyzer(data_file, json_file, enable_profiling=False)
    
    while True:
        print("\nODMR Analysis Options:")
        print("1. Perform experiment fitting and save parameters")
        print("2. Analyze single pixel spectrum")
        print("3. Batch process directory")
        print("4. Check fitted results")
        print("5. Exit")
        
        choice = input("Enter your choice (1/2/3/4/5): ").strip()
        
        if choice in ['1', '2'] and analyzer is None:
            if not all(os.path.exists(f) for f in [data_file, json_file]):
                print("Error: One or more input files not found.")
                continue
            analyzer = ODMRAnalyzer(data_file, json_file, enable_profiling=False)
        
        if choice == '1':
            method_choice = input("Choose optimization method (trf/lm): ").strip().lower()
            try:
                n_pairs = int(input("Enter number of dip pairs to fit (default: 2): ").strip() or "2")
                if n_pairs < 1:
                    print("Number of pairs must be at least 1. Using 2 pairs.")
                    n_pairs = 2
            except ValueError:
                print("Invalid input. Using default (2 pairs).")
                n_pairs = 2
            
            # Convert n_pairs to n_dips (2 dips per pair)
            n_dips = n_pairs * 2
            
            output_dir = input("Enter output directory path (press Enter for default './fitted_parameters'): ").strip() or "./fitted_parameters"
            output_dir, fitted_params_file = analyzer.fit_multi_lorentzian_with_asymmetric_pairs(
                n_pairs=n_pairs,
                method=method_choice,
                output_dir=output_dir
            )

            if fitted_params_file and output_dir:
                inspect_choice = input("\nDo you want to inspect the fitting results now? (y/n): ").strip().lower()
                if inspect_choice.startswith('y'):
                    try:
                        analyzer.check_fits(fitted_params_file)
                    except Exception as e:
                        print(f"Error launching interactive viewer: {str(e)}")
                        traceback.print_exc()
        
        elif choice == '2':
            if analyzer is None:
                print("Error: Analyzer not initialized.")
                continue
            
            try:
                x = int(input(f"Enter x coordinate (0-{analyzer.data.shape[0]-1}): ").strip())
                y = int(input(f"Enter y coordinate (0-{analyzer.data.shape[1]-1}): ").strip())
                print(f"Attempting to process pixel at coordinates: x={x}, y={y}")
                
                try:
                    n_pairs = int(input("Enter number of dip pairs to fit (default: 2): ").strip() or "2")
                    if n_pairs < 1:
                        print("Number of pairs must be at least 1. Using 2 pairs.")
                        n_pairs = 2
                except ValueError:
                    print("Invalid input. Using default (2 pairs).")
                    n_pairs = 2
                
                # Convert n_pairs to n_dips (2 dips per pair)
                n_dips = n_pairs * 2
                
                pixel = analyzer.data[x, y, :]
                single_pixel_params = analyzer.fit_single_pixel_with_independent_dips(
                    pixel, analyzer.freq_axis, n_dips=n_dips, method=method_choice
                )
                
                # Display results - modified to show individual dip information
                print("\nFitted Parameters:")
                print(f"I0: {single_pixel_params['I0']:.4f}")
                print(f"f_center: {single_pixel_params['f_center']:.4f}")
                
                # Display individual dips
                for i in range(1, n_dips + 1):
                    if f"A_{i}" in single_pixel_params:
                        print(f"\nDip {i}:")
                        print(f"  A_{i}: {single_pixel_params[f'A_{i}']:.4e}")
                        print(f"  width_{i}: {single_pixel_params[f'width_{i}']:.4f}")
                        print(f"  position: {single_pixel_params[f'f_pos_{i}']:.4f}")
                
                # Display derived pair information if available
                print("\nDerived Pair Information:")
                pair_count = 0
                while f"f_delta_{pair_count+1}" in single_pixel_params:
                    pair_count += 1
                
                for j in range(1, pair_count + 1):
                    if f"f_delta_{j}" in single_pixel_params:
                        print(f"  Pair {j}:")
                        print(f"    f_delta_{j}: {single_pixel_params[f'f_delta_{j}']:.4f}")
                        if f"pair_center_{j}" in single_pixel_params:
                            print(f"    pair_center_{j}: {single_pixel_params[f'pair_center_{j}']:.4f}")
                
                if 'quality_score' in single_pixel_params:
                    print(f"\nFit quality: {single_pixel_params['quality_score']:.4f}")
                
            except Exception as e:
                print(f"Error processing pixel: {e}")
                traceback.print_exc()
        
        elif choice == '3':
            directory = input("Enter directory path containing ODMR datasets: ").strip()
            method_choice = input("Choose optimization method (trf/lm): ").strip().lower()
            while method_choice not in ['trf', 'lm']:
                print("Invalid choice. Please choose 'trf' or 'lm'")
                method_choice = input("Choose optimization method (trf/lm): ").strip().lower()
            
            try:
                n_pairs = int(input("Enter number of dip pairs to fit (default: 2): ").strip() or "2")
                if n_pairs < 1:
                    print("Number of pairs must be at least 1. Using 2 pairs.")
                    n_pairs = 2
            except ValueError:
                print("Invalid input. Using default (2 pairs).")
                n_pairs = 2
            
            # Convert n_pairs to n_dips (2 dips per pair)
            n_dips = n_pairs * 2
            
            output_dir = input("Enter output directory path (press Enter for default './fitted_parameters'): ").strip() or "./fitted_parameters"
            
            try:
                # Define a function to process directories with independent dips
                def process_directory_with_independent_dips(directory, method='trf', output_dir='./fitted_parameters', n_dips=4):
                    """
                    Process all ODMR data files in a directory using independent dips approach
                    
                    Args:
                        directory: Directory containing ODMR data (.npy and .json pairs)
                        method: Curve fitting method ('trf' or 'lm')
                        output_dir: Output directory for fitted parameters
                        n_dips: Number of independent dips to fit
                    """
                    # Create output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Find all .npy files in the directory
                    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy') and 'ODMR_scan' in f]
                    
                    if not npy_files:
                        print(f"No ODMR scan files found in {directory}")
                        return
                    
                    print(f"Found {len(npy_files)} potential ODMR data files.")
                    
                    # Process each file if corresponding JSON exists
                    success_count = 0
                    for npy_file in npy_files:
                        base_name = os.path.splitext(npy_file)[0]
                        json_file = os.path.join(directory, base_name + '.json')
                        npy_path = os.path.join(directory, npy_file)
                        
                        if not os.path.exists(json_file):
                            print(f"Skipping {npy_file} - no matching JSON file found.")
                            continue
                        
                        print(f"\nProcessing {npy_file}")
                        try:
                            # Initialize analyzer for this file
                            analyzer = ODMRAnalyzer(npy_path, json_file, enable_profiling=False)
                            
                            # Create experiment-specific subdirectory
                            exp_output_dir = os.path.join(output_dir, f"experiment_{base_name}")
                            os.makedirs(exp_output_dir, exist_ok=True)

                            print(f"Using independent dips fitting with {n_dips} dips")
                            output_dir, fitted_params_file = analyzer.fit_multi_lorentzian_with_independent_dips(
                                n_dips=n_dips,
                                method=method,
                                output_dir=exp_output_dir
                            )
                            
                            if fitted_params_file:
                                success_count += 1
                                print(f"Successfully processed {npy_file}")
                            else:
                                print(f"Processing {npy_file} completed but no output file was generated.")
                                
                        except Exception as e:
                            print(f"Error processing {npy_file}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    print(f"\nBatch processing complete. Successfully processed {success_count}/{len(npy_files)} files.")
                    print(f"Results saved in {output_dir}")
                
                # Use the independent dips approach
                process_directory_with_independent_dips(directory, method=method_choice, output_dir=output_dir, n_dips=n_dips)
            except Exception as e:
                print(f"Error during batch processing: {e}")
                traceback.print_exc()
        
        elif choice == '4':
            print("\nChecking fitted results - please specify the required files:")
            fitted_params_file = input("\nEnter path to fitted parameters file (.npy): ").strip()
            data_file = input("Enter path to original data file (.npy): ").strip()
            json_file = input("Enter path to JSON parameters file (.json): ").strip()
            
            if not all(os.path.exists(f) for f in [fitted_params_file, data_file, json_file]):
                print("Error: One or more specified files do not exist.")
                continue
            
            try:
                checker = ODMRFitChecker(fitted_params_file, data_file, json_file)
                checker.create_interactive_viewer()
            except Exception as e:
                print(f"\nError launching fit checker: {e}")
                traceback.print_exc()
        
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()
