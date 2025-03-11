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

def process_pixel_row_with_fixed_center(args):
    """
    Process a row of pixels with fixed center approach.
    
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
    
    # Add parameters for each potential pair
    for i in range(n_pairs):
        row_results[f"A_{i+1}_1"] = np.zeros(N)
        row_results[f"A_{i+1}_2"] = np.zeros(N)
        row_results[f"width_{i+1}_1"] = np.zeros(N)
        row_results[f"width_{i+1}_2"] = np.zeros(N)
        row_results[f"f_delta_{i+1}"] = np.zeros(N)
    
    # Add compatibility parameters
    row_results["A"] = np.zeros(N)
    row_results["width"] = np.zeros(N)
    row_results["f_delta"] = np.zeros(N)
    
    # Process each pixel in the row
    for n in range(N):
        pixel_data = data[m, n, :]
        
        try:
            result = ODMRAnalyzer.fit_single_pixel_with_fixed_center(
                pixel_data, freq_axis, n_pairs=n_pairs, 
                default_values=default_values, method=method
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
        self.n_pairs = (param_count - 3) // 5  # Subtract 2 for I0, f_center and 1 for quality score
        
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
        """Calculate multi-Lorentzian function with given parameters."""
        # Extract baseline and center frequency
        I0 = params[self.param_indices['I0']]
        f_center = params[self.param_indices['f_center']]
        
        # Start with baseline
        result = I0
        
        # Subtract each dip pair
        for i in range(1, self.n_pairs + 1):
            A_1 = params[self.param_indices[f'A_{i}_1']]
            A_2 = params[self.param_indices[f'A_{i}_2']]
            width_1 = params[self.param_indices[f'width_{i}_1']]
            width_2 = params[self.param_indices[f'width_{i}_2']]
            f_delta = params[self.param_indices[f'f_delta_{i}']]
            
            # Calculate dip positions
            f_1 = f_center - 0.5 * f_delta
            f_2 = f_center + 0.5 * f_delta
            
            # Add dips
            result -= A_1 / (1 + ((f_1 - f) / width_1) ** 2)
            result -= A_2 / (1 + ((f_2 - f) / width_2) ** 2)
        
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

    def plot_pixel_spectrum(self, x, y, smooth=True, fit_result=None):
        """
        Plot spectrum from a specific pixel with original and optional fitted data
        
        Args:
            x (int): x-coordinate of pixel
            y (int): y-coordinate of pixel
            smooth (bool): Apply Savitzky-Golay smoothing
            fit_result (dict, optional): Fitted parameters to overlay on plot
        """
        spectrum = self.data[x, y, :]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot original data points
        ax.scatter(self.freq_axis, spectrum, color='red', alpha=0.5, label='Raw Data Points', zorder=2)
        
        if smooth:
            # Apply Savitzky-Golay filter for smoothing
            smoothed_spectrum = savgol_filter(spectrum, window_length=7, polyorder=3)
            ax.plot(self.freq_axis, smoothed_spectrum, 'b-', linewidth=2, label='Smoothed Spectrum', zorder=3)
        
        # Find dips
        inverted = -spectrum
        peaks, properties = find_peaks(inverted, prominence=0.01)
        
        # Plot dip positions
        for peak in peaks:
            ax.axvline(x=self.freq_axis[peak], color='g', alpha=0.3, linestyle='--')
            ax.text(self.freq_axis[peak], spectrum[peak], 
                f'{self.freq_axis[peak]:.3f} GHz', 
                rotation=90, verticalalignment='bottom')
        
        # Plot fitted curve if parameters are provided
        if fit_result is not None:
            # Modify this part to handle scalar parameters
            fitted_params = [
                fit_result['I0'],  # Remove [x, y] indexing
                fit_result['A'], 
                fit_result['width'], 
                fit_result['f_center'], 
                fit_result['f_delta']
            ]
            fitted_spectrum = self.double_dip_func(self.freq_axis, *fitted_params)
            ax.plot(self.freq_axis, fitted_spectrum, 'r--', label='Fitted Curve', linewidth=2)
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('ODMR Signal (a.u.)')
        ax.set_title(f'ODMR Spectrum at Pixel ({x}, {y})')
        ax.set_xlim([self.freq_axis[0], self.freq_axis[-1]])
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        return fig, ax


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


    def plot_contrast_map(self, fitted_params=None):
        """
        Create a 2D heat map of the ODMR contrast
        
        Args:
            fitted_params (dict, optional): Use fitted parameters for contrast calculation
        """
        # Calculate contrast based on data or fitted parameters
        if fitted_params is None:
            contrast_map = np.ptp(self.data, axis=2)
        else:
            # Use amplitude from fitted parameters as contrast measure
            contrast_map = fitted_params['A']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(contrast_map.T, origin='lower', cmap='viridis',
                      extent=[self.params['x1'], self.params['x2'],
                             self.params['y1'], self.params['y2']])
        plt.colorbar(im, ax=ax, label='Contrast')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_title('ODMR Contrast Map')
        return fig, ax, contrast_map

    def save_fitted_parameters(self, fitted_params, base_output_dir):
        """
        Save fitted parameters and organize related files in a single experiment directory.
        
        This method creates a directory structure like:
        base_output_dir/
            experiment_[number]/
                lorentzian_params_[number].npy
                2D_ODMR_scan_[number].npy
                2D_ODMR_scan_[number].json
        """
        # Create the experiment-specific directory
        experiment_dir = os.path.join(base_output_dir, f"experiment_{self.experiment_number}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Define all file paths within the experiment directory
        params_file = os.path.join(experiment_dir, f"lorentzian_params_{self.experiment_number}.npy")
        new_data_file = os.path.join(experiment_dir, os.path.basename(self.data_file))
        new_json_file = os.path.join(experiment_dir, os.path.basename(self.json_file))
        
        # Save the fitted parameters
        param_order = ['I0', 'A', 'width', 'f_center', 'f_delta']
        stacked_params = np.stack([fitted_params[param] for param in param_order], axis=-1)
        np.save(params_file, stacked_params)
        
        # Copy the original data files
        import shutil
        shutil.copy2(self.data_file, new_data_file)
        shutil.copy2(self.json_file, new_json_file)
        
        print(f"\nOrganized experiment files in: {experiment_dir}")
        print(f"Saved:")
        print(f"  - Fitted parameters: {os.path.basename(params_file)}")
        print(f"  - Original data: {os.path.basename(new_data_file)}")
        print(f"  - JSON parameters: {os.path.basename(new_json_file)}")
        print(f"Saved array shape: {stacked_params.shape}")
        
        return experiment_dir, params_file
    
    def analyze_spectrum(self, x, y, fitted_params=None):
        """
        Comprehensive spectrum analysis for a specific pixel

        Args:
            x (int): x-coordinate of pixel
            y (int): y-coordinate of pixel
            fitted_params (dict, optional): Pre-computed fitted parameters
        
        Returns:
            dict: Spectrum analysis results
        """
        spectrum = self.data[x, y, :]
        smoothed = savgol_filter(spectrum, window_length=7, polyorder=3)
        
        # Find dips
        inverted = -smoothed
        peaks, properties = find_peaks(inverted, prominence=0.01)
        
        dip_frequencies = self.freq_axis[peaks]
        dip_depths = spectrum[peaks]
        
        # Calculate contrast
        contrast = np.ptp(spectrum)
        
        # Additional analysis using fitted parameters if provided
        fitted_analysis = {}
        if fitted_params is not None:
            # Calculate the row index from the pixel coordinates
            row = x
            
            fitted_analysis = {
                'I0': fitted_params['I0'][row],
                'amplitude': fitted_params['A'][row],
                'width': fitted_params['width'][row],
                'center_frequency': fitted_params['f_center'][row],
                'frequency_splitting': fitted_params['f_delta'][row]
            }
        
        return {
            'dip_frequencies': dip_frequencies,
            'dip_depths': dip_depths,
            'contrast': contrast,
            'num_dips': len(peaks),
            'fitted_parameters': fitted_analysis
        }

    @staticmethod
    def double_dip_func(f, I0, A, width, f_center, f_delta):
        """
        Static method version of double Lorentzian dip function.
        Must be static for multiprocessing to work.
        """
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)


    def optimize_pixel_fitting_with_fixed_center_fixed(self, x, y, method='trf', max_pairs=4):
        """
        Interactive optimization of pixel fitting using the fixed center approach.
        Uses outermost dips to determine center frequency.
        
        Args:
            x (int): x-coordinate of pixel
            y (int): y-coordinate of pixel
            method (str): Fitting method ('trf' or 'lm')
            max_pairs (int): Maximum number of pairs to try
        
        Returns:
            dict: Best fitted parameters
        """
        # Get pixel data
        pixel_data = self.data[x, y, :]
        
        # Try fitting with different numbers of pairs
        results = []
        
        for n_pairs in range(1, max_pairs + 1):
            print(f"Trying fit with {n_pairs} dip pairs...")
            result = ODMRAnalyzer.fit_single_pixel_with_fixed_center(
                pixel_data, self.freq_axis, n_pairs=n_pairs, method=method
            )
            quality = result['quality_score']
            print(f"  Quality score: {quality:.4f}")
            results.append((result, quality))
            
            # If we've reached excellent quality, we can stop
            if quality > 0.95:
                break
        
        # Select the best result
        best_result, best_quality = max(results, key=lambda x: x[1])
        
        print(f"\nBest fit has quality score: {best_quality:.4f} with {(len(best_result) - 3) // 5} pairs")
        print(f"I0 = {best_result['I0']:.3e}, f_center = {best_result['f_center']:.4f}")
        
        for i in range(1, (len(best_result) - 3) // 5 + 1):  # Count pairs based on parameters
            print(f"Pair {i}: A1={best_result[f'A_{i}_1']:.3e}, A2={best_result[f'A_{i}_2']:.3e}, "
                f"w1={best_result[f'width_{i}_1']:.4f}, w2={best_result[f'width_{i}_2']:.4f}, "
                f"fd={best_result[f'f_delta_{i}']:.4f}")
        
        # Visualize the result with our new plotting function
        n_pairs_best = (len(best_result) - 3) // 5
        fig, ax = self.plot_multi_dip_fit_fixed(x, y, best_result, n_pairs=n_pairs_best)
        plt.show()
        
        return best_result

    @staticmethod
    def multi_dip_func_shared_center(f, log_I0, *params):
        """
        Generalized n-dip Lorentzian function with a shared center frequency.
        Each pair has its own splitting (f_delta), but all pairs are centered around the same frequency.
        
        Args:
            f: Frequency values
            log_I0: Log of baseline intensity
            *params: List of parameters in format [log_A_1_1, log_A_1_2, log_w_1_1, log_w_1_2, f_c, f_d_1, 
                                            log_A_2_1, log_A_2_2, log_w_2_1, log_w_2_2, f_d_2, ...]
        
        Returns:
            Combined signal with n dips
        """
        # Safety check for parameter length
        if len(params) < 6:  # At minimum need log_A_1_1, log_A_1_2, log_w_1_1, log_w_1_2, f_c, f_d_1
            # Return baseline if not enough parameters
            return np.exp(log_I0) * np.ones_like(f)
        
        I0 = np.exp(log_I0)
        result = I0
        
        # Extract shared center frequency (always at index 4)
        f_center = params[4]
        
        # Calculate number of pairs based on parameter length
        # Each pair after the first needs 5 parameters
        remaining_params = len(params) - 6  # Subtract first pair's params + center frequency
        additional_pairs = remaining_params // 5
        n_pairs = 1 + additional_pairs
        
        for i in range(n_pairs):
            try:
                if i == 0:
                    # First pair - parameters are at the beginning
                    A_i_1 = np.exp(params[0])
                    A_i_2 = np.exp(params[1])
                    w_i_1 = np.exp(params[2])
                    w_i_2 = np.exp(params[3])
                    f_d_i = abs(params[5])  # Force positive splitting
                else:
                    # Subsequent pairs - parameters start after the center frequency and first pair
                    base_idx = 6 + (i-1) * 5  # 6 = 5 for first pair params + 1 for center freq
                    
                    # Check if we have enough parameters for this pair
                    if base_idx + 4 >= len(params):
                        break  # Not enough parameters for this pair, skip it
                    
                    A_i_1 = np.exp(params[base_idx])
                    A_i_2 = np.exp(params[base_idx + 1])
                    w_i_1 = np.exp(params[base_idx + 2])
                    w_i_2 = np.exp(params[base_idx + 3])
                    f_d_i = abs(params[base_idx + 4])  # Force positive splitting
                
                # Calculate dip positions using shared center
                f_1 = f_center - 0.5 * f_d_i
                f_2 = f_center + 0.5 * f_d_i
                
                # Calculate dips
                dip_1 = A_i_1 / (1 + ((f_1 - f) / w_i_1) ** 2)
                dip_2 = A_i_2 / (1 + ((f_2 - f) / w_i_2) ** 2)
                
                # Subtract dips from baseline
                result -= (dip_1 + dip_2)
                
            except Exception as e:
                # Log error but continue with result so far
                print(f"Error processing pair {i}: {str(e)}")
                continue
        
        return result

    @staticmethod
    def multi_dip_func_hybrid(f, log_I0, *params):
        """
        Generalized n-dip Lorentzian function with hybrid parameter space.
        I0, amplitudes, and widths are in log space, while f_center and f_delta are linear.
        
        Args:
            f: Frequency values
            log_I0: Log of baseline intensity
            *params: List of parameters in format [log_A_1_1, log_A_1_2, log_w_1_1, log_w_1_2, f_c, f_d_1, 
                                                log_A_2_1, log_A_2_2, log_w_2_1, log_w_2_2, f_d_2, ...]
        
        Returns:
            Combined signal with n dips
        """
        I0 = np.exp(log_I0)
        result = I0
        
        # Number of dip pairs (each pair has 5 parameters: log_A1, log_A2, log_w1, log_w2, f_d)
        n_pairs = len(params) // 5
        f_c = params[4]  # f_center is shared and at index 4
        
        for i in range(n_pairs):
            # Extract parameters for this dip pair
            A_i_1 = np.exp(params[i*5])
            A_i_2 = np.exp(params[i*5 + 1])
            w_i_1 = np.exp(params[i*5 + 2])
            w_i_2 = np.exp(params[i*5 + 3])
            f_d_i = params[i*5 + 4]
            
            # Calculate dip positions
            f_1 = f_c - 0.5 * f_d_i
            f_2 = f_c + 0.5 * f_d_i
            
            # Calculate dips
            dip_1 = A_i_1 / (1 + ((f_1 - f) / w_i_1) ** 2)
            dip_2 = A_i_2 / (1 + ((f_2 - f) / w_i_2) ** 2)
            
            # Subtract dips from baseline
            result -= (dip_1 + dip_2)
        
        return result

    @staticmethod
    def find_dip_pairs_from_center(freq_axis, pixel_data_norm, center_freq, max_pairs=3, prominence_levels=None):
        """
        Find dip pairs based on a fixed center frequency, with enhanced error handling.
        
        Args:
            freq_axis: Frequency axis values
            pixel_data_norm: Normalized pixel data
            center_freq: Central frequency around which dips should be organized
            max_pairs: Maximum number of dip pairs to find
            prominence_levels: Prominence levels to try for peak finding
        
        Returns:
            List of dip pairs (indices) and filtered dip info
        """
        inverted = -pixel_data_norm
        
        # Try progressively lower prominence values to find all potential dips
        if prominence_levels is None:
            prominence_levels = [0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]  # Add even lower levels
        
        # Find all dips at any prominence level
        all_dips = []
        
        for prominence in prominence_levels:
            try:
                peaks, properties = find_peaks(
                    inverted, 
                    prominence=prominence, 
                    width=1  # Allow narrower dips
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
        
        # If still no dips after filtering, return empty lists
        if not filtered_dips:
            print("No dips after filtering duplicates")
            return [], None
        
        # Sort dips into left and right of center
        left_dips = [dip for dip in filtered_dips if dip['frequency'] < center_freq]
        right_dips = [dip for dip in filtered_dips if dip['frequency'] >= center_freq]
        
        # If no dips on one side, create synthetic dips
        if not left_dips and right_dips:
            print("No left dips, using synthetic ones")
            # Create a synthetic left dip for each right dip
            for right_dip in right_dips:
                right_freq = right_dip['frequency']
                # Create symmetric left dip
                left_freq = 2 * center_freq - right_freq
                left_idx = np.argmin(np.abs(freq_axis - left_freq))
                
                synthetic_dip = {
                    'index': left_idx,
                    'frequency': freq_axis[left_idx],
                    'value': pixel_data_norm[left_idx],
                    'prominence': right_dip['prominence'] * 0.5,  # Lower prominence
                    'distance_from_center': abs(freq_axis[left_idx] - center_freq),
                    'synthetic': True
                }
                left_dips.append(synthetic_dip)
        
        elif not right_dips and left_dips:
            print("No right dips, using synthetic ones")
            # Create a synthetic right dip for each left dip
            for left_dip in left_dips:
                left_freq = left_dip['frequency']
                # Create symmetric right dip
                right_freq = 2 * center_freq - left_freq
                right_idx = np.argmin(np.abs(freq_axis - right_freq))
                
                synthetic_dip = {
                    'index': right_idx,
                    'frequency': freq_axis[right_idx],
                    'value': pixel_data_norm[right_idx],
                    'prominence': left_dip['prominence'] * 0.5,  # Lower prominence
                    'distance_from_center': abs(freq_axis[right_idx] - center_freq),
                    'synthetic': True
                }
                right_dips.append(synthetic_dip)
        
        # If still no dips on either side, create default pairs
        if not left_dips and not right_dips:
            print("No dips on either side, creating default pairs")
            # Create default pairs around center frequency
            center_idx = np.argmin(np.abs(freq_axis - center_freq))
            freq_range = freq_axis[-1] - freq_axis[0]
            offset_pts = int(0.05 * len(freq_axis))  # 5% of points
            
            left_idx = max(0, center_idx - offset_pts)
            right_idx = min(len(freq_axis) - 1, center_idx + offset_pts)
            
            dip_pairs = [(left_idx, right_idx)]
            return dip_pairs, filtered_dips
        
        # Sort by distance from center
        left_dips.sort(key=lambda x: x['distance_from_center'])
        right_dips.sort(key=lambda x: x['distance_from_center'])
        
        # Form pairs based on similar distance from center
        dip_pairs = []
        used_left = set()
        used_right = set()
        
        # Match left and right dips to form pairs
        for left_dip in left_dips:
            if left_dip['index'] in used_left:
                continue
                
            best_match = None
            best_score = float('inf')
            
            for right_dip in right_dips:
                if right_dip['index'] in used_right:
                    continue
                    
                # Score based on distance from center difference
                dist_diff = abs(left_dip['distance_from_center'] - right_dip['distance_from_center'])
                score = dist_diff / max(1e-10, center_freq)  # Avoid division by zero
                
                if score < best_score:
                    best_score = score
                    best_match = right_dip
            
            if best_match:
                dip_pairs.append((left_dip['index'], best_match['index']))
                used_left.add(left_dip['index'])
                used_right.add(best_match['index'])
        
        # Use any remaining dips as single-dip pairs
        remaining_left = [dip for dip in left_dips if dip['index'] not in used_left]
        remaining_right = [dip for dip in right_dips if dip['index'] not in used_right]
        remaining_dips = remaining_left + remaining_right
        
        # Sort by prominence
        remaining_dips.sort(key=lambda x: -x['prominence'])
        
        # Add single-dip pairs
        for dip in remaining_dips:
            if len(dip_pairs) >= max_pairs:
                break
            dip_pairs.append((dip['index'], dip['index']))
        
        # Ensure we have at least one pair
        if not dip_pairs:
            print("No pairs formed, creating default pair")
            # Create a default pair around center frequency
            center_idx = np.argmin(np.abs(freq_axis - center_freq))
            freq_range = freq_axis[-1] - freq_axis[0]
            offset_pts = int(0.05 * len(freq_axis))  # 5% of points
            
            left_idx = max(0, center_idx - offset_pts)
            right_idx = min(len(freq_axis) - 1, center_idx + offset_pts)
            
            dip_pairs = [(left_idx, right_idx)]
        
        return dip_pairs[:max_pairs], filtered_dips

    @staticmethod
    def estimate_multi_dip_from_center(freq_axis, pixel_data_norm, dip_pairs, center_freq, filtered_dips=None):
        """
        Estimate parameters for multiple dip pairs using a fixed center frequency.
        Fixed to handle empty dip_pairs safely.
        
        Args:
            freq_axis: Frequency axis values
            pixel_data_norm: Normalized pixel data
            dip_pairs: List of dip pair indices (might be empty)
            center_freq: Fixed center frequency
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
        
        # Parameter list for all pairs
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
        
        # If no dip pairs were found, create synthetic ones based on the center frequency
        if not dip_pairs:
            print("No real dip pairs found, creating synthetic ones")
            # Create parameter estimates for default dips around the center
            freq_range = freq_axis[-1] - freq_axis[0]
            
            # First pair - narrow splitting (10% of range)
            A_est_1 = 0.1 * np.ptp(pixel_data_norm)
            w_est_1 = TYPICAL_WIDTH
            f_delta_1 = freq_range * 0.1
            params.extend([A_est_1, A_est_1, w_est_1, w_est_1, f_delta_1])
            
            return I0_est, params
        
        # Process each dip pair
        for pair in dip_pairs:
            # Ensure pair has correct format (might be a single value for synthetic dips)
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
                
            dip1, dip2 = pair
            
            # Check if it's a symmetric pair or a single dip
            if dip1 == dip2:
                # Single dip, create two identical dips
                f_dip = freq_axis[dip1]
                A_est = max(I0_est - smoothed_data[dip1], 0.01 * I0_est) * 2.0
                w_est = estimate_width(dip1)
                
                # Calculate delta based on distance from center
                f_delta = 2 * abs(f_dip - center_freq)
                # Ensure minimum splitting
                f_delta = max(f_delta, 0.010)
                
                params.extend([A_est, A_est, w_est, w_est, f_delta])
            else:
                # Real pair with two different dips
                f_dip_1 = freq_axis[dip1]
                f_dip_2 = freq_axis[dip2]
                
                # Calculate amplitude estimates for each dip
                A_1_est = max(I0_est - smoothed_data[dip1], 0.01 * I0_est) * 2.0  # Multiply by 2.0
                A_2_est = max(I0_est - smoothed_data[dip2], 0.01 * I0_est) * 2.0  # Multiply by 2.0
                
                # Estimate width for each dip
                w_1_est = estimate_width(dip1)
                w_2_est = estimate_width(dip2)
                
                # Calculate splitting based on dip positions
                f_delta = abs(f_dip_2 - f_dip_1)
                
                # If splitting is too small, use a minimum value
                f_delta = max(f_delta, 0.005)
                
                params.extend([A_1_est, A_2_est, w_1_est, w_2_est, f_delta])
        
        # If we still have no parameters, create default ones
        if not params:
            print("Creating default parameters")
            A_est = 0.1 * np.ptp(pixel_data_norm)
            w_est = TYPICAL_WIDTH
            f_delta = 0.010
            params = [A_est, A_est, w_est, w_est, f_delta]
        
        return I0_est, params


    @staticmethod
    def fit_single_pixel_with_fixed_center(pixel_data, freq_axis, n_pairs=3, default_values=None, method='trf', hierarchical=True):
        """
        ODMR fitting with a fixed center frequency determined from outermost dips.
        Can use either simultaneous or hierarchical fitting approach.
        
        Args:
            pixel_data: Raw pixel data
            freq_axis: Frequency axis values
            n_pairs: Number of dip pairs to fit
            default_values: Default parameter values if fitting fails
            method: Fitting method ('trf', 'lm', etc.)
            hierarchical: If True, use hierarchical approach (fit one pair at a time)
            
        Returns:
            dict: Fitted parameters
        """
        # Normalize data
        scale_factor = np.max(np.abs(pixel_data))
        if scale_factor == 0:
            scale_factor = 1
        pixel_data_norm = pixel_data / scale_factor
        
        # Apply smoothing for analysis
        window_length = min(11, len(pixel_data_norm)//2*2+1)
        smoothed_data = savgol_filter(pixel_data_norm, window_length=window_length, polyorder=3)
        
        try:
            # Find outermost dips to determine center frequency
            print("Finding outermost dips...")
            left_idx, right_idx, center_freq = ODMRAnalyzer.find_outermost_dips(freq_axis, smoothed_data)
            print(f"Found center frequency: {center_freq}")
            
            # Ensure we have a valid center frequency
            if center_freq is None or np.isnan(center_freq):
                center_freq = np.mean(freq_axis)
                print(f"Using default center frequency: {center_freq}")
            
            # Calculate reasonable bounds for f_delta
            freq_range = freq_axis[-1] - freq_axis[0]
            max_delta = freq_range * 0.6  # Maximum 60% of frequency range
            
            if hierarchical:
                # Initialize result dictionary with the shared center frequency
                result = {"f_center": center_freq}
                
                # Current data to fit (initially the original data)
                current_data = np.copy(pixel_data_norm)
                
                # Fit one pair at a time
                for pair_idx in range(1, n_pairs + 1):
                    print(f"Fitting pair {pair_idx}...")
                    
                    try:
                        # Find the most prominent dips in the current data
                        dip_pairs, filtered_dips = ODMRAnalyzer.find_dip_pairs_from_center(
                            freq_axis, current_data, center_freq, max_pairs=1
                        )
                        
                        # For pairs after the first, prevent overlapping with previous pairs
                        if pair_idx > 1 and filtered_dips:
                            # Store the original dips before filtering
                            filtered_dips_original = filtered_dips.copy()
                            
                            # Calculate the positions of all previous pairs' dips
                            exclusion_zones = []
                            for prev_idx in range(1, pair_idx):
                                if f"f_delta_{prev_idx}" in result:
                                    prev_f_delta = result[f"f_delta_{prev_idx}"]
                                    prev_width1 = result[f"width_{prev_idx}_1"]
                                    prev_width2 = result[f"width_{prev_idx}_2"]
                                    
                                    # Calculate the positions of the previous pair's dips
                                    prev_f_1 = result["f_center"] - 0.5 * prev_f_delta
                                    prev_f_2 = result["f_center"] + 0.5 * prev_f_delta
                                    
                                    # Define exclusion zones around previous dips (3x width)
                                    exclusion_width1 = 3.0 * prev_width1
                                    exclusion_width2 = 3.0 * prev_width2
                                    
                                    exclusion_zones.append((prev_f_1 - exclusion_width1, prev_f_1 + exclusion_width1))
                                    exclusion_zones.append((prev_f_2 - exclusion_width2, prev_f_2 + exclusion_width2))
                            
                            # Filter out dips that overlap with exclusion zones
                            if exclusion_zones:
                                filtered_dips_new = []
                                for dip in filtered_dips:
                                    dip_freq = dip['frequency']
                                    
                                    # Check if this dip is inside any exclusion zone
                                    in_exclusion_zone = False
                                    for zone_start, zone_end in exclusion_zones:
                                        if zone_start <= dip_freq <= zone_end:
                                            in_exclusion_zone = True
                                            break
                                    
                                    if not in_exclusion_zone:
                                        filtered_dips_new.append(dip)
                                
                                # Replace filtered_dips with the non-overlapping ones
                                filtered_dips = filtered_dips_new
                                
                                # Re-form dip pairs with the filtered dips
                                if filtered_dips:
                                    # Create a temporary function to find pairs from provided dips
                                    dip_pairs_new = []
                                    
                                    # Sort dips into left and right of center
                                    left_dips = [dip for dip in filtered_dips if dip['frequency'] < center_freq]
                                    right_dips = [dip for dip in filtered_dips if dip['frequency'] >= center_freq]
                                    
                                    # Sort by distance from center for better pairing
                                    left_dips.sort(key=lambda x: x['distance_from_center'])
                                    right_dips.sort(key=lambda x: x['distance_from_center'])
                                    
                                    # Match left and right dips to form pairs
                                    if left_dips and right_dips:
                                        # Use most prominent dips from each side
                                        left_dips.sort(key=lambda x: -x['prominence'])
                                        right_dips.sort(key=lambda x: -x['prominence'])
                                        dip_pairs_new = [(left_dips[0]['index'], right_dips[0]['index'])]
                                    elif left_dips:
                                        # Only left dips available
                                        left_dips.sort(key=lambda x: -x['prominence'])
                                        dip_pairs_new = [(left_dips[0]['index'], left_dips[0]['index'])]
                                    elif right_dips:
                                        # Only right dips available
                                        right_dips.sort(key=lambda x: -x['prominence'])
                                        dip_pairs_new = [(right_dips[0]['index'], right_dips[0]['index'])]
                                    
                                    # Use the newly formed pairs if any
                                    if dip_pairs_new:
                                        dip_pairs = dip_pairs_new
                                    else:
                                        print(f"No non-overlapping dips found for pair {pair_idx}")
                                        continue  # Skip this pair
                                else:
                                    print(f"No non-overlapping dips found for pair {pair_idx}")
                                    continue  # Skip this pair
                        
                        if not dip_pairs:
                            print(f"No more significant dips found for pair {pair_idx}")
                            break
                        
                        # Estimate parameters for this pair
                        I0_est, pair_params = ODMRAnalyzer.estimate_multi_dip_from_center(
                            freq_axis, current_data, dip_pairs, center_freq, filtered_dips
                        )
                        
                        # For second pair onward, ensure different f_delta
                        if pair_idx > 1:
                            # Get all previously fitted f_delta values
                            prev_f_deltas = [result[f"f_delta_{i}"] for i in range(1, pair_idx) 
                                            if f"f_delta_{i}" in result]
                            
                            # Generate a set of candidate f_delta values across the range
                            candidate_deltas = [0.01, 0.03, 0.05, 0.08, 0.12, 0.2]
                            
                            # Find the candidate that's most different from existing values
                            best_delta = None
                            max_diff = -1
                            
                            for candidate in candidate_deltas:
                                # Skip candidates larger than max allowed delta
                                if candidate > max_delta:
                                    continue
                                    
                                # Calculate minimum difference from existing deltas
                                min_diff = float('inf')
                                for prev_delta in prev_f_deltas:
                                    diff = abs(candidate - prev_delta)
                                    min_diff = min(min_diff, diff)
                                
                                # If this candidate is more distinct than previous best, use it
                                if min_diff > max_diff:
                                    max_diff = min_diff
                                    best_delta = candidate
                            
                            # Use the most distinct delta if found
                            if best_delta is not None:
                                # Override the estimated f_delta
                                pair_params[4] = best_delta
                                print(f"Setting initial f_delta_{pair_idx} = {best_delta} to differ from previous pairs")
                        
                        # Convert to log space for parameters that need it
                        epsilon = 1e-10
                        log_I0_est = np.log(max(I0_est, epsilon))
                        log_A_1 = np.log(max(pair_params[0], epsilon))
                        log_A_2 = np.log(max(pair_params[1], epsilon))
                        log_w_1 = np.log(max(pair_params[2], epsilon))
                        log_w_2 = np.log(max(pair_params[3], epsilon))
                        f_delta = min(pair_params[4], max_delta)  # Restrict to reasonable range
                        
                        # Initial parameters for single pair fitting
                        p0 = [log_I0_est, log_A_1, log_A_2, log_w_1, log_w_2, center_freq, f_delta]
                        
                        # Set reasonable bounds
                        if method == 'trf':
                            center_margin = freq_range * 0.05
                            f_min = max(freq_axis[0], center_freq - center_margin)
                            f_max = min(freq_axis[-1], center_freq + center_margin)
                            
                            lower_bounds = [
                                np.log(0.0001),  # log_I0
                                np.log(1e-6),    # log_A_1
                                np.log(1e-6),    # log_A_2
                                np.log(0.001),   # log_w_1
                                np.log(0.001),   # log_w_2
                                f_min,           # f_center
                                0.005            # f_delta
                            ]
                            
                            upper_bounds = [
                                np.log(1000.0),  # log_I0
                                np.log(100.0),   # log_A_1
                                np.log(100.0),   # log_A_2
                                np.log(0.005),    # log_w_1
                                np.log(0.005),    # log_w_2
                                f_max,           # f_center
                                max_delta        # f_delta - use calculated max
                            ]
                            
                            bounds = (lower_bounds, upper_bounds)
                            
                            # Clip initial parameters to bounds
                            for i, (param, (lower, upper)) in enumerate(zip(p0, zip(*bounds))):
                                p0[i] = np.clip(param, lower, upper)
                        else:
                            bounds = (-np.inf, np.inf)
                        
                        # Define single-pair fitting function
                        def single_pair_func(f, log_I0, log_A_1, log_A_2, log_w_1, log_w_2, f_center, f_delta):
                            I0 = np.exp(log_I0)
                            A_1 = np.exp(log_A_1)
                            A_2 = np.exp(log_A_2)
                            w_1 = np.exp(log_w_1)
                            w_2 = np.exp(log_w_2)
                            
                            # Calculate dip positions
                            f_1 = f_center - 0.5 * abs(f_delta)
                            f_2 = f_center + 0.5 * abs(f_delta)
                            
                            # Calculate dips
                            dip_1 = A_1 / (1 + ((f_1 - f) / w_1) ** 2)
                            dip_2 = A_2 / (1 + ((f_2 - f) / w_2) ** 2)
                            
                            return I0 - dip_1 - dip_2
                        
                        # Create weights that emphasize dips and give more importance to f_delta and amplitude
                        weights = np.ones_like(freq_axis)

                        # Add extra weight around detected dips
                        for pair in dip_pairs:
                            for dip_idx in pair:
                                # Create a window of higher weight around each dip
                                center = dip_idx
                                width = 5  # Points on each side
                                left = max(0, center - width)
                                right = min(len(weights) - 1, center + width)
                                weights[left:right+1] = 5.0  # 5x weight in dip regions
                        
                        # Fit this pair
                        popt, pcov = curve_fit(
                            single_pair_func,
                            freq_axis,
                            current_data,
                            p0=p0,
                            bounds=bounds if method == 'trf' else (-np.inf, np.inf),
                            method=method,
                            sigma=1/weights,  # Lower sigma means higher weight
                            maxfev=10000,
                            ftol=1e-6,
                            xtol=1e-6
                        )
                        
                        # Store results for this pair
                        if pair_idx == 1:
                            # For first pair, store I0 and center frequency
                            result["I0"] = np.exp(popt[0]) * scale_factor
                            result["f_center"] = popt[5]  # Update center frequency
                        
                        # Store pair-specific parameters
                        result[f"A_{pair_idx}_1"] = np.exp(popt[1]) * scale_factor
                        result[f"A_{pair_idx}_2"] = np.exp(popt[2]) * scale_factor
                        result[f"width_{pair_idx}_1"] = np.exp(popt[3])
                        result[f"width_{pair_idx}_2"] = np.exp(popt[4])
                        result[f"f_delta_{pair_idx}"] = abs(popt[6])
                        
                        # Calculate fitted curve for this pair
                        fitted_pair = single_pair_func(freq_axis, *popt)
                        
                        # Calculate residuals for next iteration
                        current_data = current_data - (fitted_pair - np.exp(popt[0]))
                        
                        # Stop if residuals are small
                        residual_amplitude = np.ptp(current_data)
                        if residual_amplitude < 0.01:
                            print(f"Residuals are small, stopping after {pair_idx} pairs")
                            break
                            
                    except Exception as e:
                        print(f"Error fitting pair {pair_idx}: {str(e)}")
                        break
                
                # After all pairs are fitted individually, do a final joint refinement
                try:
                    print("Performing final joint refinement of all pairs...")
                    
                    # Count how many pairs we actually fitted
                    fitted_pairs = 0
                    for i in range(1, n_pairs + 1):
                        if f"A_{i}_1" in result:
                            fitted_pairs += 1
                    
                    if fitted_pairs >= 1:
                        # Build initial parameter vector for joint optimization
                        joint_p0 = [np.log(result["I0"] / scale_factor)]
                        
                        # Add all fitted pairs
                        for i in range(1, fitted_pairs + 1):
                            # Check if this pair was successfully fitted
                            if f"A_{i}_1" in result:
                                log_A_i_1 = np.log(max(result[f"A_{i}_1"] / scale_factor, epsilon))
                                log_A_i_2 = np.log(max(result[f"A_{i}_2"] / scale_factor, epsilon))
                                log_w_i_1 = np.log(max(result[f"width_{i}_1"], epsilon))
                                log_w_i_2 = np.log(max(result[f"width_{i}_2"], epsilon))
                                
                                if i == 1:
                                    # First pair includes center frequency
                                    joint_p0.extend([
                                        log_A_i_1, log_A_i_2, log_w_i_1, log_w_i_2,
                                        result["f_center"], result[f"f_delta_{i}"]
                                    ])
                                else:
                                    # Subsequent pairs only need f_delta
                                    joint_p0.extend([
                                        log_A_i_1, log_A_i_2, log_w_i_1, log_w_i_2,
                                        result[f"f_delta_{i}"]
                                    ])
                        
                        # Create bounds for joint optimization
                        if method == 'trf':
                            joint_lower_bounds = [np.log(0.0001)]  # log_I0
                            joint_upper_bounds = [np.log(1000.0)]  # log_I0
                            
                            center_margin = freq_range * 0.05
                            f_min = max(freq_axis[0], center_freq - center_margin)
                            f_max = min(freq_axis[-1], center_freq + center_margin)
                            
                            for i in range(1, fitted_pairs + 1):
                                # Add bounds for each pair's parameters
                                joint_lower_bounds.extend([
                                    np.log(1e-6),  # log_A_i_1
                                    np.log(1e-6),  # log_A_i_2
                                    np.log(0.001), # log_w_i_1
                                    np.log(0.001)  # log_w_i_2
                                ])
                                
                                joint_upper_bounds.extend([
                                    np.log(100.0),  # log_A_i_1
                                    np.log(100.0),  # log_A_i_2
                                    np.log(0.005),   # log_w_i_1
                                    np.log(0.005)    # log_w_i_2
                                ])
                                
                                if i == 1:
                                    # First pair includes center frequency bounds
                                    joint_lower_bounds.extend([f_min, 0.005])
                                    joint_upper_bounds.extend([f_max, max_delta])
                                else:
                                    # Subsequent pairs only have f_delta bounds
                                    joint_lower_bounds.append(0.005)
                                    joint_upper_bounds.append(max_delta)
                            
                            joint_bounds = (joint_lower_bounds, joint_upper_bounds)
                        else:
                            joint_bounds = (-np.inf, np.inf)
                        
                        # Define multi-dip function for joint fitting
                        def multi_dip_func_for_joint_fit(x, *params):
                            return ODMRAnalyzer.multi_dip_func_shared_center(x, *params)
                        
                        # Create weights emphasizing dip regions for joint fit
                        joint_weights = np.ones_like(freq_axis)
                        
                        # Add weight around all fitted dips
                        for i in range(1, fitted_pairs + 1):
                            if f"f_delta_{i}" in result:
                                f_c = result["f_center"]
                                f_d = result[f"f_delta_{i}"]
                                
                                # Calculate dip positions
                                dip1_freq = f_c - 0.5 * f_d
                                dip2_freq = f_c + 0.5 * f_d
                                
                                # Find nearest indices
                                dip1_idx = np.argmin(np.abs(freq_axis - dip1_freq))
                                dip2_idx = np.argmin(np.abs(freq_axis - dip2_freq))
                                
                                # Add weight around these dips
                                for dip_idx in [dip1_idx, dip2_idx]:
                                    width = 5  # Points on each side
                                    left = max(0, dip_idx - width)
                                    right = min(len(joint_weights) - 1, dip_idx + width)
                                    joint_weights[left:right+1] = 5.0  # 5x weight
                        
                        # Perform joint optimization
                        joint_popt, joint_pcov = curve_fit(
                            multi_dip_func_for_joint_fit,
                            freq_axis,
                            pixel_data_norm,
                            p0=joint_p0,
                            bounds=joint_bounds if method == 'trf' else (-np.inf, np.inf),
                            method=method,
                            sigma=1/joint_weights,
                            maxfev=10000,
                            ftol=1e-6,
                            xtol=1e-6
                        )
                        
                        # Update result with jointly optimized parameters
                        # Extract baseline
                        result["I0"] = np.exp(joint_popt[0]) * scale_factor
                        
                        # Extract center frequency
                        result["f_center"] = joint_popt[5]
                        
                        # Extract parameters for each pair
                        param_index = 1  # Start after log_I0
                        
                        for i in range(1, fitted_pairs + 1):
                            if i == 1:
                                # First pair
                                result[f"A_{i}_1"] = np.exp(joint_popt[param_index]) * scale_factor
                                result[f"A_{i}_2"] = np.exp(joint_popt[param_index+1]) * scale_factor
                                result[f"width_{i}_1"] = np.exp(joint_popt[param_index+2])
                                result[f"width_{i}_2"] = np.exp(joint_popt[param_index+3])
                                result[f"f_delta_{i}"] = abs(joint_popt[param_index+5])
                                param_index += 6  # log_A1, log_A2, log_w1, log_w2, f_c, f_d
                            else:
                                # Subsequent pairs
                                result[f"A_{i}_1"] = np.exp(joint_popt[param_index]) * scale_factor
                                result[f"A_{i}_2"] = np.exp(joint_popt[param_index+1]) * scale_factor
                                result[f"width_{i}_1"] = np.exp(joint_popt[param_index+2])
                                result[f"width_{i}_2"] = np.exp(joint_popt[param_index+3])
                                result[f"f_delta_{i}"] = abs(joint_popt[param_index+4])
                                param_index += 5  # log_A1, log_A2, log_w1, log_w2, f_d
                        
                        print(f"Joint refinement complete. New parameters:")
                        print(f"I0 = {result['I0']:.3e}, f_center = {result['f_center']:.4f}")
                        for i in range(1, fitted_pairs + 1):
                            print(f"Pair {i}: A1={result[f'A_{i}_1']:.3e}, A2={result[f'A_{i}_2']:.3e}, "
                                f"w1={result[f'width_{i}_1']:.4f}, w2={result[f'width_{i}_2']:.4f}, "
                                f"fd={result[f'f_delta_{i}']:.4f}")
                except Exception as e:
                    print(f"Joint refinement failed: {str(e)}")
                    # Continue with individual fit results
                
                # Fill in compatibility parameters with first pair if we fit any pairs
                if "A_1_1" in result:
                    result["A"] = (result["A_1_1"] + result["A_1_2"]) / 2
                    result["width"] = (result["width_1_1"] + result["width_1_2"]) / 2
                    result["f_delta"] = result["f_delta_1"]
                else:
                    # No pairs were successfully fit, return default values
                    result = {
                        "I0": np.mean(pixel_data),
                        "f_center": center_freq,
                        "A": np.ptp(pixel_data) * 0.1,
                        "width": 0.006,
                        "f_delta": 0.010,
                    }
                    
                    # Add pair-specific defaults
                    for i in range(n_pairs):
                        result[f"A_{i+1}_1"] = result["A"]
                        result[f"A_{i+1}_2"] = result["A"]
                        result[f"width_{i+1}_1"] = result["width"]
                        result[f"width_{i+1}_2"] = result["width"]
                        result[f"f_delta_{i+1}"] = result["f_delta"] + i * 0.005
                
                # Calculate overall fit quality
                fitted_curve = np.ones_like(freq_axis) * result["I0"] / scale_factor
                
                # Subtract all fitted pairs
                for i in range(1, n_pairs + 1):
                    if f"A_{i}_1" in result:
                        A_i_1 = result[f"A_{i}_1"] / scale_factor
                        A_i_2 = result[f"A_{i}_2"] / scale_factor
                        w_i_1 = result[f"width_{i}_1"]
                        w_i_2 = result[f"width_{i}_2"]
                        f_d_i = result[f"f_delta_{i}"]
                        f_c = result["f_center"]
                        
                        # Calculate dip positions
                        f_1 = f_c - 0.5 * f_d_i
                        f_2 = f_c + 0.5 * f_d_i
                        
                        # Calculate and subtract dips
                        dip_1 = A_i_1 / (1 + ((f_1 - freq_axis) / w_i_1) ** 2)
                        dip_2 = A_i_2 / (1 + ((f_2 - freq_axis) / w_i_2) ** 2)
                        fitted_curve -= (dip_1 + dip_2)
                
                # Calculate quality score
                result["quality_score"] = calculate_fit_quality(pixel_data_norm, fitted_curve)
                print(f"Final quality score: {result['quality_score']:.3f}")
                
                return result
                
            else:
                # Original simultaneous fitting approach
                # Find dip pairs based on fixed center frequency
                print("Finding dip pairs...")
                dip_pairs, filtered_dips = ODMRAnalyzer.find_dip_pairs_from_center(
                    freq_axis, smoothed_data, center_freq, max_pairs=n_pairs
                )
                print(f"Found {len(dip_pairs)} dip pairs")
                
                # Estimate initial parameters using fixed center frequency
                print("Estimating parameters...")
                I0_est, pair_params = ODMRAnalyzer.estimate_multi_dip_from_center(
                    freq_axis, smoothed_data, dip_pairs, center_freq, filtered_dips
                )
                print(f"Estimated I0: {I0_est}, params length: {len(pair_params)}")
                
                # Ensure we have enough parameters for requested n_pairs
                while len(pair_params) < 5 * n_pairs:
                    # Add default parameters for additional pairs
                    A_est = 0.05 * np.ptp(pixel_data_norm)
                    w_est = 0.006
                    f_delta = 0.010 + (len(pair_params) // 5) * 0.005  # Slightly different spacing
                    
                    pair_params.extend([A_est, A_est, w_est, w_est, f_delta])
                
                # Convert to log space for parameters that need it
                epsilon = 1e-10
                log_I0_est = np.log(max(I0_est, epsilon))
                
                # Initial parameter vector: [log_I0, log_A_1_1, log_A_1_2, log_w_1_1, log_w_1_2, f_center, f_d_1, ...]
                p0 = [log_I0_est]

                # First pair includes amplitudes, widths, center frequency, and delta
                log_A_1_1 = np.log(max(pair_params[0], epsilon))
                log_A_1_2 = np.log(max(pair_params[1], epsilon))
                log_w_1_1 = np.log(max(pair_params[2], epsilon))
                log_w_1_2 = np.log(max(pair_params[3], epsilon))
                f_delta_1 = min(pair_params[4], max_delta)  # Restrict to reasonable range

                # Add first pair parameters with center frequency
                p0.extend([log_A_1_1, log_A_1_2, log_w_1_1, log_w_1_2, center_freq, f_delta_1])

                # Add subsequent pairs
                for i in range(1, n_pairs):
                    idx = i * 5  # Each pair has 5 parameters
                    
                    # Convert amplitude and width to log space
                    log_A_i_1 = np.log(max(pair_params[idx], epsilon))
                    log_A_i_2 = np.log(max(pair_params[idx + 1], epsilon))
                    log_w_i_1 = np.log(max(pair_params[idx + 2], epsilon))
                    log_w_i_2 = np.log(max(pair_params[idx + 3], epsilon))
                    f_delta_i = min(pair_params[idx + 4], max_delta)  # Restrict to reasonable range
                    
                    # Each subsequent pair only adds 5 parameters (no center frequency)
                    p0.extend([log_A_i_1, log_A_i_2, log_w_i_1, log_w_i_2, f_delta_i])
                                        
                # Set bounds for TRF method
                if method == 'trf':
                    # Lower bounds
                    lower_bounds = [np.log(0.0001)]  # log_I0
                    
                    # Narrow bounds for center frequency since we're confident about it
                    center_margin = (freq_axis[-1] - freq_axis[0]) * 0.05  # 5% margin
                    f_min = max(np.min(freq_axis), center_freq - center_margin)
                    f_max = min(np.max(freq_axis), center_freq + center_margin)
                    
                    for i in range(n_pairs):
                        # Amplitude and width bounds
                        lower_bounds.extend([
                            np.log(1e-6),  # log_A_i_1
                            np.log(1e-6),  # log_A_i_2
                            np.log(0.002), # log_w_i_1
                            np.log(0.002)  # log_w_i_2
                        ])
                        
                        # Center frequency and splitting bounds
                        if i == 0:
                            lower_bounds.extend([
                                f_min,      # Center frequency with tight bounds
                                0.005       # Minimum splitting
                            ])
                        else:
                            lower_bounds.append(0.005)  # Minimum splitting for other pairs
                    
                    # Upper bounds
                    upper_bounds = [np.log(1000.0)]  # log_I0 
                    
                    for i in range(n_pairs):
                        # Amplitude and width bounds
                        upper_bounds.extend([
                            np.log(100.0),  # log_A_i_1
                            np.log(100.0),  # log_A_i_2
                            np.log(0.005),   # log_w_i_1
                            np.log(0.005)    # log_w_i_2
                        ])
                        
                        # Center frequency and splitting bounds
                        if i == 0:
                            upper_bounds.extend([
                                f_max,     # Center frequency with tight bounds
                                max_delta  # Maximum splitting - use calculated max
                            ])
                        else:
                            upper_bounds.append(max_delta)  # Maximum splitting for other pairs
                    
                    bounds = (lower_bounds, upper_bounds)
                    
                    # Clip initial parameters to bounds
                    for i, (param, (lower, upper)) in enumerate(zip(p0, zip(*bounds))):
                        p0[i] = np.clip(param, lower, upper)
                else:
                    bounds = (-np.inf, np.inf)
                
                # Prepare function for fitting with appropriate parameters
                def fitting_function(x, *params):
                    return ODMRAnalyzer.multi_dip_func_shared_center(x, *params)
                
                # Perform fitting
                print("Starting curve fitting...")
                print(f"Initial parameters: {p0}")
                
                popt, pcov = curve_fit(
                    fitting_function,
                    freq_axis,
                    pixel_data_norm,
                    p0=p0,
                    bounds=bounds if method == 'trf' else (-np.inf, np.inf),
                    method=method,
                    maxfev=10000,  # Increase max iterations
                    ftol=1e-5,
                    xtol=1e-5
                )
                
                # Calculate fit quality
                fitted_curve = fitting_function(freq_axis, *popt)
                quality_score = calculate_fit_quality(pixel_data_norm, fitted_curve)
                
                # Process results into a more usable format
                result = {}
                
                # Extract baseline
                result["I0"] = np.exp(popt[0]) * scale_factor
                
                # Extract center frequency (index 5 after log_I0 and first pair's amplitude/width params)
                result["f_center"] = popt[5]
                
                # Extract parameters for each pair
                for i in range(n_pairs):
                    if i == 0:
                        # First pair includes the center frequency
                        base_idx = 1  # Start after log_I0
                        f_d_idx = 6   # After center frequency
                    else:
                        # Subsequent pairs
                        base_idx = 6 + (i-1) * 5  # f_center + params for previous pairs
                        f_d_idx = base_idx + 4    # Last parameter in the group
                    
                    result[f"A_{i+1}_1"] = np.exp(popt[base_idx]) * scale_factor
                    result[f"A_{i+1}_2"] = np.exp(popt[base_idx + 1]) * scale_factor
                    result[f"width_{i+1}_1"] = np.exp(popt[base_idx + 2])
                    result[f"width_{i+1}_2"] = np.exp(popt[base_idx + 3])
                    result[f"f_delta_{i+1}"] = abs(popt[f_d_idx])  # Force positive delta
                
                # For compatibility with double-dip code
                if n_pairs >= 1:
                    result["A"] = (result["A_1_1"] + result["A_1_2"]) / 2
                    result["width"] = (result["width_1_1"] + result["width_1_2"]) / 2
                    result["f_delta"] = result["f_delta_1"]
                
                # Add quality score
                result["quality_score"] = quality_score
                
                return result

        except Exception as e:
            print(f"Fixed-center fitting failed: {str(e)}")
            traceback.print_exc()  # Add traceback to see full error
            
            # Return default values or create basic defaults
            if default_values is None:
                default_values = {
                    "I0": np.mean(pixel_data),
                    "f_center": np.mean(freq_axis),  # Use mean frequency as fallback
                }
                
                # Add pair-specific defaults
                for i in range(n_pairs):
                    default_values[f"A_{i+1}_1"] = np.ptp(pixel_data) * 0.1
                    default_values[f"A_{i+1}_2"] = np.ptp(pixel_data) * 0.1
                    default_values[f"width_{i+1}_1"] = 0.006
                    default_values[f"width_{i+1}_2"] = 0.006
                    default_values[f"f_delta_{i+1}"] = 0.010 + i * 0.005  # Different spacing for each pair
                
                # Add compatibility defaults
                default_values["A"] = np.ptp(pixel_data) * 0.1
                default_values["width"] = 0.006
                default_values["f_delta"] = 0.010
                default_values["quality_score"] = 0.0
            
            return default_values


    def fit_multi_lorentzian_with_fixed_center(self, n_pairs=2, method='trf', output_dir=None):
        """
        Parallel version of multi-dip Lorentzian fitting using the fixed center approach.
        
        Args:
            n_pairs: Number of dip pairs to fit (fixed)
            method: Fitting method ('trf', 'lm', etc.)
            output_dir: Directory to save results
        
        Returns:
            tuple: (output_directory, fitted_parameters_file)
        """
        self.start_profiling()
        
        M, N, F = self.data.shape
        
        # Initialize parameter storage
        fitted_params = {
            "I0": np.zeros((M, N)),
            "f_center": np.zeros((M, N)),
            "quality_score": np.zeros((M, N)),
        }
        
        # Add parameters for each potential pair
        for i in range(n_pairs):
            fitted_params[f"A_{i+1}_1"] = np.zeros((M, N))
            fitted_params[f"A_{i+1}_2"] = np.zeros((M, N))
            fitted_params[f"width_{i+1}_1"] = np.zeros((M, N))
            fitted_params[f"width_{i+1}_2"] = np.zeros((M, N))
            fitted_params[f"f_delta_{i+1}"] = np.zeros((M, N))
        
        # Add compatibility parameters
        fitted_params["A"] = np.zeros((M, N))
        fitted_params["width"] = np.zeros((M, N))
        fitted_params["f_delta"] = np.zeros((M, N))
        
        # Default values
        default_values = {
            "I0": 1.0,
            "f_center": np.mean(self.freq_axis)
        }
        
        # Add defaults for each potential pair
        for i in range(n_pairs):
            default_values[f"A_{i+1}_1"] = 0.1
            default_values[f"A_{i+1}_2"] = 0.1
            default_values[f"width_{i+1}_1"] = 0.006
            default_values[f"width_{i+1}_2"] = 0.006
            default_values[f"f_delta_{i+1}"] = 0.010 + 0.005 * i  # Slightly different spacing
        
        # Add compatibility defaults
        default_values["A"] = 0.1
        default_values["width"] = 0.006
        default_values["f_delta"] = 0.010
        default_values["quality_score"] = 0.0

        print(f"Processing {M*N} pixels using multiprocessing...")
        print(f"Fitting model: {n_pairs} dip pair(s) with fixed center approach")
        
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
            for m, row_results in tqdm(pool.imap(process_pixel_row_with_fixed_center, row_args), 
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
            fitted_params_file = os.path.join(output_dir, f"{base_name}_fixed_center_params.npy")
            quality_stats_file = os.path.join(output_dir, f"{base_name}_fixed_center_quality_stats.txt")
            
            # Save all parameters
            param_order = ['I0', 'f_center']
            
            # Add parameters for each pair
            for i in range(n_pairs):
                param_order.extend([
                    f"A_{i+1}_1", f"A_{i+1}_2", 
                    f"width_{i+1}_1", f"width_{i+1}_2", 
                    f"f_delta_{i+1}"
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
                f.write(f"Fixed Center Multi-Dip Fitting Quality Statistics:\n")
                f.write(f"Success Rate (score >= 0.9): {success_rate:.1f}%\n")
                f.write(f"Mean Quality Score: {mean_quality:.3f}\n")
                f.write(f"Median Quality Score: {median_quality:.3f}\n")
                
                # Add performance metrics to the same file
                f.write(f"\nODMR Fixed Center Fitting Performance Statistics:\n")
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
            print(f"  - Fixed center parameters: {os.path.basename(fitted_params_file)}")
            print(f"  - Compatible parameters: {os.path.basename(compat_params_file)}")
            print(f"  - Quality statistics: {os.path.basename(quality_stats_file)}")
            print(f"  - Original data: {os.path.basename(new_data_file)}")
            print(f"  - JSON parameters: {os.path.basename(new_json_file)}")
            
            return output_dir, fitted_params_file
        
        self.stop_profiling()
        return None, None

    def plot_multi_dip_fit_fixed(self, x, y, fit_result=None, n_pairs=2):
        """
        Plot spectrum with multi-dip fit using the fixed center approach
        
        Args:
            x (int): x-coordinate of pixel
            y (int): y-coordinate of pixel
            fit_result (dict, optional): Fitted parameters to overlay
            n_pairs (int): Number of dip pairs to plot
        
        Returns:
            tuple: (figure, axes)
        """
        spectrum = self.data[x, y, :]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot original data points
        ax.scatter(self.freq_axis, spectrum, color='red', alpha=0.5, label='Raw Data Points', zorder=2)
        
        # Apply Savitzky-Golay filter for smoothing
        window_length = min(11, len(spectrum)//2*2+1)
        smoothed_spectrum = savgol_filter(spectrum, window_length=window_length, polyorder=3)
        ax.plot(self.freq_axis, smoothed_spectrum, 'b-', linewidth=2, label='Smoothed Spectrum', zorder=3)
        
        # Find center frequency from outermost dips
        left_idx, right_idx, center_freq = self.find_outermost_dips(self.freq_axis, smoothed_spectrum)
        
        # Find dips based on center frequency and plot them
        dip_pairs, filtered_dips = self.find_dip_pairs_from_center(
            self.freq_axis, smoothed_spectrum, center_freq, max_pairs=n_pairs
        )
        
        # Plot detected dips
        for i, pair in enumerate(dip_pairs):
            dip1, dip2 = pair
            for dip_idx in pair:
                ax.axvline(x=self.freq_axis[dip_idx], color='g', alpha=0.3, linestyle='--')
                ax.text(self.freq_axis[dip_idx], spectrum[dip_idx], 
                        f'{self.freq_axis[dip_idx]:.3f} GHz', 
                        rotation=90, verticalalignment='bottom')
        
        # Plot center frequency
        ax.axvline(x=center_freq, color='black', alpha=0.3, linestyle='-.')
        ax.text(center_freq, np.min(spectrum), 
                f'Center: {center_freq:.3f} GHz', 
                rotation=90, verticalalignment='bottom', color='black')
        
        # Plot fitted curve if parameters are provided
        if fit_result is not None:
            # Build parameters for the multi dip function
            I0 = fit_result['I0']
            f_center = fit_result['f_center']
            
            # Parameters list for the fitting function
            params = []
            colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Colors for different dip pairs
            
            # Plot combined fitted curve
            fitted_y = np.ones_like(self.freq_axis) * I0
            
            for i in range(1, n_pairs+1):
                # Check if this pair's parameters exist
                if f"A_{i}_1" in fit_result:
                    A_i_1 = fit_result[f"A_{i}_1"]
                    A_i_2 = fit_result[f"A_{i}_2"]
                    w_i_1 = fit_result[f"width_{i}_1"]
                    w_i_2 = fit_result[f"width_{i}_2"]
                    f_d_i = fit_result[f"f_delta_{i}"]
                    
                    # Calculate dip positions
                    f_1 = f_center - 0.5 * f_d_i
                    f_2 = f_center + 0.5 * f_d_i
                    
                    # Calculate individual dips
                    dip_1 = A_i_1 / (1 + ((f_1 - self.freq_axis) / w_i_1) ** 2)
                    dip_2 = A_i_2 / (1 + ((f_2 - self.freq_axis) / w_i_2) ** 2)
                    
                    # Subtract dips from fitted curve
                    fitted_y -= (dip_1 + dip_2)
                    
                    # Plot individual pair with distinct color
                    pair_curve = I0 - dip_1 - dip_2
                    color = colors[i % len(colors)]
                    ax.plot(self.freq_axis, pair_curve, '--', color=color, alpha=0.5, 
                            label=f'Pair {i} (Δf = {f_d_i:.3f} GHz)', linewidth=1.5)
                    
                    # Mark dip centers
                    ax.axvline(x=f_1, color=color, alpha=0.2, linestyle=':')
                    ax.axvline(x=f_2, color=color, alpha=0.2, linestyle=':')
                    
                    # Add annotations
                    ax.text(f_1, I0 * 0.97, f'Dip {i}.1: {f_1:.3f} GHz\nA: {A_i_1:.3e}, w: {w_i_1:.3f}', 
                            rotation=90, verticalalignment='top', fontsize=8, color=color)
                    ax.text(f_2, I0 * 0.97, f'Dip {i}.2: {f_2:.3f} GHz\nA: {A_i_2:.3e}, w: {w_i_2:.3f}', 
                            rotation=90, verticalalignment='top', fontsize=8, color=color)
            
            # Plot overall fitted curve
            ax.plot(self.freq_axis, fitted_y, 'r-', label='Combined Fit', linewidth=2, zorder=4)
            
            # Add fit quality information
            if 'quality_score' in fit_result:
                ax.text(0.02, 0.02, f'Fit Quality: {fit_result["quality_score"]:.3f}', 
                        transform=ax.transAxes, fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('ODMR Signal (a.u.)')
        ax.set_title(f'Multi-Dip ODMR Spectrum with Fixed Center ({x}, {y})')
        ax.set_xlim([self.freq_axis[0], self.freq_axis[-1]])
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig, ax

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
            
            output_dir = input("Enter output directory path (press Enter for default './fitted_parameters'): ").strip() or "./fitted_parameters"
            output_dir, fitted_params_file = analyzer.fit_multi_lorentzian_with_fixed_center(
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
                
                pixel = analyzer.data[x, y, :]
                single_pixel_params = analyzer.fit_single_pixel_multi_dip(
                    pixel, analyzer.freq_axis, n_pairs=n_pairs
                )
                
                fig_spectrum, ax_spectrum = analyzer.plot_multi_dip_fit(
                    x, y, fit_result=single_pixel_params, n_pairs=n_pairs
                )
                plt.show()
                
                print("\nFitted Parameters:")
                print(f"I0: {single_pixel_params['I0']:.4f}")
                print(f"f_center: {single_pixel_params['f_center']:.4f}")
                
                for i in range(1, n_pairs + 1):
                    print(f"\nPair {i}:")
                    print(f"  A_{i}_1: {single_pixel_params[f'A_{i}_1']:.4e}")
                    print(f"  A_{i}_2: {single_pixel_params[f'A_{i}_2']:.4e}")
                    print(f"  width_{i}_1: {single_pixel_params[f'width_{i}_1']:.4f}")
                    print(f"  width_{i}_2: {single_pixel_params[f'width_{i}_2']:.4f}")
                    print(f"  f_delta_{i}: {single_pixel_params[f'f_delta_{i}']:.4f}")
                
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
            except ValueError:
                print("Invalid input. Using default (2 pairs).")
                n_pairs = 2
            
            output_dir = input("Enter output directory path (press Enter for default './fitted_parameters'): ").strip() or "./fitted_parameters"
            
            try:
                process_directory(directory, method=method_choice, output_dir=output_dir, n_pairs=n_pairs)
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