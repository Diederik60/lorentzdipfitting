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

def process_directory(directory_path, method='trf', output_dir="./fitted_parameters"):
    """
    Recursively process ODMR datasets and organize results
    """
    directory = Path(directory_path)
    
    # Process all subdirectories first
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if subdirs:
        print(f"\nFound {len(subdirs)} subdirectories in {directory_path}")
        for subdir in subdirs:
            print(f"\nProcessing subdirectory: {subdir.name}")
            # Create corresponding output subdirectory
            subdir_output = Path(output_dir) / subdir.name
            process_directory(subdir, method=method, output_dir=str(subdir_output))
    
    # Find all ODMR scan files in current directory
    npy_files = list(directory.glob('2D_ODMR_scan_*.npy'))
    total_files = len(npy_files)
    
    if total_files > 0:
        print(f"\nFound {total_files} matching ODMR scan files in {directory_path}")
        
        for idx, npy_file in enumerate(npy_files, 1):
            experiment_number = get_experiment_number(str(npy_file))
            if not experiment_number:
                print(f"Could not extract experiment number from {npy_file}")
                continue
                
            json_file = directory / f'2D_ODMR_scan_{experiment_number}.json'
            if not json_file.exists():
                print(f"No matching JSON file found for {npy_file}")
                continue
            
            print(f"\nProcessing file {idx}/{total_files}: {npy_file.name}")
            print(f"Processing experiment {experiment_number}")
            
            try:
                # Create experiment-specific output directory
                experiment_output_dir = os.path.join(output_dir, f"experiment_{experiment_number}")
                
                # Initialize analyzer and process data
                analyzer = ODMRAnalyzer(str(npy_file), str(json_file), experiment_number)
                fitted_params = analyzer.fit_double_lorentzian(
                    method=method,
                    output_dir=experiment_output_dir
                )
                
                print(f"Successfully processed experiment {experiment_number}")
                
            except Exception as e:
                print(f"Error processing experiment {experiment_number}: {str(e)}")
                import traceback
                traceback.print_exc()

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


def process_pixel_row(args):
    m, data, freq_axis, default_values, method = args
    N = data.shape[1]
    row_results = {
        "I0": np.zeros(N),
        "A": np.zeros(N),
        "width": np.zeros(N),
        "f_center": np.zeros(N),
        "f_delta": np.zeros(N),
        "quality_score": np.zeros(N),  # Added quality score array
    }
    
    for n in range(N):
        result = ODMRAnalyzer.fit_single_pixel(data[m, n, :], freq_axis, 
                                             default_values, method)
        
        # Calculate fitted curve for quality assessment
        fitted_curve = ODMRAnalyzer.double_dip_func(
            freq_axis,
            result['I0'],
            result['A'],
            result['width'],
            result['f_center'],
            result['f_delta']
        )
        
        # Calculate quality score
        quality_score = calculate_fit_quality(data[m, n, :], fitted_curve)
        
        # Store all results including quality score
        for key in row_results:
            if key == 'quality_score':
                row_results[key][n] = quality_score
            else:
                row_results[key][n] = result[key]
    
    return m, row_results

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


class ODMRFitChecker:
    def __init__(self, fitted_params_file, original_data_file, json_params_file):
        """Initialize with quality assessment capabilities"""
        # Load the fitted parameters (shape: M x N x 6 now, including quality score)
        self.fitted_params = np.load(fitted_params_file)
        
        # Load original data
        self.original_data = np.load(original_data_file)
        
        # Load frequency parameters
        with open(json_params_file, 'r') as f:
            params = json.load(f)
        
        # Create frequency axis
        self.freq_axis = np.linspace(
            params['min_freq'] / 1e9,
            params['max_freq'] / 1e9,
            params['num_measurements']
        )
        
        # Get data dimensions
        self.M, self.N = self.fitted_params.shape[:2]
        
        # Extract quality scores (last parameter in the stack)
        self.quality_scores = self.fitted_params[:, :, -1]
        
        # Calculate success statistics
        self.success_rate = np.mean(self.quality_scores >= 0.9) * 100
        self.mean_quality = np.mean(self.quality_scores)
        
        # Calculate maps
        self.pl_map = np.mean(self.original_data, axis=2)
        self.raw_contrast = np.ptp(self.original_data, axis=2)

        # Store the fitting parameters and quality scores separately
        # First 5 columns are fitting parameters, last column is quality score
        self.fitting_params = self.fitted_params[:, :, :5]  # Extract only fitting parameters
        self.quality_scores = self.fitted_params[:, :, -1]  # Extract quality scores
        
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
                'label': 'Raw Contrast (a.u.)'
            },
            'Fit Contrast': {
                'data': self.fitting_params[:, :, 1] / self.fitting_params[:, :, 0] * 100,
                'cmap': 'viridis',
                'label': 'Fitted Contrast (%)'
            },
            'Peak Splitting': {
                'data': self.fitting_params[:, :, 4],
                'cmap': 'magma',
                'label': 'Peak Splitting (GHz)'
            },
            'Frequency Shift': {
                'data': self.fitting_params[:, :, 3],
                'cmap': 'magma',
                'label': 'Center Frequency (GHz)'
            },
            'Baseline (I0)': {
                'data': self.fitting_params[:, :, 0],
                'cmap': 'viridis',
                'label': 'Baseline Intensity (a.u.)'
            },
            'Amplitude (A)': {
                'data': self.fitting_params[:, :, 1],
                'cmap': 'viridis',
                'label': 'Dip Amplitude (a.u.)'
            },
            'Width': {
                'data': self.fitting_params[:, :, 2],
                'cmap': 'plasma',
                'label': 'Dip Width (GHz)'
            }
        }

        
    def double_lorentzian(self, f, I0, A, width, f_center, f_delta):
        """Calculate double Lorentzian function with given parameters."""
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - \
               A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)
    
    def create_interactive_viewer(self):
        """Create interactive viewer with quality assessment display"""
        # Set up the figure with adjusted spacing
        self.fig = plt.figure(figsize=(16, 8))
        
        # Increase left margin to make room for radio buttons
        plt.subplots_adjust(bottom=0.25, left=0.2)  # Increased left margin
        
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])
        self.ax_data = self.fig.add_subplot(gs[0])
        self.ax_map = self.fig.add_subplot(gs[1])
        
        # Initial state
        self.x_idx, self.y_idx = 0, 0
        self.full_range = True
        self.current_viz = 'Fit Quality'
        self.local_scaling = True  # Add this line for scaling state
        
        # Get initial data range for consistent y-axis limits
        y_data = self.original_data[self.x_idx, self.y_idx]
        self.y_min = np.min(self.original_data)
        self.y_max = np.max(self.original_data)
        y_range = self.y_max - self.y_min
        self.y_margin = y_range * 0.3
                
        # Plot initial spectrum with fixed y-axis limits
        self.spectrum_line, = self.ax_data.plot(self.freq_axis, y_data, 'b.', label='Data')
        self.ax_data.set_ylim(self.y_min - self.y_margin, self.y_max + self.y_margin)
        
        # Calculate and plot initial fit
        params = self.fitting_params[self.x_idx, self.y_idx]
        fitted_curve = self.double_lorentzian(self.freq_axis, *params)
        self.fit_line, = self.ax_data.plot(self.freq_axis, fitted_curve, 'r-', label='Fit')
        
        # Set initial x-axis limits
        self.ax_data.set_xlim(self.freq_axis[0], self.freq_axis[-1])
        
        # Create initial parameter map
        viz_data = self.viz_options[self.current_viz]['data']
        self.map_img = self.ax_map.imshow(viz_data.T, origin='lower', 
                                        cmap=self.viz_options[self.current_viz]['cmap'])
        self.pixel_marker, = self.ax_map.plot(self.x_idx, self.y_idx, 'rx')
        
        # Add colorbar
        self.colorbar = plt.colorbar(self.map_img, ax=self.ax_map)
        self.colorbar.set_label(self.viz_options[self.current_viz]['label'])
        
        # Set up axis labels and titles
        self.ax_data.set_xlabel('Frequency (GHz)')
        self.ax_data.set_ylabel('ODMR Signal (a.u.)')
        self.ax_data.legend()
        self.ax_map.set_xlabel('X Position')
        self.ax_map.set_ylabel('Y Position')
        
        # Add overall statistics to title
        self.ax_map.set_title(
            f'Success Rate: {self.success_rate:.1f}%\n'
            f'Mean Quality: {self.mean_quality:.3f}'
        )
        
        # Keep track of scaling mode
        self.local_scaling = True  # Start with local scaling by default
        
        # Add button for toggling x-axis range and y-axis scaling
        ax_range_button = plt.axes([0.85, 0.15, 0.1, 0.04])  # X-axis range toggle
        ax_scale_button = plt.axes([0.85, 0.08, 0.1, 0.04])  # Y-axis scaling toggle
        self.range_button = Button(ax_range_button, 'Toggle Range')
        self.scale_button = Button(ax_scale_button, 'Toggle Scale')
        
        # Calculate global data range once
        self.global_y_min = np.min(self.original_data)
        self.global_y_max = np.max(self.original_data)
        self.global_y_range = self.global_y_max - self.global_y_min
        self.y_margin_factor = 0.05  # 5% margin

        # Create sliders for x and y coordinates
        ax_x = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_y = plt.axes([0.2, 0.05, 0.6, 0.03])
        
        self.x_slider = Slider(ax_x, 'X', 0, self.M-1, valinit=0, valstep=1)
        self.y_slider = Slider(ax_y, 'Y', 0, self.N-1, valinit=0, valstep=1)
        
        # Add button for toggling x-axis range
        ax_button = plt.axes([0.85, 0.15, 0.1, 0.04])
        self.range_button = Button(ax_button, 'Toggle Range')
        
        # Move radio buttons further left and make them smaller
        ax_radio = plt.axes([0.02, 0.25, 0.1, 0.6])  # Adjusted position
        self.viz_radio = RadioButtons(ax_radio, list(self.viz_options.keys()), 
                                    active=list(self.viz_options.keys()).index(self.current_viz))
        
        # Make radio button labels smaller and adjust their position
        for label in self.viz_radio.labels:
            label.set_fontsize(7)  # Smaller font
        
        # Define update functions
        def update_viz(label):
            self.current_viz = label
            viz_data = self.viz_options[label]['data']
            
            # Update colormap
            self.map_img.set_data(viz_data.T)
            self.map_img.set_cmap(self.viz_options[label]['cmap'])
            
            # Update colorbar label
            self.colorbar.set_label(self.viz_options[label]['label'])
            
            # Update colorbar limits
            self.map_img.autoscale()
            self.fig.canvas.draw_idle()
        
        def toggle_range(event):
            self.full_range = not self.full_range
            update(None)
        
        def toggle_scaling(event):  # Add this new function
            self.local_scaling = not self.local_scaling
            if self.local_scaling:
                print("Using local y-axis scaling")
            else:
                print("Using global y-axis scaling")
            update(None)

        def update(val):
            self.x_idx = int(self.x_slider.val)
            self.y_idx = int(self.y_slider.val)
            
            # Update spectrum plot
            y_data = self.original_data[self.x_idx, self.y_idx]
            self.spectrum_line.set_ydata(y_data)
            
            # Update fit plot
            params = self.fitting_params[self.x_idx, self.y_idx]
            fitted_curve = self.double_lorentzian(self.freq_axis, *params)
            self.fit_line.set_ydata(fitted_curve)
            
            # Update pixel marker
            self.pixel_marker.set_data([self.x_idx], [self.y_idx])
            
            # Update title with info
            param_names = ['I0', 'A', 'w', 'fc', 'fd']
            param_str = ', '.join([f'{name}={val:.3f}' for name, val in zip(param_names, params)])
            raw_contrast = np.ptp(y_data)
            fit_contrast = params[1] / params[0] * 100
            quality_score = self.quality_scores[self.x_idx, self.y_idx]
            param_str += f'\nraw_contrast={raw_contrast:.3f}, fit_contrast={fit_contrast:.1f}%'
            param_str += f'\nquality_score={quality_score:.3f}'
            self.ax_data.set_title(f'Pixel ({self.x_idx}, {self.y_idx})\n{param_str}')
            
            # Calculate local values first, as we need them in both cases
            local_min = min(np.min(y_data), np.min(fitted_curve))
            local_max = max(np.max(y_data), np.max(fitted_curve))
            local_range = local_max - local_min
            y_center = (local_max + local_min) / 2
            
            # Y-axis limits calculation based on scaling mode
            if self.local_scaling:
                # Use local min/max for current pixel
                y_margin = local_range * self.y_margin_factor
                self.ax_data.set_ylim(
                    y_center - local_range/2 - y_margin,
                    y_center + local_range/2 + y_margin
                )
            else:
                # Use global min/max across all pixels
                y_margin = self.global_y_range * self.y_margin_factor
                self.ax_data.set_ylim(
                    self.global_y_min - y_margin,
                    self.global_y_max + y_margin
                )
            
            # Update x-axis limits based on view mode
            if self.full_range:
                self.ax_data.set_xlim(self.freq_axis[0], self.freq_axis[-1])
            else:
                f_center = params[3]
                f_delta = params[4]
                width = params[2]
                x_margin = max(width * 4, f_delta * 1.5)
                self.ax_data.set_xlim(f_center - x_margin, f_center + x_margin)
            
            self.fig.canvas.draw_idle()

        # Connect callbacks
        self.viz_radio.on_clicked(update_viz)
        self.range_button.on_clicked(toggle_range)
        self.scale_button.on_clicked(toggle_scaling)  # Add this line
        self.x_slider.on_changed(update)
        self.y_slider.on_changed(update)
        
        # Initialize plot
        update(None)
        
        # Show the plot
        plt.show()

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

    # lorentzian double dip function without log scaling 
    @staticmethod
    def double_dip_func(f, I0, A, width, f_center, f_delta):
        """
        Static method version of double Lorentzian dip function.
        Must be static for multiprocessing to work.
        """
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)
    
    # lorentzian double dip function with log scaling
    @staticmethod
    def double_dip_func_full_log(f, log_I0, log_A, log_width, log_f_center, log_f_delta):
        """
        Double Lorentzian dip function with all parameters in logarithmic space
        """
        # Convert all parameters from log space
        I0 = np.exp(log_I0)
        A = np.exp(log_A)
        width = np.exp(log_width)
        f_center = np.exp(log_f_center)
        f_delta = np.exp(log_f_delta)
        
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)


    @staticmethod
    def find_dips_robustly(freq_axis, pixel_data_norm):
        """Helper method that looks for dips using multiple approaches"""
        inverted = -pixel_data_norm
        dips = []
        
        # Try different prominence levels - start strict, then get more lenient
        prominence_levels = [0.01, 0.005, 0.002]
        for prominence in prominence_levels:
            peaks, properties = find_peaks(inverted, prominence=prominence, 
                                        width=2,  # Minimum width requirement
                                        distance=3)  # Minimum separation
            if len(peaks) >= 2:
                # Sort by prominence and take top two
                peak_prominences = properties['prominences']
                sorted_indices = np.argsort(peak_prominences)[::-1]
                peaks = peaks[sorted_indices]
                dips = peaks[:2]
                break
            elif len(peaks) == 1 and len(dips) == 0:
                dips = [peaks[0]]
        
        # If we still haven't found two dips, try with smoothed data
        if len(dips) < 2:
            smoothed = savgol_filter(pixel_data_norm, window_length=7, polyorder=3)
            inverted_smooth = -smoothed
            
            for prominence in prominence_levels:
                peaks, properties = find_peaks(inverted_smooth, prominence=prominence,
                                            width=2, distance=3)
                if len(peaks) >= 2:
                    peak_prominences = properties['prominences']
                    sorted_indices = np.argsort(peak_prominences)[::-1]
                    peaks = peaks[sorted_indices]
                    dips = peaks[:2]
                    break
                elif len(peaks) == 1 and len(dips) == 0:
                    dips = [peaks[0]]
        
        return np.array(dips)

    @staticmethod
    def estimate_parameters_from_dips(freq_axis, pixel_data_norm, dips):
        """Helper method that makes smarter initial parameter estimates"""
        freq_range = freq_axis[-1] - freq_axis[0]
        
        # Estimate baseline from highest points
        sorted_data = np.sort(pixel_data_norm)
        I0_est = np.mean(sorted_data[-int(len(sorted_data)*0.1):])
        
        if len(dips) >= 2:
            # We found two dips - use their properties
            f_dip_1 = freq_axis[dips[0]]
            f_dip_2 = freq_axis[dips[1]]
            
            f_center_est = np.mean([f_dip_1, f_dip_2])
            f_delta_est = abs(f_dip_2 - f_dip_1)
            
            # Estimate width from dip shapes
            dip_depth = np.mean([I0_est - pixel_data_norm[dip] for dip in dips[:2]])
            half_max_level = I0_est - dip_depth/2
            
            # Find typical width at half maximum
            widths = []
            for dip in dips[:2]:
                left_idx = dip
                while left_idx > 0 and pixel_data_norm[left_idx] < half_max_level:
                    left_idx -= 1
                
                right_idx = dip
                while right_idx < len(pixel_data_norm)-1 and pixel_data_norm[right_idx] < half_max_level:
                    right_idx += 1
                
                widths.append(freq_axis[right_idx] - freq_axis[left_idx])
            
            width_est = np.mean(widths) * 0.5  # Convert FWHM to Lorentzian width
            A_est = dip_depth
            
        elif len(dips) == 1:
            # Single dip - check if it might be merged
            dip_idx = dips[0]
            f_dip = freq_axis[dip_idx]
            
            # Measure the dip width
            dip_depth = I0_est - pixel_data_norm[dip_idx]
            half_max_level = I0_est - dip_depth/2
            
            left_idx = dip_idx
            while left_idx > 0 and pixel_data_norm[left_idx] < half_max_level:
                left_idx -= 1
            
            right_idx = dip_idx
            while right_idx < len(pixel_data_norm)-1 and pixel_data_norm[right_idx] < half_max_level:
                right_idx += 1
            
            measured_width = freq_axis[right_idx] - freq_axis[left_idx]
            
            if measured_width > freq_range * 0.1:
                # Probably merged dips
                width_est = measured_width * 0.25
                f_center_est = f_dip
                f_delta_est = width_est * 2
            else:
                # Probably single dip
                width_est = measured_width * 0.5
                f_center_est = f_dip
                f_delta_est = freq_range * 0.05
            
            A_est = dip_depth
            
        else:
            # No clear dips - use conservative estimates
            width_est = freq_range * 0.1
            f_center_est = np.mean(freq_axis)
            f_delta_est = freq_range * 0.2
            A_est = np.ptp(pixel_data_norm) * 0.3
        
        return I0_est, A_est, width_est, f_center_est, f_delta_est

    @staticmethod
    def fit_single_pixel(pixel_data, freq_axis, default_values=None, method='trf'):
        """
        Fit single pixel data using full logarithmic scale optimization
        Now treating all parameters in log space to respect their physical positivity
        """
        # Normalize data for numerical stability
        scale_factor = np.max(np.abs(pixel_data))
        pixel_data_norm = pixel_data / scale_factor
        
        # Small constant to avoid log(0)
        epsilon = 1e-10
        
        # Initial parameter estimation
        edge_points = np.concatenate([pixel_data_norm[:5], pixel_data_norm[-5:]])
        I0_est = np.mean(edge_points)
        A_est = np.ptp(pixel_data_norm) * 0.5
        freq_range = freq_axis[-1] - freq_axis[0]
        
        # Find dips for initial estimates
        inverted = -pixel_data_norm
        peaks, _ = find_peaks(inverted, prominence=0.01)
        
        # # Initial parameter estimates with physical consideration
        # if len(peaks) == 0:
        #     width_est = freq_range * 0.1
        #     f_center_est = np.mean(freq_axis)
        #     f_delta_est = freq_range * 0.2
        # elif len(peaks) == 1:
        #     f_dip = freq_axis[peaks[0]]
        #     width_est = freq_range * 0.05
        #     f_center_est = f_dip
        #     f_delta_est = freq_range * 0.1
        # else:
        #     peak_depths = inverted[peaks]
        #     two_deepest = peaks[np.argsort(peak_depths)[-2:]]
        #     f_dip_1 = freq_axis[two_deepest[0]]
        #     f_dip_2 = freq_axis[two_deepest[1]]
        #     width_est = abs(f_dip_2 - f_dip_1) * 0.3
        #     f_center_est = np.mean([f_dip_1, f_dip_2])
        #     f_delta_est = abs(f_dip_2 - f_dip_1)
        
        # Find dips using robust method
        dips = ODMRAnalyzer.find_dips_robustly(freq_axis, pixel_data_norm)
        
        # Get parameter estimates
        I0_est, A_est, width_est, f_center_est, f_delta_est = \
            ODMRAnalyzer.estimate_parameters_from_dips(freq_axis, pixel_data_norm, dips)
        
        # Convert to log space (keep your existing log conversion code)
        log_I0_est = np.log(max(I0_est, epsilon))
        log_A_est = np.log(max(A_est, epsilon))
        log_width_est = np.log(max(width_est, epsilon))
        log_f_center_est = np.log(max(f_center_est, epsilon))
        log_f_delta_est = np.log(max(f_delta_est, epsilon))
        
        # Initial parameter vector (all in log space)
        p0 = [log_I0_est, log_A_est, log_width_est, log_f_center_est, log_f_delta_est]
        
        # Set bounds for TRF method
        if method == 'trf':
            # All bounds in log space
            bounds = ([
                np.log(epsilon),          # log_I0 lower
                np.log(epsilon),          # log_A lower
                np.log(freq_range*1e-4),  # log_width lower
                np.log(freq_axis[0]),     # log_f_center lower
                np.log(epsilon)           # log_f_delta lower
            ], [
                np.log(1e2),             # log_I0 upper
                np.log(1e2),             # log_A upper
                np.log(freq_range),      # log_width upper
                np.log(freq_axis[-1]),   # log_f_center upper
                np.log(freq_range)       # log_f_delta upper
            ])
        else:
            bounds = (-np.inf, np.inf)
        
        try:
            # Fit using curve_fit with all logarithmic parameters
            if method == 'trf':
                popt, pcov = curve_fit(
                    ODMRAnalyzer.double_dip_func_full_log,
                    freq_axis,
                    pixel_data_norm,
                    p0=p0,
                    bounds=bounds,
                    method='trf',
                    maxfev=3000,
                    ftol=1e-4,
                    xtol=1e-4
                )
            else:
                popt, pcov = curve_fit(
                    ODMRAnalyzer.double_dip_func_full_log,
                    freq_axis,
                    pixel_data_norm,
                    p0=p0,
                    method='lm',
                    maxfev=3000
                )
            
            # Convert all parameters back to linear space and original scale
            I0 = np.exp(popt[0]) * scale_factor
            A = np.exp(popt[1]) * scale_factor
            width = np.exp(popt[2])
            f_center = np.exp(popt[3])
            f_delta = np.exp(popt[4])
            
            # Calculate fit quality
            fitted_curve = ODMRAnalyzer.double_dip_func_full_log(freq_axis, *popt)
            log_data = np.log(pixel_data_norm + epsilon)
            log_fitted = np.log(fitted_curve + epsilon)
            mse_log = np.mean((log_data - log_fitted)**2)
            
            # Return default values if fit is poor
            if mse_log > 1.0:
                if default_values is None:
                    default_values = {
                        "I0": np.mean(pixel_data),
                        "A": np.ptp(pixel_data) * 0.1,
                        "width": freq_range * 0.1,
                        "f_center": np.mean(freq_axis),
                        "f_delta": freq_range * 0.2
                    }
                return default_values
                
            return {
                "I0": I0,
                "A": A,
                "width": width,
                "f_center": f_center,
                "f_delta": f_delta
            }
            
        except Exception as e:
            print(f"Fitting failed with error: {str(e)}")
            if default_values is None:
                default_values = {
                    "I0": np.mean(pixel_data),
                    "A": np.ptp(pixel_data) * 0.1,
                    "width": freq_range * 0.1,
                    "f_center": np.mean(freq_axis),
                    "f_delta": freq_range * 0.2
                }
            return default_values

    @timing_decorator    
    def fit_double_lorentzian(self, method='trf', output_dir=None):
        """
        Parallel version of double Lorentzian fitting
        Now includes quality assessment for each fit
        """
        self.start_profiling()
        
        M, N, F = self.data.shape
        fitted_params = {
            "I0": np.zeros((M, N)),
            "A": np.zeros((M, N)),
            "width": np.zeros((M, N)),
            "f_center": np.zeros((M, N)),
            "f_delta": np.zeros((M, N)),
            "quality_score": np.zeros((M, N)),  # Added quality score array
        }
        
        default_values = {"I0": 1.0, "A": 0, "width": 1.0, 
                         "f_center": np.mean(self.freq_axis), "f_delta": 0.0}

        print(f"Processing {M*N} pixels using multiprocessing...")
        
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores")
        
        # Process rows in parallel
        row_args = [(m, self.data, self.freq_axis, default_values, method) 
                   for m in range(M)]
        
        total_processed = 0
        start_time = time.time()
        
        with mp.Pool(processes=num_cores) as pool:
            for m, row_results in tqdm(pool.imap(process_pixel_row, row_args), 
                                     total=M, 
                                     desc="Processing rows"):
                for key in fitted_params:
                    fitted_params[key][m] = row_results[key]
                
                total_processed += N
                if total_processed % (N * 2) == 0:
                    elapsed_time = time.time() - start_time
                    pixels_per_second = total_processed / elapsed_time
                    remaining_pixels = (M * N) - total_processed
                    remaining_time = remaining_pixels / pixels_per_second
                    
                    print(f"\nProcessing speed: {pixels_per_second:.2f} pixels/second")
                    print(f"Estimated time remaining: {remaining_time/60:.2f} minutes")
        
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.data_file))[0]
            fitted_params_file = os.path.join(output_dir, f"{base_name}_fitted_params.npy")
            quality_stats_file = os.path.join(output_dir, f"{base_name}_quality_stats.txt")
            
            # Save the fitted parameters including quality scores
            param_order = ['I0', 'A', 'width', 'f_center', 'f_delta', 'quality_score']
            stacked_params = np.stack([fitted_params[param] for param in param_order], axis=-1)
            np.save(fitted_params_file, stacked_params)
            
            # Calculate and save quality statistics
            quality_scores = fitted_params['quality_score']
            success_rate = np.mean(quality_scores >= 0.9) * 100
            mean_quality = np.mean(quality_scores)
            median_quality = np.median(quality_scores)
            
            with open(quality_stats_file, 'w') as f:
                f.write(f"Fitting Quality Statistics:\n")
                f.write(f"Success Rate (score >= 0.9): {success_rate:.1f}%\n")
                f.write(f"Mean Quality Score: {mean_quality:.3f}\n")
                f.write(f"Median Quality Score: {median_quality:.3f}\n")
            
            # Copy the original files
            new_data_file = os.path.join(output_dir, os.path.basename(self.data_file))
            new_json_file = os.path.join(output_dir, os.path.basename(self.json_file))
            import shutil
            shutil.copy2(self.data_file, new_data_file)
            shutil.copy2(self.json_file, new_json_file)
            
            print(f"\nSaved files in: {output_dir}")
            print(f"Saved:")
            print(f"  - Fitted parameters: {os.path.basename(fitted_params_file)}")
            print(f"  - Quality statistics: {os.path.basename(quality_stats_file)}")
            print(f"  - Original data: {os.path.basename(new_data_file)}")
            print(f"  - JSON parameters: {os.path.basename(new_json_file)}")
            
            return output_dir, fitted_params_file
        
        self.stop_profiling()
        return None, None

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
    

def main():
    data_file = r"C:\Users\Diederik\Documents\BEP\measurement_stuff_new\oct-nov-2024 biosample\2D_ODMR_scan_1731005879.npy"
    json_file = r"C:\Users\Diederik\Documents\BEP\measurement_stuff_new\oct-nov-2024 biosample\2D_ODMR_scan_1731005879.json"

    # Initialize analyzer at the start
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
        
        choice = input("Enter your choice (1/2/3/4/5): ")
        
        # Check if analyzer is needed and not initialized
        if choice in ['1', '2'] and analyzer is None:
            if not all(os.path.exists(f) for f in [data_file, json_file]):
                print("Error: One or more input files not found.")
                continue
            analyzer = ODMRAnalyzer(data_file, json_file, enable_profiling=False)

        if choice == '1':
            method_choice = input("Choose optimization method (trf/lm): ").lower()
            while method_choice not in ['trf', 'lm']:
                print("Invalid choice. Please choose 'trf' or 'lm'")
                method_choice = input("Choose optimization method (trf/lm): ").lower()
            
            output_dir = input("Enter output directory path (press Enter for default './fitted_parameters'): ")
            if not output_dir:
                output_dir = "./fitted_parameters"
            
            output_dir, fitted_params_file = analyzer.fit_double_lorentzian(
                method=method_choice,
                output_dir=output_dir
            )
            
            if fitted_params_file:
                print("\nWould you like to check the fitted results?")
                check_choice = input("Enter y/n: ").lower()
                if check_choice == 'y':
                    try:
                        checker = ODMRFitChecker(fitted_params_file, data_file, json_file)
                        checker.create_interactive_viewer()
                    except Exception as e:
                        print(f"\nError launching fit checker: {str(e)}")
                        import traceback
                        traceback.print_exc()

        elif choice == '2':
            # Single pixel analysis
            try:
                x = int(input("Enter x coordinate (0-" + str(analyzer.data.shape[0]-1) + "): "))
                y = int(input("Enter y coordinate (0-" + str(analyzer.data.shape[1]-1) + "): "))
                
                print(f"Attempting to process pixel at coordinates: x={x}, y={y}")
                
                try:
                    pixel = analyzer.data[x, y, :]
                    single_pixel_params = analyzer.fit_single_pixel(pixel, analyzer.freq_axis)
                    
                    # Plot the pixel spectrum with fitted curve
                    fig_spectrum, ax_spectrum = analyzer.plot_pixel_spectrum(
                        x, y, 
                        fit_result=single_pixel_params
                    )
                    plt.show()

                    # Analyze and display results
                    analysis = analyzer.analyze_spectrum(x, y, 
                        fitted_params={
                            'I0': np.array([[single_pixel_params['I0']]]),
                            'A': np.array([[single_pixel_params['A']]]),
                            'width': np.array([[single_pixel_params['width']]]),
                            'f_center': np.array([[single_pixel_params['f_center']]]),
                            'f_delta': np.array([[single_pixel_params['f_delta']]])
                        })
                    
                    print("\nSpectrum Analysis Results:")
                    print(f"Number of dips found: {analysis['num_dips']}")
                    print("\nDip frequencies (GHz):")
                    for freq, depth in zip(analysis['dip_frequencies'], analysis['dip_depths']):
                        print(f"  {freq:.3f} GHz (depth: {depth:.3e})")
                    print(f"\nTotal contrast: {analysis['contrast']:.3e}")
                    
                    print("\nFitted Parameters:")
                    for key, value in single_pixel_params.items():
                        print(f"  {key}: {value:.4f}")
                
                except Exception as e:
                    print(f"Error processing pixel: {e}")
                    import traceback
                    traceback.print_exc()
            
            except ValueError as ve:
                print(f"Input error: {ve}")
                print("Please enter valid integer coordinates.")

        elif choice == '3':
            # Batch processing
            directory = input("Enter directory path containing ODMR datasets: ")
            method_choice = input("Choose optimization method (trf/lm): ").lower()
            while method_choice not in ['trf', 'lm']:
                print("Invalid choice. Please choose 'trf' or 'lm'")
                method_choice = input("Choose optimization method (trf/lm): ").lower()
            
            output_dir = input("Enter output directory path (press Enter for default './fitted_parameters'): ")
            if not output_dir:
                output_dir = "./fitted_parameters"
            
            try:
                process_directory(directory, method=method_choice, output_dir=output_dir)
            except Exception as e:
                print(f"Error during batch processing: {e}")
                import traceback
                traceback.print_exc()

        elif choice == '4':
            print("\nChecking fitted results - please specify the required files:")
            print("Note: You'll need to provide paths to three files:")
            print("1. The fitted parameters file (.npy)")
            print("2. The original data file (.npy)")
            print("3. The JSON parameters file (.json)")
            
            # Get file paths from user
            fitted_params_file = input("\nEnter path to fitted parameters file (.npy): ").strip()
            data_file = input("Enter path to original data file (.npy): ").strip()
            json_file = input("Enter path to JSON parameters file (.json): ").strip()
            
            # Verify file existence and extensions
            files_to_check = [
                (fitted_params_file, '.npy', "Fitted parameters"),
                (data_file, '.npy', "Original data"),
                (json_file, '.json', "JSON parameters")
            ]
            
            files_ok = True
            for file_path, expected_ext, file_type in files_to_check:
                if not os.path.exists(file_path):
                    print(f"\nError: {file_type} file not found at: {file_path}")
                    files_ok = False
                elif not file_path.lower().endswith(expected_ext):
                    print(f"\nWarning: {file_type} file doesn't have expected {expected_ext} extension")
                    confirm = input("Continue anyway? (y/n): ").lower()
                    if confirm != 'y':
                        files_ok = False
            
            if not files_ok:
                continue
            
            print("\nInitializing fit checker with provided files...")
            try:
                checker = ODMRFitChecker(fitted_params_file, data_file, json_file)
                checker.create_interactive_viewer()
            except Exception as e:
                print(f"\nError launching fit checker: {str(e)}")
                import traceback
                traceback.print_exc()

        elif choice == '5':
            break

if __name__ == "__main__":
    main()
