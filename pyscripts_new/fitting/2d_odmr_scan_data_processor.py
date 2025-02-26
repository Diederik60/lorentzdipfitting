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
'''

Statistics of data provided by Dylan, this data is concerns 20x20x100 2D ODMR scan data

Parameter statistics:
                 I0              A         width      f_center       f_delta
count  2.560000e+04   25600.000000  25600.000000  25600.000000  2.560000e+04
mean   4.608335e+05   28471.238802      0.006178      2.871805  3.546591e-02
std    4.529519e+05   25914.354677      0.004521      0.008753  1.671464e-02
min    6.029727e+04      93.602962      0.000064      2.820555  3.344105e-09
25%    1.525285e+05    7803.165582      0.004589      2.869750  3.079085e-02
50%    2.024858e+05   23177.294963      0.005706      2.870441  3.653382e-02
75%    8.490758e+05   36695.043713      0.006545      2.878413  4.990010e-02
max    1.442024e+06  250318.277408      0.140000      2.940000  1.344750e-01


'''


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
    # # widefield    
    # for n in range(N):
    #     result = ODMRAnalyzer.fit_single_pixel_widefield(data[m, n, :], freq_axis, 
    #                                          default_values, method)
        
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

        # Add new parameters for neighborhood averaging
        self.enable_averaging = False  # Toggle for neighborhood averaging
        self.quality_threshold = 0.80  # Quality score threshold
        
        # Create cached averaged data dictionary
        self.averaged_data = {}
        
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

        # Then initialize log_scale_states
        self.log_scale_states = {key: False for key in self.viz_options.keys()}

        self._update_timer = None  # For debouncing
        self.averaged_data = {}    # For caching averaged data

    def double_lorentzian(self, f, I0, A, width, f_center, f_delta):
        """Calculate double Lorentzian function with given parameters."""
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - \
               A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)
    
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
        fitted_curve = self.double_lorentzian(self.freq_axis, *params)
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
            f_center = params[3]
            f_delta = params[4]
            width = params[2]
            x_margin = max(width * 4, f_delta * 1.5)
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
        ax_radio = plt.axes([0.02, 0.25, 0.1, 0.6])
        self.viz_radio = RadioButtons(ax_radio, list(self.viz_options.keys()), 
                                    active=list(self.viz_options.keys()).index(self.current_viz))
        
        # Improve radio button appearance
        for label in self.viz_radio.labels:
            label.set_fontsize(7)
        
        self.viz_radio.on_clicked(self._on_viz_change)

    def _create_control_buttons(self):
        """Create control buttons with consistent layout"""
        button_width = 0.1
        button_height = 0.04
        button_left = 0.85
        
        buttons_config = [
            ('log_button', 'Toggle Log Scale', 0.29),
            ('avg_button', 'Toggle Averaging', 0.22),
            ('range_button', 'Toggle Range', 0.15),
            ('scale_button', 'Toggle Scale', 0.08)
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
            f_center = params[3]
            f_delta = params[4]
            width = params[2]
            x_margin = max(width * 4, f_delta * 1.5)
            self.ax_data.set_xlim(f_center - x_margin, f_center + x_margin)

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

    def force_update(self):
        """Force a complete update of all plot elements"""
        try:
            # Update data plot1
            y_data = self.original_data[self.x_idx, self.y_idx]
            params = self.fitting_params[self.x_idx, self.y_idx]
            fitted_curve = self.double_lorentzian(self.freq_axis, *params)
            
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
                for i, param_name in enumerate(['I0', 'A', 'w', 'fc', 'fd']):
                    averaged_val = self.calculate_neighborhood_average(
                        self.fitting_params[:, :, i], 
                        self.x_idx, 
                        self.y_idx,
                    )
                    param_str += f'\n{param_name}={averaged_val:.3f}'
            else:
                param_names = ['I0', 'A', 'w', 'fc', 'fd']
                param_str = ', '.join([f'{name}={val:.3f}' for name, val in zip(param_names, params)])
            param_str += f'\nquality_score={quality_score:.3f}'
            
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
    
    #hybrid version for log scaling, f_center and f_delta are linear because the range allows it
    @staticmethod
    def double_dip_func_hybrid(f, log_I0, log_A, log_width, f_center, f_delta):
        """
        Double Lorentzian dip function with hybrid parameter space.
        I0, A, and width are in log space, while f_center and f_delta are linear.
        """
        # Convert amplitude parameters from log space
        I0 = np.exp(log_I0)
        A = np.exp(log_A)
        width = np.exp(log_width)
        
        # Calculate double dip
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - \
            A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)


    @staticmethod
    def find_outer_dips(freq_axis, pixel_data_norm):
        """Find the outermost dips in a spectrum that may contain multiple dip pairs"""
        inverted = -pixel_data_norm
        
        # Try different prominence levels to find dips
        prominence_levels = [0.01, 0.005, 0.002]
        all_peaks = []
        
        for prominence in prominence_levels:
            peaks, properties = find_peaks(inverted, prominence=prominence, 
                                        width=2, distance=3)
            if len(peaks) >= 2:
                # Sort peaks by prominence
                peak_prominences = properties['prominences']
                peak_positions = freq_axis[peaks]
                
                # Filter peaks that are too close to the edges
                edge_margin = (freq_axis[-1] - freq_axis[0]) * 0.05
                valid_peaks = peaks[
                    (peak_positions > freq_axis[0] + edge_margin) & 
                    (peak_positions < freq_axis[-1] - edge_margin)
                ]
                
                if len(valid_peaks) >= 2:
                    # Find the leftmost and rightmost prominent peaks
                    leftmost_idx = valid_peaks[0]
                    rightmost_idx = valid_peaks[-1]
                    
                    # Only accept if the peaks form a reasonable pair
                    min_separation = (freq_axis[-1] - freq_axis[0]) * 0.1
                    if freq_axis[rightmost_idx] - freq_axis[leftmost_idx] >= min_separation:
                        return np.array([leftmost_idx, rightmost_idx])
                    
        return np.array([])

    @staticmethod
    def estimate_parameters_from_dips(freq_axis, pixel_data_norm, dips):
        """Modified parameter estimation focusing on outer dips"""
        # Physics-based constraints
        TYPICAL_WIDTH = 0.006
        MIN_SPLITTING = 0.010
        MAX_SPLITTING = 0.150
        
        # Calculate frequency range and mean
        freq_range = freq_axis[-1] - freq_axis[0]
        mean_freq = np.mean(freq_axis)
        
        # Baseline estimation using 95th percentile
        I0_est = np.percentile(pixel_data_norm, 95)
        
        # Apply Savitzky-Golay filter
        window_length = min(11, len(pixel_data_norm)//2*2+1)
        smoothed_data = savgol_filter(pixel_data_norm, window_length=window_length, polyorder=3)
        
        if len(dips) >= 2:
            # Use outermost dips
            f_dip_1, f_dip_2 = sorted([freq_axis[d] for d in [dips[0], dips[-1]]])
            f_center_est = (f_dip_1 + f_dip_2) / 2
            f_delta_est = f_dip_2 - f_dip_1
            
            # Calculate width and amplitude from outer dips
            outer_depths = [I0_est - smoothed_data[d] for d in [dips[0], dips[-1]]]
            width_est = TYPICAL_WIDTH
            A_est = np.mean(outer_depths)
            
        else:
            # Fallback to default values
            f_center_est = 2.87
            f_delta_est = MIN_SPLITTING
            width_est = TYPICAL_WIDTH
            A_est = 0.1 * np.ptp(pixel_data_norm)
        
        # Apply constraints
        width_est = np.clip(width_est, TYPICAL_WIDTH*0.5, TYPICAL_WIDTH*2)
        f_delta_est = np.clip(f_delta_est, MIN_SPLITTING, MAX_SPLITTING)
        f_center_est = np.clip(f_center_est, mean_freq - 0.1, mean_freq + 0.1)
        
        return I0_est, A_est, width_est, f_center_est, f_delta_est

    @staticmethod
    def fit_single_pixel(pixel_data, freq_axis, default_values=None, method='trf'):
        """
        Enhanced single pixel fitting with better handling of merged dips
        """
        # Normalize data
        scale_factor = np.max(np.abs(pixel_data))
        if scale_factor == 0:
            scale_factor = 1
        pixel_data_norm = pixel_data / scale_factor
        
        # Apply Savitzky-Golay filter for smoother analysis
        window_length = min(11, len(pixel_data_norm)//2*2+1)
        smoothed_data = savgol_filter(pixel_data_norm, window_length=window_length, polyorder=3)
        
        # Find dips in smoothed data
        inverted = -smoothed_data
        
        # Try to find dips with different prominence levels
        prominences = [0.01, 0.005, 0.002]
        all_peaks = []
        peak_properties = None
        
        for prominence in prominences:
            peaks, properties = find_peaks(inverted, 
                                        prominence=prominence,
                                        width=2,
                                        distance=3)
            if len(peaks) > 0:
                all_peaks = peaks
                peak_properties = properties
                break
        
        # Initial parameter estimation
        I0_est = np.percentile(pixel_data_norm, 95)
        
        # Case 1: Multiple distinct peaks found
        if len(all_peaks) >= 2:
            # Use the two most prominent peaks
            peak_prominences = peak_properties['prominences']
            sorted_indices = np.argsort(peak_prominences)[::-1]
            main_peaks = all_peaks[sorted_indices[:2]]
            
            f_dip_1, f_dip_2 = sorted([freq_axis[idx] for idx in main_peaks])
            f_center_est = (f_dip_1 + f_dip_2) / 2
            f_delta_est = f_dip_2 - f_dip_1
            width_est = 0.006  # typical NV linewidth
            A_est = np.mean([I0_est - smoothed_data[idx] for idx in main_peaks])
            
        # Case 2: Single peak found - potential merged dips
        elif len(all_peaks) == 1:
            peak_idx = all_peaks[0]
            peak_width = peak_properties['widths'][0]
            
            # Check if this might be a merged peak
            if peak_width > 6:  # wider than typical single dip
                f_center_est = freq_axis[peak_idx]
                f_delta_est = 0.010  # minimum splitting
                width_est = 0.006
                A_est = I0_est - smoothed_data[peak_idx]
            else:
                # Treat as single dip with minimal splitting
                f_center_est = freq_axis[peak_idx]
                f_delta_est = 0.005  # very small splitting
                width_est = peak_width * (freq_axis[1] - freq_axis[0]) / 4
                A_est = I0_est - smoothed_data[peak_idx]
        
        # Case 3: No peaks found
        else:
            f_center_est = 2.87  # typical zero-field splitting
            f_delta_est = 0.010
            width_est = 0.006
            A_est = 0.1 * np.ptp(pixel_data_norm)
        
        # Convert amplitude parameters to log space
        epsilon = 1e-10
        log_I0_est = np.log(max(I0_est, epsilon))
        log_A_est = np.log(max(A_est, epsilon))
        log_width_est = np.log(max(width_est, epsilon))
        
        # Initial parameter vector
        p0 = [log_I0_est, log_A_est, log_width_est, f_center_est, f_delta_est]
        
        # Set bounds for TRF method
        if method == 'trf':
            
            # #40x40 sample
            # bounds = (
            #     # Lower bounds
            #     [np.log(0.01), np.log(1e-3), np.log(0.001), 2.80, 0.003],  # Wider frequency range
            #     # Upper bounds
            #     [np.log(10.0), np.log(1.0), np.log(0.1), 2.96, 0.16]      # and splitting
            # )

            #widefield
            bounds = (
                # Lower bounds
                [np.log(0.0001),# Allow for very low baseline (previously 0.1)
                np.log(1e-6),   # Allow for very small amplitudes
                np.log(0.001),  # Width can stay the same
                2.75,           # Frequency center
                0.001],         # Splitting
                # Upper bounds
                [np.log(1000.0),  # Allow for higher intensity regions
                np.log(100.0),   # Allow for large amplitudes in bright regions
                np.log(0.1),    # Width upper bound
                3.00,           # Frequency center
                0.20]           # Splitting
            )

            # Clip initial parameters to bounds
            for i, (param, (lower, upper)) in enumerate(zip(p0, zip(*bounds))):
                p0[i] = np.clip(param, lower, upper)
        else:
            bounds = (-np.inf, np.inf)
        
        try:
            # Try first fit
            popt, pcov = curve_fit(
                ODMRAnalyzer.double_dip_func_hybrid,
                freq_axis,
                pixel_data_norm,
                p0=p0,
                bounds=bounds if method == 'trf' else (-np.inf, np.inf),
                method=method,
                maxfev=3000,
                ftol=1e-4,
                xtol=1e-4
            )
            
            # Calculate fit quality
            fitted_curve = ODMRAnalyzer.double_dip_func_hybrid(freq_axis, *popt)
            quality_score = calculate_fit_quality(pixel_data_norm, fitted_curve)
            
            # If poor fit, try again with single dip assumption
            if quality_score < 0.6:
                p0[4] = np.log(0.005)  # Very small f_delta
                popt, pcov = curve_fit(
                    ODMRAnalyzer.double_dip_func_hybrid,
                    freq_axis,
                    pixel_data_norm,
                    p0=p0,
                    bounds=bounds if method == 'trf' else (-np.inf, np.inf),
                    method=method,
                    maxfev=3000,
                    ftol=1e-4,
                    xtol=1e-4
                )
                
                # Recalculate quality
                fitted_curve = ODMRAnalyzer.double_dip_func_hybrid(freq_axis, *popt)
                quality_score = calculate_fit_quality(pixel_data_norm, fitted_curve)
            
            # Convert back to linear space and original scale
            I0 = np.exp(popt[0]) * scale_factor
            A = np.exp(popt[1]) * scale_factor
            width = np.exp(popt[2])
            f_center = popt[3]
            f_delta = popt[4]
            
            # Final quality check
            if quality_score < 0.5:
                return default_values if default_values is not None else {
                    "I0": np.mean(pixel_data),
                    "A": np.ptp(pixel_data) * 0.1,
                    "width": 0.006,
                    "f_center": np.mean(freq_axis),
                    "f_delta": 0.010
                }
            
            return {
                "I0": I0,
                "A": A,
                "width": width,
                "f_center": f_center,
                "f_delta": f_delta
            }
            
        except Exception as e:
            print(f"Fitting failed: {str(e)}") 
            return default_values if default_values is not None else {
                "I0": np.mean(pixel_data),
                "A": np.ptp(pixel_data) * 0.1,
                "width": 0.006,
                "f_center": np.mean(freq_axis),
                "f_delta": 0.010
            }

    # method for widefield odmr                
    @staticmethod
    def fit_single_pixel_widefield(pixel_data, freq_axis, default_values=None, method='trf'):
        """
        Simplified fitting function specifically for widefield ODMR with low SNR.
        Prioritizes speed and robustness over accuracy.
        """
        # Quick check for very low signal pixels
        if np.max(pixel_data) < 0.1:
            return default_values
        
        # Simple normalization - no need for complex scaling
        pixel_data_norm = pixel_data / np.max(pixel_data)
        
        # Apply heavy smoothing to handle low SNR
        window_length = min(31, len(pixel_data_norm)//2*2+1)  # Larger window for more smoothing
        smoothed = savgol_filter(pixel_data_norm, window_length=window_length, polyorder=2)
        
        # Use simpler peak finding with relaxed criteria
        inverted = -smoothed
        peaks, _ = find_peaks(inverted, prominence=0.005, distance=5)
        
        # Very simple parameter estimation
        I0_est = 1.0  # Since we normalized
        
        if len(peaks) >= 2:
            # Just take the two deepest dips
            main_peaks = peaks[np.argsort(inverted[peaks])[-2:]]
            f_dip_1, f_dip_2 = sorted([freq_axis[idx] for idx in main_peaks])
            f_center_est = (f_dip_1 + f_dip_2) / 2
            f_delta_est = f_dip_2 - f_dip_1
            A_est = 0.1  # Simplified amplitude estimate
        else:
            # Default parameters centered around typical NV values
            f_center_est = 2.87
            f_delta_est = 0.01
            A_est = 0.1
        
        # Use fixed width for simplicity
        width_est = 0.006
        
        # Very relaxed bounds for fitting
        bounds = (
            # Lower bounds - much more relaxed
            [np.log(0.1), np.log(0.001), np.log(0.001), 2.7, 0.001],
            # Upper bounds - wide range
            [np.log(10.0), np.log(1.0), np.log(0.1), 3.0, 0.3]
        )
        
        # Initial parameters
        p0 = [
            np.log(I0_est),
            np.log(A_est),
            np.log(width_est),
            f_center_est,
            f_delta_est
        ]
        
        try:
            # Quick fit with relaxed convergence criteria
            popt, _ = curve_fit(
                ODMRAnalyzer.double_dip_func_hybrid,
                freq_axis,
                pixel_data_norm,
                p0=p0,
                bounds=bounds,
                method=method,
                maxfev=1000,    # Reduced iterations
                ftol=1e-2,      # Very relaxed tolerance
                xtol=1e-2       # Very relaxed tolerance
            )
            
            # Convert back to original scale
            return {
                "I0": np.exp(popt[0]) * np.max(pixel_data),
                "A": np.exp(popt[1]) * np.max(pixel_data),
                "width": np.exp(popt[2]),
                "f_center": popt[3],
                "f_delta": popt[4]
            }
            
        except Exception as e:
            # Return default values if fit fails
            if default_values is None:
                default_values = {
                    "I0": np.mean(pixel_data),
                    "A": 0.1 * np.max(pixel_data),
                    "width": 0.006,
                    "f_center": 2.87,
                    "f_delta": 0.01
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
