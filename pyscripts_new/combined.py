import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm
import datetime
import time
import cProfile
import pstats
from functools import wraps
import io
import multiprocessing as mp
from functools import partial
from pathlib import Path
import re


def get_experiment_number(filename):
    """Extract experiment number from filename with format '2D_ODMR_scan_{number}.npy'"""
    pattern = r'2D_ODMR_scan_(\d+)\.npy$'
    match = re.search(pattern, filename)
    return int(match.group(1)) if match else None

def process_directory(directory_path, method='trf', output_dir="./fitted_parameters"):
    """
    Process ODMR datasets in the specified directory that match the pattern '2D_ODMR_scan_{number}'.
    
    Args:
        directory_path (str): Path to directory containing .npy and .json files
        method (str): Optimization method ('trf' or 'lm')
        output_dir (str): Directory to save fitted parameters
    """
    output_name = Path(directory_path).name
    directory = Path(directory_path)
    output_path = Path(output_dir) / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get only .npy files that match the specific pattern
    npy_files = list(directory.glob('2D_ODMR_scan_*.npy'))
    total_files = len(npy_files)
    
    print(f"Found {total_files} matching ODMR scan files in {directory_path}")
    
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
            # Initialize analyzer for this dataset
            analyzer = ODMRAnalyzer(str(npy_file), str(json_file), experiment_number)
            
            # Create experiment-specific output directory
            exp_output_dir = output_path 
            exp_output_dir.mkdir(exist_ok=True)
            
            # Fit the data
            fitted_params = analyzer.fit_double_lorentzian(
                method=method,
                output_dir=str(exp_output_dir)
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
    }
    
    for n in range(N):
        result = ODMRAnalyzer.fit_single_pixel(data[m, n, :], freq_axis, 
                                             default_values, method)
        for key in row_results:
            row_results[key][n] = result[key]
    
    return m, row_results

class ODMRAnalyzer:
    def __init__(self, data_file, json_file, experiment_number, enable_profiling=False):
        """
        Initialize ODMR analyzer with data and parameters
        
        Args:
            data_file (str): Path to numpy data file
            json_file (str): Path to JSON parameter file
            experiment_number (int): Experiment identifier
            enable_profiling (bool): Whether to enable profiling functionality, turned off by default
        """
        #profiler for collectiong information about how much time is spent in every part of the code
        # Add a flag to control profiling
        self.profiling_enabled = enable_profiling
        
        # Only create profiler if enabled
        if self.profiling_enabled:
            self.profiler = cProfile.Profile()
        else:
            self.profiler = None     
        
        self.experiment_number = experiment_number

        # loading data using the load_data method with timing decorator to inspect the time it takes 
        self.load_data(data_file, json_file)

        # loading data directly on initialisation
        self.data = np.load(data_file)
        with open(json_file, 'r') as f:
            self.params = json.load(f)

        self.freq_axis = np.linspace( # creating fequency axis
            self.params['min_freq'] / 1e9,  # Convert to GHz
            self.params['max_freq'] / 1e9,
            self.params['num_measurements']
        )
        self.mean_spectrum = np.mean(self.data, axis=(0, 1)) # calculating mean spectrum

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
        This reflects the physical reality that all parameters must be positive
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

    
    @timing_decorator  # Add this decorator to measure the overall function execution time
    def fit_double_lorentzian(self, tail=5, thresholds=[3, 5], 
                         error_threshold=0.1, default_values=None, 
                         method='trf', output_dir=None):
        """
        Parallel version of double Lorentzian fitting
        """
        self.start_profiling()
        
        M, N, F = self.data.shape
        fitted_params = {
            "I0": np.zeros((M, N)),
            "A": np.zeros((M, N)),
            "width": np.zeros((M, N)),
            "f_center": np.zeros((M, N)),
            "f_delta": np.zeros((M, N)),
        }
        
        if default_values is None:
            default_values = {"I0": 1.0, "A": 0, "width": 1.0, 
                            "f_center": np.mean(self.freq_axis), "f_delta": 0.0}

        print(f"Processing {M*N} pixels using multiprocessing...")
        
        # Prepare the data for parallel processing
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} CPU cores")
        
        # Create arguments for each row
        row_args = [(m, self.data, self.freq_axis, default_values, method) 
                    for m in range(M)]
                
        # Process rows in parallel
        total_processed = 0
        start_time = time.time()
        
        with mp.Pool(processes=num_cores) as pool:
            for m, row_results in tqdm(pool.imap(process_pixel_row, row_args), 
                                    total=M, 
                                    desc="Processing rows"):
                # Store results
                for key in fitted_params:
                    fitted_params[key][m] = row_results[key]
                
                total_processed += N
                if total_processed % (N * 2) == 0:  # Report every 2 rows
                    elapsed_time = time.time() - start_time
                    pixels_per_second = total_processed / elapsed_time
                    remaining_pixels = (M * N) - total_processed
                    remaining_time = remaining_pixels / pixels_per_second
                    
                    # print statements for checking
                    print(f"\nProcessing speed: {pixels_per_second:.2f} pixels/second")
                    print(f"Estimated time remaining: {remaining_time/60:.2f} minutes")
        
        self.stop_profiling()
        
        if output_dir is not None:
            self.save_fitted_parameters(fitted_params, output_dir)
        
        return fitted_params

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

    def save_fitted_parameters(self, fitted_params, output_dir):
        """
        Save fitted parameters as a single .npy file
        
        Args:
            fitted_params (dict): Dictionary of fitted parameter arrays
            output_dir (str): Directory to save parameters
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect parameters in a specific order
        param_order = ['I0', 'A', 'width', 'f_center', 'f_delta']
        
        # Stack parameters along the third axis
        stacked_params = np.stack([fitted_params[param] for param in param_order], axis=-1)
        
        # Generate filename
        filepath = os.path.join(output_dir, f"lorentzian_params_{self.experiment_number}.npy")
        
        # Save as single .npy file
        np.save(filepath, stacked_params)
        print(f"Fitted parameters saved to {filepath}")
        print(f"Saved array shape: {stacked_params.shape}")
        return filepath

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
    experiment_number = 1730338486

    data_file = fr'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_2_biosample\2D_ODMR_scan_{experiment_number}.npy'
    json_file = fr'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_2_biosample\2D_ODMR_scan_{experiment_number}.json'
    
    # Initialize analyzer
    analyzer = ODMRAnalyzer(data_file, json_file, experiment_number, enable_profiling=False)

    # # Debug: Print detailed information about the data
    # print("Data shape:", analyzer.data.shape)
    # print("Data type:", type(analyzer.data))
    # print("Data dimensions:", len(analyzer.data.shape))
    # print("First few data values:", analyzer.data[:5, :5, 0])

    # Interactive mode selection
    while True:
        print("\nODMR Analysis Options:")
        print("1. Perform full dataset fitting and save parameters")
        print("2. Analyze single pixel spectrum")
        print("3. Batch process directory")
        print("4. Exit")
        
        choice = input("Enter your choice (1/2/3/4): ")
        
        if choice == '1':
            # Ask for optimization method
            method_choice = input("Choose optimization method (trf/lm): ").lower()
            while method_choice not in ['trf', 'lm']:
                print("Invalid choice. Please choose 'trf' or 'lm'")
                method_choice = input("Choose optimization method (trf/lm): ").lower()
            
            # Perform full dataset fitting with chosen method
            fitted_params = analyzer.fit_double_lorentzian(method=method_choice, output_dir="./fitted_parameters")
            # # Optional: Plot contrast map
            # fig_map, ax_map, contrast_map = analyzer.plot_contrast_map(fitted_params)
            # plt.show()
        
        elif choice == '2':
            # Single pixel analysis
            try:
                x = int(input("Enter x coordinate (0-" + str(analyzer.data.shape[0]-1) + "): "))
                y = int(input("Enter y coordinate (0-" + str(analyzer.data.shape[1]-1) + "): "))
                
                print(f"Attempting to process pixel at coordinates: x={x}, y={y}")
                
                try:

                    pixel = analyzer.data[x, y, :]
                    single_pixel_params = analyzer.fit_single_pixel(pixel, analyzer.freq_axis)

                    # Debug: Print the specific pixel data
                    # print(f"Pixel data shape: {pixel.shape}")
                    # print(f"Pixel data: {pixel}")

                    
                    # Plot the pixel spectrum with fitted curve
                    fig_spectrum, ax_spectrum = analyzer.plot_pixel_spectrum(
                        x, y, 
                        fit_result={
                            'I0': single_pixel_params['I0'],
                            'A': single_pixel_params['A'],
                            'width': single_pixel_params['width'],
                            'f_center': single_pixel_params['f_center'],
                            'f_delta': single_pixel_params['f_delta']
                        }
                    )
                    plt.show()

                    # Similarly, update the analyze_spectrum call
                    analysis = analyzer.analyze_spectrum(x, y, 
                        fitted_params={
                            'I0': single_pixel_params['I0'],
                            'A': single_pixel_params['A'],
                            'width': single_pixel_params['width'],
                            'f_center': single_pixel_params['f_center'],
                            'f_delta': single_pixel_params['f_delta']
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
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()