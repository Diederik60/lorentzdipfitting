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
    """
    Function to process a single row of pixels.
    Must be outside the class for multiprocessing.
    """
    m, data, freq_axis, default_values = args
    N = data.shape[1]
    row_results = {
        "I0": np.zeros(N),
        "A": np.zeros(N),
        "width": np.zeros(N),
        "f_center": np.zeros(N),
        "f_delta": np.zeros(N),
    }
    
    for n in range(N):
        result = ODMRAnalyzer.fit_single_pixel(data[m, n, :], freq_axis, default_values)
        for key in row_results:
            row_results[key][n] = result[key]
    
    return m, row_results

class ODMRAnalyzer:
    def __init__(self, data_file, json_file, experiment_number):
        """
        Initialize ODMR analyzer with data and parameters
        
        Args:
            data_file (str): Path to numpy data file
            json_file (str): Path to JSON parameter file
        """

        #profiler for collectiong information about how much time is spent in every part of the code
        self.profiler = cProfile.Profile()

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
        """Start profiling code execution"""
        self.profiler.enable()

    def stop_profiling(self):
        """Stop profiling and print results"""
        self.profiler.disable()
        s = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        stats.print_stats()
        print(s.getvalue())

    @staticmethod
    def double_dip_func(f, I0, A, width, f_center, f_delta):
        """
        Static method version of double Lorentzian dip function.
        Must be static for multiprocessing to work.
        """
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)
    
    @staticmethod
    def fit_single_pixel(pixel_data, freq_axis, default_values=None):
        """
        Static method for fitting a single pixel.
        Takes explicit parameters instead of using class attributes.
        """
        if default_values is None:
            default_values = {
                "I0": 1.0, 
                "A": 0, 
                "width": 1.0, 
                "f_center": np.mean(freq_axis), 
                "f_delta": 0.0
            }
            
        # Estimate noise level
        tail = 5  # You can make this a parameter if needed
        tail_values = np.concatenate((pixel_data[:tail], pixel_data[-tail:]))
        noise_std = np.std(tail_values)
        I0_est = np.average(tail_values)
        A_est = np.max(pixel_data) - np.min(pixel_data) - 2*noise_std

        # Peak detection
        inverted = -pixel_data
        peaks, _ = find_peaks(inverted, prominence=0.01)
        dips = [(p, p) for p in peaks]

        # Initial parameter estimation
        if len(dips) == 0:
            width_est = 1
            f_center_est = np.average(freq_axis)
            f_delta_est = (np.max(freq_axis) - np.min(freq_axis)) / 4
        elif len(dips) == 1:
            f_dip = freq_axis[dips[0][0]]
            width_est = 0.001
            f_center_est = f_dip
            f_delta_est = 0.003
            A_est *= 0.5
        else:
            f_dip_1 = freq_axis[dips[0][0]]
            f_dip_2 = freq_axis[dips[1][0]]
            width_est = abs(f_dip_2 - f_dip_1) / 2
            f_center_est = np.mean([f_dip_1, f_dip_2])
            f_delta_est = abs(f_dip_2 - f_dip_1)

        # Initial parameter bounds and guess
        p0 = [I0_est, A_est, width_est, f_center_est, f_delta_est]
        bounds = ([
            I0_est * 0.5, 0, 0.0001, freq_axis[0], 0
        ], [
            I0_est * 1.5, A_est * 2, np.ptp(freq_axis), freq_axis[-1], np.ptp(freq_axis)
        ])

        try:
            popt, _ = curve_fit(ODMRAnalyzer.double_dip_func, freq_axis, pixel_data, 
                              p0=p0, bounds=bounds, method='trf')
            
            return {
                "I0": popt[0],
                "A": popt[1],
                "width": popt[2],
                "f_center": popt[3],
                "f_delta": popt[4]
            }
        except RuntimeError:
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
        row_args = [(m, self.data, self.freq_axis, default_values) 
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
            fitted_analysis = {
                'I0': fitted_params['I0'][x, y],
                'amplitude': fitted_params['A'][x, y],
                'width': fitted_params['width'][x, y],
                'center_frequency': fitted_params['f_center'][x, y],
                'frequency_splitting': fitted_params['f_delta'][x, y]
            }
        
        return {
            'dip_frequencies': dip_frequencies,
            'dip_depths': dip_depths,
            'contrast': contrast,
            'num_dips': len(peaks),
            'fitted_parameters': fitted_analysis
        }

def main():
    experiment_number = 1730558912

    data_file = fr'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_1_biosample\2D_ODMR_scan_{experiment_number}.npy'
    json_file = fr'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_1_biosample\2D_ODMR_scan_{experiment_number}.json'
    
    # Initialize analyzer
    analyzer = ODMRAnalyzer(data_file, json_file, experiment_number)
    
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
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            # Perform full dataset fitting
            fitted_params = analyzer.fit_double_lorentzian(output_dir="./fitted_parameters")
            
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
                    single_pixel_params = analyzer.fit_single_pixel(pixel)

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
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()