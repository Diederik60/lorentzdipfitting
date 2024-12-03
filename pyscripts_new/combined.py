"""
ODMR (Optically Detected Magnetic Resonance) Analysis Module
Provides tools for analyzing ODMR spectroscopy data with double Lorentzian fitting.
Includes execution time tracking for performance optimization.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from functools import wraps
from datetime import datetime

def track_time(func):
    """
    Decorator that tracks execution time of functions.
    Stores timing data in the class's timing_log dictionary.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        execution_time = time.perf_counter() - start_time
        
        # Store timing information
        if not hasattr(self, 'timing_log'):
            self.timing_log = {}
        
        func_name = func.__name__
        if func_name not in self.timing_log:
            self.timing_log[func_name] = []
            
        self.timing_log[func_name].append({
            'timestamp': datetime.now(),
            'execution_time': execution_time,
            'args': str(args),
            'kwargs': str(kwargs)
        })
        
        # for checking the speed per iteration
        # print(f"{func_name} completed in {execution_time:.3f} seconds")
        return result
    return wrapper

class ODMRAnalyzer:
    """Analyzes ODMR spectroscopy data using double Lorentzian fitting."""
    
    def __init__(self, data_file, json_file, experiment_number):
        """
        Initialize analyzer with experimental data and parameters.
        
        Args:
            data_file (str): Path to .npy data file containing ODMR measurements
            json_file (str): Path to JSON file containing experiment parameters
            experiment_number (int): Unique identifier for the experiment
        """
        self.experiment_number = experiment_number
        self.timing_log = {}  # Initialize timing log
        self.load_data(data_file, json_file)

    @track_time
    def load_data(self, data_file, json_file):
        """Load and prepare experimental data and parameters."""
        self.data = np.load(data_file)
        
        with open(json_file, 'r') as f:
            self.params = json.load(f)
        
        # Create frequency axis in GHz
        self.freq_axis = np.linspace(
            self.params['min_freq'] / 1e9,
            self.params['max_freq'] / 1e9,
            self.params['num_measurements']
        )
        self.mean_spectrum = np.mean(self.data, axis=(0, 1))

    @staticmethod
    def double_lorentzian(f, I0, A, width, f_center, f_delta):
        """
        Double Lorentzian dip function for ODMR spectrum fitting.
        
        Args:
            f (array): Frequency points
            I0 (float): Baseline intensity
            A (float): Dip amplitude
            width (float): Dip width
            f_center (float): Center frequency
            f_delta (float): Frequency splitting between dips
        """
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) \
               - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)

    @track_time
    def fit_pixel(self, pixel_data, method='trf'):
        """
        Fit double Lorentzian to single pixel data.
        
        Args:
            pixel_data (array): Intensity data for one pixel
            method (str): Optimization method ('trf' or 'lm')
        """
        # Estimate initial parameters
        noise_std = np.std(np.concatenate((pixel_data[:5], pixel_data[-5:])))
        I0_est = np.mean(np.concatenate((pixel_data[:5], pixel_data[-5:])))
        A_est = np.ptp(pixel_data) - 2*noise_std

        # Find dips
        peaks, _ = find_peaks(-pixel_data, prominence=0.01)
        
        # Set initial parameters based on peak detection
        if len(peaks) == 0:
            p0 = [I0_est, A_est, 1.0, np.mean(self.freq_axis), 
                  (self.freq_axis[-1] - self.freq_axis[0])/4]
        elif len(peaks) == 1:
            p0 = [I0_est, A_est*0.5, 0.001, self.freq_axis[peaks[0]], 0.003]
        else:
            f_dips = self.freq_axis[peaks[:2]]
            p0 = [I0_est, A_est, np.diff(f_dips)[0]/2, 
                  np.mean(f_dips), np.diff(f_dips)[0]]

        # Perform fitting with bounds
        try:
            bounds = ([I0_est*0.5, 0, 0.0001, self.freq_axis[0], 0],
                     [I0_est*1.5, A_est*2, np.ptp(self.freq_axis), 
                      self.freq_axis[-1], np.ptp(self.freq_axis)])
            
            popt, _ = curve_fit(self.double_lorentzian, self.freq_axis, 
                              pixel_data, p0=p0, bounds=bounds, method=method)
            
            return dict(zip(['I0', 'A', 'width', 'f_center', 'f_delta'], popt))
        
        except RuntimeError:
            return dict(zip(['I0', 'A', 'width', 'f_center', 'f_delta'], p0))

    @track_time
    def fit_all_pixels(self, method='trf', output_dir=None):
        """
        Parallel fitting of all pixels in the dataset.
        
        Args:
            method (str): Optimization method ('trf' or 'lm')
            output_dir (str): Directory to save results
        """
        M, N, _ = self.data.shape
        start_time = time.perf_counter()
        processed_pixels = 0
        
        # Prepare parallel processing
        with Pool(cpu_count()) as pool:
            results = []
            for m in range(M):
                for n in range(N):
                    results.append(pool.apply_async(self.fit_pixel, 
                                 (self.data[m,n,:], method)))
            
            # Collect results with progress bar and timing updates
            fitted_params = {key: np.zeros((M,N)) for key in 
                           ['I0', 'A', 'width', 'f_center', 'f_delta']}
            
            for idx, result in tqdm(enumerate(results), total=M*N):
                m, n = divmod(idx, N)
                params = result.get()
                for key in fitted_params:
                    fitted_params[key][m,n] = params[key]
                
                # Update processing statistics every 100 pixels
                processed_pixels += 1
                if processed_pixels % 100 == 0:
                    elapsed_time = time.perf_counter() - start_time
                    pixels_per_second = processed_pixels / elapsed_time
                    remaining_pixels = (M*N) - processed_pixels
                    eta = remaining_pixels / pixels_per_second
                    print(f"\nProcessing speed: {pixels_per_second:.1f} pixels/second")
                    print(f"Estimated time remaining: {eta/60:.1f} minutes")

        if output_dir:
            self.save_results(fitted_params, output_dir)
        
        return fitted_params

    def plot_spectrum(self, x, y, fit_result=None):
        """
        Plot spectrum from a specific pixel with optional fit overlay.
        
        Args:
            x, y (int): Pixel coordinates
            fit_result (dict): Optional fitted parameters
        """
        spectrum = self.data[x, y, :]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.freq_axis, spectrum, alpha=0.5, label='Data')
        
        if fit_result:
            fitted_curve = self.double_lorentzian(self.freq_axis, 
                          fit_result['I0'], fit_result['A'], 
                          fit_result['width'], fit_result['f_center'], 
                          fit_result['f_delta'])
            plt.plot(self.freq_axis, fitted_curve, 'r--', label='Fit')
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('ODMR Signal (a.u.)')
        plt.title(f'ODMR Spectrum at Pixel ({x}, {y})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf(), plt.gca()

    @track_time
    def save_results(self, fitted_params, output_dir):
        """Save fitted parameters and timing logs to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save fitted parameters
        params_file = os.path.join(output_dir, 
                     f"odmr_fit_{self.experiment_number}.npy")
        stacked_params = np.stack([fitted_params[key] for key in 
                       ['I0', 'A', 'width', 'f_center', 'f_delta']], axis=-1)
        np.save(params_file, stacked_params)
        
        # Save timing log
        timing_file = os.path.join(output_dir,
                     f"timing_log_{self.experiment_number}.json")
        with open(timing_file, 'w') as f:
            json.dump(self.timing_log, f, default=str, indent=2)
        
        return params_file

def main():
    """Command-line interface for ODMR analysis."""
    experiment_number = 1730558912
    
    data_file = fr'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_1_biosample\2D_ODMR_scan_{experiment_number}.npy'
    json_file = fr'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_1_biosample\2D_ODMR_scan_{experiment_number}.json'
    
    analyzer = ODMRAnalyzer(data_file, json_file, experiment_number)
    
    while True:
        print("\nODMR Analysis Options:")
        print("1. Fit all pixels")
        print("2. Analyze single pixel")
        print("3. View timing statistics")
        print("4. Exit")
        
        choice = input("Choice (1-4): ")
        
        if choice == '1':
            method = input("Optimization method (trf/lm): ")
            fitted_params = analyzer.fit_all_pixels(method=method, 
                          output_dir="./results")
        
        elif choice == '2':
            x = int(input(f"X coordinate (0-{analyzer.data.shape[0]-1}): "))
            y = int(input(f"Y coordinate (0-{analyzer.data.shape[1]-1}): "))
            
            fit_result = analyzer.fit_pixel(analyzer.data[x,y,:])
            analyzer.plot_spectrum(x, y, fit_result)
            plt.show()
            
            print("\nFitted Parameters:")
            for key, value in fit_result.items():
                print(f"{key}: {value:.4f}")
        
        elif choice == '3':
            print("\nTiming Statistics:")
            for func_name, timings in analyzer.timing_log.items():
                times = [t['execution_time'] for t in timings]
                print(f"\n{func_name}:")
                print(f"  Calls: {len(times)}")
                print(f"  Average time: {np.mean(times):.3f} seconds")
                print(f"  Min time: {np.min(times):.3f} seconds")
                print(f"  Max time: {np.max(times):.3f} seconds")
        
        elif choice == '4':
            break

if __name__ == "__main__":
    main()