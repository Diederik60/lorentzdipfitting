import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
import os

def double_dip_func(f, I0, A, width, f_center, f_delta):
    '''Double Lorentzian dip.'''
    return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)

def fit_double_lorentzian(data, freq, tail=5, thresholds=[3, 5], error_threshold=0.1, default_values=None, method='trf', output_dir=None):
    '''
    Fits the double Lorentzian model to the (M, N, F) shaped input data.
    
    Arguments:
    - data: numpy array of shape (M, N, F) with the intensity values
    - freq: numpy array of shape (F,) with the frequency values
    - tail: number of pixels to take from the side as a sample for baseline and noise
    - thresholds: relative values used to detect dips
    - error_threshold: maximum acceptable fitting error
    - default_values: dictionary of default parameter values if fitting fails
    - method: scipy curve_fit optimization method
    - output_dir: directory to save fitted parameters (optional)

    Returns:
    - A dictionary containing five (M, N) numpy arrays for the fitted parameters: I0, A, width, f_center, f_delta.
    '''
    M, N, F = data.shape

    # Initialize arrays for initial guesses and fitted parameters
    fitted_params = {
        "I0": np.zeros((M, N)),
        "A": np.zeros((M, N)),
        "width": np.zeros((M, N)),
        "f_center": np.zeros((M, N)),
        "f_delta": np.zeros((M, N)),
    }
    
    # Default values if not provided
    if default_values is None:
        default_values = {"I0": 1.0, "A": 0, "width": 1.0, "f_center": 2.87, "f_delta": 0.0}

    print("Starting parameter estimation and fitting")
    
    for m in tqdm(range(M)):
        for n in range(N):
            # Get the intensity data for the current pixel
            intens = data[m, n, :]

            # Estimate noise level
            tail_values = np.concatenate((intens[:tail], intens[-tail:]))
            noise_std = np.std(tail_values)
            I0_est = np.average(tail_values)
            A_est = np.max(intens) - np.min(intens) - 2*noise_std

            dips = []
            dip_start = 0
            in_dip = False
            
            # Calculate absolute threshold positions
            threshold_low = np.min(intens) + thresholds[0]*noise_std
            threshold_high = np.min(intens) + thresholds[1]*noise_std

            for i in range(F):
                if in_dip:
                    if intens[i] >= threshold_high:
                        dips.append((dip_start, i))
                        in_dip = False
                else:
                    if intens[i] <= threshold_low:
                        dip_start = i
                        in_dip = True

            # Initial parameter estimation logic (same as previous implementation)
            if len(dips) == 0:
                width_est = 1
                f_center_est = np.average(freq)
                f_delta_est = (np.max(freq) - np.min(freq)) / 4
            elif len(dips) == 1:
                f_dip_start = freq[dips[0][0]]
                f_dip_finish = freq[dips[0][1]]
                width_est = 0.5 * (f_dip_finish - f_dip_start)
                f_center_est = 0.5 * (f_dip_start + f_dip_finish)
                f_delta_est = 0.003 #Assume the nuclear splitting as the only splitting 
                A_est *= 0.5
            elif len(dips) == 2:
                f_dip_start_1 = freq[dips[0][0]]
                f_dip_finish_1 = freq[dips[0][1]]
                f_dip_start_2 = freq[dips[1][0]]
                f_dip_finish_2 = freq[dips[1][1]]
                width_est = 0.25 * (f_dip_finish_1 + f_dip_finish_2 - f_dip_start_1 - f_dip_start_2)
                middle_1 = 0.5 * (f_dip_finish_1 + f_dip_start_1) 
                middle_2 = 0.5 * (f_dip_finish_2 + f_dip_start_2) 
                f_center_est = 0.5 * (middle_1 + middle_2)
                f_delta_est = middle_2 - middle_1
            else:
                # For multiple dips, use average and standard deviation
                width_est = 0
                f_center_est = 0
                for i in range(len(dips)):
                    f_dip_start = freq[dips[i][0]]
                    f_dip_finish = freq[dips[i][1]]
                    width_est += 0.5 * (f_dip_finish - f_dip_start)
                    f_center_est += 0.5 * (f_dip_finish + f_dip_start)
                
                width_est /= len(dips)
                f_center_est /= len(dips)
                
                # Estimate f_delta using standard deviation
                middles = np.zeros(len(dips))
                for i in range(len(dips)):
                    f_dip_start = freq[dips[i][0]]
                    f_dip_finish = freq[dips[i][1]]
                    middles[i] = 0.5 * (f_dip_finish - f_dip_start)
                f_delta_est = np.std(middles)

            # Initial parameter bounds and guess
            p0 = [I0_est, A_est, width_est, f_center_est, f_delta_est]
            bounds = ([
                I0_est * 0.5, 0, 0.0001, freq[0], 0
            ], [
                I0_est * 1.5, A_est * 2, np.ptp(freq), freq[-1], np.ptp(freq)
            ])

            try:
                # Curve fitting
                popt, pcov = curve_fit(double_dip_func, freq, intens, 
                                       p0=p0, 
                                       bounds=bounds, 
                                       method=method)
                
                # Store fitted parameters
                fitted_params["I0"][m, n] = popt[0]
                fitted_params["A"][m, n] = popt[1]
                fitted_params["width"][m, n] = popt[2]
                fitted_params["f_center"][m, n] = popt[3]
                fitted_params["f_delta"][m, n] = popt[4]

            except RuntimeError:
                # If fitting fails, use default or initial estimates
                for key in fitted_params:
                    fitted_params[key][m, n] = default_values[key]
    
    # Save parameters if output directory is specified
    if output_dir is not None:
        save_fitted_parameters(fitted_params, output_dir)
    
    return fitted_params

def save_fitted_parameters(fitted_params, output_dir):
    """
    Save fitted parameters as a single .npy file with shape (M, N, P)
    
    Arguments:
    - fitted_params: Dictionary of fitted parameter arrays
    - output_dir: Directory to save the parameters
    
    Saved file structure:
    - Single .npy file with shape (M, N, 5)
    - Dimensions order: [I0, A, width, f_center, f_delta]
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect parameters in a specific order
    param_order = ['I0', 'A', 'width', 'f_center', 'f_delta']
    
    # Stack parameters along the third axis
    stacked_params = np.stack([fitted_params[param] for param in param_order], axis=-1)
    
    # Generate filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"lorentzian_params_{timestamp}.npy")
    
    # Save as single .npy file
    np.save(filepath, stacked_params)
    print(f"Fitted parameters saved to {filepath}")
    print(f"Saved array shape: {stacked_params.shape}")

def load_fitted_parameters(filepath):
    """
    Load fitted parameters from .npy file
    
    Returns:
    - NumPy array with shape (M, N, 5)
    - Parameter order: [I0, A, width, f_center, f_delta]
    """
    return np.load(filepath)

if __name__ == "__main__":
    M, N, F = 50, 50, 100  # Example dimensions
    freq = np.linspace(2.85, 2.89, F)

    # Generate synthetic data with two dips
    intensity = np.zeros((M, N, F))
    for m in range(M):
        for n in range(N):
            intensity[m, n, :] = double_dip_func(freq, 1, 0.15, np.random.uniform(0.001, 0.004), 2.87, np.random.uniform(0.0, 0.010))
            intensity[m, n, :] += np.random.normal(0, 0.005, F)

    # Call the fitting function with an output directory
    fitted_params = fit_double_lorentzian(intensity, freq, output_dir="./fitted_parameters")