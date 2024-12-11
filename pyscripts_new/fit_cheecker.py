import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.widgets import Slider, Button
import json

class ODMRFitChecker:
    def __init__(self, fitted_params_file, original_data_file, json_params_file):
        """
        Initialize the ODMR fit checker with data files
        
        Args:
            fitted_params_file (str): Path to .npy file with fitted parameters
            original_data_file (str): Path to original ODMR scan data
            json_params_file (str): Path to JSON file with frequency parameters
        """
        # Load the fitted parameters (shape: M x N x 5)
        self.fitted_params = np.load(fitted_params_file)
        
        # Load original data
        self.original_data = np.load(original_data_file)
        
        # Load frequency parameters
        with open(json_params_file, 'r') as f:
            params = json.load(f)
            
        # Create frequency axis
        self.freq_axis = np.linspace(
            params['min_freq'] / 1e9,  # Convert to GHz
            params['max_freq'] / 1e9,
            params['num_measurements']
        )
        
        # Get data dimensions
        self.M, self.N = self.fitted_params.shape[:2]
        
    def double_lorentzian(self, f, I0, A, width, f_center, f_delta):
        """Calculate double Lorentzian function with given parameters"""
        return I0 - A/(1 + ((f_center - 0.5*f_delta - f)/width)**2) - \
               A/(1 + ((f_center + 0.5*f_delta - f)/width)**2)
    
    def create_interactive_viewer(self):
        """Create an interactive plot to browse through pixels"""
        # Set up the figure and subplots
        self.fig, (self.ax_data, self.ax_map) = plt.subplots(1, 2, figsize=(15, 6))
        plt.subplots_adjust(bottom=0.25)  # Make room for sliders
        
        # Initial pixel coordinates
        self.x_idx, self.y_idx = 0, 0
        
        # Plot initial spectrum
        self.spectrum_line, = self.ax_data.plot(self.freq_axis, 
                                              self.original_data[self.x_idx, self.y_idx], 
                                              'b.', label='Data')
        
        # Calculate and plot initial fit
        params = self.fitted_params[self.x_idx, self.y_idx]
        fitted_curve = self.double_lorentzian(self.freq_axis, *params)
        self.fit_line, = self.ax_data.plot(self.freq_axis, fitted_curve, 'r-', 
                                          label='Fit')
        
        # Create contrast map
        contrast_map = self.fitted_params[:, :, 1]  # Using amplitude (A) parameter
        self.map_img = self.ax_map.imshow(contrast_map.T, origin='lower', 
                                         cmap='viridis')
        self.pixel_marker, = self.ax_map.plot(self.x_idx, self.y_idx, 'rx')
        
        # Add colorbar
        plt.colorbar(self.map_img, ax=self.ax_map, label='Contrast (A)')
        
        # Set up axis labels
        self.ax_data.set_xlabel('Frequency (GHz)')
        self.ax_data.set_ylabel('ODMR Signal (a.u.)')
        self.ax_data.legend()
        self.ax_map.set_xlabel('X Position')
        self.ax_map.set_ylabel('Y Position')
        
        # Create sliders for x and y coordinates
        ax_x = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_y = plt.axes([0.2, 0.05, 0.6, 0.03])
        
        self.x_slider = Slider(ax_x, 'X', 0, self.M-1, valinit=0, valstep=1)
        self.y_slider = Slider(ax_y, 'Y', 0, self.N-1, valinit=0, valstep=1)
        
        # Add update function for sliders
        def update(val):
            self.x_idx = int(self.x_slider.val)
            self.y_idx = int(self.y_slider.val)
            
            # Update spectrum plot
            self.spectrum_line.set_ydata(self.original_data[self.x_idx, self.y_idx])
            
            # Update fit plot
            params = self.fitted_params[self.x_idx, self.y_idx]
            fitted_curve = self.double_lorentzian(self.freq_axis, *params)
            self.fit_line.set_ydata(fitted_curve)
            
            # Update pixel marker
            self.pixel_marker.set_data([self.x_idx], [self.y_idx])
            
            # Update title with fit parameters
            self.ax_data.set_title(
                f'Pixel ({self.x_idx}, {self.y_idx})\n' + 
                f'I0={params[0]:.3f}, A={params[1]:.3f}, w={params[2]:.3f}, ' + 
                f'fc={params[3]:.3f}, fd={params[4]:.3f}'
            )
            
            # Rescale y-axis to fit data
            y_data = self.original_data[self.x_idx, self.y_idx]
            y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
            self.ax_data.set_ylim(np.min(y_data) - y_margin, 
                                np.max(y_data) + y_margin)
            
            self.fig.canvas.draw_idle()
        
        # Connect sliders to update function
        self.x_slider.on_changed(update)
        self.y_slider.on_changed(update)
        
        # Initialize plot
        update(None)
        
        plt.show()

def main():
    # Replace these with your actual file paths
    experiment_number = 1730664256
    base_path = r'C:\Users\Diederik\Documents\BEP\lorentzdipfitting\data\dataset_1_biosample'
    
    fitted_params_file = f"./fitted_parameters/lorentzian_params_{experiment_number}.npy"
    original_data_file = f"{base_path}/2D_ODMR_scan_{experiment_number}.npy"
    json_params_file = f"{base_path}/2D_ODMR_scan_{experiment_number}.json"
    
    # Create checker instance
    checker = ODMRFitChecker(fitted_params_file, original_data_file, json_params_file)
    
    # Launch interactive viewer
    checker.create_interactive_viewer()

if __name__ == "__main__":
    main()