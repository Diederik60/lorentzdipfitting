import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from matplotlib.widgets import RectangleSelector

class ODMRAnalyzer:
    def __init__(self, data_file, json_file):
        # Load data and parameters
        self.data = np.load(data_file)
        with open(json_file, 'r') as f:
            self.params = json.load(f)
        
        # Create frequency axis
        self.freq_axis = np.linspace(
            self.params['min_freq'] / 1e9,  # Convert to GHz
            self.params['max_freq'] / 1e9,
            self.params['num_measurements']
        )
        
        # Calculate mean spectrum
        self.mean_spectrum = np.mean(self.data, axis=(0, 1))
        
    def plot_pixel_spectrum(self, x, y, smooth=True):
            """Plot spectrum from a specific pixel with original data points"""
            spectrum = self.data[x, y, :]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot original data points
            ax.scatter(self.freq_axis, spectrum, color='red', alpha=0.5, label='Raw Data Points', zorder=2)
            
            if smooth:
                # Apply Savitzky-Golay filter for smoothing
                smoothed_spectrum = savgol_filter(spectrum, window_length=7, polyorder=3)
                ax.plot(self.freq_axis, smoothed_spectrum, 'b-', linewidth=2, label='Smoothed Spectrum', zorder=3)
            
            # Find dips
            # Invert spectrum for peak finding since we're looking for dips
            inverted = -spectrum
            peaks, properties = find_peaks(inverted, prominence=0.01)
            
            # Plot dip positions
            for peak in peaks:
                ax.axvline(x=self.freq_axis[peak], color='g', alpha=0.3, linestyle='--')
                ax.text(self.freq_axis[peak], spectrum[peak], 
                    f'{self.freq_axis[peak]:.3f} GHz', 
                    rotation=90, verticalalignment='bottom')
            
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('ODMR Signal (a.u.)')
            ax.set_title(f'ODMR Spectrum at Pixel ({x}, {y})')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            return fig, ax
    
    def plot_contrast_map(self):
        """Create a 2D heat map of the ODMR contrast"""
        # Calculate contrast as max-min for each pixel
        contrast_map = np.ptp(self.data, axis=2)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(contrast_map.T, origin='lower', cmap='viridis',
                      extent=[self.params['x1'], self.params['x2'],
                             self.params['y1'], self.params['y2']])
        plt.colorbar(im, ax=ax, label='Contrast')
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_title('ODMR Contrast Map')
        return fig, ax, contrast_map
    
    def analyze_spectrum(self, x, y):
        """Analyze the spectrum at a given pixel"""
        spectrum = self.data[x, y, :]
        smoothed = savgol_filter(spectrum, window_length=7, polyorder=3)
        
        # Find dips
        inverted = -smoothed
        peaks, properties = find_peaks(inverted, prominence=0.01)
        
        dip_frequencies = self.freq_axis[peaks]
        dip_depths = spectrum[peaks]
        
        # Calculate contrast
        contrast = np.ptp(spectrum)
        
        return {
            'dip_frequencies': dip_frequencies,
            'dip_depths': dip_depths,
            'contrast': contrast,
            'num_dips': len(peaks)
        }

if __name__ == "__main__":
    analyzer = ODMRAnalyzer(r'C:\Users\Diederik\Documents\BEP\measurement_stuff_new\nov-2024 bonded sample\2D_ODMR_scan_1731345033.npy', r'C:\Users\Diederik\Documents\BEP\measurement_stuff_new\nov-2024 bonded sample\2D_ODMR_scan_1731345033.json')
    
    # Plot contrast map
    #fig_map, ax_map, contrast_map = analyzer.plot_contrast_map()
    #plt.show()
    
    # Example: analyze center pixel
    center_x, center_y = 23, 12
    fig_spectrum, ax_spectrum = analyzer.plot_pixel_spectrum(center_x, center_y)
    plt.show()
    
    # Print analysis results
    analysis = analyzer.analyze_spectrum(center_x, center_y)
    print("\nSpectrum Analysis Results:")
    print(f"Number of dips found: {analysis['num_dips']}")
    print("\nDip frequencies (GHz):")
    for freq, depth in zip(analysis['dip_frequencies'], analysis['dip_depths']):
        print(f"  {freq:.3f} GHz (depth: {depth:.3e})")
    print(f"\nTotal contrast: {analysis['contrast']:.3e}")