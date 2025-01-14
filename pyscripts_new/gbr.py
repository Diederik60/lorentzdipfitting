import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import psutil
import gc
import pickle
import os
from datetime import datetime
import json

def get_memory_usage():
    """Return current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

class ODMRBoostingModel:
    """Physics-informed gradient boosting model for ODMR spectrum analysis."""
    
    def __init__(self, freq_axis, config=None):
        self.freq_axis = freq_axis
        self.feature_scaler = StandardScaler()
        self.param_scalers = {
            'I0': StandardScaler(),
            'A': StandardScaler(),
            'width': StandardScaler(),
            'f_center': StandardScaler(),
            'f_delta': StandardScaler()
        }
        
        # Processing parameters
        self.batch_size = 100  # Process 100 spectra at a time
        
        # Model hyperparameters
        self.n_estimators = 100 if config is None else config.n_estimators
        self.learning_rate = 0.1 if config is None else config.learning_rate
        self.max_depth = 5 if config is None else config.max_depth
        
        # Initialize models for each parameter
        self.models = {
            'I0': GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                loss='huber'
            ),
            'A': GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                loss='huber'
            ),
            'width': GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                loss='huber'
            ),
            'f_center': GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                loss='squared_error'
            ),
            'f_delta': GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                loss='squared_error'
            )
        }
    
    def extract_physics_features(self, spectrum):
        """Extract physically meaningful features from an ODMR spectrum."""
        try:
            features = []
            
            # 1. Baseline and noise estimation
            tail_size = 10
            tails = np.concatenate([spectrum[:tail_size], spectrum[-tail_size:]])
            baseline_mean = np.mean(tails)
            baseline_std = np.std(tails)
            features.extend([baseline_mean, baseline_std])
            
            # 2. Multi-scale gradient information
            for sigma in [1, 2]:
                smoothed = gaussian_filter1d(spectrum, sigma)
                grad = np.gradient(smoothed)
                curvature = np.gradient(grad)
                
                features.extend([
                    np.mean(grad), np.std(grad),
                    np.mean(curvature), np.std(curvature),
                    np.min(grad), np.max(grad),
                    np.min(curvature), np.max(curvature)
                ])
            
            # 3. Peak detection features
            inverted = baseline_mean - spectrum
            peaks, properties = find_peaks(inverted, prominence=0.1*np.max(inverted))
            
            if len(peaks) >= 2:
                peak_heights = properties['prominences']
                peak_distances = np.diff(self.freq_axis[peaks])
                features.extend([
                    np.mean(peak_heights),
                    np.std(peak_heights),
                    np.mean(peak_distances),
                    np.min(peak_distances),
                    np.max(peak_distances)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # 4. Local properties around dips
            for peak_idx in peaks[:2]:
                window = slice(max(0, peak_idx-5), min(len(spectrum), peak_idx+6))
                local_region = spectrum[window]
                features.extend([
                    np.mean(local_region),
                    np.min(local_region),
                    np.max(local_region),
                    np.std(local_region)
                ])
            
            while len(peaks) < 2:
                features.extend([0, 0, 0, 0])
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}")
            raise
    
    def fit(self, X, y):
        """Train the model on spectrum data and corresponding parameters."""
        print(f"\nInitial memory usage: {get_memory_usage():.1f} MB")
        print("\nExtracting physics features from spectra...")
        
        # Get feature dimensionality
        sample_features = self.extract_physics_features(X[0])
        feature_dim = len(sample_features)
        print(f"Each spectrum will generate {feature_dim} features")
        
        # Initialize feature array
        total_spectra = len(X)
        X_features = np.zeros((total_spectra, feature_dim), dtype=np.float32)
        
        # Process in batches
        num_batches = (total_spectra + self.batch_size - 1) // self.batch_size
        
        for batch in range(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, total_spectra)
            
            print(f"\nProcessing batch {batch + 1}/{num_batches}")
            print(f"Memory usage: {get_memory_usage():.1f} MB")
            
            for i in range(start_idx, end_idx):
                if (i + 1) % 10 == 0:
                    print(f"  Spectrum {i + 1}/{total_spectra}")
                X_features[i] = self.extract_physics_features(X[i])
            
            gc.collect()
        
        print(f"\nFeature extraction complete!")
        print(f"Memory usage after feature extraction: {get_memory_usage():.1f} MB")
        
        # Scale features and parameters
        print("\nScaling features and parameters...")
        X_features = self.feature_scaler.fit_transform(X_features)
        y_scaled = np.zeros_like(y, dtype=np.float32)
        
        for i, param_name in enumerate(self.models.keys()):
            y_scaled[:, i] = self.param_scalers[param_name].fit_transform(
                y[:, i].reshape(-1, 1)
            ).ravel()
        
        # Train individual models
        print("\nTraining individual parameter models:")
        for i, (param_name, model) in enumerate(self.models.items()):
            print(f"\nTraining {param_name} model ({i+1}/5)...")
            print(f"Memory usage before training: {get_memory_usage():.1f} MB")
            model.fit(X_features, y_scaled[:, i])
            print(f"Memory usage after training: {get_memory_usage():.1f} MB")
            gc.collect()
    
    def predict(self, X):
        """Predict parameters for new spectra."""
        total_spectra = len(X)
        feature_dim = len(self.extract_physics_features(X[0]))
        X_features = np.zeros((total_spectra, feature_dim), dtype=np.float32)
        
        # Process in batches
        num_batches = (total_spectra + self.batch_size - 1) // self.batch_size
        
        print("\nExtracting features for prediction...")
        for batch in range(num_batches):
            start_idx = batch * self.batch_size
            end_idx = min((batch + 1) * self.batch_size, total_spectra)
            
            for i in range(start_idx, end_idx):
                X_features[i] = self.extract_physics_features(X[i])
            
            if (batch + 1) % 5 == 0:
                print(f"Processed batch {batch + 1}/{num_batches}")
        
        # Scale features
        X_features = self.feature_scaler.transform(X_features)
        
        # Generate predictions
        predictions = []
        print("\nGenerating predictions...")
        for param_name, model in self.models.items():
            print(f"Predicting {param_name}...")
            pred = model.predict(X_features)
            pred = self.param_scalers[param_name].inverse_transform(
                pred.reshape(-1, 1)
            ).ravel()
            predictions.append(pred)
        
        return np.column_stack(predictions)

def evaluate_boosting_predictions(model, X_test, y_test, param_names, save_dir, show_plot=True):
    """
    Evaluate and visualize predictions from the boosting model, saving results to disk.
    
    Args:
        model: Trained ODMRBoostingModel instance
        X_test: Test input data
        y_test: True parameter values
        param_names: List of parameter names
        save_dir: Directory to save results
        show_plot: Whether to display visualization plots
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate predictions
    print("\nGenerating predictions for evaluation...")
    y_pred = model.predict(X_test)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(y_pred, columns=param_names)
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # Calculate metrics
    metrics = {}
    for i, param in enumerate(param_names):
        mse = np.mean((y_test[:, i] - y_pred[:, i]) ** 2)
        r2 = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1] ** 2
        metrics[f'{param}_mse'] = float(mse)  # Convert to float for JSON serialization
        metrics[f'{param}_r2'] = float(r2)
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create evaluation plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot parameter predictions
    for i, param in enumerate(param_names):
        if i < len(axes) - 1:
            axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
            
            min_val = min(y_test[:, i].min(), y_pred[:, i].min())
            max_val = max(y_test[:, i].max(), y_pred[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
            
            mse = metrics[f'{param}_mse']
            r2 = metrics[f'{param}_r2']
            
            axes[i].set_title(f'{param} Predictions')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].text(0.05, 0.95, 
                       f'MSE: {mse:.2e}\nR²: {r2:.3f}',
                       transform=axes[i].transAxes,
                       verticalalignment='top')
    
    # Add feature importance plot
    if len(axes) > 5:
        importances = model.models['f_center'].feature_importances_
        top_n = min(10, len(importances))
        top_indices = np.argsort(importances)[-top_n:]
        
        axes[5].barh(range(top_n), importances[top_indices])
        axes[5].set_yticks(range(top_n))
        axes[5].set_title('Top Feature Importances\nfor f_center prediction')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    
    # Save feature importances for each parameter
    importance_data = {}
    for param, model_instance in model.models.items():
        importance_data[param] = list(model_instance.feature_importances_)
    
    with open(os.path.join(save_dir, 'feature_importances.json'), 'w') as f:
        json.dump(importance_data, f, indent=4)
    
    # Create a summary report
    with open(os.path.join(save_dir, 'summary_report.txt'), 'w') as f:
        f.write("ODMR Boosting Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Model Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        for param in param_names:
            f.write(f"\n{param}:\n")
            f.write(f"  MSE:  {metrics[f'{param}_mse']:.2e}\n")
            f.write(f"  R²:   {metrics[f'{param}_r2']:.3f}\n")
        
        f.write("\nFiles Generated:\n")
        f.write("-" * 30 + "\n")
        f.write("- predictions.csv: Model predictions for test set\n")
        f.write("- metrics.json: Detailed performance metrics\n")
        f.write("- evaluation_plots.png: Visualization of model performance\n")
        f.write("- feature_importances.json: Feature importance scores\n")
        f.write("- trained_model.pkl: Saved model file\n")
    
    return y_pred, metrics

def main():
    print("\n=== Starting ODMR Boosting Model Training ===\n")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'odmr_boosting_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Redirect stdout to a log file
    log_file = open(os.path.join(save_dir, 'training_log.txt'), 'w')
    import sys
    sys.stdout = log_file
    
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    try:
        # Load and prepare data
        print("Loading data...")
        df = pd.read_csv('test.csv')
        print(f"Successfully loaded data with {len(df)} rows")
        
        # Previous data preparation code remains the same...
        
        # Train model
        print("\nTraining model...")
        model = ODMRBoostingModel(freq_axis)
        model.fit(X_train, y_train)
        print("Model training completed successfully!")
        
        # Save trained model
        print("\nSaving trained model...")
        with open(os.path.join(save_dir, 'trained_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        
        # Evaluate model
        print("\nEvaluating model performance...")
        y_pred, metrics = evaluate_boosting_predictions(
            model, X_test, y_test, param_cols, 
            save_dir=save_dir, show_plot=False
        )
        
        print("\nAll results have been saved to:", save_dir)
        return model, y_pred, metrics
        
    except Exception as e:
        print(f"\nAn error occurred during execution:")
        print(str(e))
        import traceback
        traceback.print_exc()
        return None, None, None
    
    finally:
        # Restore stdout and close log file
        sys.stdout = sys.__stdout__
        log_file.close()

if __name__ == "__main__":
    model, predictions, metrics = main()