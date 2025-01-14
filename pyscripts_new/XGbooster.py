import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ODMRDataProcessor:
    def __init__(self):
        self.input_scaler = StandardScaler()
        self.raw_freq_columns = [f'raw_freq_{i}' for i in range(100)]
        self.param_columns = ['I0', 'A', 'width', 'f_center', 'f_delta']
        self.scalers = {param: StandardScaler() for param in self.param_columns}
        self.f_center_mean = None

    def preprocess_data(self, df):
        X = df[self.raw_freq_columns].values
        y = df[self.param_columns].values
        
        # Print data ranges before preprocessing
        for param in self.param_columns:
            data = df[param].values
            logger.info(f"{param} range: {data.min():.6f} to {data.max():.6f}")
        
        X_scaled = self.input_scaler.fit_transform(X)
        y_processed = np.zeros_like(y)
        
        # Store f_center mean for later use
        self.f_center_mean = np.mean(y[:, self.param_columns.index('f_center')])
        
        for i, param in enumerate(self.param_columns):
            data = y[:, i].reshape(-1, 1)
            if param == 'f_center':
                y_processed[:, i] = ((data - self.f_center_mean) / self.f_center_mean).ravel()
            elif param in ['width', 'f_delta']:
                y_processed[:, i] = (data / self.f_center_mean).ravel()
            else:
                y_processed[:, i] = self.scalers[param].fit_transform(np.log1p(data)).ravel()
        
        return X_scaled, y_processed

    def inverse_transform_outputs(self, y_scaled):
        y_orig = np.zeros_like(y_scaled)
        
        for i, param in enumerate(self.param_columns):
            data = y_scaled[:, i].reshape(-1, 1)
            if param == 'f_center':
                y_orig[:, i] = (data * self.f_center_mean + self.f_center_mean).ravel()
            elif param in ['width', 'f_delta']:
                y_orig[:, i] = (data * self.f_center_mean).ravel()
            else:
                y_orig[:, i] = np.expm1(self.scalers[param].inverse_transform(data)).ravel()
        
        return y_orig

class ODMRRegressor:
    def __init__(self):
        self.models = {}
        self.param_columns = ['I0', 'A', 'width', 'f_center', 'f_delta']
        self.training_history = {param: {'train': [], 'val': []} for param in self.param_columns}
        
    def create_base_model(self, param):
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 1
        }
        
        # Parameter-specific adjustments
        if param in ['width', 'f_delta']:
            params = {
                **base_params,
                'learning_rate': 0.005,
                'max_depth': 4,
                'min_child_weight': 3
            }
        elif param == 'f_center':
            params = {
                **base_params,
                'learning_rate': 0.008,
                'max_depth': 8
            }
        else:  # I0, A
            params = base_params
            
        return params
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train separate models for each parameter"""
        for i, param in enumerate(self.param_columns):
            logger.info(f"Training model for {param}...")
            params = self.create_base_model(param)
            
            dtrain = xgb.DMatrix(X, label=y[:, i])
            evaluation = {'train': [], 'eval': []}
            
            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val[:, i])
                evallist = [(dtrain, 'train'), (dval, 'eval')]
                
                # Create a proper callback class
                class CustomCallback(xgb.callback.TrainingCallback):
                    def __init__(self, evaluation_dict):
                        self.evaluation_dict = evaluation_dict

                    def after_iteration(self, model, epoch, evals_log):
                        for key in evals_log:
                            metric_value = evals_log[key]['rmse'][-1]
                            if key == 'train':
                                self.evaluation_dict['train'].append(metric_value)
                            else:
                                self.evaluation_dict['eval'].append(metric_value)
                        return False
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=evallist,
                    early_stopping_rounds=50,
                    verbose_eval=100,
                    callbacks=[CustomCallback(evaluation)]
                )
            else:
                model = xgb.train(params, dtrain, num_boost_round=100)
            
            self.models[param] = model
            self.training_history[param] = evaluation
            
            # Log feature importance
            importance = model.get_score(importance_type='gain')
            logger.info(f"\nTop 10 important features for {param}:")
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat, imp in sorted_imp:
                logger.info(f"Feature {feat}: {imp:.4f}")
        
    def predict(self, X):
        """Predict all parameters for given spectra"""
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X)
        predictions = np.zeros((X.shape[0], len(self.param_columns)))
        
        # Get predictions for each parameter
        for i, param in enumerate(self.param_columns):
            predictions[:, i] = self.models[param].predict(dtest)
        
        return predictions

def evaluate_predictions(model, X_test, y_test, processor, show_plot=True):
    y_pred = model.predict(X_test)
    y_test_orig = processor.inverse_transform_outputs(y_test)
    y_pred_orig = processor.inverse_transform_outputs(y_pred)

    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Plot parameter predictions (first 5 subplots)
        for i, param in enumerate(processor.param_columns):
            axes[i].scatter(y_test_orig[:, i], y_pred_orig[:, i], alpha=0.5)
            min_val = min(y_test_orig[:, i].min(), y_pred_orig[:, i].min())
            max_val = max(y_test_orig[:, i].max(), y_pred_orig[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

            mse = np.mean((y_test_orig[:, i] - y_pred_orig[:, i]) ** 2)
            r2 = np.corrcoef(y_test_orig[:, i], y_pred_orig[:, i])[0, 1] ** 2

            axes[i].set_title(f'{param} Predictions')
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].text(0.05, 0.95, f'MSE: {mse:.2e}\nR²: {r2:.3f}',
                         transform=axes[i].transAxes,
                         verticalalignment='top')

        # Plot training history in the sixth subplot
        axes[5].set_title('Model Learning Curves')
        axes[5].set_xlabel('Iterations')
        axes[5].set_ylabel('RMSE')
        
        # Plot learning curves for each parameter
        for param in processor.param_columns:
            if model.training_history[param]:  # Check if history exists
                history = model.training_history[param]
                train_loss = history['train']
                val_loss = history['eval']
                iters = range(1, len(train_loss) + 1)
                
                axes[5].plot(iters, train_loss, '--', label=f'{param} (train)', alpha=0.5)
                axes[5].plot(iters, val_loss, '-', label=f'{param} (val)', alpha=0.5)
        
        axes[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[5].grid(True)

        plt.tight_layout()
        plt.show()

    metrics = {f'{param}_mse': np.mean((y_test_orig[:, i] - y_pred_orig[:, i]) ** 2)
               for i, param in enumerate(processor.param_columns)}

    return y_pred_orig, metrics

def main():
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(r"C:\Users\Diederik\Documents\BEP\test.csv")
    
    # Initialize processor and preprocess data
    processor = ODMRDataProcessor()
    X_scaled, y_scaled = processor.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    logger.info("Training models...")
    model = ODMRRegressor()
    model.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate
    logger.info("Evaluating models...")
    y_pred_orig, metrics = evaluate_predictions(model, X_test, y_test, processor)
    
    # Print metrics
    logger.info("\nFinal Metrics:")
    for param, mse in metrics.items():
        logger.info(f"{param}: {mse:.2e}")
    
    return model, processor

if __name__ == "__main__":
    model, processor = main()