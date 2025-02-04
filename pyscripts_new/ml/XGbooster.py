
import numpy as np
import pandas as pd
import xgboost as xgb
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
        
        # Scale raw frequency features separately
        scaler_raw_freq = MinMaxScaler()  # Alternative: StandardScaler
        X_scaled = scaler_raw_freq.fit_transform(X)

        # Handle target variables
        y_processed = np.zeros_like(y)
        self.f_center_mean = np.mean(y[:, self.param_columns.index('f_center')])

        for i, param in enumerate(self.param_columns):
            data = y[:, i].reshape(-1, 1)

            # Log-transform skewed targets
            if param in ['I0', 'A']:
                logger.info(f"Applying log transformation to {param}")
                data = np.log1p(data)
                y_processed[:, i] = self.scalers[param].fit_transform(data).ravel()
            elif param == 'f_center':
                y_processed[:, i] = ((data - self.f_center_mean) / self.f_center_mean).ravel()
            elif param in ['width', 'f_delta']:
                y_processed[:, i] = (data / self.f_center_mean).ravel()
        
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

    def remove_outliers(self, df, lower_percentile=1, upper_percentile=99):
        """Remove outliers based on percentiles."""
        for param in self.param_columns:
            lower_bound = df[param].quantile(lower_percentile / 100.0)
            upper_bound = df[param].quantile(upper_percentile / 100.0)
            df = df[(df[param] >= lower_bound) & (df[param] <= upper_bound)]
            logger.info(f"Removed outliers for {param}: [{lower_bound}, {upper_bound}]")
        return df

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

        if param in ['width', 'f_delta']:
            params = {**base_params, 'learning_rate': 0.005, 'max_depth': 4, 'min_child_weight': 3}
        elif param == 'f_center':
            params = {**base_params, 'learning_rate': 0.008, 'max_depth': 8}
        else:
            params = base_params

        return params

    def fit(self, X, y, X_val=None, y_val=None):
        for i, param in enumerate(self.param_columns):
            logger.info(f"Training model for {param}...")
            params = self.create_base_model(param)
            dtrain = xgb.DMatrix(X, label=y[:, i])
            evaluation = {'train': [], 'eval': []}

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val[:, i])
                evallist = [(dtrain, 'train'), (dval, 'eval')]

                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=evallist,
                    early_stopping_rounds=50,
                    verbose_eval=100
                )
            else:
                model = xgb.train(params, dtrain, num_boost_round=100)

            self.models[param] = model
            self.training_history[param] = evaluation
            importance = model.get_score(importance_type='gain')
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"\nTop 10 important features for {param}:")
            for feat, imp in sorted_imp:
                logger.info(f"Feature {feat}: {imp:.4f}")

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        predictions = np.zeros((X.shape[0], len(self.param_columns)))
        for i, param in enumerate(self.param_columns):
            predictions[:, i] = self.models[param].predict(dtest)
        return predictions

def evaluate_predictions(model, X_test, y_test, processor, show_plot=True):
    y_pred = model.predict(X_test)
    y_test_orig = processor.inverse_transform_outputs(y_test)
    y_pred_orig = processor.inverse_transform_outputs(y_pred)

    metrics = {f'{param}_mse': np.mean((y_test_orig[:, i] - y_pred_orig[:, i]) ** 2)
               for i, param in enumerate(processor.param_columns)}

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

        # Add a placeholder for the sixth subplot if unused
        if len(processor.param_columns) < 6:
            axes[-1].axis('off')

        # Show the plot
        plt.tight_layout()
        plt.show()

               
if __name__ == "__main__":
    # Example usage
    logger.info("Starting script...")

    # Load sample data
    if os.path.exists("test.csv"):
        df = pd.read_csv("test.csv")
        processor = ODMRDataProcessor()
        regressor = ODMRRegressor()

        # Preprocess the data
        X_scaled, y_processed = processor.preprocess_data(df)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_processed, test_size=0.2, random_state=42)

        # Train the model
        regressor.fit(X_train, y_train, X_test, y_test)

        # Evaluate the model
        evaluate_predictions(regressor, X_test, y_test, processor)
    else:
        logger.error("Sample data file 'sample_data.csv' not found.")
