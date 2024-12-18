import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class ODMRDataProcessor:
    def __init__(self):
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.raw_freq_columns = [f'raw_freq_{i}' for i in range(100)]
        self.param_columns = ['I0', 'A', 'width', 'f_center', 'f_delta']
        
    def prepare_data(self, df):
        """
        Prepare ODMR data for training by separating and scaling inputs/outputs
        """
        # Extract raw frequency data (features)
        X = df[self.raw_freq_columns].values
        
        # Extract Lorentzian parameters (targets)
        y = df[self.param_columns].values
        
        # Fit and transform the scalers
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)
        
        return X_scaled, y_scaled
    
    def inverse_transform_outputs(self, y_scaled):
        """
        Transform scaled outputs back to original scale
        """
        return self.output_scaler.inverse_transform(y_scaled)

def create_lorentzian_surrogate_model(input_dim=100):
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # Deeper feature extraction
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Residual blocks
    def residual_block(x, units):
        skip = x
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if skip.shape[-1] == units:
            x = tf.keras.layers.Add()([x, skip])
        return x
    
    x = residual_block(x, 256)
    x = residual_block(x, 128)
    x = residual_block(x, 64)
    
    # Shared high-level features
    shared = tf.keras.layers.Dense(64, activation='relu')(x)
    shared = tf.keras.layers.BatchNormalization()(shared)
    
    # Parameter-specific branches with positive outputs
    outputs = []
    for param_name in ['I0', 'A', 'width', 'f_center', 'f_delta']:
        branch = tf.keras.layers.Dense(32, activation='relu')(shared)
        branch = tf.keras.layers.BatchNormalization()(branch)
        output = tf.keras.layers.Dense(1, activation='softplus', name=param_name)(branch)
        outputs.append(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def custom_loss():
    """
    Custom loss function that can handle different scales of parameters
    """
    def loss(y_true, y_pred):
        # MSE loss
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        # Add relative error component
        relative_error = tf.abs((y_true - y_pred) / (y_true + 1e-7))
        
        return mse + 0.1 * relative_error
    return loss

def plot_training_history(history):
    """
    Plot training and validation loss over epochs
    """
    plt.figure(figsize=(12, 8))
    
    # Plot total loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot individual parameter losses
    plt.subplot(2, 1, 2)
    for param in ['I0', 'A', 'width', 'f_center', 'f_delta']:
        plt.plot(history.history[f'{param}_loss'], label=f'{param} Loss')
    plt.title('Individual Parameter Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_predictions(y_true, y_pred, processor):
    """
    Evaluate and visualize prediction accuracy for each parameter
    """
    # Convert scaled values back to original scale
    y_true_orig = processor.inverse_transform_outputs(y_true)
    y_pred_orig = processor.inverse_transform_outputs(y_pred)
    
    # Create scatter plots for each parameter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(processor.param_columns):
        axes[i].scatter(y_true_orig[:, i], y_pred_orig[:, i], alpha=0.5)
        axes[i].plot([y_true_orig[:, i].min(), y_true_orig[:, i].max()], 
                    [y_true_orig[:, i].min(), y_true_orig[:, i].max()], 
                    'r--')
        axes[i].set_title(f'{param} Predictions')
        axes[i].set_xlabel('True Values')
        axes[i].set_ylabel('Predicted Values')
        
        # Calculate and display metrics
        mse = np.mean((y_true_orig[:, i] - y_pred_orig[:, i])**2)
        r2 = np.corrcoef(y_true_orig[:, i], y_pred_orig[:, i])[0, 1]**2
        axes[i].text(0.05, 0.95, f'MSE: {mse:.2e}\nR²: {r2:.3f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def train_model(X_train, y_train, X_val, y_val, epochs=100):
    model = create_lorentzian_surrogate_model()
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=[custom_loss() for _ in range(5)],
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        X_train,
        [y_train[:, i] for i in range(5)],
        validation_data=(X_val, [y_val[:, i] for i in range(5)]),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history

def main():
    # Load your data
    df = pd.read_csv('test.csv')  # Replace with your data file path
    
    # Initialize data processor
    processor = ODMRDataProcessor()
    
    # Prepare data
    X_scaled, y_scaled = processor.prepare_data(df)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions on validation set
    y_pred_scaled = np.column_stack(model.predict(X_val))
    
    # Evaluate predictions
    evaluate_predictions(y_val, y_pred_scaled, processor)
    
    # Function to make predictions on new data
    def predict_parameters(X_new):
        X_new_scaled = processor.input_scaler.transform(X_new)
        y_pred_scaled = model.predict(X_new_scaled)
        y_pred_scaled = np.column_stack(y_pred_scaled)
        return processor.inverse_transform_outputs(y_pred_scaled)
    
    return model, predict_parameters, processor

if __name__ == "__main__":
    model, predict_params, processor = main()