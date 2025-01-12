import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import wandb

print(f"wandb version: {wandb.__version__}")
print(f"tensorflow version: {tf.__version__}")


class TrainingConfig:
    def __init__(self,
                 initial_units=1024,
                 second_units=512,
                 dropout_rate=0.2,
                 learning_rate=1e-3,
                 batch_size=128,
                 epochs=20,
                 I0_loss_weight=1.0,
                 A_loss_weight=1.0,
                 width_loss_weight=1.0,
                 f_center_loss_weight=1.0,
                 f_delta_loss_weight=1.0):
        # Network architecture
        self.initial_units = initial_units
        self.second_units = second_units
        self.dropout_rate = dropout_rate

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Loss weights
        self.I0_loss_weight = I0_loss_weight
        self.A_loss_weight = A_loss_weight
        self.width_loss_weight = width_loss_weight
        self.f_center_loss_weight = f_center_loss_weight
        self.f_delta_loss_weight = f_delta_loss_weight

    def __str__(self):
        """Provides a nice string representation of the configuration"""
        return (
            f"\nTraining Configuration:"
            f"\n----------------------"
            f"\nNetwork Architecture:"
            f"\n  Initial units: {self.initial_units}"
            f"\n  Second units: {self.second_units}"
            f"\n  Dropout rate: {self.dropout_rate}"
            f"\n"
            f"\nTraining Parameters:"
            f"\n  Learning rate: {self.learning_rate}"
            f"\n  Batch size: {self.batch_size}"
            f"\n  Epochs: {self.epochs}"
            f"\n"
            f"\nLoss Weights:"
            f"\n  I0: {self.I0_loss_weight}"
            f"\n  A: {self.A_loss_weight}"
            f"\n  Width: {self.width_loss_weight}"
            f"\n  f_center: {self.f_center_loss_weight}"
            f"\n  f_delta: {self.f_delta_loss_weight}"
        )


# Create default configurations for different training scenarios
def get_training_config(config_name='default'):
    """Returns a predefined training configuration.

    This function serves as a central place to define different training configurations.
    You can easily add new configurations or modify existing ones here.
    """
    configs = {
        'default': TrainingConfig(
            # These are the standard parameters that work well for most cases
            initial_units=1024,
            second_units=512,
            dropout_rate=0.2,
            learning_rate=1e-3,
            batch_size=128,
            epochs=20,
            I0_loss_weight=1.0,
            A_loss_weight=1.0,
            width_loss_weight=1.0,
            f_center_loss_weight=1.0,
            f_delta_loss_weight=1.0
        ),

        'focus_on_frequency': TrainingConfig(
            initial_units=2048,
            second_units=1024,
            dropout_rate=0.1,
            learning_rate=1e-5,  # Even slower learning
            batch_size=32,  # Smaller batches
            epochs=25,  # More epochs
            I0_loss_weight=1.0,
            A_loss_weight=1.0,
            width_loss_weight=10.0,
            f_center_loss_weight=10.0,
            f_delta_loss_weight=10.0
        ),

        'quick_test': TrainingConfig(
            # Configuration for quick testing
            initial_units=512,
            second_units=256,
            dropout_rate=0.2,
            learning_rate=1e-3,
            batch_size=128,
            epochs=5,
            I0_loss_weight=1.0,
            A_loss_weight=1.0,
            width_loss_weight=1.0,
            f_center_loss_weight=1.0,
            f_delta_loss_weight=1.0
        )
    }

    return configs.get(config_name, configs['default'])


class ODMRDataProcessor:
    def __init__(self):
        self.input_scaler = StandardScaler()
        self.raw_freq_columns = [f'raw_freq_{i}' for i in range(100)]
        self.param_columns = ['I0', 'A', 'width', 'f_center', 'f_delta']
        self.scalers = {param: StandardScaler() for param in self.param_columns}

    def preprocess_data(self, df):
        X = df[self.raw_freq_columns].values
        y = df[self.param_columns].values
        X_scaled = self.input_scaler.fit_transform(X)
        y_processed = np.zeros_like(y)
        for i, param in enumerate(self.param_columns):
            if param in ['width', 'f_delta']:
                # Don't log transform width and f_delta
                y_processed[:, i] = self.scalers[param].fit_transform(
                    y[:, i].reshape(-1, 1)
                ).ravel()
            else:
                # Log transform other parameters
                y_processed[:, i] = self.scalers[param].fit_transform(
                    np.log1p(y[:, i].reshape(-1, 1))
                ).ravel()
        return X_scaled, y_processed

    def inverse_transform_outputs(self, y_scaled):
        y_orig = np.zeros_like(y_scaled)
        for i, param in enumerate(self.param_columns):
            y_orig[:, i] = np.expm1(
                self.scalers[param].inverse_transform(y_scaled[:, i].reshape(-1, 1))
            ).ravel()
        return y_orig


def parameter_specific_losses(config=None):
    """
    Creates improved loss functions for each parameter with appropriate weighting and constraints.
    """
    def get_weight(param):
        if config is None:
            return 1.0
        return getattr(config, f'{param.lower()}_loss_weight', 1.0)
    
    def custom_loss(param):
        weight = get_weight(param)
        
        def loss(y_true, y_pred):
            # Base MSE loss
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            
            # Relative error with safe denominator
            relative_error = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-7)
            
            # Parameter-specific additional terms
            if param in ['width', 'f_delta']:
                # Ensure positive values and reasonable ranges
                positivity_penalty = tf.reduce_mean(tf.maximum(0.0, -y_pred))
                range_penalty = tf.reduce_mean(tf.maximum(0.0, y_pred - 1.0))
                
                return weight * (mse + 0.1 * relative_error + positivity_penalty + range_penalty)
            
            elif param in ['I0', 'A']:
                # Ensure positive values for intensity parameters
                positivity_penalty = tf.reduce_mean(tf.maximum(0.0, -y_pred))
                return weight * (mse + 0.1 * relative_error + positivity_penalty)
            
            # Default case (f_center)
            return weight * (mse + 0.1 * relative_error)
        
        return loss
    
    return {
        'I0': custom_loss('I0'),
        'A': custom_loss('A'),
        'width': custom_loss('width'),
        'f_center': custom_loss('f_center'),
        'f_delta': custom_loss('f_delta')
    }

def create_improved_model(input_dim=100, config=None):
    """
    Creates an improved ODMR model with multi-scale convolutions and separate branches.
    """
    if config is None:
        config = TrainingConfig()
        
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # 1. Initial spectrum encoding with multi-scale convolutions
    x = tf.keras.layers.Reshape((input_dim, 1))(inputs)
    
    # Multi-scale feature extraction
    conv_3 = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    conv_5 = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    conv_7 = tf.keras.layers.Conv1D(64, kernel_size=7, padding='same', activation='relu')(x)
    
    # Combine features
    x = tf.keras.layers.Concatenate()([conv_3, conv_5, conv_7])
    
    # Additional feature processing
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Global features
    x_avg = tf.keras.layers.GlobalAveragePooling1D()(x)
    x_max = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Concatenate()([x_avg, x_max])
    
    # 2. Separate processing branches
    # Frequency branch (more complex features)
    frequency_branch = tf.keras.layers.Dense(1024, activation='relu')(x)
    frequency_branch = tf.keras.layers.BatchNormalization()(frequency_branch)
    frequency_branch = tf.keras.layers.Dropout(0.2)(frequency_branch)
    frequency_branch = tf.keras.layers.Dense(512, activation='relu')(frequency_branch)
    frequency_branch = tf.keras.layers.BatchNormalization()(frequency_branch)
    frequency_branch = tf.keras.layers.Dense(256, activation='relu')(frequency_branch)
    frequency_branch = tf.keras.layers.BatchNormalization()(frequency_branch)
    
    # Intensity branch (simpler features)
    intensity_branch = tf.keras.layers.Dense(256, activation='relu')(x)
    intensity_branch = tf.keras.layers.BatchNormalization()(intensity_branch)
    intensity_branch = tf.keras.layers.Dense(128, activation='relu')(intensity_branch)
    intensity_branch = tf.keras.layers.BatchNormalization()(intensity_branch)
    
    # 3. Output layers with appropriate constraints
    # Intensity outputs
    i0_output = tf.keras.layers.Dense(1, activation='softplus', name='I0')(intensity_branch)
    a_output = tf.keras.layers.Dense(1, activation='softplus', name='A')(intensity_branch)
    
    # Frequency outputs
    # Width needs to be small but positive
    width_output = tf.keras.layers.Dense(1, activation='sigmoid', name='width')(frequency_branch)
    
    # Center frequency should be in measurement range
    f_center_output = tf.keras.layers.Dense(1, name='f_center')(frequency_branch)
    
    # Frequency separation should be positive
    f_delta_output = tf.keras.layers.Dense(1, activation='sigmoid', name='f_delta')(frequency_branch)
    
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[i0_output, a_output, width_output, f_center_output, f_delta_output]
    )
    
    return model


def train_improved_model(X_train, y_train, X_val, y_val, config=None, epochs=None, use_wandb=False):
    """
    Improved training function with better learning rate scheduling and monitoring.
    """
    model = create_improved_model(X_train.shape[1], config)
    
    # Print model summary
    print("\nModel Architecture:")
    print("-" * 50)
    model.summary(expand_nested=True, show_trainable=True, line_length=80)
    
    # Setup training parameters
    learning_rate = config.learning_rate if config else 1e-3
    batch_size = config.batch_size if config else 128
    epochs = epochs or (config.epochs if config else 20)
    
    # Use parameter specific losses
    losses = parameter_specific_losses(config)  # Changed from improved_loss_functions
    
    # Setup optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0  # Add gradient clipping
    )
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['mae']
    )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    if use_wandb:
        try:
            from wandb.keras import WandbCallback
            callbacks.append(WandbCallback(monitor='val_loss'))
        except ImportError:
            print("Warning: wandb not available")
    
    # Train model
    history = model.fit(
        X_train,
        [y_train[:, i] for i in range(5)],
        validation_data=(X_val, [y_val[:, i] for i in range(5)]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, [history]  # Wrap the history in a list


def evaluate_predictions(model, X_test, y_test, processor, histories=None, show_plot=True):
    y_pred = np.column_stack(model.predict(X_test, verbose=0))
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

        # Plot loss history in the sixth subplot
        if histories:
            train_losses = []
            val_losses = []
            for history in histories:
                train_losses.extend(history.history['loss'])
                val_losses.extend(history.history['val_loss'])

            epochs = range(1, len(train_losses) + 1)
            axes[5].plot(epochs, train_losses, 'b-', label='Training Loss')
            axes[5].plot(epochs, val_losses, 'r-', label='Validation Loss')
            axes[5].set_title('Model Loss')
            axes[5].set_xlabel('Epoch')
            axes[5].set_ylabel('Loss')
            axes[5].legend()
            axes[5].grid(True)

            # Add vertical lines to separate stages
            stage_length = len(histories[0].history['loss'])
            for stage in range(1, 3):
                axes[5].axvline(x=stage * stage_length, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

    metrics = {f'{param}_mse': np.mean((y_test_orig[:, i] - y_pred_orig[:, i]) ** 2)
               for i, param in enumerate(processor.param_columns)}

    return y_pred_orig, metrics


def train_model_wandb(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config

        # Print current configuration
        print("\nCurrent sweep configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

        df = pd.read_csv('test.csv')
        processor = ODMRDataProcessor()
        X_scaled, y_scaled = processor.preprocess_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Create TrainingConfig object from wandb config
        training_config = TrainingConfig(
            initial_units=config.initial_units,
            second_units=config.second_units,
            dropout_rate=config.dropout_rate,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            I0_loss_weight=config.I0_loss_weight,
            A_loss_weight=config.A_loss_weight,
            width_loss_weight=config.width_loss_weight,
            f_center_loss_weight=config.f_center_loss_weight,
            f_delta_loss_weight=config.f_delta_loss_weight
        )

        model, histories = train_improved_model(
            X_train, y_train, X_val, y_val,
            config=training_config,
            use_wandb=True
        )
        # Evaluate and log metrics
        _, metrics = evaluate_predictions(model, X_test, y_test, processor, show_plot=False)
        wandb.log(metrics)


sweep_config = {
    'method': 'bayes',  # Using Bayesian optimization for parameter search
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'initial_units': {
            'values': [512, 1024, 2048]
        },
        'second_units': {
            'values': [256, 512, 1024]
        },
        'dropout_rate': {
            'values': [0.1, 0.2, 0.3]
        },
        'learning_rate': {
            'values': [1e-3, 3e-4, 1e-4]
        },
        'batch_size': {
            'values': [64, 128, 256]
        },
        'epochs': {'value': 15},  # Fixed value for all runs
        'I0_loss_weight': {
            'values': [0.5, 1.0, 2.0]
        },
        'A_loss_weight': {
            'values': [0.5, 1.0, 2.0]
        },
        'width_loss_weight': {
            'values': [1.0, 2.0, 4.0]
        },
        'f_center_loss_weight': {
            'values': [1.0, 2.0, 4.0]
        },
        'f_delta_loss_weight': {
            'values': [1.0, 2.0, 4.0]
        }
    }
}


def regular_training(config):
    df = pd.read_csv('test.csv')
    processor = ODMRDataProcessor()
    X_scaled, y_scaled = processor.preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    model, histories = train_improved_model(
        X_train, y_train, X_val, y_val,
        epochs=config.epochs,
        config=config
    )

    y_pred, _ = evaluate_predictions(model, X_test, y_test, processor, histories=histories, show_plot=True)
    return model, processor, y_pred


def main():
    parser = argparse.ArgumentParser(description='ODMR Model Training')
    parser.add_argument('--mode', type=str, default='regular',
                        choices=['regular', 'wandb'],
                        help='Training mode: regular or wandb sweep')
    args = parser.parse_args()

    if args.mode == 'wandb':
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, project="odmr-fitting")

        # Specify number of runs to try different configurations
        wandb.agent(sweep_id, train_model_wandb, count=5)  # Will run 20 different configurations
        return None, None, None
    else:
        config = get_training_config('focus_on_frequency')
        print(config)
        return regular_training(config)


if __name__ == "__main__":
    model, processor, predictions = main()