import tensorflow as tf

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

        'focus_on_width_and_frequency': TrainingConfig(
            # Configuration that emphasizes width and frequency predictions
            initial_units=2048,
            second_units=1024,
            dropout_rate=0.3,
            learning_rate=3e-4,
            batch_size=256,
            epochs=10,
            I0_loss_weight=0.5,
            A_loss_weight=0.5,
            width_loss_weight=2.0,
            f_center_loss_weight=2.0,
            f_delta_loss_weight=2.0
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
    def get_weight(param):
        if config is None:
            return 1.0
        return getattr(config, f'{param.lower()}_loss_weight', 1.0)

    def combined_loss(param):
        weight = get_weight(param)

        def loss(y_true, y_pred):
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            relative_error = tf.abs((y_true - y_pred) / (y_true + 1e-7))
            return weight * (mse + 0.1 * relative_error)

        return loss

    return {
        'I0': combined_loss('I0'),
        'A': combined_loss('A'),
        'width': combined_loss('width'),
        'f_center': combined_loss('f_center'),
        'f_delta': combined_loss('f_delta')
    }


def create_tuned_model(input_dim=100, config=None):
    if config is None:
        class DefaultConfig:
            def __init__(self):
                self.initial_units = 1024
                self.second_units = 512
                self.dropout_rate = 0.2

        config = DefaultConfig()

    inputs = tf.keras.Input(shape=(input_dim,))

    x = tf.keras.layers.Dense(config.initial_units, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(config.dropout_rate)(x)

    skip1 = x
    x = tf.keras.layers.Dense(config.initial_units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(config.initial_units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, skip1])

    skip2 = x
    x = tf.keras.layers.Dense(config.second_units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(config.second_units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, tf.keras.layers.Dense(config.second_units)(skip2)])

    outputs = []
    param_branches = {
        'I0': [256],
        'A': [256],
        'width': [512, 256],
        'f_center': [512, 256, 128],
        'f_delta': [512, 256, 128]
    }

    for param, layers in param_branches.items():
        branch = x
        for units in layers:
            branch = tf.keras.layers.Dense(units, activation='relu')(branch)
            branch = tf.keras.layers.BatchNormalization()(branch)
        output = tf.keras.layers.Dense(1, name=param)(branch)
        outputs.append(output)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def train_improved_model(X_train, y_train, X_val, y_val, epochs=20, use_wandb=False, config=None):
    model = create_tuned_model(config=config)

    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    print(f"\nTotal parameters: {trainable_params + non_trainable_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    losses = parameter_specific_losses(config)
    learning_rate = config.learning_rate if config else 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['mae']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_weights=True,
            verbose=0
        )
    ]

    if use_wandb:
        callbacks.append(
            wandb.keras.callbacks.WandbCallback(
                monitor='val_loss',
                save_model=False,
                log_weights=False
            )
        )

    print("\nTraining Progress:", flush=True)
    histories = []
    for stage in range(3):
        print(f"\nStage {stage + 1}/3:")
        history = model.fit(
            X_train,
            [y_train[:, i] for i in range(5)],
            validation_data=(X_val, [y_val[:, i] for i in range(5)]),
            epochs=epochs,
            batch_size=config.batch_size if config else 128,
            callbacks=callbacks,
            verbose=1  # Changed from 0 to 1 to show progress bar
        )
        histories.append(history)

        val_loss = history.history['val_loss'][-1]
        print(f"Stage {stage + 1} completed - Final val_loss: {val_loss:.4f}")

        current_lr = float(optimizer.learning_rate)
        optimizer.learning_rate = current_lr * 0.1
        print(f"Learning rate reduced to: {current_lr * 0.1:.2e}")

    return model, histories


def evaluate_predictions(model, X_test, y_test, processor):
    y_pred = np.column_stack(model.predict(X_test, verbose=0))
    y_test_orig = processor.inverse_transform_outputs(y_test)
    y_pred_orig = processor.inverse_transform_outputs(y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics = {}
    for i, param in enumerate(processor.param_columns):
        axes[i].scatter(y_test_orig[:, i], y_pred_orig[:, i], alpha=0.5)

        min_val = min(y_test_orig[:, i].min(), y_pred_orig[:, i].min())
        max_val = max(y_test_orig[:, i].max(), y_pred_orig[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')

        mse = np.mean((y_test_orig[:, i] - y_pred_orig[:, i]) ** 2)
        r2 = np.corrcoef(y_test_orig[:, i], y_pred_orig[:, i])[0, 1] ** 2

        metrics[f'{param}_mse'] = mse
        metrics[f'{param}_r2'] = r2

        axes[i].set_title(f'{param} Predictions')
        axes[i].set_xlabel('True Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].text(0.05, 0.95, f'MSE: {mse:.2e}\nR²: {r2:.3f}',
                     transform=axes[i].transAxes,
                     verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return y_pred_orig, metrics


def train_model_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config

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
            use_wandb=True,
            config=config
        )

        _, metrics = evaluate_predictions(model, X_test, y_test, processor)
        wandb.log(metrics)


sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'initial_units': {'values': [512, 1024, 2048]},
        'second_units': {'values': [256, 512, 1024]},
        'dropout_rate': {'values': [0.1, 0.2, 0.3]},
        'learning_rate': {'values': [1e-3, 3e-4, 1e-4]},
        'batch_size': {'values': [64, 128, 256]},
        'epochs': {'value': 20},
        'I0_loss_weight': {'values': [0.5, 1.0, 2.0]},
        'A_loss_weight': {'values': [0.5, 1.0, 2.0]},
        'width_loss_weight': {'values': [1.0, 2.0, 4.0]},
        'f_center_loss_weight': {'values': [1.0, 2.0, 4.0]},
        'f_delta_loss_weight': {'values': [1.0, 2.0, 4.0]}
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

    # Pass config.epochs instead of the entire config object
    model, histories = train_improved_model(
        X_train, y_train, X_val, y_val,
        epochs=config.epochs,  # Changed from epochs=epochs
        config=config  # Add this to pass the full config
    )
    y_pred, _ = evaluate_predictions(model, X_test, y_test, processor)

    return model, processor, y_pred


def main():
    parser = argparse.ArgumentParser(description='ODMR Model Training')
    parser.add_argument('--mode', type=str, default='regular',
                        choices=['regular', 'wandb'],
                        help='Training mode: regular or wandb sweep')
    args = parser.parse_args()

    if args.mode == 'wandb':
        # WandB sweep configuration remains unchanged
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, project="odmr-fitting",
                               entity="diederikdekker-tu-delft")
        wandb.agent(sweep_id, train_model_wandb, count=20)
        return None, None, None
    else:
        # Choose your configuration here
        # config = get_training_config('default')
        config = get_training_config('focus_on_width_and_frequency')
        # config = get_training_config('quick_test')
        print(config)  # Print the configuration before training
        return regular_training(config)


if __name__ == "__main__":
    model, processor, predictions = main()