import tensorflow as tf
import numpy as np

def create_lorentzian_surrogate_model(input_dim=100):
    # Input layer for the frequency sweep
    inputs = tf.keras.Input(shape=(input_dim,))
    
    # First, let's create a feature extraction block that can identify key spectral characteristics
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Add residual connections to help preserve spectral information
    def residual_block(x, units):
        skip = x
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Add skip connection if dimensions match
        if skip.shape[-1] == units:
            x = tf.keras.layers.Add()([x, skip])
        return x
    
    # Multiple residual blocks for deep feature extraction
    x = residual_block(x, 128)
    x = residual_block(x, 64)
    
    # Split the network into parameter-specific branches
    # This allows the network to specialize in predicting each parameter
    
    # Branch for I0 (baseline intensity)
    i0_branch = tf.keras.layers.Dense(32, activation='relu')(x)
    i0_output = tf.keras.layers.Dense(1, activation='linear', name='i0')(i0_branch)
    
    # Branch for amplitude
    amp_branch = tf.keras.layers.Dense(32, activation='relu')(x)
    amp_output = tf.keras.layers.Dense(1, activation='linear', name='amplitude')(amp_branch)
    
    # Branch for width
    width_branch = tf.keras.layers.Dense(32, activation='relu')(x)
    width_output = tf.keras.layers.Dense(1, activation='positive', name='width')(width_branch)
    
    # Branch for center frequency
    fcenter_branch = tf.keras.layers.Dense(32, activation='relu')(x)
    fcenter_output = tf.keras.layers.Dense(1, activation='linear', name='fcenter')(fcenter_branch)
    
    # Branch for frequency delta
    fdelta_branch = tf.keras.layers.Dense(32, activation='relu')(x)
    fdelta_output = tf.keras.layers.Dense(1, activation='linear', name='fdelta')(fdelta_branch)
    
    # Combine all outputs
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[i0_output, amp_output, width_output, fcenter_output, fdelta_output]
    )
    
    return model

# Custom loss function that incorporates physics-based constraints
def physics_informed_loss(y_true, y_pred):
    # Standard MSE loss
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Add physics-based regularization
    # For example, ensure width is positive, amplitude is reasonable, etc.
    physics_penalty = tf.maximum(0.0, -y_pred[2])  # Penalty for negative width
    
    return mse_loss + 0.1 * physics_penalty

# Custom learning rate scheduler
def create_lr_schedule():
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )

# Training function with validation
def train_model(model, train_data, train_labels, val_data, val_labels, epochs=100):
    lr_schedule = create_lr_schedule()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'i0': 'mse',
            'amplitude': 'mse',
            'width': 'mse',
            'fcenter': 'mse',
            'fdelta': 'mse'
        },
        metrics=['mae']  # Mean Absolute Error for interpretable results
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    return history