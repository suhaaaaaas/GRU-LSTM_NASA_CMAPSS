import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, GRU, Dense, Dropout, BatchNormalization, 
    Bidirectional, Concatenate, GlobalAveragePooling1D, 
    LayerNormalization, Conv1D, Add, Attention, TimeDistributed
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention mechanism to focus on important time steps."""
    
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        score = tf.nn.tanh(self.W(inputs))  # (batch_size, seq_len, units)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch_size, seq_len, 1)
        context_vector = attention_weights * inputs  # (batch_size, seq_len, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch_size, features)
        return context_vector, attention_weights

def residual_block(x, units, dropout_rate=0.3):
    """Residual block with skip connection for better gradient flow."""
    # Store input for residual connection
    residual = x
    
    # Apply GRU with LayerNormalization and Dropout
    gru_output = Bidirectional(GRU(units, return_sequences=True))(x)
    normalized = LayerNormalization()(gru_output)
    dropped = Dropout(dropout_rate)(normalized)
    
    # If input and output dimensions don't match, use a projection
    if residual.shape[-1] != dropped.shape[-1]:
        residual = Conv1D(dropped.shape[-1], kernel_size=1, padding='same')(residual)
    
    # Add residual connection
    output = Add()([dropped, residual])
    return output

def build_refined_model(sequence_length, num_features, group_features=0):
    """
    Build an enhanced RUL prediction model with attention, residual connections,
    and other advanced techniques.
    
    Args:
        sequence_length: Length of the input time sequence
        num_features: Number of sensor/input features
        group_features: Number of engine group features (default 0)
        
    Returns:
        Compiled keras model
    """
    # Main sequence input - time series data
    sequence_input = Input(shape=(sequence_length, num_features - group_features), name='sequence_input')
    
    # Engine group input (if used)
    if group_features > 0:
        group_input = Input(shape=(sequence_length, group_features), name='group_input')
        # Merge inputs
        merged_input = Concatenate(axis=2)([sequence_input, group_input])
    else:
        merged_input = sequence_input
    
    # 1D Convolutional layer for feature extraction
    conv = Conv1D(
        filters=64, 
        kernel_size=3, 
        padding='same', 
        activation='relu', 
        kernel_regularizer=l2(0.0005)
    )(merged_input)
    
    # First residual block
    res1 = residual_block(conv, units=64, dropout_rate=0.2)
    
    # Second residual block
    res2 = residual_block(res1, units=96, dropout_rate=0.2)
    
    # Bidirectional GRU with increased units for more capacity
    bi_gru = Bidirectional(
        GRU(128, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.0005))
    )(res2)
    
    # Apply batch normalization
    normalized = BatchNormalization()(bi_gru)
    
    # Attention mechanism to focus on important time steps
    attention = AttentionLayer(128)
    context_vector, attention_weights = attention(normalized)
    
    # Additional fully connected layers with regularization
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(context_vector)
    dropout1 = Dropout(0.3)(dense1)
    batch_norm1 = BatchNormalization()(dropout1)
    
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(batch_norm1)
    dropout2 = Dropout(0.2)(dense2)
    batch_norm2 = BatchNormalization()(dropout2)
    
    # Output layer
    output = Dense(1, name='rul_output')(batch_norm2)
    
    # Define the model
    if group_features > 0:
        model = Model(inputs=[sequence_input, group_input], outputs=output)
    else:
        model = Model(inputs=sequence_input, outputs=output)
    
    # AdamW optimizer with learning rate schedule and gradient clipping
    optimizer = AdamW(
        learning_rate=0.001,
        weight_decay=0.0001,
        clipnorm=1.0  # Gradient clipping
    )
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model

def apply_prediction_smoothing(predictions, window_size=15):
    """
    Apply smoothing to model predictions to reduce oscillations.
    
    Args:
        predictions: Numpy array of raw model predictions
        window_size: Size of the moving average window
        
    Returns:
        Smoothed predictions
    """
    import numpy as np
    
    # Apply moving average smoothing
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(predictions.flatten(), kernel, mode='same')
    
    # Handle edge effects
    # For the first and last window_size//2 elements, gradually reduce the kernel size
    for i in range(window_size // 2):
        left_kernel_size = 2 * i + 1
        smoothed[i] = np.sum(predictions.flatten()[0:left_kernel_size]) / left_kernel_size
        
        right_kernel_size = 2 * i + 1
        smoothed[-(i+1)] = np.sum(predictions.flatten()[-(right_kernel_size):]) / right_kernel_size
    
    return smoothed