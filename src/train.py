import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from src.data_preprocessing import load_and_split_data, scale_data
from src.engine_groups import preprocess_with_engine_groups, create_training_data_with_groups
from src.utils import create_training_data, plot_history
from src.model import build_refined_model, apply_prediction_smoothing

def prepare_refined_data(file_path, use_engine_groups=True, num_groups=10, seq_len=50):
    """
    Prepare data for the refined model, with optional engine groups.
    
    Args:
        file_path: Path to the CSV data file
        use_engine_groups: Whether to use engine grouping
        num_groups: Number of engine groups to create
        seq_len: Sequence length for time series
        
    Returns:
        Processed training and testing data, scalers
    """
    if use_engine_groups:
        # Process with engine groups
        train_df, test_df, feature_scaler, target_scaler, group_cols = preprocess_with_engine_groups(
            file_path, num_groups
        )
        
        # Create training data with group features
        X_train, y_train, X_test, y_test = create_training_data_with_groups(
            train_df, test_df, group_cols, seq_len
        )
        
        # Split features and group data for the refined model's inputs
        feature_cols_count = len(group_cols)
        
        # Extract group columns as separate inputs
        X_train_features = X_train[:, :, :-feature_cols_count]
        X_train_groups = X_train[:, :, -feature_cols_count:]
        
        X_test_features = X_test[:, :, :-feature_cols_count]
        X_test_groups = X_test[:, :, -feature_cols_count:]
        
        return (X_train_features, X_train_groups), y_train, (X_test_features, X_test_groups), y_test, feature_scaler, target_scaler, feature_cols_count
    else:
        # Process without engine groups
        train_df, test_df = load_and_split_data(file_path)
        train_df, test_df, feature_scaler, target_scaler = scale_data(train_df, test_df)
        
        # Create training data
        X_train, y_train, X_test, y_test = create_training_data(train_df, test_df)
        
        return X_train, y_train, X_test, y_test, feature_scaler, target_scaler, 0

def train_refined_model(file_path, use_engine_groups=True, num_groups=10, seq_len=50, batch_size=32, epochs=10):
    """
    Train the refined RUL prediction model.
    
    Args:
        file_path: Path to the CSV data file
        use_engine_groups: Whether to use engine grouping
        num_groups: Number of engine groups to create
        seq_len: Sequence length for time series
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        
    Returns:
        Trained model, training history, test data, and scalers
    """
    # Ensure directories exist
    os.makedirs('results/model_checkpoint', exist_ok=True)
    
    # Prepare data
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler, group_features = prepare_refined_data(
        file_path, use_engine_groups, num_groups, seq_len
    )
    
    # Determine model input shape
    if use_engine_groups:
        num_features = X_train[0].shape[2] + X_train[1].shape[2]
    else:
        num_features = X_train.shape[2]
    
    # Build model
    model = build_refined_model(seq_len, num_features, group_features if use_engine_groups else 0)
    
    # Print model summary
    model.summary()
    
    # Train-validation split
    if use_engine_groups:
        X_train_features, X_val_features, X_train_groups, X_val_groups, y_train_split, y_val = train_test_split(
            X_train[0], X_train[1], y_train, test_size=0.2, random_state=42
        )
        X_train_split = [X_train_features, X_train_groups]
        X_val = [X_val_features, X_val_groups]
    else:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # Define callbacks
    model_name = 'refined_model_with_groups.keras' if use_engine_groups else 'refined_model.keras'
    checkpoint_callback = ModelCheckpoint(
        f'results/model_checkpoint/{model_name}',
        save_best_only=True,
        monitor='val_loss'
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        update_freq='epoch'
    )
    
    # Train model
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            lr_scheduler,
            tensorboard_callback
        ],
        verbose=1
    )
    
    # Plot training history
    history_path = 'results/refined_training_history.png'
    if use_engine_groups:
        history_path = 'results/refined_training_history_with_groups.png'
    
    plot_history(history, save_path=history_path)
    
    return model, history, X_test, y_test, target_scaler, use_engine_groups

def evaluate_refined_model(model, X_test, y_test, target_scaler, use_engine_groups=True):
    """
    Evaluate the refined model and visualize results.
    
    Args:
        model: Trained refined model
        X_test: Test features
        y_test: Test targets
        target_scaler: Scaler for the target values
        use_engine_groups: Whether engine groups were used
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Unscale predictions and y_test
    y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_unscaled = target_scaler.inverse_transform(predictions).flatten()
    
    # Apply smoothing to reduce oscillations
    predictions_smoothed = apply_prediction_smoothing(predictions_unscaled, window_size=15)
    
    # Calculate metrics with both raw and smoothed predictions
    mae_raw = mean_absolute_error(y_test_unscaled, predictions_unscaled)
    rmse_raw = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
    
    mae_smoothed = mean_absolute_error(y_test_unscaled, predictions_smoothed)
    rmse_smoothed = np.sqrt(mean_squared_error(y_test_unscaled, predictions_smoothed))
    
    # Print evaluation metrics
    print("\n--- Evaluation Metrics ---")
    print(f"Raw Predictions:")
    print(f"  Mean Absolute Error (MAE): {mae_raw:.2f}")
    print(f"  Root Mean Square Error (RMSE): {rmse_raw:.2f}")
    print(f"Smoothed Predictions:")
    print(f"  Mean Absolute Error (MAE): {mae_smoothed:.2f}")
    print(f"  Root Mean Square Error (RMSE): {rmse_smoothed:.2f}")
    print("-------------------------\n")
    
    # Plot actual vs predicted RUL
    plt.figure(figsize=(14, 8))
    
    # Plot actual RUL
    plt.plot(y_test_unscaled, label='Actual RUL', color='blue', linewidth=2)
    
    # Plot raw predictions with low alpha
    plt.plot(
        predictions_unscaled, 
        label='Raw Predictions', 
        color='orange', 
        alpha=0.4, 
        linewidth=1
    )
    
    # Plot smoothed predictions
    plt.plot(
        predictions_smoothed, 
        label='Smoothed Predictions', 
        color='red', 
        linewidth=1.5
    )
    
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Remaining Useful Life (RUL)', fontsize=12)
    plt.title('Refined Model: Actual vs Predicted RUL', fontsize=14)
    if use_engine_groups:
        plt.title('Refined Model with Engine Groups: Actual vs Predicted RUL', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add performance metrics to the plot
    plt.figtext(
        0.15, 0.02, 
        f"Smoothed MAE: {mae_smoothed:.2f} | RMSE: {rmse_smoothed:.2f}", 
        ha="left", 
        fontsize=12, 
        bbox={"facecolor":"white", "alpha":0.8, "pad":5}
    )
    
    # Save the figure
    plot_path = 'results/refined_predictions.png'
    if use_engine_groups:
        plot_path = 'results/refined_predictions_with_groups.png'
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    return mae_smoothed, rmse_smoothed, plot_path