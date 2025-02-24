import os
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data_preprocessing import load_and_split_data, scale_data
from src.utils import create_training_data
from src.model import build_model

SEQ_LEN = 100
BATCH_SIZE = 32
EPOCHS = 25
FEATURE_COLUMNS = [str(i) for i in range(10)] + \
                  [f"ewma_{i}" for i in range(10)]
TARGET_COLUMN = 'RUL'
CYCLE_COLUMN = 'cycle'

# Ensure model save directory exists
if not os.path.exists('results/model_checkpoint'):
    os.makedirs('results/model_checkpoint')

def train_model(file_path):
    # Load and preprocess data
    train_df, test_df = load_and_split_data(file_path)
    train_df, test_df, feature_scaler, target_scaler = scale_data(train_df, test_df)

    X_train, y_train, X_test, y_test = create_training_data(train_df, test_df)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build model
    model = build_model(SEQ_LEN, X_train.shape[2])

    # Define callbacks
    checkpoint_callback = ModelCheckpoint('results/model_checkpoint/model.keras', save_best_only=True)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    

    # Train model with callbacks
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler]
    )

    return model, history, X_test, y_test, target_scaler
