from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2

def build_model(sequence_length, num_features):
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        GRU(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        GRU(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(1)
    ])
    # Add gradient clipping by setting clipnorm or clipvalue
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)  # Clip gradients by their L2 norm
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model
