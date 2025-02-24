import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEQ_LEN = 100
BATCH_SIZE = 32
EPOCHS = 25
FEATURE_COLUMNS = [str(i) for i in range(10)] + \
                  [f"ewma_{i}" for i in range(10)]
TARGET_COLUMN = 'RUL'
CYCLE_COLUMN = 'cycle'

def plot_history(history, save_path=None):
    import os
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()



def preprocess_engine_data(df, seq_len=SEQ_LEN):
    processed_data = []
    
    # Unique engine identifier
    df['engine_id'] = df['esn'].astype(str) + '_' + df['UERID'].astype(str)
    grouped = df.groupby('engine_id')

    for engine, data in grouped:
        data = data.sort_values(by=CYCLE_COLUMN).reset_index(drop=True)

        for i in range(len(data)):
            sequence = data.iloc[max(0, i - (seq_len - 1)): i + 1][FEATURE_COLUMNS].values

            # Pad sequences shorter than seq_len
            if sequence.shape[0] < seq_len:
                padding = np.zeros((seq_len - sequence.shape[0], len(FEATURE_COLUMNS)))
                sequence = np.vstack((padding, sequence))

            rul = data.iloc[i][TARGET_COLUMN]
            processed_data.append((sequence, rul))

    return processed_data

def create_training_data(train_df, test_df):
    train_sequences = preprocess_engine_data(train_df, SEQ_LEN)
    test_sequences = preprocess_engine_data(test_df, SEQ_LEN)

    X_train = np.array([seq for seq, _ in train_sequences])
    y_train = np.array([rul for _, rul in train_sequences])
    X_test = np.array([seq for seq, _ in test_sequences])
    y_test = np.array([rul for _, rul in test_sequences])

    return X_train, y_train, X_test, y_test
