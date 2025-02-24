import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import os

SEQ_LEN = 100
BATCH_SIZE = 32
EPOCHS = 25
FEATURE_COLUMNS = [str(i) for i in range(10)] + \
                  [f"ewma_{i}" for i in range(10)]
TARGET_COLUMN = 'RUL'
CYCLE_COLUMN = 'cycle'

def load_and_split_data(file_path):
    data = pd.read_csv(file_path)

    # Ensure RUL column exists
    columns_to_drop = ["indicator"]  # Remove indicator
    data = data.drop(columns=columns_to_drop)

    if 'RUL' not in data.columns:
        rul = data.groupby('esn')['Unnamed: 0'].max().reset_index()
        rul.columns = ['esn', 'max_cycle']
        data = data.merge(rul, on='esn', how='left')
        data['RUL'] = data['max_cycle'] - data['Unnamed: 0']
        data.drop(columns=['max_cycle'], inplace=True)

    sensor_columns = [str(i) for i in range(10)]  # Sensor columns: "0" to "9"
    for sensor in sensor_columns:
        data[f"ewma_{sensor}"] = data.groupby("esn")[sensor].transform(lambda x: x.ewm(span=50).mean())

    # Split Train and Test Data
    lifecycles = data[['esn', 'UERID']].drop_duplicates()
    random_lifecycles = random.sample(lifecycles.values.tolist(), 5)

    test_mask = data[['esn', 'UERID']].apply(lambda x: x.tolist() in random_lifecycles, axis=1)
    train_data = data[~test_mask]
    test_data = data[test_mask]

    # Rename cycle column
    train_data = train_data.rename(columns={'Unnamed: 0': 'cycle'})
    test_data = test_data.rename(columns={'Unnamed: 0': 'cycle'})

    # Create '../data' directory if it does not exist
    output_dir = os.path.join(os.path.dirname(file_path), '../data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    return train_data, test_data

def scale_data(train_df, test_df):
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_columns = [str(i) for i in range(10)]

    train_df[feature_columns] = feature_scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = feature_scaler.transform(test_df[feature_columns])

    train_df[TARGET_COLUMN] = target_scaler.fit_transform(train_df[[TARGET_COLUMN]])
    test_df[TARGET_COLUMN] = target_scaler.transform(test_df[[TARGET_COLUMN]])

    # Data Augmentation: Add Gaussian noise to training features
    train_df[feature_columns] += np.random.normal(0, 0.005, train_df[feature_columns].shape)  # Reduce noise


    return train_df, test_df, feature_scaler, target_scaler

