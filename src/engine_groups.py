import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os
from src.config import SEQ_LEN, FEATURE_COLUMNS, TARGET_COLUMN, CYCLE_COLUMN

def add_engine_group_feature(data_path, num_groups=10):
    """
    Assigns group IDs to engines based on their initial RUL values and adds
    this as a feature to the dataset.
    
    Args:
        data_path: Path to the CSV file containing engine data
        num_groups: Number of groups to create (default: 10)
        
    Returns:
        DataFrame with added group_id column
    """
    # Load the data
    data = pd.read_csv(data_path)
    
    # Create unique engine identifier 
    data['engine_id'] = data['esn'].astype(str) + '_' + data['UERID'].astype(str)
    
    # Get initial RUL for each engine
    engine_initial_ruls = {}
    
    for engine_id, group in data.groupby('engine_id'):
        # Find the minimum cycle (starting point)
        min_cycle_idx = group['Unnamed: 0'].idxmin()
        initial_rul = group.loc[min_cycle_idx, 'RUL']
        engine_initial_ruls[engine_id] = initial_rul

    # Convert to DataFrame for easier analysis
    initial_rul_df = pd.DataFrame([
        {'engine_id': engine_id, 'initial_rul': rul} 
        for engine_id, rul in engine_initial_ruls.items()
    ])
    
    # Calculate bin edges
    min_rul = initial_rul_df['initial_rul'].min()
    max_rul = initial_rul_df['initial_rul'].max()
    bin_size = (max_rul - min_rul) / num_groups
    
    # Create bin edges
    bin_edges = [min_rul + i * bin_size for i in range(num_groups + 1)]
    
    # Assign group IDs
    initial_rul_df['group_id'] = pd.cut(
        initial_rul_df['initial_rul'], 
        bins=bin_edges, 
        labels=range(1, num_groups + 1), 
        include_lowest=True
    )
    
    # Plot the distribution of engines across groups
    plt.figure(figsize=(12, 6))
    plt.hist(initial_rul_df['initial_rul'], bins=bin_edges, edgecolor='black')
    plt.title('Distribution of Initial RULs Across Engine Groups')
    plt.xlabel('Initial RUL')
    plt.ylabel('Number of Engines')
    plt.xticks(bin_edges, rotation=45)
    plt.grid(alpha=0.3)
    
    for i, edge in enumerate(bin_edges[:-1]):
        plt.text(
            (edge + bin_edges[i+1])/2, 
            1, 
            f'Group {i+1}', 
            ha='center', 
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    # Ensure the directory exists
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/engine_group_distribution.png')
    plt.close()
    
    # Merge group_id back to the original data
    group_mapping = initial_rul_df[['engine_id', 'group_id']]
    data = data.merge(group_mapping, on='engine_id', how='left')
    
    # Convert to categorical feature
    data['group_id'] = data['group_id'].astype(int)
    
    # Log some statistics about the groups
    print(f"Engine grouping statistics:")
    group_stats = initial_rul_df.groupby('group_id')['initial_rul'].agg(['min', 'max', 'count'])
    print(group_stats)
    
    return data

def preprocess_with_engine_groups(file_path, num_groups=10):
    """
    Enhanced preprocessing function that includes engine group features.
    
    Args:
        file_path: Path to the data file
        num_groups: Number of engine groups to create
        
    Returns:
        Enhanced train and test dataframes with group features
    """
    from src.data_preprocessing import load_and_split_data, scale_data
    
    # First, add engine group features
    data_with_groups = add_engine_group_feature(file_path, num_groups)
    
    # Save the enhanced data to a temporary file
    temp_file = 'temp_data_with_groups.csv'
    data_with_groups.to_csv(temp_file, index=False)
    
    # Load and split the data using the existing function
    train_df, test_df = load_and_split_data(temp_file)
    
    # One-hot encode the group_id feature
    encoder = OneHotEncoder(sparse_output=False)
    
    # Fit the encoder on train data to avoid data leakage
    group_encoded_train = encoder.fit_transform(train_df[['group_id']])
    group_encoded_test = encoder.transform(test_df[['group_id']])
    
    # Create column names for encoded features
    group_cols = [f'group_{i}' for i in range(1, num_groups + 1)]
    
    # Add encoded group features to dataframes
    for i, col in enumerate(group_cols):
        train_df[col] = group_encoded_train[:, i]
        test_df[col] = group_encoded_test[:, i]
    
    # Scale the data using the existing function
    train_df, test_df, feature_scaler, target_scaler = scale_data(train_df, test_df)
    
    # Clean up temporary file
    import os
    os.remove(temp_file)
    
    return train_df, test_df, feature_scaler, target_scaler, group_cols

def preprocess_engine_data_with_groups(df, feature_columns, seq_len=SEQ_LEN):
    """
    Modified version of preprocess_engine_data that handles enhanced feature columns.
    
    Args:
        df: DataFrame containing engine data
        feature_columns: List of feature columns including group features
        seq_len: Sequence length for time series
        
    Returns:
        List of (sequence, target) pairs
    """
    processed_data = []
    
    # Unique engine identifier (already created in earlier steps)
    if 'engine_id' not in df.columns:
        df['engine_id'] = df['esn'].astype(str) + '_' + df['UERID'].astype(str)
        
    grouped = df.groupby('engine_id')

    for engine, data in grouped:
        data = data.sort_values(by=CYCLE_COLUMN).reset_index(drop=True)

        for i in range(len(data)):
            sequence = data.iloc[max(0, i - (seq_len - 1)): i + 1][feature_columns].values

            # Pad sequences shorter than seq_len
            if sequence.shape[0] < seq_len:
                padding = np.zeros((seq_len - sequence.shape[0], len(feature_columns)))
                sequence = np.vstack((padding, sequence))

            rul = data.iloc[i][TARGET_COLUMN]
            processed_data.append((sequence, rul))

    return processed_data

def create_training_data_with_groups(train_df, test_df, group_cols, seq_len=SEQ_LEN):
    """
    Enhanced version of create_training_data that includes group features.
    
    Args:
        train_df: Training dataframe with group features
        test_df: Testing dataframe with group features
        group_cols: List of group feature columns
        seq_len: Sequence length for time series
        
    Returns:
        Training and testing data with group features included
    """
    # Enhanced feature columns including group features
    enhanced_features = FEATURE_COLUMNS + group_cols
    
    # Preprocess data with engine groups
    processed_train = preprocess_engine_data_with_groups(train_df, enhanced_features, seq_len)
    processed_test = preprocess_engine_data_with_groups(test_df, enhanced_features, seq_len)
    
    X_train = np.array([seq for seq, _ in processed_train])
    y_train = np.array([rul for _, rul in processed_train])
    X_test = np.array([seq for seq, _ in processed_test])
    y_test = np.array([rul for _, rul in processed_test])
    
    return X_train, y_train, X_test, y_test

def train_model_with_groups(file_path, num_groups=10):
    """
    Enhanced training function that incorporates engine group features.
    
    Args:
        file_path: Path to the data file
        num_groups: Number of engine groups
        
    Returns:
        Trained model, history, and evaluation data
    """
    from src.model import build_model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from src.config import BATCH_SIZE, EPOCHS
    
    # Ensure model save directory exists
    os.makedirs('results/model_checkpoint', exist_ok=True)
    
    # Preprocess data with engine groups
    train_df, test_df, feature_scaler, target_scaler, group_cols = preprocess_with_engine_groups(file_path, num_groups)
    
    # Create training data with enhanced features
    X_train, y_train, X_test, y_test = create_training_data_with_groups(train_df, test_df, group_cols, SEQ_LEN)
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Build model with enhanced features (more input features now)
    num_features = X_train.shape[2]  # This will include the group features
    model = build_model(SEQ_LEN, num_features)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint('results/model_checkpoint/model_with_groups.keras', save_best_only=True)
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
    
    return model, history, X_test, y_test, target_scaler, group_cols