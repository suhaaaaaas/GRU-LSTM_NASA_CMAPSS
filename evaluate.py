from tensorflow.keras.models import load_model
from src.train import train_model
from src.utils import plot_history
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load the saved model
model = load_model('results/model_checkpoint/model.keras')

# Load and preprocess test data
#_, test_df = load_and_split_data('/Users/suhaasnadella/Documents/vscode/ML/Pw Data with RUL.csv')  # Assuming this splits train/test

model, history, X_test, y_test, target_scaler = train_model('file path here')
plot_history(history, save_path='results/training_history.png')


predictions = model.predict(X_test)

#test_df.to_csv("test.csv")
#_.to_csv("_.csv")
#_, test_df, feature_scaler, target_scaler = scale_data(_, test_df)

# Select 5 random engines for testing (with seed for reproducibility)
#unique_engines = test_df['esn'].unique()
#selected_engines = random.sample(list(unique_engines), 5)
#test_subset = test_df[test_df['esn'].isin(selected_engines)]

# Create test sequences for the selected engines
#test_sequences = preprocess_engine_data(test_df)
#X_test = np.array([seq for seq, _ in test_sequences])
#y_test = np.array([rul for _, rul in test_sequences])

# Generate predictions
#predictions = model.predict(X_test)

# Unscale predictions and y_test
y_test_unscaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
predictions_unscaled = target_scaler.inverse_transform(predictions).flatten()

# Plot actual vs predicted RUL for the selected engines
plt.figure(figsize=(12, 8))
plt.plot(y_test_unscaled, label='Actual RUL', linestyle='-', marker='o')
plt.plot(predictions_unscaled, label='Predicted RUL', linestyle='-', marker='x')
plt.xlabel('Sample Index')
plt.ylabel('Remaining Useful Life (RUL)')
plt.title('Actual vs Predicted RUL for Selected Engines')
plt.legend()
plt.grid()
plt.savefig('results/predictions_subset.png')
plt.show()

# Print evaluation metrics
mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Square Error (RMSE): {rmse}")
