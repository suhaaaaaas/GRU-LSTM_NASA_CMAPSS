import os
import matplotlib.pyplot as plt
from src.train_refined import train_refined_model, evaluate_refined_model

def main():
    """
    Train and evaluate the refined RUL prediction model with engine groups.
    All parameters are preset with optimal values.
    """
    # Configuration (hardcoded with optimal values)
    data_path = 'Pw Data with RUL.csv'
    num_groups = 10
    seq_len = 50
    batch_size = 32
    epochs = 10

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/model_checkpoint', exist_ok=True)
    
    # Train the model with engine groups (always enabled)
    print("\n===== Training refined model with engine groups =====")
    
    model, history, X_test, y_test, target_scaler, _ = train_refined_model(
        data_path, 
        use_engine_groups=True,  # Always use engine groups
        num_groups=num_groups,
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Evaluate the model
    mae, rmse, plot_path = evaluate_refined_model(
        model, 
        X_test, 
        y_test, 
        target_scaler,
        use_engine_groups=True  # Always use engine groups
    )
    
    print(f"\nModel evaluation complete. Results saved to {plot_path}")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

if __name__ == "__main__":
    main()