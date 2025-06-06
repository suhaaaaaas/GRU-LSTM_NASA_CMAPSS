# RUL Prediction with GRU RNN

This project predicts the Remaining Useful Life (RUL) of NASA CMAPPS using a GRU-based RNN.

## File Structure
- `data/`: Contains raw data files.
- `src/`: Core code files.
- `results/`: Stores plots and model checkpoints.
- `evaluate.py`: Main script to run the pipeline (trains and tests).

## Setup
1. Install dependencies
2. Copy data path to load_and_split_data function in evaluate.py
3. Run evaluate.py

## Final Model
<img width="255" alt="image" src="https://github.com/user-attachments/assets/bb054280-fdfa-40d1-bd5f-dfc8ebad008f" />
1. Pass input sequence through an LSTM to produce vector of hidden states, H
2. Each h ∈ H is scored via a trainable projection (vᵀ · tanh(W·h + b))
3. Normalize weights using Softmax then aggregate via weighted sum
4. Dropout layer to regularize, then final dense layer produces a scalar output
Idea: learn which moments in the sequence matter most by scoring and weighting each hidden state, then aggregate them into concise context vector for prediction.

## Results
<img width="486" alt="image" src="https://github.com/user-attachments/assets/ea48d20a-11ef-4ece-acaa-70b2846a07de" />


