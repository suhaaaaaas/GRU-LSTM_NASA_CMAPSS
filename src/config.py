import random

# Global Constants
SEQ_LEN = 100
BATCH_SIZE = 32
EPOCHS = 25
FEATURE_COLUMNS = [str(i) for i in range(10)] + \
                  [f"ewma_{i}" for i in range(10)]
TARGET_COLUMN = 'RUL'
CYCLE_COLUMN = 'cycle'

# Random seed for reproducibility
random.seed(42)
