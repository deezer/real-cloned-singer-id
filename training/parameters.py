# Model name
MODEL_CONFIG = "vocal2vocal"

# Learning parameters
LEARNING_RATE = 0.0001
STEP_SIZE = 25
FACTOR = 0.5
EARLY_STOP = 100

# Model parameters
FEATURE_DIM = 2048
HIDDEN_DIM = 1024
PROJ_DIM = 2048

# Data sets and loaders parameters
BATCH_SIZE = 128
TRAIN_STEPS = 32
VAL_STEPS = 32
NUM_WORKERS = 32
CHECKPOINT_PATH = f"/workspace/checkpoint/{MODEL_CONFIG}_{FEATURE_DIM}_{BATCH_SIZE}_{LEARNING_RATE}"

# Loss parameters
TEMPERATURE = 0.2
EPSILON = 1e-9
