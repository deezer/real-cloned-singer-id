from training.parameters import CHECKPOINT_PATH, MODEL_CONFIG

# Data sets and loaders parameters
BATCH_SIZE = 100
LEARNING_RATE = 0.01
STEP_SIZE = 10
FACTOR = 0.1
EARLY_STOP = 20
BACKBONE_WEIGHTS_PATH = f"{CHECKPOINT_PATH}/best_epoch.pt"
STORAGE_PATH = f"/workspace/eval_data/{MODEL_CONFIG}"
PICKLES_PATH = f"{STORAGE_PATH}/pickles"
MTG_PICKLES_PATH = f"{STORAGE_PATH}/mtg_pickles"
FMA_PICKLES_PATH = f"{STORAGE_PATH}/fma_pickles"

# Number of artists experiment
LOWER_BOUND = 100
UPPER_BOUND = 1000
STEP = 100
RUNS = 10

# We cannot display cloned artist IDs unfortunately...
CLONED_DATASET_ARTISTS = []
CLONED_DATASET_DIR = "data/clonedDataset/"