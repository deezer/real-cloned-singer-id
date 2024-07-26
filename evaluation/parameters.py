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
CLONED_DATASET_ARTISTS = [
    "cloned_artist_id_1",
    "cloned_artist_id_2",
    "cloned_artist_id_3",
    "cloned_artist_id_4",
    "cloned_artist_id_5",
    "cloned_artist_id_6",
    "cloned_artist_id_7",
    "cloned_artist_id_8",
    "cloned_artist_id_9",
    "cloned_artist_id_10",
    "cloned_artist_id_11",
    "cloned_artist_id_12",
    "cloned_artist_id_13",
    "cloned_artist_id_14",
    "cloned_artist_id_15",
    "cloned_artist_id_16",
    "cloned_artist_id_17",
    "cloned_artist_id_18",
    "cloned_artist_id_19",
    "cloned_artist_id_20",
    "cloned_artist_id_21",
    "cloned_artist_id_22",
    "cloned_artist_id_23",
    "cloned_artist_id_24",
    "cloned_artist_id_25",
    "cloned_artist_id_26",
    "cloned_artist_id_27",
    "cloned_artist_id_28",
    "cloned_artist_id_29",
    "cloned_artist_id_30",
    "cloned_artist_id_31",
    "cloned_artist_id_32",
    "cloned_artist_id_33",
    "cloned_artist_id_34",
    "cloned_artist_id_35",
    "cloned_artist_id_36",
    "cloned_artist_id_37",
    "cloned_artist_id_38",
    "cloned_artist_id_39",
    "cloned_artist_id_40",
    "cloned_artist_id_41",
    "cloned_artist_id_42",
    "cloned_artist_id_43",
    "cloned_artist_id_44",
    "cloned_artist_id_45",
    "cloned_artist_id_46",
    "cloned_artist_id_47",
    "cloned_artist_id_48",
    "cloned_artist_id_49",
    "cloned_artist_id_50",
    "cloned_artist_id_51",
    "cloned_artist_id_52",
    "cloned_artist_id_53",
    "cloned_artist_id_54",
    "cloned_artist_id_55",
    "cloned_artist_id_56",
    "cloned_artist_id_57",
    "cloned_artist_id_58",
    "cloned_artist_id_59",
    "cloned_artist_id_60",
    "cloned_artist_id_61",
    "cloned_artist_id_62",
    "cloned_artist_id_63",
    "cloned_artist_id_64",
    "cloned_artist_id_65",
    "cloned_artist_id_66",
    "cloned_artist_id_67",
]
CLONED_DATASET_DIR = "eval_data/clonedDataset/"
