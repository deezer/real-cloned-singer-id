import gc
import json
import os
import pickle

import pandas as pd

from evaluation.parameters import (
    BATCH_SIZE,
    EARLY_STOP,
    FACTOR,
    LEARNING_RATE,
    LOWER_BOUND,
    RUNS,
    STEP,
    STEP_SIZE,
    STORAGE_PATH,
    UPPER_BOUND,
)
from evaluation.train import Trainer
from training.parameters import EPSILON, FEATURE_DIM, HIDDEN_DIM, NUM_WORKERS

if __name__ == "__main__":
    print("Loading .json files...")
    with open("data/test_set.json", "r+") as f:
        test_set = json.load(f)

    if os.path.isfile(f"{STORAGE_PATH}/full_results.pickle"):
        with open(f"{STORAGE_PATH}/full_results.pickle", "rb") as handle:
            full_results = pickle.load(handle)

        with open(f"{STORAGE_PATH}/lower_bound.pickle", "rb") as handle:
            lower_bound = pickle.load(handle)
            lower_bound += STEP

    else:
        lower_bound = LOWER_BOUND
        full_results = []

    for num_labels in range(lower_bound, UPPER_BOUND + 1, STEP):
        results = []

        print(f"\nNumber of classes: {num_labels}")

        for run in range(0, RUNS, 1):
            print(f"\nRun: {run}")

            trainer = Trainer(
                test_set=test_set,
                num_classes=num_labels,
                feature_dim=FEATURE_DIM,
                hidden_dim=HIDDEN_DIM,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                learning_rate=LEARNING_RATE,
                factor=FACTOR,
                step_size=STEP_SIZE,
                early_stop=EARLY_STOP,
                epsilon=EPSILON,
            )
            trainer.train()
            _, _, run_results = trainer.test(run)

            results += run_results
            gc.collect()

        full_results += results

        df = pd.DataFrame(
            full_results,
            columns=[
                "num_classes",
                "trial",
                "artist_id",
                "artist_name",
                "path",
                "macro_genre",
                "lang",
                "num_train_tracks",
                "predicted_artist_id",
                "predicted_artist_name",
                "is_top1",
                "is_top5",
            ],
        )
        df.to_excel(f"{STORAGE_PATH}/all_results.xlsx", index=False)

        with open(f"{STORAGE_PATH}/full_results.pickle", "wb") as handle:
            pickle.dump(full_results, handle, pickle.HIGHEST_PROTOCOL)

        with open(f"{STORAGE_PATH}/lower_bound.pickle", "wb") as handle:
            pickle.dump(num_labels, handle, pickle.HIGHEST_PROTOCOL)
