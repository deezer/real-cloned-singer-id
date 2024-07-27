import numpy as np

from evaluation.parameters import (
    BATCH_SIZE,
    EARLY_STOP,
    FACTOR,
    LEARNING_RATE,
    STEP_SIZE,
)
from evaluation.train import Trainer
from training.parameters import EPSILON, FEATURE_DIM, HIDDEN_DIM, NUM_WORKERS

if __name__ == "__main__":
    top1_accs = []
    top5_accs = []

    for trial in range(10):
        trainer = Trainer(
            test_set={},
            num_classes=1019,
            feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            learning_rate=LEARNING_RATE,
            factor=FACTOR,
            step_size=STEP_SIZE,
            early_stop=EARLY_STOP,
            epsilon=EPSILON,
            results="fma",
        )
        trainer.train()
        top1_test, top5_test = trainer.test_fma_mtg()
        top1_accs.append(top1_test)
        top5_accs.append(top5_test)

    print(
        "\nMean test top-1 accuracy: {:.3f} +/- {:.3f}\nMean test top-5 accuracy: {:.3f} +/- {:.3f}".format(
            float(np.mean(top1_accs)),
            float(np.std(top1_accs)),
            float(np.mean(top5_accs)),
            float(np.std(top5_accs)),
        )
    )
