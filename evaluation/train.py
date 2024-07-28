import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchmetrics
from torchinfo import summary
from tqdm import tqdm, trange

from evaluation.dataset import (
    ClonedDataset,
    FMATestDataset,
    FMATrainDataset,
    FMAValDataset,
    MTGTestDataset,
    MTGTrainDataset,
    MTGValDataset,
    TestDataset,
    TrainDataset,
    ValDataset,
)
from evaluation.model import EvaluationWrapper
from evaluation.parameters import (
    BACKBONE_WEIGHTS_PATH,
    BATCH_SIZE,
    CLONED_DATASET_ARTISTS,
    EARLY_STOP,
    FACTOR,
    LEARNING_RATE,
    STEP_SIZE,
    STORAGE_PATH,
)
from foundation.constants import N_FRAMES, N_MELS
from foundation.model import ASTModel
from training.parameters import EPSILON, FEATURE_DIM, HIDDEN_DIM, NUM_WORKERS


class Trainer(object):
    """
    Object for training and evaluation a classifier upon voice embeddings.
    """

    def __init__(
        self,
        test_set: Dict[str, Any],
        num_classes: int,
        feature_dim: int,
        hidden_dim: int,
        batch_size: int,
        num_workers: int,
        learning_rate: float,
        factor: float,
        step_size: int,
        early_stop: int,
        epsilon: float,
        results: str = "cloned",
    ) -> None:
        """
        Args:
            test_set (Dict[str, Any]):
                Test set dictionary.
            num_classes (int):
                Number of artists to classify.
            feature_dim (int):
                Embedding feature dimension.
            hidden_dim (int):
                Hidden dimension.
            batch_size (int):
                Batch size.
            num_workers (int):
                Number of workers for torch multiprocessing.
            learning_rate (float):
                Learning rate.
            factor (float):
                Factor at which learning rate is decreased.
            step_size (int):
                Number of steps after which learning rate decreases.
            early_stop (int):
                Number of non-diminishing loss epochs before stopping training.
            epsilon (float):
                Epsilon ADAM optimizer.
            results (str):
                Run cloned, vanilla, fma, or mtg dataset results?
        """

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.num_workers = num_workers
        self.early_stop = early_stop
        self.train_steps = self.num_workers
        self.cuda_available = torch.cuda.is_available()

        # Artist IDs used for training the classifier and random class
        if results == "cloned":
            if num_classes < len(CLONED_DATASET_ARTISTS):
                raise ValueError(
                    f"Number of classes to evaluate cloned dataset must be superior to {len(CLONED_DATASET_ARTISTS)}!"
                )
            else:
                artist_set = set(CLONED_DATASET_ARTISTS)

                while len(artist_set) < num_classes:
                    artist_set.add(random.choice(list(test_set.keys())))

                artist_ids = list(artist_set)
                self.clonedDataset = ClonedDataset(test_set, artist_ids)

            # Import feature model
            self.feature_model = ASTModel(
                n_mels=N_MELS,
                n_frames=N_FRAMES,
                in_channels=2,
                feature_dim=FEATURE_DIM,
                model_size="small224",
            )
            pretrained_weights = torch.load(BACKBONE_WEIGHTS_PATH)["model_state_dict"]
            filtered_weights = {
                k.replace("feature_model.", ""): v for k, v in pretrained_weights.items() if "feature_model" in k
            }
            self.feature_model.load_state_dict(filtered_weights)

            if self.cuda_available:
                self.feature_model = self.feature_model.cuda()

        elif results == "vanilla":
            artist_ids = random.sample(list(test_set.keys()), k=self.num_classes)

        # Declare datasets
        if results == "vanilla" or results == "cloned":
            trainDataset: Any = TrainDataset(test_set, artist_ids)
            valDataset: Any = ValDataset(test_set, artist_ids)
            self.testDataset: Any = TestDataset(test_set, artist_ids)
        elif results == "fma":
            trainDataset = FMATrainDataset()
            valDataset = FMAValDataset(trainDataset.artist_ids)
            self.testDataset = FMATestDataset(trainDataset.artist_ids)
        elif results == "mtg":
            trainDataset = MTGTrainDataset()
            valDataset = MTGValDataset(trainDataset.artist_ids)
            self.testDataset = MTGTestDataset(trainDataset.artist_ids)
        else:
            raise NotImplementedError("Only fma, mtg, vanilla, or cloned for now.")

        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            valDataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.testDataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )
        if results == "cloned":
            self.cloned_loader = torch.utils.data.DataLoader(
                self.clonedDataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.num_workers,
            )
        self.cache()

        # Create model and load pre-trained backbone weights
        self.model = EvaluationWrapper(
            feature_dim,
            hidden_dim,
            self.num_classes,
        )
        summary(self.model.eval(), input_size=(self.batch_size, feature_dim))

        # Declare torch modules
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.top1_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes,
            top_k=1,
            average=None,
        )
        self.top5_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=self.num_classes,
            top_k=5,
            average=None,
        )
        self.optimizer = torch.optim.Adam(  # type: ignore
            self.model.classifier.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=self.epsilon,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=factor,
            patience=step_size,
            min_lr=learning_rate * 0.001,
            eps=epsilon,
        )
        summary(
            self.criterion,
            input_size=(
                (self.batch_size, self.num_classes),
                (self.batch_size, self.num_classes),
            ),
        )

        # Pass objects to GPU
        if self.cuda_available:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            self.softmax = self.softmax.cuda()
            self.top1_accuracy = self.top1_accuracy.cuda()
            self.top5_accuracy = self.top5_accuracy.cuda()

        # Initialize variables
        self.initialize()

    def train(self) -> None:
        """
        Train model!
        """
        self.train_iterator = iter(self.train_loader)

        while self.plateau < self.early_stop:
            print("\n--- Epoch {} ---".format(self.epoch))

            train_loss_mean, train_top1_acc_mean, train_top5_acc_mean = self.train_epoch()
            val_mean, val_top1_acc_mean, val_top5_acc_mean = self.val_epoch()

            print(
                "training   set - loss: {:.3f}, top-1 accuracy: {:.3f}, top-5 accuracy: {:.3f}".format(
                    train_loss_mean, train_top1_acc_mean, train_top5_acc_mean
                )
            )
            print(
                "validation set - loss: {:.3f}, top-1 accuracy: {:.3f}, top-5 accuracy: {:.3f}".format(
                    val_mean, val_top1_acc_mean, val_top5_acc_mean
                )
            )

            self.save_epoch(val_mean)
            self.scheduler.step(val_mean)

            self.epoch += 1
            gc.collect()

        return

    def test(self, trial: int = 0) -> Tuple[float, float, List[List[Any]]]:
        """
        Test model!
        """
        # Testing...
        best_weights = torch.load(f"{STORAGE_PATH}/weights.pt")["model_state_dict"]
        self.model.load_state_dict(best_weights)

        print("\nTest set:")
        test_top1_acc, test_top5_acc, metrics_for_csv = self.test_epoch(trial)

        print("Top-1 accuracy: {:.3f}\nTop-5 accuracy: {:.3f}".format(test_top1_acc, test_top5_acc))

        return test_top1_acc, test_top5_acc, metrics_for_csv

    def test_fma_mtg(self) -> Tuple[float, float]:
        """
        Test model!
        """
        # Testing...
        best_weights = torch.load(f"{STORAGE_PATH}/weights.pt")["model_state_dict"]
        self.model.load_state_dict(best_weights)

        print("\nTest set:")
        test_top1_acc, test_top5_acc = self.fma_mtg_test_epoch()

        print("Top-1 accuracy: {:.3f}\nTop-5 accuracy: {:.3f}".format(test_top1_acc, test_top5_acc))

        return test_top1_acc, test_top5_acc

    def train_epoch(self) -> Tuple[float, float, float]:
        """
        One epoch of training :)
        """
        self.model.train()

        train_loss = []
        targets = None
        preds = None

        for _ in trange(self.train_steps):
            # Fetch anchors and positives
            embedding, tag = next(self.train_iterator)

            # Move to GPU if available
            if self.cuda_available:
                embedding, tag = embedding.cuda(), tag.cuda()

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(embedding)

            # Loss computation and backward pass
            loss = self.criterion(output, tag)
            loss.backward()

            # Optimize
            self.optimizer.step()

            # Monitor metrics
            loss_value = loss.item()
            train_loss.append(loss_value)

            softmax_classes = torch.exp(self.softmax(output.detach()))
            tag_class = torch.argmax(tag, dim=-1)

            if preds is None:
                preds = softmax_classes
            else:
                preds = torch.cat([preds, softmax_classes], dim=0)

            if targets is None:
                targets = tag_class
            else:
                targets = torch.cat([targets, tag_class], dim=0)

        top1_accuracy = self.top1_accuracy(preds, targets).mean()
        top5_accuracy = self.top5_accuracy(preds, targets).mean()

        return np.mean(train_loss), top1_accuracy, top5_accuracy

    def val_epoch(self) -> Tuple[float, float, float]:
        """
        One epoch of validation :)
        """
        self.model.eval()

        val_loss = []
        targets = None
        preds = None

        with torch.inference_mode():
            for embedding, tag in tqdm(self.val_loader):
                # Move to GPU if available
                if self.cuda_available:
                    embedding, tag = embedding.cuda(), tag.cuda()

                # Forward pass
                output = self.model(embedding)

                # Loss computation and backward pass
                loss = self.criterion(output, tag)

                # Monitor metrics
                val_loss.append(loss.item())

                softmax_classes = torch.exp(self.softmax(output.detach()))
                tag_class = torch.argmax(tag, dim=-1)

                if preds is None:
                    preds = softmax_classes
                else:
                    preds = torch.cat([preds, softmax_classes], dim=0)

                if targets is None:
                    targets = tag_class
                else:
                    targets = torch.cat([targets, tag_class], dim=0)

            top1_accuracy = self.top1_accuracy(preds, targets).mean()
            top5_accuracy = self.top5_accuracy(preds, targets).mean()

        return np.mean(val_loss), top1_accuracy, top5_accuracy

    def cloned_evaluation(self) -> Tuple[float, float, List[List[Any]]]:
        """
        One epoch of cloned dataset testing :)
        """
        self.model.eval()
        self.feature_model.eval()

        metrics_for_csv = []
        top1_results = []
        top5_results = []

        print("\nCloned dataset:")

        with torch.inference_mode():
            for audio_batches, tag, metadata in tqdm(self.cloned_loader):
                # Move to GPU if available
                if self.cuda_available:
                    audio_batches, tag = audio_batches.cuda(), tag.cuda()

                # Forward pass
                embedding = self.feature_model(audio_batches.squeeze(0))
                output = self.model(embedding)

                # Gather outputs
                softmax_classes = torch.exp(self.softmax(output.detach()))
                softmax_classes = torch.argmax(softmax_classes, dim=-1)
                softmax_classes = torch.nn.functional.one_hot(softmax_classes, num_classes=self.num_classes).float()
                softmax_classes = softmax_classes.sum(dim=0).unsqueeze(0)
                softmax_classes /= softmax_classes.sum()

                # Top-1 accuracy
                predicted_index = int(torch.argmax(softmax_classes, dim=-1).item())
                predicted_artist_id = self.clonedDataset.artist_ids[predicted_index]
                predicted_artist_name = self.clonedDataset.data_dict[predicted_artist_id]["name"]

                if tag[:, predicted_index] > 0.5:
                    is_top1 = 1.0
                else:
                    is_top1 = 0.0

                # Top-5 accuracy
                _, top5_predictions = torch.topk(softmax_classes, 5)
                top5_list = top5_predictions.squeeze(0).tolist()

                is_top5 = 0.0
                for p in top5_list:
                    if tag[:, p] > 0.5:
                        is_top5 = 1.0

                csv_result = [
                    self.num_classes,
                    trial,
                    ",".join(metadata["artist_names"][0]),
                    predicted_artist_name,
                    ",".join(metadata["artist_ids"][0]),
                    predicted_artist_id,
                    ",".join([str(x) for x in metadata["artist_indices"][0].tolist()]),
                    predicted_index,
                    is_top1,
                    is_top5,
                    metadata["file_id"][0],
                    metadata["num_train_tracks"][0].item(),
                ]
                metrics_for_csv.append(csv_result)

                top1_results.append(is_top1)
                top5_results.append(is_top5)

                gc.collect()

        cloned_top1_acc, cloned_top5_acc = float(np.mean(top1_results)), float(np.mean(top5_results))
        print("Top-1 accuracy: {:.3f}\nTop-5 accuracy: {:.3f}".format(cloned_top1_acc, cloned_top5_acc))

        return cloned_top1_acc, cloned_top5_acc, metrics_for_csv

    def test_epoch(self, trial: int) -> Tuple[float, float, List[List[Any]]]:
        """
        One epoch of testing :)
        """
        self.model.eval()

        metrics_for_csv = []
        targets = []
        preds = None

        with torch.inference_mode():
            for embedding, tag, metadata in tqdm(self.test_loader):
                # Move to GPU if available
                if self.cuda_available:
                    embedding, tag = embedding.cuda(), tag.cuda()

                # Forward pass
                output = self.model(embedding.squeeze(0))

                # Gather outputs
                softmax_classes = torch.exp(self.softmax(output.detach()))
                softmax_classes = torch.argmax(softmax_classes, dim=-1)
                softmax_classes = torch.nn.functional.one_hot(softmax_classes, num_classes=self.num_classes).float()
                softmax_classes = softmax_classes.sum(dim=0).unsqueeze(0)
                softmax_classes /= softmax_classes.sum()
                tag_argmax = torch.argmax(tag, dim=-1)
                tag_class = tag_argmax.item()

                # Stuff for monitoring
                if preds is None:
                    preds = softmax_classes
                else:
                    preds = torch.cat([preds, softmax_classes], dim=0)

                targets.append(tag_class)
                _, predicted_artist_id = self.testDataset.artist_ids[int(torch.argmax(softmax_classes, dim=-1).item())]
                predicted_artist_name = self.testDataset.data_dict[predicted_artist_id]["name"]

                # Stuff for CSV
                is_top1 = self.top1_accuracy(softmax_classes, tag_argmax)[tag_class].int().item()
                is_top5 = self.top5_accuracy(softmax_classes, tag_argmax)[tag_class].int().item()

                csv_result = [
                    self.num_classes,
                    trial,
                    metadata["artist_id"][0],
                    metadata["artist_name"][0],
                    metadata["path"][0],
                    metadata["macro_genre"][0],
                    metadata["lang"][0],
                    metadata["num_train_tracks"].item(),
                    predicted_artist_id,
                    predicted_artist_name,
                    is_top1,
                    is_top5,
                ]
                metrics_for_csv.append(csv_result)
                gc.collect()

            # Monitor metrics
            targets_tensor = torch.tensor(targets)

            if self.cuda_available:
                targets_tensor = targets_tensor.cuda()

            top1_list_accuracy = self.top1_accuracy(preds, targets_tensor)
            top5_list_accuracy = self.top5_accuracy(preds, targets_tensor)

        return top1_list_accuracy.mean().item(), top5_list_accuracy.mean().item(), metrics_for_csv

    def fma_mtg_test_epoch(self) -> Tuple[float, float]:
        """
        One epoch of testing :)
        """
        self.model.eval()

        targets = []
        preds = None

        with torch.inference_mode():
            for embedding, tag in tqdm(self.test_loader):
                # Move to GPU if available
                if self.cuda_available:
                    embedding, tag = embedding.cuda(), tag.cuda()

                # Forward pass
                output = self.model(embedding.squeeze(0))

                # Gather outputs
                softmax_classes = torch.exp(self.softmax(output.detach()))
                softmax_classes = torch.argmax(softmax_classes, dim=-1)
                softmax_classes = torch.nn.functional.one_hot(softmax_classes, num_classes=self.num_classes).float()
                softmax_classes = softmax_classes.sum(dim=0).unsqueeze(0)
                softmax_classes /= softmax_classes.sum()
                tag_argmax = torch.argmax(tag, dim=-1)
                tag_class = tag_argmax.item()

                # Stuff for monitoring
                if preds is None:
                    preds = softmax_classes
                else:
                    preds = torch.cat([preds, softmax_classes], dim=0)

                targets.append(tag_class)

            # Monitor metrics
            targets_tensor = torch.tensor(targets)

            if self.cuda_available:
                targets_tensor = targets_tensor.cuda()

            top1_list_accuracy = self.top1_accuracy(preds, targets_tensor)
            top5_list_accuracy = self.top5_accuracy(preds, targets_tensor)

        return top1_list_accuracy.mean().item(), top5_list_accuracy.mean().item()

    def save_epoch(self, val_loss: float) -> None:
        """
        Save best and last epoch training objects.
        """
        if val_loss < self.best_loss_value:
            self.best_loss_value: float = val_loss

            self.plateau = 0

            best_epoch: Dict[str, Any] = {
                "model_state_dict": self.model.state_dict(),
                "best_loss_value": self.best_loss_value,
            }
            torch.save(best_epoch, f"{STORAGE_PATH}/weights.pt")
            print("...Saved best model checkpoint.")

        else:
            self.plateau += 1
            print(f"...Plateau set at {self.plateau}.")

        return

    def initialize(self) -> None:
        """
        Initialize variables for training.
        """
        if not os.path.isdir(STORAGE_PATH):
            Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.best_loss_value = float("+inf")
        self.plateau = 0

        return

    def cache(self) -> None:
        """
        Cache and save validation and test datasets.
        """

        print("\nCaching validation set...")

        for embedding, tag in tqdm(self.val_loader):
            if self.cuda_available:
                embedding, tag = embedding.cuda(), tag.cuda()

        self.val_loader.dataset.set_use_cache(use_cache=True)  # type: ignore
        self.val_loader.dataset.num_workers = self.num_workers  # type: ignore

        return


if __name__ == "__main__":
    print("Loading .json files...")
    with open("data/test_set.json", "r+") as f:
        test_set = json.load(f)

    test_top1_accs = []
    test_top5_accs = []
    cloned_top1_accs = []
    cloned_top5_accs = []

    csv_results = []

    for num_classes in range(100, 1100, 100):
        for trial in range(10):
            trainer = Trainer(
                test_set=test_set,
                num_classes=num_classes,
                feature_dim=FEATURE_DIM,
                hidden_dim=HIDDEN_DIM,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                learning_rate=LEARNING_RATE,
                factor=FACTOR,
                step_size=STEP_SIZE,
                early_stop=EARLY_STOP,
                epsilon=EPSILON,
                results="cloned",
            )
            trainer.train()
            top1_test, top5_test, _ = trainer.test(trial)
            test_top1_accs.append(top1_test)
            test_top5_accs.append(top5_test)

            top1_cloned, top5_cloned, metrics_for_csv = trainer.cloned_evaluation()
            cloned_top1_accs.append(top1_cloned)
            cloned_top5_accs.append(top5_cloned)

            csv_results += metrics_for_csv

        print(
            "\nMean test top-1 accuracy: {:.3f} +/- {:.3f}\nMean test top-5 accuracy: {:.3f} +/- {:.3f}".format(
                float(np.mean(test_top1_accs)),
                float(np.std(test_top1_accs)),
                float(np.mean(test_top5_accs)),
                float(np.std(test_top5_accs)),
            )
        )

        print(
            "Mean cloned top-1 accuracy: {:.3f} +/- {:.3f}\nMean cloned top-5 accuracy: {:.3f} +/- {:.3f}".format(
                float(np.mean(cloned_top1_accs)),
                float(np.std(cloned_top1_accs)),
                float(np.mean(cloned_top5_accs)),
                float(np.std(cloned_top5_accs)),
            )
        )
        df = pd.DataFrame(
            csv_results,
            columns=[
                "num_classes",
                "trial",
                "artist_names",
                "predicted_artist_name",
                "artist_ids",
                "predicted_artist_id",
                "artist_indices",
                "predicted_index",
                "is_top1",
                "is_top5",
                "file_id",
                "num_train_tracks",
            ],
        )
        df.to_excel(f"{STORAGE_PATH}/cloned_results.xlsx", index=False)
