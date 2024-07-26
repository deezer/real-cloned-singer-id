import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary
from tqdm import tqdm, trange

from foundation.constants import N_FRAMES, N_MELS, SAMPLING_FREQUENCY
from foundation.model import TrainingWrapper
from training.dataset import ContrastiveTrainDataset, ContrastiveValDataset
from training.loss import NT_Xent
from training.parameters import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EARLY_STOP,
    EPSILON,
    FACTOR,
    FEATURE_DIM,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_CONFIG,
    NUM_WORKERS,
    PROJ_DIM,
    STEP_SIZE,
    TEMPERATURE,
    TRAIN_STEPS,
)


class Trainer(object):
    """
    Object for pre-training a model for the voice similarity contrastive learning task :)
    """

    def __init__(
        self,
        train_dict: Dict[str, Any],
        val_dict: Dict[str, Any],
        batch_size: int,
        num_workers: int,
        train_steps: int,
        sampling_frequency: int,
        n_mels: int,
        n_frames: int,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float,
        epsilon: float,
        temperature: float,
        early_stop: int,
        step_size: int,
        factor: float,
        checkpoint_path: str,
        model_config: str,
    ) -> None:
        """
        Args:
            train_dict (Dict[str, Any]):
                Train set dictionary.
            val_dict (Dict[str, Any]):
                Validation set dictionary.
            batch_size (int):
                Batch size.
            num_workers (int):
                Number of workers for torch multiprocessing.
            train_steps (int):
                Number of training steps.
            sampling_frequency (int):
                Sampling frequency in Hz.
            n_mels (int):
                Number of mel filters.
            n_frames (int):
                Number of time frames.
            feature_dim (int):
                Embedding feature dimension.
            hidden_dim (int):
                Hidden dimension.
            output_dim (int):
                Output dimension.
            learning_rate (float):
                Learning rate.
            epsilon (float):
                Epsilon for NT-Xent and ADAM optimizer.
            temperature (float):
                Temperature for NT-Xent loss.
            early_stop (int):
                Number of non-diminishing loss epochs before stopping training.
            step_size (int):
                Number of steps after which learning rate decreases.
            factor (float):
                Factor at which learning rate is decreased.
            checkpoint_path (str):
                Path to training checkpoint.
            model_config (str):
                Type of pre-training task. Can be:
                * mix2mix
                * vocal2vocal
                * anything else: combines both of the above
        """

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.epsilon = epsilon
        self.temperature = temperature
        self.checkpoint_path = checkpoint_path
        self.cache_path = self.checkpoint_path + "/cache_val.pkl"
        self.early_stop = early_stop

        trainDataset = ContrastiveTrainDataset(train_dict, sampling_frequency, model_config)

        self.model = TrainingWrapper(
            n_mels=n_mels,
            n_frames=n_frames,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.train_loader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        self.criterion = NT_Xent(self.batch_size, self.temperature)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
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

        self.cuda_available = torch.cuda.is_available()

        if self.cuda_available:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.checkpoint()

        summary(self.model.eval(), input_size=(self.batch_size, 2, n_mels, n_frames))
        summary(
            self.criterion,
            input_size=((self.batch_size, output_dim), (self.batch_size, output_dim)),
        )

        if not os.path.isfile(self.cache_path):
            self.val_loader = torch.utils.data.DataLoader(
                ContrastiveValDataset(val_dict, sampling_frequency, model_config, use_cache=False),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
            self.cache_val()

        else:
            self.load_val()

    def train(self) -> None:
        """
        Train model!
        """
        self.train_iterator = iter(self.train_loader)

        while self.plateau < self.early_stop:
            print("\n--- Epoch {} ---".format(self.epoch))

            train_mean, train_std = self.train_epoch()
            val_mean, val_std = self.val_epoch()

            self.mean_train_loss.append(train_mean)
            self.std_train_loss.append(train_std)
            self.mean_val_loss.append(val_mean)
            self.std_val_loss.append(val_std)

            print("training set - loss: {:.3f} +/- {:.3f}".format(train_mean, train_std))
            print("validation set - loss: {:.3f} +/- {:.3f}".format(val_mean, val_std))

            self.save_epoch(val_mean)
            self.scheduler.step(val_mean)

            if self.epoch % 25 == 0:
                self.plot()

            self.epoch += 1
            torch.cuda.empty_cache()

        self.plot()

        return

    def train_epoch(self) -> Tuple[Any, Any]:
        """
        One epoch of training :)
        """
        self.model.train()
        train_loss = []

        for _ in trange(self.train_steps):
            # Fetch anchors and positives
            anchors, positives = next(self.train_iterator)

            # Move to GPU if available
            if self.cuda_available:
                anchors, positives = anchors.cuda(), positives.cuda()

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            anchor_outs = self.model(anchors)
            positive_outs = self.model(positives)

            # Loss computation and backward pass
            loss = self.criterion(anchor_outs, positive_outs)
            loss.backward()

            # Optimize
            self.optimizer.step()

            # Monitor metrics
            loss_value = loss.item()
            train_loss.append(loss_value)

        return np.mean(train_loss), np.std(train_loss)

    def val_epoch(self) -> Tuple[Any, Any]:
        """
        One epoch of validation :)
        """
        self.model.eval()
        val_loss = []

        with torch.inference_mode():
            for anchors, positives in tqdm(self.val_loader):
                # Move to GPU if available
                if self.cuda_available:
                    anchors, positives = anchors.cuda(), positives.cuda()

                # Forward pass
                anchor_outs = self.model(anchors)
                positive_outs = self.model(positives)

                # Loss computation and backward pass
                loss = self.criterion(anchor_outs, positive_outs)

                # Monitor metrics
                val_loss.append(loss.item())

        return np.mean(val_loss), np.std(val_loss)

    def save_epoch(self, val_loss: float) -> None:
        """
        Save best and last epoch training objects.
        """
        if val_loss < self.best_loss_value:
            self.best_loss_value: float = val_loss

            self.plateau = 0

            best_epoch: Dict[str, Any] = {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "best_loss_value": self.best_loss_value,
            }
            torch.save(
                best_epoch,
                self.checkpoint_path + "/best_epoch.pt",
            )
            print("...Saved best model checkpoint.")

        else:
            self.plateau += 1
            print(f"...Plateau set at {self.plateau}.")

        last_epoch: Dict[str, Any] = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "criterion_state_dict": self.criterion.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss_value": self.best_loss_value,
            "plateau": self.plateau,
            "mean_val_loss": self.mean_val_loss,
            "std_val_loss": self.std_val_loss,
            "mean_train_loss": self.mean_train_loss,
            "std_train_loss": self.std_train_loss,
        }
        torch.save(
            last_epoch,
            self.checkpoint_path + "/last_epoch.pt",
        )

        return

    def plot(self) -> None:
        """
        Plot loss curve.
        """
        plt.plot(self.mean_train_loss, "b")
        plt.plot(self.mean_val_loss, "r")
        plt.legend(["Train", "Val"])
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.ylim(0, math.ceil(max(self.mean_train_loss + self.mean_val_loss)))
        plt.xlim(0, self.epoch)
        plt.title("Train and Validation Loss Evolution")
        plt.tight_layout()
        plt.savefig(self.checkpoint_path + "/loss.pdf", dpi=300)

        return

    def checkpoint(self) -> None:
        """
        Load training bricks from checkpoint if one exists.
        """
        if not os.path.isfile(self.checkpoint_path + "/last_epoch.pt"):
            Path(self.checkpoint_path).mkdir(parents=True, exist_ok=True)

            self.epoch = 0

            self.best_loss_value = float("+inf")

            self.mean_val_loss: List[float] = []
            self.std_val_loss: List[float] = []
            self.mean_train_loss: List[float] = []
            self.std_train_loss: List[float] = []

            self.plateau = 0

        else:
            checkpoint = torch.load(self.checkpoint_path + "/last_epoch.pt")

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.epoch = checkpoint["epoch"] + 1

            self.best_loss_value = checkpoint["best_loss_value"]

            self.mean_val_loss = checkpoint["mean_val_loss"]
            self.std_val_loss = checkpoint["std_val_loss"]
            self.mean_train_loss = checkpoint["mean_train_loss"]
            self.std_train_loss = checkpoint["std_train_loss"]

            self.plateau = checkpoint["plateau"]

        return

    def load_val(self) -> None:
        """
        Load validation set.
        """
        with open(self.cache_path, "rb") as fp:
            self.val_loader = pickle.load(fp)

        return

    def cache_val(self) -> None:
        """
        Cache and save validation dataset.
        """
        print("\nCaching validation set...")

        for anchor, positive in tqdm(self.val_loader):
            if self.cuda_available:
                anchor, positive = (
                    anchor.cuda(non_blocking=True),
                    positive.cuda(non_blocking=True),
                )

        self.val_loader.dataset.set_use_cache(use_cache=True)  # type: ignore
        self.val_loader.dataset.num_workers = self.num_workers  # type: ignore

        with open(self.cache_path, "wb") as fp:
            pickle.dump(self.val_loader, fp, pickle.HIGHEST_PROTOCOL)

        return


if __name__ == "__main__":
    print("Loading .json files, datasets, and data loaders...")
    with open("data/train_set.json", "r+") as f:
        train_set = json.load(f)

    with open("data/val_set.json", "r+") as f:
        val_set = json.load(f)

    trainer = Trainer(
        train_set,
        val_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        train_steps=TRAIN_STEPS,
        sampling_frequency=SAMPLING_FREQUENCY,
        n_mels=N_MELS,
        n_frames=N_FRAMES,
        feature_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=PROJ_DIM,
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON,
        temperature=TEMPERATURE,
        early_stop=EARLY_STOP,
        step_size=STEP_SIZE,
        factor=FACTOR,
        checkpoint_path=CHECKPOINT_PATH,
        model_config=MODEL_CONFIG,
    )
    trainer.train()
