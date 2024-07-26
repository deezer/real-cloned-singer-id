import json
import random
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np
import torch

from foundation.audio import AudioReader
from foundation.constants import SAMPLING_FREQUENCY, SEG_DURATION
from training.parameters import MODEL_CONFIG

path2mixture: Callable[[str], str] = lambda f: f"data/mixtures/{f}.mp3"
path2vocal: Callable[[str], str] = lambda f: f"data/vocals/{f}.mp3"


class ContrastiveTrainDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    Train dataset for voice similarity contrastive learning task.
    """

    def __init__(
        self,
        data_dict: Dict[str, Any],
        sampling_frequency: int,
        model_config: str,
    ) -> None:
        """
        Args:
            data_dict (Dict[str, Any]):
                Dictionary with artist as keys and tracks as values.
            sampling_frequency (int):
                Sampling frequency.
            model_config (str):
                Type of pre-training task. Can be:
                * mix2mix
                * vocal2vocal
                * anything else: combines both of the above
        """
        self.data_dict = data_dict
        self.model_config = model_config
        self.artist_ids = list(self.data_dict.keys())
        self.audioReader = AudioReader(sampling_frequency)

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of artists in train set.
        """
        return len(self.artist_ids)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        ---
        Returns:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                Anchor and positive segments.
        """
        while True:
            artist_id = random.choice(self.artist_ids)
            track1, track2 = tuple(random.sample(self.data_dict[artist_id]["tracks"], k=2))
            md51, md52 = track1["md5"], track2["md5"]
            offset1, offset2 = random.choice(track1["instru_vocal"]), random.choice(track2["instru_vocal"])

            # Get mel-specs of segments
            md51_paths = [path2vocal(md51), path2mixture(md51)]
            md52_paths = [path2vocal(md52), path2mixture(md52)]

            if self.model_config == "mix2mix":
                anchor = self.audioReader(md51_paths[1], offset1, SEG_DURATION)
                positive = self.audioReader(md52_paths[1], offset2, SEG_DURATION)
            elif self.model_config == "vocal2vocal":
                anchor = self.audioReader(md51_paths[0], offset1, SEG_DURATION)
                positive = self.audioReader(md52_paths[0], offset2, SEG_DURATION)
            else:
                anchor = self.audioReader(random.choice(md51_paths), offset1, SEG_DURATION)
                positive = self.audioReader(random.choice(md52_paths), offset2, SEG_DURATION)

            yield anchor, positive


class ContrastiveValDataset(torch.utils.data.Dataset):  # type: ignore
    """
    Validation dataset for voice similarity contrastive learning task.
    """

    def __init__(
        self,
        data_dict: Dict[str, Any],
        sampling_frequency: int,
        model_config: str,
        use_cache: bool = False,
    ) -> None:
        """
        Args:
            data_dict (Dict[str, Any]):
                Dictionary with artist as keys and tracks as values.
            sampling_frequency (int):
                Sampling frequency.
            model_config (str):
                Type of pre-training task. Can be:
                * mix2mix
                * vocal2vocal
                * anything else: combines both of the above
            use_cache (bool):
                Use cached dataset? Set to True after first epoch with set_use_cache :)
        """
        self.cached_data: Any = []
        self.use_cache = use_cache
        self.data_dict = data_dict
        self.model_config = model_config
        self.artist_ids = np.array(list(self.data_dict.keys()))
        self.audioReader = AudioReader(sampling_frequency)

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of artists in train set.
        """
        return len(self.artist_ids)

    def set_use_cache(self, use_cache: bool) -> None:
        """
        Function for creating cached torch dataset.
        """
        if use_cache:
            self.cached_data = torch.stack(self.cached_data)

        else:
            self.cached_data = []

        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Anchor and positive segments for index.
        """
        if not self.use_cache:
            artist_id = self.artist_ids[index]
            track1, track2 = tuple(random.sample(self.data_dict[artist_id]["tracks"], k=2))
            md51, md52 = track1["md5"], track2["md5"]
            offset1, offset2 = track1["instru_vocal"], track2["instru_vocal"]

            # Get mel-specs of segments
            md51_paths = [path2vocal(md51), path2mixture(md51)]
            md52_paths = [path2vocal(md52), path2mixture(md52)]

            if self.model_config == "mix2mix":
                anchor = self.audioReader(md51_paths[1], offset1, SEG_DURATION)
                positive = self.audioReader(md52_paths[1], offset2, SEG_DURATION)
            elif self.model_config == "vocal2vocal":
                anchor = self.audioReader(md51_paths[0], offset1, SEG_DURATION)
                positive = self.audioReader(md52_paths[0], offset2, SEG_DURATION)
            else:
                anchor = self.audioReader(random.choice(md51_paths), offset1, SEG_DURATION)
                positive = self.audioReader(random.choice(md52_paths), offset2, SEG_DURATION)

            self.cached_data.append(torch.stack([anchor, positive]))

        else:
            anchor, positive = self.cached_data[index]

        return anchor, positive


# FOR DEBUGGING
if __name__ == "__main__":
    print("Loading .json files...")
    with open("data/train_set.json", "r+") as f:
        train_set = json.load(f)

    with open("data/val_set.json", "r+") as f:
        val_set = json.load(f)

    print("\nTrain dataset:")
    trainDataset = iter(ContrastiveTrainDataset(train_set, SAMPLING_FREQUENCY, MODEL_CONFIG))
    train_anchor, train_positive = next(trainDataset)
    print("Anchor shape is: {}.".format(train_anchor.shape))
    print("Positive shape is: {}.".format(train_positive.shape))

    print("\nValidation dataset:")
    valDataset = ContrastiveValDataset(val_set, SAMPLING_FREQUENCY, MODEL_CONFIG, use_cache=False)
    idx = random.randint(0, len(valDataset) - 1)
    val_anchor, val_positive = valDataset[idx]
    print("Artist name: {}.".format(valDataset.artist_ids[idx]))
    print("Anchor shape is: {}.".format(val_anchor.shape))
    print("Positive shape is: {}.".format(val_positive.shape))
