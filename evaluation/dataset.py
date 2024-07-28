import json
import math
import pickle
import random
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pandas as pd
import torch
import torchaudio  # type: ignore

from evaluation.parameters import (
    CLONED_DATASET_ARTISTS,
    CLONED_DATASET_DIR,
    FMA_PICKLES_PATH,
    MTG_PICKLES_PATH,
    PICKLES_PATH,
)
from foundation.audio import AudioReader
from foundation.constants import EPSILON, HOP_LENGTH, MAX_DURATION, N_FFT, N_MELS, SAMPLING_FREQUENCY, SEG_DURATION
from training.parameters import FEATURE_DIM, MODEL_CONFIG

path2pickle: Callable[[str], str] = lambda f: f"{PICKLES_PATH}/{f}.pickle"


class TrainDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    Train dataset for singer identification downstream task.
    """

    def __init__(self, data_dict: Dict[str, Any], artist_ids: List[str]) -> None:
        """
        Args:
            data_dict (Dict[str, Any]):
                Dictionary with artist as keys and tracks as values.
            artist_ids (List[str]):
                Artists to use for downstream artists identification.
        """
        self.data_dict = data_dict
        self.num_artists = len(artist_ids)
        self.artist_ids = [(idx, artist_id) for idx, artist_id in zip(range(len(artist_ids)), artist_ids)]

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of artists in train set.
        """
        return self.num_artists

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        ---
        Returns:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                Track mel-spec and tag.
        """
        while True:
            idx, artist_id = random.choice(self.artist_ids)

            track = random.choice(self.data_dict[artist_id]["tracks"][1:])
            path = track["path"]
            offset = random.choice(track["vocal"])

            # Get embedding
            with open(path2pickle(path), "rb") as handle:
                embeddings = pickle.load(handle)

            embedding = embeddings[offset]

            tag = torch.zeros(self.num_artists)
            tag[idx] = 1.0

            yield embedding, tag


class ValDataset(torch.utils.data.Dataset):  # type: ignore
    """
    Validation dataset for singer identification downstream task.
    """

    def __init__(
        self, data_dict: Dict[str, Any], artist_ids: List[str], factor: int = 4, use_cache: bool = False
    ) -> None:
        """
        Args:
            data_dict (Dict[str, Any]):
                Dictionary with artist as keys and tracks as values.
            artist_ids (List[str]):
                Artists to use for downstream artists identification.
            factor (int):
                Number of segments per validation track.
            use_cache (bool):
                Use cached dataset? Set to True after first epoch with set_use_cache :)
        """
        self.data_dict = data_dict
        self.num_artists = len(artist_ids)

        repeated_artist_ids = [(idx, artist_id) for idx, artist_id in zip(range(len(artist_ids)), artist_ids)]
        self.artist_ids = repeated_artist_ids * factor
        self.cached_data: Dict[int, Any] = {}
        self.use_cache = use_cache

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
        if not use_cache:
            self.cached_data = {}

        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Mel-spec and tag for index.
        """
        if not self.use_cache:
            idx, artist_id = self.artist_ids[index]

            track = self.data_dict[artist_id]["tracks"][0]
            path = track["path"]
            offset = random.choice(track["vocal"])

            # Get embedding
            with open(path2pickle(path), "rb") as handle:
                embeddings = pickle.load(handle)

            embedding = embeddings[offset]

            tag = torch.zeros(self.num_artists)
            tag[idx] = 1.0

            self.cached_data[index] = (embedding, tag)

        else:
            embedding, tag = self.cached_data[index]

        return embedding, tag


class TestDataset(torch.utils.data.Dataset):  # type: ignore
    """
    Test dataset for singer identification downstream task.
    """

    def __init__(self, data_dict: Dict[str, Any], artist_ids: List[str]) -> None:
        """
        Args:
            data_dict (Dict[str, Any]):
                Dictionary with artist as keys and tracks as values.
            artist_ids (List[str]):
                Artists to use for downstream artists identification.
        """
        self.data_dict = data_dict
        self.num_artists = len(artist_ids)

        self.artist_ids = [(idx, artist_id) for idx, artist_id in zip(range(len(artist_ids)), artist_ids)]

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of tracks in validation set.
        """
        return self.num_artists

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                Batch of mel-specs, tag, and metadata.
        """
        idx, artist_id = self.artist_ids[index]

        track = self.data_dict[artist_id]["test_track"]
        path = track["path"]
        metadata = {
            "artist_name": self.data_dict[artist_id]["name"],
            "artist_id": artist_id,
            "path": path,
            "macro_genre": track["macro_genre"],
            "lang": track["lang"],
            "num_train_tracks": len(self.data_dict[artist_id]["tracks"]) - 1,
        }

        # Get embedding
        with open(path2pickle(path), "rb") as handle:
            embeddings = pickle.load(handle)

        # Create batch of mel-specs
        embedding_batch = torch.zeros(len(track["vocal"]), FEATURE_DIM)

        for x, (_, embedding) in enumerate(embeddings.items()):
            embedding_batch[x, :] = embedding

        # Create tag
        tag = torch.zeros(self.num_artists)
        tag[idx] = 1.0

        return embedding_batch, tag, metadata


class ClonedDataset(torch.utils.data.Dataset):  # type: ignore
    """
    Cloned dataset for singer identification downstream task.
    """

    def __init__(self, data_dict: Dict[str, Any], artist_ids: List[str]) -> None:
        """
        Args:
            data_dict (Dict[str, Any]):
                Dictionary with artist as keys and tracks as values.
            artist_ids (List[str]):
                Artists to use for downstream artists identification.
        """
        # Dict with artist id as key and its index as value
        self.num_artists = len(artist_ids)
        artist_indices = {artist_id: idx for idx, artist_id in zip(range(len(artist_ids)), artist_ids)}
        self.artist_ids = {idx: artist_id for idx, artist_id in zip(range(len(artist_ids)), artist_ids)}
        self.data_dict = data_dict

        # Load cloned dataset metadata and save it for in-depth results
        cloned_df = pd.read_excel(CLONED_DATASET_DIR + "/cloned_dataset.xlsx")
        self.tracks = []

        for _, row in cloned_df.iterrows():
            list_row = list(row)

            cloned_ids = str(list_row[1]).split(" ")
            cloned_indices = [artist_indices[id] for id in cloned_ids]

            num_train_tracks = [len(self.data_dict[id]["tracks"]) - 1 for id in cloned_ids]
            data = {
                "artist_names": list_row[0].replace(", ", ",").split(","),
                "artist_ids": cloned_ids,
                "artist_indices": cloned_indices,
                "file_id": list_row[2],
                "num_train_tracks": num_train_tracks,
            }
            self.tracks.append(data)

        # Load cloned dataset vocal results
        with open(CLONED_DATASET_DIR + "/vocal.json", "r+") as f:
            self.vocal = json.load(f)

        # Audio parameters
        self.sampling_frequency = SAMPLING_FREQUENCY
        self.audioReader = AudioReader(self.sampling_frequency)
        self.seg_duration = SEG_DURATION
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_frequency,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.epsilon = EPSILON
        self.time_dim_len = int(self.sampling_frequency / HOP_LENGTH * SEG_DURATION)

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of tracks in cloned dataset.
        """
        return len(self.tracks)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                Batch of mel-specs, tag, and metadata.
        """
        # Get track data and file path
        track = self.tracks[index]
        audio_path = track["file_id"]
        artist_indices = track["artist_indices"]
        audio_vocal = self.vocal[audio_path]

        # Get audio
        if MODEL_CONFIG == "vocal2vocal":
            full_audio_path = f"{CLONED_DATASET_DIR}/stems/mdx_extra/{audio_path[:-4]}/vocals.mp3"
        else:
            full_audio_path = f"{CLONED_DATASET_DIR}/mixtures/{audio_path}"

        np_waveform = self.audioReader.load_audio(full_audio_path, 0.0, MAX_DURATION)
        waveform = torch.tensor(np_waveform)

        # Create batch of mel-specs
        waveform_batch = torch.zeros(len(audio_vocal), 2, self.sampling_frequency * self.seg_duration)

        for x, offset in enumerate(audio_vocal):
            start = int(self.sampling_frequency * offset)
            end = start + self.seg_duration * self.sampling_frequency
            waveform_batch[x, :, :] = waveform[:, start:end]

        melspec_features = self.melspec_transform(waveform_batch)
        melspec_features = torch.log(torch.add(melspec_features, self.epsilon))
        time_dim = melspec_features.shape[3]
        closest_time = self.time_dim_len // 2 * math.floor(time_dim / (self.time_dim_len // 2))
        melspec_features = melspec_features[:, :, :, :closest_time]

        # Create tag
        tag = torch.zeros(self.num_artists)

        for idx in artist_indices:
            tag[int(idx)] = 1.0

        return melspec_features, tag, track


class FMATrainDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    FMA train dataset for voice similarity downstream task.
    """

    def __init__(self, data_path: str = "splits/FMA/fma_train_set.json") -> None:
        """
        Args:
            data_path (optional, str):
                Path to .json file.
        """
        with open(data_path, "r+") as f:
            self.data_dict = json.load(f)

        self.num_artists = len(self.data_dict)
        self.artist_ids = [(idx, artist_id) for idx, artist_id in zip(range(self.num_artists), self.data_dict.keys())]

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of artists in train set.
        """
        return self.num_artists

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        ---
        Returns:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                Track mel-spec and tag.
        """
        while True:
            idx, artist_id = random.choice(self.artist_ids)

            track = random.choice(self.data_dict[artist_id]["track"])
            file_id = track["file_id"][:-4]
            offset = random.choice(track["vocal_segs"])

            # Get embedding
            with open(f"{FMA_PICKLES_PATH}/{file_id}.pickle", "rb") as handle:
                embeddings = pickle.load(handle)

            embedding = embeddings[offset]

            tag = torch.zeros(self.num_artists)
            tag[idx] = 1.0

            yield embedding, tag


class FMAValDataset(torch.utils.data.Dataset):  # type: ignore
    """
    FMA Validation dataset for voice similarity downstream task.
    """

    def __init__(
        self,
        artist_ids: List[Tuple[int, str]],
        data_path: str = "splits/FMA/fma_val_set.json",
        factor: int = 4,
        use_cache: bool = False,
    ) -> None:
        """
        Args:
            artist_ids (List[Tuple[int, str]]):
                Artist IDs with correct indices.
            data_path (optional, str):
                Path to .json file.
            factor (optional, int):
                Number of segments per validation track.
            use_cache (optional, bool):
                Use cached dataset? Set to True after first epoch with set_use_cache :)
        """
        with open(data_path, "r+") as f:
            self.data_dict = json.load(f)

        self.artist_ids = artist_ids
        self.num_artists = len(self.artist_ids)

        self.repeated_artist_ids = self.artist_ids * factor
        self.cached_data: Dict[int, Any] = {}
        self.use_cache = use_cache

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of val set tracks.
        """
        return len(self.repeated_artist_ids)

    def set_use_cache(self, use_cache: bool) -> None:
        """
        Function for creating cached torch dataset.
        """
        if not use_cache:
            self.cached_data = {}

        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Mel-spec and tag for index.
        """
        if not self.use_cache:
            idx, artist_id = self.repeated_artist_ids[index]

            track = self.data_dict[artist_id]["track"]
            file_id = track["file_id"][:-4]
            offset = random.choice(track["vocal_segs"])

            # Get embedding
            with open(f"{FMA_PICKLES_PATH}/{file_id}.pickle", "rb") as handle:
                embeddings = pickle.load(handle)

            embedding = embeddings[offset]

            tag = torch.zeros(self.num_artists)
            tag[idx] = 1.0

            self.cached_data[index] = (embedding, tag)

        else:
            embedding, tag = self.cached_data[index]

        return embedding, tag


class FMATestDataset(torch.utils.data.Dataset):  # type: ignore
    """
    FMA test dataset for voice similarity downstream task.
    """

    def __init__(
        self,
        artist_ids: List[Tuple[int, str]],
        data_path: str = "splits/FMA/fma_test_set.json",
    ) -> None:
        """
        Args:
            artist_ids (List[Tuple[int, str]]):
                Artist IDs with correct indices.
            data_path (optional, str):
                Path to .json file.
        """
        with open(data_path, "r+") as f:
            self.data_dict = json.load(f)

        self.artist_ids = artist_ids
        self.num_artists = len(self.artist_ids)

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of tracks in validation set.
        """
        return self.num_artists

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                Batch of mel-specs, tag, and metadata.
        """
        idx, artist_id = self.artist_ids[index]

        track = self.data_dict[artist_id]["track"]
        file_id = track["file_id"][:-4]

        # Get embedding
        with open(f"{FMA_PICKLES_PATH}/{file_id}.pickle", "rb") as handle:
            embeddings = pickle.load(handle)

        # Create batch of mel-specs
        embedding_batch = torch.zeros(len(track["vocal_segs"]), FEATURE_DIM)

        for x, (_, embedding) in enumerate(embeddings.items()):
            embedding_batch[x, :] = embedding

        # Create tag
        tag = torch.zeros(self.num_artists)
        tag[idx] = 1.0

        return embedding_batch, tag


class MTGTrainDataset(torch.utils.data.IterableDataset):  # type: ignore
    """
    MTG train dataset for voice similarity downstream task.
    """

    def __init__(self, data_path: str = "splits/mtg-jamendo-dataset/mtg_train_set.json") -> None:
        """
        Args:
            data_path (optional, str):
                Path to .json file.
        """
        with open(data_path, "r+") as f:
            self.data_dict = json.load(f)

        self.num_artists = len(self.data_dict)
        self.artist_ids = [(idx, artist_id) for idx, artist_id in zip(range(self.num_artists), self.data_dict.keys())]

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of artists in train set.
        """
        return self.num_artists

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        ---
        Returns:
            Iterator[Tuple[torch.Tensor, torch.Tensor]]:
                Track mel-spec and tag.
        """
        while True:
            idx, artist_id = random.choice(self.artist_ids)

            track = random.choice(self.data_dict[artist_id])
            file_id = track["file_id"].split("/")[1][:-4]
            offset = random.choice(track["vocal_segs"])

            # Get embedding
            with open(f"{MTG_PICKLES_PATH}/{file_id}.pickle", "rb") as handle:
                embeddings = pickle.load(handle)

            embedding = embeddings[offset]

            tag = torch.zeros(self.num_artists)
            tag[idx] = 1.0

            yield embedding, tag


class MTGValDataset(torch.utils.data.Dataset):  # type: ignore
    """
    MTG Validation dataset for voice similarity downstream task.
    """

    def __init__(
        self,
        artist_ids: List[Tuple[int, str]],
        data_path: str = "splits/mtg-jamendo-dataset/mtg_val_set.json",
        factor: int = 4,
        use_cache: bool = False,
    ) -> None:
        """
        Args:
            artist_ids (List[Tuple[int, str]]):
                Artist IDs with correct indices.
            data_path (optional, str):
                Path to .json file.
            factor (optional, int):
                Number of segments per validation track.
            use_cache (optional, bool):
                Use cached dataset? Set to True after first epoch with set_use_cache :)
        """
        with open(data_path, "r+") as f:
            self.data_dict = json.load(f)

        self.artist_ids = artist_ids
        self.num_artists = len(self.artist_ids)

        self.repeated_artist_ids = self.artist_ids * factor
        self.cached_data: Dict[int, Any] = {}
        self.use_cache = use_cache

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of val set tracks.
        """
        return len(self.repeated_artist_ids)

    def set_use_cache(self, use_cache: bool) -> None:
        """
        Function for creating cached torch dataset.
        """
        if not use_cache:
            self.cached_data = {}

        self.use_cache = use_cache

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Mel-spec and tag for index.
        """
        if not self.use_cache:
            idx, artist_id = self.repeated_artist_ids[index]

            track = self.data_dict[artist_id]
            file_id = track["file_id"].split("/")[1][:-4]
            offset = random.choice(track["vocal_segs"])

            # Get embedding
            with open(f"{MTG_PICKLES_PATH}/{file_id}.pickle", "rb") as handle:
                embeddings = pickle.load(handle)

            embedding = embeddings[offset]

            tag = torch.zeros(self.num_artists)
            tag[idx] = 1.0

            self.cached_data[index] = (embedding, tag)

        else:
            embedding, tag = self.cached_data[index]

        return embedding, tag


class MTGTestDataset(torch.utils.data.Dataset):  # type: ignore
    """
    MTG test dataset for voice similarity downstream task.
    """

    def __init__(
        self,
        artist_ids: List[Tuple[int, str]],
        data_path: str = "splits/mtg-jamendo-dataset/mtg_test_set.json",
    ) -> None:
        """
        Args:
            artist_ids (List[Tuple[int, str]]):
                Artist IDs with correct indices.
            data_path (optional, str):
                Path to .json file.
        """
        with open(data_path, "r+") as f:
            self.data_dict = json.load(f)

        self.artist_ids = artist_ids
        self.num_artists = len(self.artist_ids)

    def __len__(self) -> int:
        """
        Dataset length.
        ---
        Returns:
            int:
                Number of tracks in validation set.
        """
        return self.num_artists

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ---
        Args:
            index (int):
                Index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
                Batch of mel-specs, tag, and metadata.
        """
        idx, artist_id = self.artist_ids[index]

        track = self.data_dict[artist_id]
        file_id = track["file_id"].split("/")[1][:-4]

        # Get embedding
        with open(f"{MTG_PICKLES_PATH}/{file_id}.pickle", "rb") as handle:
            embeddings = pickle.load(handle)

        # Create batch of mel-specs
        embedding_batch = torch.zeros(len(track["vocal_segs"]), FEATURE_DIM)

        for x, (_, embedding) in enumerate(embeddings.items()):
            embedding_batch[x, :] = embedding

        # Create tag
        tag = torch.zeros(self.num_artists)
        tag[idx] = 1.0

        return embedding_batch, tag


# FOR DEBUGGING
if __name__ == "__main__":
    print("Loading .json files...")
    with open("data/test_set.json", "r+") as f:
        test_set = json.load(f)

    artist_ids = random.sample(list(test_set.keys()), k=200)

    print("\nTrain set:")
    trainDataset = TrainDataset(test_set, artist_ids)
    iterTrainDataset = iter(trainDataset)
    train_segment, train_tag = next(iterTrainDataset)
    print("Segment shape is: {}.".format(train_segment.shape))
    print("Tag shape is: {}.".format(train_tag.shape))
    print(f"Length: {len(trainDataset)}")

    print("\nValidation set:")
    valDataset = ValDataset(test_set, artist_ids)
    idx = random.randint(0, len(valDataset) - 1)
    val_segment, val_tag = valDataset[idx]
    print("Segment shape is: {}.".format(train_segment.shape))
    print("Tag shape is: {}.".format(val_tag.shape))
    print(f"Length: {len(valDataset)}")

    print("\nTest set:")
    testDataset = TestDataset(test_set, artist_ids)
    idx = random.randint(0, len(testDataset) - 1)
    test_segment, test_tag, metadata = testDataset[idx]
    print("Segment shape is: {}.".format(test_segment.shape))
    print("Tag shape is: {}.".format(test_tag.shape))
    print(f"Metadata: {metadata}")
    print(f"Length: {len(testDataset)}")

    print("\nCloned set:")
    clonedDataset = ClonedDataset(test_set, CLONED_DATASET_ARTISTS)
    idx = random.randint(0, len(clonedDataset) - 1)
    cloned_segment, cloned_tag, metadata = clonedDataset[idx]
    print("Segment shape is: {}.".format(cloned_segment.shape))
    print("Tag shape is: {}.".format(cloned_tag.shape))
    print(f"Metadata: {metadata}")
    print(f"Length: {len(clonedDataset)}")

    print("\nFMA Train set:")
    fmaTrainDataset = FMATrainDataset()
    iterFMATrainDataset = iter(fmaTrainDataset)
    fma_train_segment, fma_train_tag = next(iterFMATrainDataset)
    print("Segment shape is: {}.".format(fma_train_segment.shape))
    print("Tag shape is: {}.".format(fma_train_tag.shape))
    print(f"Length: {len(fmaTrainDataset)}")

    print("\nFMA Val set:")
    fmaValDataset = FMAValDataset(fmaTrainDataset.artist_ids)
    idx = random.randint(0, len(fmaValDataset) - 1)
    fma_val_segment, fma_val_tag = fmaValDataset[idx]
    print("Segment shape is: {}.".format(fma_val_segment.shape))
    print("Tag shape is: {}.".format(fma_val_tag.shape))
    print(f"Length: {len(fmaValDataset)}")

    print("\nFMA Test set:")
    fmaTestDataset = FMATestDataset(fmaTrainDataset.artist_ids)
    idx = random.randint(0, len(fmaTestDataset) - 1)
    fma_test_segment, fma_test_tag = fmaTestDataset[idx]
    print("Segment shape is: {}.".format(fma_test_segment.shape))
    print("Tag shape is: {}.".format(fma_test_tag.shape))
    print(f"Length: {len(fmaTestDataset)}")

    print("\nMTG Train set:")
    mtgTrainDataset = MTGTrainDataset()
    iterMTGTrainDataset = iter(mtgTrainDataset)
    mtg_train_segment, mtg_train_tag = next(iterMTGTrainDataset)
    print("Segment shape is: {}.".format(mtg_train_segment.shape))
    print("Tag shape is: {}.".format(mtg_train_tag.shape))
    print(f"Length: {len(mtgTrainDataset)}")

    print("\nMTG Val set:")
    mtgValDataset = MTGValDataset(mtgTrainDataset.artist_ids)
    idx = random.randint(0, len(mtgValDataset) - 1)
    mtg_val_segment, mtg_val_tag = mtgValDataset[idx]
    print("Segment shape is: {}.".format(mtg_val_segment.shape))
    print("Tag shape is: {}.".format(mtg_val_tag.shape))
    print(f"Length: {len(mtgValDataset)}")

    print("\nMTG Test set:")
    mtgTestDataset = MTGTestDataset(mtgTrainDataset.artist_ids)
    idx = random.randint(0, len(mtgTestDataset) - 1)
    mtg_test_segment, mtg_test_tag = mtgTestDataset[idx]
    print("Segment shape is: {}.".format(mtg_test_segment.shape))
    print("Tag shape is: {}.".format(mtg_test_tag.shape))
    print(f"Length: {len(mtgTestDataset)}")
