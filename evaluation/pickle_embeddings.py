import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from torchinfo import summary
from tqdm import tqdm

from evaluation.parameters import (
    BACKBONE_WEIGHTS_PATH,
    BATCH_SIZE,
    FMA_PICKLES_PATH,
    MTG_PICKLES_PATH,
    PICKLES_PATH,
)
from foundation.audio import AudioReader
from foundation.constants import MAX_DURATION, N_FRAMES, N_MELS, SAMPLING_FREQUENCY, SEG_DURATION
from foundation.model import ASTModel
from training.parameters import FEATURE_DIM, MODEL_CONFIG

path2mixture: Callable[[str], str] = lambda f: f"data/mixtures/{f}.mp3"
path2vocal: Callable[[str], str] = lambda f: f"data/vocals/{f}.mp3"

fma_fid2path: Callable[[str], str] = lambda f: f"data/FMA/fma_full/{f[:3]}/{f}"
fma_fid2vocal: Callable[[str], str] = lambda f: f"data/FMA/stems/mdx_extra/{f[:-4]}/vocals.mp3"

mtg_paths = "data/mtg-jamendo-dataset/audio/"
mtg_vocals = "data/mtg-jamendo-dataset/stems/mdx_extra/"


def generatePickledEmbeddings(track: Dict[str, Any]) -> None:
    """
    Pickle file name: md5.pickle
    Pickle file values: {
        vocal segment time (s): embedding value (2048 floats)
    }
    ---
    track (Dict[str, Any]):
        Test set track in dict format.
    """
    md5 = track["md5"]

    if not os.path.exists(f"{PICKLES_PATH}/{md5}.pickle"):
        vocal = track["vocal"]
        pickle_dict = {}

        if MODEL_CONFIG == "vocal2vocal":
            audio = audioReader.load_audio(path2vocal(md5), 0, MAX_DURATION)
        else:
            audio = audioReader.load_audio(path2mixture(md5), 0, MAX_DURATION)

        for offset in vocal:
            start_index = int(offset * SAMPLING_FREQUENCY)
            waveform = audio[:, start_index : start_index + (SEG_DURATION * SAMPLING_FREQUENCY)]
            mel_spec = audioReader.tf_transform(waveform)

            if cuda_available:
                mel_spec = mel_spec.cuda()

            output = feature_model(mel_spec.unsqueeze(0))
            pickle_dict[offset] = output.detach().squeeze(0).cpu()

        with open(f"{PICKLES_PATH}/{md5}.pickle", "wb") as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def generateMTGEmbeddings(track: Dict[str, Any]) -> None:
    """
    Pickle file name: file_id.pickle
    Pickle file values: {
        vocal segment time (s): embedding value (2048 floats)
    }
    ---
    track (Dict[str, Any]):
        MTG track in dict format.
    """
    file_id = track["file_id"].split("/")[1][:-4]

    if not os.path.exists(f"{MTG_PICKLES_PATH}/{file_id}.pickle"):
        vocal = track["segs"]
        pickle_dict = {}

        if MODEL_CONFIG == "vocal2vocal":
            audio = audioReader.load_audio(mtg_vocals + file_id + "/vocals.mp3", 0, MAX_DURATION)
        else:
            audio = audioReader.load_audio(mtg_paths + track["file_id"], 0, MAX_DURATION)

        for offset in vocal:
            start_index = int(offset * SAMPLING_FREQUENCY)
            waveform = audio[:, start_index : start_index + (SEG_DURATION * SAMPLING_FREQUENCY)]
            mel_spec = audioReader.tf_transform(waveform)

            if cuda_available:
                mel_spec = mel_spec.cuda()

            output = feature_model(mel_spec.unsqueeze(0))
            pickle_dict[offset] = output.detach().squeeze(0).cpu()

        with open(f"{MTG_PICKLES_PATH}/{file_id}.pickle", "wb") as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


def generateFMAEmbeddings(track: Dict[str, Any]) -> None:
    """
    Pickle file name: file_id.pickle
    Pickle file values: {
        vocal segment time (s): embedding value (2048 floats)
    }
    ---
    track (Dict[str, Any]):
        FMA track in dict format.
    """
    file_id = track["file_id"][:-4]

    if not os.path.exists(f"{FMA_PICKLES_PATH}/{file_id}.pickle"):
        vocal = track["vocal_segs"]
        pickle_dict = {}

        if MODEL_CONFIG == "vocal2vocal":
            audio = audioReader.load_audio(fma_fid2vocal(track["file_id"]), 0, MAX_DURATION)
        else:
            audio = audioReader.load_audio(fma_fid2path(track["file_id"]), 0, MAX_DURATION)

        for offset in vocal:
            start_index = int(offset * SAMPLING_FREQUENCY)
            waveform = audio[:, start_index : start_index + (SEG_DURATION * SAMPLING_FREQUENCY)]
            mel_spec = audioReader.tf_transform(waveform)

            if cuda_available:
                mel_spec = mel_spec.cuda()

            output = feature_model(mel_spec.unsqueeze(0))
            pickle_dict[offset] = output.detach().squeeze(0).cpu()

        with open(f"{FMA_PICKLES_PATH}/{file_id}.pickle", "wb") as handle:
            pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


if __name__ == "__main__":
    with open("data/test_set.json", "r+") as f:
        test_set = json.load(f)

    with open("data/mtg_vocal.json", "r+") as f:
        mtg = json.load(f)

    with open("data/fma_vocal.json", "r+") as f:
        fma = json.load(f)

    audioReader = AudioReader(SAMPLING_FREQUENCY)

    # Import pre-trained model
    feature_model = ASTModel(
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
    feature_model.load_state_dict(filtered_weights)
    summary(feature_model.eval(), input_size=(BATCH_SIZE, 2, N_MELS, N_FRAMES))

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        feature_model = feature_model.cuda()

    if not os.path.isdir(PICKLES_PATH):
        Path(PICKLES_PATH).mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(MTG_PICKLES_PATH):
        Path(MTG_PICKLES_PATH).mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(FMA_PICKLES_PATH):
        Path(FMA_PICKLES_PATH).mkdir(parents=True, exist_ok=True)

    # Now iterate through test set md5s and save embedding values
    with torch.inference_mode():
        feature_model.eval()

        for _, artist_values in tqdm(test_set.items()):
            # Process test track
            track = artist_values["test_track"]
            generatePickledEmbeddings(track)

            # Process training tracks
            for track in artist_values["tracks"]:
                generatePickledEmbeddings(track)

        for _, artist_values in tqdm(mtg.items()):
            for track in artist_values:
                generateMTGEmbeddings(track)

        for _, artist_values in tqdm(fma.items()):
            for track in artist_values["valid_files"]:
                generateFMAEmbeddings(track)
