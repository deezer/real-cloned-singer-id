import datetime
import math
import re
import subprocess
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torchaudio  # type: ignore

from foundation.constants import (
    EPSILON,
    HOP_LENGTH,
    MAX_DURATION,
    N_FFT,
    N_MELS,
    SAMPLING_FREQUENCY,
    SEG_DURATION,
)


class AudioReader:
    def __init__(self, sampling_frequency: int) -> None:
        self.sampling_frequency = sampling_frequency
        self.seg_duration = SEG_DURATION
        self.epsilon = EPSILON
        self.time_dim_len = int(self.sampling_frequency / HOP_LENGTH * SEG_DURATION)

        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_frequency,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

    @staticmethod
    def _execute_command(command: list[str]) -> bytes:
        """
        Execute (ffmpeg) command.
        ---
        Args:
            command (list[str]):
                Command to execute.

        Returns:
            bytes:
                Bytes that are the result of the command.

        Raises:
            IOError:
                If error occurs during command run.
        """
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=None)
        out, err = proc.communicate(input=None)

        if proc.returncode != 0:
            message = err.decode("utf-8")
            message = re.sub(r"\n+", " | ", message.strip())
            message = re.sub(r"\s+", " ", message)
            raise IOError(f"An error occurs with cmd '{' '.join(command)}' [RC={proc.returncode}] : {message}")

        return out

    def _ffmpeg_load(self, audio_path: str, offset: float, duration: float) -> npt.NDArray[np.float32]:
        """
        Load audio using ffmpeg.
        ---
        Args:
            audio_path (str):
                Path to audio file.
            offset (float):
                Offset in seconds.
            duration (float):
                Duration in seconds.

        Returns:
            npt.NDArray[np.float32]:
                Waveform (channels-last).

        Raises:
            ValueError:
                When number of channels exceeds 2.
        """
        # Create and execute load command
        command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            audio_path,
            "-f",
            "f32le",
            "-ar",
            str(self.sampling_frequency),
            "-ss",
            str(datetime.timedelta(seconds=offset)),
            "-t",
            str(datetime.timedelta(seconds=duration)),
            "-",
        ]
        buffer = AudioReader._execute_command(command)

        # Fetch number of channels through ffprobe
        command = [
            "ffprobe",
            audio_path,
            "-show_entries",
            "stream=channels",
            "-select_streams",
            "a",
            "-of",
            "compact=p=0:nk=1",
            "-v",
            "0",
        ]
        channels = int(AudioReader._execute_command(command).decode("utf-8"))

        if channels > 2:
            raise ValueError(f"Audio in file {audio_path} has {channels} channels!")

        # Convert to proper np array (copy for writeable array; float32 to be in line with torch)
        waveform = np.frombuffer(buffer, dtype="<f4").reshape(-1, channels).copy()

        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        return waveform

    def load_audio(
        self,
        audio_path: str,
        offset: float,
        duration: float,
    ) -> Any:
        """
        Load audio from file path.
        ---
        Args:
            audio_path (str):
                Path to audio file.
            offset (float):
                Offset in seconds.
            duration (float):
                Duration in seconds.

        Returns:
            Any:
                Waveform.
        """
        # Load, normalize, and stereo if needed
        waveform = self._ffmpeg_load(audio_path, offset, duration)
        waveform_max = np.abs(waveform).max()

        if waveform_max != 0:
            waveform /= waveform_max

        if waveform.shape[-1] == 1:
            waveform = np.concatenate([waveform, waveform], axis=-1)

        # No pad to appropriate number of samples if waveform is too short
        load_len = waveform.shape[0]
        pad_duration = (
            self.sampling_frequency
            * self.seg_duration
            * math.ceil(load_len / (self.sampling_frequency * self.seg_duration))
        )
        waveform = np.vstack([waveform, np.zeros((pad_duration - load_len, 2), dtype=np.float32)])

        return waveform.T

    def tf_transform(self, audio_signal: Any) -> Any:
        """
        Transform waveform into mel-spectrogram.
        ---
        Args:
            audio_signal (Any):
                Waveform.

        Returns:
            Any:
                Mel-spectrogram.
        """
        melspec_features = self.melspec_transform(torch.from_numpy(audio_signal))
        log_melspec_features = torch.log(torch.add(melspec_features, self.epsilon))

        time_dim = log_melspec_features.shape[2]

        closest_time = self.time_dim_len // 2 * math.floor(time_dim / (self.time_dim_len // 2))

        return log_melspec_features[:, :, :closest_time]

    def __call__(
        self,
        audio_path: str,
        offset: float = 0.0,
        duration: float = MAX_DURATION,
    ) -> Any:
        waveform = self.load_audio(audio_path, offset, duration)
        return self.tf_transform(waveform)


if __name__ == "__main__":
    audio_reader = AudioReader(SAMPLING_FREQUENCY)

    waveform = audio_reader.load_audio("audio_example.mp3", 0.0, MAX_DURATION)
    mel_spec = audio_reader.tf_transform(waveform)
    print(f"Full waveform shape: {waveform.shape}")
    print(f"Full mel-spectrogram shape: {mel_spec.shape}")

    waveform = audio_reader.load_audio("audio_example.mp3", offset=10.0, duration=SEG_DURATION)
    mel_spec = audio_reader.tf_transform(waveform)
    print(f"\nSegment duration waveform shape: {waveform.shape}")
    print(f"Segment duration mel-spectrogram shape: {mel_spec.shape}")
