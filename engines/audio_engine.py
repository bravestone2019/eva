from pathlib import Path

import numpy as np

import librosa
import sounddevice as sd
import soundfile as sf

from scipy.signal import (
    butter,
    sosfilt
)

from app.config import (
    TARGET_SAMPLE_RATE,
    HIGH_PASS_CUTOFF
)


class AudioEngine:

    def __init__(self):

        self.target_sr = TARGET_SAMPLE_RATE

    def load_audio(
        self,
        input_file
    ):

        return librosa.load(
            input_file,
            sr=None,
            mono=False
        )

    def convert_to_mono(
        self,
        audio
    ):

        audio = np.asarray(audio)

        if audio.ndim == 1:
            return audio

        if audio.ndim == 2:
            return librosa.to_mono(audio)

        raise ValueError(
            f"Unsupported audio shape: {audio.shape}"
        )

    def remove_dc_offset(
        self,
        audio
    ):

        return audio - np.mean(audio)

    def resample_audio(
        self,
        audio,
        original_sr
    ):

        if original_sr == self.target_sr:
            return audio

        return librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=self.target_sr
        ).astype(np.float32)

    def highpass_filter(
        self,
        audio
    ):

        sos = butter(
            4,
            HIGH_PASS_CUTOFF,
            btype="highpass",
            fs=self.target_sr,
            output="sos"
        )

        return sosfilt(
            sos,
            audio
        ).astype(np.float32)

    def peak_protection(
        self,
        audio
    ):

        peak = np.max(
            np.abs(audio)
        )

        if peak > 0.99:

            audio = (
                audio / peak
            ) * 0.99

        return audio

    def preprocess(
        self,
        input_file,
        output_file
    ):

        audio, original_sr = (
            self.load_audio(
                input_file
            )
        )

        audio = self.convert_to_mono(
            audio
        )

        audio = audio.astype(
            np.float32
        )

        audio = self.remove_dc_offset(
            audio
        )

        audio = self.resample_audio(
            audio,
            original_sr
        )

        audio = self.highpass_filter(
            audio
        )

        audio = self.peak_protection(
            audio
        )

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        sf.write(
            output_file,
            audio,
            self.target_sr,
            subtype="FLOAT"
        )

        return audio

    def record_microphone(
        self,
        output_file,
        duration_seconds
    ):

        if duration_seconds <= 0:
            raise ValueError(
                "Recording duration must be greater than zero."
            )

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        audio = sd.rec(
            int(duration_seconds * self.target_sr),
            samplerate=self.target_sr,
            channels=1,
            dtype="float32"
        )

        sd.wait()

        sf.write(
            output_file,
            audio[:, 0],
            self.target_sr,
            subtype="FLOAT"
        )

        return output_file
