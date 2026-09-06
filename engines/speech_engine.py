import numpy as np
import torch
import soundfile as sf

from silero_vad import get_speech_timestamps

from config import (
    TARGET_SAMPLE_RATE,
    VAD_THRESHOLD,
    SPEECH_PAD_MS,
    MIN_SPEECH_DURATION_MS,
    MIN_SILENCE_DURATION_MS
)


class SpeechEngine:

    def __init__(
        self,
        model_manager
    ):

        self.vad_model = (
            model_manager.vad_model
        )

        self.whisper_model = (
            model_manager.whisper_model
        )

    def detect_speech(
        self,
        audio
    ):

        audio_tensor = (
            torch.from_numpy(
                audio
            )
        )

        timestamps = get_speech_timestamps(

            audio_tensor,

            self.vad_model,

            sampling_rate=
            TARGET_SAMPLE_RATE,

            threshold=
            VAD_THRESHOLD,

            min_speech_duration_ms=
            MIN_SPEECH_DURATION_MS,

            min_silence_duration_ms=
            MIN_SILENCE_DURATION_MS,

            speech_pad_ms=
            SPEECH_PAD_MS
        )

        return timestamps

    def extract_speech(
        self,
        audio,
        timestamps
    ):

        if not timestamps:
            return None

        segments = []

        for segment in timestamps:

            start = segment["start"]

            end = segment["end"]

            segments.append(
                audio[start:end]
            )

        return np.concatenate(
            segments
        )

    def save_speech(
        self,
        audio,
        output_file
    ):

        sf.write(
            output_file,
            audio,
            TARGET_SAMPLE_RATE,
            subtype="FLOAT"
        )

    def transcribe(
        self,
        speech_file
    ):

        if speech_file is None:
            return ""

        result = (
            self.whisper_model.transcribe(

                speech_file,

                fp16=torch.cuda.is_available()
            )
        )

        return result["text"].strip()