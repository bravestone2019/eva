from pathlib import Path

import numpy as np
import warnings

import torch
import soundfile as sf

from silero_vad import get_speech_timestamps

from app.config import (
    TARGET_SAMPLE_RATE,
    VAD_THRESHOLD,
    SPEECH_PAD_MS,
    MIN_SPEECH_DURATION_MS,
    MIN_SILENCE_DURATION_MS,
    WHISPER_LANGUAGE,
)


warnings.filterwarnings(
    "ignore",
    message=(
        "Support for mismatched key_padding_mask and attn_mask "
        "is deprecated.*"
    ),
    category=UserWarning
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
        self.whisper_processor = getattr(model_manager, "whisper_processor", None)

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

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        sf.write(
            output_file,
            audio,
            TARGET_SAMPLE_RATE,
            subtype="FLOAT"
        )

    def transcribe(
        self,
        audio
    ):

        if audio is None:
            return ""

        if self.whisper_processor is not None:
            waveform = np.asarray(audio, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            inputs = self.whisper_processor.feature_extractor(
                waveform,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.whisper_model.device)
            generation_options = {}
            if WHISPER_LANGUAGE != "auto":
                language_names = {"english": "english", "hindi": "hindi"}
                language = language_names.get(WHISPER_LANGUAGE, WHISPER_LANGUAGE)
                generation_options["language"] = language
                generation_options["task"] = "transcribe"
            else:
                # Avoid forcing the language used during fine-tuning when the
                # operator explicitly requests automatic detection.
                generation_options["language"] = None
                generation_options["task"] = "transcribe"
                self.whisper_model.generation_config.forced_decoder_ids = None
            with torch.inference_mode():
                predicted_ids = self.whisper_model.generate(
                    input_features,
                    **generation_options,
                )
            return self.whisper_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0].strip()

        result = (
            self.whisper_model.transcribe(

                audio,

                fp16=torch.cuda.is_available()
            )
        )

        return result["text"].strip()
