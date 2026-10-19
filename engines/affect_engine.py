"""Estimate broad affect cues from text and optional speech audio.

These are uncertain interaction signals, not claims about a person's internal
state. Models are loaded lazily and kept on CPU so they do not compete with
EVA's language model for GPU memory.
"""

from threading import Lock

import numpy as np
import torch

from app.config import (
    AFFECT_AUDIO_MODEL_NAME,
    AFFECT_CONFIDENCE_THRESHOLD,
    AFFECT_TEXT_MODEL_NAME,
    TARGET_SAMPLE_RATE,
)


RESPONSE_STYLES = {
    "angry": "concise_direct",
    "confused": "step_by_step",
    "sad": "warm_supportive",
    "fearful": "warm_supportive",
    "happy": "encouraging",
    "neutral": "standard",
    "calm": "standard",
    "disgust": "standard",
    "surprised": "standard",
    "uncertain": "standard",
}

VOICE_EMOTION_LABELS = (
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
)
AFFECT_LABELS = (*VOICE_EMOTION_LABELS, "confused")

TEXT_LABELS = {
    "angry": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fearful": ("fear", "nervousness"),
    "happy": ("joy", "amusement", "excitement", "optimism", "love", "pride", "gratitude", "relief"),
    "calm": (),
    "surprised": ("surprise",),
    "neutral": ("neutral",),
    "confused": ("confusion",),
    "sad": ("sadness", "disappointment", "grief", "remorse"),
}

AUDIO_LABELS = {
    # Keep aliases for both the pretrained SUPERB head and older EVA checkpoints.
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "frustrated": "angry",
    "hap": "happy",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "excited": "happy",
    "excitement": "happy",
    "dis": "disgust",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fea": "fearful",
    "fear": "fearful",
    "fearful": "fearful",
    "sur": "surprised",
    "surprise": "surprised",
    "surprised": "surprised",
    "cal": "calm",
    "calm": "calm",
    "sad": "sad",
    "sadness": "sad",
    "neu": "neutral",
    "neutral": "neutral",
}

AUDIO_WINDOW_SECONDS = 2.0
AUDIO_HOP_SECONDS = 1.0
AUDIO_SMOOTHING_ALPHA = 0.4
FUSION_DISAGREEMENT_THRESHOLD = 0.55


class AffectEngine:
    def __init__(self):
        self._model_lock = Lock()
        self._text_tokenizer = None
        self._text_model = None
        self._audio_processor = None
        self._audio_model = None

    def analyze(
        self,
        text="",
        audio=None,
        sample_rate=TARGET_SAMPLE_RATE,
        audio_scores=None,
    ):
        """Fuse transcript and voice evidence without hiding strong conflicts."""
        text_scores = self._analyze_text(text) if text.strip() else None
        if audio_scores is None and audio is not None:
            audio_scores = self._analyze_audio(audio, sample_rate)
        return self.combine_scores(text_scores=text_scores, audio_scores=audio_scores)

    def analyze_voice_window(self, audio, sample_rate=TARGET_SAMPLE_RATE):
        """Analyze one rolling audio window for low-latency affect feedback."""
        return self._analyze_audio(audio, sample_rate)

    def combine_scores(self, text_scores=None, audio_scores=None):
        """Fuse calibrated-at-runtime model scores, abstaining on conflicts."""

        sources = []
        if text_scores is not None:
            sources.append("text")
        if audio_scores is not None:
            sources.append("voice")

        if not sources:
            return self._result("uncertain", 0.0, sources)

        source_tops = {}
        for source, source_scores in (("text", text_scores), ("voice", audio_scores)):
            if not source_scores:
                continue
            source_label, source_confidence = max(
                source_scores.items(), key=lambda item: item[1]
            )
            source_tops[source] = (source_label, float(source_confidence))

        # Do not let averaging make contradictory, confident signals look
        # certain. The normal response style is safer until the user clarifies.
        confident_labels = {
            label
            for label, confidence in source_tops.values()
            if confidence >= FUSION_DISAGREEMENT_THRESHOLD
        }
        if len(confident_labels) > 1:
            scores = self._average_scores(text_scores, audio_scores)
            top_confidence = max(confidence for _, confidence in source_tops.values())
            return self._result(
                "uncertain",
                top_confidence,
                sources,
                scores=scores,
                source_scores={"text": text_scores, "voice": audio_scores},
                disagreement=True,
            )

        scores = self._average_scores(text_scores, audio_scores)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, confidence = ranked[0]
        margin = confidence - ranked[1][1]
        if confidence < AFFECT_CONFIDENCE_THRESHOLD or margin < 0.08:
            label = "uncertain"
        return self._result(
            label,
            confidence,
            sources,
            scores=scores,
            source_scores={"text": text_scores, "voice": audio_scores},
        )

    @staticmethod
    def _average_scores(text_scores, audio_scores):
        combined = {label: [] for label in AFFECT_LABELS}
        for source_scores in (text_scores, audio_scores):
            if source_scores is None:
                continue
            for label in combined:
                if label in source_scores:
                    combined[label].append(float(source_scores[label]))
        return {
            label: float(np.mean(values)) if values else 0.0
            for label, values in combined.items()
        }

    def analyze_file(self, text, audio_path, audio_scores=None):
        if audio_scores is not None:
            return self.analyze(text=text, audio_scores=audio_scores)
        try:
            from engines.audio_engine import load_audio_file

            audio, sample_rate = load_audio_file(
                audio_path,
                sample_rate=TARGET_SAMPLE_RATE,
                mono=True,
            )
        except Exception:
            audio = None
            sample_rate = TARGET_SAMPLE_RATE
        return self.analyze(text=text, audio=audio, sample_rate=sample_rate)

    @staticmethod
    def _result(
        label,
        confidence,
        sources,
        scores=None,
        source_scores=None,
        disagreement=False,
    ):
        response_style = RESPONSE_STYLES.get(label, "standard")
        adapted = response_style != "standard" and label != "uncertain"
        return {
            "label": label,
            "confidence": round(float(confidence), 3),
            "source": sources,
            "response_style": response_style,
            "adapted": adapted,
            "scores": {key: round(float(value), 4) for key, value in (scores or {}).items()},
            "source_scores": {
                source: {
                    key: round(float(value), 4)
                    for key, value in values.items()
                }
                for source, values in (source_scores or {}).items()
                if values is not None
            },
            "disagreement": disagreement,
        }

    def _load_text_model(self):
        if self._text_model is not None:
            return
        with self._model_lock:
            if self._text_model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(AFFECT_TEXT_MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(AFFECT_TEXT_MODEL_NAME)
                self._text_tokenizer = tokenizer
                self._text_model = model.to("cpu").eval()

    def _analyze_text(self, text):
        try:
            self._load_text_model()
            inputs = self._text_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            with torch.inference_mode():
                logits = self._text_model(**inputs).logits[0]
                probabilities = torch.sigmoid(logits).cpu().tolist()
            labels = self._text_model.config.id2label
            raw = {
                str(labels.get(index, labels.get(str(index), ""))).lower(): float(score)
                for index, score in enumerate(probabilities)
            }
            return {
                category: max((raw.get(name, 0.0) for name in names), default=0.0)
                for category, names in TEXT_LABELS.items()
                if names
            }
        except Exception:
            # Chat still works if an optional classifier cannot be downloaded
            # or loaded; it simply uses EVA's normal response style.
            return None

    def _load_audio_model(self):
        if self._audio_model is not None:
            return
        with self._model_lock:
            if self._audio_model is None:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                processor = AutoFeatureExtractor.from_pretrained(AFFECT_AUDIO_MODEL_NAME)
                model = AutoModelForAudioClassification.from_pretrained(AFFECT_AUDIO_MODEL_NAME)
                self._audio_processor = processor
                self._audio_model = model.to("cpu").eval()

    def _analyze_audio(self, audio, sample_rate):
        try:
            self._load_audio_model()
            waveform = np.asarray(audio, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            if sample_rate != TARGET_SAMPLE_RATE:
                import librosa

                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=TARGET_SAMPLE_RATE,
                )
            if waveform.size < TARGET_SAMPLE_RATE // 2:
                return None

            # Analyze overlapping short windows so brief changes in delivery
            # aren't diluted inside one long recording. The final reply is
            # still generated at the end of the user's turn.
            window_size = round(TARGET_SAMPLE_RATE * AUDIO_WINDOW_SECONDS)
            hop_size = round(TARGET_SAMPLE_RATE * AUDIO_HOP_SECONDS)
            if waveform.size <= window_size:
                chunks = [waveform]
            else:
                last_start = waveform.size - window_size
                starts = list(range(0, last_start + 1, hop_size))
                if starts[-1] != last_start:
                    starts.append(last_start)
                chunks = [waveform[start:start + window_size] for start in starts]

            smoothed_probabilities = None
            smoothed_windows = []
            for chunk in chunks:
                inputs = self._audio_processor(
                    chunk,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                with torch.inference_mode():
                    logits = self._audio_model(**inputs).logits[0]
                    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                if smoothed_probabilities is None:
                    smoothed_probabilities = probabilities
                else:
                    smoothed_probabilities = (
                        (1.0 - AUDIO_SMOOTHING_ALPHA) * smoothed_probabilities
                        + AUDIO_SMOOTHING_ALPHA * probabilities
                    )
                smoothed_windows.append(smoothed_probabilities.copy())
            turn_probabilities = np.mean(smoothed_windows, axis=0)
            id2label = self._audio_model.config.id2label
            mapped = {}
            for index, score in enumerate(turn_probabilities):
                raw_label = str(id2label.get(index, id2label.get(str(index), ""))).lower()
                category = AUDIO_LABELS.get(raw_label.strip().lower())
                if category:
                    mapped[category] = max(mapped.get(category, 0.0), float(score))
            return mapped or None
        except Exception:
            return None
