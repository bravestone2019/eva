import os
from pathlib import Path

import torch


# ============================================================
# EVA V5 CONFIGURATION
# ============================================================

PROJECT_NAME = "EVA"
VERSION = "5.0"

# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# AUDIO
# ------------------------------------------------------------

TARGET_SAMPLE_RATE = 16000

HIGH_PASS_CUTOFF = 80

# ------------------------------------------------------------
# VAD
# ------------------------------------------------------------

VAD_THRESHOLD = 0.5

SPEECH_PAD_MS = 200

MIN_SPEECH_DURATION_MS = 250

MIN_SILENCE_DURATION_MS = 300

# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

WHISPER_BACKEND = os.getenv("EVA_WHISPER_BACKEND", "openai").strip().lower()
WHISPER_DEFAULT_MODEL = (
    "openai/whisper-large-v3-turbo"
    if WHISPER_BACKEND == "transformers"
    else "large-v3-turbo"
)
WHISPER_MODEL_NAME = os.getenv("EVA_WHISPER_MODEL", WHISPER_DEFAULT_MODEL)
WHISPER_LANGUAGE = os.getenv("EVA_WHISPER_LANGUAGE", "auto").strip().lower()

WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# Optional affect classifiers, loaded only when emotion adaptation is enabled.
_LOCAL_TEXT_AFFECT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "eva-text-affect-go-emotions"
)
AFFECT_TEXT_MODEL_NAME = os.getenv(
    "EVA_AFFECT_TEXT_MODEL",
    str(_LOCAL_TEXT_AFFECT_MODEL)
    if (_LOCAL_TEXT_AFFECT_MODEL / "config.json").is_file()
    else "SamLowe/roberta-base-go_emotions",
)
_LOCAL_AUDIO_AFFECT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "eva-voice-affect-ravdess-cremad-8class"
)
_LEGACY_LOCAL_AUDIO_AFFECT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "eva-voice-affect-ravdess-cremad"
)
AFFECT_AUDIO_MODEL_NAME = os.getenv(
    "EVA_AFFECT_AUDIO_MODEL",
    str(_LOCAL_AUDIO_AFFECT_MODEL)
    if (_LOCAL_AUDIO_AFFECT_MODEL / "config.json").is_file()
    else str(_LEGACY_LOCAL_AUDIO_AFFECT_MODEL)
    if (_LEGACY_LOCAL_AUDIO_AFFECT_MODEL / "config.json").is_file()
    else "superb/wav2vec2-base-superb-er",
)
AFFECT_CONFIDENCE_THRESHOLD = float(
    os.getenv("EVA_AFFECT_CONFIDENCE_THRESHOLD", "0.55")
)

# Qwen is public (unlike some Gemma checkpoints) and runs well in Kaggle's
# free T4/P100 GPUs when loaded in 4-bit mode.  Override it with EVA_LLM_MODEL
# if you already have another Hugging Face model available.
LLM_MODEL_NAME = os.getenv("EVA_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# Used automatically when CUDA is unavailable.  This public, compact
# instruction model makes the application usable on CPU-only machines.
CPU_LLM_MODEL_NAME = os.getenv(
    "EVA_CPU_LLM_MODEL", "HuggingFaceTB/SmolLM2-360M-Instruct"
)

# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------

MAX_NEW_TOKENS = 300

TEMPERATURE = 0.7

# ------------------------------------------------------------
# MEMORY
# ------------------------------------------------------------

MAX_CONVERSATION_HISTORY = 10

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = "."

DATA_DIR = "data"

AUDIO_DIR = "data/audio"

DOCUMENT_DIR = "data/documents"

MEMORY_DIR = "data/memory"

CHECKPOINT_DIR = "checkpoints"

OUTPUT_DIR = "outputs"

OUTPUT_AUDIO_DIR = "outputs/audio"
