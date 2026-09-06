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

WHISPER_MODEL_NAME = "large-v3-turbo"

WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

GEMMA_MODEL_NAME = "google/gemma-3-4b-it"

# Used automatically when CUDA is unavailable.  This public, compact
# instruction model makes the application usable on CPU-only machines.
CPU_LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

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
