import os
from pathlib import Path

import certifi
import torch
import whisper

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoFeatureExtractor,
    AutoModel,
    BitsAndBytesConfig
)

from silero_vad import load_silero_vad

from app.config import (
    CPU_LLM_MODEL_NAME,
    DEVICE,
    LLM_MODEL_NAME,
    WAVLM_MODEL_NAME,
    WHISPER_MODEL_NAME
)


def configure_certificate_bundle():
    """Recover from a stale SSL_CERT_FILE setting in the launch shell.

    Hugging Face respects SSL_CERT_FILE.  When that variable points to a
    deleted file, HTTP clients cannot even create an SSL context.  Keep a
    valid custom certificate path intact, and otherwise use Certifi's bundle.
    """

    certificate_file = os.environ.get("SSL_CERT_FILE")

    if certificate_file and not Path(certificate_file).is_file():
        os.environ["SSL_CERT_FILE"] = certifi.where()


configure_certificate_bundle()


def can_load_gemma():
    """Return whether the 4B Gemma model has enough CUDA memory to load."""

    if not torch.cuda.is_available():
        return False

    minimum_vram_bytes = 6 * 1024**3
    available_vram_bytes = torch.cuda.get_device_properties(0).total_memory

    return available_vram_bytes >= minimum_vram_bytes


class ModelManager:

    def __init__(self):

        self.device = DEVICE

        self.llm = None
        self.tokenizer = None

        self.whisper_model = None

        self.wavlm_model = None
        self.wavlm_processor = None

        self.vad_model = None

    # ========================================================
    # GEMMA
    # ========================================================

    def load_llm(self):

        print("\nLoading language model...")

        if not can_load_gemma():

            if torch.cuda.is_available():
                fallback_reason = (
                    "The GPU has less than 6 GiB of VRAM"
                )
            else:
                fallback_reason = "CUDA is unavailable"

            print(
                f"{fallback_reason}; loading the lightweight model: "
                f"{CPU_LLM_MODEL_NAME}"
            )

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    CPU_LLM_MODEL_NAME
                )
            )

            self.llm = (
                AutoModelForCausalLM
                .from_pretrained(
                    CPU_LLM_MODEL_NAME,
                    torch_dtype=(
                        torch.float16
                        if torch.cuda.is_available()
                        else torch.float32
                    ),
                    low_cpu_mem_usage=True
                )
                .to(self.device)
            )

            self.llm.eval()

            print("Lightweight fallback model loaded.")
            return

        quant_config = BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=torch.float16,

            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

        self.llm = (
            AutoModelForCausalLM
            .from_pretrained(

                LLM_MODEL_NAME,

                quantization_config=quant_config,

                device_map="auto"
            )
        )

        print("Language model loaded.")

    # ========================================================
    # WHISPER
    # ========================================================

    def load_whisper(self):

        print("\nLoading Whisper...")

        self.whisper_model = (
            whisper.load_model(
                WHISPER_MODEL_NAME
            )
        )

        print("Whisper loaded.")

    # ========================================================
    # WAVLM
    # ========================================================

    def load_wavlm(self):

        print("\nLoading WavLM...")

        self.wavlm_processor = (
            AutoFeatureExtractor
            .from_pretrained(
                WAVLM_MODEL_NAME
            )
        )

        self.wavlm_model = (
            AutoModel
            .from_pretrained(
                WAVLM_MODEL_NAME
            )
            .to(self.device)
        )

        self.wavlm_model.eval()

        print("WavLM loaded.")

    # ========================================================
    # SILERO VAD
    # ========================================================

    def load_vad(self):

        print("\nLoading Silero VAD...")

        self.vad_model = load_silero_vad()

        print("Silero VAD loaded.")

    # ========================================================
    # ALL MODELS
    # ========================================================

    def load_all(self):

        self.load_llm()

        self.load_whisper()

        self.load_wavlm()

        self.load_vad()


model_manager = ModelManager()
