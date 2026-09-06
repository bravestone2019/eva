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

from config import (
    DEVICE,
    GEMMA_MODEL_NAME,
    WAVLM_MODEL_NAME,
    WHISPER_MODEL_NAME
)


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

        print("\nLoading Gemma...")

        quant_config = BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=torch.float16,

            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                GEMMA_MODEL_NAME
            )
        )

        self.llm = (
            AutoModelForCausalLM
            .from_pretrained(

                GEMMA_MODEL_NAME,

                quantization_config=quant_config,

                device_map="auto"
            )
        )

        print("Gemma loaded.")

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