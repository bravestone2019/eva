import torch
import librosa

from transformers import (
    AutoFeatureExtractor,
    AutoModel
)


class WavLMEncoder:

    def __init__(
        self,
        model_name="microsoft/wavlm-base-plus"
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Loading WavLM: {model_name}")
        print(f"Device: {self.device}")

        self.processor = (
            AutoFeatureExtractor
            .from_pretrained(model_name)
        )

        self.model = (
            AutoModel
            .from_pretrained(model_name)
            .to(self.device)
        )

        self.model.eval()

        print("WavLM loaded successfully.")

    @torch.no_grad()
    def extract_embedding(
        self,
        audio,
        sr=16000
    ):

        if sr != 16000:

            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000
            )

        audio = audio.astype("float32")

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        hidden = outputs.last_hidden_state

        embedding = hidden.mean(
            dim=1
        )

        return (
            embedding
            .squeeze(0)
            .cpu()
            .numpy()
        )