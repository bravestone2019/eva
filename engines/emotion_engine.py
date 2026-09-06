import torch

from emotion.features import (
    extract_acoustic_features
)


class EmotionEngine:

    def __init__(
        self,
        model_manager
    ):

        self.wavlm_model = (
            model_manager.wavlm_model
        )

        self.wavlm_processor = (
            model_manager.wavlm_processor
        )

        self.device = (
            model_manager.device
        )

        # EmotionModel will be connected
        # after training.
        self.trained_model = None

    @torch.no_grad()
    def extract_wavlm_embedding(
        self,
        audio
    ):

        inputs = self.wavlm_processor(

            audio,

            sampling_rate=16000,

            return_tensors="pt",

            padding=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.wavlm_model(
            **inputs
        )

        hidden = (
            outputs.last_hidden_state
        )

        embedding = hidden.mean(
            dim=1
        )

        return (
            embedding
            .squeeze(0)
        )

    def analyze(
        self,
        audio
    ):

        acoustic_features = (
            extract_acoustic_features(
                audio
            )
        )

        embedding = (
            self.extract_wavlm_embedding(
                audio
            )
        )

        result = {

            "status":
            "representation_only",

            "acoustic_features":
            acoustic_features,

            "wavlm_embedding":
            embedding.cpu().numpy()
        }

        # ------------------------------------------
        # Trained emotion model
        # ------------------------------------------

        if self.trained_model is not None:

            prediction = (
                self.predict(
                    embedding
                )
            )

            result["emotion"] = prediction

        else:

            result["emotion"] = None

        return result

    @torch.no_grad()
    def predict(
        self,
        embedding
    ):

        output = (
            self.trained_model(
                embedding.unsqueeze(0)
            )
        )

        emotion_logits = (
            output["emotion_logits"]
        )

        probabilities = torch.softmax(
            emotion_logits,
            dim=-1
        )

        emotion_id = (
            torch.argmax(
                probabilities,
                dim=-1
            )
        )

        return {

            "valence":
            float(
                output["valence"][0][0]
            ),

            "arousal":
            float(
                output["arousal"][0][0]
            ),

            "dominance":
            float(
                output["dominance"][0][0]
            ),

            "stress":
            float(
                output["stress"][0][0]
            ),

            "confidence":
            float(
                output["confidence"][0][0]
            ),

            "intensity":
            float(
                output["intensity"][0][0]
            ),

            "emotion_id":
            int(emotion_id[0]),

            "emotion_probability":
            float(
                probabilities[
                    0,
                    emotion_id[0]
                ]
            )
        }