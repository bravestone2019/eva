import torch
import torch.nn as nn


class EmotionModel(nn.Module):

    def __init__(
        self,
        input_dim=768,
        num_emotions=7
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2)
        )

        # ----------------------------------------------
        # Dimensional emotion
        # ----------------------------------------------

        self.valence_head = nn.Linear(
            256,
            1
        )

        self.arousal_head = nn.Linear(
            256,
            1
        )

        self.dominance_head = nn.Linear(
            256,
            1
        )

        # ----------------------------------------------
        # Additional dimensions
        # ----------------------------------------------

        self.stress_head = nn.Linear(
            256,
            1
        )

        self.confidence_head = nn.Linear(
            256,
            1
        )

        self.intensity_head = nn.Linear(
            256,
            1
        )

        # ----------------------------------------------
        # Categorical emotion
        # ----------------------------------------------

        self.emotion_head = nn.Linear(
            256,
            num_emotions
        )

    def forward(
        self,
        x
    ):

        features = self.encoder(x)

        valence = torch.tanh(
            self.valence_head(features)
        )

        arousal = torch.sigmoid(
            self.arousal_head(features)
        )

        dominance = torch.sigmoid(
            self.dominance_head(features)
        )

        stress = torch.sigmoid(
            self.stress_head(features)
        )

        confidence = torch.sigmoid(
            self.confidence_head(features)
        )

        intensity = torch.sigmoid(
            self.intensity_head(features)
        )

        emotion_logits = self.emotion_head(
            features
        )

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "stress": stress,
            "confidence": confidence,
            "intensity": intensity,
            "emotion_logits": emotion_logits
        }