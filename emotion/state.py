from dataclasses import dataclass


@dataclass
class EmotionState:

    valence: float = 0.0

    arousal: float = 0.0

    dominance: float = 0.5

    stress: float = 0.0

    confidence: float = 0.5

    intensity: float = 0.0

    primary_emotion: str = "neutral"

    confidence_score: float = 0.0

    source: str = "unknown"


def create_neutral_emotion():

    return EmotionState(
        valence=0.0,
        arousal=0.2,
        dominance=0.5,
        stress=0.0,
        confidence=0.5,
        intensity=0.1,
        primary_emotion="neutral",
        confidence_score=0.0,
        source="default"
    )