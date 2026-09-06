from collections import deque
from datetime import datetime


class EmotionMemory:

    def __init__(
        self,
        max_history=50
    ):

        self.history = deque(
            maxlen=max_history
        )

    def add(
        self,
        emotion_state
    ):

        record = {

            "timestamp":
            datetime.now().isoformat(),

            "valence":
            emotion_state.valence,

            "arousal":
            emotion_state.arousal,

            "dominance":
            emotion_state.dominance,

            "stress":
            emotion_state.stress,

            "confidence":
            emotion_state.confidence,

            "intensity":
            emotion_state.intensity,

            "emotion":
            emotion_state.primary_emotion
        }

        self.history.append(
            record
        )

        return record

    def get_recent(
        self,
        n=5
    ):

        return list(
            self.history
        )[-n:]

    def calculate_trend(
        self,
        n=5
    ):

        records = self.get_recent(n)

        if len(records) < 2:

            return {
                "status":
                "insufficient_data"
            }

        first = records[0]

        last = records[-1]

        return {

            "valence_change":
            last["valence"]
            -
            first["valence"],

            "arousal_change":
            last["arousal"]
            -
            first["arousal"],

            "stress_change":
            last["stress"]
            -
            first["stress"],

            "intensity_change":
            last["intensity"]
            -
            first["intensity"]
        }