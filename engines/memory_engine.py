from config import (
    MAX_CONVERSATION_HISTORY
)


class MemoryEngine:

    def __init__(self):

        self.conversation_history = []

        self.emotion_history = []

    def add_conversation(
        self,
        user_message,
        assistant_message
    ):

        self.conversation_history.append({

            "user":
            user_message,

            "assistant":
            assistant_message
        })

        self.conversation_history = (
            self.conversation_history[
                -MAX_CONVERSATION_HISTORY:
            ]
        )

    def get_history(self):

        return self.conversation_history

    def add_emotion(
        self,
        emotion_state
    ):

        self.emotion_history.append(
            emotion_state
        )

    def get_emotion_history(self):

        return self.emotion_history