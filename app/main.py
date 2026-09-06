from models.model_manager import (
    model_manager
)

from engines.audio_engine import (
    AudioEngine
)

from engines.speech_engine import (
    SpeechEngine
)

from engines.emotion_engine import (
    EmotionEngine
)

from engines.memory_engine import (
    MemoryEngine
)

from engines.llm_engine import (
    LLMEngine
)

from engines.rag_engine import (
    RAGEngine
)


class EVA:

    def __init__(self):

        print()
        print("=" * 60)
        print("INITIALIZING EVA V5")
        print("=" * 60)

        # --------------------------------------------
        # Models
        # --------------------------------------------

        model_manager.load_all()

        # --------------------------------------------
        # Engines
        # --------------------------------------------

        self.audio_engine = (
            AudioEngine()
        )

        self.speech_engine = (
            SpeechEngine(
                model_manager
            )
        )

        self.emotion_engine = (
            EmotionEngine(
                model_manager
            )
        )

        self.memory_engine = (
            MemoryEngine()
        )

        self.llm_engine = (
            LLMEngine(
                model_manager
            )
        )

        self.rag_engine = (
            RAGEngine()
        )

        print()
        print("EVA V5 READY")
        print("=" * 60)

    def process_voice(
        self,
        input_file
    ):

        # ====================================================
        # 1. AUDIO PROCESSING
        # ====================================================

        processed_file = (
            "outputs/processed_audio.wav"
        )

        audio = (
            self.audio_engine.preprocess(

                input_file,

                processed_file
            )
        )

        # ====================================================
        # 2. VAD
        # ====================================================

        timestamps = (
            self.speech_engine.detect_speech(
                audio
            )
        )

        if not timestamps:

            return {

                "success":
                False,

                "message":
                "No speech detected."
            }

        # ====================================================
        # 3. SPEECH AUDIO
        # ====================================================

        speech_audio = (
            self.speech_engine.extract_speech(

                audio,

                timestamps
            )
        )

        speech_file = (
            "outputs/speech_only.wav"
        )

        self.speech_engine.save_speech(

            speech_audio,

            speech_file
        )

        # ====================================================
        # 4. WHISPER
        # ====================================================

        text = (
            self.speech_engine.transcribe(
                speech_file
            )
        )

        # ====================================================
        # 5. EMOTION
        # ====================================================

        emotion_result = (
            self.emotion_engine.analyze(
                speech_audio
            )
        )

        # ====================================================
        # 6. RAG
        # ====================================================

        retrieved = (
            self.rag_engine.retrieve(
                text
            )
        )

        rag_context = "\n\n".join(

            document["text"]

            for document in retrieved
        )

        # ====================================================
        # 7. MEMORY
        # ====================================================

        history = (
            self.memory_engine.get_history()
        )

        # ====================================================
        # 8. LLM
        # ====================================================

        response = (
            self.llm_engine.generate(

                user_message=text,

                emotion_state=
                emotion_result.get(
                    "emotion"
                ),

                conversation_history=
                history,

                rag_context=
                rag_context
            )
        )

        # ====================================================
        # 9. MEMORY UPDATE
        # ====================================================

        self.memory_engine.add_conversation(

            text,

            response
        )

        return {

            "success":
            True,

            "transcription":
            text,

            "emotion":
            emotion_result,

            "response":
            response
        }


if __name__ == "__main__":

    eva = EVA()

    print("\nEVA initialized successfully.")