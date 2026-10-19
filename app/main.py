import argparse

from models.model_manager import (
    model_manager
)

from engines.audio_engine import (
    AudioEngine
)

from engines.speech_engine import (
    SpeechEngine
)

from engines.affect_engine import (
    AffectEngine
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

from engines.tts_engine import (
    TTSEngine
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

        # The experimental EmotionModel has no trained checkpoint. The active
        # web flow uses AffectEngine's pretrained text/audio classifiers.
        self.emotion_engine = None

        self.affect_engine = AffectEngine()

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

        self.tts_engine = (
            TTSEngine()
        )

        print()
        print("EVA V5 READY")
        print("=" * 60)

    def process_voice(
        self,
        input_file,
        adapt_emotion=True,
        conversation_history=None,
    ):

        # ====================================================
        # 1. AUDIO PROCESSING
        # ====================================================

        processed_file = (
            "output/processed_audio.wav"
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

        # ====================================================
        # 3. SPEECH AUDIO
        # ====================================================

        # VAD can miss quiet speech, accents, or recordings with a noisy
        # microphone. Let Whisper inspect the complete recording in that case
        # instead of rejecting it before transcription is attempted.
        speech_audio = (
            self.speech_engine.extract_speech(audio, timestamps)
            if timestamps
            else audio
        )

        speech_file = (
            "output/speech_only.wav"
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
                speech_audio
            )
        )

        # A VAD-trimmed clip can also remove the first/last words. Retry the
        # original preprocessed recording before reporting a recognition
        # failure to the user.
        if not text.strip() and speech_audio is not audio:
            speech_audio = audio
            text = self.speech_engine.transcribe(speech_audio)

        if not text.strip():
            return {
                "success": False,
                "message": "I couldn't recognize any words. Move closer to the microphone, reduce background noise, and try speaking again.",
            }

        # ====================================================
        # 5. EMOTION
        # ====================================================

        affect_result = (
            self.affect_engine.analyze(
                text=text,
                audio=speech_audio
            )
            if adapt_emotion
            else self.affect_engine.analyze()
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
            f"[Source: {document['metadata'].get('source', 'local document')}]\n{document['text']}"
            for document in retrieved
        )

        # ====================================================
        # 7. MEMORY
        # ====================================================

        history = conversation_history if conversation_history is not None else self.memory_engine.get_history()

        # ====================================================
        # 8. LLM
        # ====================================================

        response = (
            self.llm_engine.generate(

                user_message=text,

                conversation_history=
                history,

                rag_context=
                rag_context,

                response_style=(
                    affect_result["response_style"]
                    if adapt_emotion
                    else "standard"
                )
            )
        )

        # ====================================================
        # 9. MEMORY UPDATE
        # ====================================================

        if conversation_history is None:
            self.memory_engine.add_conversation(text, response)

        return {

            "success":
            True,

            "transcription":
            text,

            "affect":
            affect_result,

            "response":
            response
        }


def main():

    parser = argparse.ArgumentParser(
        description="Run EVA with a microphone recording or WAV input file."
    )

    input_group = parser.add_mutually_exclusive_group(
        required=True
    )

    input_group.add_argument(
        "--record-seconds",
        type=float,
        help="Record from the default microphone for this many seconds."
    )

    input_group.add_argument(
        "--input-file",
        help="Path to an existing audio file."
    )

    args = parser.parse_args()

    eva = EVA()

    if args.record_seconds is not None:

        input_file = "output/recorded_audio.wav"

        print(
            f"\nRecording for {args.record_seconds:g} seconds. "
            "Speak now..."
        )

        eva.audio_engine.record_microphone(
            input_file,
            args.record_seconds
        )

        print("Recording complete. Processing audio...")

    else:
        input_file = args.input_file

    result = eva.process_voice(input_file)

    if not result["success"]:
        print(f"\nEVA: {result['message']}")
        return

    print(f"\nYou: {result['transcription']}")
    print(f"EVA: {result['response']}")

    print("\nEVA is speaking...")
    eva.tts_engine.speak(result["response"], play=True)


if __name__ == "__main__":

    main()
