"""Text-to-speech generation and playback for EVA responses."""

import asyncio
import os
from pathlib import Path

import edge_tts


class TTSEngine:

    def __init__(
        self,
        voice="en-US-AriaNeural"
    ):

        self.voice = voice

    async def _save_speech(
        self,
        text,
        output_file
    ):

        communicator = edge_tts.Communicate(
            text,
            self.voice
        )

        await communicator.save(output_file)

    def speak(
        self,
        text,
        output_file="output/eva_response.mp3",
        play=False
    ):

        if not text.strip():
            return None

        Path(output_file).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        asyncio.run(
            self._save_speech(
                text,
                output_file
            )
        )

        # A Kaggle/API server has no desktop audio player.  The caller can
        # return this file to Flutter, while the local CLI may opt in to play.
        if play and os.name == "nt":
            os.startfile(Path(output_file).resolve())

        return output_file
