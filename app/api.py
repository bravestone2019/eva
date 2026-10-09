"""HTTP API for Flutter or any other client.

Start with: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import numpy as np
import edge_tts
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import Response

from app.main import EVA


app = FastAPI(title="EVA Voice API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your Flutter app's domain in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_eva: EVA | None = None
_model_lock = Lock()
_inference_lock = Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


def get_eva() -> EVA:
    """Load heavy models once, only when the first request arrives."""
    global _eva
    with _model_lock:
        if _eva is None:
            _eva = EVA()
    return _eva


def json_safe(value: Any) -> Any:
    """Convert NumPy/Torch-compatible result values to JSON primitives."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        # WavLM embeddings are large and not useful to the mobile client.
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/voice")
async def process_voice(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a WAV/M4A/MP3 upload and return transcription plus EVA reply."""
    suffix = Path(audio.filename or "recording.wav").suffix or ".wav"
    if suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload a supported audio file.")

    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_file.write(await audio.read())
        input_path = temporary_file.name

    try:
        # The models and in-memory conversation history are shared by this
        # notebook process, so process one GPU request at a time.
        with _inference_lock:
            result = get_eva().process_voice(input_path)
        if result.get("emotion"):
            # This 768-value internal feature vector is not an API result and
            # wastes mobile bandwidth.
            result["emotion"].pop("wavlm_embedding", None)
        return json_safe(result)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        Path(input_path).unlink(missing_ok=True)


@app.post("/v1/chat")
def process_chat(request: ChatRequest) -> dict[str, Any]:
    """Generate a text reply using the conversation supplied by the client."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    try:
        with _inference_lock:
            eva = get_eva()
            retrieved = eva.rag_engine.retrieve(message)
            rag_context = "\n\n".join(
                document["text"] for document in retrieved
            )
            response = eva.llm_engine.generate(
                user_message=message,
                conversation_history=request.history,
                rag_context=rag_context,
            )
        return {
            "success": True,
            "message": message,
            "response": response,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/v1/tts")
async def synthesize_speech(request: SpeechRequest) -> Response:
    """Return a spoken response using EVA's feminine Aria Neural voice."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be blank.")

    try:
        communicator = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        audio = bytearray()
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        if not audio:
            raise RuntimeError("Speech service returned no audio.")
        return Response(content=bytes(audio), media_type="audio/mpeg")
    except Exception as error:
        raise HTTPException(status_code=502, detail="Speech generation is unavailable.") from error
