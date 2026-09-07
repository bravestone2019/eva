"""HTTP API for Flutter or any other client.

Start with: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
