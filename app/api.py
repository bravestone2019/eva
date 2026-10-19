"""HTTP API for Flutter or any other client.

Start with: uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import numpy as np
import edge_tts
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from app.main import EVA
from app import account_store
from engines.affect_engine import AFFECT_LABELS, VOICE_EMOTION_LABELS
from engines.rag_engine import RAGEngine


logger = logging.getLogger(__name__)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="EVA Voice API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv(
        "EVA_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_eva: EVA | None = None
_rag_engine = RAGEngine()
_model_lock = Lock()
_inference_lock = Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    history: list[dict[str, str]] = Field(default_factory=list, max_length=10)
    adapt_emotion: bool = True


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class AccountRequest(BaseModel):
    email: str = Field(min_length=6, max_length=254)
    password: str = Field(min_length=6, max_length=1024)


class GoogleAccountRequest(BaseModel):
    credential: str = Field(min_length=100, max_length=10000)


class ConversationRequest(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(default="New conversation", max_length=120)
    updatedAt: int
    messages: list[dict[str, Any]] = Field(max_length=5000)


def require_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    scheme, _, token = (authorization or "").partition(" ")
    user = account_store.verify_token(token) if scheme.lower() == "bearer" else None
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to access saved conversations.")
    return user


@app.post("/v1/auth/register")
def register_account(request: AccountRequest) -> dict[str, Any]:
    try:
        user = account_store.create_user(request.email, request.password)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"user": user, "token": account_store.issue_token(user["id"])}


@app.post("/v1/auth/login")
def login_account(request: AccountRequest) -> dict[str, Any]:
    user = account_store.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    return {"user": user, "token": account_store.issue_token(user["id"])}


@app.post("/v1/auth/google")
def google_login(request: GoogleAccountRequest) -> dict[str, Any]:
    client_id = os.getenv("EVA_GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not configured on this EVA server.")
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
    except ImportError as error:
        raise HTTPException(status_code=503, detail="Install the google-auth package to enable Google sign-in.") from error
    try:
        claims = id_token.verify_oauth2_token(
            request.credential,
            google_requests.Request(),
            audience=client_id,
        )
    except ValueError as error:
        # Never log the ID token itself; the exception identifies common audience,
        # expiry, and signature issues without exposing a user's credential.
        verification_reason = str(error)
        logger.warning("Google ID token verification failed: %s", verification_reason)
        if "audience" in verification_reason.lower():
            detail = (
                "Google issued this sign-in for a different OAuth client ID. Make sure the "
                "frontend and backend use the same Web application client ID, then restart both servers."
            )
        elif "expired" in verification_reason.lower():
            detail = "This Google sign-in expired before EVA could verify it. Please try again."
        else:
            detail = (
                "Google could not verify this sign-in. Check the Web application OAuth client ID, "
                "the site's authorized JavaScript origin, your server connection, and restart both servers."
            )
        raise HTTPException(
            status_code=401,
            detail=detail,
        ) from error
    except Exception as error:
        logger.exception("Unexpected Google ID token verification failure (%s)", type(error).__name__)
        raise HTTPException(
            status_code=502,
            detail="Google sign-in verification is temporarily unavailable. Check your connection and try again.",
        ) from error
    if claims.get("email_verified") not in (True, "true"):
        raise HTTPException(status_code=401, detail="Google must verify your email address before sign-in.")
    try:
        user = account_store.google_user(claims.get("email", ""), claims.get("sub", ""))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"user": user, "token": account_store.issue_token(user["id"])}


@app.get("/v1/auth/me")
def current_account(user: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
    return {"user": user}


@app.get("/v1/conversations")
def get_conversations(user: dict[str, str] = Depends(require_user)) -> dict[str, Any]:
    return {"conversations": account_store.list_conversations(user["id"])}


@app.put("/v1/conversations/{conversation_id}")
def put_conversation(conversation_id: str, request: ConversationRequest,
                     user: dict[str, str] = Depends(require_user)) -> dict[str, bool]:
    if conversation_id != request.id:
        raise HTTPException(status_code=422, detail="Conversation ID does not match the URL.")
    try:
        serialize = getattr(request, "model_dump", None) or request.dict
        account_store.save_conversation(user["id"], serialize())
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"success": True}


@app.delete("/v1/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, user: dict[str, str] = Depends(require_user)) -> dict[str, bool]:
    if not account_store.delete_conversation(user["id"], conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return {"success": True}


def get_eva() -> EVA:
    """Load heavy models once, only when the first request arrives."""
    global _eva
    with _model_lock:
        if _eva is None:
            _eva = EVA()
            _eva.rag_engine = _rag_engine
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


@app.get("/v1/documents")
def list_documents() -> dict[str, Any]:
    """List private local text/Markdown files available to retrieval."""
    return {"documents": _rag_engine.list_documents()}


@app.post("/v1/documents")
async def add_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Add a small UTF-8 text/Markdown file to EVA's local knowledge store."""
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in {".txt", ".md"}:
        raise HTTPException(status_code=415, detail="Only .txt and .md knowledge files are supported.")
    contents = await file.read(2 * 1024 * 1024 + 1)
    if len(contents) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Knowledge files must be 2 MB or smaller.")
    try:
        text = contents.decode("utf-8-sig").strip()
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=415, detail="Knowledge files must use UTF-8 text encoding.") from error
    if not text:
        raise HTTPException(status_code=422, detail="The selected knowledge file is empty.")
    try:
        document = _rag_engine.save_document(filename, text)
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail="Could not save the knowledge file.") from error
    return {"success": True, "document": document}


@app.delete("/v1/documents/{document_id}")
def remove_document(document_id: str) -> dict[str, bool]:
    if not _rag_engine.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Knowledge file was not found.")
    return {"success": True}


@app.websocket("/v1/affect/stream")
async def stream_affect(websocket: WebSocket) -> None:
    """Analyze rolling voice windows and return privacy-minimal score updates."""
    await websocket.accept()
    try:
        settings = await websocket.receive_json()
        sample_rate = int(settings.get("sample_rate", 0))
        if not 8_000 <= sample_rate <= 96_000:
            await websocket.send_json({"type": "error", "message": "Unsupported microphone sample rate."})
            await websocket.close(code=1008)
            return

        window_samples = sample_rate * 2
        hop_samples = sample_rate
        rolling_audio = np.empty(0, dtype=np.float32)
        samples_since_analysis = 0
        smoothed_scores: dict[str, float] | None = None
        score_totals: dict[str, float] = {}
        score_updates = 0

        async def analyze_window(window: np.ndarray) -> dict[str, float] | None:
            def infer() -> dict[str, float] | None:
                with _inference_lock:
                    eva = get_eva()
                    return eva.affect_engine.analyze_voice_window(window, sample_rate)

            return await run_in_threadpool(infer)

        async def send_final() -> None:
            if not score_updates:
                await websocket.send_json({"type": "final", "scores": None, "estimate": None})
                return
            average = {key: value / score_updates for key, value in score_totals.items()}
            eva = get_eva()
            estimate = eva.affect_engine.combine_scores(audio_scores=average)
            await websocket.send_json({"type": "final", "scores": average, "estimate": estimate})

        while True:
            packet = await websocket.receive()
            if packet["type"] == "websocket.disconnect":
                break
            if packet.get("text"):
                try:
                    control = json.loads(packet["text"])
                except json.JSONDecodeError:
                    continue
                if control.get("type") == "finish":
                    await send_final()
                    await websocket.close()
                    break
                continue

            frame = packet.get("bytes")
            if not frame or len(frame) % 4:
                continue
            chunk = np.frombuffer(frame, dtype="<f4").astype(np.float32, copy=False)
            if not np.isfinite(chunk).all():
                continue
            rolling_audio = np.concatenate((rolling_audio, chunk))[-window_samples:]
            samples_since_analysis += len(chunk)
            if len(rolling_audio) < window_samples or samples_since_analysis < hop_samples:
                continue
            samples_since_analysis %= hop_samples
            if float(np.sqrt(np.mean(np.square(rolling_audio)))) < 0.008:
                continue

            current_scores = await analyze_window(rolling_audio.copy())
            if not current_scores:
                continue
            current_scores = {
                label: float(current_scores[label])
                for label in VOICE_EMOTION_LABELS
                if label in current_scores
            }
            if smoothed_scores is None:
                smoothed_scores = current_scores
            else:
                smoothed_scores = {
                    label: 0.65 * smoothed_scores.get(label, score) + 0.35 * score
                    for label, score in current_scores.items()
                }
            score_updates += 1
            for label, score in smoothed_scores.items():
                score_totals[label] = score_totals.get(label, 0.0) + score

            average = {key: value / score_updates for key, value in score_totals.items()}
            eva = get_eva()
            estimate = eva.affect_engine.combine_scores(audio_scores=average)
            await websocket.send_json({"type": "affect", "scores": average, "estimate": estimate})
    except WebSocketDisconnect:
        return
    except Exception as error:
        try:
            await websocket.send_json({"type": "error", "message": str(error)})
            await websocket.close(code=1011)
        except Exception:
            return


@app.post("/v1/voice")
async def process_voice(
    audio: UploadFile = File(...),
    adapt_emotion: bool = Form(True),
    history_json: str = Form(""),
) -> dict[str, Any]:
    """Accept a WAV/M4A/MP3 upload and return transcription plus EVA reply."""
    suffix = Path(audio.filename or "recording.wav").suffix or ".wav"
    if suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload a supported audio file.")

    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_file.write(await audio.read())
        input_path = temporary_file.name

    try:
        history = None
        if history_json:
            try:
                history = json.loads(history_json)
                if not isinstance(history, list) or len(history) > 10 or any(
                    not isinstance(item, dict) or not isinstance(item.get("user"), str) or not isinstance(item.get("assistant"), str)
                    for item in history
                ):
                    raise ValueError("Invalid history shape.")
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise HTTPException(status_code=422, detail="Invalid conversation history.") from error
        # The models and in-memory conversation history are shared by this
        # notebook process, so process one GPU request at a time.
        with _inference_lock:
            result = get_eva().process_voice(
                input_path,
                adapt_emotion=adapt_emotion,
                conversation_history=history,
            )
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
            affect = (
                eva.affect_engine.analyze(text=message)
                if request.adapt_emotion
                else {
                    "label": "disabled",
                    "confidence": 0.0,
                    "source": [],
                    "response_style": "standard",
                    "adapted": False,
                }
            )
            retrieved = eva.rag_engine.retrieve(message)
            rag_context = "\n\n".join(
                f"[Source: {document['metadata'].get('source', 'local document')}]\n{document['text']}"
                for document in retrieved
            )
            response = eva.llm_engine.generate(
                user_message=message,
                conversation_history=request.history,
                rag_context=rag_context,
                response_style=affect["response_style"],
            )
        return {
            "success": True,
            "message": message,
            "response": response,
            "affect": affect,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/v1/turn")
async def process_voice_turn(
    message: str = Form(..., min_length=1, max_length=10000),
    history_json: str = Form("[]"),
    adapt_emotion: bool = Form(True),
    live_affect_json: str = Form(""),
    audio: UploadFile = File(...),
) -> dict[str, Any]:
    """Analyze one transcript and its audio, then generate an adapted reply."""
    try:
        history = json.loads(history_json)
        request = ChatRequest(
            message=message,
            history=history,
            adapt_emotion=adapt_emotion,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail="Invalid conversation history.") from error

    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    if suffix.lower() not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        raise HTTPException(status_code=415, detail="Upload a supported audio file.")

    with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_file.write(await audio.read())
        input_path = temporary_file.name

    try:
        live_affect_scores = None
        if live_affect_json:
            try:
                parsed_scores = json.loads(live_affect_json)
                if isinstance(parsed_scores, dict):
                    allowed_labels = set(AFFECT_LABELS)
                    live_affect_scores = {
                        key: max(0.0, min(1.0, float(value)))
                        for key, value in parsed_scores.items()
                        if key in allowed_labels
                    }
            except (json.JSONDecodeError, TypeError, ValueError):
                live_affect_scores = None
        with _inference_lock:
            eva = get_eva()
            affect = (
                eva.affect_engine.analyze_file(
                    message,
                    input_path,
                    audio_scores=live_affect_scores,
                )
                if request.adapt_emotion
                else {
                    "label": "disabled",
                    "confidence": 0.0,
                    "source": [],
                    "response_style": "standard",
                    "adapted": False,
                }
            )
            retrieved = eva.rag_engine.retrieve(request.message.strip())
            rag_context = "\n\n".join(
                f"[Source: {document['metadata'].get('source', 'local document')}]\n{document['text']}"
                for document in retrieved
            )
            response = eva.llm_engine.generate(
                user_message=request.message.strip(),
                conversation_history=request.history,
                rag_context=rag_context,
                response_style=affect["response_style"],
            )
        return {
            "success": True,
            "message": request.message.strip(),
            "response": response,
            "affect": affect,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        Path(input_path).unlink(missing_ok=True)


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
