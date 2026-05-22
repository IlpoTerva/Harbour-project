"""
FastAPI server for the Harbour Agent — runs on the Jetson Orin NX.

All ML inference (YOLOv11, Fast-Plate-OCR, Faster-Whisper, Piper TTS,
Llama 3.2 1B) and SQLite lookups happen here.  The laptop frontend calls
these endpoints; all processing stays strictly local and offline.

Start:
    uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000

New dependencies (add to requirements):
    fastapi
    uvicorn[standard]
    python-multipart
"""

#Shutting down gracefully on SIGINT is tricky with uvicorn
import signal, threading, os

def _force_exit_after(seconds: int = 8) -> None:
    """Called in a daemon thread — kills the process if shutdown hangs."""
    import time
    time.sleep(seconds)
    logger.warning(f"Shutdown still hanging after {seconds}s — forcing exit.")
    os._exit(1)   # bypasses Python cleanup entirely, kills all threads

def _handle_sigint(sig, frame):
    logger.info("SIGINT received — starting shutdown (forced exit in 8 s).")
    t = threading.Thread(target=_force_exit_after, args=(8,), daemon=True)
    t.start()
    raise SystemExit(0)   # lets uvicorn lifespan run normally

signal.signal(signal.SIGINT,  _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)



import base64
import io
import logging
import os
import sqlite3
import wave
from typing import Any, Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from scripts.audio_pipeline import LanguageModel, Listener, Speaker
from scripts.orchestrator import create_mock_db
from scripts.vision_pipeline import VisionPipeline, read_config




logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Harbour Agent API")

# Module-level singletons — loaded once at startup, reused for every request.
_config: Dict[str, Any] = {}
_vision: Optional[VisionPipeline] = None
_listener: Optional[Listener] = None
_speaker: Optional[Speaker] = None
_llm: Optional[LanguageModel] = None
_db: Optional[sqlite3.Connection] = None


def get_device() -> str:
    """Return 'cuda' if CUDAExecutionProvider is available, else 'cpu'."""
    import onnxruntime as ort
    providers = ort.get_available_providers()
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"

DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")

# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup() -> None:
    global _config, _vision, _listener, _speaker, _llm, _db
    _config = read_config("utils/config.yaml")
    db_path = _config["database"]["db_path"]
    if not os.path.exists(db_path):
        create_mock_db(db_path)
    try:
        _vision = VisionPipeline(config=_config, device=DEVICE, onnx=True)
        logger.info("VisionPipeline loaded in ONNX mode.")
    except Exception as e:
        logger.warning(f"ONNX init failed ({e}), falling back to PyTorch mode.")
        _vision = VisionPipeline(config=_config,device=DEVICE, onnx=False)
    _listener = Listener(conf=_config, device=DEVICE)
    _speaker = Speaker(conf=_config, device=DEVICE)
    _llm = LanguageModel(conf=_config, device=DEVICE)
    _db = sqlite3.connect(db_path, check_same_thread=False)
    logger.info("All models loaded — server ready.")


# ── Audio codec helpers ───────────────────────────────────────────────────────

def _int16_to_wav(pcm: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_to_float32(wav_bytes: bytes) -> np.ndarray:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


# ── Request / response schemas ────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str

class ParsePlateRequest(BaseModel):
    transcription: str

class ParseYesNoRequest(BaseModel):
    transcription: str

class ExtractNameRequest(BaseModel):
    transcription: str

class VerifyNameRequest(BaseModel):
    db_name: str
    spoken_name: str

class LookupRequest(BaseModel):
    plate: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "models": {
            "vision":  _vision   is not None,
            "whisper": _listener is not None,
            "piper":   _speaker  is not None,
            "llm":     _llm      is not None,
            "db":      _db       is not None,
        },
    }


@app.post("/vision/detect")
async def detect_plate(image: UploadFile = File(...)) -> Optional[Dict[str, Any]]:
    """Detect and OCR a license plate in an uploaded JPEG/PNG image.

    Returns a JSON object with plate, conf, bbox, and the YOLO-annotated image
    encoded as base64 JPEG — or null if no plate was found.
    """
    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = _vision.read_plate(bgr)
    if result is None:
        return None

    _, jpg_bytes = cv2.imencode(".jpg", result["visual"])
    return {
        "plate":      result["plate"],
        "conf":       result["conf"],
        "bbox":       list(result["bbox"]),
        "visual_b64": base64.b64encode(jpg_bytes).decode(),
    }


@app.post("/tts/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    """Synthesise text with Piper TTS.

    Returns raw WAV audio bytes (audio/wav).  The laptop plays them locally.
    """
    chunks = [
        np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        for chunk in _speaker.pipeline.synthesize(req.text)
    ]
    if not chunks:
        raise HTTPException(status_code=500, detail="TTS produced no audio.")

    pcm = np.concatenate(chunks)
    wav = _int16_to_wav(pcm, _speaker.pipeline.config.sample_rate)
    return Response(content=wav, media_type="audio/wav")


@app.post("/stt/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> Dict[str, str]:
    """Transcribe a WAV audio file with Faster-Whisper.

    Expects mono 16 kHz audio (matching the laptop's recording settings).
    Returns {"text": "..."}.
    """
    wav_bytes = await audio.read()
    samples = _wav_to_float32(wav_bytes)
    text = _listener.transcribe(samples)
    return {"text": text}


@app.post("/llm/parse_plate")
def parse_plate(req: ParsePlateRequest) -> Dict[str, str]:
    plate = _llm.parse_plate_from_transcription(req.transcription)
    return {"plate": plate}


@app.post("/llm/parse_yes_no")
def parse_yes_no(req: ParseYesNoRequest) -> Dict[str, bool]:
    return {"result": _llm.parse_yes_no(req.transcription)}


@app.post("/llm/extract_name")
def extract_name(req: ExtractNameRequest) -> Dict[str, str]:
    return {"name": _llm.extract_name(req.transcription)}


@app.post("/llm/verify_name")
def verify_name(req: VerifyNameRequest) -> Dict[str, bool]:
    return {"match": _llm.verify_name_similarity(req.db_name, req.spoken_name)}


@app.post("/db/lookup")
def lookup_plate(req: LookupRequest) -> Dict[str, Any]:
    cursor = _db.cursor()
    cursor.execute(
        "SELECT plate, driver_name, cargo, dock, arrival_window "
        "FROM license_plates WHERE plate = ?",
        (req.plate,),
    )
    row = cursor.fetchone()
    if row is None:
        return {"found": False, "entry": None}
    return {
        "found": True,
        "entry": {
            "plate":          row[0],
            "driver_name":    row[1],
            "cargo":          row[2],
            "dock":           row[3],
            "arrival_window": row[4],
        },
    }
