import base64
import cv2
import numpy as np
import io
import wave
import requests
from typing import Any, Dict, Optional, Tuple


_SAMPLE_RATE = 16_000  # Hz — must match the server's Whisper input expectation

# ── Audio codec helpers ───────────────────────────────────────────────────────

def _float32_to_wav(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Encode a float32 numpy array as a mono WAV byte string for upload."""
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _wav_to_float32(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes (from the TTS endpoint) to (float32 array, sample_rate)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate
# ── HTTP client ───────────────────────────────────────────────────────────────

class HarbourClient:
    """
    Thin wrapper around the Jetson API endpoints defined in scripts/api_server.py.

    All network calls are synchronous; the caller is responsible for running
    them off the main thread when needed (RemoteOrchestrator is always called
    from RemoteGUI's background worker thread).
    """

    def __init__(self, base_url: str, timeout: int = 120) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        return requests.get(f"{self.base}/health", timeout=10).json()

    def detect_plate(self, image_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """Upload a BGR image and return the detection result, or None."""
        _, jpg = cv2.imencode(".jpg", image_bgr)
        resp = requests.post(
            f"{self.base}/vision/detect",
            files={"image": ("image.jpg", jpg.tobytes(), "image/jpeg")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if data is None:
            return None
        jpg_bytes = base64.b64decode(data["visual_b64"])
        arr = np.frombuffer(jpg_bytes, np.uint8)
        visual = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return {
            "plate":  data["plate"],
            "conf":   data["conf"],
            "bbox":   tuple(data["bbox"]),
            "visual": visual,
        }

    def synthesize(self, text: str, language: str = "en") -> Tuple[np.ndarray, int]:
        """Request TTS synthesis in the given language. Returns (float32 audio array, sample_rate)."""
        resp = requests.post(
            f"{self.base}/tts/synthesize",
            json={"text": text, "language": language},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return _wav_to_float32(resp.content)

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        """Upload recorded audio. Returns (transcription_text, detected_language_code)."""
        wav_bytes = _float32_to_wav(audio)
        resp = requests.post(
            f"{self.base}/stt/transcribe",
            files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["text"], data.get("language", "en")

    def parse_plate(self, transcription: str) -> str:
        resp = requests.post(
            f"{self.base}/llm/parse_plate",
            json={"transcription": transcription},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["plate"]

    def parse_yes_no(self, transcription: str) -> bool:
        resp = requests.post(
            f"{self.base}/llm/parse_yes_no",
            json={"transcription": transcription},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def extract_name(self, transcription: str) -> str:
        resp = requests.post(
            f"{self.base}/llm/extract_name",
            json={"transcription": transcription},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["name"]

    def verify_name(self, db_name: str, spoken_name: str) -> bool:
        resp = requests.post(
            f"{self.base}/llm/verify_name",
            json={"db_name": db_name, "spoken_name": spoken_name},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["match"]

    def lookup_plate(self, plate: str) -> Optional[Dict[str, Any]]:
        resp = requests.post(
            f"{self.base}/db/lookup",
            json={"plate": plate},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["entry"] if data["found"] else None