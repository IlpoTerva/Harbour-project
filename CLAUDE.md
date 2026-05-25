# Harbour Agent — CLAUDE.md

## Project Overview

Harbour Agent is an automated harbour gate system for controlling truck access. When a truck arrives, the system captures an image, detects and reads the license plate (vision pipeline), cross-references the plate against a SQLite vehicle manifest, and conducts a spoken voice exchange with the driver to verify identity (audio pipeline). Based on confidence and database lookup results, it either grants access with dock instructions, requests manual verification, or denies entry.

**All ML processing runs on the Jetson Orin NX.** The laptop connects to the Jetson over HTTP — no ML packages are required on the laptop. No cloud APIs or internet connectivity are required or permitted at runtime.

**Target hardware:** NVIDIA Jetson Orin NX (server) + laptop (client)  
**Python version:** 3.10

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| License plate detection | YOLOv11 (ultralytics) |
| License plate OCR | Fast-Plate-OCR |
| Speech-to-text | Faster-Whisper (CTranslate2) |
| Language model | Llama 3.2 1B Instruct (llama-cpp-python, GGUF Q4_K_M) |
| Text-to-speech | Piper TTS |
| Database | SQLite |
| Server framework | FastAPI + uvicorn |
| GUI | Tkinter |
| Config | YAML (`utils/config.yaml`) |

---

## Architecture

### Deployment topology

```
Jetson Orin NX (server)              Laptop (client)
────────────────────────             ────────────────────────
scripts/api_server.py                UI/remote_frontend.py
  ├─ VisionPipeline (ONNX)      ←→   UI/client.py (HarbourClient)
  ├─ Listener (Whisper)              UI/remote_orchestrator.py
  ├─ Speaker (Piper TTS)               ├─ audio I/O via sounddevice
  ├─ LanguageModel (Llama 3.2)         └─ gate-flow logic (3 flows)
  └─ SQLite DB
```

All ML inference and database lookups happen on the Jetson. Audio I/O (recording and playback via `sounddevice`) happens on the laptop.

### Three-Flow Decision Logic (`RemoteOrchestrator.run_automated_entry`)

```
Flow A — Vision confident (conf ≥ 0.5) + plate found in DB
  → Ask driver to confirm name → grant access or deny

Flow B — Vision confident + plate NOT in DB
  → Ask driver to confirm plate reading
    → Confirmed: alert gate worker
    → Denied: fall back to audio plate request → name verify

Flow C — Vision low-confidence (< 0.5) or no plate detected
  → Skip to audio plate request → database lookup → name verify
```

### Four-Layer Plate Extraction (`LanguageModel.parse_plate_from_transcription`)

1. Direct regex match on raw Whisper transcription
2. NATO phonetic alphabet word-by-word conversion (deterministic)
3. Llama 3.2 1B NLU for natural-language input (temperature=0.1)
4. Regex fallback on longest alphanumeric run

---

## Main Classes

### `scripts/api_server.py`

FastAPI server that runs on the Jetson. Loads all ML models and the SQLite database at startup (as module-level singletons). Exposes the following REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model/DB load status |
| `/vision/detect` | POST | Upload image → plate, conf, bbox, annotated image (base64 JPEG) |
| `/tts/synthesize` | POST | Text → WAV audio bytes |
| `/stt/transcribe` | POST | WAV audio → Whisper transcription |
| `/llm/parse_plate` | POST | Transcription → extracted plate string |
| `/llm/parse_yes_no` | POST | Transcription → bool |
| `/llm/extract_name` | POST | Transcription → name string |
| `/llm/verify_name` | POST | db_name + spoken_name → bool |
| `/db/lookup` | POST | plate → DB entry or null |

Start with: `./start_server.sh` (uses `harbourenv` conda env; wraps `uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 10`)

---

### `UI/client.py`

**`HarbourClient`** — Thin synchronous HTTP wrapper around every server endpoint. All calls use `requests`. Audio encoding/decoding helpers (`_float32_to_wav`, `_wav_to_float32`) convert between numpy float32 arrays and WAV bytes for the STT and TTS endpoints.

---

### `UI/remote_orchestrator.py`

**`RemoteOrchestrator`** — Mirrors the old `Orchestrator` interface but delegates all ML and DB work to the Jetson via `HarbourClient`. Audio I/O (recording + playback via `sounddevice`) runs locally on the laptop. The single public entry point is `run_automated_entry(image)` which returns `(result_dict, vision_output)`. Implements all three flows. Result statuses: `"success"`, `"alert_worker"`, `"name_mismatch"`. Accepts an optional `on_vision_result` callback invoked right after plate detection (used by `RemoteGUI` to update the image display immediately).

---

### `UI/remote_frontend.py`

**`RemoteGUI`** — Tkinter laptop frontend. Functionally identical to the old `UI/GUI.py` but carries no ML imports. Two-column layout: annotated image (left) + live log (right). Runs `RemoteOrchestrator` in a background thread. Thread-safe helpers: `log()`, `set_status()`, `set_button_enabled()`. Includes image file import dialog.

Run with: `python UI/remote_frontend.py --host http://<jetson-ip>:8000`

---

### `scripts/vision_pipeline.py`

**`VisionPipeline`** — License plate detection and OCR. Uses YOLOv11 for bounding-box detection and Fast-Plate-OCR (`cct-s-v2-global-model`) for character recognition. Supports ONNX model variant (preferred on Jetson). Main method: `read_plate(image)` returns a dict with `plate`, `conf`, `bbox`, `visual` (annotated image), or `None` if no plate is found. `_select_best_box()` picks the highest-confidence detection when multiple plates are visible.

---

### `scripts/audio_pipeline.py`

**`Listener`** — Transcribes audio with Faster-Whisper. `listen(duration)` records from microphone; `transcribe(audio)` accepts a numpy float32 array and returns a string.

**`Speaker`** — Synthesises speech with Piper TTS and plays back via `sounddevice`. On the server, `speak()` is not used — the `/tts/synthesize` endpoint returns WAV bytes instead.

**`LanguageModel`** — Llama 3.2 1B wrapper used exclusively for *understanding* driver speech. Key methods:
- `parse_plate_from_transcription(text)` — Four-layer extraction (see above)
- `parse_yes_no(text)` → bool — Binary yes/no classification
- `extract_name(text)` → str — Extracts a name in Title Case
- `verify_name_similarity(db_name, spoken_name)` → bool — Strict fuzzy match (last name must match; common English nicknames accepted for first name)

---

### `scripts/helpers.py`

Utility functions shared across server and tests:

- **`create_mock_db(db_path)`** — Seeds a SQLite database with six demo trucks (plate, driver_name, cargo, dock, arrival_window).
- **`read_config(path)`** — Loads `utils/config.yaml` and returns it as a dict.


---

## Configuration

All model paths and the database path are defined in `utils/config.yaml`:

```yaml
models:
  model_path: "models/license_plate_model.pt"
  model_path_onnx: "models/license_plate_model.onnx"
  whisper_model_path: "models/whisper_small/"
  fast_ocr_model_path: "models/cct_s_v2_global_model_config.yaml"
  piper_model_path: "models/en_US-hfc_female-medium.onnx"
  llm_model_path: "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
database:
  db_path: "utils/license_plates_database.db"
images:
  path: "database_imgs/"
```

---

## Entry Points

- **Jetson server:** `./start_server.sh` (launches uvicorn via the `harbourenv` conda environment with graceful shutdown)
- **Laptop GUI:** `python UI/remote_frontend.py --host http://<jetson-ip>:8000`
- **Testing/debug:** `python tests/CLI.py` — CLI with multi-mode testing (runs locally, requires ML models)
- **Tests:** `pytest tests/` — ML and hardware dependencies are mocked via `sys.modules` injection in `tests/conftest.py`; a real temporary SQLite database is used for data-access tests

---

## Deployment Notes (Jetson Orin NX)

- USB microphone and USB speaker are likely required (onboard audio may not work)
- The server uses ONNX mode by default (`model_path_onnx`) for better GPU performance; falls back to PyTorch automatically if ONNX init fails
- SQLite database uses `check_same_thread=False` for concurrent request handling
- Laptop dependencies (no ML packages required): `requests sounddevice numpy opencv-python Pillow`
