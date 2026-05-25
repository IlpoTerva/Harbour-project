# Harbour Agent: Multimodal Edge AI Assistant

> **High-performance, real-time multimodal AI agent designed specifically for edge deployment.**

Optimized for the **NVIDIA Jetson Orin NX**, this project integrates state-of-the-art vision, speech, and reasoning models to interact with its environment autonomously.

## Project Overview

The Harbour Agent acts as an intelligent observer and interlocutor. By combining local Large Language Models (LLMs) with high-speed vision and audio pipelines, it can "see" objects and license plates, "hear" verbal commands, and "speak" back to the user with minimal latency.

The primary objective is to fully automate the harbor gate entry process. By autonomously scanning license plates, verifying driver identities via voice, and instantly assigning cargo unloading docks, the system drastically reduces the need for manual checkpoints and keeps logistics moving smoothly.

The system supports multilingual driver interactions in **English, Spanish, French, and Portuguese** — language is auto-detected from the driver's first response and all subsequent prompts switch accordingly.

---

## Core Technology Stack

> **The project leverages a specialized stack chosen for the balance between accuracy and edge-device throughput:**

### Computer Vision
- **YOLOv11** (ultralytics): Real-time license plate bounding-box detection. Runs in ONNX mode on the Jetson for optimal GPU performance; falls back to PyTorch automatically.
- **Fast-Plate-OCR** (`cct-s-v2-global-model`): Optimized optical character recognition specifically for vehicle identification.

### Language & Reasoning
- **Llama 3.2 1B Instruct** (llama-cpp-python, GGUF Q4_K_M): A compact but powerful LLM running locally with full CUDA acceleration. Used exclusively for understanding driver speech — plate parsing, yes/no classification, name extraction, and fuzzy name matching.

### Audio Pipeline
- **Faster-Whisper** (CTranslate2): Ultra-fast Speech-to-Text with automatic language detection (ISO 639-1). Returns both the transcription and the detected language code.
- **Piper TTS**: Fast, local neural text-to-speech with per-language voice models (EN, ES, FR, PT). Models are lazy-loaded on first use and cached; adding a new language requires only a config entry.

### Infrastructure
- **FastAPI + uvicorn**: REST server running on the Jetson — all ML inference and DB lookups stay on-device.
- **SQLite**: Vehicle manifest database.
- **Tkinter**: Laptop-side GUI; no ML packages required on the client.

---

## Environment & Prerequisites

To achieve maximum throughput and avoid legacy dependency conflicts, the Harbour Agent is built for modern environments.

**Python:** `>= 3.10`

**Edge Deployment (NVIDIA Jetson):**
- **OS:** Ubuntu 22.04 LTS
- **Drivers:** JetPack 6.2+ (provides native Python 3.10 support and CUDA 12.x drivers)

**Local Development & Testing:**
- **OS:** Windows 10/11 or standard Linux desktop
- **Drivers:** Standard NVIDIA Display Drivers and CUDA Toolkit 12.x (if testing with a discrete GPU)

**Laptop client dependencies** (no ML packages required):
```
requests sounddevice numpy opencv-python Pillow
```

---

## Architecture

### Deployment Topology

```
Jetson Orin NX (server)              Laptop (client)
────────────────────────             ────────────────────────
scripts/api_server.py                UI/remote_frontend.py
  ├─ VisionPipeline (ONNX)      ←→   UI/client.py (HarbourClient)
  ├─ Listener (Whisper + lang)       UI/remote_orchestrator.py
  ├─ Speaker (Piper, multi-lang)       ├─ VAD recording (sounddevice)
  ├─ LanguageModel (Llama 3.2)         ├─ language auto-switch
  └─ SQLite DB                         └─ gate-flow logic (3 flows)
                                     UI/i18n.py (prompt translations)
```

All ML inference and database lookups happen on the Jetson. Audio I/O (VAD recording and playback via `sounddevice`) happens on the laptop.

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
2. NATO phonetic alphabet word-by-word conversion (deterministic; digit words for EN/ES/FR/PT included)
3. Llama 3.2 1B NLU for natural-language input (temperature=0.1)
4. Regex fallback on longest alphanumeric run

### Voice Activity Detection (`RemoteOrchestrator._record_vad`)

Recording uses energy-threshold VAD rather than a fixed duration:
- ~90 ms of speech above the RMS threshold triggers recording onset
- ~600 ms of silence after speech ends the recording
- 300 ms pre-roll buffer ensures speech onset is not clipped
- Hard cap of 15 s prevents runaway recordings

---

## Main Modules

### `scripts/api_server.py`

FastAPI server that runs on the Jetson. Loads all ML models and the SQLite database at startup as module-level singletons. Exposes the following REST endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns model/DB load status |
| `/vision/detect` | POST | Upload image → plate, conf, bbox, annotated image (base64 JPEG) |
| `/tts/synthesize` | POST | Text + language → WAV audio bytes |
| `/stt/transcribe` | POST | WAV audio → `{text, language}` (ISO 639-1 detected language) |
| `/llm/parse_plate` | POST | Transcription → extracted plate string |
| `/llm/parse_yes_no` | POST | Transcription → bool |
| `/llm/extract_name` | POST | Transcription → name string |
| `/llm/verify_name` | POST | db_name + spoken_name → bool |
| `/db/lookup` | POST | plate → DB entry or null |

Start with: `./start_server.sh` (uses `harbourenv` conda env; wraps `uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 10`)

---

### `UI/client.py`

**`HarbourClient`** — Thin synchronous HTTP wrapper around every server endpoint. All calls use `requests`. Audio encoding/decoding helpers (`_float32_to_wav`, `_wav_to_float32`) convert between numpy float32 arrays and WAV bytes for the STT and TTS endpoints. The `synthesize` method accepts an optional `language` parameter passed through to the Jetson.

---

### `UI/remote_orchestrator.py`

**`RemoteOrchestrator`** — Delegates all ML and DB work to the Jetson via `HarbourClient`. Audio I/O runs locally on the laptop. The single public entry point is `run_automated_entry(image)` which returns `(result_dict, vision_output)`.

Language is auto-detected from the first driver transcription and used for all subsequent TTS prompts within that session. Between sessions the language resets to the configured default (`"en"`).

Result statuses: `"success"`, `"alert_worker"`, `"name_mismatch"`. Accepts an optional `on_vision_result` callback invoked right after plate detection (used by `RemoteGUI` to update the image display immediately).

---

### `UI/remote_frontend.py`

**`RemoteGUI`** — Tkinter laptop frontend. No ML imports required. Two-column layout: annotated image (left) + live log (right). Runs `RemoteOrchestrator` in a background thread. Thread-safe helpers: `log()`, `set_status()`, `set_button_enabled()`. Includes image file import dialog.

Run with: `python UI/remote_frontend.py --host http://<jetson-ip>:8000`

---

### `UI/i18n.py`

Driver-facing spoken prompt translations. Supports **English (`en`), Spanish (`es`), French (`fr`), and Portuguese (`pt`)**. All user-facing strings (plate confirmation, name request, access granted/denied messages) are defined here. Adding a new language requires only adding its ISO 639-1 code to `SUPPORTED_LANGS` and providing translations in `_T` — no other file changes needed.

---

### `scripts/vision_pipeline.py`

**`VisionPipeline`** — License plate detection and OCR. Uses YOLOv11 for bounding-box detection and Fast-Plate-OCR (`cct-s-v2-global-model`) for character recognition. Supports ONNX model variant (preferred on Jetson). Main method: `read_plate(image)` returns a dict with `plate`, `conf`, `region`, `bbox`, `visual` (annotated image), or `None` if no plate is found. `_select_best_box()` picks the highest-confidence detection when multiple plates are visible.

---

### `scripts/audio_pipeline.py`

**`Listener`** — Transcribes audio with Faster-Whisper. `transcribe(audio)` accepts a numpy float32 array and returns `(text, detected_language_code)`.

**`Speaker`** — Synthesises speech with Piper TTS. Maintains a lazy registry of per-language `PiperVoice` models; models are loaded on first use and cached for subsequent calls. The default language model is pre-loaded at startup to avoid first-call latency. On the server, `synthesize_chunks()` feeds the `/tts/synthesize` endpoint; `speak()` is available for direct local playback.

**`LanguageModel`** — Llama 3.2 1B wrapper used exclusively for understanding driver speech. Key methods:
- `parse_plate_from_transcription(text)` — Four-layer extraction (see above)
- `parse_yes_no(text)` → bool — Binary yes/no classification (temperature=0.0)
- `extract_name(text)` → str — Extracts a name in Title Case
- `verify_name_similarity(db_name, spoken_name)` → bool — Strict fuzzy match (last name must match; common English nicknames accepted for first name)

---

### `scripts/helpers.py`

Utility functions shared across server and tests:

- **`create_mock_db(db_path)`** — Seeds a SQLite database with six demo trucks (plate, driver_name, cargo, dock, arrival_window).
- **`read_config(path)`** — Loads `utils/config.yaml` and returns it as a dict.

---

### `tests/CLI.py`

**`CLI`** — Command-line testing interface. Modes: full flow, vision only, audio only, batch (process a folder of images). Uses ANSI colours for readable terminal output. Requires a local `Orchestrator` instance (not the remote client).

---

## Configuration

All model paths, database path, and language settings are defined in `utils/config.yaml`:

```yaml
database:
  db_path: "utils/license_plates_database.db"
images:
  path: "database_imgs/"
speech:
  default_language: "en"
  piper_models:
    en: "models/piper_models/en/en_US-hfc_female-medium.onnx"
    es: "models/piper_models/es/es_ES-sharvard-medium.onnx"
    fr: "models/piper_models/fr/fr_FR-siwis-medium.onnx"
    pt: "models/piper_models/pt/pt_PT-tugao-medium.onnx"
vision:
  yolo:
    pt: "models/vision/license_plate_model.pt"
    onnx: "models/vision/license_plate_model.onnx"
  fast_ocr:
    model_path: "models/vision/cct_s_v2_global_model_config.yaml"
tts:
  whisper: "models/whisper_small/"
  llm: "models/LLM/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
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
- The server uses ONNX mode by default for better GPU performance; falls back to PyTorch automatically if ONNX init fails
- SQLite database uses `check_same_thread=False` for concurrent request handling
- Piper TTS models for each supported language must be placed under `models/piper_models/<lang>/` as configured in `config.yaml`
- The server handles SIGINT/SIGTERM gracefully; if shutdown hangs beyond 8 s, the process is force-killed to prevent stale uvicorn workers
