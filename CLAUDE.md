# Harbour Agent — CLAUDE.md

## Project Overview

Harbour Agent is an automated harbour gate system for controlling truck access. When a truck arrives, the system captures an image, detects and reads the license plate (vision pipeline), cross-references the plate against a SQLite vehicle manifest, and conducts a spoken voice exchange with the driver to verify identity (audio pipeline). Based on confidence and database lookup results, it either grants access with dock instructions, requests manual verification, or denies entry.

**All processing is strictly local and offline.** No cloud APIs, no internet connectivity required or permitted at runtime.

**Target hardware:** NVIDIA Jetson Orin NX  
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
| GUI | Tkinter |
| Config | YAML (`utils/config.yaml`) |

---

## Architecture

### Three-Flow Decision Logic (`Orchestrator.run_automated_entry`)

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

### `scripts/orchestrator.py`

**`Orchestrator`** — Top-level coordinator. Owns the `VisionPipeline`, `AudioPipeline`, and SQLite connection. The single public entry point is `run_automated_entry(image)` which runs one full gate-check cycle and returns `(result_dict, vision_output)`. Implements all three flows. Supports context manager (`with` statement). Result statuses: `"success"`, `"alert_worker"`, `"name_mismatch"`.

**`create_mock_db(db_path)`** — Seeds a SQLite database with six demo trucks for development and testing.

---

### `scripts/vision_pipeline.py`

**`VisionPipeline`** — License plate detection and OCR. Uses YOLOv11 for bounding-box detection and Fast-Plate-OCR for character recognition. Supports ONNX model variant. Main method: `read_plate(image)` returns a dict with `plate`, `conf`, `bbox`, `visual` (annotated image), or `None` if no plate is found. `_select_best_box()` picks the highest-confidence detection when multiple plates are visible.

**`read_config(path)`** — Loads `utils/config.yaml`.

---

### `scripts/audio_pipeline.py`

**`Listener`** — Records microphone audio (16kHz, float32 via `sounddevice`) and transcribes it with Faster-Whisper.

**`Speaker`** — Synthesises speech with Piper TTS and plays back via `sounddevice`.

**`LanguageModel`** — Llama 3.2 1B wrapper. Used only for *understanding* driver speech, not for free-form generation. Key methods:
- `parse_plate_from_transcription(text)` — Four-layer extraction (see above)
- `parse_yes_no(text)` → bool — Binary yes/no classification
- `extract_name(text)` → str — Extracts a name in Title Case
- `verify_name_similarity(db_name, spoken_name)` → bool — Lenient fuzzy match

**`AudioPipeline`** — Coordinates `Speaker`, `Listener`, and `LanguageModel`. Called by `Orchestrator`:
- `request_plate_from_driver(vision_output)` — Asks driver to say their plate, returns transcription + parsed plate
- `confirm_plate_with_driver(plate)` → bool — Asks driver to confirm a plate reading
- `request_driver_name()` → str — Asks driver for their name
- `give_dock_instructions(db_entry)` — Speaks access-granted message with dock number
- `speak(text)` — Direct TTS for alerts

---

### `UI/GUI.py`

**`GUI`** — Tkinter production interface. Two-column layout: annotated image (left) + live log (right). Runs the orchestrator in a background thread to keep the UI responsive. Thread-safe helpers: `log()`, `set_status()`, `set_button_enabled()`. Status indicator uses colour coding (green/red/orange). Includes image file import dialog.

### `UI/CLI.py`

**`CLI`** — Command-line testing interface. Modes: full flow, vision only, audio only, batch (process a folder of images). Uses ANSI colours for readable terminal output.

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
```

---

## Entry Points

- **Production:** `python UI/GUI.py` — Tkinter GUI for gate deployment
- **Testing/debug:** `python UI/CLI.py` — CLI with multi-mode testing
- **Tests:** `pytest tests/` — ML and hardware dependencies are mocked via `sys.modules` injection in `tests/conftest.py`; a real temporary SQLite database is used for data-access tests

---

## Deployment Notes (Jetson Orin NX)

- USB microphone and USB speaker are likely required (onboard audio may not work)
- Tkinter GUI may not work on Jetson; fall back to `CLI.py` if necessary
- Use the ONNX model variant (`model_path_onnx`) for better CPU/GPU performance on Jetson
- SQLite database uses `check_same_thread=False` for the GUI's background worker thread

---

## Known Issues

- `LanguageModel` occasionally returns an empty string for plate parsing despite correct input — tracked in `Documentation.txt`
- Full Jetson deployment testing is pending; local CPU testing is functional
