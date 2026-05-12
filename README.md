# ⚓ Harbour Agent: Multimodal Edge AI Assistant

> **High-performance, real-time multimodal AI agent designed specifically for edge deployment.**

Optimized for the **NVIDIA Jetson Orin NX**, this project integrates state-of-the-art vision, speech, and reasoning models to interact with its environment autonomously.

---

# 🛠️ Core Technology Stack

> **The project leverages a specialized stack chosen for the balance between accuracy and edge-device throughput:**

### 👁️ Computer Vision
- YOLO (You Only Look Once): Real-time object detection for environmental awareness.

- Fast-Plate-OCR: Optimized optical character recognition specifically for vehicle identification and logistics.

### 🧠 Language & Reasoning
- Llama 3.2 (1B): A compact but powerful LLM optimized for edge-tier reasoning, running locally via llama-cpp-python with full CUDA acceleration.

### 🎙️ Audio Pipeline
- Faster-Whisper: A re-implementation of OpenAI’s Whisper model using CTranslate2 for ultra-fast Speech-to-Text (STT).

- Piper-TTS: A fast, local neural text-to-speech system that provides natural-sounding voice synthesis without requiring an internet connection.
