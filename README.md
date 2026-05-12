# ⚓ Harbour Agent: Multimodal Edge AI Assistant

> **High-performance, real-time multimodal AI agent designed specifically for edge deployment.**

Optimized for the **NVIDIA Jetson Orin NX**, this project integrates state-of-the-art vision, speech, and reasoning models to interact with its environment autonomously.

## 🚀 Project Overview

The Harbour Agent acts as an intelligent observer and interlocutor. By combining local Large Language Models (LLMs) with high-speed vision and audio pipelines, it can "see" objects and license plates, "hear" verbal commands, and "speak" back to the user with minimal latency. 
The primary objective is to fully automate the harbor gate entry process. By autonomously scanning license plates, verifying driver identities via voice, and instantly assigning cargo unloading docks, the system drastically reduces the need for manual checkpoints and keeps logistics moving smoothly.

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


---

---

## ⚙️ Environment & Prerequisites

To achieve maximum throughput and avoid legacy dependency conflicts, the Harbour Agent is built for modern environments. 

* **Python:** `>= 3.10` (Required for modern library compatibility and performance optimizations).

**Edge Deployment (NVIDIA Jetson):**
* **OS:** Ubuntu 22.04 LTS. 
* **Drivers:** **JetPack 6.2+** is highly recommended. It provides native Python 3.10 support and updated CUDA 12.x drivers, bypassing the dependency limitations of older JetPack 5 systems.

**Local Development & Testing:**
* **OS:** Windows 10/11 or standard Linux desktop.
* **Drivers:** Standard NVIDIA Display Drivers and CUDA Toolkit 12.x (if testing with a discrete GPU).



- Piper-TTS: A fast, local neural text-to-speech system that provides natural-sounding voice synthesis without requiring an internet connection.
