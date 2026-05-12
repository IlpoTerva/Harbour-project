import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Dict, Any, Optional
from piper import PiperVoice
from llama_cpp import Llama

logger = logging.getLogger(__name__)


# ── Listener ──────────────────────────────────────────────────────────────────

class Listener:
    """Records microphone audio and transcribes it with faster-whisper."""

    def __init__(self, conf: Dict[str, Any], sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.model: WhisperModel = WhisperModel(
            model_size_or_path=conf["models"]["whisper_model_path"],
            device="cpu",
        )

    def listen(self, duration: int = 5) -> np.ndarray:
        """Block and record `duration` seconds. Returns a 1-D float32 array."""
        logger.info(f"Listening for {duration}s…")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"Transcribed: {text!r}")
        return text


# ── Speaker ───────────────────────────────────────────────────────────────────

class Speaker:
    """Converts text to speech with Piper and plays it back synchronously."""

    def __init__(self, conf: Dict[str, Any]) -> None:
        self.pipeline = PiperVoice.load(conf["models"]["piper_model_path"])

    def speak(self, text: str) -> None:
        audio_data = []
        for chunk in self.pipeline.synthesize(text):
            audio_data.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not audio_data:
            logger.warning("TTS produced no audio — is the text empty?")
            return
        full_audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
        sd.play(full_audio, samplerate=self.pipeline.config.sample_rate)
        sd.wait()


# ── LanguageModel ─────────────────────────────────────────────────────────────

class LanguageModel:
    """
    Wraps Llama 3.2 1B for all NLU tasks: plate parsing, yes/no classification,
    name extraction, and fuzzy name matching.

    Spoken prompts (what the system says to the driver) are plain Python strings —
    no LLM needed for generation since they are deterministic templates.
    The LLM is used exclusively for *understanding* what the driver says back.

    GPU note: set n_gpu_layers=-1 once CUDA is confirmed working on the device.
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        self.model = Llama(
            model_path=conf["models"]["llm_model_path"],
            n_ctx=2048,
            verbose=False,
            # n_gpu_layers=-1,  # ← uncomment to offload all layers to GPU
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _prompt(self, system: str, user: str) -> str:
        """Build a clean Llama 3.2 Instruct chat prompt.

        Using explicit string concatenation instead of an indented f-string
        ensures no accidental leading whitespace enters the prompt, which
        causes models to output garbage or repeat the whitespace pattern.
        """
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            + system.strip() + "\n<|eot_id|>\n"
            + "<|start_header_id|>user<|end_header_id|>\n"
            + user.strip() + "\n<|eot_id|>\n"
            + "<|start_header_id|>assistant<|end_header_id|>"
        )

    def _call(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Run the model and return the stripped text response."""
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "\n"],
            temperature=temperature,
        )
        return response["choices"][0]["text"].strip()

    # ── Task 1: Plate extraction ──────────────────────────────────────────────

    def parse_plate_from_transcription(self, transcription: str) -> str:
        """Extract the alphanumeric plate from free-form driver speech.

        Handles NATO phonetic alphabet (Romeo Kilo six one two → RK612).
        Falls back to uppercased raw transcription if the LLM call fails.
        """
        system = (
            "You are a harbor gate assistant. "
            "Extract ONLY the alphanumeric license plate from the driver's speech. "
            "Convert NATO phonetic alphabet words to their single letters: "
            "Alpha=A, Bravo=B, Charlie=C, Delta=D, Echo=E, Foxtrot=F, Golf=G, "
            "Hotel=H, India=I, Juliet=J, Kilo=K, Lima=L, Mike=M, November=N, "
            "Oscar=O, Papa=P, Quebec=Q, Romeo=R, Sierra=S, Tango=T, "
            "Uniform=U, Victor=V, Whiskey=W, X-ray=X, Yankee=Y, Zulu=Z. "
            "Digits are spoken as individual numbers (six=6, one=1, two=2). "
            "If no plate can be found, return UNKNOWN. "
            "Output ONLY the plate string — no spaces, no punctuation, no explanation."
        )
        user = 'Driver said: "' + transcription + '"\nExtracted plate:'

        try:
            raw = self._call(self._prompt(system, user), max_tokens=20, temperature=0.1)
            clean = raw.upper().replace(" ", "").replace(".", "").replace("-", "")
            logger.info(f"Plate parsed: {transcription!r} → {clean!r}")
            return clean
        except Exception:
            logger.exception("LLM plate parsing failed — using raw transcription as fallback.")
            return transcription.strip().upper().replace(" ", "")

    # ── Task 2: Yes/No classification ────────────────────────────────────────

    def parse_yes_no(self, transcription: str) -> bool:
        """Return True if the driver's response expresses agreement/confirmation.

        Uses temperature=0.0 for maximum determinism on this binary task.
        Falls back to keyword matching if the LLM call fails.
        """
        system = (
            "You are a yes/no classifier. "
            "Decide whether the speaker is confirming (yes) or denying (no). "
            "Output ONLY the single word YES or NO. No other text whatsoever."
        )
        user = 'Speaker said: "' + transcription + '"\nClassification:'

        try:
            result = self._call(self._prompt(system, user), max_tokens=5, temperature=0.0).upper()
            answer = result.startswith("YES")
            logger.info(f"Yes/No: {transcription!r} → {answer}")
            return answer
        except Exception:
            logger.exception("LLM yes/no classification failed — using keyword fallback.")
            lower = transcription.lower()
            return any(w in lower for w in ["yes", "correct", "right", "yeah", "yep", "confirmed"])

    # ── Task 3: Name extraction ───────────────────────────────────────────────

    def extract_name(self, transcription: str) -> str:
        """Extract a person's full name from free-form speech.

        Returns the name in Title Case (e.g. 'John Smith').
        Returns 'UNKNOWN' if no name is identifiable.
        """
        system = (
            "You are a name extractor. "
            "Extract the speaker's full name from their speech. "
            "Return it in Title Case (e.g. John Smith). "
            "If no name can be found, return UNKNOWN. "
            "Output ONLY the name — no explanation, no punctuation."
        )
        user = 'Speaker said: "' + transcription + '"\nExtracted name:'

        try:
            name = self._call(self._prompt(system, user), max_tokens=20, temperature=0.1)
            logger.info(f"Name extracted: {transcription!r} → {name!r}")
            return name
        except Exception:
            logger.exception("LLM name extraction failed — using raw transcription as fallback.")
            return transcription.strip().title()

    # ── Task 4: Fuzzy name matching ───────────────────────────────────────────

    def verify_name_similarity(self, db_name: str, spoken_name: str) -> bool:
        """Fuzzy-match a spoken name against the name on file.

        Lenient matching: nicknames, partial names, different pronunciations,
        and cross-language equivalents (e.g. Juan ≈ John, Matti ≈ Matthew).
        Uses temperature=0.0 — this is a binary access-control decision.
        Falls back to first-token substring match if the LLM call fails.
        """
        system = (
            "You are a name verification assistant at a harbor gate. "
            "Decide if two names refer to the same person. "
            "Be lenient: allow for nicknames, partial names, different pronunciations, "
            "and cross-language equivalents (e.g. Juan ≈ John, Matti ≈ Matthew, "
            "Meikalainen ≈ Meikäläinen). "
            "Output ONLY the single word YES or NO. No other text whatsoever."
        )
        user = (
            'Name on file: "' + db_name + '"\n'
            'Name spoken by driver: "' + spoken_name + '"\n'
            "Same person?"
        )

        try:
            result = self._call(self._prompt(system, user), max_tokens=5, temperature=0.0).upper()
            match = result.startswith("YES")
            logger.info(f"Name match: {db_name!r} vs {spoken_name!r} → {match}")
            return match
        except Exception:
            logger.exception("LLM name matching failed — using first-token fallback.")
            s = spoken_name.lower().strip().split()
            e = db_name.lower().strip().split()
            return bool(s and e and s[0] == e[0])


# ── AudioPipeline ─────────────────────────────────────────────────────────────

class AudioPipeline:
    """
    Coordinates Speaker → Listener → LanguageModel for all driver interactions.

    Public interface used by Orchestrator:
        request_plate_from_driver(vision_output)  → dict  (plate + transcription)
        confirm_plate_with_driver(plate)           → bool
        request_driver_name()                      → str   (parsed name)
        give_dock_instructions(db_entry)           → None
        speak(text)                                → None  (direct TTS for alerts)
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        self.listener = Listener(conf)
        self.speaker = Speaker(conf)
        self.language_model = LanguageModel(conf)

    # ── Low-level primitives ──────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak `text` to the driver directly (used for alert / denial messages)."""
        logger.info(f"Speaking: {text!r}")
        self.speaker.speak(text)

    def _listen_and_transcribe(self, duration: int) -> Dict[str, Any]:
        """Record for `duration` seconds and return transcription + raw audio."""
        audio = self.listener.listen(duration)
        transcription = self.listener.transcribe(audio)
        return {"transcription": transcription, "audio": audio}

    # ── Plate interactions ────────────────────────────────────────────────────

    def request_plate_from_driver(
        self,
        vision_output: Optional[Dict[str, Any]] = None,
        duration: int = 6,
    ) -> Dict[str, Any]:
        """Speak a context-aware prompt, listen, and return the parsed plate.

        The prompt differs depending on whether vision produced a low-confidence
        candidate or detected nothing at all.

        Returns:
            {
                "plate":         str         – parsed plate text (upper-cased)
                "transcription": str         – raw Whisper output
                "audio":         np.ndarray  – recorded buffer
            }
        """
        if vision_output and vision_output.get("conf", 0) >= 0.5:
            plate_letters = " ".join(vision_output["plate"])
            prompt_text = (
                f"I detected the plate {plate_letters} but I am not fully confident. "
                "Could you please confirm your licence plate out loud?"
            )
        else:
            prompt_text = "I could not read your licence plate. Please say it out loud now."

        self.speak(prompt_text)
        result = self._listen_and_transcribe(duration)
        plate = self.language_model.parse_plate_from_transcription(result["transcription"])
        result["plate"] = plate
        logger.info(f"Driver-provided plate: {plate!r}")
        return result

    def confirm_plate_with_driver(self, plate: str, duration: int = 4) -> bool:
        """Read a plate back to the driver and return their yes/no answer.

        Used when vision is confident but the plate is absent from the DB —
        to rule out a single OCR character error before alerting a worker.
        """
        plate_letters = " ".join(plate)
        self.speak(
            f"I read your licence plate as {plate_letters}. "
            "Is that correct? Please say yes or no."
        )
        result = self._listen_and_transcribe(duration)
        confirmed = self.language_model.parse_yes_no(result["transcription"])
        logger.info(f"Plate {plate!r} confirmed by driver: {confirmed}")
        return confirmed

    # ── Name interaction ──────────────────────────────────────────────────────

    def request_driver_name(self, duration: int = 5) -> str:
        """Speak the name-request prompt, listen, extract and return the name string.

        Returns the LLM-extracted name in Title Case, or 'UNKNOWN' on failure.
        The caller (Orchestrator) passes this directly to verify_name_similarity.
        """
        self.speak("Please say your full name for verification.")
        result = self._listen_and_transcribe(duration)
        name = self.language_model.extract_name(result["transcription"])
        logger.info(f"Driver name: {name!r}")
        return name

    # ── Gate outcome messages ─────────────────────────────────────────────────

    def give_dock_instructions(self, db_entry: Dict[str, Any]) -> None:
        """Speak the full access-granted message with dock, cargo, and window."""
        text = (
            f"Access granted. Welcome, {db_entry['driver_name']}. "
            f"Please proceed to {db_entry['dock']}. "
            f"Your cargo is {db_entry['cargo']}. "
            f"Your arrival window is {db_entry['arrival_window']}. "
            "Have a safe unloading."
        )
        self.speak(text)
