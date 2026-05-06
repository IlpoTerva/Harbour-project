import logging
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
from faster_whisper import WhisperModel
from typing import Dict, Any, Optional
from piper import PiperVoice

logger = logging.getLogger(__name__)

# Kokoro always synthesises at 24 kHz regardless of input.
KOKORO_SAMPLE_RATE = 24_000


class Listener:
    """Records microphone audio and transcribes it with faster-whisper."""

    def __init__(self,conf: Dict[str, Any], sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.model: WhisperModel = WhisperModel(model_size_or_path=conf["models"]["whisper_model_path"], device="cpu")

    def listen(self, duration: int = 5) -> np.ndarray:
        """Block and record `duration` seconds from the default microphone.

        Returns a 1-D float32 array at `self.sample_rate`.
        """
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
        """Transcribe `audio` and return the joined segment text."""
        segments, _ = self.model.transcribe(audio, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"Transcribed: {text!r}")
        return text


class Speaker:
    """Converts text to speech with Kokoro and plays it back synchronously."""

    def __init__(self,conf: Dict[str, Any], kokoro=False) -> None:
        # lang_code='a' → American English; swap to 'e' for Spanish, 'p' for Portuguese, etc.
        if kokoro:
            self.pipeline: KPipeline = KPipeline(lang_code="a", device="cpu")
            self.use_kokoro = True
        else:
            self.pipeline = PiperVoice.load(conf["models"]["piper_model_path"])
            self.use_kokoro = False

    def speak(self, text: str, voice: str = "af_heart", speed: float = 1.0) -> None:
        """Synthesise `text` and play it, blocking until playback is finished.

        Kokoro yields (graphemes, phonemes, audio_chunk) tuples; we concatenate
        all chunks before handing them to sounddevice so there are no gaps.
        """
        if self.use_kokoro:
            chunks = [audio for _, _, audio in self.pipeline(text, voice=voice, speed=speed)]
            if not chunks:
                logger.warning("TTS produced no audio — is the text empty?")
                return
            full_audio = np.concatenate(chunks)
            sample_rate = KOKORO_SAMPLE_RATE
        else:
            audio_data = []
            for chunk in self.pipeline.synthesize(text):
                audio_chunk = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                audio_data.append(audio_chunk)
            if not audio_data:
                logger.warning("TTS produced no audio — is the text empty?")
                return
            full_audio = np.concatenate(audio_data)
            sample_rate = self.pipeline.config.sample_rate
        if full_audio.dtype == np.int16:
            full_audio = full_audio.astype(np.float32) / 32768.0
        sd.play(full_audio, samplerate=sample_rate)
        sd.wait()


class LanguageModel:
    """Prompt generation and plate parsing logic.

    Currently rule-based. Swap `generate_prompt` and `parse_plate_from_transcription`
    for llama-cpp-python / mlx / transformers calls once the Llama 3.2 1B model
    is deployed on the Pi — the interface stays the same.
    """

    def generate_prompt(self, vision_output: Optional[Dict[str, Any]]) -> str:
        """Return a spoken prompt that gives the driver helpful context."""
        if vision_output and vision_output.get("conf", 0) >= 0.5:
            plate = vision_output["plate"]
            return (
                f"I detected the plate {' '.join(plate)} but I'm not fully confident. "
                "Could you please confirm your licence plate out loud?"
            )
        return "I could not read your licence plate. Please say it out loud now."

    def parse_plate_from_transcription(self, transcription: str) -> str:
        """Extract a plate string from free-form driver speech.

        Placeholder: strips whitespace and upper-cases the raw transcription.
        Replace with a Llama call to handle natural phrasing such as
        'my plate is Romeo Kilo six one two alpha lima'.
        """
        return transcription.strip().upper()


class AudioPipeline:
    """Coordinates Speaker → Listener → LanguageModel for the voice fallback flow.

    Typical call from the Orchestrator when vision confidence is too low:

        result = audio_pipeline.request_plate_from_driver(vision_output)
        spoken_plate = result["plate"]
    """

    def __init__(self, conf: Dict[str, Any]) -> None:
        self.listener: Listener = Listener(conf)
        self.speaker: Speaker = Speaker(conf)
        self.language_model: LanguageModel = LanguageModel()

    def speak(self, text: str) -> None:
        """Speak `text` to the driver."""
        logger.info(f"Speaking: {text!r}")
        self.speaker.speak(text)

    def listen_and_transcribe(self, duration: int = 5) -> Dict[str, Any]:
        """Record for `duration` seconds and return the transcription and raw audio."""
        audio = self.listener.listen(duration)
        transcription = self.listener.transcribe(audio)
        return {"transcription": transcription, "audio": audio}

    def request_plate_from_driver(
        self,
        vision_output: Optional[Dict[str, Any]] = None,
        duration: int = 6,
    ) -> Dict[str, Any]:
        """Full voice fallback flow:

        1. Generate a context-aware prompt and speak it to the driver.
        2. Listen for the driver's spoken response.
        3. Parse the plate text from the Whisper transcription.

        Args:
            vision_output: Output from VisionPipeline.process(), used to tailor
                           the spoken prompt. Pass None when there was no detection.
            duration:      Recording length in seconds.

        Returns:
            {
                "plate":         str         – parsed plate text (upper-cased)
                "transcription": str         – raw Whisper output
                "audio":         np.ndarray  – recorded audio buffer
            }
        """
        prompt = self.language_model.generate_prompt(vision_output)
        self.speak(prompt)

        result = self.listen_and_transcribe(duration)
        plate = self.language_model.parse_plate_from_transcription(result["transcription"])
        result["plate"] = plate
        logger.info(f"Driver-provided plate: {plate!r}")
        return result
