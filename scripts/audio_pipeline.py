import logging
import numpy as np
import sounddevice as sd
from kokoro import KPipeline
from faster_whisper import WhisperModel
from typing import Dict, Any, Optional
from piper import PiperVoice
from llama_cpp import Llama

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

    """
    def __init__(self) -> None:
        self.model = Llama(model_path="models/llama-3.2-1B-Instruct-Q4_K_M.gguf", n_ctx=2048, verbose=False)# NOTE: n_gpu_layers=-1 

    def generic_prompt(self, vision_output: Optional[Dict[str, Any]]) -> str:
        """Generate a spoken prompt based on `vision_output` to ask the driver for their plate."""
        if vision_output and vision_output.get("conf", 0) >= 0.5:
            plate = vision_output["plate"]
            return (
                f"I detected the plate {' '.join(plate)} but I'm not fully confident. "
                "Could you please confirm your licence plate out loud?"
            )
        return "I could not read your licence plate. Please say it out loud now."

    def driver_instructions_prompt(self, inDatabase: bool, read_plate: str = None) -> str:
        """Generate a spoken prompt to instruct the driver on where to go next."""
        if inDatabase:
            return "Thank you. Your vehicle is registered in our database. Please proceed to the gate."
        else:
            return f"Thank you. However, I couldn't find your vehicle in our database. Is the plate you provided, {read_plate}, correct? If not, please say it again."        

    def parse_plate_from_transcription(self, transcription: str) -> str:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a harbor gate assistant. Extract ONLY the alphanumeric license plate from the text.
            Convert NATO phonetic alphabet (Alpha, Bravo, etc.) to single characters.
            If no plate is found, return 'UNKNOWN'.
            Output format: ONLY the string.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Driver said: "{transcription}"
            Extracted Plate:<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>"""

        try:
            # temperature=0.1 makes the model more deterministic (less creative)
            response = self.model(
                prompt, 
                max_tokens=20, 
                stop=["<|eot_id|>", "\n"], 
                temperature=0.1
            )
            
            result = response["choices"][0]["text"].strip()
            
            # Post-processing: ensure it's uppercase and has no spaces
            clean_plate = result.upper().replace(" ", "").replace(".", "")
            
            logger.info(f"LLM Parsed '{transcription}' into '{clean_plate}'")
            return clean_plate

        except Exception as e:
            logger.error(f"LLM Parsing failed: {e}")
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
    
    def driver_instructions(self, inDatabase: bool, read_plate: str = None) -> None:
        """Generate and speak instructions for the driver based on whether their plate is in the database."""
        prompt = self.language_model.driver_instructions_prompt(inDatabase, read_plate)
        self.speak(prompt)
