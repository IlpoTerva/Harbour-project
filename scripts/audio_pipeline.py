import logging
import re
import numpy as np
from faster_whisper import WhisperModel
from typing import Dict, Any, Optional
from piper import PiperVoice
from llama_cpp import Llama

logger = logging.getLogger(__name__)

# ── Plate parsing constants ───────────────────────────────────────────────────

# Regex that matches a valid plate: 3–9 uppercase alphanumeric characters.
# European plates are typically 5-8 chars; 3 is the floor to avoid false hits.
_PLATE_RE = re.compile(r'[A-Z0-9]{3,9}')

# NATO phonetic alphabet + spoken digit words → single character
_NATO = {
    "alpha":"A", "bravo":"B", "charlie":"C", "delta":"D", "echo":"E",
    "foxtrot":"F", "golf":"G", "hotel":"H", "india":"I", "juliet":"J",
    "kilo":"K", "lima":"L", "mike":"M", "november":"N", "oscar":"O",
    "papa":"P", "quebec":"Q", "romeo":"R", "sierra":"S", "tango":"T",
    "uniform":"U", "victor":"V", "whiskey":"W", "x-ray":"X", "xray":"X",
    "yankee":"Y", "zulu":"Z",
    "zero":"0", "one":"1", "two":"2", "three":"3", "four":"4",
    "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9",
}


# ── Listener ──────────────────────────────────────────────────────────────────

class Listener:
    """Records microphone audio and transcribes it with faster-whisper."""

    def __init__(self, conf: Dict[str, Any], sample_rate: int = 16_000, device: str = "cpu") -> None:
        self.sample_rate = sample_rate
        self.model: WhisperModel = WhisperModel(
            model_size_or_path=conf["models"]["whisper_model_path"],
            device="cpu",
            compute_type="int8",
        )

    def listen(self, duration: int = 5) -> np.ndarray:
        """Block and record `duration` seconds. Returns a 1-D float32 array."""
        import sounddevice as sd
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
        segments, _ = self.model.transcribe(audio, language="en", task="transcribe", beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"Transcribed: {text!r}")
        return text


# ── Speaker ───────────────────────────────────────────────────────────────────

class Speaker:
    """Converts text to speech with Piper and plays it back synchronously."""

    def __init__(self, conf: Dict[str, Any], device="cpu") -> None:
        if device == "cuda":
            self.pipeline = PiperVoice.load(
                conf["models"]["piper_model_path"],
                use_cuda=True,
            )
        else:  # CPU-only fallback
            self.pipeline = PiperVoice.load(conf["models"]["piper_model_path"])

    def speak(self, text: str) -> None:
        import sounddevice as sd
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

    def __init__(self, conf: Dict[str, Any], device="cpu") -> None:
        if device == "cuda":
            self.model = Llama(
                model_path=conf["models"]["llm_model_path"],
                n_ctx=2048,
                verbose=False,
                n_gpu_layers=-1,  # ← uncomment to offload all layers to GPU
            )
        else:# CPU-only inference (fallback)
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
        """Run the model and return the first non-empty line of the response.

        "\n" is intentionally NOT in the stop list because small models (1B)
        often emit a leading newline before the actual answer. Stopping there
        gives an empty string. Instead we let the model run to <|eot_id|> and
        take the first meaningful line ourselves.
        """
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>"],
            temperature=temperature,
        )
        raw = response["choices"][0]["text"]
        first_line = next((l.strip() for l in raw.split("\n") if l.strip()), raw.strip())
        return first_line

    # ── Task 1: Plate extraction ──────────────────────────────────────────────

    def parse_plate_from_transcription(self, transcription: str) -> str:
        """Extract the alphanumeric plate from free-form driver speech.

        Four-layer approach (fastest / most reliable first):

        1. Direct match  — Whisper already returned a bare plate ("LM633BD").
                           No LLM needed; just clean and return.
        2. NATO words    — every word is a NATO callsign or spoken digit.
                           Deterministic conversion, no LLM needed.
        3. LLM           — natural language ("my plate is LM633BD") or mixed
                           input that layers 1-2 cannot handle.
        4. Regex fallback — LLM returned empty or garbage; pull the longest
                           alphanumeric run from the raw transcription.
        """
        text = transcription.strip()

        # ── Layer 1: direct plate match ───────────────────────────────────────
        direct = re.sub(r'[^A-Z0-9]', '', text.upper())
        if _PLATE_RE.fullmatch(direct):
            logger.info(f"Plate direct match (no LLM): {direct!r}")
            return direct

        # ── Layer 2: NATO phonetic word-by-word conversion ────────────────────
        nato_chars = []
        all_nato = True
        for word in text.lower().split():
            if word in _NATO:
                nato_chars.append(_NATO[word])
            elif len(word) == 1 and word.isalnum():
                nato_chars.append(word.upper())
            else:
                all_nato = False
                break
        if all_nato and nato_chars:
            nato_plate = "".join(nato_chars)
            if _PLATE_RE.fullmatch(nato_plate):
                logger.info(f"Plate via NATO: {text!r} -> {nato_plate!r}")
                return nato_plate

        # ── Layer 3: LLM for natural-language input ───────────────────────────
        system = (
            "You are a harbor gate assistant. "
            "Extract ONLY the alphanumeric license plate from the driver's speech. "
            "Convert NATO phonetic alphabet words to their single letters: "
            "Alpha=A, Bravo=B, Charlie=C, Delta=D, Echo=E, Foxtrot=F, Golf=G, "
            "Hotel=H, India=I, Juliet=J, Kilo=K, Lima=L, Mike=M, November=N, "
            "Oscar=O, Papa=P, Quebec=Q, Romeo=R, Sierra=S, Tango=T, "
            "Uniform=U, Victor=V, Whiskey=W, X-ray=X, Yankee=Y, Zulu=Z. "
            "Digits: zero=0 one=1 two=2 three=3 four=4 five=5 six=6 seven=7 eight=8 nine=9. "
            "If no plate can be found, return UNKNOWN. "
            "Output ONLY the plate string, no spaces, no punctuation, no explanation."
        )
        user = 'Driver said: "' + text + '"' + '\nExtracted plate:'
        try:
            raw = self._call(self._prompt(system, user), max_tokens=20, temperature=0.1)
            clean = re.sub(r'[^A-Z0-9]', '', raw.upper())
            if clean and clean != "UNKNOWN":
                logger.info(f"Plate via LLM: {text!r} -> {clean!r}")
                return clean
        except Exception:
            logger.exception("LLM plate parsing failed.")

        candidates = _PLATE_RE.findall(re.sub(r'\s+', '', text.upper()))
        if candidates:
            best = max(candidates, key=len)
            logger.info(f"Plate via regex fallback: {text!r} -> {best!r}")
            return best

        logger.warning(f"Could not extract plate from: {text!r}")
        return "UNKNOWN"


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
            "You are a name extractor at a harbor gate. "
            "Extract the speaker's full name (first and last) from their speech. "
            "Return it in Title Case (e.g. John Smith). "
            "If no clear full name can be found, return UNKNOWN. "
            "Output ONLY the name — no explanation, no punctuation."
        )
        user = 'Speaker said: "' + transcription + '"\nExtracted name:'

        try:
            name = self._call(self._prompt(system, user), max_tokens=20, temperature=0.1)
            # Reject single-word results from the LLM — a full name is expected.
            if name.upper() == "UNKNOWN" or len(name.strip().split()) < 2:
                logger.info(f"Name extraction returned single token or UNKNOWN: {name!r}")
                return "UNKNOWN"
            logger.info(f"Name extracted: {transcription!r} → {name!r}")
            return name
        except Exception:
            logger.exception("LLM name extraction failed.")
            return "UNKNOWN"


    def verify_name_similarity(self, db_name: str, spoken_name: str) -> bool:
        """Strict match of a spoken name against the name on file.

        The last name must match exactly (or near-exactly for spelling variants).
        Common English short-form nicknames (Bob/Robert, Bill/William, Kate/Katherine)
        are accepted for the first name. First-name-only responses, cross-language
        substitutions (Juan/John), and partial matches are rejected.
        Uses temperature=0.0. Falls back to a strict token-overlap check.
        """
        if spoken_name in ("UNKNOWN", ""):
            logger.info("Name match rejected: spoken name is UNKNOWN or empty.")
            return False

        system = (
            "You are a strict name verification assistant at a harbor gate. "
            "Decide if the spoken name refers to the SAME person as the name on file. "
            "Rules (apply all of them):\n"
            "1. The LAST NAME must match — exact spelling or a very minor variation "
            "(e.g. a missing accent). Different last names are NEVER a match.\n"
            "2. If the name on file includes a last name, a first-name-only response is NOT a match.\n"
            "3. The first name may be a well-known English short-form nickname for the registered "
            "first name (e.g. Bob for Robert, Bill for William, Kate for Katherine). "
            "Cross-language substitutions such as Juan for John are NOT a match.\n"
            "4. If in doubt, output NO.\n"
            "Output ONLY the single word YES or NO. No other text whatsoever."
        )
        user = (
            'Name on file: "' + db_name + '"\n'
            'Name spoken by driver: "' + spoken_name + '"\n'
            "Same person?"
        )

        try:
            result = self._call(self._prompt(system, user), max_tokens=5, temperature=0.0).upper()
            # Accept only an explicit YES; anything else (NO, empty, garbage) → False.
            match = result.startswith("YES")
            logger.info(f"Name match: {db_name!r} vs {spoken_name!r} → {match}")
            return match
        except Exception:
            logger.exception("LLM name matching failed — using strict token fallback.")
            spoken_parts = spoken_name.lower().strip().split()
            db_parts = db_name.lower().strip().split()
            if not spoken_parts or not db_parts:
                return False
            # Last names must match exactly.
            if spoken_parts[-1] != db_parts[-1]:
                return False
            # If the registered name has a first name, require it to appear in the spoken name too.
            if len(db_parts) >= 2:
                return db_parts[0] in spoken_parts
            return True