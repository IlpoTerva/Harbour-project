import logging
import re
import numpy as np
from faster_whisper import WhisperModel
from typing import Dict, Any, Iterator, Optional, Tuple
from piper import PiperVoice
from llama_cpp import Llama

logger = logging.getLogger(__name__)

# ── Plate parsing constants ───────────────────────────────────────────────────

# Regex that matches a valid plate: 3–9 uppercase alphanumeric characters.
# European plates are typically 5-8 chars; 3 is the floor to avoid false hits.
_PLATE_RE = re.compile(r'[A-Z0-9]{3,9}')

# NATO phonetic alphabet (international standard) + spoken digit words.
# Digit words are included for EN/ES/FR/PT so layer-2 deterministic conversion
# works for drivers who spell out digits in their native language.
_NATO = {
    # NATO letters (language-agnostic international standard)
    "alpha":"A", "bravo":"B", "charlie":"C", "delta":"D", "echo":"E",
    "foxtrot":"F", "golf":"G", "hotel":"H", "india":"I", "juliet":"J",
    "kilo":"K", "lima":"L", "mike":"M", "november":"N", "oscar":"O",
    "papa":"P", "quebec":"Q", "romeo":"R", "sierra":"S", "tango":"T",
    "uniform":"U", "victor":"V", "whiskey":"W", "x-ray":"X", "xray":"X",
    "yankee":"Y", "zulu":"Z",
    # English digits
    "zero":"0", "one":"1", "two":"2", "three":"3", "four":"4",
    "five":"5", "six":"6", "seven":"7", "eight":"8", "nine":"9",
    # Spanish digits (seis/six already covered above)
    "cero":"0", "uno":"1", "dos":"2", "tres":"3", "cuatro":"4",
    "cinco":"5", "siete":"7", "ocho":"8", "nueve":"9",
    # French digits (six already covered above)
    "zéro":"0", "un":"1", "deux":"2", "trois":"3", "quatre":"4",
    "cinq":"5", "sept":"7", "huit":"8", "neuf":"9",
    # Portuguese digits (zero/seis/cinco already covered above)
    "um":"1", "dois":"2", "três":"3", "quatro":"4",
    "sete":"7", "oito":"8", "nove":"9",
}

# Module-level yes/no keyword fallback — used when the LLM call fails.
# Covers EN / ES / FR / PT affirmatives.
_YES_KEYWORDS = [
    # English
    "yes", "correct", "right", "yeah", "yep", "confirmed",
    # Spanish
    "sí", "si", "correcto", "exacto",
    # French
    "oui", "exact",
    # Portuguese
    "sim", "correto", "certo",
]


# ── Listener ──────────────────────────────────────────────────────────────────

class Listener:
    """Records microphone audio and transcribes it with faster-whisper."""

    def __init__(self, conf: Dict[str, Any], sample_rate: int = 16_000, device: str = "cpu") -> None:
        self.sample_rate = sample_rate
        # int8 is fast on CPU; float16 gets CUDA tensor cores on GPU.
        compute_type = "float16" if device == "cuda" else "int8"
        self.model: WhisperModel = WhisperModel(
            model_size_or_path=conf["tts"]["whisper"],
            device=device,
            compute_type=compute_type,
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

    def transcribe(self, audio: np.ndarray) -> Tuple[str, str]:
        """Transcribe audio and return (text, detected_language_code)."""
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,                      # greedy decoding — ~3-5× faster than beam_size=5
            vad_filter=True,                  # skip silent padding via bundled silero-VAD
            condition_on_previous_text=False,  # single utterances need no inter-segment context
            without_timestamps=True,          # skip timestamp computation
        )
        text = " ".join(seg.text.strip() for seg in segments)
        detected_lang = getattr(info, "language", "en") or "en"
        logger.info(f"Transcribed ({detected_lang}): {text!r}")
        return text, detected_lang


# ── Speaker ───────────────────────────────────────────────────────────────────

class Speaker:
    """Converts text to speech with Piper TTS.

    Supports multiple languages via a lazy model registry: models are loaded on
    first use and cached. Add a new language by adding its code and model path
    to config.yaml under speech.piper_models — no code changes needed.
    """

    def __init__(self, conf: Dict[str, Any], device: str = "cpu") -> None:
        self._use_cuda = (device == "cuda")
        # Build model-path map from new config layout, with fallback to legacy key.
        speech_conf = conf.get("speech", {})
        piper_map = speech_conf.get("piper_models", {})
        if not piper_map:
            legacy = conf.get("models", {}).get("piper_model_path")
            if legacy:
                piper_map = {"en": legacy}
        self._model_paths: Dict[str, str] = piper_map
        self._pipelines: Dict[str, PiperVoice] = {}
        # Pre-load the default language so the first call has no extra latency.
        default_lang = speech_conf.get("default_language", "en")
        self._load(default_lang)

    def _load(self, lang: str) -> Optional[PiperVoice]:
        if lang in self._pipelines:
            return self._pipelines[lang]
        path = self._model_paths.get(lang)
        if path is None:
            logger.warning(f"No Piper model configured for language {lang!r}.")
            return None
        voice = (
            PiperVoice.load(path, use_cuda=True)
            if self._use_cuda
            else PiperVoice.load(path)
        )
        self._pipelines[lang] = voice
        logger.info(f"Piper model loaded for language {lang!r}.")
        return voice

    def synthesize_chunks(self, text: str, language: str = "en") -> Tuple[Iterator, int]:
        """Return (chunk_iterator, sample_rate) for use by the API server."""
        pipeline = self._load(language) or self._pipelines.get("en")
        if pipeline is None:
            raise RuntimeError("No TTS pipeline available.")
        return pipeline.synthesize(text), pipeline.config.sample_rate

    def speak(self, text: str, language: str = "en") -> None:
        """Synthesise `text` and play it back through the local speaker."""
        import sounddevice as sd
        pipeline = self._load(language) or self._pipelines.get("en")
        if pipeline is None:
            logger.error("No TTS pipeline available — cannot speak.")
            return
        audio_data = []
        for chunk in pipeline.synthesize(text):
            audio_data.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not audio_data:
            logger.warning("TTS produced no audio — is the text empty?")
            return
        full_audio = np.concatenate(audio_data).astype(np.float32) / 32768.0
        sd.play(full_audio, samplerate=pipeline.config.sample_rate)
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
                model_path=conf["tts"]["llm"],
                n_ctx=2048,
                verbose=False,
                n_gpu_layers=-1,  # ← uncomment to offload all layers to GPU
            )
        else:# CPU-only inference (fallback)
            self.model = Llama(
                model_path=conf["tts"]["llm"],
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
            "The driver may speak any language; the plate itself is always alphanumeric. "
            "Convert NATO phonetic alphabet words to their single letters: "
            "Alpha=A, Bravo=B, Charlie=C, Delta=D, Echo=E, Foxtrot=F, Golf=G, "
            "Hotel=H, India=I, Juliet=J, Kilo=K, Lima=L, Mike=M, November=N, "
            "Oscar=O, Papa=P, Quebec=Q, Romeo=R, Sierra=S, Tango=T, "
            "Uniform=U, Victor=V, Whiskey=W, X-ray=X, Yankee=Y, Zulu=Z. "
            "Digits (EN/ES/FR/PT): zero/cero/zéro=0, one/uno/un/um=1, two/dos/deux/dois=2, "
            "three/tres/trois/três=3, four/cuatro/quatre/quatro=4, five/cinco/cinq=5, "
            "six/seis=6, seven/siete/sept/sete=7, eight/ocho/huit/oito=8, nine/nueve/neuf/nove=9. "
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
            return any(w in lower for w in _YES_KEYWORDS)

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
            name = name.strip().title()
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
        db_name = db_name.strip().title()
        spoken_name = spoken_name.strip().title()

        if spoken_name in ("Unknown", ""):
            logger.info("Name match rejected: spoken name is UNKNOWN or empty.")
            return False

        system = (
            "You are a name verification assistant at a harbor gate. "
            "Decide if the spoken name and the name on file refer to the SAME person.\n"
            "Rules:\n"
            "1. If both names are identical or nearly identical (minor spelling or accent difference), output YES.\n"
            "2. The last name must match — different last names are never a match.\n"
            "3. If the name on file has a last name but only a first name is spoken, output NO.\n"
            "4. A common English nickname is acceptable for the first name (e.g. Bob for Robert, Bill for William).\n"
            "5. Cross-language equivalents are not a match (e.g. Juan for John → NO).\n"
            "Output ONLY the single word YES or NO."
        )
        #Debugging the name verification
        logger.info(f"Verifying name similarity: on file {db_name!r} vs spoken {spoken_name!r}")
        user = (
            'Name on file: "' + db_name.lower() + '"\n'
            'Name spoken by driver: "' + spoken_name.lower() + '"\n'
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