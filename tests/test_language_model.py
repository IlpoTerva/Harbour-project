"""
test_language_model.py — unit tests for LanguageModel.

Tests cover all four NLU tasks:
  1. parse_plate_from_transcription  – extraction + NATO alphabet + fallback
  2. parse_yes_no                    – binary classification + keyword fallback
  3. extract_name                    – name extraction + fallback
  4. verify_name_similarity          – fuzzy match + fallback

Each test controls the mocked Llama response via make_llm_response() and
verifies that the post-processing and fallback logic works correctly.
"""

import pytest
from unittest.mock import patch, MagicMock

from scripts.audio_pipeline import LanguageModel
from tests.conftest import make_llm_response


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def lm(mock_config):
    """LanguageModel with a mocked Llama instance."""
    with patch("scripts.audio_pipeline.Llama") as MockLlama:
        model = LanguageModel(mock_config)
        # Expose the mock so individual tests can set return values
        model._mock_llama = MockLlama.return_value
    return model


# ── 1. Plate parsing ──────────────────────────────────────────────────────────

class TestParsePlate:

    def test_direct_plate_returned_uppercase(self, lm):
        lm._mock_llama.return_value = make_llm_response("rk612al")
        assert lm.parse_plate_from_transcription("my plate is rk612al") == "RK612AL"

    def test_nato_alphabet_converted(self, lm):
        """'Romeo Kilo six one two alpha lima' → RK612AL (post-processed from LLM)."""
        lm._mock_llama.return_value = make_llm_response("RK612AL")
        result = lm.parse_plate_from_transcription(
            "Romeo Kilo six one two alpha lima"
        )
        assert result == "RK612AL"

    def test_spaces_stripped_from_llm_output(self, lm):
        """LLM sometimes returns 'R K 6 1 2 A L' — spaces must be removed."""
        lm._mock_llama.return_value = make_llm_response("R K 6 1 2 A L")
        assert lm.parse_plate_from_transcription("Romeo...") == "RK612AL"

    def test_dots_stripped_from_llm_output(self, lm):
        lm._mock_llama.return_value = make_llm_response("R.K.612.AL")
        assert lm.parse_plate_from_transcription("...") == "RK612AL"

    def test_unknown_returned_when_llm_says_unknown(self, lm):
        lm._mock_llama.return_value = make_llm_response("UNKNOWN")
        assert lm.parse_plate_from_transcription("I don't know") == "UNKNOWN"

    def test_fallback_on_llm_exception(self, lm):
        """When LLM raises, fall back to uppercased raw transcription."""
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        result = lm.parse_plate_from_transcription("rk612al")
        assert result == "RK612AL"


# ── 2. Yes/No classification ──────────────────────────────────────────────────

class TestParseYesNo:

    def test_yes_response_returns_true(self, lm):
        lm._mock_llama.return_value = make_llm_response("YES")
        assert lm.parse_yes_no("yes that is correct") is True

    def test_no_response_returns_false(self, lm):
        lm._mock_llama.return_value = make_llm_response("NO")
        assert lm.parse_yes_no("no that is wrong") is False

    def test_case_insensitive_yes(self, lm):
        lm._mock_llama.return_value = make_llm_response("yes")
        assert lm.parse_yes_no("yeah") is True

    def test_keyword_fallback_yes_on_llm_exception(self, lm):
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        assert lm.parse_yes_no("yeah that's correct") is True

    def test_keyword_fallback_no_on_llm_exception(self, lm):
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        assert lm.parse_yes_no("that is completely wrong") is False

    @pytest.mark.parametrize("word", ["yes", "correct", "right", "yeah", "yep", "confirmed"])
    def test_all_yes_keywords_trigger_fallback_true(self, lm, word):
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        assert lm.parse_yes_no(f"I think {word}") is True


# ── 3. Name extraction ────────────────────────────────────────────────────────

class TestExtractName:

    def test_extracts_name_from_full_sentence(self, lm):
        lm._mock_llama.return_value = make_llm_response("Jane Doe")
        result = lm.extract_name("My name is Jane Doe, I'm the driver")
        assert result == "Jane Doe"

    def test_strips_whitespace_from_llm_output(self, lm):
        lm._mock_llama.return_value = make_llm_response("  Bob Wilson  ")
        result = lm.extract_name("Bob Wilson")
        assert result == "Bob Wilson"

    def test_unknown_returned_when_llm_says_unknown(self, lm):
        lm._mock_llama.return_value = make_llm_response("UNKNOWN")
        result = lm.extract_name("I forgot")
        assert result == "UNKNOWN"

    def test_fallback_on_llm_exception(self, lm):
        """Falls back to title-cased raw transcription."""
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        result = lm.extract_name("jane doe")
        assert result == "Jane Doe"


# ── 4. Name similarity ────────────────────────────────────────────────────────

class TestVerifyNameSimilarity:

    def test_exact_match_returns_true(self, lm):
        lm._mock_llama.return_value = make_llm_response("YES")
        assert lm.verify_name_similarity("Jane Doe", "Jane Doe") is True

    def test_different_person_returns_false(self, lm):
        lm._mock_llama.return_value = make_llm_response("NO")
        assert lm.verify_name_similarity("Jane Doe", "John Smith") is False

    def test_llm_yes_is_case_insensitive(self, lm):
        lm._mock_llama.return_value = make_llm_response("yes")
        assert lm.verify_name_similarity("Matti Meikalainen", "matti") is True

    def test_fallback_first_token_match(self, lm):
        """First names match → fallback returns True."""
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        assert lm.verify_name_similarity("Jane Doe", "Jane Smith") is True

    def test_fallback_first_token_mismatch(self, lm):
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        assert lm.verify_name_similarity("Jane Doe", "John Smith") is False

    def test_fallback_handles_empty_strings(self, lm):
        lm._mock_llama.side_effect = RuntimeError("LLM crashed")
        assert lm.verify_name_similarity("", "") is False
