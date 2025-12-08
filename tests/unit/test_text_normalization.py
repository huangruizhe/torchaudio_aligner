"""
Tests for text normalization functionality.

Based on test_text_frontend.ipynb test cases.

Tests cover:
- Word count preservation through normalization
- Apostrophe normalization (ASCII vs Unicode - the PDF issue)
- Number/currency expansion with word count preservation
- CJK preprocessing
- Romanization alignment
- Multilingual word count preservation

Key invariant: len(original.split()) == len(normalized.split())
This enables lossless recovery via word index for alignment.
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for text_frontend imports"
)


class TestWordCountPreservation:
    """Tests that normalization preserves word count (Test 3 from notebook)."""

    def test_basic_normalization_preserves_word_count(self):
        """Test that basic normalization preserves word count."""
        from text_frontend.normalization import normalize_for_mms

        sample = "Hello, World! This is Q1 2025. Numbers: 123 and symbols."
        normalized = normalize_for_mms(sample)

        orig_words = sample.split()
        norm_words = normalized.split()

        assert len(orig_words) == len(norm_words), \
            f"Word count changed! {len(orig_words)} -> {len(norm_words)}"

    def test_word_by_word_mapping(self):
        """Test that each original word maps to exactly one normalized word."""
        from text_frontend.normalization import normalize_for_mms

        sample = "Hello, World! This is a test."
        normalized = normalize_for_mms(sample)

        orig_words = sample.split()
        norm_words = normalized.split()

        # Same count
        assert len(orig_words) == len(norm_words)

        # Each word maps 1:1
        for i, (orig, norm) in enumerate(zip(orig_words, norm_words)):
            # Normalized word should be lowercase letters, apostrophe, or *
            assert norm, f"Word {i} '{orig}' became empty"

    def test_lossless_recovery_via_index(self):
        """Test that we can recover original words using indices."""
        from text_frontend.normalization import normalize_for_mms

        sample = "Hello, World! This is Q1 2025. Numbers: 123 and symbols."
        normalized = normalize_for_mms(sample)

        orig_words = sample.split()
        norm_words = normalized.split()

        # Simulate alignment result: indices of aligned words
        aligned_indices = [0, 1, 3, 4]

        # Can recover original text using indices
        recovered_original = [orig_words[i] for i in aligned_indices]
        recovered_normalized = [norm_words[i] for i in aligned_indices]

        assert recovered_original == ["Hello,", "World!", "is", "Q1"]
        assert len(recovered_original) == len(recovered_normalized)


class TestApostropheNormalization:
    """Tests for apostrophe/quote normalization (Test 3b++ from notebook).

    CRITICAL: PDFs often contain "smart quotes" (curly quotes) which must be
    normalized to straight apostrophe (') for MMS model compatibility.
    """

    def test_right_single_quote_normalized(self):
        """Test right single quotation mark (U+2019) - most common in PDFs."""
        from text_frontend.normalization import _normalize_word_for_mms

        # U+2019 Right single quotation mark
        result = _normalize_word_for_mms("Meta's")  # curly apostrophe
        assert result == "meta's", f"Expected 'meta's', got '{result}'"

    def test_contractions_with_curly_quotes(self):
        """Test common contractions with curly quotes."""
        from text_frontend.normalization import _normalize_word_for_mms

        # These use U+2019 (right single quotation mark)
        test_cases = [
            ("it's", "it's"),
            ("don't", "don't"),
            ("we're", "we're"),
            ("I'm", "i'm"),
            ("can't", "can't"),
        ]

        for input_word, expected in test_cases:
            result = _normalize_word_for_mms(input_word)
            assert result == expected, \
                f"'{input_word}' -> expected '{expected}', got '{result}'"

    def test_left_single_quote(self):
        """Test left single quotation mark (U+2018)."""
        from text_frontend.normalization import _normalize_word_for_mms

        # U+2018 Left single quotation mark
        result = _normalize_word_for_mms("'twas")
        assert "'" in result or result == "'twas"

    def test_straight_apostrophe_passthrough(self):
        """Test straight apostrophe (U+0027) passes through."""
        from text_frontend.normalization import _normalize_word_for_mms

        result = _normalize_word_for_mms("Meta's")  # straight apostrophe
        assert result == "meta's"

    def test_backtick_normalized(self):
        """Test backtick/grave accent (U+0060) normalized."""
        from text_frontend.normalization import _normalize_word_for_mms

        result = _normalize_word_for_mms("Meta`s")
        assert result == "meta's", f"Expected 'meta's', got '{result}'"

    def test_acute_accent_normalized(self):
        """Test acute accent (U+00B4) normalized."""
        from text_frontend.normalization import _normalize_word_for_mms

        result = _normalize_word_for_mms("Meta´s")
        assert result == "meta's", f"Expected 'meta's', got '{result}'"

    def test_full_sentence_with_mixed_apostrophes(self):
        """Test full sentence with mixed apostrophe variants."""
        from text_frontend.normalization import normalize_for_mms

        # Contains curly quotes from PDF-style text
        mixed_sentence = "Meta's Q1 results show it's growing. We're confident."
        normalized = normalize_for_mms(mixed_sentence)

        # No words should become * due to apostrophe issues
        norm_words = normalized.split()
        star_words = [w for w in norm_words if w == "*"]
        assert len(star_words) == 0, \
            f"Words became '*' due to apostrophe issues: {star_words}"

        # Word count preserved
        orig_words = mixed_sentence.split()
        assert len(orig_words) == len(norm_words)


class TestNumberExpansion:
    """Tests for number/currency expansion (Test 3b, 3b+ from notebook)."""

    def test_number_expansion_preserves_word_count(self):
        """Test that number expansion preserves word count with word_joiner."""
        from text_frontend.normalization import expand_numbers_in_text

        sample = "The price is $66 and we sold 123 items."
        expanded = expand_numbers_in_text(sample, word_joiner="")

        orig_words = sample.split()
        expanded_words = expanded.split()

        assert len(orig_words) == len(expanded_words), \
            f"Word count changed! {len(orig_words)} -> {len(expanded_words)}"

    def test_currency_dollar(self):
        """Test dollar currency expansion."""
        from text_frontend.normalization import expand_number

        result = expand_number("$66", word_joiner="")
        assert "sixtysixdollars" in result.lower()

    def test_currency_with_cents(self):
        """Test currency with cents."""
        from text_frontend.normalization import expand_number

        result = expand_number("$7.50", word_joiner="")
        assert "dollars" in result.lower()
        assert "cents" in result.lower() or "fifty" in result.lower()

    def test_euro_currency(self):
        """Test euro currency."""
        from text_frontend.normalization import expand_number

        result = expand_number("€100", word_joiner="")
        assert "euro" in result.lower()

    def test_percentage(self):
        """Test percentage expansion."""
        from text_frontend.normalization import expand_number

        result = expand_number("50%", word_joiner="")
        assert "percent" in result.lower()

    def test_ordinal(self):
        """Test ordinal expansion."""
        from text_frontend.normalization import expand_number

        test_cases = [
            ("1st", "first"),
            ("2nd", "second"),
            ("3rd", "third"),
            ("21st", "twentyfirst"),
        ]

        for input_num, expected_pattern in test_cases:
            result = expand_number(input_num, word_joiner="")
            assert expected_pattern in result.lower(), \
                f"'{input_num}' -> expected '{expected_pattern}' in '{result}'"

    def test_decimal(self):
        """Test decimal expansion."""
        from text_frontend.normalization import expand_number

        result = expand_number("3.14", word_joiner="")
        assert "point" in result.lower() or "three" in result.lower()

    def test_mixed_letter_number(self):
        """Test mixed letter-number like COVID19, B2B."""
        from text_frontend.normalization import expand_number

        test_cases = [
            ("COVID19", "nineteen"),
            ("MP3", "three"),
            ("4K", "four"),
        ]

        for input_val, expected_pattern in test_cases:
            result = expand_number(input_val, word_joiner="")
            assert expected_pattern in result.lower(), \
                f"'{input_val}' -> expected '{expected_pattern}' in '{result}'"

    def test_comma_separated_numbers(self):
        """Test comma-separated numbers like 1,000,000."""
        from text_frontend.normalization import expand_number

        result = expand_number("1,000", word_joiner="")
        assert "thousand" in result.lower()

        result = expand_number("1,000,000", word_joiner="")
        assert "million" in result.lower()

    def test_full_normalization_with_tn(self):
        """Test full normalization with text normalization enabled."""
        from text_frontend.normalization import normalize_for_mms

        sample = "The price is $66 and we sold 123 items for €7.50 each."
        normalized = normalize_for_mms(sample, expand_numbers=True, word_joiner="")

        orig_words = sample.split()
        norm_words = normalized.split()

        # Word count MUST be preserved
        assert len(orig_words) == len(norm_words), \
            f"Word count changed! {len(orig_words)} -> {len(norm_words)}"


class TestCJKPreprocessing:
    """Tests for CJK (Chinese/Japanese/Korean) preprocessing."""

    def test_chinese_character_split(self):
        """Test Chinese characters are split into individual chars."""
        from text_frontend.romanization import preprocess_cjk

        chinese = "子曰學而時習之"
        processed = preprocess_cjk(chinese)
        chars = processed.split()

        # Each character should become a separate "word"
        assert len(chars) == 7, f"Expected 7 chars, got {len(chars)}"

    def test_japanese_character_split(self):
        """Test Japanese characters are split."""
        from text_frontend.romanization import preprocess_cjk

        japanese = "風立ちぬ"
        processed = preprocess_cjk(japanese)
        chars = processed.split()

        assert len(chars) == 4, f"Expected 4 chars, got {len(chars)}"

    def test_korean_character_split(self):
        """Test Korean characters are split."""
        from text_frontend.romanization import preprocess_cjk

        korean = "세계 인권"
        processed = preprocess_cjk(korean)
        chars = processed.split()

        # Korean syllables split: 세 계 인 권
        assert len(chars) >= 4


class TestRomanizationAlignment:
    """Tests for romanization with word count preservation."""

    def test_romanize_text_aligned_preserves_count(self):
        """Test romanize_text_aligned preserves word count."""
        try:
            from text_frontend.romanization import romanize_text_aligned

            # Portuguese with accents
            portuguese = "A música portuguesa é bonita"
            romanized = romanize_text_aligned(portuguese, language="por")

            orig_words = portuguese.split()
            rom_words = romanized.split()

            assert len(orig_words) == len(rom_words), \
                f"Word count changed! {len(orig_words)} -> {len(rom_words)}"
        except ImportError:
            pytest.skip("uroman not available")

    def test_romanize_chinese_with_cjk_split(self):
        """Test romanizing Chinese after CJK split."""
        try:
            from text_frontend.romanization import preprocess_cjk, romanize_text_aligned

            chinese = "子曰學"
            processed = preprocess_cjk(chinese)
            romanized = romanize_text_aligned(processed, language="cmn")

            processed_words = processed.split()
            rom_words = romanized.split()

            assert len(processed_words) == len(rom_words), \
                f"Word count changed! {len(processed_words)} -> {len(rom_words)}"
        except ImportError:
            pytest.skip("uroman not available")


class TestMultilingualWordCount:
    """Tests for multilingual word count preservation (Test 3c from notebook)."""

    def test_english_word_count(self):
        """Test English text normalization preserves word count."""
        from text_frontend.normalization import normalize_for_mms

        sample = "Hello World! The price is $123."
        normalized = normalize_for_mms(sample, expand_numbers=True, word_joiner="")

        orig_words = sample.split()
        norm_words = normalized.split()

        assert len(orig_words) == len(norm_words)

    def test_portuguese_word_count(self):
        """Test Portuguese text with accents."""
        from text_frontend.normalization import normalize_for_mms

        sample = "A música portuguesa é muito bonita"
        normalized = normalize_for_mms(sample)

        orig_words = sample.split()
        norm_words = normalized.split()

        assert len(orig_words) == len(norm_words)

    def test_hindi_word_count(self):
        """Test Hindi (Devanagari) text."""
        try:
            from text_frontend.romanization import romanize_text_aligned
            from text_frontend.normalization import normalize_for_mms

            sample = "मानव अधिकारों की"
            romanized = romanize_text_aligned(sample, language="hin")
            normalized = normalize_for_mms(romanized)

            orig_words = sample.split()
            norm_words = normalized.split()

            assert len(orig_words) == len(norm_words), \
                f"Hindi word count changed! {len(orig_words)} -> {len(norm_words)}"
        except ImportError:
            pytest.skip("uroman not available")


class TestEdgeCases:
    """Edge case tests for normalization."""

    def test_empty_string(self):
        """Test empty string normalization."""
        from text_frontend.normalization import normalize_for_mms

        result = normalize_for_mms("")
        assert result == ""

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        from text_frontend.normalization import normalize_for_mms

        result = normalize_for_mms("   ")
        assert result.strip() == ""

    def test_single_word(self):
        """Test single word normalization."""
        from text_frontend.normalization import normalize_for_mms

        result = normalize_for_mms("Hello")
        assert result == "hello"

    def test_pure_numbers_become_star(self):
        """Test pure numbers become * without expansion."""
        from text_frontend.normalization import _normalize_word_for_mms

        result = _normalize_word_for_mms("123")
        # Without number expansion, pure numbers may become *
        assert result in ["*", "onetwothree", "onehundredtwentythree"]

    def test_emoji_becomes_star(self):
        """Test emoji becomes * (unknown)."""
        from text_frontend.normalization import _normalize_word_for_mms

        result = _normalize_word_for_mms("😀")
        assert result == "*"

    def test_special_characters_removed(self):
        """Test special characters are removed, keeping letters."""
        from text_frontend.normalization import _normalize_word_for_mms

        result = _normalize_word_for_mms("hello!")
        assert result == "hello"

        result = _normalize_word_for_mms("world?")
        assert result == "world"
