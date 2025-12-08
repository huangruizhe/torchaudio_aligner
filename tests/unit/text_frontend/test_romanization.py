"""
Tests for romanization module.

Based on test_text_frontend.ipynb CJK and romanization tests.

Tests cover:
- preprocess_cjk function
- get_available_romanizers
- romanize_text_aligned (word count preservation)
- CJK character splitting
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for text_frontend imports"
)


class TestGetAvailableRomanizers:
    """Tests for get_available_romanizers function."""

    def test_returns_dict(self):
        """Test that get_available_romanizers returns a dict."""
        from text_frontend.romanization import get_available_romanizers

        romanizers = get_available_romanizers()
        assert isinstance(romanizers, dict)

    def test_contains_expected_keys(self):
        """Test that result contains expected romanizer keys."""
        from text_frontend.romanization import get_available_romanizers

        romanizers = get_available_romanizers()
        expected_keys = ["uroman", "cutlet"]

        for key in expected_keys:
            assert key in romanizers


class TestPreprocessCJK:
    """Tests for preprocess_cjk function (CJK character splitting)."""

    def test_chinese_character_split(self):
        """Test Chinese characters are split into individual chars."""
        from text_frontend.romanization import preprocess_cjk

        chinese = "子曰學而時習之"
        processed = preprocess_cjk(chinese)
        chars = processed.split()

        # Each character should become a separate "word"
        assert len(chars) == 7

    def test_japanese_character_split(self):
        """Test Japanese characters are split."""
        from text_frontend.romanization import preprocess_cjk

        japanese = "風立ちぬ"
        processed = preprocess_cjk(japanese)
        chars = processed.split()

        assert len(chars) == 4

    def test_korean_character_split(self):
        """Test Korean characters are split."""
        from text_frontend.romanization import preprocess_cjk

        # Without spaces
        korean = "세계인권"
        processed = preprocess_cjk(korean)
        chars = processed.split()

        # Korean syllables split: 세 계 인 권
        assert len(chars) == 4

    def test_cjk_removes_punctuation(self):
        """Test that CJK preprocessing removes punctuation."""
        from text_frontend.romanization import preprocess_cjk

        chinese_with_punct = "子曰。學而時習之！"
        processed = preprocess_cjk(chinese_with_punct)

        # Should not contain punctuation
        assert "。" not in processed
        assert "！" not in processed

    def test_cjk_removes_whitespace(self):
        """Test that CJK preprocessing removes whitespace."""
        from text_frontend.romanization import preprocess_cjk

        korean = "세계 인권"  # Space in middle
        processed = preprocess_cjk(korean)
        chars = processed.split()

        # All characters should be split
        assert len(chars) == 4

    def test_cjk_mixed_content(self):
        """Test CJK with mixed content (numbers, ASCII)."""
        from text_frontend.romanization import preprocess_cjk

        mixed = "日本語123ABC"
        processed = preprocess_cjk(mixed)
        chars = processed.split()

        # Should split all chars including numbers and ASCII
        assert len(chars) >= 9  # 3 Japanese + 3 numbers + 3 ASCII

    def test_cjk_empty_string(self):
        """Test CJK preprocessing with empty string."""
        from text_frontend.romanization import preprocess_cjk

        result = preprocess_cjk("")
        assert result == ""


class TestAlignRomanizedToOriginal:
    """Tests for align_romanized_to_original function."""

    def test_equal_length(self):
        """Test alignment when lengths are equal."""
        from text_frontend.romanization import align_romanized_to_original

        original = ["hello", "world"]
        romanized = ["hello", "world"]

        result = align_romanized_to_original(original, romanized)
        assert result == romanized

    def test_more_romanized_words(self):
        """Test alignment when more romanized words than original."""
        from text_frontend.romanization import align_romanized_to_original

        original = ["東京"]
        romanized = ["tou", "kyou"]

        result = align_romanized_to_original(original, romanized)
        assert len(result) == 1  # Same as original
        assert result[0] == "toukyou"  # Merged

    def test_fewer_romanized_words(self):
        """Test alignment when fewer romanized words than original."""
        from text_frontend.romanization import align_romanized_to_original

        original = ["a", "b", "c"]
        romanized = ["ab"]

        result = align_romanized_to_original(original, romanized, unk_token="*")
        assert len(result) == 3  # Same as original

    def test_empty_romanized(self):
        """Test alignment with empty romanized list."""
        from text_frontend.romanization import align_romanized_to_original

        original = ["hello", "world"]
        romanized = []

        result = align_romanized_to_original(original, romanized, unk_token="*")
        assert result == ["*", "*"]


class TestRomanizeTextAligned:
    """Tests for romanize_text_aligned function (word count preservation)."""

    def test_romanize_aligned_requires_uroman(self):
        """Test that romanize_text_aligned requires uroman."""
        from text_frontend.romanization import get_available_romanizers

        if not get_available_romanizers()["uroman"]:
            pytest.skip("uroman not available")

        from text_frontend.romanization import romanize_text_aligned

        # Just verify it can be called
        result = romanize_text_aligned("hello world", language="eng")
        assert isinstance(result, str)

    def test_romanize_preserves_word_count(self):
        """Test that romanization preserves word count."""
        from text_frontend.romanization import get_available_romanizers

        if not get_available_romanizers()["uroman"]:
            pytest.skip("uroman not available")

        from text_frontend.romanization import romanize_text_aligned

        text = "A música portuguesa"
        romanized = romanize_text_aligned(text, language="por")

        orig_words = text.split()
        rom_words = romanized.split()

        assert len(orig_words) == len(rom_words)


class TestRomanizeJapanese:
    """Tests for Japanese romanization (if cutlet available)."""

    def test_japanese_romanization_availability(self):
        """Test checking Japanese romanization availability."""
        from text_frontend.romanization import get_available_romanizers

        romanizers = get_available_romanizers()
        assert "cutlet" in romanizers

    def test_japanese_morphological_romanization(self):
        """Test Japanese morphological romanization."""
        from text_frontend.romanization import get_available_romanizers

        if not get_available_romanizers()["cutlet"]:
            pytest.skip("cutlet not available")

        from text_frontend.romanization import romanize_japanese_morphemes

        japanese = "東京"
        result = romanize_japanese_morphemes(japanese)

        # Should return romanized text
        assert isinstance(result, str)
        assert len(result) > 0

    def test_japanese_aligned_romanization(self):
        """Test Japanese romanization with word count preservation."""
        from text_frontend.romanization import get_available_romanizers

        if not get_available_romanizers()["cutlet"]:
            pytest.skip("cutlet not available")

        from text_frontend.romanization import romanize_japanese_morphemes_aligned

        # Pre-split Japanese characters
        text = "東 京"
        result = romanize_japanese_morphemes_aligned(text)

        orig_words = text.split()
        rom_words = result.split()

        assert len(orig_words) == len(rom_words)


class TestRomanizeText:
    """Tests for basic romanize_text function."""

    def test_romanize_requires_uroman(self):
        """Test that romanize_text raises ImportError if uroman missing."""
        from text_frontend.romanization import get_available_romanizers

        if get_available_romanizers()["uroman"]:
            pytest.skip("uroman is available")

        from text_frontend.romanization import romanize_text

        with pytest.raises(ImportError):
            romanize_text("test")
