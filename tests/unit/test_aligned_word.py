"""
Tests for AlignedWord class.

Tests cover:
- Constructor and attributes
- start_seconds() / end_seconds() / duration_seconds() methods
- duration_frames property
- display_text property
- to_dict() method
- __repr__() method

Note: All tests import alignment.base inside test methods or fixtures
to allow pytest to collect tests even if torch is not available.
Tests will be skipped at runtime if torch is missing.
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for alignment.base imports"
)


class TestAlignedWordConstructor:
    """Tests for AlignedWord constructor and basic attributes."""

    def test_required_attributes(self):
        """Test that word, start_frame, end_frame are required."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="hello", start_frame=100, end_frame=150)
        assert word.word == "hello"
        assert word.start_frame == 100
        assert word.end_frame == 150

    def test_default_attributes(self):
        """Test default values for optional attributes."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="hello", start_frame=100, end_frame=150)
        assert word.score == 0.0
        assert word.original is None
        assert word.index == -1
        assert word.chars == []

    def test_all_attributes(self):
        """Test setting all attributes."""
        from alignment.base import AlignedWord, AlignedChar
        chars = [AlignedChar(char="h", start=2.0, end=2.1)]
        word = AlignedWord(
            word="hello",
            start_frame=100,
            end_frame=150,
            score=0.95,
            original="Hello!",
            index=5,
            chars=chars,
        )
        assert word.word == "hello"
        assert word.start_frame == 100
        assert word.end_frame == 150
        assert word.score == 0.95
        assert word.original == "Hello!"
        assert word.index == 5
        assert word.chars == chars


class TestAlignedWordTimeMethods:
    """Tests for time conversion methods."""

    def test_start_seconds_default_duration(self, aligned_word_simple):
        """Test start_seconds() with default frame duration."""
        # 100 frames * 0.02 seconds/frame = 2.0 seconds
        assert aligned_word_simple.start_seconds() == 2.0

    def test_end_seconds_default_duration(self, aligned_word_simple):
        """Test end_seconds() with default frame duration."""
        # 150 frames * 0.02 seconds/frame = 3.0 seconds
        assert aligned_word_simple.end_seconds() == 3.0

    def test_duration_seconds_default_duration(self, aligned_word_simple):
        """Test duration_seconds() with default frame duration."""
        # (150 - 100) frames * 0.02 seconds/frame = 1.0 seconds
        assert aligned_word_simple.duration_seconds() == 1.0

    def test_start_seconds_custom_duration(self, aligned_word_simple):
        """Test start_seconds() with custom frame duration."""
        # 100 frames * 0.01 seconds/frame = 1.0 seconds
        assert aligned_word_simple.start_seconds(frame_duration=0.01) == 1.0

    def test_end_seconds_custom_duration(self, aligned_word_simple):
        """Test end_seconds() with custom frame duration."""
        # 150 frames * 0.01 seconds/frame = 1.5 seconds
        assert aligned_word_simple.end_seconds(frame_duration=0.01) == 1.5

    def test_duration_seconds_custom_duration(self, aligned_word_simple):
        """Test duration_seconds() with custom frame duration."""
        # (150 - 100) frames * 0.01 seconds/frame = 0.5 seconds
        assert aligned_word_simple.duration_seconds(frame_duration=0.01) == 0.5

    def test_zero_start_frame(self):
        """Test word starting at frame 0."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="start", start_frame=0, end_frame=50)
        assert word.start_seconds() == 0.0
        assert word.end_seconds() == 1.0

    def test_large_frame_numbers(self):
        """Test with large frame numbers (long audio)."""
        from alignment.base import AlignedWord
        # 1 hour at 50fps = 180000 frames
        word = AlignedWord(word="late", start_frame=180000, end_frame=180050)
        assert word.start_seconds() == 3600.0  # 1 hour
        assert word.end_seconds() == 3601.0


class TestAlignedWordDurationFrames:
    """Tests for duration_frames property."""

    def test_duration_frames_simple(self, aligned_word_simple):
        """Test duration_frames property."""
        assert aligned_word_simple.duration_frames == 50  # 150 - 100

    def test_duration_frames_zero(self):
        """Test word with zero duration."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="instant", start_frame=100, end_frame=100)
        assert word.duration_frames == 0

    def test_duration_frames_large(self):
        """Test word with large duration."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="long", start_frame=0, end_frame=1000)
        assert word.duration_frames == 1000


class TestAlignedWordDisplayText:
    """Tests for display_text property."""

    def test_display_text_no_original(self, aligned_word_simple):
        """Test display_text when original is None."""
        assert aligned_word_simple.display_text == "hello"

    def test_display_text_same_as_word(self):
        """Test display_text when original equals word."""
        from alignment.base import AlignedWord
        word = AlignedWord(
            word="hello",
            start_frame=100,
            end_frame=150,
            original="hello",
        )
        assert word.display_text == "hello"

    def test_display_text_different_original(self, aligned_word_with_original):
        """Test display_text when original differs from word."""
        # original="Hello!", word="hello"
        assert aligned_word_with_original.display_text == "Hello! (hello)"

    def test_display_text_with_punctuation(self):
        """Test display_text with punctuation in original."""
        from alignment.base import AlignedWord
        word = AlignedWord(
            word="dont",
            start_frame=100,
            end_frame=150,
            original="Don't",
        )
        assert word.display_text == "Don't (dont)"

    def test_display_text_case_difference(self):
        """Test display_text with only case difference."""
        from alignment.base import AlignedWord
        word = AlignedWord(
            word="hello",
            start_frame=100,
            end_frame=150,
            original="HELLO",
        )
        assert word.display_text == "HELLO (hello)"


class TestAlignedWordConfidence:
    """Tests for confidence property (alias for score)."""

    def test_confidence_equals_score(self, aligned_word_with_score):
        """Test that confidence property returns score."""
        assert aligned_word_with_score.confidence == aligned_word_with_score.score
        assert aligned_word_with_score.confidence == 0.88

    def test_confidence_default(self, aligned_word_simple):
        """Test default confidence (0.0)."""
        assert aligned_word_simple.confidence == 0.0


class TestAlignedWordToDict:
    """Tests for to_dict() method."""

    def test_to_dict_basic(self, aligned_word_simple):
        """Test basic to_dict() output."""
        d = aligned_word_simple.to_dict()
        assert d["word"] == "hello"
        assert d["start"] == 2.0  # 100 * 0.02
        assert d["end"] == 3.0    # 150 * 0.02
        assert "original" not in d  # None original not included

    def test_to_dict_with_original(self, aligned_word_with_original):
        """Test to_dict() includes original when different."""
        d = aligned_word_with_original.to_dict()
        assert d["word"] == "hello"
        assert d["original"] == "Hello!"
        assert d["score"] == 0.95

    def test_to_dict_with_score(self, aligned_word_with_score):
        """Test to_dict() includes non-zero score."""
        d = aligned_word_with_score.to_dict()
        assert d["score"] == 0.88

    def test_to_dict_custom_frame_duration(self, aligned_word_simple):
        """Test to_dict() with custom frame duration."""
        d = aligned_word_simple.to_dict(frame_duration=0.01)
        assert d["start"] == 1.0  # 100 * 0.01
        assert d["end"] == 1.5    # 150 * 0.01

    def test_to_dict_rounding(self):
        """Test that times are rounded to 3 decimal places."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="test", start_frame=33, end_frame=67)
        d = word.to_dict()
        # 33 * 0.02 = 0.66, 67 * 0.02 = 1.34
        assert d["start"] == 0.66
        assert d["end"] == 1.34


class TestAlignedWordRepr:
    """Tests for __repr__() method."""

    def test_repr_simple(self, aligned_word_simple):
        """Test repr for simple word."""
        r = repr(aligned_word_simple)
        assert "AlignedWord" in r
        assert "hello" in r
        assert "2.00s" in r
        assert "3.00s" in r

    def test_repr_with_original(self, aligned_word_with_original):
        """Test repr shows original when different."""
        r = repr(aligned_word_with_original)
        assert "Hello!" in r
        assert "hello" in r
        assert "score=0.95" in r

    def test_repr_with_score(self, aligned_word_with_score):
        """Test repr includes score when non-zero."""
        r = repr(aligned_word_with_score)
        assert "score=0.88" in r

    def test_repr_no_score_when_zero(self, aligned_word_simple):
        """Test repr omits score when zero."""
        r = repr(aligned_word_simple)
        assert "score=" not in r


class TestAlignedWordEdgeCases:
    """Edge case tests."""

    def test_empty_word(self):
        """Test word with empty string."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="", start_frame=0, end_frame=10)
        assert word.word == ""
        assert word.display_text == ""

    def test_unicode_word(self):
        """Test word with unicode characters."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="café", start_frame=0, end_frame=10)
        assert word.word == "café"

    def test_very_short_duration(self):
        """Test word with 1 frame duration."""
        from alignment.base import AlignedWord
        word = AlignedWord(word="blip", start_frame=100, end_frame=101)
        assert word.duration_frames == 1
        assert word.duration_seconds() == 0.02

    def test_negative_frames_not_prevented(self):
        """Test that negative frames are allowed (caller's responsibility)."""
        from alignment.base import AlignedWord
        # This tests current behavior - negative frames aren't prevented
        word = AlignedWord(word="bug", start_frame=-10, end_frame=10)
        assert word.start_frame == -10
        assert word.duration_frames == 20
