"""
Tests for AlignedChar class.

Tests cover:
- Constructor and attributes
- to_dict() method
- __repr__() method

Note: Tests import alignment.base lazily to allow pytest collection
even if torch is not available. Tests skip at runtime if torch is missing.
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for alignment.base imports"
)


class TestAlignedCharConstructor:
    """Tests for AlignedChar constructor and basic attributes."""

    def test_required_attributes(self):
        """Test that char, start, end are required."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="h", start=2.0, end=2.1)
        assert char.char == "h"
        assert char.start == 2.0
        assert char.end == 2.1

    def test_default_attributes(self):
        """Test default values for optional attributes."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="h", start=2.0, end=2.1)
        assert char.score == 0.0
        assert char.word_index == -1

    def test_all_attributes(self):
        """Test setting all attributes."""
        from alignment.base import AlignedChar
        char = AlignedChar(
            char="h",
            start=2.0,
            end=2.1,
            score=0.95,
            word_index=5,
        )
        assert char.char == "h"
        assert char.start == 2.0
        assert char.end == 2.1
        assert char.score == 0.95
        assert char.word_index == 5


class TestAlignedCharDuration:
    """Tests for duration calculation."""

    def test_duration_simple(self, aligned_char_simple):
        """Test basic duration calculation."""
        # end - start = 2.1 - 2.0 = 0.1
        duration = aligned_char_simple.end - aligned_char_simple.start
        assert abs(duration - 0.1) < 0.0001

    def test_duration_zero(self):
        """Test char with zero duration."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="x", start=1.0, end=1.0)
        assert char.end - char.start == 0.0

    def test_duration_precise(self):
        """Test precise duration values."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="a", start=0.123, end=0.456)
        duration = char.end - char.start
        assert abs(duration - 0.333) < 0.0001


class TestAlignedCharToDict:
    """Tests for to_dict() method."""

    def test_to_dict_basic(self, aligned_char_simple):
        """Test basic to_dict() output."""
        d = aligned_char_simple.to_dict()
        assert d["char"] == "h"
        assert d["start"] == 2.0
        assert d["end"] == 2.1

    def test_to_dict_with_all_fields(self):
        """Test to_dict() with all fields set."""
        from alignment.base import AlignedChar
        char = AlignedChar(
            char="x",
            start=1.234,
            end=1.567,
            score=0.89,
            word_index=3,
        )
        d = char.to_dict()
        assert d["char"] == "x"
        assert d["start"] == 1.234
        assert d["end"] == 1.567
        assert d["score"] == 0.89

    def test_to_dict_rounding(self):
        """Test that values are rounded appropriately."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="r", start=1.23456789, end=2.34567890)
        d = char.to_dict()
        # Should be rounded to 4 decimal places
        assert d["start"] == 1.2346
        assert d["end"] == 2.3457


class TestAlignedCharRepr:
    """Tests for __repr__() method."""

    def test_repr_simple(self, aligned_char_simple):
        """Test repr for simple char."""
        r = repr(aligned_char_simple)
        assert "AlignedChar" in r
        assert "'h'" in r
        assert "2.000s" in r
        assert "2.100s" in r

    def test_repr_format(self):
        """Test repr format with various chars."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="x", start=0.0, end=0.5)
        r = repr(char)
        assert "AlignedChar('x'" in r


class TestAlignedCharEdgeCases:
    """Edge case tests."""

    def test_space_char(self):
        """Test space character."""
        from alignment.base import AlignedChar
        char = AlignedChar(char=" ", start=1.0, end=1.1)
        assert char.char == " "

    def test_special_char(self):
        """Test special characters."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="'", start=1.0, end=1.1)
        assert char.char == "'"

    def test_unicode_char(self):
        """Test unicode character."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="é", start=1.0, end=1.1)
        assert char.char == "é"

    def test_empty_char(self):
        """Test empty string as char."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="", start=1.0, end=1.1)
        assert char.char == ""

    def test_multi_char_string(self):
        """Test that multi-char string is allowed (caller's responsibility)."""
        from alignment.base import AlignedChar
        # This tests current behavior - multi-char not prevented
        char = AlignedChar(char="abc", start=1.0, end=1.1)
        assert char.char == "abc"

    def test_zero_start_time(self):
        """Test char at time 0."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="s", start=0.0, end=0.1)
        assert char.start == 0.0

    def test_large_times(self):
        """Test with large time values (long audio)."""
        from alignment.base import AlignedChar
        char = AlignedChar(char="l", start=3600.0, end=3600.1)  # 1 hour
        assert char.start == 3600.0
        assert char.end == 3600.1


class TestAlignedCharList:
    """Tests for working with lists of AlignedChars."""

    def test_chars_for_word(self, aligned_chars_for_word):
        """Test list of chars forming a word."""
        assert len(aligned_chars_for_word) == 5
        word = "".join(c.char for c in aligned_chars_for_word)
        assert word == "hello"

    def test_chars_sorted_by_time(self, aligned_chars_for_word):
        """Test that chars are in time order."""
        times = [c.start for c in aligned_chars_for_word]
        assert times == sorted(times)

    def test_chars_contiguous(self, aligned_chars_for_word):
        """Test that chars are contiguous (end of one = start of next)."""
        for i in range(len(aligned_chars_for_word) - 1):
            assert aligned_chars_for_word[i].end == aligned_chars_for_word[i + 1].start

    def test_all_same_word_index(self, aligned_chars_for_word):
        """Test that all chars have same word_index."""
        indices = [c.word_index for c in aligned_chars_for_word]
        assert all(i == 0 for i in indices)
