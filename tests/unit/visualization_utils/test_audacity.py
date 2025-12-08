"""
Tests for Audacity label export utilities.

Tests cover:
- get_audacity_labels function
- save_audacity_labels function
- Label format validation
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for visualization_utils imports"
)


class TestGetAudacityLabels:
    """Tests for get_audacity_labels function."""

    def test_basic_labels(self):
        """Test basic label generation."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        word_alignment = {
            0: AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
            1: AlignedWord(word="world", start_frame=160, end_frame=220, index=1),
        }

        labels = get_audacity_labels(word_alignment)

        # Should have 2 lines
        lines = labels.strip().split("\n")
        assert len(lines) == 2

    def test_label_format(self):
        """Test label format is start<TAB>end<TAB>word."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        word_alignment = {
            0: AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
        }

        labels = get_audacity_labels(word_alignment)
        parts = labels.strip().split("\t")

        assert len(parts) == 3
        # First two should be numbers (start, end times)
        assert float(parts[0]) >= 0
        assert float(parts[1]) > float(parts[0])
        # Third should be the word
        assert parts[2] == "hello"

    def test_time_conversion(self):
        """Test frame to seconds conversion."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        # Frame 100 with 0.02s frame duration = 2.0s
        word_alignment = {
            0: AlignedWord(word="test", start_frame=100, end_frame=150, index=0),
        }

        labels = get_audacity_labels(word_alignment, frame_duration=0.02)
        parts = labels.strip().split("\t")

        assert float(parts[0]) == 2.0  # 100 * 0.02
        assert float(parts[1]) == 3.0  # 150 * 0.02

    def test_custom_frame_duration(self):
        """Test with custom frame duration."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        word_alignment = {
            0: AlignedWord(word="test", start_frame=100, end_frame=150, index=0),
        }

        labels = get_audacity_labels(word_alignment, frame_duration=0.01)
        parts = labels.strip().split("\t")

        assert float(parts[0]) == 1.0  # 100 * 0.01
        assert float(parts[1]) == 1.5  # 150 * 0.01

    def test_sorted_by_word_index(self):
        """Test that labels are sorted by word index."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        # Out of order word indices
        word_alignment = {
            2: AlignedWord(word="third", start_frame=300, end_frame=350, index=2),
            0: AlignedWord(word="first", start_frame=100, end_frame=150, index=0),
            1: AlignedWord(word="second", start_frame=200, end_frame=250, index=1),
        }

        labels = get_audacity_labels(word_alignment)
        lines = labels.strip().split("\n")

        # Check order
        assert "first" in lines[0]
        assert "second" in lines[1]
        assert "third" in lines[2]

    def test_empty_word_fallback(self):
        """Test fallback to index when word is empty."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        word_alignment = {
            0: AlignedWord(word="", start_frame=100, end_frame=150, index=0),
        }

        labels = get_audacity_labels(word_alignment)
        parts = labels.strip().split("\t")

        # Should use [0] as label
        assert parts[2] == "[0]"


class TestSaveAudacityLabels:
    """Tests for save_audacity_labels function."""

    def test_save_creates_file(self, tmp_path):
        """Test that save_audacity_labels creates a file."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import save_audacity_labels

        word_alignment = {
            0: AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
        }

        output_path = tmp_path / "labels.txt"
        result_path = save_audacity_labels(word_alignment, output_path)

        assert output_path.exists()
        assert result_path == str(output_path)

    def test_save_content_matches(self, tmp_path):
        """Test that saved content matches get_audacity_labels."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import save_audacity_labels, get_audacity_labels

        word_alignment = {
            0: AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
            1: AlignedWord(word="world", start_frame=160, end_frame=220, index=1),
        }

        output_path = tmp_path / "labels.txt"
        save_audacity_labels(word_alignment, output_path)

        # Read back and compare
        with open(output_path) as f:
            saved_content = f.read()

        expected_content = get_audacity_labels(word_alignment)
        assert saved_content == expected_content

    def test_save_utf8_encoding(self, tmp_path):
        """Test that file is saved with UTF-8 encoding."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import save_audacity_labels

        word_alignment = {
            0: AlignedWord(word="café", start_frame=100, end_frame=150, index=0),
        }

        output_path = tmp_path / "labels.txt"
        save_audacity_labels(word_alignment, output_path)

        # Read back with UTF-8
        with open(output_path, encoding="utf-8") as f:
            content = f.read()

        assert "café" in content

    def test_save_accepts_path_object(self, tmp_path):
        """Test that save accepts Path objects."""
        from pathlib import Path
        from alignment.base import AlignedWord
        from visualization_utils.audacity import save_audacity_labels

        word_alignment = {
            0: AlignedWord(word="test", start_frame=100, end_frame=150, index=0),
        }

        output_path = Path(tmp_path) / "labels.txt"
        save_audacity_labels(word_alignment, output_path)

        assert output_path.exists()

    def test_save_accepts_string_path(self, tmp_path):
        """Test that save accepts string paths."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import save_audacity_labels

        word_alignment = {
            0: AlignedWord(word="test", start_frame=100, end_frame=150, index=0),
        }

        output_path = str(tmp_path / "labels.txt")
        save_audacity_labels(word_alignment, output_path)

        assert (tmp_path / "labels.txt").exists()
