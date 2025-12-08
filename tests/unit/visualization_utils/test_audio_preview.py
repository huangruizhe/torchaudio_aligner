"""
Tests for audio preview utilities.

These tests are limited since IPython.display.Audio is not available
outside of notebook environments.

Tests cover:
- Function existence and signatures
- Basic parameter validation
- Non-IPython functionality (sample extraction)
"""

import pytest
from unittest.mock import Mock, patch

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for visualization_utils imports"
)


class TestPreviewWordSecondsFunction:
    """Tests for preview_word_seconds function signature."""

    def test_function_exists(self):
        """Test that preview_word_seconds function exists."""
        from visualization_utils.audio_preview import preview_word_seconds

        assert callable(preview_word_seconds)

    @patch("visualization_utils.audio_preview.Audio")
    def test_basic_call(self, mock_audio):
        """Test basic function call with mocked IPython."""
        import torch
        from alignment.base import AlignedWord
        from visualization_utils.audio_preview import preview_word_seconds

        waveform = torch.randn(16000)  # 1 second
        word = AlignedWord(word="hello", start_frame=0, end_frame=25, index=0)

        mock_audio.return_value = Mock()

        result = preview_word_seconds(waveform, word, sample_rate=16000)

        # Should have called Audio
        assert mock_audio.called

    @patch("visualization_utils.audio_preview.Audio")
    def test_with_padding(self, mock_audio):
        """Test preview_word_seconds with padding."""
        import torch
        from alignment.base import AlignedWord
        from visualization_utils.audio_preview import preview_word_seconds

        waveform = torch.randn(32000)  # 2 seconds
        word = AlignedWord(word="hello", start_frame=25, end_frame=50, index=0)

        mock_audio.return_value = Mock()

        result = preview_word_seconds(waveform, word, padding=0.1)

        assert mock_audio.called


class TestPreviewSegmentSecondsFunction:
    """Tests for preview_segment_seconds function."""

    def test_function_exists(self):
        """Test that preview_segment_seconds function exists."""
        from visualization_utils.audio_preview import preview_segment_seconds

        assert callable(preview_segment_seconds)

    @patch("visualization_utils.audio_preview.Audio")
    def test_empty_words(self, mock_audio):
        """Test preview_segment_seconds with empty words list."""
        import torch
        from visualization_utils.audio_preview import preview_segment_seconds

        waveform = torch.randn(16000)

        audio, text = preview_segment_seconds(waveform, [])

        assert audio is None
        assert text == ""

    @patch("visualization_utils.audio_preview.Audio")
    def test_with_words(self, mock_audio):
        """Test preview_segment_seconds with words."""
        import torch
        from alignment.base import AlignedWord
        from visualization_utils.audio_preview import preview_segment_seconds

        waveform = torch.randn(32000)  # 2 seconds
        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=25, index=0),
            AlignedWord(word="world", start_frame=30, end_frame=50, index=1),
        ]

        mock_audio.return_value = Mock()

        audio, text = preview_segment_seconds(waveform, words)

        assert mock_audio.called
        assert "hello" in text
        assert "world" in text


class TestPreviewRandomSegmentSecondsFunction:
    """Tests for preview_random_segment_seconds function."""

    def test_function_exists(self):
        """Test that preview_random_segment_seconds function exists."""
        from visualization_utils.audio_preview import preview_random_segment_seconds

        assert callable(preview_random_segment_seconds)


class TestLegacyPreviewWord:
    """Tests for legacy preview_word function."""

    def test_function_exists(self):
        """Test that preview_word function exists."""
        from visualization_utils.audio_preview import preview_word

        assert callable(preview_word)

    @patch("visualization_utils.audio_preview.Audio")
    def test_word_not_found(self, mock_audio, capsys):
        """Test preview_word with missing word index."""
        import torch
        from visualization_utils.audio_preview import preview_word

        waveform = torch.randn(16000)
        word_alignment = {}  # Empty

        result = preview_word(waveform, word_alignment, word_idx=0)

        assert result is None
        captured = capsys.readouterr()
        assert "not found" in captured.out

    @patch("visualization_utils.audio_preview.Audio")
    def test_basic_preview(self, mock_audio):
        """Test basic preview_word call."""
        import torch
        from alignment.base import AlignedWord
        from visualization_utils.audio_preview import preview_word

        waveform = torch.randn(32000)  # 2 seconds
        word_alignment = {
            0: AlignedWord(word="hello", start_frame=25, end_frame=50, index=0),
        }

        mock_audio.return_value = Mock()

        result = preview_word(waveform, word_alignment, word_idx=0)

        assert mock_audio.called


class TestLegacyPreviewSegment:
    """Tests for legacy preview_segment function."""

    def test_function_exists(self):
        """Test that preview_segment function exists."""
        from visualization_utils.audio_preview import preview_segment

        assert callable(preview_segment)

    @patch("visualization_utils.audio_preview.Audio")
    def test_out_of_range(self, mock_audio, capsys):
        """Test preview_segment with out of range start_idx."""
        import torch
        from visualization_utils.audio_preview import preview_segment

        waveform = torch.randn(16000)
        word_alignment = {}

        result, words = preview_segment(waveform, word_alignment, start_idx=100)

        assert result is None
        assert words == []


class TestPreviewWordByIndex:
    """Tests for preview_word_by_index function."""

    def test_function_exists(self):
        """Test that preview_word_by_index function exists."""
        from visualization_utils.audio_preview import preview_word_by_index

        assert callable(preview_word_by_index)

    @patch("visualization_utils.audio_preview.preview_word")
    def test_out_of_range(self, mock_preview, capsys):
        """Test preview_word_by_index with out of range index."""
        import torch
        from visualization_utils.audio_preview import preview_word_by_index

        waveform = torch.randn(16000)
        word_alignment = {0: Mock()}

        result = preview_word_by_index(waveform, word_alignment, alignment_idx=100)

        assert result is None
        captured = capsys.readouterr()
        assert "out of range" in captured.out


class TestPreviewAllWords:
    """Tests for preview_all_words function."""

    def test_function_exists(self):
        """Test that preview_all_words function exists."""
        from visualization_utils.audio_preview import preview_all_words

        assert callable(preview_all_words)
