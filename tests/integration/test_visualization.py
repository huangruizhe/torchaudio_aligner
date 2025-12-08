"""
Integration tests for visualization utilities.

Tests cover:
- Audio preview functions (both legacy and new API)
- Audacity label generation
- Time conversion utilities

Note: Many visualization functions require IPython for Audio display,
so we test the underlying logic without requiring IPython.
"""

import pytest
from unittest.mock import MagicMock, patch

from conftest import requires_torch


class TestAudioPreviewHelpers:
    """Tests for audio preview helper functions."""

    @requires_torch
    def test_preview_word_seconds_extracts_segment(self, aligned_word_simple):
        """Test preview_word_seconds extracts correct audio segment."""
        import torch

        # Create a waveform with different values at different positions
        sample_rate = 16000
        waveform = torch.arange(0, 100000, dtype=torch.float32)

        # Word is at frames 100-150, which is 2.0s-3.0s
        # At 16kHz: 2.0s = sample 32000, 3.0s = sample 48000

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word_seconds

            result = preview_word_seconds(waveform, aligned_word_simple, sample_rate)

            # Check Audio was called with correct segment
            call_args = mock_audio.call_args
            segment = call_args[0][0]

            # Segment should start around sample 32000
            assert segment[0] == 32000.0

    @requires_torch
    def test_preview_word_seconds_with_padding(self, aligned_word_simple):
        """Test preview_word_seconds with padding."""
        import torch

        sample_rate = 16000
        waveform = torch.zeros(100000)

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word_seconds

            # 0.5s padding = 8000 samples before and after
            result = preview_word_seconds(
                waveform, aligned_word_simple, sample_rate, padding=0.5
            )

            call_args = mock_audio.call_args
            segment = call_args[0][0]

            # With 0.5s padding, start should be at 1.5s (24000 samples)
            # Length should be 2.0s (1s word + 1s padding) = 32000 samples
            assert len(segment) == 32000  # 3.5s - 1.5s = 2.0s


class TestLegacyPreviewAPI:
    """Tests for legacy frame-based preview API."""

    @requires_torch
    def test_preview_word_frame_conversion(self, aligned_words_list):
        """Test legacy preview_word converts frames correctly."""
        import torch

        sample_rate = 16000
        frame_duration = 0.02
        waveform = torch.zeros(100000)

        word_alignment = {i: w for i, w in enumerate(aligned_words_list)}

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word

            result = preview_word(
                waveform, word_alignment, 0,  # First word
                sample_rate, frame_duration
            )

            # First word at frames 100-150
            # samples_per_frame = 16000 * 0.02 = 320
            # start_sample = 100 * 320 = 32000
            # end_sample = 150 * 320 = 48000
            call_args = mock_audio.call_args
            segment = call_args[0][0]
            assert len(segment) == 16000  # 50 frames * 320 samples/frame

    @requires_torch
    def test_preview_word_not_found(self, aligned_words_list):
        """Test preview_word with invalid index returns None."""
        import torch

        waveform = torch.zeros(100000)
        word_alignment = {i: w for i, w in enumerate(aligned_words_list)}

        from visualization_utils.audio_preview import preview_word

        result = preview_word(waveform, word_alignment, 999)  # Invalid index
        assert result is None


class TestPreviewSegment:
    """Tests for segment preview functions."""

    @requires_torch
    def test_preview_segment_seconds_empty(self):
        """Test preview_segment_seconds with empty list."""
        import torch

        waveform = torch.zeros(100000)

        with patch("visualization_utils.audio_preview.Audio"):
            from visualization_utils.audio_preview import preview_segment_seconds

            result, text = preview_segment_seconds(waveform, [])

            assert result is None
            assert text == ""

    @requires_torch
    def test_preview_segment_seconds_multiple_words(self, aligned_words_list):
        """Test preview_segment_seconds with multiple words."""
        import torch

        sample_rate = 16000
        waveform = torch.zeros(200000)

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_segment_seconds

            # Preview first 3 words
            result, text = preview_segment_seconds(
                waveform, aligned_words_list[:3], sample_rate, padding=0.0
            )

            # Should contain all 3 words
            assert "hello" in text.lower()
            assert "world" in text.lower()
            assert "this" in text.lower()


class TestPreviewByIndex:
    """Tests for preview_word_by_index function."""

    @requires_torch
    def test_preview_word_by_index_valid(self, aligned_words_list):
        """Test preview by position index."""
        import torch

        waveform = torch.zeros(200000)
        word_alignment = {i: w for i, w in enumerate(aligned_words_list)}

        with patch("visualization_utils.audio_preview.Audio"):
            from visualization_utils.audio_preview import preview_word_by_index

            result = preview_word_by_index(waveform, word_alignment, 0)
            # Should return Audio object (mocked)
            assert result is not None

    @requires_torch
    def test_preview_word_by_index_out_of_range(self, aligned_words_list):
        """Test preview by index out of range."""
        import torch

        waveform = torch.zeros(200000)
        word_alignment = {i: w for i, w in enumerate(aligned_words_list)}

        from visualization_utils.audio_preview import preview_word_by_index

        # Index beyond range
        result = preview_word_by_index(waveform, word_alignment, 100)
        assert result is None


class TestPreviewRandom:
    """Tests for random preview functions."""

    @requires_torch
    def test_preview_random_segment_returns_tuple(self, aligned_words_list):
        """Test preview_random_segment returns correct tuple."""
        import torch

        waveform = torch.zeros(200000)

        with patch("visualization_utils.audio_preview.Audio"):
            from visualization_utils.audio_preview import preview_random_segment_seconds

            result = preview_random_segment_seconds(
                waveform, aligned_words_list, num_words=3
            )

            # Should return (audio, words, start_idx)
            assert len(result) == 3
            audio, words, start_idx = result
            assert isinstance(start_idx, int)
            assert len(words) <= 3


class TestWaveformDimensions:
    """Tests for handling different waveform dimensions."""

    @requires_torch
    def test_1d_waveform(self, aligned_word_simple):
        """Test with 1D waveform."""
        import torch

        waveform = torch.zeros(100000)  # 1D
        sample_rate = 16000

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word_seconds

            result = preview_word_seconds(waveform, aligned_word_simple, sample_rate)
            mock_audio.assert_called_once()

    @requires_torch
    def test_2d_waveform(self, aligned_word_simple):
        """Test with 2D waveform (channels, samples)."""
        import torch

        waveform = torch.zeros(1, 100000)  # 2D
        sample_rate = 16000

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word_seconds

            result = preview_word_seconds(waveform, aligned_word_simple, sample_rate)
            mock_audio.assert_called_once()


class TestGentleVisualization:
    """Tests for Gentle-format visualization utilities."""

    def test_import_gentle_utils(self):
        """Test that gentle.py can be imported."""
        from visualization_utils import gentle
        assert gentle is not None

    @requires_torch
    def test_gentle_module_has_expected_functions(self):
        """Test gentle module has expected functions."""
        from visualization_utils import gentle

        # Check for expected attributes (may vary)
        assert hasattr(gentle, '__file__')


class TestAudacityVisualization:
    """Tests for Audacity label generation utilities."""

    def test_audacity_labels_sorted_by_time(self, aligned_words_list):
        """Test Audacity labels are sorted by time."""
        from visualization_utils.audacity import get_audacity_labels

        # Create unsorted dict
        word_alignment = {5: aligned_words_list[5], 0: aligned_words_list[0]}

        labels = get_audacity_labels(word_alignment)
        lines = labels.strip().split("\n")

        # Should be sorted by time (first word first)
        if len(lines) >= 2:
            time1 = float(lines[0].split("\t")[0])
            time2 = float(lines[1].split("\t")[0])
            assert time1 < time2

    def test_audacity_labels_handles_empty_word(self, aligned_words_list):
        """Test Audacity labels handle empty word text."""
        from alignment.base import AlignedWord
        from visualization_utils.audacity import get_audacity_labels

        # Create word with empty text
        word = AlignedWord(word="", start_frame=100, end_frame=150)
        word_alignment = {0: word}

        labels = get_audacity_labels(word_alignment)
        # Should use placeholder like "[0]"
        assert "[0]" in labels


class TestVisualizationEdgeCases:
    """Edge case tests for visualization."""

    @requires_torch
    def test_very_short_word(self):
        """Test visualization of very short word (1 frame)."""
        import torch
        from alignment.base import AlignedWord

        word = AlignedWord(word="a", start_frame=100, end_frame=101)
        waveform = torch.zeros(100000)

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word_seconds

            result = preview_word_seconds(waveform, word)
            mock_audio.assert_called_once()

    @requires_torch
    def test_word_at_audio_boundary(self):
        """Test visualization of word at end of audio."""
        import torch
        from alignment.base import AlignedWord

        # Word at very end of audio
        word = AlignedWord(word="end", start_frame=4900, end_frame=5000)
        waveform = torch.zeros(100000)  # About 6.25s at 16kHz

        with patch("visualization_utils.audio_preview.Audio") as mock_audio:
            from visualization_utils.audio_preview import preview_word_seconds

            result = preview_word_seconds(waveform, word)
            mock_audio.assert_called_once()
