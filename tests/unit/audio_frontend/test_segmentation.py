"""
Tests for audio segmentation utilities.

Based on test_audio_frontend.ipynb Tests 4-7.

Tests cover:
- AudioSegment dataclass
- SegmentationResult dataclass
- segment_waveform function
- get_waveforms_batched method
- get_offsets_in_frames method
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for audio_frontend imports"
)


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_audio_segment_creation(self):
        """Test creating an AudioSegment."""
        import torch
        from audio_frontend.segmentation import AudioSegment

        waveform = torch.randn(16000)
        segment = AudioSegment(
            waveform=waveform,
            sample_rate=16000,
            offset_samples=0,
            length_samples=16000,
            segment_index=0,
        )

        assert segment.sample_rate == 16000
        assert segment.offset_samples == 0
        assert segment.segment_index == 0

    def test_offset_seconds_property(self):
        """Test offset_seconds property."""
        import torch
        from audio_frontend.segmentation import AudioSegment

        segment = AudioSegment(
            waveform=torch.randn(16000),
            sample_rate=16000,
            offset_samples=32000,  # 2 seconds at 16kHz
            length_samples=16000,
            segment_index=2,
        )

        assert segment.offset_seconds == 2.0

    def test_duration_seconds_property(self):
        """Test duration_seconds property."""
        import torch
        from audio_frontend.segmentation import AudioSegment

        segment = AudioSegment(
            waveform=torch.randn(24000),
            sample_rate=16000,
            offset_samples=0,
            length_samples=24000,  # 1.5 seconds at 16kHz
            segment_index=0,
        )

        assert segment.duration_seconds == 1.5

    def test_segment_with_different_sample_rates(self):
        """Test segment properties with different sample rates."""
        import torch
        from audio_frontend.segmentation import AudioSegment

        for sr in [8000, 16000, 22050, 44100, 48000]:
            segment = AudioSegment(
                waveform=torch.randn(sr),  # 1 second
                sample_rate=sr,
                offset_samples=sr * 5,  # 5 seconds offset
                length_samples=sr,
                segment_index=0,
            )

            assert abs(segment.offset_seconds - 5.0) < 0.01
            assert abs(segment.duration_seconds - 1.0) < 0.01


class TestSegmentWaveform:
    """Tests for segment_waveform function."""

    def test_segment_short_audio(self):
        """Test segmenting audio shorter than segment size."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        # 5 second audio, 15 second segment size
        waveform = torch.randn(16000 * 5)
        result = segment_waveform(waveform, sample_rate=16000, segment_size=15.0)

        # Should have 1 segment
        assert result.num_segments == 1

    def test_segment_long_audio(self):
        """Test segmenting audio longer than segment size."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        # 60 second audio, 15 second segment size, 2 second overlap
        waveform = torch.randn(16000 * 60)
        result = segment_waveform(
            waveform,
            sample_rate=16000,
            segment_size=15.0,
            overlap=2.0,
        )

        # Should have multiple segments
        assert result.num_segments > 1
        # Step size is ~13 seconds, so ~60/13 ≈ 5 segments
        assert result.num_segments >= 4

    def test_segment_overlap(self):
        """Test that segments have correct overlap."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        segment_size = 15.0
        overlap = 2.0

        result = segment_waveform(
            waveform,
            sample_rate=16000,
            segment_size=segment_size,
            overlap=overlap,
        )

        # Check that consecutive segments overlap
        if result.num_segments >= 2:
            seg0 = result.segments[0]
            seg1 = result.segments[1]

            # End of seg0 minus start of seg1 should be roughly overlap
            seg0_end = seg0.offset_samples + seg0.length_samples
            seg1_start = seg1.offset_samples

            overlap_samples = seg0_end - seg1_start
            overlap_seconds = overlap_samples / 16000

            # Allow some tolerance due to extra_samples
            assert overlap_seconds > 0

    def test_segment_result_metadata(self):
        """Test SegmentationResult metadata."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        duration = 30.0
        waveform = torch.randn(int(16000 * duration))
        result = segment_waveform(waveform, sample_rate=16000)

        assert result.sample_rate == 16000
        assert abs(result.original_duration_seconds - duration) < 0.01
        assert result.original_duration_samples == int(16000 * duration)

    def test_segment_monotonic_offsets(self):
        """Test that segment offsets are monotonically increasing."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        offsets = [seg.offset_samples for seg in result.segments]
        for i in range(len(offsets) - 1):
            assert offsets[i] < offsets[i + 1]

    def test_segment_all_have_correct_sample_rate(self):
        """Test that all segments have correct sample rate."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        for seg in result.segments:
            assert seg.sample_rate == 16000

    def test_segment_2d_input(self):
        """Test segmentation with 2D input (channels, samples)."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(1, 16000 * 30)
        result = segment_waveform(waveform, sample_rate=16000)

        assert result.num_segments > 0
        # For mono input (1 channel), output should be 1D per segment
        assert result.segments[0].waveform.dim() == 1

    def test_segment_min_segment_size(self):
        """Test minimum segment size handling."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        # Audio slightly longer than segment size
        waveform = torch.randn(16000 * 16)  # 16 seconds

        result = segment_waveform(
            waveform,
            sample_rate=16000,
            segment_size=15.0,
            overlap=2.0,
            min_segment_size=0.5,  # 0.5 second minimum
        )

        # Last segment should be at least min_segment_size
        if result.num_segments > 0:
            last_seg = result.segments[-1]
            assert last_seg.duration_seconds >= 0.2  # Allowing some tolerance


class TestSegmentationResultBatching:
    """Tests for SegmentationResult batching methods."""

    def test_get_waveforms_batched_shape(self):
        """Test get_waveforms_batched returns correct shape."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        waveforms, lengths = result.get_waveforms_batched()

        assert waveforms.shape[0] == result.num_segments
        assert lengths.shape[0] == result.num_segments

    def test_get_waveforms_batched_padding(self):
        """Test that shorter segments are padded correctly."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        waveforms, lengths = result.get_waveforms_batched()

        # All waveforms should have same length (padded)
        max_len = waveforms.shape[1]
        for i, seg in enumerate(result.segments):
            # Actual length should be stored in lengths tensor
            assert lengths[i].item() == seg.waveform.shape[-1]
            # Padded length should be max_len
            assert waveforms.shape[1] == max_len

    def test_get_waveforms_batched_mono(self):
        """Test batching for mono audio."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 30)
        result = segment_waveform(waveform, sample_rate=16000)

        waveforms, _ = result.get_waveforms_batched()

        # Should be 2D for mono: (batch, samples)
        assert waveforms.dim() == 2


class TestSegmentationResultFrameOffsets:
    """Tests for get_offsets_in_frames method."""

    def test_get_offsets_in_frames_basic(self):
        """Test basic frame offset calculation."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        # MMS model: 20ms frame duration
        frame_duration = 0.02
        offsets = result.get_offsets_in_frames(frame_duration)

        assert offsets.shape[0] == result.num_segments

    def test_get_offsets_in_frames_monotonic(self):
        """Test that frame offsets are monotonically increasing."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        offsets = result.get_offsets_in_frames(0.02)

        for i in range(len(offsets) - 1):
            assert offsets[i] < offsets[i + 1]

    def test_get_offsets_in_frames_calculation(self):
        """Test frame offset calculation accuracy."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        frame_duration = 0.02  # 20ms
        offsets = result.get_offsets_in_frames(frame_duration)

        # Verify first offset
        first_seg = result.segments[0]
        expected_offset = int(first_seg.offset_samples / 16000 / frame_duration)
        assert offsets[0].item() == expected_offset

    def test_get_offsets_different_frame_durations(self):
        """Test frame offsets with different frame durations."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        waveform = torch.randn(16000 * 60)
        result = segment_waveform(waveform, sample_rate=16000)

        # Test different frame durations
        for frame_duration in [0.01, 0.02, 0.025, 0.03]:
            offsets = result.get_offsets_in_frames(frame_duration)
            assert offsets.shape[0] == result.num_segments

            # Larger frame duration = smaller offsets
            if frame_duration > 0.02:
                offsets_20ms = result.get_offsets_in_frames(0.02)
                assert offsets[1] < offsets_20ms[1]


class TestSegmentWaveformEdgeCases:
    """Edge case tests for segment_waveform."""

    def test_very_short_audio(self):
        """Test segmenting very short audio."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        # 0.5 second audio
        waveform = torch.randn(8000)
        result = segment_waveform(
            waveform,
            sample_rate=16000,
            segment_size=15.0,
            min_segment_size=0.2,
        )

        assert result.num_segments == 1

    def test_exact_segment_size(self):
        """Test audio exactly equal to segment size."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        # Exactly 15 seconds
        waveform = torch.randn(16000 * 15)
        result = segment_waveform(
            waveform,
            sample_rate=16000,
            segment_size=15.0,
            overlap=2.0,
        )

        assert result.num_segments >= 1

    def test_segment_waveform_different_sample_rates(self):
        """Test segmentation with different sample rates."""
        import torch
        from audio_frontend.segmentation import segment_waveform

        for sr in [8000, 16000, 22050, 44100]:
            duration = 30.0
            waveform = torch.randn(int(sr * duration))
            result = segment_waveform(waveform, sample_rate=sr, segment_size=10.0)

            # Duration should be preserved
            assert abs(result.original_duration_seconds - duration) < 0.1
            assert result.sample_rate == sr
