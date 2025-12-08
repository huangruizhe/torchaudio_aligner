"""
Tests for audio preprocessing utilities.

Based on test_audio_frontend.ipynb Tests 2, 3, 9.

Tests cover:
- resample function
- to_mono function
- normalize_peak function
- preprocess function (full pipeline)
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for audio_frontend imports"
)


class TestResample:
    """Tests for resample function."""

    def test_resample_downsamples(self):
        """Test resampling from higher to lower sample rate."""
        import torch
        from audio_frontend.preprocessing import resample

        # 1 second at 44100 Hz
        waveform = torch.randn(1, 44100)
        resampled = resample(waveform, 44100, 16000)

        # Expected samples: 44100 * (16000/44100) = 16000
        expected_samples = 16000
        assert abs(resampled.shape[1] - expected_samples) < 10

    def test_resample_upsamples(self):
        """Test resampling from lower to higher sample rate."""
        import torch
        from audio_frontend.preprocessing import resample

        # 1 second at 16000 Hz
        waveform = torch.randn(1, 16000)
        resampled = resample(waveform, 16000, 44100)

        # Expected samples: 16000 * (44100/16000) = 44100
        expected_samples = 44100
        assert abs(resampled.shape[1] - expected_samples) < 10

    def test_resample_no_change(self):
        """Test resampling with same rate returns same waveform."""
        import torch
        from audio_frontend.preprocessing import resample

        waveform = torch.randn(1, 16000)
        resampled = resample(waveform, 16000, 16000)

        assert resampled.shape == waveform.shape
        assert torch.allclose(resampled, waveform)

    def test_resample_preserves_channels(self):
        """Test that resampling preserves number of channels."""
        import torch
        from audio_frontend.preprocessing import resample

        # Stereo audio
        waveform = torch.randn(2, 44100)
        resampled = resample(waveform, 44100, 16000)

        assert resampled.shape[0] == 2

    def test_resample_duration_preserved(self):
        """Test that resampled audio has approximately same duration."""
        import torch
        from audio_frontend.preprocessing import resample

        orig_sr = 44100
        target_sr = 16000
        duration = 2.0  # seconds
        waveform = torch.randn(1, int(orig_sr * duration))

        resampled = resample(waveform, orig_sr, target_sr)
        resampled_duration = resampled.shape[1] / target_sr

        assert abs(resampled_duration - duration) < 0.01


class TestToMono:
    """Tests for to_mono function."""

    def test_mono_from_stereo(self):
        """Test converting stereo to mono."""
        import torch
        from audio_frontend.preprocessing import to_mono

        stereo = torch.randn(2, 16000)
        mono = to_mono(stereo)

        assert mono.shape[0] == 1
        assert mono.shape[1] == 16000

    def test_mono_from_mono(self):
        """Test that mono input stays mono."""
        import torch
        from audio_frontend.preprocessing import to_mono

        mono_input = torch.randn(1, 16000)
        mono = to_mono(mono_input)

        assert mono.shape[0] == 1
        assert torch.allclose(mono, mono_input)

    def test_mono_averages_channels(self):
        """Test that mono conversion averages channels."""
        import torch
        from audio_frontend.preprocessing import to_mono

        # Create stereo with known values
        stereo = torch.zeros(2, 100)
        stereo[0, :] = 1.0  # Left channel
        stereo[1, :] = 3.0  # Right channel

        mono = to_mono(stereo)

        # Average should be (1 + 3) / 2 = 2
        assert torch.allclose(mono, torch.full((1, 100), 2.0))

    def test_mono_preserves_length(self):
        """Test that mono conversion preserves sample length."""
        import torch
        from audio_frontend.preprocessing import to_mono

        stereo = torch.randn(2, 12345)
        mono = to_mono(stereo)

        assert mono.shape[1] == 12345

    def test_mono_from_multichannel(self):
        """Test converting multi-channel (>2) to mono."""
        import torch
        from audio_frontend.preprocessing import to_mono

        multichannel = torch.randn(6, 16000)  # 5.1 surround
        mono = to_mono(multichannel)

        assert mono.shape[0] == 1


class TestNormalizePeak:
    """Tests for normalize_peak function."""

    def test_normalize_to_target_db(self):
        """Test normalization to target dB."""
        import torch
        from audio_frontend.preprocessing import normalize_peak

        waveform = torch.randn(1, 16000) * 0.5  # Random with max ~0.5
        normalized = normalize_peak(waveform, target_db=-3.0)

        expected_peak = 10 ** (-3.0 / 20)  # ~0.708
        actual_peak = normalized.abs().max().item()

        assert abs(actual_peak - expected_peak) < 0.01

    def test_normalize_different_targets(self):
        """Test normalization with different target dB values."""
        import torch
        from audio_frontend.preprocessing import normalize_peak

        waveform = torch.randn(1, 16000) * 0.3

        for target_db in [-1.0, -3.0, -6.0, -12.0]:
            normalized = normalize_peak(waveform.clone(), target_db=target_db)
            expected_peak = 10 ** (target_db / 20)
            actual_peak = normalized.abs().max().item()
            assert abs(actual_peak - expected_peak) < 0.01

    def test_normalize_zero_audio(self):
        """Test normalization of silence (all zeros)."""
        import torch
        from audio_frontend.preprocessing import normalize_peak

        waveform = torch.zeros(1, 16000)
        normalized = normalize_peak(waveform, target_db=-3.0)

        # Should remain zeros (no division by zero)
        assert torch.allclose(normalized, torch.zeros_like(normalized))

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves waveform shape."""
        import torch
        from audio_frontend.preprocessing import normalize_peak

        waveform = torch.randn(2, 32000)
        normalized = normalize_peak(waveform, target_db=-3.0)

        assert normalized.shape == waveform.shape


class TestPreprocess:
    """Tests for preprocess function (full pipeline)."""

    def test_preprocess_resamples(self):
        """Test that preprocess resamples to target rate."""
        import torch
        from audio_frontend.preprocessing import preprocess

        waveform = torch.randn(1, 44100)  # 1 second at 44100 Hz
        processed = preprocess(waveform, sample_rate=44100, target_sample_rate=16000)

        # Should be approximately 16000 samples
        assert abs(processed.shape[-1] - 16000) < 100

    def test_preprocess_converts_to_mono(self):
        """Test that preprocess converts to mono by default."""
        import torch
        from audio_frontend.preprocessing import preprocess

        stereo = torch.randn(2, 16000)
        processed = preprocess(stereo, sample_rate=16000, mono=True)

        assert processed.shape[0] == 1

    def test_preprocess_keeps_stereo(self):
        """Test that preprocess keeps stereo when mono=False."""
        import torch
        from audio_frontend.preprocessing import preprocess

        stereo = torch.randn(2, 16000)
        processed = preprocess(stereo, sample_rate=16000, mono=False)

        assert processed.shape[0] == 2

    def test_preprocess_normalizes(self):
        """Test that preprocess normalizes when requested."""
        import torch
        from audio_frontend.preprocessing import preprocess

        waveform = torch.randn(1, 16000) * 0.3
        processed = preprocess(
            waveform,
            sample_rate=16000,
            normalize=True,
            normalize_db=-3.0,
        )

        expected_peak = 10 ** (-3.0 / 20)
        actual_peak = processed.abs().max().item()

        assert abs(actual_peak - expected_peak) < 0.01

    def test_preprocess_no_normalize(self):
        """Test that preprocess doesn't normalize by default."""
        import torch
        from audio_frontend.preprocessing import preprocess

        waveform = torch.randn(1, 16000) * 0.3
        original_peak = waveform.abs().max().item()

        processed = preprocess(waveform, sample_rate=16000, normalize=False)
        processed_peak = processed.abs().max().item()

        # Peaks should be similar (not normalized)
        assert abs(original_peak - processed_peak) < 0.01

    def test_preprocess_full_pipeline(self):
        """Test full preprocessing pipeline."""
        import torch
        from audio_frontend.preprocessing import preprocess

        # Stereo at 44100 Hz
        waveform = torch.randn(2, 44100 * 2)  # 2 seconds

        processed = preprocess(
            waveform,
            sample_rate=44100,
            target_sample_rate=16000,
            mono=True,
            normalize=True,
            normalize_db=-3.0,
        )

        # Should be mono
        assert processed.shape[0] == 1

        # Should be ~2 seconds at 16000 Hz
        assert abs(processed.shape[1] - 32000) < 100

        # Should be normalized
        expected_peak = 10 ** (-3.0 / 20)
        assert abs(processed.abs().max().item() - expected_peak) < 0.01

    def test_preprocess_custom_preprocessor(self):
        """Test preprocess with custom preprocessor function."""
        import torch
        from audio_frontend.preprocessing import preprocess

        # Custom preprocessor that doubles the amplitude
        def double_amplitude(waveform, sample_rate):
            return waveform * 2

        waveform = torch.ones(1, 16000) * 0.1
        processed = preprocess(
            waveform,
            sample_rate=16000,
            custom_preprocessors=[double_amplitude],
        )

        # Should be doubled
        assert torch.allclose(processed, waveform * 2)
