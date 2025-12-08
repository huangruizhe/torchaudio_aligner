"""
Tests for audio loading utilities.

Based on test_audio_frontend.ipynb Tests 1-3.

Tests cover:
- load_audio function with different backends
- get_available_backends
- FileNotFoundError handling
- Waveform shape and sample rate validation
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for audio_frontend imports"
)


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_torchaudio_always_available(self):
        """Test that torchaudio is always in available backends."""
        from audio_frontend.loaders import get_available_backends

        backends = get_available_backends()
        assert "torchaudio" in backends

    def test_returns_list(self):
        """Test that get_available_backends returns a list."""
        from audio_frontend.loaders import get_available_backends

        backends = get_available_backends()
        assert isinstance(backends, list)


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent file."""
        from audio_frontend.loaders import load_audio

        fake_path = tmp_path / "nonexistent.wav"
        with pytest.raises(FileNotFoundError):
            load_audio(fake_path)

    def test_load_returns_tuple(self, tmp_path):
        """Test that load_audio returns (waveform, sample_rate) tuple."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        # Create a test audio file
        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)  # 1 second at 16kHz
        torchaudio.save(str(test_audio), waveform, 16000)

        result = load_audio(test_audio)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_load_waveform_shape(self, tmp_path):
        """Test that loaded waveform has shape (channels, samples)."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        # Create a test audio file
        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        loaded_waveform, sample_rate = load_audio(test_audio)
        assert loaded_waveform.dim() == 2
        assert loaded_waveform.shape[0] >= 1  # At least 1 channel

    def test_load_sample_rate_positive(self, tmp_path):
        """Test that sample rate is positive."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        _, sample_rate = load_audio(test_audio)
        assert sample_rate > 0

    def test_load_preserves_sample_rate(self, tmp_path):
        """Test that load preserves original sample rate."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "test.wav"
        original_sr = 22050
        waveform = torch.randn(1, original_sr)
        torchaudio.save(str(test_audio), waveform, original_sr)

        _, sample_rate = load_audio(test_audio)
        assert sample_rate == original_sr

    def test_load_with_path_object(self, tmp_path):
        """Test load_audio accepts Path objects."""
        import torch
        import torchaudio
        from pathlib import Path

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        # Pass as Path object
        loaded_waveform, _ = load_audio(Path(test_audio))
        assert loaded_waveform is not None

    def test_load_with_string_path(self, tmp_path):
        """Test load_audio accepts string paths."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        # Pass as string
        loaded_waveform, _ = load_audio(str(test_audio))
        assert loaded_waveform is not None

    def test_load_stereo_audio(self, tmp_path):
        """Test loading stereo audio."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "stereo.wav"
        waveform = torch.randn(2, 16000)  # Stereo
        torchaudio.save(str(test_audio), waveform, 16000)

        loaded_waveform, _ = load_audio(test_audio)
        assert loaded_waveform.shape[0] == 2

    def test_load_backend_torchaudio(self, tmp_path):
        """Test explicit torchaudio backend."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        loaded_waveform, _ = load_audio(test_audio, backend="torchaudio")
        assert loaded_waveform is not None

    def test_load_backend_auto(self, tmp_path):
        """Test auto backend selection."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio

        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        loaded_waveform, _ = load_audio(test_audio, backend="auto")
        assert loaded_waveform is not None


class TestLoadWithSoundfile:
    """Tests for soundfile backend (if available)."""

    def test_soundfile_backend_if_available(self, tmp_path):
        """Test soundfile backend if installed."""
        import torch
        import torchaudio

        from audio_frontend.loaders import load_audio, get_available_backends

        if "soundfile" not in get_available_backends():
            pytest.skip("soundfile not available")

        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000)
        torchaudio.save(str(test_audio), waveform, 16000)

        loaded_waveform, _ = load_audio(test_audio, backend="soundfile")
        assert loaded_waveform is not None
        assert loaded_waveform.dim() == 2
