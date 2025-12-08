"""
Tests for audio enhancement utilities.

Based on test_audio_frontend.ipynb Tests 11-18.

Tests cover:
- TimeMappingManager class
- EnhancementResult class
- AudioEnhancement class
- Silence removal
- Enhancement availability checks
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for audio_frontend imports"
)


class TestTimeMappingManager:
    """Tests for TimeMappingManager (timestamp recovery after silence removal)."""

    def test_empty_silence_intervals(self):
        """Test TimeMappingManager with no silence intervals."""
        from audio_frontend.enhancement import TimeMappingManager

        mapper = TimeMappingManager(silence_intervals=[])

        # Should pass through unchanged
        assert mapper.map_to_original(0.0) == 0.0
        assert mapper.map_to_original(5.0) == 5.0
        assert mapper.map_to_original(10.0) == 10.0

    def test_single_silence_interval(self):
        """Test TimeMappingManager with single silence interval."""
        from audio_frontend.enhancement import TimeMappingManager

        # Silence from 2-4 seconds
        mapper = TimeMappingManager(silence_intervals=[(2.0, 4.0)])

        # Before silence: unchanged
        assert abs(mapper.map_to_original(1.0) - 1.0) < 0.01

        # After silence removed: shifted by 2 seconds
        # processed_time=2.0 should map to original_time=4.0
        assert abs(mapper.map_to_original(2.0) - 4.0) < 0.01

    def test_multiple_silence_intervals(self):
        """Test TimeMappingManager with multiple silence intervals."""
        from audio_frontend.enhancement import TimeMappingManager

        # Silence from 0-1, 3-5, 6-8 seconds
        mapper = TimeMappingManager(silence_intervals=[(0, 1), (3, 5), (6, 8)])

        # Test various mappings
        test_cases = [
            (0, 1),      # After removing 0-1s silence, time 0 maps to 1
            (2, 5),      # After removing 3-5s silence, time 2 maps to 5
            (3, 8),      # After removing 6-8s silence, time 3 maps to 8
        ]

        for processed, expected_original in test_cases:
            actual = mapper.map_to_original(processed)
            assert abs(actual - expected_original) < 0.1, \
                f"map_to_original({processed}) = {actual}, expected {expected_original}"

    def test_map_to_new(self):
        """Test mapping from original to processed (new) time."""
        from audio_frontend.enhancement import TimeMappingManager

        # Silence from 2-4 seconds (2 seconds of silence)
        mapper = TimeMappingManager(silence_intervals=[(2.0, 4.0)])

        # Before silence: unchanged
        assert abs(mapper.map_to_new(1.0) - 1.0) < 0.01

        # After silence: shifted back by silence duration
        # original_time=5.0 should map to processed_time=3.0 (5-2=3)
        assert abs(mapper.map_to_new(5.0) - 3.0) < 0.01

    def test_overlapping_intervals_merged(self):
        """Test that overlapping silence intervals are merged."""
        from audio_frontend.enhancement import TimeMappingManager

        # Overlapping intervals: should be merged to (1, 5)
        mapper = TimeMappingManager(silence_intervals=[(1, 3), (2, 5)])

        # Check merged intervals
        assert len(mapper.silence_intervals) == 1
        assert mapper.silence_intervals[0] == (1, 5)

    def test_round_trip_mapping(self):
        """Test that map_to_original and map_to_new are inverses."""
        from audio_frontend.enhancement import TimeMappingManager

        mapper = TimeMappingManager(silence_intervals=[(2.0, 4.0), (6.0, 8.0)])

        # Test round trip for various times
        for original_time in [0.0, 1.0, 4.5, 5.0, 8.5, 10.0]:
            processed = mapper.map_to_new(original_time)
            recovered = mapper.map_to_original(processed)
            # Note: Exact round-trip may not work if original_time is in silence
            # But for times outside silence, should be close
            if original_time < 2.0 or (4.0 <= original_time < 6.0) or original_time >= 8.0:
                assert abs(recovered - original_time) < 0.1, \
                    f"Round trip failed for {original_time}: {processed} -> {recovered}"


class TestEnhancementResult:
    """Tests for EnhancementResult dataclass."""

    def test_enhancement_result_creation(self):
        """Test creating an EnhancementResult."""
        import torch
        from audio_frontend.enhancement import EnhancementResult, TimeMappingManager

        waveform = torch.randn(16000)
        mapper = TimeMappingManager([])

        result = EnhancementResult(
            waveform=waveform,
            sample_rate=16000,
            time_mapping_managers=[mapper],
            original_duration_seconds=2.0,
            enhanced_duration_seconds=1.5,
        )

        assert result.sample_rate == 16000
        assert result.original_duration_seconds == 2.0
        assert result.enhanced_duration_seconds == 1.5

    def test_enhancement_result_mapping(self):
        """Test EnhancementResult timestamp mapping methods."""
        import torch
        from audio_frontend.enhancement import EnhancementResult, TimeMappingManager

        # Single silence removal
        mapper = TimeMappingManager([(2.0, 4.0)])

        result = EnhancementResult(
            waveform=torch.randn(16000),
            sample_rate=16000,
            time_mapping_managers=[mapper],
            original_duration_seconds=10.0,
            enhanced_duration_seconds=8.0,
        )

        # Test mapping
        original = result.map_to_original(3.0)  # 3s processed -> 5s original
        assert abs(original - 5.0) < 0.1


class TestGetAvailableEnhancementBackends:
    """Tests for enhancement backend availability checking."""

    def test_returns_dict(self):
        """Test that get_available_enhancement_backends returns a dict."""
        from audio_frontend.enhancement import get_available_enhancement_backends

        backends = get_available_enhancement_backends()
        assert isinstance(backends, dict)

    def test_contains_expected_keys(self):
        """Test that result contains expected backend keys."""
        from audio_frontend.enhancement import get_available_enhancement_backends

        backends = get_available_enhancement_backends()
        expected_keys = [
            "demucs",
            "pyloudnorm",
            "silero_vad",
            "noisereduce",
            "deepfilternet",
            "resemble_enhance",
        ]

        for key in expected_keys:
            assert key in backends

    def test_values_are_boolean(self):
        """Test that backend availability values are boolean."""
        from audio_frontend.enhancement import get_available_enhancement_backends

        backends = get_available_enhancement_backends()
        for key, value in backends.items():
            assert isinstance(value, bool), f"{key} should be boolean"


class TestAudioEnhancementInit:
    """Tests for AudioEnhancement initialization."""

    def test_init_default_device(self):
        """Test AudioEnhancement initializes with default device."""
        from audio_frontend.enhancement import AudioEnhancement

        enhancer = AudioEnhancement()
        assert enhancer.device is not None

    def test_init_cpu_device(self):
        """Test AudioEnhancement initializes on CPU."""
        from audio_frontend.enhancement import AudioEnhancement
        import torch

        enhancer = AudioEnhancement(device="cpu")
        assert enhancer.device == torch.device("cpu")

    def test_init_creates_temp_dir(self, tmp_path):
        """Test AudioEnhancement creates temp directory."""
        from audio_frontend.enhancement import AudioEnhancement

        temp_dir = tmp_path / "enhancement_temp"
        enhancer = AudioEnhancement(temp_dir=temp_dir)

        assert enhancer.temp_dir.exists()


class TestAudioEnhancementSilenceRemoval:
    """Tests for AudioEnhancement.remove_silence method."""

    def test_remove_silence_from_silent_audio(self):
        """Test silence removal from all-silent audio."""
        import torch
        from audio_frontend.enhancement import AudioEnhancement

        enhancer = AudioEnhancement(device="cpu")

        # All zeros (silence)
        waveform = torch.zeros(16000 * 5)  # 5 seconds of silence
        result, mapper = enhancer.remove_silence(
            waveform,
            sample_rate=16000,
            silence_threshold_db=-50.0,
        )

        # Most should be removed (may keep some due to padding)
        assert result.shape[0] < waveform.shape[0]

    def test_remove_silence_from_speech_audio(self):
        """Test silence removal from audio with speech."""
        import torch
        from audio_frontend.enhancement import AudioEnhancement

        enhancer = AudioEnhancement(device="cpu")

        # Create audio: [noise][silence][noise]
        sr = 16000
        noise1 = torch.randn(sr * 2) * 0.5  # 2s noise
        silence = torch.zeros(sr * 3)        # 3s silence
        noise2 = torch.randn(sr * 2) * 0.5  # 2s noise
        waveform = torch.cat([noise1, silence, noise2])

        result, mapper = enhancer.remove_silence(
            waveform,
            sample_rate=sr,
            silence_threshold_db=-30.0,
            min_silence_duration=0.5,
        )

        # Result should be shorter than original
        original_duration = waveform.shape[0] / sr
        result_duration = result.shape[0] / sr
        assert result_duration < original_duration

    def test_remove_silence_returns_mapper(self):
        """Test that remove_silence returns a TimeMappingManager."""
        import torch
        from audio_frontend.enhancement import AudioEnhancement, TimeMappingManager

        enhancer = AudioEnhancement(device="cpu")
        waveform = torch.randn(16000 * 5) * 0.5

        _, mapper = enhancer.remove_silence(waveform, sample_rate=16000)

        assert isinstance(mapper, TimeMappingManager)


class TestAudioEnhancementNormalize:
    """Tests for AudioEnhancement.normalize_audio method."""

    def test_normalize_audio_basic(self):
        """Test basic audio normalization."""
        import torch
        from audio_frontend.enhancement import AudioEnhancement, get_available_enhancement_backends

        if not get_available_enhancement_backends().get("pyloudnorm", False):
            pytest.skip("pyloudnorm not available")

        enhancer = AudioEnhancement(device="cpu")
        waveform = torch.randn(16000 * 2) * 0.3

        normalized = enhancer.normalize_audio(waveform, sample_rate=16000)

        # Should return a tensor
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape[0] == waveform.shape[0]


class TestAudioEnhancementNoisereduce:
    """Tests for noisereduce denoising (if available)."""

    def test_noisereduce_basic(self):
        """Test basic noisereduce denoising."""
        import torch
        from audio_frontend.enhancement import (
            AudioEnhancement,
            get_available_enhancement_backends,
        )

        if not get_available_enhancement_backends().get("noisereduce", False):
            pytest.skip("noisereduce not available")

        enhancer = AudioEnhancement(device="cpu")
        waveform = torch.randn(16000 * 2)  # 2 seconds

        denoised = enhancer.apply_noisereduce(
            waveform,
            sample_rate=16000,
            stationary=True,
        )

        # Should return same shape
        assert denoised.shape == waveform.shape

    def test_noisereduce_reduces_noise(self):
        """Test that noisereduce reduces noise in noisy audio."""
        import torch
        from audio_frontend.enhancement import (
            AudioEnhancement,
            get_available_enhancement_backends,
        )

        if not get_available_enhancement_backends().get("noisereduce", False):
            pytest.skip("noisereduce not available")

        enhancer = AudioEnhancement(device="cpu")

        # Create noisy signal: clean tone + noise
        t = torch.linspace(0, 2, 32000)
        clean = torch.sin(2 * 3.14159 * 440 * t) * 0.5  # 440 Hz tone
        noise = torch.randn(32000) * 0.3
        noisy = clean + noise

        denoised = enhancer.apply_noisereduce(noisy, sample_rate=16000)

        # Denoised should have different characteristics than noisy
        # (Not a strict test, just checking it runs)
        assert denoised.shape == noisy.shape


class TestAudioEnhancementEnhancePipeline:
    """Tests for full enhance() pipeline."""

    def test_enhance_returns_result(self, tmp_path):
        """Test that enhance() returns EnhancementResult."""
        import torch
        import torchaudio
        from audio_frontend.enhancement import AudioEnhancement, EnhancementResult

        # Create test audio file
        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000 * 5)
        torchaudio.save(str(test_audio), waveform, 16000)

        enhancer = AudioEnhancement(device="cpu")
        result = enhancer.enhance(
            test_audio,
            denoise_method=None,
            remove_silence=True,
            apply_vad=False,  # Skip VAD for faster test
        )

        assert isinstance(result, EnhancementResult)
        assert result.sample_rate == 16000

    def test_enhance_with_tensor_input(self):
        """Test enhance() with tensor input."""
        import torch
        from audio_frontend.enhancement import AudioEnhancement, EnhancementResult

        waveform = torch.randn(16000 * 5)

        enhancer = AudioEnhancement(device="cpu")
        result = enhancer.enhance(
            waveform,
            sample_rate=16000,
            denoise_method=None,
            remove_silence=True,
            apply_vad=False,
        )

        assert isinstance(result, EnhancementResult)

    def test_enhance_tracks_duration(self):
        """Test that enhance() tracks duration changes."""
        import torch
        from audio_frontend.enhancement import AudioEnhancement

        # Create audio with silence
        sr = 16000
        noise = torch.randn(sr * 2) * 0.5
        silence = torch.zeros(sr * 3)
        waveform = torch.cat([noise, silence, noise])

        enhancer = AudioEnhancement(device="cpu")
        result = enhancer.enhance(
            waveform,
            sample_rate=sr,
            remove_silence=True,
            apply_vad=False,
        )

        # Should have original and enhanced durations
        assert result.original_duration_seconds > 0
        assert result.enhanced_duration_seconds > 0
        # Enhanced should be shorter due to silence removal
        assert result.enhanced_duration_seconds <= result.original_duration_seconds


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_enhance_audio_function(self, tmp_path):
        """Test enhance_audio convenience function."""
        import torch
        import torchaudio
        from audio_frontend.enhancement import enhance_audio, EnhancementResult

        # Create test audio file
        test_audio = tmp_path / "test.wav"
        waveform = torch.randn(1, 16000 * 3)
        torchaudio.save(str(test_audio), waveform, 16000)

        result = enhance_audio(
            test_audio,
            denoise_method=None,
            remove_silence=True,
            apply_vad=False,
            device="cpu",
        )

        assert isinstance(result, EnhancementResult)
