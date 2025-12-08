"""
Tests for high-level API (api.py).

Tests cover:
- align_long_audio function signature
- Module exports (__all__)
- Backwards compatibility aliases
- Configuration options

Note: Full integration tests require model download and are marked separately.
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for API imports"
)


class TestModuleExports:
    """Tests for module-level exports."""

    def test_align_long_audio_exported(self):
        """Test that align_long_audio is exported."""
        from api import align_long_audio

        assert callable(align_long_audio)

    def test_alignment_result_exported(self):
        """Test that AlignmentResult is exported."""
        from api import AlignmentResult

        assert AlignmentResult is not None

    def test_aligned_word_exported(self):
        """Test that AlignedWord is exported."""
        from api import AlignedWord

        assert AlignedWord is not None

    def test_aligned_char_exported(self):
        """Test that AlignedChar is exported."""
        from api import AlignedChar

        assert AlignedChar is not None

    def test_alignment_config_exported(self):
        """Test that AlignmentConfig is exported."""
        from api import AlignmentConfig

        assert AlignmentConfig is not None

    def test_all_exports(self):
        """Test that __all__ contains expected items."""
        from api import __all__

        expected = [
            "align_long_audio",
            "AlignmentResult",
            "AlignedWord",
            "AlignedChar",
            "AlignmentConfig",
        ]

        for name in expected:
            assert name in __all__


class TestBackwardsCompatibility:
    """Tests for backwards compatibility."""

    def test_long_form_alignment_result_alias(self):
        """Test LongFormAlignmentResult is an alias for AlignmentResult."""
        from api import LongFormAlignmentResult, AlignmentResult

        assert LongFormAlignmentResult is AlignmentResult


class TestAlignLongAudioSignature:
    """Tests for align_long_audio function signature."""

    def test_accepts_audio_path(self):
        """Test function signature accepts audio as string path."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "audio" in sig.parameters

    def test_accepts_text_path(self):
        """Test function signature accepts text as string path."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "text" in sig.parameters

    def test_has_language_parameter(self):
        """Test function has language parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "language" in sig.parameters
        # Default should be "eng"
        assert sig.parameters["language"].default == "eng"

    def test_has_segment_size_parameter(self):
        """Test function has segment_size parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "segment_size" in sig.parameters
        assert sig.parameters["segment_size"].default == 15.0

    def test_has_overlap_parameter(self):
        """Test function has overlap parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "overlap" in sig.parameters
        assert sig.parameters["overlap"].default == 2.0

    def test_has_batch_size_parameter(self):
        """Test function has batch_size parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "batch_size" in sig.parameters
        assert sig.parameters["batch_size"].default == 32

    def test_has_device_parameter(self):
        """Test function has device parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "device" in sig.parameters

    def test_has_verbose_parameter(self):
        """Test function has verbose parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "verbose" in sig.parameters
        assert sig.parameters["verbose"].default is True

    def test_has_expand_numbers_parameter(self):
        """Test function has expand_numbers parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "expand_numbers" in sig.parameters
        assert sig.parameters["expand_numbers"].default is True

    def test_has_romanize_parameter(self):
        """Test function has romanize parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "romanize" in sig.parameters
        assert sig.parameters["romanize"].default is False

    def test_has_model_name_parameter(self):
        """Test function has model_name parameter."""
        from api import align_long_audio
        import inspect

        sig = inspect.signature(align_long_audio)
        assert "model_name" in sig.parameters
        assert sig.parameters["model_name"].default == "mms-fa"


class TestAlignmentResultIntegration:
    """Tests for AlignmentResult as returned by API."""

    def test_alignment_result_has_words(self):
        """Test AlignmentResult has words attribute."""
        from api import AlignmentResult, AlignedWord

        words = [AlignedWord(word="test", start_frame=0, end_frame=50)]
        result = AlignmentResult(words=words)

        assert len(result.words) == 1
        assert result.words[0].word == "test"

    def test_alignment_result_iteration(self):
        """Test AlignmentResult can be iterated."""
        from api import AlignmentResult, AlignedWord

        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, index=0),
            AlignedWord(word="world", start_frame=60, end_frame=100, index=1),
        ]
        result = AlignmentResult(words=words)

        word_texts = [w.word for w in result]
        assert word_texts == ["hello", "world"]

    def test_alignment_result_len(self):
        """Test len(AlignmentResult) returns word count."""
        from api import AlignmentResult, AlignedWord

        words = [AlignedWord(word=f"w{i}", start_frame=i*50, end_frame=(i+1)*50) for i in range(5)]
        result = AlignmentResult(words=words)

        assert len(result) == 5

    def test_alignment_result_text_property(self):
        """Test AlignmentResult.text returns joined words."""
        from api import AlignmentResult, AlignedWord

        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50),
            AlignedWord(word="world", start_frame=60, end_frame=100),
        ]
        result = AlignmentResult(words=words)

        assert result.text == "hello world"

    def test_alignment_result_has_export_methods(self):
        """Test AlignmentResult has export methods."""
        from api import AlignmentResult, AlignedWord

        words = [AlignedWord(word="test", start_frame=0, end_frame=50)]
        result = AlignmentResult(words=words)

        # Should have export methods
        assert hasattr(result, "to_audacity_labels")
        assert hasattr(result, "to_srt")
        assert hasattr(result, "to_json")
        assert hasattr(result, "to_ctm")

    def test_aligned_word_has_start_seconds_method(self):
        """Test AlignedWord has start_seconds() method."""
        from api import AlignedWord

        word = AlignedWord(word="test", start_frame=100, end_frame=150)

        # start_seconds() should be callable and return time in seconds
        start = word.start_seconds()
        assert isinstance(start, float)
        assert start == 2.0  # 100 * 0.02

    def test_aligned_word_has_end_seconds_method(self):
        """Test AlignedWord has end_seconds() method."""
        from api import AlignedWord

        word = AlignedWord(word="test", start_frame=100, end_frame=150)

        end = word.end_seconds()
        assert isinstance(end, float)
        assert end == 3.0  # 150 * 0.02


class TestAlignmentConfigIntegration:
    """Tests for AlignmentConfig integration with API."""

    def test_alignment_config_defaults(self):
        """Test AlignmentConfig default values."""
        from api import AlignmentConfig
        from unittest.mock import patch

        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()

        assert config.frame_duration == 0.02
        assert config.segment_size == 15.0
        assert config.overlap == 2.0

    def test_alignment_config_custom_values(self):
        """Test AlignmentConfig with custom values."""
        from api import AlignmentConfig
        from unittest.mock import patch

        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(
                segment_size=20.0,
                overlap=3.0,
            )

        assert config.segment_size == 20.0
        assert config.overlap == 3.0
