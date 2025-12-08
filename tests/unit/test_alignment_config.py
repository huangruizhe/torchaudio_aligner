"""
Tests for AlignmentConfig class.

Tests cover:
- Constructor and default values
- Custom values
- __post_init__ device selection
- All configuration attributes
"""

import pytest
from unittest.mock import patch

from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for alignment.base imports"
)


class TestAlignmentConfigDefaults:
    """Tests for default configuration values."""

    def test_default_language(self):
        """Test default language is English."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.language == "eng"

    def test_default_sample_rate(self):
        """Test default sample rate is 16000."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.sample_rate == 16000

    def test_default_segment_size(self):
        """Test default segment size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.segment_size == 15.0

    def test_default_overlap(self):
        """Test default overlap."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.overlap == 2.0

    def test_default_batch_size(self):
        """Test default batch size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.batch_size == 32

    def test_default_frame_duration(self):
        """Test default frame duration (20ms)."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.frame_duration == 0.02

    def test_default_skip_penalty(self):
        """Test default skip penalty."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.skip_penalty == -0.5

    def test_default_return_penalty(self):
        """Test default return penalty."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.return_penalty == -18.0

    def test_default_blank_penalty(self):
        """Test default blank penalty."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.blank_penalty == 0.0

    def test_default_neighborhood_size(self):
        """Test default neighborhood size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()
        assert config.neighborhood_size == 5


class TestAlignmentConfigDeviceSelection:
    """Tests for automatic device selection."""

    def test_device_auto_with_cuda_available(self):
        """Test auto device selection when CUDA is available."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=True):
            config = AlignmentConfig(device="auto")
        assert config.device == "cuda"

    def test_device_auto_without_cuda(self):
        """Test auto device selection when CUDA is not available."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(device="auto")
        assert config.device == "cpu"

    def test_device_explicit_cpu(self):
        """Test explicit CPU device."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=True):
            config = AlignmentConfig(device="cpu")
        assert config.device == "cpu"

    def test_device_explicit_cuda(self):
        """Test explicit CUDA device."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(device="cuda")
        assert config.device == "cuda"


class TestAlignmentConfigCustomValues:
    """Tests for custom configuration values."""

    def test_custom_language(self):
        """Test setting custom language."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(language="deu")
        assert config.language == "deu"

    def test_custom_language_chinese(self):
        """Test Chinese language code."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(language="cmn")
        assert config.language == "cmn"

    def test_custom_sample_rate(self):
        """Test custom sample rate."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(sample_rate=22050)
        assert config.sample_rate == 22050

    def test_custom_segment_size(self):
        """Test custom segment size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(segment_size=30.0)
        assert config.segment_size == 30.0

    def test_custom_overlap(self):
        """Test custom overlap."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(overlap=5.0)
        assert config.overlap == 5.0

    def test_custom_batch_size(self):
        """Test custom batch size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(batch_size=16)
        assert config.batch_size == 16

    def test_custom_frame_duration(self):
        """Test custom frame duration."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(frame_duration=0.01)
        assert config.frame_duration == 0.01

    def test_custom_penalties(self):
        """Test custom penalty values."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(
                skip_penalty=-1.0,
                return_penalty=-20.0,
                blank_penalty=-0.1,
            )
        assert config.skip_penalty == -1.0
        assert config.return_penalty == -20.0
        assert config.blank_penalty == -0.1

    def test_custom_neighborhood_size(self):
        """Test custom neighborhood size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(neighborhood_size=10)
        assert config.neighborhood_size == 10

    def test_multiple_custom_values(self):
        """Test setting multiple custom values."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(
                language="fra",
                sample_rate=8000,
                segment_size=20.0,
                overlap=3.0,
                batch_size=8,
                device="cpu",
            )
        assert config.language == "fra"
        assert config.sample_rate == 8000
        assert config.segment_size == 20.0
        assert config.overlap == 3.0
        assert config.batch_size == 8
        assert config.device == "cpu"


class TestAlignmentConfigEdgeCases:
    """Edge case tests."""

    def test_zero_overlap(self):
        """Test zero overlap (no overlap)."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(overlap=0.0)
        assert config.overlap == 0.0

    def test_small_segment_size(self):
        """Test small segment size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(segment_size=1.0)
        assert config.segment_size == 1.0

    def test_large_batch_size(self):
        """Test large batch size."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(batch_size=128)
        assert config.batch_size == 128

    def test_batch_size_one(self):
        """Test batch size of 1."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(batch_size=1)
        assert config.batch_size == 1

    def test_high_sample_rate(self):
        """Test high sample rate (48kHz)."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(sample_rate=48000)
        assert config.sample_rate == 48000

    def test_very_small_frame_duration(self):
        """Test very small frame duration."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(frame_duration=0.001)
        assert config.frame_duration == 0.001

    def test_zero_penalties(self):
        """Test zero penalties."""
        from alignment.base import AlignmentConfig
        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig(
                skip_penalty=0.0,
                return_penalty=0.0,
                blank_penalty=0.0,
            )
        assert config.skip_penalty == 0.0
        assert config.return_penalty == 0.0
        assert config.blank_penalty == 0.0
