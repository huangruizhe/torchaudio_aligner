"""
Tests for emission extraction utilities.

Tests cover:
- EmissionResult dataclass
- EmissionResult properties and methods
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for labeling_utils imports"
)


class TestEmissionResult:
    """Tests for EmissionResult dataclass."""

    def test_emission_result_creation(self):
        """Test creating an EmissionResult."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        # Create mock data
        emissions = torch.randn(100, 50)  # 100 frames, vocab size 50
        lengths = torch.tensor(100)
        vocab_info = VocabInfo(
            labels=list("abcdefg"),
            label_to_id={c: i for i, c in enumerate("abcdefg")},
            id_to_label={i: c for i, c in enumerate("abcdefg")},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
            frame_duration=0.02,
            sample_rate=16000,
        )

        assert result.emissions.shape == (100, 50)
        assert result.sample_rate == 16000

    def test_num_frames_property(self):
        """Test num_frames property."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        emissions = torch.randn(150, 50)
        lengths = torch.tensor(150)
        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
        )

        assert result.num_frames == 150

    def test_num_frames_batched(self):
        """Test num_frames with batched emissions."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        # Batched: (batch, frames, vocab)
        emissions = torch.randn(4, 100, 50)
        lengths = torch.tensor([100, 100, 100, 100])
        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
        )

        assert result.num_frames == 100

    def test_vocab_size_property(self):
        """Test vocab_size property."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        emissions = torch.randn(100, 75)  # vocab size 75
        lengths = torch.tensor(100)
        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
        )

        assert result.vocab_size == 75

    def test_duration_property(self):
        """Test duration property."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        emissions = torch.randn(100, 50)  # 100 frames
        lengths = torch.tensor(100)
        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
            frame_duration=0.02,  # 20ms
        )

        # 100 frames * 0.02s = 2.0s
        assert result.duration == 2.0

    def test_get_frame_timestamps(self):
        """Test get_frame_timestamps method."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        emissions = torch.randn(50, 10)  # 50 frames
        lengths = torch.tensor(50)
        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
            frame_duration=0.02,
        )

        timestamps = result.get_frame_timestamps()

        assert timestamps.shape == (50,)
        assert timestamps[0] == 0.0
        assert abs(timestamps[1] - 0.02) < 0.001
        assert abs(timestamps[-1] - 0.98) < 0.001  # (50-1) * 0.02

    def test_to_device(self):
        """Test to() method for moving tensors."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        emissions = torch.randn(50, 10)
        lengths = torch.tensor(50)
        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=emissions,
            lengths=lengths,
            vocab_info=vocab_info,
        )

        moved = result.to("cpu")

        assert moved.emissions.device.type == "cpu"
        assert moved.lengths.device.type == "cpu"

    def test_default_frame_duration(self):
        """Test default frame duration is 0.02s (20ms)."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=torch.randn(100, 10),
            lengths=torch.tensor(100),
            vocab_info=vocab_info,
        )

        assert result.frame_duration == 0.02

    def test_default_sample_rate(self):
        """Test default sample rate is 16000."""
        import torch
        from labeling_utils.emissions import EmissionResult
        from labeling_utils.base import VocabInfo

        vocab_info = VocabInfo(
            labels=["a"],
            label_to_id={"a": 0},
            id_to_label={0: "a"},
        )

        result = EmissionResult(
            emissions=torch.randn(100, 10),
            lengths=torch.tensor(100),
            vocab_info=vocab_info,
        )

        assert result.sample_rate == 16000
