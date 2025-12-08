"""
Tests for stitching_utils base classes.

Tests cover:
- AlignedToken dataclass
- SegmentAlignment dataclass
- StitchingResult dataclass
- StitchingConfig dataclass
- StitcherBackend abstract class
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for stitching_utils imports"
)


class TestAlignedToken:
    """Tests for AlignedToken dataclass."""

    def test_aligned_token_creation(self):
        """Test creating an AlignedToken."""
        from stitching_utils.base import AlignedToken

        token = AlignedToken(
            token_id=5,
            timestamp=100,
            score=0.95,
            attr={"wid": 3},
        )

        assert token.token_id == 5
        assert token.timestamp == 100
        assert token.score == 0.95
        assert token.attr["wid"] == 3

    def test_has_word_index_true(self):
        """Test has_word_index when wid is present."""
        from stitching_utils.base import AlignedToken

        token = AlignedToken(token_id=1, timestamp=10, attr={"wid": 0})
        assert token.has_word_index is True

    def test_has_word_index_false(self):
        """Test has_word_index when wid is absent."""
        from stitching_utils.base import AlignedToken

        token = AlignedToken(token_id=1, timestamp=10)
        assert token.has_word_index is False

    def test_default_score(self):
        """Test default score is 1.0."""
        from stitching_utils.base import AlignedToken

        token = AlignedToken(token_id=1, timestamp=10)
        assert token.score == 1.0

    def test_default_attr(self):
        """Test default attr is empty dict."""
        from stitching_utils.base import AlignedToken

        token = AlignedToken(token_id=1, timestamp=10)
        assert token.attr == {}


class TestSegmentAlignment:
    """Tests for SegmentAlignment dataclass."""

    def test_segment_alignment_creation(self):
        """Test creating a SegmentAlignment."""
        from stitching_utils.base import AlignedToken, SegmentAlignment

        tokens = [
            AlignedToken(token_id=1, timestamp=10, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=20, attr={"wid": 1}),
        ]

        seg = SegmentAlignment(
            tokens=tokens,
            segment_index=0,
            frame_offset=100,
        )

        assert len(seg) == 2
        assert seg.segment_index == 0
        assert seg.frame_offset == 100

    def test_get_word_indices(self):
        """Test get_word_indices extracts wid from tokens."""
        from stitching_utils.base import AlignedToken, SegmentAlignment

        tokens = [
            AlignedToken(token_id=1, timestamp=10, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=20),  # No wid
            AlignedToken(token_id=3, timestamp=30, attr={"wid": 2}),
        ]

        seg = SegmentAlignment(tokens=tokens)
        indices = seg.get_word_indices()

        assert indices == [0, 2]

    def test_get_tokens_with_word_index(self):
        """Test get_tokens_with_word_index filters correctly."""
        from stitching_utils.base import AlignedToken, SegmentAlignment

        tokens = [
            AlignedToken(token_id=1, timestamp=10, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=20),  # No wid
            AlignedToken(token_id=3, timestamp=30, attr={"wid": 2}),
        ]

        seg = SegmentAlignment(tokens=tokens)
        word_tokens = seg.get_tokens_with_word_index()

        assert len(word_tokens) == 2
        assert word_tokens[0].attr["wid"] == 0
        assert word_tokens[1].attr["wid"] == 2

    def test_default_rejected_false(self):
        """Test default rejected is False."""
        from stitching_utils.base import SegmentAlignment

        seg = SegmentAlignment(tokens=[])
        assert seg.rejected is False

    def test_len_returns_token_count(self):
        """Test __len__ returns number of tokens."""
        from stitching_utils.base import AlignedToken, SegmentAlignment

        tokens = [AlignedToken(token_id=i, timestamp=i * 10) for i in range(5)]
        seg = SegmentAlignment(tokens=tokens)

        assert len(seg) == 5


class TestStitchingResult:
    """Tests for StitchingResult dataclass."""

    def test_stitching_result_creation(self):
        """Test creating a StitchingResult."""
        from stitching_utils.base import AlignedToken, StitchingResult

        tokens = [AlignedToken(token_id=i, timestamp=i * 10) for i in range(3)]

        result = StitchingResult(
            tokens=tokens,
            unaligned_regions=[(5, 8)],
            segment_mapping=[0, 0, 1],
        )

        assert result.num_aligned_tokens == 3
        assert result.num_unaligned_regions == 1

    def test_num_aligned_tokens_property(self):
        """Test num_aligned_tokens property."""
        from stitching_utils.base import AlignedToken, StitchingResult

        tokens = [AlignedToken(token_id=i, timestamp=i) for i in range(10)]
        result = StitchingResult(tokens=tokens)

        assert result.num_aligned_tokens == 10

    def test_num_unaligned_regions_property(self):
        """Test num_unaligned_regions property."""
        from stitching_utils.base import StitchingResult

        result = StitchingResult(
            tokens=[],
            unaligned_regions=[(0, 5), (10, 15), (20, 25)],
        )

        assert result.num_unaligned_regions == 3

    def test_get_word_indices(self):
        """Test get_word_indices from stitched result."""
        from stitching_utils.base import AlignedToken, StitchingResult

        tokens = [
            AlignedToken(token_id=1, timestamp=10, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=20, attr={"wid": 1}),
            AlignedToken(token_id=3, timestamp=30, attr={"wid": 3}),  # Gap
        ]

        result = StitchingResult(tokens=tokens)
        indices = result.get_word_indices()

        assert indices == [0, 1, 3]

    def test_default_empty_lists(self):
        """Test default empty lists for optional fields."""
        from stitching_utils.base import StitchingResult

        result = StitchingResult(tokens=[])

        assert result.unaligned_regions == []
        assert result.segment_mapping == []
        assert result.metadata == {}


class TestStitchingConfig:
    """Tests for StitchingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from stitching_utils.base import StitchingConfig

        config = StitchingConfig()

        assert config.method == "lis"
        assert config.neighborhood_size == 5
        assert config.outlier_scan_range == 3
        assert config.outlier_threshold == 50
        assert config.frame_duration == 0.02

    def test_custom_config(self):
        """Test custom configuration values."""
        from stitching_utils.base import StitchingConfig

        config = StitchingConfig(
            method="edit_distance",
            neighborhood_size=10,
            insertion_cost=2.0,
        )

        assert config.method == "edit_distance"
        assert config.neighborhood_size == 10
        assert config.insertion_cost == 2.0

    def test_edit_distance_costs(self):
        """Test edit distance cost defaults."""
        from stitching_utils.base import StitchingConfig

        config = StitchingConfig()

        assert config.insertion_cost == 1.0
        assert config.deletion_cost == 1.0
        assert config.substitution_cost == 1.0


class TestStitcherBackend:
    """Tests for StitcherBackend abstract class."""

    def test_backend_name_property(self):
        """Test backend name property."""
        from stitching_utils.base import StitcherBackend, StitchingConfig, SegmentAlignment, StitchingResult

        class TestStitcher(StitcherBackend):
            METHOD_NAME = "test_stitcher"

            def stitch(self, segment_alignments, **kwargs):
                return StitchingResult(tokens=[])

        stitcher = TestStitcher()
        assert stitcher.name == "test_stitcher"

    def test_default_config(self):
        """Test default config is created."""
        from stitching_utils.base import StitcherBackend, StitchingResult

        class TestStitcher(StitcherBackend):
            def stitch(self, segment_alignments, **kwargs):
                return StitchingResult(tokens=[])

        stitcher = TestStitcher()
        assert stitcher.config is not None
        assert stitcher.config.method == "lis"

    def test_custom_config(self):
        """Test custom config is used."""
        from stitching_utils.base import StitcherBackend, StitchingConfig, StitchingResult

        class TestStitcher(StitcherBackend):
            def stitch(self, segment_alignments, **kwargs):
                return StitchingResult(tokens=[])

        config = StitchingConfig(method="custom")
        stitcher = TestStitcher(config)
        assert stitcher.config.method == "custom"

    def test_flatten_tokens(self):
        """Test _flatten_tokens helper."""
        from stitching_utils.base import (
            StitcherBackend, AlignedToken, SegmentAlignment, StitchingResult
        )

        class TestStitcher(StitcherBackend):
            def stitch(self, segment_alignments, **kwargs):
                return StitchingResult(tokens=[])

        stitcher = TestStitcher()

        segments = [
            SegmentAlignment(
                tokens=[AlignedToken(token_id=i, timestamp=i) for i in range(3)],
                segment_index=0,
            ),
            SegmentAlignment(
                tokens=[AlignedToken(token_id=i+3, timestamp=i+3) for i in range(2)],
                segment_index=1,
            ),
        ]

        tokens, indices = stitcher._flatten_tokens(segments)

        assert len(tokens) == 5
        assert indices == [0, 0, 0, 1, 1]

    def test_flatten_tokens_skips_rejected(self):
        """Test _flatten_tokens skips rejected segments."""
        from stitching_utils.base import (
            StitcherBackend, AlignedToken, SegmentAlignment, StitchingResult
        )

        class TestStitcher(StitcherBackend):
            def stitch(self, segment_alignments, **kwargs):
                return StitchingResult(tokens=[])

        stitcher = TestStitcher()

        segments = [
            SegmentAlignment(
                tokens=[AlignedToken(token_id=0, timestamp=0)],
                segment_index=0,
                rejected=False,
            ),
            SegmentAlignment(
                tokens=[AlignedToken(token_id=1, timestamp=1)],
                segment_index=1,
                rejected=True,  # Should be skipped
            ),
        ]

        tokens, indices = stitcher._flatten_tokens(segments)

        assert len(tokens) == 1
        assert indices == [0]

    def test_validate_input_empty(self):
        """Test _validate_input raises for empty input."""
        from stitching_utils.base import StitcherBackend, StitchingResult

        class TestStitcher(StitcherBackend):
            def stitch(self, segment_alignments, **kwargs):
                return StitchingResult(tokens=[])

        stitcher = TestStitcher()

        with pytest.raises(ValueError, match="Empty"):
            stitcher._validate_input([])
