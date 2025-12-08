"""
Tests for Edit Distance stitching utilities.

Tests cover:
- EditDistanceStitcher class
- Edit distance computation
- Alignment transfer
"""

import pytest

# Import markers from conftest
from test_utils import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for stitching_utils imports"
)


class TestEditDistanceStitcher:
    """Tests for EditDistanceStitcher class."""

    def test_stitcher_creation(self):
        """Test creating EditDistanceStitcher."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()
        assert stitcher.name == "edit_distance"

    def test_stitcher_with_config(self):
        """Test EditDistanceStitcher with custom config."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import StitchingConfig

        config = StitchingConfig(
            insertion_cost=2.0,
            deletion_cost=2.0,
            substitution_cost=3.0,
        )
        stitcher = EditDistanceStitcher(config)

        assert stitcher.config.insertion_cost == 2.0

    def test_compute_alignment_identical(self):
        """Test _compute_alignment with identical strings."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()

        predicted = ["hello", "world"]
        reference = ["hello", "world"]

        result = stitcher._compute_alignment(predicted, reference)

        assert result["distance"] == 0
        # All operations should be "match"
        assert all(op == "match" for op, _, _ in result["alignment"])

    def test_compute_alignment_one_substitution(self):
        """Test _compute_alignment with one substitution."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()

        predicted = ["hello", "world"]
        reference = ["hello", "there"]

        result = stitcher._compute_alignment(predicted, reference)

        assert result["distance"] == 1.0  # One substitution
        # Should have one "sub" operation
        ops = [op for op, _, _ in result["alignment"]]
        assert "sub" in ops

    def test_compute_alignment_insertion(self):
        """Test _compute_alignment with insertion needed."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()

        predicted = ["hello"]
        reference = ["hello", "world"]

        result = stitcher._compute_alignment(predicted, reference)

        assert result["distance"] == 1.0  # One insertion
        ops = [op for op, _, _ in result["alignment"]]
        assert "ins" in ops

    def test_compute_alignment_deletion(self):
        """Test _compute_alignment with deletion needed."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()

        predicted = ["hello", "world"]
        reference = ["hello"]

        result = stitcher._compute_alignment(predicted, reference)

        assert result["distance"] == 1.0  # One deletion
        ops = [op for op, _, _ in result["alignment"]]
        assert "del" in ops

    def test_compute_alignment_case_insensitive(self):
        """Test that alignment is case insensitive."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()

        predicted = ["HELLO", "WORLD"]
        reference = ["hello", "world"]

        result = stitcher._compute_alignment(predicted, reference)

        # Should match despite case difference
        assert result["distance"] == 0

    def test_compute_alignment_custom_costs(self):
        """Test _compute_alignment with custom costs."""
        from stitching_utils.edit_distance import EditDistanceStitcher

        stitcher = EditDistanceStitcher()

        predicted = ["hello", "world"]
        reference = ["hello", "there"]

        # Higher substitution cost
        result = stitcher._compute_alignment(
            predicted, reference,
            ins_cost=1.0,
            del_cost=1.0,
            sub_cost=5.0,
        )

        assert result["distance"] == 5.0  # Substitution cost

    def test_extract_words_from_segments(self):
        """Test _extract_words_from_segments method."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = EditDistanceStitcher()

        segments = [
            SegmentAlignment(
                tokens=[
                    AlignedToken(token_id=1, timestamp=10, attr={"word": "hello"}),
                    AlignedToken(token_id=2, timestamp=20, attr={"word": "world"}),
                ],
                segment_index=0,
            ),
        ]

        words, tokens = stitcher._extract_words_from_segments(segments)

        assert words == ["hello", "world"]
        assert len(tokens) == 2

    def test_extract_words_skips_rejected(self):
        """Test _extract_words_from_segments skips rejected segments."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = EditDistanceStitcher()

        segments = [
            SegmentAlignment(
                tokens=[AlignedToken(token_id=1, timestamp=10, attr={"word": "hello"})],
                segment_index=0,
                rejected=False,
            ),
            SegmentAlignment(
                tokens=[AlignedToken(token_id=2, timestamp=20, attr={"word": "rejected"})],
                segment_index=1,
                rejected=True,
            ),
        ]

        words, tokens = stitcher._extract_words_from_segments(segments)

        assert words == ["hello"]

    def test_dedupe_by_position(self):
        """Test _dedupe_by_position method."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = EditDistanceStitcher()

        # Overlapping segments with same timestamp
        segments = [
            SegmentAlignment(
                tokens=[AlignedToken(token_id=1, timestamp=10)],
                segment_index=0,
            ),
            SegmentAlignment(
                tokens=[
                    AlignedToken(token_id=2, timestamp=10),  # Duplicate timestamp
                    AlignedToken(token_id=3, timestamp=20),
                ],
                segment_index=1,
            ),
        ]

        result = stitcher._dedupe_by_position(segments)

        # Should only have one token per timestamp
        timestamps = [t.timestamp for t in result.tokens]
        assert len(timestamps) == len(set(timestamps))

    def test_stitch_without_reference(self):
        """Test stitch without reference (dedupe mode)."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = EditDistanceStitcher()

        segments = [
            SegmentAlignment(
                tokens=[
                    AlignedToken(token_id=1, timestamp=10, attr={"word": "hello"}),
                ],
                segment_index=0,
            ),
        ]

        result = stitcher.stitch(segments, reference_words=None)

        assert result.metadata["mode"] == "dedupe_by_position"

    def test_stitch_with_reference(self):
        """Test stitch with reference transcript."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = EditDistanceStitcher()

        segments = [
            SegmentAlignment(
                tokens=[
                    AlignedToken(token_id=1, timestamp=10, attr={"word": "hello"}),
                    AlignedToken(token_id=2, timestamp=20, attr={"word": "world"}),
                ],
                segment_index=0,
            ),
        ]

        reference = ["hello", "world"]

        result = stitcher.stitch(segments, reference_words=reference)

        assert result.metadata["method"] == "edit_distance"
        assert result.metadata["predicted_words"] == 2
        assert result.metadata["reference_words"] == 2

    def test_stitch_empty_segments(self):
        """Test stitch with no extractable words."""
        from stitching_utils.edit_distance import EditDistanceStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = EditDistanceStitcher()

        segments = [
            SegmentAlignment(
                tokens=[AlignedToken(token_id=1, timestamp=10)],  # No word attr
                segment_index=0,
            ),
        ]

        reference = ["hello"]

        result = stitcher.stitch(segments, reference_words=reference)

        assert result.metadata.get("error") == "no_predicted_words"
