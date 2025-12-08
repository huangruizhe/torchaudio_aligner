"""
Tests for LIS (Longest Increasing Subsequence) stitching utilities.

Based on test_stitching.ipynb.

Tests cover:
- compute_lis function
- remove_outliers function
- remove_isolated_words function
- find_unaligned_regions function
- merge_segments function
- LISStitcher class
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE, requires_lis

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for stitching_utils imports"
)


class TestComputeLIS:
    """Tests for compute_lis function."""

    @requires_lis
    def test_simple_increasing_sequence(self):
        """Test LIS on simple increasing sequence."""
        from stitching_utils.lis import compute_lis

        sequence = [1, 2, 3, 4, 5]
        result = compute_lis(sequence)

        assert result == [1, 2, 3, 4, 5]

    @requires_lis
    def test_decreasing_sequence(self):
        """Test LIS on decreasing sequence."""
        from stitching_utils.lis import compute_lis

        sequence = [5, 4, 3, 2, 1]
        result = compute_lis(sequence)

        # LIS of decreasing sequence should have length 1
        assert len(result) == 1

    @requires_lis
    def test_mixed_sequence(self):
        """Test LIS on mixed sequence."""
        from stitching_utils.lis import compute_lis

        sequence = [1, 5, 2, 6, 3, 7, 4, 8]
        result = compute_lis(sequence)

        # Should be monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    @requires_lis
    def test_empty_sequence(self):
        """Test LIS on empty sequence."""
        from stitching_utils.lis import compute_lis

        result = compute_lis([])
        assert result == []

    @requires_lis
    def test_single_element(self):
        """Test LIS on single element."""
        from stitching_utils.lis import compute_lis

        result = compute_lis([42])
        assert result == [42]

    @requires_lis
    def test_duplicates(self):
        """Test LIS with duplicate values."""
        from stitching_utils.lis import compute_lis

        # LIS should be strictly increasing, so duplicates shouldn't all be in result
        sequence = [1, 2, 2, 3, 3, 3, 4]
        result = compute_lis(sequence)

        # Result should be strictly increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]


class TestRemoveOutliers:
    """Tests for remove_outliers function."""

    def test_no_outliers(self):
        """Test remove_outliers with no outliers."""
        from stitching_utils.lis import remove_outliers

        sequence = list(range(1, 101))  # 1 to 100
        result = remove_outliers(sequence, scan_range=10, outlier_threshold=50)

        # Should be unchanged
        assert result == sequence

    def test_left_outlier(self):
        """Test remove_outliers with left outlier."""
        from stitching_utils.lis import remove_outliers

        # First element is outlier (gap > 60)
        sequence = [0] + list(range(100, 200))
        result = remove_outliers(sequence, scan_range=10, outlier_threshold=60)

        # First element should be removed
        assert result[0] >= 100

    def test_right_outlier(self):
        """Test remove_outliers with right outlier."""
        from stitching_utils.lis import remove_outliers

        # Last element is outlier (gap > 60)
        sequence = list(range(1, 50)) + [1000]
        result = remove_outliers(sequence, scan_range=10, outlier_threshold=60)

        # Last element should be removed
        assert result[-1] < 100

    def test_short_sequence_unchanged(self):
        """Test that short sequences are unchanged."""
        from stitching_utils.lis import remove_outliers

        sequence = [1, 2, 3, 4, 5]
        result = remove_outliers(sequence, scan_range=10, outlier_threshold=60)

        # Short sequences (<=10) should be unchanged
        assert result == sequence


class TestRemoveIsolatedWords:
    """Tests for remove_isolated_words function."""

    def test_no_isolated_words(self):
        """Test remove_isolated_words with dense sequence."""
        from stitching_utils.lis import remove_isolated_words

        sequence = list(range(0, 100))
        result = remove_isolated_words(sequence, neighborhood_size=5)

        # Dense sequence should be mostly unchanged
        assert len(result) >= len(sequence) * 0.9

    def test_isolated_word_in_middle(self):
        """Test removal of isolated word in middle."""
        from stitching_utils.lis import remove_isolated_words

        # Word 50 is isolated (neighbors 40-49 and 51-60 are missing)
        sequence = list(range(0, 40)) + [50] + list(range(61, 100))
        result = remove_isolated_words(sequence, neighborhood_size=5)

        # Word 50 might be removed as isolated
        # (Depends on neighbor_threshold)

    def test_empty_sequence(self):
        """Test remove_isolated_words with empty sequence."""
        from stitching_utils.lis import remove_isolated_words

        result = remove_isolated_words([])
        assert result == []


class TestFindUnalignedRegions:
    """Tests for find_unaligned_regions function."""

    def test_no_gaps(self):
        """Test find_unaligned_regions with no gaps."""
        from stitching_utils.lis import find_unaligned_regions

        aligned = set(range(0, 100))
        regions = find_unaligned_regions(0, 99, aligned)

        assert regions == []

    def test_single_gap(self):
        """Test find_unaligned_regions with single gap."""
        from stitching_utils.lis import find_unaligned_regions

        # Gap at indices 10-15
        aligned = set(range(0, 10)) | set(range(16, 30))
        regions = find_unaligned_regions(0, 29, aligned)

        # Should find gap including indices 10-15
        assert len(regions) >= 1
        # Check that gap region is in result
        gap_found = any(start <= 10 and end >= 15 for start, end in regions)
        assert gap_found

    def test_multiple_gaps(self):
        """Test find_unaligned_regions with multiple gaps."""
        from stitching_utils.lis import find_unaligned_regions

        # Gaps at 5-8 and 15-18
        aligned = {0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 19, 20}
        regions = find_unaligned_regions(0, 20, aligned, merge_threshold=0)

        # Should find at least 2 gaps
        assert len(regions) >= 2

    def test_gap_at_start(self):
        """Test gap at start of range."""
        from stitching_utils.lis import find_unaligned_regions

        # First few indices missing
        aligned = set(range(5, 20))
        regions = find_unaligned_regions(0, 19, aligned, merge_threshold=0)

        # Should find gap at start
        assert any(start == 0 for start, end in regions)


class TestMergeSegments:
    """Tests for merge_segments function."""

    def test_no_merge_needed(self):
        """Test merge_segments with non-overlapping segments."""
        from stitching_utils.lis import merge_segments

        segments = [(0, 5), (10, 15), (20, 25)]
        merged = merge_segments(segments, threshold=2)

        assert len(merged) == 3

    def test_merge_close_segments(self):
        """Test merge_segments with close segments."""
        from stitching_utils.lis import merge_segments

        # Gap of 2 between segments, threshold is 3
        segments = [(0, 5), (8, 15)]
        merged = merge_segments(segments, threshold=3)

        assert len(merged) == 1
        assert merged[0] == (0, 15)

    def test_merge_overlapping_segments(self):
        """Test merge_segments with overlapping segments."""
        from stitching_utils.lis import merge_segments

        segments = [(0, 10), (5, 15), (12, 20)]
        merged = merge_segments(segments, threshold=3)

        assert len(merged) == 1
        assert merged[0][0] == 0
        assert merged[0][1] == 20

    def test_empty_segments(self):
        """Test merge_segments with empty list."""
        from stitching_utils.lis import merge_segments

        merged = merge_segments([], threshold=3)
        assert merged == []

    def test_single_segment(self):
        """Test merge_segments with single segment."""
        from stitching_utils.lis import merge_segments

        segments = [(5, 10)]
        merged = merge_segments(segments, threshold=3)

        assert merged == [(5, 10)]


class TestLISStitcher:
    """Tests for LISStitcher class."""

    def test_stitcher_creation(self):
        """Test creating LISStitcher."""
        from stitching_utils.lis import LISStitcher

        stitcher = LISStitcher()
        assert stitcher.name == "lis"

    def test_stitcher_with_config(self):
        """Test LISStitcher with custom config."""
        from stitching_utils.lis import LISStitcher
        from stitching_utils.base import StitchingConfig

        config = StitchingConfig(neighborhood_size=10)
        stitcher = LISStitcher(config)

        assert stitcher.config.neighborhood_size == 10

    @requires_lis
    def test_stitch_empty_returns_empty(self):
        """Test stitching with no word indices returns empty."""
        from stitching_utils.lis import LISStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = LISStitcher()

        # Tokens without word indices
        segments = [
            SegmentAlignment(
                tokens=[AlignedToken(token_id=1, timestamp=10)],
                segment_index=0,
            ),
        ]

        result = stitcher.stitch(segments)
        # Should return empty or with error in metadata
        assert result.metadata.get("error") or len(result.tokens) == 0

    @requires_lis
    def test_stitch_simple_alignment(self):
        """Test stitching with simple alignment."""
        from stitching_utils.lis import LISStitcher
        from stitching_utils.base import AlignedToken, SegmentAlignment

        stitcher = LISStitcher()

        # Simple increasing word indices
        segments = [
            SegmentAlignment(
                tokens=[
                    AlignedToken(token_id=1, timestamp=10, attr={"wid": 0}),
                    AlignedToken(token_id=2, timestamp=20, attr={"wid": 1}),
                    AlignedToken(token_id=3, timestamp=30, attr={"wid": 2}),
                ],
                segment_index=0,
            ),
        ]

        result = stitcher.stitch(segments)
        assert result.metadata["method"] == "lis"

    def test_check_lis_available(self):
        """Test _check_lis_available method."""
        from stitching_utils.lis import LISStitcher

        stitcher = LISStitcher()
        # Just verify it returns boolean
        assert isinstance(stitcher._check_lis_available(), bool)
