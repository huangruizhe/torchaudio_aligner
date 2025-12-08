"""
Integration tests for WFSTAligner.

These tests require k2 and lis dependencies to be installed.
Tests will be skipped if dependencies are not available.
"""

import pytest
import sys
from pathlib import Path

# Import markers from conftest
from conftest import requires_k2, requires_lis, requires_k2_and_lis, requires_torch


class TestWFSTAlignerImport:
    """Tests for importing WFSTAligner components."""

    @requires_k2_and_lis
    def test_import_wfst_aligner(self):
        """Test that WFSTAligner can be imported."""
        from alignment.wfst.aligner import WFSTAligner
        assert WFSTAligner is not None

    @requires_k2
    def test_import_factor_transducer(self):
        """Test that factor_transducer can be imported."""
        from alignment.wfst.factor_transducer import (
            make_factor_transducer_word_level_index_with_skip,
        )
        assert make_factor_transducer_word_level_index_with_skip is not None

    @requires_k2
    def test_import_k2_utils(self):
        """Test that k2_utils can be imported."""
        from alignment.wfst.k2_utils import (
            get_best_paths,
            get_texts_with_timestamp,
            concat_alignments,
            get_final_word_alignment,
        )
        assert get_best_paths is not None

    @requires_lis
    def test_import_lis_utils(self):
        """Test that lis_utils can be imported."""
        from alignment.wfst.lis_utils import (
            compute_lis,
            remove_outliers,
            remove_isolated_words,
            find_unaligned_regions,
        )
        assert compute_lis is not None


class TestLISUtilsFunctions:
    """Tests for LIS utility functions."""

    @requires_lis
    def test_compute_lis_simple(self):
        """Test compute_lis with simple sequence."""
        from alignment.wfst.lis_utils import compute_lis

        # Simple increasing sequence
        result = compute_lis([1, 2, 3, 4, 5])
        assert result == [1, 2, 3, 4, 5]

    @requires_lis
    def test_compute_lis_with_decreasing(self):
        """Test compute_lis with non-monotonic sequence."""
        from alignment.wfst.lis_utils import compute_lis

        # Sequence with some decreasing parts
        result = compute_lis([1, 3, 2, 4, 5])
        # LIS should be [1, 2, 4, 5] or [1, 3, 4, 5]
        assert len(result) == 4

    @requires_lis
    def test_compute_lis_empty(self):
        """Test compute_lis with empty list."""
        from alignment.wfst.lis_utils import compute_lis

        result = compute_lis([])
        assert result == []

    @requires_lis
    def test_compute_lis_single(self):
        """Test compute_lis with single element."""
        from alignment.wfst.lis_utils import compute_lis

        result = compute_lis([5])
        assert result == [5]

    def test_remove_outliers_simple(self):
        """Test remove_outliers with no outliers."""
        from alignment.wfst.lis_utils import remove_outliers

        # Sequential list with no outliers
        result = remove_outliers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def test_remove_outliers_with_gap(self):
        """Test remove_outliers with large gap at start."""
        from alignment.wfst.lis_utils import remove_outliers

        # List with outlier at start (jump from 1 to 100)
        test_list = [1] + list(range(100, 120))
        result = remove_outliers(test_list, outlier_threshold=60)
        assert 1 not in result

    def test_remove_outliers_short_list(self):
        """Test remove_outliers with short list (< 10 elements)."""
        from alignment.wfst.lis_utils import remove_outliers

        result = remove_outliers([1, 2, 3])
        assert result == [1, 2, 3]

    def test_remove_isolated_words_empty(self):
        """Test remove_isolated_words with empty list."""
        from alignment.wfst.lis_utils import remove_isolated_words

        result = remove_isolated_words([])
        assert result == []

    def test_remove_isolated_words_continuous(self):
        """Test remove_isolated_words with continuous sequence."""
        from alignment.wfst.lis_utils import remove_isolated_words

        # Continuous sequence - nothing should be removed
        result = remove_isolated_words([1, 2, 3, 4, 5])
        assert result == [1, 2, 3, 4, 5]

    def test_find_unaligned_regions_no_gaps(self):
        """Test find_unaligned_regions with no gaps."""
        from alignment.wfst.lis_utils import find_unaligned_regions

        aligned = {1, 2, 3, 4, 5}
        result = find_unaligned_regions(1, 5, aligned)
        assert result == []

    def test_find_unaligned_regions_with_gap(self):
        """Test find_unaligned_regions with a gap."""
        from alignment.wfst.lis_utils import find_unaligned_regions

        aligned = {1, 2, 5, 6}  # Missing 3, 4
        result = find_unaligned_regions(1, 6, aligned)
        assert (3, 4) in result

    def test_merge_segments_adjacent(self):
        """Test merge_segments with adjacent segments."""
        from alignment.wfst.lis_utils import merge_segments

        segments = [(1, 2), (4, 5)]
        result = merge_segments(segments, threshold=2)
        assert result == [(1, 5)]

    def test_merge_segments_far_apart(self):
        """Test merge_segments with far apart segments."""
        from alignment.wfst.lis_utils import merge_segments

        segments = [(1, 2), (10, 11)]
        result = merge_segments(segments, threshold=2)
        assert result == [(1, 2), (10, 11)]


class TestSegmentAlignmentResult:
    """Tests for SegmentAlignmentResult class."""

    @requires_k2_and_lis
    def test_segment_alignment_result_constructor(self):
        """Test SegmentAlignmentResult constructor."""
        from alignment.wfst.aligner import SegmentAlignmentResult
        from alignment.base import AlignedToken

        tokens = [
            AlignedToken(token_id=1, timestamp=10, score=0.9, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=15, score=0.8, attr={"wid": 1}),
        ]

        result = SegmentAlignmentResult(
            tokens=tokens,
            segment_index=0,
            frame_offset=100,
        )

        assert len(result) == 2
        assert result.segment_index == 0
        assert result.frame_offset == 100
        assert result.rejected == False

    @requires_k2_and_lis
    def test_segment_alignment_result_word_indices(self):
        """Test get_word_indices method."""
        from alignment.wfst.aligner import SegmentAlignmentResult
        from alignment.base import AlignedToken

        tokens = [
            AlignedToken(token_id=1, timestamp=10, score=0.9, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=15, score=0.8, attr={"wid": 1}),
            AlignedToken(token_id=3, timestamp=20, score=0.85),  # No wid
        ]

        result = SegmentAlignmentResult(tokens=tokens, segment_index=0)
        word_indices = result.get_word_indices()

        assert word_indices == [0, 1]

    @requires_k2_and_lis
    def test_segment_alignment_result_rejected(self):
        """Test rejected segment."""
        from alignment.wfst.aligner import SegmentAlignmentResult

        result = SegmentAlignmentResult(
            tokens=[],
            segment_index=0,
            rejected=True,
            score=-100.0,
        )

        assert result.rejected == True
        assert result.score == -100.0


class TestWFSTAlignerInit:
    """Tests for WFSTAligner initialization."""

    @requires_k2_and_lis
    def test_aligner_init(self):
        """Test WFSTAligner initialization."""
        from alignment.wfst.aligner import WFSTAligner
        from alignment.base import AlignmentConfig

        config = AlignmentConfig(
            language="eng",
            segment_size=15.0,
            device="cpu",
        )
        aligner = WFSTAligner(config)

        assert aligner.BACKEND_NAME == "wfst"
        assert aligner.config.segment_size == 15.0
        assert aligner._model is None
        assert aligner._loaded == False

    @requires_k2_and_lis
    def test_aligner_load_without_model(self):
        """Test that load() raises error without set_model()."""
        from alignment.wfst.aligner import WFSTAligner
        from alignment.base import AlignmentConfig

        config = AlignmentConfig(device="cpu")
        aligner = WFSTAligner(config)

        with pytest.raises(RuntimeError, match="Call set_model"):
            aligner.load()


class TestAlignedToken:
    """Tests for AlignedToken class."""

    @requires_torch
    def test_aligned_token_constructor(self):
        """Test AlignedToken constructor."""
        from alignment.base import AlignedToken

        token = AlignedToken(
            token_id=5,
            timestamp=100,
            score=-2.3,
            attr={"wid": 3, "tk": "h"},
        )

        assert token.token_id == 5
        assert token.timestamp == 100
        assert token.score == -2.3
        assert token.attr["wid"] == 3
        assert token.attr["tk"] == "h"

    @requires_torch
    def test_aligned_token_defaults(self):
        """Test AlignedToken default values."""
        from alignment.base import AlignedToken

        token = AlignedToken(token_id=1, timestamp=50)

        assert token.score == 0.0
        assert token.attr == {}

    @requires_torch
    def test_aligned_token_string_id(self):
        """Test AlignedToken with string token_id."""
        from alignment.base import AlignedToken

        token = AlignedToken(token_id="h", timestamp=100)

        assert token.token_id == "h"
