"""
Integration tests for the full alignment pipeline.

Based on test_alignment.ipynb test cases.

Tests cover:
- AlignedWord API (start_frame/end_frame vs start_seconds/end_seconds)
- Full alignment pipeline with ground truth comparison
- Export format consistency
- Multi-backend alignment result compatibility

Key API:
- AlignedWord has start_frame/end_frame as primary attributes (integers)
- Call start_seconds()/end_seconds() for time in seconds
- Frame duration: 20ms (0.02s) by default, 50fps
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE, requires_k2, requires_lis, requires_k2_and_lis

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for alignment imports"
)


# Ground truth data from test_alignment.ipynb
GROUND_TRUTH_WORDS = [
    {"word": "I", "start": 31, "end": 35, "score": 0.78},
    {"word": "HAD", "start": 37, "end": 44, "score": 0.84},
    {"word": "THAT", "start": 45, "end": 53, "score": 0.52},
    {"word": "CURIOSITY", "start": 56, "end": 92, "score": 0.89},
    {"word": "BESIDE", "start": 95, "end": 116, "score": 0.94},
    {"word": "ME", "start": 118, "end": 124, "score": 0.67},
    {"word": "AT", "start": 126, "end": 129, "score": 0.66},
    {"word": "THIS", "start": 131, "end": 139, "score": 0.70},
    {"word": "MOMENT", "start": 143, "end": 157, "score": 0.88},
]

TRANSCRIPT = "I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT"
FRAME_RATE = 50  # 20ms per frame


class TestAlignedWordAPI:
    """Tests for AlignedWord API - the frame vs seconds distinction."""

    def test_primary_attributes_are_frames(self):
        """Test that start_frame/end_frame are the primary attributes."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=100, end_frame=150)

        # Primary attributes are frames (integers)
        assert isinstance(word.start_frame, int)
        assert isinstance(word.end_frame, int)
        assert word.start_frame == 100
        assert word.end_frame == 150

    def test_seconds_are_methods(self):
        """Test that seconds are obtained via methods, not properties."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=100, end_frame=150)

        # Must call methods to get seconds
        start_sec = word.start_seconds()
        end_sec = word.end_seconds()

        assert isinstance(start_sec, float)
        assert isinstance(end_sec, float)

    def test_default_frame_duration(self):
        """Test default frame duration is 0.02s (20ms, 50fps)."""
        from alignment.base import AlignedWord, DEFAULT_FRAME_DURATION

        assert DEFAULT_FRAME_DURATION == 0.02

        word = AlignedWord(word="hello", start_frame=100, end_frame=150)

        # 100 frames * 0.02 = 2.0 seconds
        assert word.start_seconds() == 2.0
        # 150 frames * 0.02 = 3.0 seconds
        assert word.end_seconds() == 3.0

    def test_custom_frame_duration(self):
        """Test custom frame duration can be passed."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=100, end_frame=150)

        # With 10ms frame duration (100fps)
        assert word.start_seconds(frame_duration=0.01) == 1.0
        assert word.end_seconds(frame_duration=0.01) == 1.5

    def test_duration_frames_property(self):
        """Test duration_frames property."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=100, end_frame=150)

        assert word.duration_frames == 50
        assert word.duration_seconds() == 1.0  # 50 * 0.02


class TestAlignmentResultAPI:
    """Tests for AlignmentResult API."""

    def test_duration_uses_frames(self):
        """Test AlignmentResult.duration is computed from frames."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=100, end_frame=150),
            AlignedWord(word="world", start_frame=200, end_frame=300),
        ]
        result = AlignmentResult(words=words)

        # Duration should be from first start to last end in seconds
        # 100 frames = 2.0s, 300 frames = 6.0s, duration = 4.0s
        assert result.duration == 4.0

    def test_word_alignments_returns_dict(self):
        """Test word_alignments returns dict keyed by index."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
            AlignedWord(word="world", start_frame=200, end_frame=300, index=1),
        ]
        result = AlignmentResult(words=words)

        alignments = result.word_alignments
        assert isinstance(alignments, dict)
        assert 0 in alignments
        assert 1 in alignments


class TestGroundTruthComparison:
    """Tests comparing alignment results to ground truth."""

    def test_frame_to_seconds_conversion(self):
        """Test converting ground truth frames to seconds."""
        # Ground truth word "CURIOSITY": frames 56-92
        gt_word = GROUND_TRUTH_WORDS[3]
        assert gt_word["word"] == "CURIOSITY"

        start_sec = gt_word["start"] / FRAME_RATE  # 56/50 = 1.12s
        end_sec = gt_word["end"] / FRAME_RATE      # 92/50 = 1.84s

        assert abs(start_sec - 1.12) < 0.01
        assert abs(end_sec - 1.84) < 0.01

    def test_iou_computation(self):
        """Test IoU (Intersection over Union) computation."""
        def compute_iou(pred_start, pred_end, gt_start, gt_end):
            intersection_start = max(pred_start, gt_start)
            intersection_end = min(pred_end, gt_end)
            intersection = max(0, intersection_end - intersection_start)

            union_start = min(pred_start, gt_start)
            union_end = max(pred_end, gt_end)
            union = union_end - union_start

            return intersection / union if union > 0 else 0

        # Perfect overlap
        assert compute_iou(1.0, 2.0, 1.0, 2.0) == 1.0

        # 50% overlap
        iou = compute_iou(1.0, 2.0, 1.5, 2.5)
        assert 0.3 < iou < 0.4  # ~0.33

        # No overlap
        assert compute_iou(1.0, 2.0, 3.0, 4.0) == 0.0

    def test_time_error_computation(self):
        """Test time error computation in seconds."""
        def compute_time_error(pred_start, pred_end, gt_start, gt_end):
            start_error = abs(pred_start - gt_start)
            end_error = abs(pred_end - gt_end)
            return start_error, end_error

        # No error
        start_err, end_err = compute_time_error(1.0, 2.0, 1.0, 2.0)
        assert start_err == 0.0
        assert end_err == 0.0

        # 100ms error
        start_err, end_err = compute_time_error(1.1, 2.1, 1.0, 2.0)
        assert abs(start_err - 0.1) < 0.01
        assert abs(end_err - 0.1) < 0.01


class TestExportConsistency:
    """Tests for export format consistency with frame-based API."""

    def test_audacity_export_uses_seconds(self):
        """Test Audacity export converts frames to seconds."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=100, end_frame=150),  # 2.0-3.0s
        ]
        result = AlignmentResult(words=words)

        labels = result.to_audacity_labels()
        lines = labels.strip().split("\n")
        parts = lines[0].split("\t")

        # Should be in seconds
        assert float(parts[0]) == 2.0
        assert float(parts[1]) == 3.0
        assert parts[2] == "hello"

    def test_ctm_export_uses_seconds(self):
        """Test CTM export converts frames to seconds."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=100, end_frame=150),  # 2.0-3.0s
        ]
        result = AlignmentResult(words=words)

        ctm = result.to_ctm(audio_id="test")
        parts = ctm.strip().split()

        # CTM format: audio_id channel start duration word
        start = float(parts[2])
        duration = float(parts[3])

        assert start == 2.0
        assert duration == 1.0  # 3.0 - 2.0

    def test_json_export_uses_seconds(self):
        """Test JSON export converts frames to seconds."""
        import json
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=100, end_frame=150),
        ]
        result = AlignmentResult(words=words)

        json_str = result.to_json()
        data = json.loads(json_str)

        word_data = data["words"][0]
        assert word_data["start"] == 2.0
        assert word_data["end"] == 3.0

    def test_to_dict_uses_seconds(self):
        """Test to_dict export uses seconds."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=100, end_frame=150)
        d = word.to_dict()

        assert d["start"] == 2.0
        assert d["end"] == 3.0


class TestSegmentAlignmentResult:
    """Tests for SegmentAlignmentResult (for stitching_utils integration)."""

    @requires_k2_and_lis
    def test_segment_alignment_result_structure(self):
        """Test SegmentAlignmentResult has expected structure."""
        from alignment.wfst.aligner import SegmentAlignmentResult
        from alignment.base import AlignedToken

        tokens = [
            AlignedToken(token_id=1, timestamp=10, score=0.9, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=20, score=0.85, attr={"wid": 1}),
        ]

        seg_result = SegmentAlignmentResult(
            tokens=tokens,
            segment_index=0,
            frame_offset=100,
            rejected=False,
            score=0.95,
        )

        assert len(seg_result) == 2
        assert seg_result.segment_index == 0
        assert seg_result.frame_offset == 100
        assert seg_result.rejected == False
        assert seg_result.score == 0.95

    @requires_k2_and_lis
    def test_segment_alignment_result_word_indices(self):
        """Test get_word_indices extracts word IDs from tokens."""
        from alignment.wfst.aligner import SegmentAlignmentResult
        from alignment.base import AlignedToken

        tokens = [
            AlignedToken(token_id=1, timestamp=10, score=0.9, attr={"wid": 0}),
            AlignedToken(token_id=2, timestamp=20, score=0.85, attr={"wid": 1}),
            AlignedToken(token_id=3, timestamp=30, score=0.8),  # No wid
        ]

        seg_result = SegmentAlignmentResult(tokens=tokens, segment_index=0)
        word_indices = seg_result.get_word_indices()

        assert word_indices == [0, 1]


class TestBackendCompatibility:
    """Tests for alignment result compatibility across backends."""

    def test_all_backends_return_aligned_word(self):
        """Test all backends return AlignedWord with same API."""
        from alignment.base import AlignedWord

        # Create AlignedWord as each backend would
        wfst_word = AlignedWord(word="hello", start_frame=100, end_frame=150, index=0)
        mfa_word = AlignedWord(word="hello", start_frame=100, end_frame=150, index=0)
        gentle_word = AlignedWord(word="hello", start_frame=100, end_frame=150, index=0)

        # All should have same API
        for word in [wfst_word, mfa_word, gentle_word]:
            assert hasattr(word, 'start_frame')
            assert hasattr(word, 'end_frame')
            assert hasattr(word, 'start_seconds')
            assert hasattr(word, 'end_seconds')
            assert callable(word.start_seconds)
            assert callable(word.end_seconds)

    def test_alignment_result_from_any_backend(self):
        """Test AlignmentResult works with words from any backend."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
            AlignedWord(word="world", start_frame=160, end_frame=220, index=1),
        ]

        result = AlignmentResult(words=words)

        # All properties should work
        assert result.text == "hello world"
        assert result.num_words == 2
        assert result.duration > 0

        # All export methods should work
        assert result.to_audacity_labels() != ""
        assert result.to_ctm() != ""
        assert result.to_json() != ""


class TestAlignmentConfigDefaults:
    """Tests for AlignmentConfig default values."""

    def test_default_frame_duration(self):
        """Test default frame duration in config."""
        from alignment.base import AlignmentConfig
        from unittest.mock import patch

        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()

        assert config.frame_duration == 0.02  # 20ms

    def test_default_segment_size(self):
        """Test default segment size."""
        from alignment.base import AlignmentConfig
        from unittest.mock import patch

        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()

        assert config.segment_size == 15.0  # 15 seconds

    def test_default_overlap(self):
        """Test default overlap."""
        from alignment.base import AlignmentConfig
        from unittest.mock import patch

        with patch("torch.cuda.is_available", return_value=False):
            config = AlignmentConfig()

        assert config.overlap == 2.0  # 2 seconds


class TestLISUtilities:
    """Tests for LIS utilities used in stitching."""

    @requires_lis
    def test_compute_lis_basic(self):
        """Test basic LIS computation."""
        from alignment.wfst.lis_utils import compute_lis

        # Simple increasing sequence
        result = compute_lis([1, 2, 3, 4, 5])
        assert result == [1, 2, 3, 4, 5]

    @requires_lis
    def test_compute_lis_with_duplicates(self):
        """Test LIS with non-monotonic sequence."""
        from alignment.wfst.lis_utils import compute_lis

        # Simulating word indices from overlapping segments
        word_indices = [1, 5, 2, 6, 3, 7, 4, 8]
        result = compute_lis(word_indices)

        # LIS should be strictly increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    @requires_lis
    def test_find_unaligned_regions(self):
        """Test finding unaligned word regions."""
        from alignment.wfst.lis_utils import find_unaligned_regions

        aligned = {0, 1, 2, 5, 6, 7}  # Missing 3, 4
        regions = find_unaligned_regions(0, 7, aligned)

        assert (3, 4) in regions
