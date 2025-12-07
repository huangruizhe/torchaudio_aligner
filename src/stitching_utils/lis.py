"""
LIS (Longest Increasing Subsequence) based stitching.

This algorithm uses the property that word indices are monotonic in the
transcript. By finding the LIS of word indices across all segments,
we identify the globally consistent alignment path.

Complexity: O(N log N) where N is total number of tokens

Best for:
- WFST alignment with word index output labels
- Long audio with many overlapping segments
- Fuzzy transcripts where ordering matters

Requires: pip install git+https://github.com/huangruizhe/lis.git
"""

from typing import List, Tuple, Set, Dict, Optional
import logging
import itertools

from .base import (
    StitcherBackend,
    StitchingConfig,
    StitchingResult,
    SegmentAlignment,
    AlignedToken,
)

logger = logging.getLogger(__name__)


class LISStitcher(StitcherBackend):
    """
    LIS-based stitcher for combining segment alignments.

    Uses Longest Increasing Subsequence to find globally consistent
    alignment across overlapping segments.

    Example:
        >>> stitcher = LISStitcher(config)
        >>> result = stitcher.stitch(segment_alignments)
        >>> print(f"Aligned {result.num_aligned_tokens} tokens")
    """

    METHOD_NAME = "lis"

    def __init__(self, config: Optional[StitchingConfig] = None):
        super().__init__(config)
        self._lis_available = None

    def _check_lis_available(self) -> bool:
        """Check if LIS library is available."""
        if self._lis_available is None:
            try:
                import lis
                self._lis_available = True
            except ImportError:
                self._lis_available = False
        return self._lis_available

    def stitch(
        self,
        segment_alignments: List[SegmentAlignment],
        **kwargs,
    ) -> StitchingResult:
        """
        Stitch segment alignments using LIS algorithm.

        Args:
            segment_alignments: List of SegmentAlignment from each segment
            **kwargs: Override config parameters

        Returns:
            StitchingResult with globally consistent alignment
        """
        if not self._check_lis_available():
            raise ImportError(
                "LIS library required. "
                "Install with: pip install git+https://github.com/huangruizhe/lis.git"
            )

        self._validate_input(segment_alignments)

        # Get config values (allow override)
        neighborhood_size = kwargs.get(
            "neighborhood_size", self.config.neighborhood_size
        )
        scan_range = kwargs.get(
            "outlier_scan_range", self.config.outlier_scan_range
        )
        outlier_threshold = kwargs.get(
            "outlier_threshold", self.config.outlier_threshold
        )

        # Step 1: Extract word indices from all segments
        logger.info("Step 1: Extracting word indices from segments")
        all_word_indices = []
        for seg in segment_alignments:
            if not seg.rejected:
                all_word_indices.extend(seg.get_word_indices())

        if not all_word_indices:
            logger.warning("No word indices found in segment alignments")
            return StitchingResult(
                tokens=[],
                unaligned_regions=[],
                metadata={"method": self.METHOD_NAME, "error": "no_word_indices"},
            )

        logger.info(f"Found {len(all_word_indices)} word indices")

        # Step 2: Compute LIS
        logger.info("Step 2: Computing Longest Increasing Subsequence")
        lis_result = compute_lis(all_word_indices)
        logger.info(f"LIS length: {len(lis_result)}")

        # Step 3: Remove outliers
        logger.info("Step 3: Removing outliers")
        lis_result = remove_outliers(
            lis_result,
            scan_range=scan_range,
            outlier_threshold=outlier_threshold,
        )
        logger.info(f"After outlier removal: {len(lis_result)}")

        if not lis_result:
            return StitchingResult(
                tokens=[],
                unaligned_regions=[],
                metadata={"method": self.METHOD_NAME, "error": "empty_after_outlier"},
            )

        # Step 4: Remove isolated words
        logger.info("Step 4: Removing isolated words")
        lis_result = remove_isolated_words(lis_result, neighborhood_size)
        logger.info(f"After isolated word removal: {len(lis_result)}")

        if not lis_result:
            return StitchingResult(
                tokens=[],
                unaligned_regions=[],
                metadata={"method": self.METHOD_NAME, "error": "empty_after_isolated"},
            )

        # Step 5: Align LIS back to tokens
        logger.info("Step 5: Aligning LIS to original tokens")
        segment_alignments = align_lis_to_tokens(lis_result, segment_alignments)

        # Step 6: Collect resolved tokens
        logger.info("Step 6: Collecting resolved tokens")
        resolved_tokens, segment_mapping = self._collect_lis_tokens(segment_alignments)

        # Step 7: Find unaligned regions
        logger.info("Step 7: Finding unaligned regions")
        rg_min = lis_result[0]
        rg_max = lis_result[-1]
        set_lis = set(lis_result)
        unaligned_regions = find_unaligned_regions(rg_min, rg_max, set_lis)

        logger.info(
            f"Stitching complete: {len(resolved_tokens)} tokens, "
            f"{len(unaligned_regions)} unaligned regions"
        )

        return StitchingResult(
            tokens=resolved_tokens,
            unaligned_regions=unaligned_regions,
            segment_mapping=segment_mapping,
            metadata={
                "method": self.METHOD_NAME,
                "original_word_indices": len(all_word_indices),
                "lis_length": len(lis_result),
                "range": (rg_min, rg_max),
            },
        )

    def _collect_lis_tokens(
        self,
        segment_alignments: List[SegmentAlignment],
    ) -> Tuple[List[AlignedToken], List[int]]:
        """
        Collect tokens that are part of the LIS.

        Includes both word-start tokens (marked with "lis") and
        their following character tokens.

        Returns:
            (resolved_tokens, segment_indices)
        """
        resolved_tokens = []
        segment_mapping = []

        for seg in segment_alignments:
            if seg.rejected:
                continue

            word_start_flag = False
            for token in seg.tokens:
                if token.attr.get("lis", False):
                    # This is a word-start token in LIS
                    resolved_tokens.append(token)
                    segment_mapping.append(seg.segment_index)
                    word_start_flag = True
                elif "wid" in token.attr:
                    # New word, not in LIS
                    word_start_flag = False
                elif word_start_flag:
                    # Character within LIS word
                    resolved_tokens.append(token)
                    segment_mapping.append(seg.segment_index)

        return resolved_tokens, segment_mapping


# ===========================================================================
# LIS Utility Functions
# ===========================================================================

def compute_lis(word_indices: List[int]) -> List[int]:
    """
    Compute Longest Increasing Subsequence of word indices.

    Uses O(N log N) algorithm for efficiency with long sequences.

    Args:
        word_indices: List of word indices from alignment

    Returns:
        LIS as a list of word indices
    """
    try:
        import lis
    except ImportError:
        raise ImportError(
            "LIS library required for alignment concatenation. "
            "Install with: pip install git+https://github.com/huangruizhe/lis.git"
        )

    if not word_indices:
        return []

    return lis.longestIncreasingSubsequence(word_indices)


def remove_outliers(
    my_list: List[int],
    scan_range: int = 100,
    outlier_threshold: int = 60,
) -> List[int]:
    """
    Remove outlier word indices from LIS results.

    An outlier is a word index with a gap larger than outlier_threshold
    from its neighbors at the beginning or end of the sequence.

    Args:
        my_list: Sorted list of word indices
        scan_range: How far to scan from ends
        outlier_threshold: Gap size to consider as outlier

    Returns:
        List with outliers removed
    """
    if len(my_list) <= 10:
        return my_list

    scan_range = min(scan_range, int(len(my_list) / 2) - 1)

    # Find left outliers
    left = [
        i + 1 for i in range(0, scan_range)
        if my_list[i + 1] - my_list[i] > outlier_threshold
    ]
    left = left[-1] if left else 0

    # Find right outliers
    right = [
        i - 1 for i in range(-scan_range, 0)
        if my_list[i] - my_list[i - 1] > outlier_threshold
    ]
    right = right[0] + 1 if right else None

    return my_list[left:right]


def remove_isolated_words(
    lis_results: List[int],
    neighborhood_size: int = 5,
    neighbor_threshold: float = 0.4,
) -> List[int]:
    """
    Remove isolated aligned words from LIS results.

    A word is considered isolated if less than neighbor_threshold
    of its neighbors are also aligned.

    Args:
        lis_results: LIS word indices
        neighborhood_size: How many neighbors to consider
        neighbor_threshold: Minimum fraction of aligned neighbors

    Returns:
        LIS with isolated words removed
    """
    if not lis_results:
        return []

    rg_min = lis_results[0]
    rg_max = lis_results[-1]
    set_lis = set(lis_results)

    for i in range(rg_min, rg_max + 1):
        if i in set_lis:
            left_neighbors = [
                j for j in range(i - neighborhood_size, i)
                if j in set_lis
            ]
            right_neighbors = [
                j for j in range(i + 1, i + neighborhood_size + 1)
                if j in set_lis
            ]

            num_left = i - max(i - neighborhood_size, rg_min)
            num_right = min(i + neighborhood_size, rg_max) - i

            # Check if isolated on both sides
            if (num_left > 0 and num_right > 0 and
                    len(left_neighbors) < neighbor_threshold * num_left and
                    len(right_neighbors) < neighbor_threshold * num_right):
                set_lis.remove(i)

    return [i for i in lis_results if i in set_lis]


def find_unaligned_regions(
    rg_min: int,
    rg_max: int,
    aligned_indices: Set[int],
    merge_threshold: int = 3,
) -> List[Tuple[int, int]]:
    """
    Find unaligned "holes" in the transcript.

    Args:
        rg_min: Minimum word index in alignment
        rg_max: Maximum word index in alignment
        aligned_indices: Set of aligned word indices
        merge_threshold: Merge holes closer than this

    Returns:
        List of (start, end) tuples for unaligned regions (both inclusive)
    """
    # Find consecutive unaligned regions
    holes = [
        [rg_min + i for i, _ in group]
        for key, group in itertools.groupby(
            enumerate(range(rg_min, rg_max + 1)),
            key=lambda x: x[1] in aligned_indices
        )
        if not key
    ]

    # Convert to (start, end) tuples
    holes = [(group[0], group[-1]) for group in holes]

    # Merge close holes
    if merge_threshold > 0:
        holes = merge_segments(holes, merge_threshold)

    return holes


def merge_segments(
    segments: List[Tuple[int, int]],
    threshold: int,
) -> List[Tuple[int, int]]:
    """
    Merge segments that are closer than threshold.

    Args:
        segments: List of (start, end) tuples
        threshold: Maximum gap to merge

    Returns:
        Merged segments
    """
    if not segments:
        return []

    merged = []
    for start, end in sorted(segments, key=lambda x: x[0]):
        if merged and start - merged[-1][1] <= threshold:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def align_lis_to_tokens(
    lis_results: List[int],
    segment_alignments: List[SegmentAlignment],
) -> List[SegmentAlignment]:
    """
    Align LIS results back to original alignment tokens.

    Uses a heuristic for "tightness":
    - First half: align to last occurrence (tight at beginning)
    - Second half: align to first occurrence (tight at end)

    Args:
        lis_results: LIS word indices
        segment_alignments: List of SegmentAlignment

    Returns:
        segment_alignments with "lis" attr marked on LIS tokens
    """
    if not lis_results:
        return segment_alignments

    midpoint = len(lis_results) // 2

    lis_ali = dict()
    j = 0  # Index in lis_results

    outer_break = False
    for seg in segment_alignments:
        if seg.rejected:
            continue

        for token in seg.tokens:
            if j >= len(lis_results):
                outer_break = True
                break

            wid = token.attr.get("wid", None)

            if j < midpoint:
                # First half: align to last occurrence
                if lis_results[j] == wid:
                    lis_ali[lis_results[j]] = token
                    j += 1
            else:
                # Second half: align to first occurrence
                if lis_results[j] == wid and lis_results[j] not in lis_ali:
                    lis_ali[lis_results[j]] = token
                    j += 1

        if outer_break:
            break

    # Mark LIS tokens
    for token in lis_ali.values():
        token.attr["lis"] = True

    return segment_alignments
