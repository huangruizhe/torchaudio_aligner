"""
Longest Increasing Subsequence (LIS) utilities for alignment concatenation.

The LIS algorithm is used to find reliable alignments across overlapping
segments. Since word indices are monotonic in the transcript, the LIS
identifies the most consistent alignment path.

Requires: pip install git+https://github.com/huangruizhe/lis.git
"""

from typing import List, Tuple, Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


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
            if (len(left_neighbors) < neighbor_threshold * num_left and
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
        List of (start, end) tuples for unaligned regions
    """
    import itertools

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


def get_lis_alignment(
    lis_results: List[int],
    alignment_results: List,
) -> List:
    """
    Align LIS results back to original alignment tokens.

    Uses a heuristic for "tightness":
    - First half: align to last occurrence
    - Second half: align to first occurrence

    Args:
        lis_results: LIS word indices
        alignment_results: List of AlignedToken lists

    Returns:
        alignment_results with "lis" attr marked on LIS tokens
    """
    midpoint = len(lis_results) // 2

    lis_ali = dict()
    j = 0  # Index in lis_results

    outer_break = False
    for aligned_tokens in alignment_results:
        for token in aligned_tokens:
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

    return alignment_results
