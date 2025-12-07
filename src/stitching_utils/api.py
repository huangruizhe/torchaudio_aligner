"""
High-level API for stitching utilities.

Provides convenient functions for common stitching operations.
"""

from typing import List, Optional, Dict, Any

from .base import (
    StitcherBackend,
    StitchingConfig,
    StitchingResult,
    SegmentAlignment,
)
from .lis import LISStitcher
from .edit_distance import EditDistanceStitcher
from .diff import DiffStitcher

# Registry of available stitchers
_STITCHERS = {
    "lis": LISStitcher,
    "edit_distance": EditDistanceStitcher,
    "levenshtein": EditDistanceStitcher,  # Alias
    "diff": DiffStitcher,
    "gentle": DiffStitcher,  # Alias
}


def get_stitcher(
    method: str = "lis",
    config: Optional[StitchingConfig] = None,
    **kwargs,
) -> StitcherBackend:
    """
    Get a stitcher instance by method name.

    Args:
        method: Stitching method ("lis", "edit_distance", "diff")
        config: StitchingConfig (created with kwargs if not provided)
        **kwargs: Passed to StitchingConfig

    Returns:
        StitcherBackend instance

    Example:
        >>> stitcher = get_stitcher("lis", neighborhood_size=10)
        >>> result = stitcher.stitch(segment_alignments)
    """
    method_lower = method.lower()

    if method_lower not in _STITCHERS:
        raise ValueError(
            f"Unknown stitching method: {method}. "
            f"Available: {list(_STITCHERS.keys())}"
        )

    if config is None:
        config = StitchingConfig(method=method_lower, **kwargs)

    return _STITCHERS[method_lower](config)


def stitch_alignments(
    segment_alignments: List[SegmentAlignment],
    method: str = "lis",
    reference_words: Optional[List[str]] = None,
    config: Optional[StitchingConfig] = None,
    **kwargs,
) -> StitchingResult:
    """
    Stitch segment alignments into global alignment.

    This is the main entry point for stitching operations.

    Args:
        segment_alignments: List of SegmentAlignment from alignment module
        method: Stitching method ("lis", "edit_distance", "diff")
        reference_words: Reference transcript (needed for edit_distance/diff)
        config: StitchingConfig
        **kwargs: Additional parameters for stitcher

    Returns:
        StitchingResult with globally consistent alignment

    Example:
        >>> from alignment import WFSTAligner
        >>> from stitching_utils import stitch_alignments
        >>>
        >>> # Get segment alignments
        >>> segment_results = aligner.align_segments(waveform, text)
        >>>
        >>> # Stitch with LIS (default)
        >>> result = stitch_alignments(segment_results)
        >>>
        >>> # Or stitch with edit distance
        >>> result = stitch_alignments(
        ...     segment_results,
        ...     method="edit_distance",
        ...     reference_words=text.split(),
        ... )
    """
    stitcher = get_stitcher(method, config)

    # Pass reference_words for methods that need it
    if method.lower() in ("edit_distance", "levenshtein", "diff", "gentle"):
        return stitcher.stitch(
            segment_alignments,
            reference_words=reference_words,
            **kwargs,
        )
    else:
        return stitcher.stitch(segment_alignments, **kwargs)


def list_methods() -> Dict[str, Dict[str, Any]]:
    """
    List available stitching methods.

    Returns:
        Dict with method info
    """
    return {
        "lis": {
            "class": "LISStitcher",
            "description": "Longest Increasing Subsequence - O(N log N)",
            "requires_reference": False,
            "requires_word_indices": True,
            "best_for": "WFST alignment with word index output labels",
        },
        "edit_distance": {
            "class": "EditDistanceStitcher",
            "description": "Levenshtein distance alignment - O(N*M)",
            "requires_reference": True,
            "requires_word_indices": False,
            "best_for": "ASR output or MFA/Gentle results",
            "aliases": ["levenshtein"],
        },
        "diff": {
            "class": "DiffStitcher",
            "description": "Python difflib-based alignment",
            "requires_reference": True,
            "requires_word_indices": False,
            "best_for": "Mostly accurate transcripts",
            "aliases": ["gentle"],
        },
    }
