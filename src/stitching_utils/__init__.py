"""
Stitching Utils - Combine overlapping segment alignments into global alignment.

This module provides algorithms to stitch together segment-wise alignment results
from the alignment module into a coherent long-form alignment.

The problem:
- Long audio is segmented with overlap (e.g., 15s segments, 2s overlap)
- Each segment is aligned independently, producing local timestamps
- Segments may have: duplicates in overlap, misalignments, gaps

Stitching algorithms:
1. LIS (Longest Increasing Subsequence) - O(N log N)
   - Uses word indices to find globally consistent alignment
   - Best for: fuzzy alignment with word index output labels

2. Edit Distance (Levenshtein) - O(N*M)
   - Aligns predicted text to reference text
   - Best for: ASR output alignment, MFA/Gentle results

3. Diff-based (Gentle style)
   - Uses diff algorithm to match aligned words to reference
   - Best for: accurate transcripts with minor errors

Usage:
    from stitching_utils import (
        stitch_alignments,
        LISStitcher,
        EditDistanceStitcher,
        DiffStitcher,
    )

    # Automatic stitching
    result = stitch_alignments(segment_alignments, method="lis")

    # Or use specific stitcher
    stitcher = LISStitcher(neighborhood_size=5)
    result = stitcher.stitch(segment_alignments)

Example:
    >>> from alignment import WFSTAligner
    >>> from stitching_utils import stitch_alignments
    >>>
    >>> # Get segment-wise alignments
    >>> aligner = WFSTAligner(config)
    >>> segment_results = aligner.align_segments(waveform, text)
    >>>
    >>> # Stitch segments together
    >>> final_result = stitch_alignments(segment_results, method="lis")

TODO: Future enhancements (inter-operation with alignment module)
-------------------------------------------------------------
After stitching, there may be "holes" or unaligned regions where:
- The reference text exists but no alignment was found
- These could be actual unspoken text OR alignment failures

Planned heuristics (following Gentle aligner / Tutorial.py approach):
1. Detect unaligned regions from StitchingResult.unaligned_regions
2. Extract the corresponding audio segment (using timestamps of surrounding aligned words)
3. Re-align or re-forced-align the missing part using the alignment module
4. Merge the new alignments back into the result

This allows recovery of alignments that failed due to:
- Segment boundary issues
- Local acoustic noise
- Temporary model confusion

See:
- Gentle: https://github.com/strob/gentle/blob/master/gentle/forced_aligner.py
- Tutorial.py: realign_unaligned_words() pattern
"""

from .base import (
    StitcherBackend,
    StitchingConfig,
    SegmentAlignment,
)
from .lis import LISStitcher
from .edit_distance import EditDistanceStitcher
from .diff import DiffStitcher, word_diff
from .api import stitch_alignments, get_stitcher, list_methods

__all__ = [
    # Base classes
    "StitcherBackend",
    "StitchingConfig",
    "SegmentAlignment",
    # Stitchers
    "LISStitcher",
    "EditDistanceStitcher",
    "DiffStitcher",
    # Utilities
    "word_diff",  # Gentle-style word diff
    # API
    "stitch_alignments",
    "get_stitcher",
    "list_methods",
]
