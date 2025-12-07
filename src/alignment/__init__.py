"""
Alignment module for speech-to-text alignment.

This module provides multiple alignment backends:
- WFST/k2-based alignment with factor transducer for fuzzy text matching
- Montreal Forced Aligner (MFA) integration
- Gentle aligner integration (English only)
- ASR + ngram LM alignment (placeholder)

Architecture:
- Alignment produces SEGMENT-WISE results (each segment aligned independently)
- For global alignment, use stitching_utils to combine segments
- The align() method can optionally do stitching for convenience

Main entry point:
    >>> from alignment import align, AlignmentConfig
    >>> from labeling_utils import load_model

    >>> # Load acoustic model
    >>> model = load_model("mms-fa")

    >>> # Align
    >>> result = align(waveform, text, model_backend=model)

    >>> # Access results
    >>> for idx, word in result.word_alignments.items():
    ...     print(f"{word.word}: {word.start_seconds:.2f}s")

Backends:
    - WFSTAligner: Fuzzy alignment for long audio with noisy transcripts
    - MFAAligner: Montreal Forced Aligner for accurate transcripts
    - GentleAligner: English-only alignment with Gentle

Example - Segment-wise alignment (for use with stitching_utils):
    >>> from alignment import WFSTAligner, AlignmentConfig
    >>> from stitching_utils import stitch_alignments, SegmentAlignment

    >>> aligner = WFSTAligner(config)
    >>> aligner.set_model(model)

    >>> # Get segment-wise results (no stitching)
    >>> segment_results = aligner.align_segments(waveform, text)

    >>> # Convert and stitch
    >>> stitch_input = [
    ...     SegmentAlignment(
    ...         tokens=seg.tokens,
    ...         segment_index=seg.segment_index,
    ...         frame_offset=seg.frame_offset,
    ...         rejected=seg.rejected,
    ...     )
    ...     for seg in segment_results
    ... ]
    >>> final = stitch_alignments(stitch_input, method="lis")

Example - Full pipeline (alignment + stitching):
    >>> from alignment import align_long_audio

    >>> result = align_long_audio(
    ...     waveform,
    ...     noisy_transcript,
    ...     model_backend=model,
    ...     segment_size=15.0,
    ...     overlap=2.0,
    ... )
"""

from .base import (
    AlignmentResult,
    AlignedWord,
    AlignedToken,
    AlignmentConfig,
    AlignerBackend,
)
from .wfst import WFSTAligner
from .wfst.aligner import SegmentAlignmentResult
from .mfa import MFAAligner
from .gentle import GentleAligner
from .api import align, align_long_audio, get_aligner, list_backends

__all__ = [
    # Data classes
    "AlignmentResult",
    "AlignedWord",
    "AlignedToken",
    "AlignmentConfig",
    "SegmentAlignmentResult",
    # Base class
    "AlignerBackend",
    # Backends
    "WFSTAligner",
    "MFAAligner",
    "GentleAligner",
    # Main API
    "align",
    "align_long_audio",
    "get_aligner",
    "list_backends",
]
