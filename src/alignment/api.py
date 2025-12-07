"""
High-level API for speech-to-text alignment.

This module provides the main entry points for alignment:
- align(): Single audio alignment
- align_long_audio(): Long-form audio with automatic segmentation
"""

from typing import Optional, Union
import logging

import torch

from .base import AlignmentConfig, AlignmentResult, AlignerBackend
from .wfst import WFSTAligner
from .mfa import MFAAligner
from .gentle import GentleAligner

logger = logging.getLogger(__name__)

# Backend registry
_BACKENDS = {
    "wfst": WFSTAligner,
    "k2": WFSTAligner,
    "mfa": MFAAligner,
    "gentle": GentleAligner,
}


def get_aligner(
    backend: str = "wfst",
    config: Optional[AlignmentConfig] = None,
    **kwargs,
) -> AlignerBackend:
    """
    Get an aligner instance by backend name.

    Args:
        backend: Backend name ("wfst", "mfa", "gentle")
        config: AlignmentConfig (created if not provided)
        **kwargs: Passed to AlignmentConfig

    Returns:
        AlignerBackend instance

    Example:
        >>> aligner = get_aligner("wfst", language="eng")
        >>> aligner.set_model(model_backend)  # For WFST
        >>> result = aligner.align(waveform, text)
    """
    if config is None:
        config = AlignmentConfig(backend=backend, **kwargs)

    backend_lower = backend.lower()
    if backend_lower not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {list(_BACKENDS.keys())}"
        )

    return _BACKENDS[backend_lower](config)


def align(
    waveform: torch.Tensor,
    text: str,
    model_backend=None,
    config: Optional[AlignmentConfig] = None,
    backend: str = "wfst",
    **kwargs,
) -> AlignmentResult:
    """
    Align audio to text.

    This is the main high-level function for alignment.

    Args:
        waveform: Audio tensor (samples,) or (1, samples)
        text: Text to align (can be noisy for WFST backend)
        model_backend: CTCModelBackend for WFST (required for WFST)
        config: AlignmentConfig
        backend: Backend to use ("wfst", "mfa", "gentle")
        **kwargs: Additional config options

    Returns:
        AlignmentResult with word alignments

    Example:
        >>> from labeling_utils import load_model
        >>> from alignment import align

        >>> # Load model
        >>> model = load_model("mms-fa")

        >>> # Simple alignment
        >>> result = align(waveform, text, model_backend=model)

        >>> # Access results
        >>> for idx, word in result.word_alignments.items():
        ...     print(f"{word.word}: {word.start_seconds:.2f}s")
    """
    if config is None:
        config = AlignmentConfig(backend=backend, **kwargs)

    aligner = get_aligner(config.backend, config)

    # Set model for WFST backend
    if config.backend in ("wfst", "k2"):
        if model_backend is None:
            raise ValueError(
                "model_backend is required for WFST alignment. "
                "Load with: model_backend = load_model('mms-fa')"
            )
        aligner.set_model(model_backend)
    else:
        aligner.load()

    return aligner.align(waveform, text)


def align_long_audio(
    waveform: torch.Tensor,
    text: str,
    model_backend=None,
    config: Optional[AlignmentConfig] = None,
    segment_size: float = 15.0,
    overlap: float = 2.0,
    **kwargs,
) -> AlignmentResult:
    """
    Align long audio with automatic segmentation.

    This function handles long audio files (hours) by:
    1. Segmenting with overlap
    2. Aligning each segment
    3. Concatenating using LIS algorithm

    Args:
        waveform: Audio tensor (can be hours long)
        text: Text to align
        model_backend: CTCModelBackend (required)
        config: AlignmentConfig
        segment_size: Segment size in seconds
        overlap: Overlap between segments in seconds
        **kwargs: Additional config options

    Returns:
        AlignmentResult

    Example:
        >>> # Align a 1-hour audio file
        >>> result = align_long_audio(
        ...     waveform,
        ...     text,
        ...     model_backend=model,
        ...     segment_size=15.0,
        ...     overlap=2.0,
        ... )
    """
    if config is None:
        config = AlignmentConfig(
            backend="wfst",
            segment_size=segment_size,
            overlap=overlap,
            **kwargs,
        )
    else:
        config.segment_size = segment_size
        config.overlap = overlap

    return align(
        waveform,
        text,
        model_backend=model_backend,
        config=config,
        backend="wfst",
    )


# ===========================================================================
# ASR + ngram LM Alignment (Placeholder)
# ===========================================================================

class ASRNgramAligner(AlignerBackend):
    """
    Placeholder for ASR + ngram LM alignment.

    This approach uses:
    1. ASR decoding with ngram language model
    2. Lattice rescoring with target text
    3. Best path extraction

    Supports:
    - CTC models
    - RNN-T models
    - Attention-based encoder-decoder

    Status: NOT IMPLEMENTED (placeholder for future work)
    """

    BACKEND_NAME = "asr_ngram"

    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        logger.warning(
            "ASRNgramAligner is a placeholder and not yet implemented. "
            "Use WFSTAligner for fuzzy alignment instead."
        )

    def load(self):
        raise NotImplementedError(
            "ASRNgramAligner is not yet implemented. "
            "This would require:\n"
            "1. ngram LM training on target text\n"
            "2. Lattice generation from ASR model\n"
            "3. Lattice rescoring with ngram LM\n"
            "4. Best path extraction with timestamps\n"
            "\n"
            "For now, use WFSTAligner which provides similar functionality "
            "using factor transducers with skip/return arcs."
        )

    def align(
        self,
        waveform: torch.Tensor,
        text: str,
        **kwargs,
    ) -> AlignmentResult:
        raise NotImplementedError("ASRNgramAligner not implemented")


# Register placeholder
_BACKENDS["asr_ngram"] = ASRNgramAligner


def list_backends() -> dict:
    """
    List available alignment backends.

    Returns:
        Dict with backend info
    """
    return {
        "wfst": {
            "class": "WFSTAligner",
            "description": "WFST/k2-based fuzzy alignment with factor transducer",
            "languages": "1100+ (with MMS)",
            "fuzzy": True,
            "requires_model": True,
        },
        "mfa": {
            "class": "MFAAligner",
            "description": "Montreal Forced Aligner (Kaldi-based)",
            "languages": "Many (with pretrained models)",
            "fuzzy": False,
            "requires_model": False,
        },
        "gentle": {
            "class": "GentleAligner",
            "description": "Gentle aligner for English",
            "languages": "English only",
            "fuzzy": False,
            "requires_model": False,
        },
        "asr_ngram": {
            "class": "ASRNgramAligner",
            "description": "ASR + ngram LM alignment (NOT IMPLEMENTED)",
            "languages": "N/A",
            "fuzzy": True,
            "requires_model": True,
            "status": "placeholder",
        },
    }
