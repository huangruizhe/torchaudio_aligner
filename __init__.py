"""
TorchAudio Long-Form Aligner

A toolkit for robust speech-to-text alignment of long-form audio with
fuzzy/noisy transcriptions. Supports 1000+ languages via MMS and other
CTC-based acoustic models.

Main features:
- Long-form audio alignment (hours of audio)
- Fuzzy text matching (handles noisy/non-verbatim transcriptions)
- Multi-lingual support (1000+ languages)
- GPU acceleration
- Any CTC model as acoustic backbone
"""

from .audio_frontend import (
    AudioFrontend,
    AudioSegment,
    SegmentationResult,
    segment_audio,
    load_audio,
    get_available_backends,
)

__version__ = "0.1.0"

__all__ = [
    "AudioFrontend",
    "AudioSegment",
    "SegmentationResult",
    "segment_audio",
    "load_audio",
    "get_available_backends",
]
