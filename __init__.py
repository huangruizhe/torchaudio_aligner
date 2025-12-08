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

Usage:
    from torchaudio_aligner import align_long_audio

    result = align_long_audio(
        audio="path/to/audio.mp3",
        text="path/to/transcript.pdf",
        language="eng",
    )

    for word in result:
        print(f"{word.word}: {word.start_seconds():.2f}s - {word.end_seconds():.2f}s")
"""

# Import from src module
from .src import (
    # High-level API
    align_long_audio,
    AlignmentResult,
    LongFormAlignmentResult,
    AlignedWord,
    AlignedChar,
    AlignmentConfig,
    # Modules
    text_frontend,
    audio_frontend,
    labeling_utils,
    alignment,
    stitching_utils,
    visualization_utils,
)

__version__ = "0.1.0"

__all__ = [
    # High-level API
    "align_long_audio",
    "AlignmentResult",
    "LongFormAlignmentResult",
    "AlignedWord",
    "AlignedChar",
    "AlignmentConfig",
    # Modules
    "text_frontend",
    "audio_frontend",
    "labeling_utils",
    "alignment",
    "stitching_utils",
    "visualization_utils",
]
