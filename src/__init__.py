"""
TorchAudio Long-Form Aligner

A modular library for forced alignment of long-form speech with text.

Usage:
    from torchaudio_aligner import align_long_audio

    result = align_long_audio(
        audio="path/to/audio.mp3",
        text="path/to/transcript.pdf",
        language="eng",
    )

Modules:
- text_frontend: Text loading, normalization, romanization, tokenization
- audio_frontend: Audio loading, preprocessing, segmentation, enhancement
- labeling_utils: Frame-wise posterior extraction from CTC models
- alignment: WFST-based flexible alignment
- stitching_utils: LIS-based segment concatenation
- visualization_utils: Audacity, Gentle visualization, audio preview
"""

from . import text_frontend
from . import audio_frontend
from . import labeling_utils
from . import alignment
from . import stitching_utils
from . import visualization_utils

# High-level API
from .api import (
    align_long_audio,
    LongFormAlignmentResult,
    second_pass_refinement,
)

__all__ = [
    # Modules
    "text_frontend",
    "audio_frontend",
    "labeling_utils",
    "alignment",
    "stitching_utils",
    "visualization_utils",
    # High-level API
    "align_long_audio",
    "LongFormAlignmentResult",
    "second_pass_refinement",
]

# Version
__version__ = "0.1.0"
