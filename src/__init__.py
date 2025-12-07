"""
TorchAudio Long-Form Aligner

A modular library for forced alignment of long-form speech with text.

Modules:
- text_frontend: Text loading, normalization, romanization, tokenization
- audio_frontend: Audio loading, preprocessing, segmentation, enhancement
- labeling_utils: Frame-wise posterior extraction from CTC models
"""

from . import text_frontend
from . import audio_frontend
from . import labeling_utils

__all__ = [
    "text_frontend",
    "audio_frontend",
    "labeling_utils",
]

# Version
__version__ = "0.1.0"
