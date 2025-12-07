"""
Audio Frontend Module for TorchAudio Long-Form Aligner

This module handles audio loading, preprocessing, and segmentation for
long-form speech-to-text alignment.

Main functionality:
- Load audio from various formats (mp3, wav, flac, etc.)
- Resample to target sample rate
- Convert to mono
- Uniform segmentation with overlap for divide-and-conquer alignment
- Optional audio enhancement (denoising, VAD)
"""

# Audio loading
from .loaders import (
    load_audio,
    get_available_backends,
    AudioBackend,
)

# Audio preprocessing
from .preprocessing import (
    resample,
    to_mono,
    normalize_peak,
    preprocess,
)

# Segmentation
from .segmentation import (
    AudioSegment,
    SegmentationResult,
    segment_waveform,
)

# Enhancement
from .enhancement import (
    AudioEnhancement,
    EnhancementResult,
    TimeMappingManager,
    enhance_audio,
    denoise_noisereduce,
    get_available_enhancement_backends,
)

# Main frontend class
from .frontend import (
    AudioFrontend,
    segment_audio,
)

__all__ = [
    # Loaders
    "load_audio",
    "get_available_backends",
    "AudioBackend",
    # Preprocessing
    "resample",
    "to_mono",
    "normalize_peak",
    "preprocess",
    # Segmentation
    "AudioSegment",
    "SegmentationResult",
    "segment_waveform",
    # Enhancement
    "AudioEnhancement",
    "EnhancementResult",
    "TimeMappingManager",
    "enhance_audio",
    "denoise_noisereduce",
    "get_available_enhancement_backends",
    # Frontend
    "AudioFrontend",
    "segment_audio",
]
