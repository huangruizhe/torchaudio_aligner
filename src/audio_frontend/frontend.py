"""
Audio Frontend Class

Main class for audio loading, preprocessing, and segmentation.
"""

from pathlib import Path
from typing import Optional, Union, List, Callable

import torch

from .loaders import load_audio, AudioBackend
from .preprocessing import resample, to_mono, normalize_peak
from .segmentation import segment_waveform, SegmentationResult, AudioSegment


class AudioFrontend:
    """
    Audio frontend for loading, preprocessing, and segmenting audio files.

    This class provides a pipeline for preparing long-form audio for alignment:
    1. Load audio from file
    2. Resample to target sample rate
    3. Convert to mono if needed
    4. Apply optional preprocessing
    5. Segment into overlapping chunks

    Example:
        >>> frontend = AudioFrontend(target_sample_rate=16000)
        >>> result = frontend.process("audio.mp3", segment_size=15.0, overlap=2.0)
        >>> print(f"Created {result.num_segments} segments")
        >>> waveforms, lengths = result.get_waveforms_batched()
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        mono: bool = True,
        normalize: bool = False,
        normalize_db: float = -3.0,
        backend: AudioBackend = "auto",
    ):
        """
        Initialize the audio frontend.

        Args:
            target_sample_rate: Target sample rate for output audio
            mono: If True, convert stereo audio to mono
            normalize: If True, apply peak normalization
            normalize_db: Target peak level in dB for normalization
            backend: Audio loading backend ("auto", "torchaudio", "soundfile")
        """
        self.target_sample_rate = target_sample_rate
        self.mono = mono
        self.normalize = normalize
        self.normalize_db = normalize_db
        self.backend = backend
        self.preprocessors: List[Callable[[torch.Tensor, int], torch.Tensor]] = []

    def load(self, audio_path: Union[str, Path]) -> tuple:
        """Load audio from file."""
        return load_audio(audio_path, backend=self.backend)

    def resample(
        self,
        waveform: torch.Tensor,
        orig_sample_rate: int,
        target_sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Resample audio to target sample rate."""
        target_sr = target_sample_rate or self.target_sample_rate
        return resample(waveform, orig_sample_rate, target_sr)

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono."""
        return to_mono(waveform)

    def apply_normalization(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply peak normalization to audio."""
        return normalize_peak(waveform, self.normalize_db)

    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """Apply all preprocessing steps to the waveform."""
        if self.mono:
            waveform = self.to_mono(waveform)
        if self.normalize:
            waveform = self.apply_normalization(waveform)
        for preprocessor in self.preprocessors:
            waveform = preprocessor(waveform, sample_rate)
        return waveform

    def segment(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment_size: float = 15.0,
        overlap: float = 2.0,
        min_segment_size: float = 0.2,
        extra_samples: int = 128,
    ) -> SegmentationResult:
        """Segment audio into overlapping chunks."""
        return segment_waveform(
            waveform,
            sample_rate,
            segment_size=segment_size,
            overlap=overlap,
            min_segment_size=min_segment_size,
            extra_samples=extra_samples,
        )

    def process(
        self,
        audio_path: Union[str, Path],
        segment_size: float = 15.0,
        overlap: float = 2.0,
        min_segment_size: float = 0.2,
        extra_samples: int = 128,
    ) -> SegmentationResult:
        """
        Full processing pipeline: load, preprocess, and segment audio.

        Args:
            audio_path: Path to audio file
            segment_size: Size of each segment in seconds
            overlap: Overlap between segments in seconds
            min_segment_size: Minimum size for last segment in seconds
            extra_samples: Extra samples for segment size

        Returns:
            SegmentationResult containing processed and segmented audio
        """
        # Load audio
        waveform, orig_sample_rate = self.load(audio_path)

        # Resample to target sample rate
        waveform = self.resample(waveform, orig_sample_rate)

        # Apply preprocessing
        waveform = self.preprocess(waveform, self.target_sample_rate)

        # Segment
        result = self.segment(
            waveform,
            self.target_sample_rate,
            segment_size=segment_size,
            overlap=overlap,
            min_segment_size=min_segment_size,
            extra_samples=extra_samples,
        )

        return result

    def process_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment_size: float = 15.0,
        overlap: float = 2.0,
        min_segment_size: float = 0.2,
        extra_samples: int = 128,
    ) -> SegmentationResult:
        """
        Process a waveform tensor instead of loading from file.

        Args:
            waveform: Input waveform tensor
            sample_rate: Sample rate of the waveform
            segment_size: Size of each segment in seconds
            overlap: Overlap between segments in seconds
            min_segment_size: Minimum size for last segment in seconds
            extra_samples: Extra samples for segment size

        Returns:
            SegmentationResult containing processed and segmented audio
        """
        # Resample to target sample rate
        waveform = self.resample(waveform, sample_rate)

        # Apply preprocessing
        waveform = self.preprocess(waveform, self.target_sample_rate)

        # Segment
        result = self.segment(
            waveform,
            self.target_sample_rate,
            segment_size=segment_size,
            overlap=overlap,
            min_segment_size=min_segment_size,
            extra_samples=extra_samples,
        )

        return result


# Convenience functions
def segment_audio(
    audio_path: Union[str, Path],
    target_sample_rate: int = 16000,
    segment_size: float = 15.0,
    overlap: float = 2.0,
    mono: bool = True,
    normalize: bool = False,
    backend: AudioBackend = "auto",
) -> SegmentationResult:
    """
    Convenience function to segment audio file with default settings.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate
        segment_size: Segment size in seconds
        overlap: Overlap in seconds
        mono: Convert to mono
        normalize: Apply peak normalization
        backend: Audio loading backend

    Returns:
        SegmentationResult with segmented audio
    """
    frontend = AudioFrontend(
        target_sample_rate=target_sample_rate,
        mono=mono,
        normalize=normalize,
        backend=backend,
    )
    return frontend.process(
        audio_path,
        segment_size=segment_size,
        overlap=overlap,
    )
