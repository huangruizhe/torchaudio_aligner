"""
Audio Frontend Module for TorchAudio Long-Form Aligner

This module handles audio loading, preprocessing, and segmentation for
long-form speech-to-text alignment.

Main functionality:
- Load audio from various formats (mp3, wav, flac, etc.)
- Resample to target sample rate
- Convert to mono
- Uniform segmentation with overlap for divide-and-conquer alignment

Supported audio backends:
- torchaudio (primary, requires torchcodec for torchaudio >= 2.8)
- soundfile (fallback)
"""

from dataclasses import dataclass
from typing import Optional, List, Callable, Tuple, Union, Literal
from pathlib import Path
import logging

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Check for soundfile backend
_SOUNDFILE_AVAILABLE = False
try:
    import soundfile as sf
    _SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None


def get_available_backends() -> List[str]:
    """Return list of available audio loading backends."""
    backends = ["torchaudio"]
    if _SOUNDFILE_AVAILABLE:
        backends.append("soundfile")
    return backends


def _load_with_torchaudio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio using torchaudio."""
    return torchaudio.load(audio_path)


def _load_with_soundfile(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load audio using soundfile."""
    if sf is None:
        raise ImportError("soundfile is not installed. Install with: pip install soundfile")

    data, sample_rate = sf.read(audio_path, dtype="float32")
    # soundfile returns (samples, channels), we need (channels, samples)
    waveform = torch.from_numpy(data)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    return waveform, sample_rate


AudioBackend = Literal["torchaudio", "soundfile", "auto"]


@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata."""
    waveform: torch.Tensor  # Shape: (num_samples,) or (channels, num_samples)
    sample_rate: int
    offset_samples: int  # Offset in samples from the start of the original audio
    length_samples: int  # Actual length of this segment (may be shorter for last segment)
    segment_index: int  # Index of this segment in the sequence

    @property
    def offset_seconds(self) -> float:
        """Offset in seconds from the start of the original audio."""
        return self.offset_samples / self.sample_rate

    @property
    def duration_seconds(self) -> float:
        """Duration of this segment in seconds."""
        return self.length_samples / self.sample_rate


@dataclass
class SegmentationResult:
    """Result of audio segmentation containing all segments and metadata."""
    segments: List[AudioSegment]
    original_duration_samples: int
    original_duration_seconds: float
    sample_rate: int
    segment_size_samples: int
    overlap_samples: int
    num_segments: int

    def get_waveforms_batched(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stack all segment waveforms into a batch tensor.

        Returns:
            waveforms: Tensor of shape (num_segments, max_segment_length)
            lengths: Tensor of shape (num_segments,) with actual lengths
        """
        max_len = max(seg.waveform.shape[-1] for seg in self.segments)
        batch_size = len(self.segments)

        # Determine number of channels
        if self.segments[0].waveform.dim() == 1:
            waveforms = torch.zeros(batch_size, max_len)
        else:
            num_channels = self.segments[0].waveform.shape[0]
            waveforms = torch.zeros(batch_size, num_channels, max_len)

        lengths = torch.zeros(batch_size, dtype=torch.long)

        for i, seg in enumerate(self.segments):
            length = seg.waveform.shape[-1]
            if seg.waveform.dim() == 1:
                waveforms[i, :length] = seg.waveform
            else:
                waveforms[i, :, :length] = seg.waveform
            lengths[i] = length

        return waveforms, lengths

    def get_offsets_in_frames(self, frame_duration_seconds: float) -> torch.Tensor:
        """
        Get segment offsets in frames (for acoustic model output).

        Args:
            frame_duration_seconds: Duration of each frame in seconds (e.g., 0.02 for 20ms)

        Returns:
            Tensor of shape (num_segments,) with frame offsets
        """
        offsets = torch.tensor([seg.offset_samples for seg in self.segments])
        return (offsets / self.sample_rate / frame_duration_seconds).long()


class AudioFrontend:
    """
    Audio frontend for loading, preprocessing, and segmenting audio files.

    This class provides a pipeline for preparing long-form audio for alignment:
    1. Load audio from file (supports various formats)
    2. Resample to target sample rate
    3. Convert to mono if needed
    4. Apply optional preprocessing (normalization)
    5. Segment into overlapping chunks for divide-and-conquer alignment

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
            target_sample_rate: Target sample rate for output audio. Default 16000 Hz
                is standard for most speech models (MMS, Wav2Vec2, etc.)
            mono: If True, convert stereo audio to mono by averaging channels.
            normalize: If True, apply peak normalization to the audio.
            normalize_db: Target peak level in dB for normalization (default -3.0 dB).
            backend: Audio loading backend to use. Options:
                - "auto": Try torchaudio first, fall back to soundfile (default)
                - "torchaudio": Use torchaudio (requires torchcodec for torchaudio >= 2.8)
                - "soundfile": Use soundfile (pip install soundfile)
        """
        self.target_sample_rate = target_sample_rate
        self.mono = mono
        self.normalize = normalize
        self.normalize_db = normalize_db
        self.backend = backend

        # Validate backend
        if backend not in ("auto", "torchaudio", "soundfile"):
            raise ValueError(f"Unknown backend: {backend}. Available: torchaudio, soundfile, auto")

        if backend == "soundfile" and not _SOUNDFILE_AVAILABLE:
            raise RuntimeError(
                "Backend 'soundfile' is not available. "
                "Install with: pip install soundfile"
            )

    def load(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """
        Load audio from file.

        Args:
            audio_path: Path to audio file (supports mp3, wav, flac, etc.)

        Returns:
            waveform: Tensor of shape (channels, num_samples)
            sample_rate: Original sample rate of the audio

        Raises:
            FileNotFoundError: If audio file does not exist
            RuntimeError: If audio file cannot be loaded
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio_path_str = str(audio_path)
        logger.info(f"Loading audio from: {audio_path}")

        if self.backend == "soundfile":
            waveform, sample_rate = _load_with_soundfile(audio_path_str)
            logger.info("Loaded with backend 'soundfile'")
        elif self.backend == "torchaudio":
            waveform, sample_rate = _load_with_torchaudio(audio_path_str)
            logger.info("Loaded with backend 'torchaudio'")
        else:  # auto
            try:
                waveform, sample_rate = _load_with_torchaudio(audio_path_str)
                logger.info("Loaded with backend 'torchaudio'")
            except Exception as e:
                logger.debug(f"torchaudio failed: {e}, trying soundfile")
                if _SOUNDFILE_AVAILABLE:
                    waveform, sample_rate = _load_with_soundfile(audio_path_str)
                    logger.info("Loaded with backend 'soundfile' (fallback)")
                else:
                    raise RuntimeError(
                        f"Failed to load audio with torchaudio: {e}\n"
                        "Install soundfile as fallback: pip install soundfile"
                    )

        logger.info(
            f"Loaded audio: shape={waveform.shape}, "
            f"sample_rate={sample_rate}, "
            f"duration={waveform.shape[1] / sample_rate:.2f}s"
        )

        return waveform, sample_rate

    def resample(
        self,
        waveform: torch.Tensor,
        orig_sample_rate: int,
        target_sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Resample audio to target sample rate.

        Args:
            waveform: Input waveform tensor
            orig_sample_rate: Original sample rate
            target_sample_rate: Target sample rate (uses self.target_sample_rate if None)

        Returns:
            Resampled waveform tensor
        """
        target_sr = target_sample_rate or self.target_sample_rate

        if orig_sample_rate == target_sr:
            return waveform

        logger.info(f"Resampling from {orig_sample_rate} Hz to {target_sr} Hz")
        waveform = torchaudio.functional.resample(waveform, orig_sample_rate, target_sr)

        return waveform

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-channel audio to mono.

        Args:
            waveform: Input waveform of shape (channels, num_samples)

        Returns:
            Mono waveform of shape (1, num_samples)
        """
        if waveform.shape[0] == 1:
            return waveform

        logger.info(f"Converting {waveform.shape[0]} channels to mono")
        return waveform.mean(dim=0, keepdim=True)

    def apply_normalization(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply peak normalization to audio.

        Args:
            waveform: Input waveform tensor

        Returns:
            Normalized waveform tensor
        """
        peak = waveform.abs().max()
        if peak > 0:
            target_peak = 10 ** (self.normalize_db / 20)
            waveform = waveform * (target_peak / peak)
            logger.info(f"Normalized audio to {self.normalize_db} dB peak")
        return waveform

    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Apply all preprocessing steps to the waveform.

        Args:
            waveform: Input waveform tensor of shape (channels, num_samples)
            sample_rate: Sample rate of the waveform

        Returns:
            Preprocessed waveform tensor
        """
        # Convert to mono if requested
        if self.mono:
            waveform = self.to_mono(waveform)

        # Apply normalization if requested
        if self.normalize:
            waveform = self.apply_normalization(waveform)

        # Apply custom preprocessors
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
        """
        Segment audio into overlapping chunks.

        This implements the divide-and-conquer approach from the paper:
        long audio is split into overlapping segments that can be processed
        independently and then merged.

        Args:
            waveform: Input waveform tensor of shape (channels, num_samples) or (num_samples,)
            sample_rate: Sample rate of the waveform
            segment_size: Size of each segment in seconds (default 15.0s)
            overlap: Overlap between consecutive segments in seconds (default 2.0s)
            min_segment_size: Minimum size for the last segment in seconds.
                If the last segment is shorter than this, it will be discarded.
            extra_samples: Extra samples to add to segment size to ensure we get
                the expected number of frames from the acoustic model (default 128).
                This accounts for edge effects in convolution-based models.

        Returns:
            SegmentationResult containing all segments and metadata
        """
        # Handle different input shapes
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension

        num_channels, total_samples = waveform.shape

        # Calculate sizes in samples
        segment_size_samples = int(sample_rate * segment_size) + extra_samples
        overlap_samples = int(sample_rate * overlap) + extra_samples
        min_segment_samples = int(sample_rate * min_segment_size)
        step_size = segment_size_samples - overlap_samples

        logger.info(
            f"Segmenting audio: total_samples={total_samples}, "
            f"segment_size={segment_size_samples}, "
            f"overlap={overlap_samples}, "
            f"step_size={step_size}"
        )

        segments = []
        segment_idx = 0
        offset = 0

        while offset < total_samples:
            # Calculate end position for this segment
            end = min(offset + segment_size_samples, total_samples)
            segment_length = end - offset

            # Skip if remaining audio is too short
            if segment_length < min_segment_samples:
                logger.debug(
                    f"Skipping final segment of {segment_length} samples "
                    f"(< {min_segment_samples} min)"
                )
                break

            # Extract segment
            segment_waveform = waveform[:, offset:end]

            # Squeeze channel dimension if mono
            if num_channels == 1:
                segment_waveform = segment_waveform.squeeze(0)

            segment = AudioSegment(
                waveform=segment_waveform,
                sample_rate=sample_rate,
                offset_samples=offset,
                length_samples=segment_length,
                segment_index=segment_idx,
            )
            segments.append(segment)

            segment_idx += 1
            offset += step_size

            # If we've reached or passed the end, stop
            if end >= total_samples:
                break

        logger.info(f"Created {len(segments)} segments")

        return SegmentationResult(
            segments=segments,
            original_duration_samples=total_samples,
            original_duration_seconds=total_samples / sample_rate,
            sample_rate=sample_rate,
            segment_size_samples=segment_size_samples,
            overlap_samples=overlap_samples,
            num_segments=len(segments),
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

        This is the main entry point for the audio frontend.

        Args:
            audio_path: Path to audio file
            segment_size: Size of each segment in seconds (default 15.0s)
            overlap: Overlap between segments in seconds (default 2.0s)
            min_segment_size: Minimum size for last segment in seconds (default 0.2s)
            extra_samples: Extra samples for segment size (default 128)

        Returns:
            SegmentationResult containing processed and segmented audio

        Example:
            >>> frontend = AudioFrontend(target_sample_rate=16000)
            >>> result = frontend.process(
            ...     "lecture.mp3",
            ...     segment_size=15.0,
            ...     overlap=2.0
            ... )
            >>> print(f"Processed {result.original_duration_seconds:.1f}s audio")
            >>> print(f"Created {result.num_segments} segments")
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


# Convenience function for simple use cases
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
        target_sample_rate: Target sample rate (default 16000 Hz)
        segment_size: Segment size in seconds (default 15.0s)
        overlap: Overlap in seconds (default 2.0s)
        mono: Convert to mono (default True)
        normalize: Apply peak normalization (default False)
        backend: Audio loading backend ("auto", "torchaudio", "soundfile")

    Returns:
        SegmentationResult with segmented audio

    Example:
        >>> result = segment_audio("lecture.mp3")
        >>> waveforms, lengths = result.get_waveforms_batched()
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


def load_audio(
    audio_path: Union[str, Path],
    backend: AudioBackend = "auto",
) -> Tuple[torch.Tensor, int]:
    """
    Convenience function to load audio file.

    Args:
        audio_path: Path to audio file
        backend: Audio loading backend ("auto", "torchaudio", "soundfile")

    Returns:
        waveform: Tensor of shape (channels, num_samples)
        sample_rate: Sample rate of the audio

    Example:
        >>> waveform, sr = load_audio("audio.mp3")
        >>> print(f"Loaded {waveform.shape[1]/sr:.2f}s of audio at {sr}Hz")
    """
    frontend = AudioFrontend(backend=backend)
    return frontend.load(audio_path)
