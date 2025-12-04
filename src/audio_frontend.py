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
        self.preprocessors = []  # Custom preprocessors (empty by default)

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


# =============================================================================
# Audio Enhancement Module (Demucs + VAD)
# =============================================================================
#
# This module provides optional audio enhancement for better alignment:
# 1. Demucs: Source separation to extract vocals from music/noise
# 2. Silero VAD: Voice Activity Detection to remove silence
#
# Reference: https://github.com/EtienneAb3d/WhisperHallu/
#
# Dependencies (optional):
#   pip install demucs
#   pip install pyloudnorm (for audio normalization)
#
# =============================================================================

# Check for optional enhancement dependencies
_DEMUCS_AVAILABLE = False
_PYLOUDNORM_AVAILABLE = False
_SILERO_VAD_AVAILABLE = False
_NOISEREDUCE_AVAILABLE = False
_DEEPFILTERNET_AVAILABLE = False
_RESEMBLE_ENHANCE_AVAILABLE = False

# 1. Demucs - Source separation (vocals extraction)
try:
    import demucs
    from demucs.pretrained import get_model_from_args
    from demucs.apply import apply_model
    _DEMUCS_AVAILABLE = True
except ImportError:
    demucs = None
    get_model_from_args = None
    apply_model = None

# 2. pyloudnorm - Audio normalization
try:
    import pyloudnorm as pyln
    _PYLOUDNORM_AVAILABLE = True
except ImportError:
    pyln = None

# 3. noisereduce - Spectral gating noise reduction (lightweight, CPU-friendly)
try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None

# 4. DeepFilterNet - Deep learning noise suppression (48kHz full-band)
try:
    from df import enhance as df_enhance, init_df
    _DEEPFILTERNET_AVAILABLE = True
except ImportError:
    df_enhance = None
    init_df = None

# 5. Resemble Enhance - AI speech denoising and enhancement
try:
    from resemble_enhance.enhancer.inference import denoise as resemble_denoise
    from resemble_enhance.enhancer.inference import enhance as resemble_enhance_fn
    _RESEMBLE_ENHANCE_AVAILABLE = True
except ImportError:
    resemble_denoise = None
    resemble_enhance_fn = None


def get_available_enhancement_backends() -> dict:
    """Return availability of audio enhancement backends."""
    return {
        "demucs": _DEMUCS_AVAILABLE,
        "pyloudnorm": _PYLOUDNORM_AVAILABLE,
        "silero_vad": _SILERO_VAD_AVAILABLE,  # Loaded lazily
        "noisereduce": _NOISEREDUCE_AVAILABLE,
        "deepfilternet": _DEEPFILTERNET_AVAILABLE,
        "resemble_enhance": _RESEMBLE_ENHANCE_AVAILABLE,
    }


@dataclass
class TimeMappingManager:
    """
    Manages timestamp mapping when silence is removed from audio.

    When we remove silence periods from audio, the timestamps change.
    This class provides utilities to map between original and processed timestamps.
    This is essential for recovering original timestamps after alignment.

    Example:
        >>> # Audio with silence removed at [(0, 1), (3, 5)] seconds
        >>> mapper = TimeMappingManager([(0, 1), (3, 5)])
        >>> # Processed timestamp 2.0 maps to original timestamp 5.0
        >>> original_time = mapper.map_to_original(2.0)
    """
    silence_intervals: List[Tuple[float, float]]

    def __post_init__(self):
        if len(self.silence_intervals) == 0:
            self._offsets = None
            self._non_silence_starts = None
            return

        # Sort and merge overlapping intervals
        self.silence_intervals = self._sort_and_merge_intervals(self.silence_intervals)

        # Pre-compute offsets for efficient mapping
        import bisect
        offsets = []
        non_sil_cumulative_dur = 0

        if self.silence_intervals[0][0] > 0:
            offsets.append((0, 0))

        prev_sil_end = 0
        for start, end in self.silence_intervals:
            non_sil_cumulative_dur += (start - prev_sil_end)
            prev_sil_end = end
            offsets.append((non_sil_cumulative_dur, end))

        self._offsets = offsets
        self._non_silence_starts = [start for start, _ in offsets]

    def _sort_and_merge_intervals(self, intervals):
        """Sort and merge overlapping intervals."""
        if not intervals:
            return intervals
        intervals = sorted(intervals)
        merged = []
        for start, end in intervals:
            if not merged or merged[-1][1] < start:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def map_to_original(self, processed_timestamp: float) -> float:
        """
        Map a timestamp from processed audio back to original audio.

        Args:
            processed_timestamp: Timestamp in the processed (silence-removed) audio

        Returns:
            Corresponding timestamp in the original audio
        """
        if self._non_silence_starts is None:
            return processed_timestamp

        import bisect
        idx = bisect.bisect_right(self._non_silence_starts, processed_timestamp) - 1

        if idx >= 0:
            return processed_timestamp - self._offsets[idx][0] + self._offsets[idx][1]
        else:
            return processed_timestamp

    def map_to_new(self, original_timestamp: float) -> float:
        """
        Map a timestamp from original audio to processed audio.

        Args:
            original_timestamp: Timestamp in the original audio

        Returns:
            Corresponding timestamp in the processed (silence-removed) audio
        """
        if self.silence_intervals is None or len(self.silence_intervals) == 0:
            return original_timestamp

        if original_timestamp <= self.silence_intervals[0][0]:
            return original_timestamp

        cum_sil_dur = 0
        for start, end in self.silence_intervals:
            if original_timestamp >= end:
                cum_sil_dur += (end - start)
            else:
                if original_timestamp <= start:
                    return original_timestamp - cum_sil_dur
                else:
                    return start - cum_sil_dur

        return original_timestamp - cum_sil_dur


@dataclass
class EnhancementResult:
    """Result of audio enhancement."""
    waveform: torch.Tensor
    sample_rate: int
    time_mapping_managers: List[TimeMappingManager]
    original_duration_seconds: float
    enhanced_duration_seconds: float

    def map_to_original(self, processed_timestamp: float) -> float:
        """Map timestamp from enhanced audio back to original."""
        original_timestamp = processed_timestamp
        for mapper in reversed(self.time_mapping_managers):
            original_timestamp = mapper.map_to_original(original_timestamp)
        return original_timestamp

    def map_to_enhanced(self, original_timestamp: float) -> float:
        """Map timestamp from original audio to enhanced audio."""
        enhanced_timestamp = original_timestamp
        for mapper in self.time_mapping_managers:
            enhanced_timestamp = mapper.map_to_new(enhanced_timestamp)
        return enhanced_timestamp


class AudioEnhancement:
    """
    Audio enhancement using Demucs source separation and Silero VAD.

    This class provides audio denoising and enhancement for better speech alignment:
    1. **Demucs**: Separates audio into sources (vocals, drums, bass, other)
       and extracts the vocals for cleaner speech
    2. **Silence Removal**: Removes long silence periods
    3. **Silero VAD**: Voice Activity Detection to keep only speech segments

    The enhancement pipeline maintains timestamp mappings so alignment results
    can be mapped back to the original audio.

    Requires: pip install demucs pyloudnorm

    Example:
        >>> enhancer = AudioEnhancement()
        >>> result = enhancer.enhance("noisy_audio.mp3")
        >>> # result.waveform is the enhanced audio
        >>> # result.map_to_original(5.0) maps timestamp back to original
    """

    def __init__(
        self,
        device: Optional[str] = None,
        temp_dir: Union[str, Path] = "tmp/",
        demucs_model: str = "htdemucs",
    ):
        """
        Initialize audio enhancement.

        Args:
            device: Device to use ("cuda", "cpu", or None for auto)
            temp_dir: Directory for temporary files
            demucs_model: Demucs model name (default: "htdemucs")
        """
        if not _DEMUCS_AVAILABLE:
            raise ImportError(
                "demucs is required for audio enhancement. "
                "Install with: pip install demucs"
            )

        # Set device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load Demucs model
        logger.info(f"Loading Demucs model: {demucs_model}")
        self.demucs_model = get_model_from_args(
            type("args", (object,), dict(name=demucs_model, repo=None))
        )
        self.demucs_model = self.demucs_model.to(self.device).eval()
        logger.info(f"Demucs sources: {self.demucs_model.sources}")

        # Set up temp directory
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Silero VAD (loaded lazily)
        self._vad_model = None
        self._vad_utils = None

    def _load_vad(self):
        """Lazily load Silero VAD model."""
        if self._vad_model is not None:
            return

        logger.info("Loading Silero VAD model")
        self._vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self._vad_model = self._vad_model.to(self.device)
        self._vad_utils = utils
        global _SILERO_VAD_AVAILABLE
        _SILERO_VAD_AVAILABLE = True

    def extract_vocals(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Extract vocals from audio using Demucs source separation.

        Args:
            waveform: Input waveform (1D or 2D tensor)
            sample_rate: Sample rate of the input

        Returns:
            vocals: Extracted vocals waveform (1D)
            sample_rate: Output sample rate
        """
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to Demucs format
        wav = demucs.audio.convert_audio(
            waveform,
            from_samplerate=sample_rate,
            to_samplerate=self.demucs_model.samplerate,
            channels=self.demucs_model.audio_channels,
        )
        wav = wav.to(self.device)

        # Handle dimensions
        if wav.dim() == 1:
            wav = wav[None, None].repeat_interleave(2, -2)
        else:
            if wav.shape[-2] == 1:
                wav = wav.repeat_interleave(2, -2)
            if wav.dim() < 3:
                wav = wav[None]

        # Apply Demucs
        logger.info("Applying Demucs source separation...")
        result = apply_model(
            self.demucs_model, wav, device=self.device, split=True, overlap=0.25
        )

        # Clear GPU memory
        if self.device.type != "cpu":
            torch.cuda.empty_cache()

        # Extract vocals
        # Model sources: ['drums', 'bass', 'other', 'vocals']
        vocals_idx = self.demucs_model.sources.index("vocals")
        vocals = result[0, vocals_idx].mean(0)  # Average channels to mono

        # Resample back to original sample rate
        vocals = torchaudio.functional.resample(
            vocals, self.demucs_model.samplerate, sample_rate
        )

        logger.info(f"Extracted vocals: {vocals.shape[0]/sample_rate:.2f}s")
        return vocals.cpu(), sample_rate

    def remove_silence(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        silence_threshold_db: float = -50.0,
        min_silence_duration: float = 0.2,
        padding: float = 0.1,
        min_keep_duration: float = 0.4,
    ) -> Tuple[torch.Tensor, TimeMappingManager]:
        """
        Remove silence periods from audio.

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate
            silence_threshold_db: Threshold for silence detection (dB)
            min_silence_duration: Minimum silence duration to detect (seconds)
            padding: Padding around silence periods (seconds)
            min_keep_duration: Minimum duration of silence to actually remove (seconds)

        Returns:
            waveform: Processed waveform with silence removed
            mapper: TimeMappingManager for timestamp recovery
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        duration = waveform.shape[0] / sample_rate

        # Detect silence using energy-based method
        # Convert threshold from dB to linear
        threshold = 10 ** (silence_threshold_db / 20)

        # Frame-based energy detection
        frame_size = int(sample_rate * 0.02)  # 20ms frames
        hop_size = int(sample_rate * 0.01)    # 10ms hop

        silence_intervals = []
        in_silence = False
        silence_start = 0

        for i in range(0, waveform.shape[0] - frame_size, hop_size):
            frame = waveform[i:i + frame_size]
            energy = frame.abs().max().item()

            if energy < threshold:
                if not in_silence:
                    in_silence = True
                    silence_start = i / sample_rate
            else:
                if in_silence:
                    in_silence = False
                    silence_end = i / sample_rate
                    if silence_end - silence_start >= min_silence_duration:
                        silence_intervals.append((silence_start, silence_end))

        # Handle trailing silence
        if in_silence:
            silence_end = waveform.shape[0] / sample_rate
            if silence_end - silence_start >= min_silence_duration:
                silence_intervals.append((silence_start, silence_end))

        # Filter and pad silence intervals
        filtered_intervals = []
        for start, end in silence_intervals:
            start += padding
            end -= padding
            if end - start >= min_keep_duration:
                filtered_intervals.append((start, end))

        # Create time mapping manager
        mapper = TimeMappingManager(filtered_intervals)

        if not filtered_intervals:
            return waveform, mapper

        # Remove silence by keeping non-silence parts
        speech_intervals = self._flip_intervals(filtered_intervals, duration)

        chunks = []
        for start, end in speech_intervals:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            chunks.append(waveform[start_sample:end_sample])

        if chunks:
            waveform = torch.cat(chunks)

        logger.info(
            f"Removed {len(filtered_intervals)} silence periods, "
            f"duration: {duration:.2f}s -> {waveform.shape[0]/sample_rate:.2f}s"
        )

        return waveform, mapper

    def apply_vad(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        threshold: float = 0.4,
        min_silence_duration_ms: int = 500,
        padding: float = 0.1,
        min_keep_duration: float = 0.4,
    ) -> Tuple[torch.Tensor, TimeMappingManager]:
        """
        Apply Voice Activity Detection to keep only speech segments.

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate
            threshold: VAD confidence threshold (0-1)
            min_silence_duration_ms: Minimum silence duration for VAD (ms)
            padding: Padding around detected silence (seconds)
            min_keep_duration: Minimum silence duration to remove (seconds)

        Returns:
            waveform: Processed waveform with non-speech removed
            mapper: TimeMappingManager for timestamp recovery
        """
        # Load VAD model lazily
        self._load_vad()

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        duration = waveform.shape[0] / sample_rate

        # Move to device for VAD
        wav_device = waveform.to(self.device)

        # Get speech timestamps
        get_speech_timestamps = self._vad_utils[0]
        collect_chunks = self._vad_utils[4]

        speech_timestamps = get_speech_timestamps(
            wav_device,
            self._vad_model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
            sampling_rate=sample_rate,
        )

        # Convert to intervals
        speech_intervals = [
            (ts['start'] / sample_rate, ts['end'] / sample_rate)
            for ts in speech_timestamps
        ]

        # Get silence intervals (complement of speech)
        silence_intervals = self._flip_intervals(speech_intervals, duration)

        # Filter silence intervals
        filtered_intervals = []
        for start, end in silence_intervals:
            start += padding
            end -= padding
            if end - start >= min_keep_duration:
                filtered_intervals.append((start, end))

        # Create time mapping manager
        mapper = TimeMappingManager(filtered_intervals)

        # Collect speech chunks
        waveform = waveform.cpu()

        if speech_timestamps:
            # Recalculate after filtering
            speech_intervals = self._flip_intervals(filtered_intervals, duration)
            filtered_timestamps = [
                {"start": int(s * sample_rate), "end": int(e * sample_rate)}
                for s, e in speech_intervals
            ]
            waveform = collect_chunks(filtered_timestamps, waveform)

        logger.info(
            f"VAD: kept {len(speech_timestamps)} speech segments, "
            f"duration: {duration:.2f}s -> {waveform.shape[0]/sample_rate:.2f}s"
        )

        return waveform, mapper

    def _flip_intervals(
        self,
        intervals: List[Tuple[float, float]],
        end_time: float,
    ) -> List[Tuple[float, float]]:
        """Convert silence intervals to speech intervals (or vice versa)."""
        if not intervals:
            return [(0, end_time)]

        flipped = []
        if intervals[0][0] > 0:
            flipped.append((0, intervals[0][0]))
        for i in range(len(intervals) - 1):
            flipped.append((intervals[i][1], intervals[i + 1][0]))
        if intervals[-1][1] < end_time:
            flipped.append((intervals[-1][1], end_time))
        return flipped

    # =========================================================================
    # Additional Enhancement Methods (noisereduce, DeepFilterNet, Resemble)
    # =========================================================================

    def apply_noisereduce(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        stationary: bool = False,
        prop_decrease: float = 1.0,
        n_fft: int = 512,
    ) -> torch.Tensor:
        """
        Apply spectral gating noise reduction using noisereduce.

        This is a lightweight, CPU-friendly noise reduction method using
        spectral gating. Good for light background noise.

        Requires: pip install noisereduce

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate
            stationary: If True, use stationary noise reduction (faster)
            prop_decrease: Proportion to reduce noise by (0-1)
            n_fft: FFT size (512 recommended for speech)

        Returns:
            Denoised waveform
        """
        if not _NOISEREDUCE_AVAILABLE:
            raise ImportError(
                "noisereduce is required. Install with: pip install noisereduce"
            )

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Convert to numpy
        audio = waveform.numpy()

        logger.info(f"Applying noisereduce (stationary={stationary})...")

        # Apply noise reduction
        reduced = nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease,
            n_fft=n_fft,
        )

        logger.info("noisereduce complete")
        return torch.from_numpy(reduced).float()

    def apply_deepfilternet(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply DeepFilterNet for deep learning-based noise suppression.

        DeepFilterNet is a Low Complexity Speech Enhancement Framework
        for Full-Band Audio (48kHz). It provides high-quality noise
        suppression using deep filtering.

        Requires: pip install deepfilternet

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate (will be resampled to 48kHz internally)

        Returns:
            enhanced_waveform: Enhanced audio
            sample_rate: Output sample rate (resampled back to original)
        """
        if not _DEEPFILTERNET_AVAILABLE:
            raise ImportError(
                "DeepFilterNet is required. Install with: pip install deepfilternet"
            )

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        logger.info("Applying DeepFilterNet...")

        # Initialize model (cached after first call)
        if not hasattr(self, '_df_model'):
            self._df_model, self._df_state, _ = init_df()

        # DeepFilterNet expects 48kHz
        target_sr = 48000
        if sample_rate != target_sr:
            waveform_48k = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        else:
            waveform_48k = waveform

        # Apply enhancement
        enhanced = df_enhance(self._df_model, self._df_state, waveform_48k.unsqueeze(0))
        enhanced = enhanced.squeeze()

        # Resample back to original sample rate
        if sample_rate != target_sr:
            enhanced = torchaudio.functional.resample(enhanced, target_sr, sample_rate)

        logger.info("DeepFilterNet complete")
        return enhanced, sample_rate

    def apply_resemble_enhance(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        denoise_only: bool = False,
        solver: str = "midpoint",
        nfe: int = 64,
        tau: float = 0.5,
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply Resemble Enhance for AI-powered speech denoising and enhancement.

        Resemble Enhance consists of two modules:
        1. Denoiser: Separates speech from noisy audio
        2. Enhancer: Boosts perceptual quality by restoring distortions
           and extending audio bandwidth

        Requires: pip install resemble-enhance

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate
            denoise_only: If True, only apply denoising (faster)
            solver: ODE solver for enhancement ("midpoint", "rk4", "euler")
            nfe: Number of function evaluations (higher = better quality but slower)
            tau: Temperature parameter for enhancement

        Returns:
            enhanced_waveform: Enhanced audio
            sample_rate: Output sample rate (44.1kHz for full enhancement)
        """
        if not _RESEMBLE_ENHANCE_AVAILABLE:
            raise ImportError(
                "Resemble Enhance is required. Install with: pip install resemble-enhance"
            )

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        logger.info(f"Applying Resemble Enhance (denoise_only={denoise_only})...")

        # Resemble Enhance works at 44.1kHz
        target_sr = 44100
        if sample_rate != target_sr:
            waveform_44k = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        else:
            waveform_44k = waveform

        if denoise_only:
            # Just denoise
            enhanced, out_sr = resemble_denoise(waveform_44k, target_sr, self.device)
        else:
            # Full enhancement (denoise + quality boost)
            enhanced, out_sr = resemble_enhance_fn(
                waveform_44k,
                target_sr,
                self.device,
                solver=solver,
                nfe=nfe,
                tau=tau,
            )

        # Resample back to original sample rate if needed
        if out_sr != sample_rate:
            enhanced = torchaudio.functional.resample(enhanced, out_sr, sample_rate)

        logger.info("Resemble Enhance complete")
        return enhanced, sample_rate

    def normalize_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        peak_db: float = -1.0,
        loudness_lufs: float = -12.0,
    ) -> torch.Tensor:
        """
        Normalize audio for consistent volume.

        Args:
            waveform: Input waveform
            sample_rate: Sample rate
            peak_db: Target peak level in dB
            loudness_lufs: Target loudness in LUFS

        Returns:
            Normalized waveform
        """
        if not _PYLOUDNORM_AVAILABLE:
            logger.warning("pyloudnorm not available, skipping normalization")
            return waveform

        # Convert to numpy for pyloudnorm
        if waveform.dim() > 1:
            waveform = waveform.mean(0)
        if isinstance(waveform, torch.Tensor):
            audio = waveform.numpy()
        else:
            audio = waveform

        # Peak normalize
        peak_normalized = pyln.normalize.peak(audio, peak_db)

        # Loudness normalize
        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        normalized = pyln.normalize.loudness(peak_normalized, loudness, loudness_lufs)

        return torch.from_numpy(normalized).float()

    def enhance(
        self,
        audio_or_path: Union[str, Path, torch.Tensor],
        sample_rate: Optional[int] = None,
        denoise_method: Optional[str] = None,
        extract_vocals: bool = False,
        remove_silence: bool = True,
        apply_vad: bool = True,
        normalize: bool = False,
    ) -> EnhancementResult:
        """
        Full audio enhancement pipeline with multiple denoising options.

        Args:
            audio_or_path: Audio file path or waveform tensor
            sample_rate: Sample rate (required if audio_or_path is tensor)
            denoise_method: Denoising method to use. Options:
                - None: No denoising (default)
                - "demucs": Demucs vocal extraction (best for music/heavy noise)
                - "noisereduce": Spectral gating (lightweight, CPU-friendly)
                - "deepfilternet": DeepFilterNet (48kHz full-band, real-time capable)
                - "resemble": Resemble Enhance (highest quality, slower)
            extract_vocals: [DEPRECATED] Use denoise_method="demucs" instead
            remove_silence: Remove silence periods
            apply_vad: Apply Voice Activity Detection
            normalize: Apply loudness normalization

        Returns:
            EnhancementResult with enhanced waveform and timestamp mappings

        Example:
            >>> enhancer = AudioEnhancement()
            >>> # Light denoising (fast, CPU)
            >>> result = enhancer.enhance("audio.mp3", denoise_method="noisereduce")
            >>> # Heavy denoising (music/noise separation)
            >>> result = enhancer.enhance("audio.mp3", denoise_method="demucs")
            >>> # Highest quality
            >>> result = enhancer.enhance("audio.mp3", denoise_method="resemble")
        """
        # Handle deprecated extract_vocals parameter
        if extract_vocals and denoise_method is None:
            denoise_method = "demucs"

        # Load audio if path
        if isinstance(audio_or_path, (str, Path)):
            waveform, sample_rate = torchaudio.load(str(audio_or_path))
            if waveform.dim() > 1:
                waveform = waveform.mean(0)  # Convert to mono
        else:
            waveform = audio_or_path
            if waveform.dim() > 1:
                waveform = waveform.mean(0)
            assert sample_rate is not None, "sample_rate required for tensor input"

        original_duration = waveform.shape[0] / sample_rate
        time_mapping_managers = []
        step = 1

        # Step: Apply denoising method
        if denoise_method:
            logger.info(f"Step {step}: Applying {denoise_method} denoising...")
            if denoise_method == "demucs":
                waveform, sample_rate = self.extract_vocals(waveform, sample_rate)
            elif denoise_method == "noisereduce":
                waveform = self.apply_noisereduce(waveform, sample_rate)
            elif denoise_method == "deepfilternet":
                waveform, sample_rate = self.apply_deepfilternet(waveform, sample_rate)
            elif denoise_method == "resemble":
                waveform, sample_rate = self.apply_resemble_enhance(waveform, sample_rate)
            else:
                raise ValueError(
                    f"Unknown denoise_method: {denoise_method}. "
                    f"Options: demucs, noisereduce, deepfilternet, resemble"
                )
            step += 1

        # Step: Remove silence
        if remove_silence:
            logger.info(f"Step {step}: Removing silence...")
            waveform, mapper = self.remove_silence(waveform, sample_rate)
            time_mapping_managers.append(mapper)
            step += 1

        # Step: Apply VAD
        if apply_vad:
            logger.info(f"Step {step}: Applying Voice Activity Detection...")
            waveform, mapper = self.apply_vad(waveform, sample_rate)
            time_mapping_managers.append(mapper)
            step += 1

        # Step: Normalize
        if normalize:
            logger.info(f"Step {step}: Normalizing audio...")
            waveform = self.normalize_audio(waveform, sample_rate)

        enhanced_duration = waveform.shape[0] / sample_rate

        logger.info(
            f"Enhancement complete: {original_duration:.2f}s -> {enhanced_duration:.2f}s "
            f"({100 * enhanced_duration / original_duration:.1f}%)"
        )

        return EnhancementResult(
            waveform=waveform,
            sample_rate=sample_rate,
            time_mapping_managers=time_mapping_managers,
            original_duration_seconds=original_duration,
            enhanced_duration_seconds=enhanced_duration,
        )


# Convenience functions
def enhance_audio(
    audio_path: Union[str, Path],
    denoise_method: Optional[str] = None,
    remove_silence: bool = True,
    apply_vad: bool = True,
    device: Optional[str] = None,
) -> EnhancementResult:
    """
    Convenience function to enhance audio file.

    Args:
        audio_path: Path to audio file
        denoise_method: Denoising method. Options:
            - None: No denoising
            - "demucs": Vocal extraction (best for music/heavy noise)
            - "noisereduce": Spectral gating (fast, CPU-friendly)
            - "deepfilternet": DeepFilterNet (48kHz, real-time capable)
            - "resemble": Resemble Enhance (highest quality)
        remove_silence: Remove silence periods
        apply_vad: Apply Voice Activity Detection
        device: Device to use ("cuda", "cpu", or None for auto)

    Returns:
        EnhancementResult with enhanced audio and timestamp mappings

    Example:
        >>> # Light denoising (fast)
        >>> result = enhance_audio("audio.mp3", denoise_method="noisereduce")
        >>> # Heavy denoising (music separation)
        >>> result = enhance_audio("audio.mp3", denoise_method="demucs")
        >>> # Highest quality
        >>> result = enhance_audio("audio.mp3", denoise_method="resemble")
    """
    enhancer = AudioEnhancement(device=device)
    return enhancer.enhance(
        audio_path,
        denoise_method=denoise_method,
        remove_silence=remove_silence,
        apply_vad=apply_vad,
    )


# Quick denoise functions for common use cases
def denoise_noisereduce(
    audio_path: Union[str, Path],
    stationary: bool = False,
) -> Tuple[torch.Tensor, int]:
    """
    Quick noisereduce denoising (lightweight, CPU-friendly).

    Args:
        audio_path: Path to audio file
        stationary: Use stationary noise reduction (faster)

    Returns:
        waveform: Denoised audio
        sample_rate: Sample rate
    """
    if not _NOISEREDUCE_AVAILABLE:
        raise ImportError("noisereduce required. Install: pip install noisereduce")

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(0)

    reduced = nr.reduce_noise(y=waveform.numpy(), sr=sr, stationary=stationary)
    return torch.from_numpy(reduced).float(), sr


def denoise_deepfilternet(
    audio_path: Union[str, Path],
) -> Tuple[torch.Tensor, int]:
    """
    Quick DeepFilterNet denoising (48kHz full-band, real-time capable).

    Args:
        audio_path: Path to audio file

    Returns:
        waveform: Denoised audio
        sample_rate: Sample rate
    """
    if not _DEEPFILTERNET_AVAILABLE:
        raise ImportError("DeepFilterNet required. Install: pip install deepfilternet")

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(0)

    model, df_state, _ = init_df()

    # Resample to 48kHz
    if sr != 48000:
        waveform = torchaudio.functional.resample(waveform, sr, 48000)

    enhanced = df_enhance(model, df_state, waveform.unsqueeze(0))
    enhanced = enhanced.squeeze()

    # Resample back
    if sr != 48000:
        enhanced = torchaudio.functional.resample(enhanced, 48000, sr)

    return enhanced, sr


def denoise_resemble(
    audio_path: Union[str, Path],
    denoise_only: bool = True,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Quick Resemble Enhance denoising (highest quality).

    Args:
        audio_path: Path to audio file
        denoise_only: If True, only denoise (faster). If False, full enhancement.
        device: Device to use

    Returns:
        waveform: Enhanced audio
        sample_rate: Sample rate
    """
    if not _RESEMBLE_ENHANCE_AVAILABLE:
        raise ImportError("Resemble Enhance required. Install: pip install resemble-enhance")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(0)

    # Resample to 44.1kHz
    if sr != 44100:
        waveform = torchaudio.functional.resample(waveform, sr, 44100)

    if denoise_only:
        enhanced, out_sr = resemble_denoise(waveform, 44100, device)
    else:
        enhanced, out_sr = resemble_enhance_fn(waveform, 44100, device)

    # Resample back
    if out_sr != sr:
        enhanced = torchaudio.functional.resample(enhanced, out_sr, sr)

    return enhanced, sr
