"""
Audio Enhancement Module

Optional audio enhancement for better alignment:
- Demucs: Source separation (vocal extraction)
- Silero VAD: Voice Activity Detection
- noisereduce: Spectral gating denoising
- DeepFilterNet: Deep learning denoising
- Resemble Enhance: AI speech enhancement
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Union
import logging

import torch
import torchaudio

logger = logging.getLogger(__name__)

# Check for optional enhancement dependencies
_DEMUCS_AVAILABLE = False
_PYLOUDNORM_AVAILABLE = False
_SILERO_VAD_AVAILABLE = False
_NOISEREDUCE_AVAILABLE = False
_DEEPFILTERNET_AVAILABLE = False
_RESEMBLE_ENHANCE_AVAILABLE = False

# 1. Demucs - Source separation
try:
    from demucs.pretrained import get_model_from_args
    from demucs.apply import apply_model
    _DEMUCS_AVAILABLE = True
except ImportError:
    get_model_from_args = None
    apply_model = None

# 2. pyloudnorm - Audio normalization
try:
    import pyloudnorm as pyln
    _PYLOUDNORM_AVAILABLE = True
except ImportError:
    pyln = None

# 3. noisereduce - Spectral gating
try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    nr = None

# 4. DeepFilterNet - Deep learning denoising
try:
    from df import enhance as df_enhance_fn, init_df
    _DEEPFILTERNET_AVAILABLE = True
except ImportError:
    df_enhance_fn = None
    init_df = None

# 5. Resemble Enhance - AI speech enhancement
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
        "silero_vad": _SILERO_VAD_AVAILABLE,
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
        """Map a timestamp from processed audio back to original audio."""
        if self._non_silence_starts is None:
            return processed_timestamp

        import bisect
        idx = bisect.bisect_right(self._non_silence_starts, processed_timestamp) - 1

        if idx >= 0:
            return processed_timestamp - self._offsets[idx][0] + self._offsets[idx][1]
        else:
            return processed_timestamp

    def map_to_new(self, original_timestamp: float) -> float:
        """Map a timestamp from original audio to processed audio."""
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
    Audio enhancement using various denoising and VAD methods.

    This class provides audio denoising and enhancement for better speech alignment:
    1. **Demucs**: Separates audio into sources and extracts vocals
    2. **Silence Removal**: Removes long silence periods
    3. **Silero VAD**: Voice Activity Detection
    4. **noisereduce**: Spectral gating (lightweight)
    5. **DeepFilterNet**: Deep learning denoising (48kHz)
    6. **Resemble Enhance**: AI speech enhancement (highest quality)

    Example:
        >>> enhancer = AudioEnhancement()
        >>> result = enhancer.enhance("noisy_audio.mp3", denoise_method="noisereduce")
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
        # Set device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Set up temp directory
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Demucs model (loaded lazily)
        self._demucs_model = None
        self._demucs_model_name = demucs_model

        # Silero VAD (loaded lazily)
        self._vad_model = None
        self._vad_utils = None

        # DeepFilterNet (loaded lazily)
        self._df_model = None
        self._df_state = None

    def _load_demucs(self):
        """Lazily load Demucs model."""
        if self._demucs_model is not None:
            return

        if not _DEMUCS_AVAILABLE:
            raise ImportError("demucs is required. Install with: pip install demucs")

        logger.info(f"Loading Demucs model: {self._demucs_model_name}")
        self._demucs_model = get_model_from_args(
            type("args", (object,), dict(name=self._demucs_model_name, repo=None))
        )
        self._demucs_model = self._demucs_model.to(self.device).eval()
        logger.info(f"Demucs sources: {self._demucs_model.sources}")

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

    def _load_deepfilternet(self):
        """Lazily load DeepFilterNet model."""
        if self._df_model is not None:
            return

        if not _DEEPFILTERNET_AVAILABLE:
            raise ImportError("DeepFilterNet is required. Install: pip install deepfilternet")

        logger.info("Loading DeepFilterNet model")
        self._df_model, self._df_state, _ = init_df()

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
        self._load_demucs()

        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Convert to stereo and resample using torchaudio (demucs.audio API removed)
        if waveform.shape[0] == 1:
            waveform_stereo = waveform.repeat(2, 1)
        else:
            waveform_stereo = waveform

        if sample_rate != self._demucs_model.samplerate:
            wav = torchaudio.functional.resample(
                waveform_stereo, sample_rate, self._demucs_model.samplerate
            )
        else:
            wav = waveform_stereo

        # Add batch dimension [B, C, T]
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)

        # Apply Demucs
        logger.info("Applying Demucs source separation...")
        with torch.no_grad():
            result = apply_model(
                self._demucs_model, wav, device=self.device, split=True, overlap=0.25
            )

        # Clear GPU memory
        if self.device.type != "cpu":
            torch.cuda.empty_cache()

        # Extract vocals
        vocals_idx = self._demucs_model.sources.index("vocals")
        vocals = result[0, vocals_idx].mean(0).cpu()  # Average channels to mono

        # Resample back to original sample rate
        vocals = torchaudio.functional.resample(
            vocals, self._demucs_model.samplerate, sample_rate
        )

        logger.info(f"Extracted vocals: {vocals.shape[0]/sample_rate:.2f}s")
        return vocals, sample_rate

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

        Requires: pip install noisereduce

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate
            stationary: If True, use stationary noise reduction (faster)
            prop_decrease: Proportion to reduce noise by (0-1)
            n_fft: FFT size

        Returns:
            Denoised waveform
        """
        if not _NOISEREDUCE_AVAILABLE:
            raise ImportError("noisereduce is required. Install: pip install noisereduce")

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        audio = waveform.numpy()

        logger.info(f"Applying noisereduce (stationary={stationary})...")
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

        Requires: pip install deepfilternet

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate

        Returns:
            enhanced_waveform: Enhanced audio
            sample_rate: Output sample rate
        """
        self._load_deepfilternet()

        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        logger.info("Applying DeepFilterNet...")

        # DeepFilterNet expects 48kHz
        target_sr = 48000
        if sample_rate != target_sr:
            waveform_48k = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        else:
            waveform_48k = waveform

        # Apply enhancement
        enhanced = df_enhance_fn(self._df_model, self._df_state, waveform_48k.unsqueeze(0))
        enhanced = enhanced.squeeze()

        # Resample back
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
        Apply Resemble Enhance for AI-powered speech enhancement.

        Requires: pip install resemble-enhance

        Args:
            waveform: Input waveform (1D tensor)
            sample_rate: Sample rate
            denoise_only: If True, only apply denoising (faster)
            solver: ODE solver for enhancement
            nfe: Number of function evaluations
            tau: Temperature parameter

        Returns:
            enhanced_waveform: Enhanced audio
            sample_rate: Output sample rate
        """
        if not _RESEMBLE_ENHANCE_AVAILABLE:
            raise ImportError("Resemble Enhance is required. Install: pip install resemble-enhance")

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
            enhanced, out_sr = resemble_denoise(waveform_44k, target_sr, self.device)
        else:
            enhanced, out_sr = resemble_enhance_fn(
                waveform_44k, target_sr, self.device,
                solver=solver, nfe=nfe, tau=tau,
            )

        # Resample back
        if out_sr != sample_rate:
            enhanced = torchaudio.functional.resample(enhanced, out_sr, sample_rate)

        logger.info("Resemble Enhance complete")
        return enhanced, sample_rate

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
        threshold = 10 ** (silence_threshold_db / 20)
        frame_size = int(sample_rate * 0.02)
        hop_size = int(sample_rate * 0.01)

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

        # Get silence intervals
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

        if waveform.dim() > 1:
            waveform = waveform.mean(0)
        if isinstance(waveform, torch.Tensor):
            audio = waveform.numpy()
        else:
            audio = waveform

        peak_normalized = pyln.normalize.peak(audio, peak_db)
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
        Full audio enhancement pipeline.

        Args:
            audio_or_path: Audio file path or waveform tensor
            sample_rate: Sample rate (required if audio_or_path is tensor)
            denoise_method: Denoising method. Options:
                - None: No denoising (default)
                - "demucs": Vocal extraction
                - "noisereduce": Spectral gating (fast)
                - "deepfilternet": DeepFilterNet (48kHz)
                - "resemble": Resemble Enhance (highest quality)
            extract_vocals: [DEPRECATED] Use denoise_method="demucs"
            remove_silence: Remove silence periods
            apply_vad: Apply Voice Activity Detection
            normalize: Apply loudness normalization

        Returns:
            EnhancementResult with enhanced waveform and timestamp mappings
        """
        # Handle deprecated extract_vocals parameter
        if extract_vocals and denoise_method is None:
            denoise_method = "demucs"

        # Load audio if path
        if isinstance(audio_or_path, (str, Path)):
            waveform, sample_rate = torchaudio.load(str(audio_or_path))
            if waveform.dim() > 1:
                waveform = waveform.mean(0)
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
        denoise_method: Denoising method (demucs, noisereduce, deepfilternet, resemble)
        remove_silence: Remove silence periods
        apply_vad: Apply Voice Activity Detection
        device: Device to use

    Returns:
        EnhancementResult with enhanced audio
    """
    enhancer = AudioEnhancement(device=device)
    return enhancer.enhance(
        audio_path,
        denoise_method=denoise_method,
        remove_silence=remove_silence,
        apply_vad=apply_vad,
    )


def denoise_noisereduce(
    audio_path: Union[str, Path],
    stationary: bool = False,
) -> Tuple[torch.Tensor, int]:
    """Quick noisereduce denoising."""
    if not _NOISEREDUCE_AVAILABLE:
        raise ImportError("noisereduce required. Install: pip install noisereduce")

    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(0)

    reduced = nr.reduce_noise(y=waveform.numpy(), sr=sr, stationary=stationary)
    return torch.from_numpy(reduced).float(), sr
