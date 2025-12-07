"""
Emission extraction utilities.

High-level functions for extracting frame-wise posteriors from audio.
"""

from dataclasses import dataclass
from typing import Optional, List, Union, Any
import logging

import torch

from .backends import CTCModelBackend, VocabInfo

logger = logging.getLogger(__name__)


@dataclass
class EmissionResult:
    """
    Result of emission extraction.

    Attributes:
        emissions: Log posteriors of shape (frames, vocab_size) or (batch, frames, vocab_size)
        lengths: Number of valid frames per batch item
        vocab_info: Vocabulary information
        frame_duration: Duration of each frame in seconds
        sample_rate: Original audio sample rate
    """
    emissions: torch.Tensor
    lengths: torch.Tensor
    vocab_info: VocabInfo
    frame_duration: float = 0.02
    sample_rate: int = 16000

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        if self.emissions.dim() == 3:
            return self.emissions.shape[1]
        return self.emissions.shape[0]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.emissions.shape[-1]

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.num_frames * self.frame_duration

    def get_frame_timestamps(self) -> torch.Tensor:
        """Get timestamp (in seconds) for each frame."""
        return torch.arange(self.num_frames) * self.frame_duration

    def to(self, device: str) -> "EmissionResult":
        """Move tensors to device."""
        return EmissionResult(
            emissions=self.emissions.to(device),
            lengths=self.lengths.to(device),
            vocab_info=self.vocab_info,
            frame_duration=self.frame_duration,
            sample_rate=self.sample_rate,
        )


def get_emissions(
    backend: CTCModelBackend,
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    normalize: bool = True,
) -> EmissionResult:
    """
    Extract frame-wise log posteriors from audio.

    This is the main entry point for emission extraction.

    Args:
        backend: Loaded CTC model backend
        waveform: Audio tensor of shape (samples,) or (channels, samples)
        sample_rate: Input audio sample rate (will resample if needed)
        normalize: Whether to apply log_softmax (usually already done)

    Returns:
        EmissionResult containing emissions and metadata

    Example:
        >>> from labeling_utils import load_model, get_emissions
        >>> backend = load_model("facebook/mms-1b-all", language="eng")
        >>> result = get_emissions(backend, waveform)
        >>> print(result.emissions.shape)  # (frames, vocab_size)
    """
    if not backend.is_loaded:
        raise RuntimeError(f"Backend {backend.name} not loaded. Call load() first.")

    # Handle multi-channel audio -> mono
    if waveform.dim() == 2:
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

    # Resample if needed
    if sample_rate != backend.sample_rate:
        try:
            import torchaudio.functional as F
            waveform = F.resample(waveform, sample_rate, backend.sample_rate)
            logger.debug(f"Resampled from {sample_rate}Hz to {backend.sample_rate}Hz")
        except ImportError:
            logger.warning(
                f"torchaudio not available for resampling. "
                f"Expected {backend.sample_rate}Hz, got {sample_rate}Hz"
            )

    # Get emissions from backend
    emissions, lengths = backend.get_emissions(waveform.unsqueeze(0))

    # Remove batch dimension for single input
    emissions = emissions.squeeze(0)
    lengths = lengths.squeeze(0) if lengths.dim() > 0 else lengths

    return EmissionResult(
        emissions=emissions,
        lengths=lengths,
        vocab_info=backend.get_vocab_info(),
        frame_duration=backend.frame_duration,
        sample_rate=backend.sample_rate,
    )


def get_emissions_batched(
    backend: CTCModelBackend,
    waveforms: List[torch.Tensor],
    sample_rate: int = 16000,
    batch_size: int = 8,
) -> List[EmissionResult]:
    """
    Extract emissions from multiple audio files efficiently.

    Args:
        backend: Loaded CTC model backend
        waveforms: List of audio tensors
        sample_rate: Input audio sample rate
        batch_size: Batch size for inference

    Returns:
        List of EmissionResult, one per input audio

    Example:
        >>> results = get_emissions_batched(backend, [wav1, wav2, wav3])
        >>> for result in results:
        ...     print(result.emissions.shape)
    """
    if not backend.is_loaded:
        raise RuntimeError(f"Backend {backend.name} not loaded. Call load() first.")

    # Preprocess: mono + resample
    processed = []
    for waveform in waveforms:
        if waveform.dim() == 2:
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

        if sample_rate != backend.sample_rate:
            try:
                import torchaudio.functional as F
                waveform = F.resample(waveform, sample_rate, backend.sample_rate)
            except ImportError:
                pass

        processed.append(waveform)

    # Get emissions
    emissions_list = backend.get_emissions_batched(processed, batch_size)

    # Wrap in EmissionResult
    vocab_info = backend.get_vocab_info()
    results = []
    for emissions in emissions_list:
        results.append(EmissionResult(
            emissions=emissions,
            lengths=torch.tensor(emissions.shape[0]),
            vocab_info=vocab_info,
            frame_duration=backend.frame_duration,
            sample_rate=backend.sample_rate,
        ))

    return results
