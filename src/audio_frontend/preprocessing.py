"""
Audio Preprocessing Utilities

Functions for audio preprocessing:
- Resampling
- Mono conversion
- Normalization
"""

from typing import Optional, List, Callable
import logging

import torch
import torchaudio

logger = logging.getLogger(__name__)


def resample(
    waveform: torch.Tensor,
    orig_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        waveform: Input waveform tensor
        orig_sample_rate: Original sample rate
        target_sample_rate: Target sample rate

    Returns:
        Resampled waveform tensor
    """
    if orig_sample_rate == target_sample_rate:
        return waveform

    logger.info(f"Resampling from {orig_sample_rate} Hz to {target_sample_rate} Hz")
    waveform = torchaudio.functional.resample(waveform, orig_sample_rate, target_sample_rate)

    return waveform


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
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


def normalize_peak(
    waveform: torch.Tensor,
    target_db: float = -3.0,
) -> torch.Tensor:
    """
    Apply peak normalization to audio.

    Args:
        waveform: Input waveform tensor
        target_db: Target peak level in dB (default -3.0 dB)

    Returns:
        Normalized waveform tensor
    """
    peak = waveform.abs().max()
    if peak > 0:
        target_peak = 10 ** (target_db / 20)
        waveform = waveform * (target_peak / peak)
        logger.info(f"Normalized audio to {target_db} dB peak")
    return waveform


def preprocess(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = False,
    normalize_db: float = -3.0,
    custom_preprocessors: Optional[List[Callable[[torch.Tensor, int], torch.Tensor]]] = None,
) -> torch.Tensor:
    """
    Apply all preprocessing steps to the waveform.

    Args:
        waveform: Input waveform tensor of shape (channels, num_samples)
        sample_rate: Sample rate of the waveform
        target_sample_rate: Target sample rate (default 16000)
        mono: If True, convert to mono
        normalize: If True, apply peak normalization
        normalize_db: Target peak level in dB
        custom_preprocessors: Optional list of custom preprocessor functions

    Returns:
        Preprocessed waveform tensor
    """
    # Resample if needed
    if sample_rate != target_sample_rate:
        waveform = resample(waveform, sample_rate, target_sample_rate)

    # Convert to mono if requested
    if mono:
        waveform = to_mono(waveform)

    # Apply normalization if requested
    if normalize:
        waveform = normalize_peak(waveform, normalize_db)

    # Apply custom preprocessors
    if custom_preprocessors:
        for preprocessor in custom_preprocessors:
            waveform = preprocessor(waveform, target_sample_rate)

    return waveform
