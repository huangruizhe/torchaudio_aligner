"""
Audio Loading Utilities

Functions for loading audio from various sources:
- torchaudio (primary backend)
- soundfile (fallback backend)
"""

from pathlib import Path
from typing import Tuple, List, Literal, Union
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


AudioBackend = Literal["torchaudio", "soundfile", "auto"]


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


def load_audio(
    audio_path: Union[str, Path],
    backend: AudioBackend = "auto",
) -> Tuple[torch.Tensor, int]:
    """
    Load audio from file.

    Args:
        audio_path: Path to audio file (supports mp3, wav, flac, etc.)
        backend: Audio loading backend ("auto", "torchaudio", "soundfile")

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

    if backend == "soundfile":
        waveform, sample_rate = _load_with_soundfile(audio_path_str)
        logger.info("Loaded with backend 'soundfile'")
    elif backend == "torchaudio":
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
