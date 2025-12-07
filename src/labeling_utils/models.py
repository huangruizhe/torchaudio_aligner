"""
Model loading utilities.

High-level functions for loading CTC models with automatic backend detection.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

import torch

from .backends import (
    CTCModelBackend,
    BackendConfig,
    HuggingFaceCTCBackend,
    get_backend,
    list_backends,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for loading a CTC model.

    Attributes:
        model_name: Model identifier (HuggingFace model ID or path)
        backend: Backend to use ("huggingface", "torchaudio", etc.)
        language: Target language (ISO 639-3 code for MMS models)
        device: Device to load model on ("cuda", "cpu", "mps")
        dtype: Model dtype (torch.float32, torch.float16)
        with_star: Whether to include <star>/<unk> token dimension
        cache_dir: Directory to cache model files
    """
    model_name: str
    backend: str = "huggingface"
    language: Optional[str] = None
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32
    with_star: bool = True
    cache_dir: Optional[str] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# Pre-defined model configurations for common models
_MODEL_PRESETS: Dict[str, ModelConfig] = {
    # MMS models (HuggingFace)
    "mms": ModelConfig(
        model_name="facebook/mms-1b-all",
        backend="huggingface",
    ),
    "mms-1b-all": ModelConfig(
        model_name="facebook/mms-1b-all",
        backend="huggingface",
    ),
    "mms-1b-fl102": ModelConfig(
        model_name="facebook/mms-1b-fl102",
        backend="huggingface",
    ),
    "mms-300m": ModelConfig(
        model_name="facebook/mms-300m",
        backend="huggingface",
    ),
    # Wav2Vec2 models
    "wav2vec2-base": ModelConfig(
        model_name="facebook/wav2vec2-base-960h",
        backend="huggingface",
    ),
    "wav2vec2-large": ModelConfig(
        model_name="facebook/wav2vec2-large-960h-lv60-self",
        backend="huggingface",
    ),
    # Forced alignment specific
    "mms-fa": ModelConfig(
        model_name="MahmoudAshraf/mms-300m-1130-forced-aligner",
        backend="huggingface",
    ),
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model preset.

    Args:
        model_name: Preset name or HuggingFace model ID

    Returns:
        Dictionary with model information
    """
    if model_name in _MODEL_PRESETS:
        config = _MODEL_PRESETS[model_name]
        return {
            "preset": model_name,
            "model_name": config.model_name,
            "backend": config.backend,
            "languages": "1100+" if "mms" in model_name.lower() else "English",
        }
    else:
        return {
            "preset": None,
            "model_name": model_name,
            "backend": "huggingface",
            "languages": "Unknown",
        }


def list_presets() -> List[str]:
    """List available model presets."""
    return list(_MODEL_PRESETS.keys())


def load_model(
    model_name: str,
    language: Optional[str] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    backend: Optional[str] = None,
    with_star: bool = True,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> CTCModelBackend:
    """
    Load a CTC model for emission extraction.

    This is the main entry point for loading models. It automatically
    selects the appropriate backend based on the model name.

    Args:
        model_name: Model preset name or HuggingFace model ID
            - Presets: "mms", "mms-1b-all", "wav2vec2-base", etc.
            - HuggingFace: "facebook/mms-1b-all", "facebook/wav2vec2-base-960h"
        language: Target language code (ISO 639-3 for MMS)
            - Examples: "eng", "fra", "cmn", "jpn", "hin"
        device: Device to load model on ("cuda", "cpu", "mps")
        dtype: Model dtype (torch.float32, torch.float16)
        backend: Force a specific backend ("huggingface", etc.)
        with_star: Include <star>/<unk> token in emissions
        cache_dir: Directory to cache downloaded models
        **kwargs: Additional backend-specific options

    Returns:
        Loaded CTCModelBackend ready for emission extraction

    Example:
        >>> # Load MMS for English
        >>> backend = load_model("mms", language="eng")
        >>>
        >>> # Load specific HuggingFace model
        >>> backend = load_model("facebook/mms-1b-all", language="fra")
        >>>
        >>> # Load on GPU with float16
        >>> backend = load_model("mms", language="eng", device="cuda", dtype=torch.float16)
    """
    # Check for preset
    if model_name in _MODEL_PRESETS:
        preset = _MODEL_PRESETS[model_name]
        actual_model_name = preset.model_name
        actual_backend = backend or preset.backend
    else:
        actual_model_name = model_name
        actual_backend = backend or "huggingface"

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build config
    config = BackendConfig(
        model_name=actual_model_name,
        language=language,
        device=device,
        dtype=dtype,
        with_star=with_star,
        cache_dir=cache_dir,
        extra_options=kwargs,
    )

    # Get backend class
    backend_class = get_backend(actual_backend)

    # Create and load
    logger.info(f"Loading model: {actual_model_name} with backend: {actual_backend}")
    model_backend = backend_class(config)
    model_backend.load()

    return model_backend


def load_from_config(config: ModelConfig) -> CTCModelBackend:
    """
    Load a model from a ModelConfig object.

    Args:
        config: ModelConfig with all settings

    Returns:
        Loaded CTCModelBackend
    """
    return load_model(
        model_name=config.model_name,
        language=config.language,
        device=config.device,
        dtype=config.dtype,
        backend=config.backend,
        with_star=config.with_star,
        cache_dir=config.cache_dir,
        **config.extra_options,
    )
