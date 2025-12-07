"""
Model loading utilities.

High-level functions for loading CTC models with automatic backend detection.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

import torch

from .base import CTCModelBackend, BackendConfig
from .registry import get_backend

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for loading a CTC model.

    Attributes:
        model_name: Model identifier (HuggingFace model ID, path, or pipeline name)
        backend: Backend to use ("huggingface", "torchaudio", etc.)
        language: Target language (ISO 639-3 code for MMS models)
        device: Device to load model on ("cuda", "cpu", "mps")
        dtype: Model dtype (torch.float32, torch.float16, torch.bfloat16)
        with_star: Whether to include <star>/<unk> token dimension
        cache_dir: Directory to cache model files
        extra_options: Additional backend-specific options
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
    # ==========================================================================
    # MMS Models (HuggingFace) - Massively Multilingual Speech
    # ==========================================================================
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

    # ==========================================================================
    # MMS Forced Alignment Models
    # ==========================================================================
    # TorchAudio pipeline version
    "mms-fa": ModelConfig(
        model_name="MMS_FA",
        backend="torchaudio",
        with_star=True,
    ),
    "mms-fa-torchaudio": ModelConfig(
        model_name="MMS_FA",
        backend="torchaudio",
        with_star=True,
    ),
    # HuggingFace version (community model)
    "mms-fa-hf": ModelConfig(
        model_name="MahmoudAshraf/mms-300m-1130-forced-aligner",
        backend="huggingface",
    ),

    # ==========================================================================
    # Wav2Vec2 Models (HuggingFace)
    # ==========================================================================
    "wav2vec2-base": ModelConfig(
        model_name="facebook/wav2vec2-base-960h",
        backend="huggingface",
    ),
    "wav2vec2-large": ModelConfig(
        model_name="facebook/wav2vec2-large-960h-lv60-self",
        backend="huggingface",
    ),
    "wav2vec2-large-lv60": ModelConfig(
        model_name="facebook/wav2vec2-large-960h-lv60-self",
        backend="huggingface",
    ),
    "wav2vec2-xlsr": ModelConfig(
        model_name="facebook/wav2vec2-large-xlsr-53",
        backend="huggingface",
    ),

    # ==========================================================================
    # Wav2Vec2 Models (TorchAudio Pipelines)
    # ==========================================================================
    "wav2vec2-base-ta": ModelConfig(
        model_name="WAV2VEC2_ASR_BASE_960H",
        backend="torchaudio",
    ),
    "wav2vec2-large-ta": ModelConfig(
        model_name="WAV2VEC2_ASR_LARGE_960H",
        backend="torchaudio",
    ),
    "wav2vec2-large-lv60k-ta": ModelConfig(
        model_name="WAV2VEC2_ASR_LARGE_LV60K_960H",
        backend="torchaudio",
    ),

    # ==========================================================================
    # HuBERT Models (TorchAudio Pipelines)
    # ==========================================================================
    "hubert-large": ModelConfig(
        model_name="HUBERT_ASR_LARGE",
        backend="torchaudio",
    ),
    "hubert-xlarge": ModelConfig(
        model_name="HUBERT_ASR_XLARGE",
        backend="torchaudio",
    ),

    # ==========================================================================
    # NeMo Models (NVIDIA)
    # ==========================================================================
    "nemo-conformer": ModelConfig(
        model_name="nvidia/stt_en_conformer_ctc_large",
        backend="nemo",
    ),
    "nemo-conformer-large": ModelConfig(
        model_name="nvidia/stt_en_conformer_ctc_large",
        backend="nemo",
    ),
    "nemo-conformer-small": ModelConfig(
        model_name="nvidia/stt_en_conformer_ctc_small",
        backend="nemo",
    ),
    "nemo-fastconformer": ModelConfig(
        model_name="nvidia/stt_en_fastconformer_hybrid_large_pc",
        backend="nemo",
    ),
    "nemo-quartznet": ModelConfig(
        model_name="nvidia/stt_en_quartznet15x5",
        backend="nemo",
    ),
    "nemo-citrinet": ModelConfig(
        model_name="nvidia/stt_en_citrinet_1024",
        backend="nemo",
    ),

    # ==========================================================================
    # OmniASR Models (Facebook/Meta - 1600+ languages)
    # ==========================================================================
    "omniasr": ModelConfig(
        model_name="omniASR_CTC_1B",
        backend="omniasr",
    ),
    "omniasr-300m": ModelConfig(
        model_name="omniASR_CTC_300M",
        backend="omniasr",
    ),
    "omniasr-1b": ModelConfig(
        model_name="omniASR_CTC_1B",
        backend="omniasr",
    ),
    "omniasr-3b": ModelConfig(
        model_name="omniASR_CTC_3B",
        backend="omniasr",
    ),
    "omniasr-7b": ModelConfig(
        model_name="omniASR_CTC_7B",
        backend="omniasr",
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

        # Determine language support
        if "omniasr" in model_name.lower():
            languages = "1600+"
        elif "mms" in model_name.lower():
            if "1b-all" in config.model_name.lower() or "mms-1b-all" in config.model_name.lower():
                languages = "1100+"
            elif "fl102" in config.model_name.lower():
                languages = "102"
            elif "fa" in model_name.lower():
                languages = "1130+"
            else:
                languages = "Multiple"
        elif "xlsr" in model_name.lower():
            languages = "53"
        elif "nemo" in model_name.lower():
            languages = "English"  # NeMo models are typically English
        else:
            languages = "English"

        return {
            "preset": model_name,
            "model_name": config.model_name,
            "backend": config.backend,
            "languages": languages,
            "with_star": config.with_star,
        }
    else:
        # Auto-detect backend for unknown models
        backend = "huggingface"
        if model_name.upper() in ["MMS_FA", "WAV2VEC2_ASR_BASE_960H", "WAV2VEC2_ASR_LARGE_960H"]:
            backend = "torchaudio"

        return {
            "preset": None,
            "model_name": model_name,
            "backend": backend,
            "languages": "Unknown",
        }


def list_presets() -> List[str]:
    """List available model presets."""
    return list(_MODEL_PRESETS.keys())


def get_preset_by_category() -> Dict[str, List[str]]:
    """Get presets organized by category."""
    categories = {
        "MMS (HuggingFace)": [],
        "MMS Forced Alignment": [],
        "Wav2Vec2 (HuggingFace)": [],
        "Wav2Vec2 (TorchAudio)": [],
        "HuBERT (TorchAudio)": [],
        "NeMo (NVIDIA)": [],
        "OmniASR (Meta)": [],
    }

    for name, config in _MODEL_PRESETS.items():
        if "mms-fa" in name:
            categories["MMS Forced Alignment"].append(name)
        elif "mms" in name:
            categories["MMS (HuggingFace)"].append(name)
        elif "wav2vec2" in name and config.backend == "torchaudio":
            categories["Wav2Vec2 (TorchAudio)"].append(name)
        elif "wav2vec2" in name:
            categories["Wav2Vec2 (HuggingFace)"].append(name)
        elif "hubert" in name:
            categories["HuBERT (TorchAudio)"].append(name)
        elif "nemo" in name:
            categories["NeMo (NVIDIA)"].append(name)
        elif "omniasr" in name:
            categories["OmniASR (Meta)"].append(name)

    return categories


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
        model_name: Model preset name or model identifier
            - Presets: "mms", "mms-1b-all", "wav2vec2-base", "mms-fa", etc.
            - HuggingFace: "facebook/mms-1b-all", "facebook/wav2vec2-base-960h"
            - TorchAudio: "MMS_FA", "WAV2VEC2_ASR_BASE_960H"
        language: Target language code (ISO 639-3 for MMS)
            - Examples: "eng", "fra", "cmn", "jpn", "hin"
        device: Device to load model on ("cuda", "cpu", "mps")
        dtype: Model dtype (torch.float32, torch.float16, torch.bfloat16)
        backend: Force a specific backend ("huggingface", "torchaudio", etc.)
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
        >>> # Load MMS_FA via TorchAudio
        >>> backend = load_model("mms-fa")
        >>>
        >>> # Load on GPU with float16
        >>> backend = load_model("mms", language="eng", device="cuda", dtype=torch.float16)
    """
    # Check for preset
    if model_name in _MODEL_PRESETS:
        preset = _MODEL_PRESETS[model_name]
        actual_model_name = preset.model_name
        actual_backend = backend or preset.backend
        # Use preset's with_star if not explicitly overridden
        if "with_star" not in kwargs:
            with_star = preset.with_star
    else:
        actual_model_name = model_name
        # Auto-detect backend based on model name patterns
        if backend is None:
            if model_name.upper() in ["MMS_FA", "WAV2VEC2_ASR_BASE_960H",
                                       "WAV2VEC2_ASR_LARGE_960H", "WAV2VEC2_ASR_LARGE_LV60K_960H",
                                       "HUBERT_ASR_LARGE", "HUBERT_ASR_XLARGE"]:
                actual_backend = "torchaudio"
            elif model_name.startswith("nvidia/") or model_name.startswith("stt_"):
                actual_backend = "nemo"
            elif model_name.startswith("omniASR"):
                actual_backend = "omniasr"
            else:
                actual_backend = "huggingface"
        else:
            actual_backend = backend

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
