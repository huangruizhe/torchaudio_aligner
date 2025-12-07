"""
Labeling Utils - Frame-wise posterior extraction from CTC models.

This module provides an extensible framework for extracting frame-wise
posteriors (emissions) from various CTC acoustic models.

Supported backends:
- HuggingFace Transformers (recommended): MMS, Wav2Vec2, etc.
- TorchAudio pipelines (legacy/deprecated)
- NeMo (planned)
- ESPnet (planned)

Usage:
    from labeling_utils import get_emissions, load_model

    # Load model (auto-detects backend)
    model = load_model("facebook/mms-1b-all", language="eng")

    # Get frame-wise posteriors
    emissions = get_emissions(model, waveform, sample_rate=16000)
"""

from .backends import (
    CTCModelBackend,
    HuggingFaceCTCBackend,
    TorchAudioPipelineBackend,
    VocabInfo,
    BackendConfig,
    get_backend,
    list_backends,
    register_backend,
)

from .emissions import (
    get_emissions,
    get_emissions_batched,
    EmissionResult,
)

from .models import (
    load_model,
    ModelConfig,
    get_model_info,
)

__all__ = [
    # Backends
    "CTCModelBackend",
    "HuggingFaceCTCBackend",
    "TorchAudioPipelineBackend",
    "VocabInfo",
    "BackendConfig",
    "get_backend",
    "list_backends",
    "register_backend",
    # Emissions
    "get_emissions",
    "get_emissions_batched",
    "EmissionResult",
    # Models
    "load_model",
    "ModelConfig",
    "get_model_info",
]
