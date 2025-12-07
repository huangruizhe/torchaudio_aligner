"""
Labeling Utils - Frame-wise posterior extraction from CTC models.

This module provides an extensible framework for extracting frame-wise
posteriors (emissions) from various CTC acoustic models.

Supported backends:
- HuggingFace Transformers (recommended): MMS, Wav2Vec2, etc.
- TorchAudio pipelines: MMS_FA, WAV2VEC2_ASR_*, etc.
- NeMo (planned): Conformer-CTC, QuartzNet
- ESPnet (planned): ESPnet2 CTC models
- OmniASR (planned): Universal ASR models

Usage:
    from labeling_utils import get_emissions, load_model

    # Load model (auto-detects backend)
    model = load_model("facebook/mms-1b-all", language="eng")

    # Get frame-wise posteriors
    result = get_emissions(model, waveform, sample_rate=16000)
    print(result.emissions.shape)  # (frames, vocab_size)

    # Or use presets
    model = load_model("mms", language="fra")  # French MMS

Adding custom backends:
    from labeling_utils import CTCModelBackend, register_backend

    class MyBackend(CTCModelBackend):
        def load(self): ...
        def get_emissions(self, waveform, lengths=None): ...
        def get_vocab_info(self): ...

    register_backend("mybackend", MyBackend, aliases=["mb"])
"""

# Core abstractions
from .base import (
    CTCModelBackend,
    VocabInfo,
    BackendConfig,
)

# Registry functions
from .registry import (
    get_backend,
    list_backends,
    register_backend,
    register_backend_lazy,
    is_backend_available,
    backend,  # Decorator
)

# Emission extraction
from .emissions import (
    get_emissions,
    get_emissions_batched,
    EmissionResult,
)

# Model loading
from .models import (
    load_model,
    load_from_config,
    ModelConfig,
    get_model_info,
    list_presets,
)

# Import backends to trigger registration
from . import backends

# Re-export backend classes for convenience (if available)
from .backends import (
    HuggingFaceCTCBackend,
    TorchAudioPipelineBackend,
)

__all__ = [
    # Core abstractions
    "CTCModelBackend",
    "VocabInfo",
    "BackendConfig",
    # Registry
    "get_backend",
    "list_backends",
    "register_backend",
    "register_backend_lazy",
    "is_backend_available",
    "backend",
    # Emissions
    "get_emissions",
    "get_emissions_batched",
    "EmissionResult",
    # Models
    "load_model",
    "load_from_config",
    "ModelConfig",
    "get_model_info",
    "list_presets",
    # Backend classes
    "HuggingFaceCTCBackend",
    "TorchAudioPipelineBackend",
]

__version__ = "0.2.0"
