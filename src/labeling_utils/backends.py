"""
CTC Model Backends - Backwards compatibility layer.

This module re-exports classes from the new modular structure for
backwards compatibility with existing code.

New code should import from:
- labeling_utils.base: CTCModelBackend, VocabInfo, BackendConfig
- labeling_utils.registry: get_backend, list_backends, register_backend
- labeling_utils.backends.huggingface: HuggingFaceCTCBackend
- labeling_utils.backends.torchaudio_backend: TorchAudioPipelineBackend

Or simply from labeling_utils directly:
    from labeling_utils import load_model, get_emissions
"""

# Re-export from base
from .base import (
    CTCModelBackend,
    VocabInfo,
    BackendConfig,
)

# Re-export from registry
from .registry import (
    get_backend,
    list_backends,
    register_backend,
    register_backend_lazy,
    is_backend_available,
)

# Re-export backend classes (if available)
try:
    from .backends.huggingface import HuggingFaceCTCBackend
except ImportError:
    HuggingFaceCTCBackend = None

try:
    from .backends.torchaudio_backend import TorchAudioPipelineBackend
except ImportError:
    TorchAudioPipelineBackend = None

__all__ = [
    # Base classes
    "CTCModelBackend",
    "VocabInfo",
    "BackendConfig",
    # Registry
    "get_backend",
    "list_backends",
    "register_backend",
    "register_backend_lazy",
    "is_backend_available",
    # Backend implementations
    "HuggingFaceCTCBackend",
    "TorchAudioPipelineBackend",
]
