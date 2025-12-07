"""
CTC Model Backends - Plugin-style architecture for different model sources.

This package provides backends for various CTC model sources:
- huggingface: HuggingFace Transformers (MMS, Wav2Vec2, etc.)
- torchaudio: TorchAudio pipelines (MMS_FA, etc.)
- nemo: NVIDIA NeMo (Conformer-CTC, QuartzNet, hybrid RNN-T/CTC)
- omniasr: Facebook OmniASR (1600+ languages)
- espnet: ESPnet (planned)

Backends are auto-discovered and registered on import.
Each backend is in its own module to allow optional dependencies.

Usage:
    from labeling_utils import get_backend, list_backends

    # List available backends
    print(list_backends())  # ['huggingface', 'torchaudio']

    # Get a backend class
    BackendClass = get_backend("huggingface")

    # Or use load_model() for simpler interface
    from labeling_utils import load_model
    backend = load_model("facebook/mms-1b-all", language="eng")
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Import backends that should always be available
# Each backend registers itself via the @backend decorator
_AVAILABLE_BACKENDS: List[str] = []


def _try_import_backend(module_name: str, backend_name: str) -> bool:
    """Try to import a backend module. Returns True if successful."""
    try:
        __import__(f"labeling_utils.backends.{module_name}", fromlist=[module_name])
        _AVAILABLE_BACKENDS.append(backend_name)
        logger.debug(f"Loaded backend: {backend_name}")
        return True
    except ImportError as e:
        logger.debug(f"Backend {backend_name} not available: {e}")
        return False


# Auto-discover backends
# Order matters - first successful import wins for aliases
_try_import_backend("huggingface", "huggingface")
_try_import_backend("torchaudio_backend", "torchaudio")
_try_import_backend("nemo_backend", "nemo")
_try_import_backend("omniasr_backend", "omniasr")

# Future backends (will be skipped if dependencies not available)
# _try_import_backend("espnet", "espnet")


def get_available_backends() -> List[str]:
    """Get list of successfully loaded backends."""
    return list(_AVAILABLE_BACKENDS)


# Re-export commonly used classes for convenience
# (Only if the backends are available)
try:
    from .huggingface import HuggingFaceCTCBackend
except ImportError:
    HuggingFaceCTCBackend = None

try:
    from .torchaudio_backend import TorchAudioPipelineBackend
except ImportError:
    TorchAudioPipelineBackend = None

try:
    from .nemo_backend import NeMoCTCBackend
except ImportError:
    NeMoCTCBackend = None

try:
    from .omniasr_backend import OmniASRBackend
except ImportError:
    OmniASRBackend = None


__all__ = [
    "get_available_backends",
    "HuggingFaceCTCBackend",
    "TorchAudioPipelineBackend",
    "NeMoCTCBackend",
    "OmniASRBackend",
]
