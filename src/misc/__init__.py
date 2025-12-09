"""
Miscellaneous utilities for torchaudio_aligner.

This package contains:
- install_utils: Functions to install k2 and other dependencies
"""

from .install_utils import (
    install_k2_if_needed,
    install_other_deps,
    install_ocr_deps,
    install_romanization_deps,
    install_all,
    check_k2_installed,
    get_k2_version,
    get_torch_info,
)

__all__ = [
    "install_k2_if_needed",
    "install_other_deps",
    "install_ocr_deps",
    "install_romanization_deps",
    "install_all",
    "check_k2_installed",
    "get_k2_version",
    "get_torch_info",
]
