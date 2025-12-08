"""
Shared test utilities.

This module provides shared constants and markers that can be imported
by test files. Unlike conftest.py, this can be imported directly.
"""

import pytest


def check_torch_available():
    """Check if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def check_k2_available():
    """Check if k2 is available."""
    try:
        import k2
        return True
    except ImportError:
        return False


def check_lis_available():
    """Check if lis is available."""
    try:
        import lis
        return True
    except ImportError:
        return False


# Availability flags
TORCH_AVAILABLE = check_torch_available()
K2_AVAILABLE = check_k2_available()
LIS_AVAILABLE = check_lis_available()

# Pytest markers
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
requires_k2 = pytest.mark.skipif(not K2_AVAILABLE, reason="k2 not installed")
requires_lis = pytest.mark.skipif(not LIS_AVAILABLE, reason="lis not installed")
requires_k2_and_lis = pytest.mark.skipif(
    not (K2_AVAILABLE and LIS_AVAILABLE),
    reason="k2 and/or lis not installed"
)
