"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides:
- Shared fixtures for tests
- Dependency availability checks
- Sample data for testing

Note: This conftest avoids importing packages that require torch at module
level, so tests can be collected even without torch installed.
"""

import pytest
import sys
from pathlib import Path

# Add src to path (for direct module imports like alignment.base)
# Important: Insert at position 0 to take precedence, and ensure we don't
# accidentally import from the root __init__.py which tries to import torch
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Remove the root project path if it was added (to avoid root __init__.py imports)
root_path = str(Path(__file__).parent.parent)
if root_path in sys.path:
    sys.path.remove(root_path)


# =============================================================================
# Dependency availability markers
# =============================================================================

def check_k2_available():
    try:
        import k2
        return True
    except ImportError:
        return False


def check_lis_available():
    try:
        import lis
        return True
    except ImportError:
        return False


def check_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


K2_AVAILABLE = check_k2_available()
LIS_AVAILABLE = check_lis_available()
TORCH_AVAILABLE = check_torch_available()

# Pytest markers for skipping tests
requires_k2 = pytest.mark.skipif(not K2_AVAILABLE, reason="k2 not installed")
requires_lis = pytest.mark.skipif(not LIS_AVAILABLE, reason="lis not installed")
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
requires_k2_and_lis = pytest.mark.skipif(
    not (K2_AVAILABLE and LIS_AVAILABLE),
    reason="k2 and/or lis not installed"
)

# Skip entire test collection if torch not available
# This is because alignment.base imports torch at module level
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_torch: mark test as requiring torch"
    )
    config.addinivalue_line(
        "markers", "requires_k2: mark test as requiring k2"
    )
    config.addinivalue_line(
        "markers", "requires_lis: mark test as requiring lis"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require unavailable dependencies."""
    if not TORCH_AVAILABLE:
        skip_torch = pytest.mark.skip(reason="torch not installed - required for alignment.base")
        for item in items:
            # Skip all tests in unit/ since they import alignment.base
            if "unit" in str(item.fspath):
                item.add_marker(skip_torch)


# =============================================================================
# Basic fixtures
# =============================================================================

@pytest.fixture
def sample_text():
    """Simple sample text for testing."""
    return "hello world this is a test"


@pytest.fixture
def sample_text_normalized():
    """Normalized sample text (uppercase, no punctuation)."""
    return "HELLO WORLD THIS IS A TEST"


@pytest.fixture
def sample_words():
    """List of sample words."""
    return ["hello", "world", "this", "is", "a", "test"]


# =============================================================================
# AlignedWord fixtures (require torch)
# =============================================================================

@pytest.fixture
def aligned_word_simple():
    """Simple AlignedWord for basic tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignedWord
    return AlignedWord(
        word="hello",
        start_frame=100,
        end_frame=150,
    )


@pytest.fixture
def aligned_word_with_original():
    """AlignedWord with original form different from normalized."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignedWord
    return AlignedWord(
        word="hello",
        start_frame=100,
        end_frame=150,
        score=0.95,
        original="Hello!",
        index=0,
    )


@pytest.fixture
def aligned_word_with_score():
    """AlignedWord with confidence score."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignedWord
    return AlignedWord(
        word="world",
        start_frame=160,
        end_frame=220,
        score=0.88,
        index=1,
    )


@pytest.fixture
def aligned_words_list():
    """List of AlignedWords for result testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignedWord
    return [
        AlignedWord(word="hello", start_frame=100, end_frame=150, index=0),
        AlignedWord(word="world", start_frame=160, end_frame=220, index=1),
        AlignedWord(word="this", start_frame=230, end_frame=270, index=2),
        AlignedWord(word="is", start_frame=280, end_frame=300, index=3),
        AlignedWord(word="a", start_frame=310, end_frame=325, index=4),
        AlignedWord(word="test", start_frame=330, end_frame=400, index=5),
    ]


# =============================================================================
# AlignedChar fixtures (require torch)
# =============================================================================

@pytest.fixture
def aligned_char_simple():
    """Simple AlignedChar for basic tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignedChar
    return AlignedChar(
        char="h",
        start=2.0,
        end=2.1,
    )


@pytest.fixture
def aligned_chars_for_word():
    """List of AlignedChars for a word "hello"."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignedChar
    return [
        AlignedChar(char="h", start=2.0, end=2.1, score=0.9, word_index=0),
        AlignedChar(char="e", start=2.1, end=2.2, score=0.85, word_index=0),
        AlignedChar(char="l", start=2.2, end=2.3, score=0.88, word_index=0),
        AlignedChar(char="l", start=2.3, end=2.4, score=0.92, word_index=0),
        AlignedChar(char="o", start=2.4, end=2.5, score=0.87, word_index=0),
    ]


# =============================================================================
# AlignmentResult fixtures (require torch)
# =============================================================================

@pytest.fixture
def alignment_result_simple(aligned_words_list):
    """Simple AlignmentResult for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignmentResult
    return AlignmentResult(
        words=aligned_words_list,
        chars=[],
        unaligned_regions=[],
        metadata={"test": True},
    )


@pytest.fixture
def alignment_result_with_unaligned(aligned_words_list):
    """AlignmentResult with unaligned regions."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignmentResult
    return AlignmentResult(
        words=aligned_words_list,
        chars=[],
        unaligned_regions=[(6, 8), (15, 17)],
        metadata={"total_words": 20},
    )


# =============================================================================
# AlignmentConfig fixtures (require torch)
# =============================================================================

@pytest.fixture
def alignment_config_default():
    """Default AlignmentConfig."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignmentConfig
    return AlignmentConfig()


@pytest.fixture
def alignment_config_wfst():
    """WFST AlignmentConfig with custom parameters."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch required for alignment.base imports")
    from alignment.base import AlignmentConfig
    return AlignmentConfig(
        language="eng",
        segment_size=15.0,
        overlap=2.0,
        skip_penalty=-0.5,
        return_penalty=-18.0,
    )


# =============================================================================
# Audio fixtures (requires torch)
# =============================================================================

@pytest.fixture
def sample_waveform():
    """Sample waveform tensor for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    import torch
    # 3 seconds of silence at 16kHz
    return torch.zeros(1, 48000)


@pytest.fixture
def sample_waveform_1d():
    """Sample 1D waveform tensor."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not available")
    import torch
    return torch.zeros(48000)


# =============================================================================
# Constants
# =============================================================================

@pytest.fixture
def default_frame_duration():
    """Default frame duration (20ms)."""
    return 0.02


@pytest.fixture
def default_sample_rate():
    """Default sample rate (16kHz)."""
    return 16000
