"""
Visualization utilities for alignment results.

This module provides tools for:
- Audacity label export
- Gentle-style HTML visualization
- Audio preview functions
- Alignment inspection

Usage:
    from visualization_utils import (
        # Simple API (recommended)
        play_word,
        play_segment,
        play_random,
        play_words_sequential,
        # Export
        get_audacity_labels,
        get_gentle_visualization,
    )
"""

# Suppress pydub SyntaxWarning about invalid escape sequences in Python 3.12+
# This is a known issue in pydub: https://github.com/jiaaro/pydub/issues/755
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

from .audacity import (
    get_audacity_labels,
    save_audacity_labels,
)

from .gentle import (
    get_gentle_visualization,
    save_gentle_html,
    create_interactive_demo,
)

from .audio_preview import (
    # Simple API (recommended - uses audio file directly)
    play_word,
    play_segment,
    play_random,
    play_words_sequential,
    # Waveform API (seconds-based)
    preview_word_seconds,
    preview_segment_seconds,
    preview_random_segment_seconds,
    # Legacy API (frame-based)
    preview_word,
    preview_word_by_index,
    preview_segment,
    preview_all_words,
    preview_random_segment,
)

__all__ = [
    # Audacity
    "get_audacity_labels",
    "save_audacity_labels",
    # Gentle
    "get_gentle_visualization",
    "save_gentle_html",
    "create_interactive_demo",
    # Audio preview - Simple API (recommended)
    "play_word",
    "play_segment",
    "play_random",
    "play_words_sequential",
    # Audio preview - Waveform API
    "preview_word_seconds",
    "preview_segment_seconds",
    "preview_random_segment_seconds",
    # Audio preview - Legacy API
    "preview_word",
    "preview_word_by_index",
    "preview_segment",
    "preview_all_words",
    "preview_random_segment",
]
