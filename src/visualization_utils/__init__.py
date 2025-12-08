"""
Visualization utilities for alignment results.

This module provides tools for:
- Audacity label export
- Gentle-style HTML visualization
- Audio preview functions
- Alignment inspection

Usage:
    from visualization_utils import (
        get_audacity_labels,
        get_gentle_visualization,
        preview_word,
        preview_all_words,
    )
"""

from .audacity import (
    get_audacity_labels,
    save_audacity_labels,
)

from .gentle import (
    get_gentle_visualization,
    save_gentle_html,
)

from .audio_preview import (
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
    # Audio preview
    "preview_word",
    "preview_word_by_index",
    "preview_segment",
    "preview_all_words",
    "preview_random_segment",
]
