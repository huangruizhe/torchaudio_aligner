"""
Audacity label export utilities.

Generates label files that can be imported into Audacity for visualization.
See: https://manual.audacityteam.org/man/importing_and_exporting_labels.html

Format: start_time<TAB>end_time<TAB>label
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path


def get_audacity_labels(
    word_alignment: Dict[int, Any],
    frame_duration: float = 0.02,
    include_unaligned: bool = False,
) -> str:
    """
    Generate Audacity label file content from word alignments.

    Args:
        word_alignment: Dict mapping word index to AlignedWord
        frame_duration: Duration of each frame in seconds (default 20ms)
        include_unaligned: Whether to include unaligned word indices

    Returns:
        String content for Audacity label file

    Example:
        >>> labels = get_audacity_labels(result.word_alignments)
        >>> with open("output.txt", "w") as f:
        ...     f.write(labels)
    """
    lines = []
    sorted_items = sorted(word_alignment.items())

    for i, (word_idx, word) in enumerate(sorted_items):
        # Get start time
        if hasattr(word.start_time, 'item'):
            start_sec = word.start_time.item() * frame_duration
        else:
            start_sec = word.start_time * frame_duration

        # Get end time
        if word.end_time is not None:
            if hasattr(word.end_time, 'item'):
                end_sec = word.end_time.item() * frame_duration
            else:
                end_sec = word.end_time * frame_duration
        elif i + 1 < len(sorted_items):
            # Use next word's start as end
            next_word = sorted_items[i + 1][1]
            if hasattr(next_word.start_time, 'item'):
                end_sec = next_word.start_time.item() * frame_duration
            else:
                end_sec = next_word.start_time * frame_duration
        else:
            # Last word - use default duration
            end_sec = start_sec + 0.5

        # Get word text
        label = word.word if word.word else f"[{word_idx}]"

        # Format: start<TAB>end<TAB>label
        lines.append(f"{start_sec:.6f}\t{end_sec:.6f}\t{label}")

    return "\n".join(lines)


def save_audacity_labels(
    word_alignment: Dict[int, Any],
    output_path: Union[str, Path],
    frame_duration: float = 0.02,
) -> str:
    """
    Save word alignments as Audacity label file.

    Args:
        word_alignment: Dict mapping word index to AlignedWord
        output_path: Path to save the label file
        frame_duration: Duration of each frame in seconds

    Returns:
        Path to the saved file

    Example:
        >>> path = save_audacity_labels(result.word_alignments, "output.txt")
        >>> print(f"Labels saved to: {path}")
    """
    output_path = Path(output_path)
    labels = get_audacity_labels(word_alignment, frame_duration)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(labels)

    return str(output_path)
