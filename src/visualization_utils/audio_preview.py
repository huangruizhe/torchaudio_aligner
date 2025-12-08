"""
Audio preview utilities for alignment verification.

Provides functions to play audio segments corresponding to aligned words.

NEW API (recommended):
    # With seconds-based AlignmentResult
    result = align_long_audio(audio, text)
    preview_word_seconds(waveform, result.words[10], sample_rate)
    preview_segment_seconds(waveform, result.words[100:120], sample_rate)

LEGACY API (for backwards compatibility):
    # With frame-based word_alignment dict
    preview_word(waveform, word_alignment, word_idx, frame_duration=0.02)
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import random


# =============================================================================
# NEW API: Works with seconds-based AlignedWord objects
# =============================================================================

def preview_word_seconds(
    waveform,
    word,  # AlignedWord with start_frame/end_frame
    sample_rate: int = 16000,
    padding: float = 0.0,  # padding in seconds
):
    """
    Preview audio for an aligned word.

    Args:
        waveform: Audio waveform tensor (1D or 2D)
        word: AlignedWord object (call start_seconds()/end_seconds() for times)
        sample_rate: Audio sample rate
        padding: Extra seconds to include before/after

    Returns:
        IPython.display.Audio object for playback

    Example:
        >>> result = align_long_audio(audio, text)
        >>> preview_word_seconds(waveform, result.words[10])
    """
    from IPython.display import Audio
    import torch

    start_sec = max(0, word.start_seconds() - padding)
    end_sec = word.end_seconds() + padding

    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # Extract segment
    if waveform.dim() == 2:
        segment = waveform[:, start_sample:end_sample]
    else:
        segment = waveform[start_sample:end_sample]

    # Convert to numpy
    if hasattr(segment, 'numpy'):
        segment_np = segment.numpy()
    else:
        segment_np = segment

    display_word = word.original if word.original else word.word
    print(f"'{display_word}': {word.start_seconds():.2f}s - {word.end_seconds():.2f}s")

    return Audio(segment_np, rate=sample_rate)


def preview_segment_seconds(
    waveform,
    words: List,  # List of AlignedWord objects
    sample_rate: int = 16000,
    padding: float = 0.1,  # padding in seconds
) -> Tuple[Any, str]:
    """
    Preview audio for a segment of words (times in seconds).

    Args:
        waveform: Audio waveform tensor
        words: List of AlignedWord objects
        sample_rate: Audio sample rate
        padding: Extra seconds at boundaries

    Returns:
        Tuple of (Audio object, text of words)

    Example:
        >>> result = align_long_audio(audio, text)
        >>> preview_segment_seconds(waveform, result.words[100:120])
    """
    from IPython.display import Audio

    if not words:
        print("No words to preview")
        return None, ""

    start_sec = max(0, words[0].start_seconds() - padding)
    end_sec = words[-1].end_seconds() + padding

    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # Extract segment
    if waveform.dim() == 2:
        segment = waveform[:, start_sample:end_sample]
    else:
        segment = waveform[start_sample:end_sample]

    # Convert to numpy
    if hasattr(segment, 'numpy'):
        segment_np = segment.numpy()
    else:
        segment_np = segment

    # Build text
    text = " ".join(w.original if w.original else w.word for w in words)
    print(f"Segment ({words[0].start_seconds():.2f}s - {words[-1].end_seconds():.2f}s):")
    print(f"  {text[:100]}{'...' if len(text) > 100 else ''}")

    return Audio(segment_np, rate=sample_rate), text


def preview_random_segment_seconds(
    waveform,
    words: List,  # List of AlignedWord objects
    num_words: int = 10,
    sample_rate: int = 16000,
) -> Tuple[Any, List, int]:
    """
    Preview a random segment of words.

    Args:
        waveform: Audio waveform tensor
        words: List of AlignedWord objects
        num_words: Number of words to include
        sample_rate: Audio sample rate

    Returns:
        Tuple of (Audio object, list of words, start_idx)

    Example:
        >>> result = align_long_audio(audio, text)
        >>> preview_random_segment_seconds(waveform, result.words)
    """
    max_start = max(0, len(words) - num_words)
    start_idx = random.randint(0, max_start)

    segment_words = words[start_idx:start_idx + num_words]
    audio, _ = preview_segment_seconds(waveform, segment_words, sample_rate)

    return audio, segment_words, start_idx


# =============================================================================
# LEGACY API: Works with frame-based word_alignment dict
# =============================================================================

def preview_word(
    waveform,
    word_alignment: Dict[int, Any],
    word_idx: int,
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
    padding_frames: int = 0,
):
    """
    Preview audio for a specific aligned word (LEGACY - frame-based).

    For new code, use preview_word_seconds() instead.

    Args:
        waveform: Audio waveform tensor (1D or 2D)
        word_alignment: Dict mapping word index to AlignedWord (frame-based)
        word_idx: Word index to preview
        sample_rate: Audio sample rate
        frame_duration: Duration of each frame in seconds
        padding_frames: Extra frames to include before/after

    Returns:
        IPython.display.Audio object for playback
    """
    from IPython.display import Audio
    import torch

    if word_idx not in word_alignment:
        print(f"Word index {word_idx} not found in alignment")
        return None

    word = word_alignment[word_idx]

    # Get start/end frames - AlignedWord now has start_frame/end_frame attributes
    start_frame = word.start_frame
    end_frame = word.end_frame

    # Add padding
    start_frame = max(0, int(start_frame) - padding_frames)
    end_frame = int(end_frame) + padding_frames

    # Convert to samples
    samples_per_frame = int(sample_rate * frame_duration)
    start_sample = start_frame * samples_per_frame
    end_sample = end_frame * samples_per_frame

    # Extract segment
    if waveform.dim() == 2:
        segment = waveform[:, start_sample:end_sample]
    else:
        segment = waveform[start_sample:end_sample]

    # Convert to numpy if needed
    if hasattr(segment, 'numpy'):
        segment_np = segment.numpy()
    else:
        segment_np = segment

    start_sec = start_frame * frame_duration
    end_sec = end_frame * frame_duration
    print(f"Word '{word.word}' [{word_idx}]: {start_sec:.2f}s - {end_sec:.2f}s")

    return Audio(segment_np, rate=sample_rate)


def preview_segment(
    waveform,
    word_alignment: Dict[int, Any],
    start_idx: int,
    num_words: int = 10,
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
    padding_frames: int = 5,
) -> Tuple[Any, List[Any]]:
    """
    Preview a segment of consecutive aligned words (LEGACY - frame-based).

    For new code, use preview_segment_seconds() instead.
    """
    from IPython.display import Audio
    import torch

    sorted_items = sorted(word_alignment.items())
    end_idx = min(start_idx + num_words, len(sorted_items))

    if start_idx >= len(sorted_items):
        print(f"Start index {start_idx} out of range")
        return None, []

    segment_words = sorted_items[start_idx:end_idx]

    # Get time range - AlignedWord now has start_frame/end_frame attributes
    first_word = segment_words[0][1]
    last_word = segment_words[-1][1]

    start_frame = first_word.start_frame
    end_frame = last_word.end_frame

    # Add padding
    start_frame = max(0, int(start_frame) - padding_frames)
    end_frame = int(end_frame) + padding_frames

    # Convert to samples
    samples_per_frame = int(sample_rate * frame_duration)
    start_sample = start_frame * samples_per_frame
    end_sample = end_frame * samples_per_frame

    # Extract segment
    if waveform.dim() == 2:
        segment = waveform[:, start_sample:end_sample]
    else:
        segment = waveform[start_sample:end_sample]

    # Convert to numpy
    if hasattr(segment, 'numpy'):
        segment_np = segment.numpy()
    else:
        segment_np = segment

    start_sec = start_frame * frame_duration
    end_sec = end_frame * frame_duration

    # Print words
    words_text = " ".join([w.word for _, w in segment_words if w.word])
    print(f"Segment [{start_idx}:{end_idx}] ({start_sec:.2f}s - {end_sec:.2f}s):")
    print(f"  {words_text}")

    return Audio(segment_np, rate=sample_rate), [w for _, w in segment_words]


def preview_random_segment(
    waveform,
    word_alignment: Dict[int, Any],
    num_words: int = 10,
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
) -> Tuple[Any, List[Any], int]:
    """
    Preview a random segment of aligned words (LEGACY - frame-based).

    For new code, use preview_random_segment_seconds() instead.
    """
    max_start = max(0, len(word_alignment) - num_words)
    start_idx = random.randint(0, max_start)

    audio, words = preview_segment(
        waveform, word_alignment, start_idx, num_words,
        sample_rate, frame_duration
    )

    return audio, words, start_idx


def preview_word_by_index(
    waveform,
    word_alignment: Dict[int, Any],
    alignment_idx: int,
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
    padding_frames: int = 2,
):
    """
    Preview audio for the Nth word in the alignment (by position, not word index).

    Args:
        waveform: Audio waveform tensor
        word_alignment: Dict mapping word index to AlignedWord
        alignment_idx: Position in the sorted alignment (0, 1, 2, ...)
        sample_rate: Audio sample rate
        frame_duration: Duration of each frame in seconds
        padding_frames: Extra frames to include

    Returns:
        IPython.display.Audio object
    """
    sorted_indices = sorted(word_alignment.keys())
    if alignment_idx < 0 or alignment_idx >= len(sorted_indices):
        print(f"Alignment index {alignment_idx} out of range [0, {len(sorted_indices)})")
        return None

    word_idx = sorted_indices[alignment_idx]
    return preview_word(
        waveform, word_alignment, word_idx,
        sample_rate, frame_duration, padding_frames
    )


def preview_all_words(
    waveform,
    word_alignment: Dict[int, Any],
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
    padding_frames: int = 2,
    max_words: int = 50,
):
    """
    Preview all aligned words (up to max_words).

    Displays each word with its audio player.

    Args:
        waveform: Audio waveform tensor
        word_alignment: Dict mapping word index to AlignedWord
        sample_rate: Audio sample rate
        frame_duration: Duration of each frame
        padding_frames: Extra frames to include
        max_words: Maximum words to display

    Example:
        >>> preview_all_words(waveform, result.word_alignments)
    """
    from IPython.display import display, HTML

    sorted_items = sorted(word_alignment.items())[:max_words]

    print(f"Previewing {len(sorted_items)} words:")
    print("=" * 60)

    for i, (word_idx, word) in enumerate(sorted_items):
        audio = preview_word(
            waveform, word_alignment, word_idx,
            sample_rate, frame_duration, padding_frames
        )
        if audio:
            display(audio)
        print("-" * 40)
