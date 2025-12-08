"""
Audio preview utilities for alignment verification.

Provides functions to play audio segments corresponding to aligned words,
following the pattern in torchaudio's forced_alignment_tutorial.py.
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import random


def preview_word(
    waveform,
    word_alignment: Dict[int, Any],
    word_idx: int,
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
    padding_frames: int = 0,
):
    """
    Preview audio for a specific aligned word.

    Args:
        waveform: Audio waveform tensor (1D or 2D)
        word_alignment: Dict mapping word index to AlignedWord
        word_idx: Word index to preview
        sample_rate: Audio sample rate
        frame_duration: Duration of each frame in seconds
        padding_frames: Extra frames to include before/after

    Returns:
        IPython.display.Audio object for playback

    Example:
        >>> audio_widget = preview_word(waveform, result.word_alignments, 10)
        >>> display(audio_widget)
    """
    from IPython.display import Audio
    import torch

    if word_idx not in word_alignment:
        print(f"Word index {word_idx} not found in alignment")
        return None

    word = word_alignment[word_idx]

    # Get start/end frames
    if hasattr(word.start_time, 'item'):
        start_frame = word.start_time.item()
    else:
        start_frame = word.start_time

    if word.end_time is not None:
        if hasattr(word.end_time, 'item'):
            end_frame = word.end_time.item()
        else:
            end_frame = word.end_time
    else:
        end_frame = start_frame + 25  # Default ~0.5s

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
    Preview a segment of consecutive aligned words.

    Args:
        waveform: Audio waveform tensor
        word_alignment: Dict mapping word index to AlignedWord
        start_idx: Starting alignment position
        num_words: Number of words to include
        sample_rate: Audio sample rate
        frame_duration: Duration of each frame in seconds
        padding_frames: Extra frames at boundaries

    Returns:
        Tuple of (Audio object, list of words in segment)
    """
    from IPython.display import Audio
    import torch

    sorted_items = sorted(word_alignment.items())
    end_idx = min(start_idx + num_words, len(sorted_items))

    if start_idx >= len(sorted_items):
        print(f"Start index {start_idx} out of range")
        return None, []

    segment_words = sorted_items[start_idx:end_idx]

    # Get time range
    first_word = segment_words[0][1]
    last_word = segment_words[-1][1]

    if hasattr(first_word.start_time, 'item'):
        start_frame = first_word.start_time.item()
    else:
        start_frame = first_word.start_time

    if last_word.end_time is not None:
        if hasattr(last_word.end_time, 'item'):
            end_frame = last_word.end_time.item()
        else:
            end_frame = last_word.end_time
    else:
        # Fallback: use last word's start_time + default duration
        if hasattr(last_word.start_time, 'item'):
            last_start = last_word.start_time.item()
        else:
            last_start = last_word.start_time
        end_frame = last_start + 25  # Default ~0.5s for last word

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
    words_text = " ".join([w.word for _, w in segment_words])
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
    Preview a random segment of aligned words.

    Args:
        waveform: Audio waveform tensor
        word_alignment: Dict mapping word index to AlignedWord
        num_words: Number of words to include
        sample_rate: Audio sample rate
        frame_duration: Duration of each frame

    Returns:
        Tuple of (Audio object, list of words, start_idx)
    """
    max_start = max(0, len(word_alignment) - num_words)
    start_idx = random.randint(0, max_start)

    audio, words = preview_segment(
        waveform, word_alignment, start_idx, num_words,
        sample_rate, frame_duration
    )

    return audio, words, start_idx


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
