"""
Audio preview utilities for alignment verification.

Provides functions to play audio segments corresponding to aligned words.

SIMPLE API (recommended - uses audio file directly):
    result = align_long_audio(audio, text)
    play_word(result, 10)  # Play word at index 10
    play_segment(result, 100, num_words=20)  # Play 20 words starting at 100
    play_random(result, num_words=30)  # Play random 30-word segment

WAVEFORM API (for when you already have waveform loaded):
    preview_word_seconds(waveform, result.words[10], sample_rate)
    preview_segment_seconds(waveform, result.words[100:120], sample_rate)

LEGACY API (for backwards compatibility with frame-based dict):
    preview_word(waveform, word_alignment, word_idx, frame_duration=0.02)
"""

from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import random


# =============================================================================
# SIMPLE API: Works with AlignmentResult and audio file path
# =============================================================================

def play_word(result, word_idx: int, audio_file: str = None):
    """
    Play audio for a specific word.

    Args:
        result: AlignmentResult from align_long_audio()
        word_idx: Index of word to play (0-based position in result.words)
        audio_file: Path to audio file (uses result.metadata if not provided)

    Returns:
        IPython.display.Audio object

    Example:
        >>> result = align_long_audio("audio.mp3", "text.txt")
        >>> play_word(result, 10)  # Play the 11th word
    """
    from IPython.display import Audio
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required: pip install pydub")

    # Get audio file
    if audio_file is None:
        audio_file = result.metadata.get("audio_file")
    if not audio_file:
        raise ValueError("audio_file not found. Pass it explicitly or use align_long_audio().")

    if word_idx < 0 or word_idx >= len(result.words):
        print(f"Word index {word_idx} out of range [0, {len(result.words)})")
        return None

    word = result.words[word_idx]
    start_sec = word.start_seconds()
    end_sec = word.end_seconds()

    # Load and slice audio
    audio = AudioSegment.from_file(audio_file).set_channels(1)
    segment = audio[start_sec * 1000:end_sec * 1000]

    # Display info
    display_text = word.original if word.original else word.word
    print(f"[{word_idx}] '{display_text}': {start_sec:.2f}s - {end_sec:.2f}s")

    return Audio(segment.get_array_of_samples(), rate=segment.frame_rate)


def play_segment(result, start_idx: int, num_words: int = 20, audio_file: str = None):
    """
    Play audio for a segment of consecutive words.

    Args:
        result: AlignmentResult from align_long_audio()
        start_idx: Starting word index
        num_words: Number of words to play
        audio_file: Path to audio file (uses result.metadata if not provided)

    Returns:
        IPython.display.Audio object

    Example:
        >>> result = align_long_audio("audio.mp3", "text.txt")
        >>> play_segment(result, 100, num_words=30)
    """
    from IPython.display import Audio
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required: pip install pydub")

    # Get audio file
    if audio_file is None:
        audio_file = result.metadata.get("audio_file")
    if not audio_file:
        raise ValueError("audio_file not found. Pass it explicitly or use align_long_audio().")

    # Get words
    end_idx = min(start_idx + num_words, len(result.words))
    words = result.words[start_idx:end_idx]

    if not words:
        print("No words in range")
        return None

    start_sec = words[0].start_seconds()
    end_sec = words[-1].end_seconds()

    # Load and slice audio
    audio = AudioSegment.from_file(audio_file).set_channels(1)
    segment = audio[start_sec * 1000:end_sec * 1000]

    # Display info - show both original and normalized/romanized forms
    # Align columns so original and romanized words line up vertically
    original_words = [w.original if w.original else w.word for w in words]
    normalized_words = [w.word for w in words]

    # Check if we need to show both rows
    show_normalized = original_words != normalized_words

    if show_normalized:
        # Use wcwidth for proper Unicode display width (CJK chars are 2 columns wide)
        try:
            import wcwidth

            def display_width(s):
                """Get display width of string (CJK chars = 2, Latin = 1)."""
                return sum(max(0, wcwidth.wcwidth(c)) for c in s)

            def pad_to_width(s, target_width):
                """Pad string to target display width."""
                current_width = display_width(s)
                return s + " " * (target_width - current_width)

        except ImportError:
            # Fallback to len() if wcwidth not installed
            def display_width(s):
                return len(s)

            def pad_to_width(s, target_width):
                return s.ljust(target_width)

        # Calculate column widths based on display width
        col_widths = [max(display_width(o), display_width(n)) for o, n in zip(original_words, normalized_words)]
        original_aligned = " ".join(pad_to_width(o, w) for o, w in zip(original_words, col_widths))
        normalized_aligned = " ".join(pad_to_width(n, w) for n, w in zip(normalized_words, col_widths))

        print(f"Words {start_idx}-{end_idx-1} ({start_sec:.2f}s - {end_sec:.2f}s):")
        print(f"  {original_aligned}")
        print(f"  {normalized_aligned}")
    else:
        original_text = " ".join(original_words)
        print(f"Words {start_idx}-{end_idx-1} ({start_sec:.2f}s - {end_sec:.2f}s):")
        print(f"  {original_text}")

    return Audio(segment.get_array_of_samples(), rate=segment.frame_rate)


def play_random(result, num_words: int = 30, audio_file: str = None):
    """
    Play a random segment of words.

    Args:
        result: AlignmentResult from align_long_audio()
        num_words: Number of words to play
        audio_file: Path to audio file (uses result.metadata if not provided)

    Returns:
        Tuple of (Audio object, start_idx)

    Example:
        >>> result = align_long_audio("audio.mp3", "text.txt")
        >>> play_random(result, num_words=30)
    """
    if len(result) < num_words:
        num_words = len(result)

    start_idx = random.randint(0, max(0, len(result) - num_words))
    audio = play_segment(result, start_idx, num_words, audio_file)
    return audio, start_idx


def play_words_sequential(result, start_idx: int = 0, num_words: int = 10, audio_file: str = None):
    """
    Play words one by one with their text displayed.

    Args:
        result: AlignmentResult from align_long_audio()
        start_idx: Starting word index
        num_words: Number of words to play
        audio_file: Path to audio file (uses result.metadata if not provided)

    Example:
        >>> result = align_long_audio("audio.mp3", "text.txt")
        >>> play_words_sequential(result, start_idx=50, num_words=10)
    """
    from IPython.display import Audio, display
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError("pydub required: pip install pydub")

    # Get audio file
    if audio_file is None:
        audio_file = result.metadata.get("audio_file")
    if not audio_file:
        raise ValueError("audio_file not found. Pass it explicitly or use align_long_audio().")

    # Load audio once
    audio = AudioSegment.from_file(audio_file).set_channels(1)

    print(f"Playing words {start_idx} to {start_idx + num_words - 1}:")
    print("=" * 50)

    end_idx = min(start_idx + num_words, len(result.words))
    words = result.words[start_idx:end_idx]

    for i, word in enumerate(words):
        start_sec = word.start_seconds()
        # Use next word's start or add buffer for end time
        if i + 1 < len(words):
            end_sec = words[i + 1].start_seconds()
        else:
            end_sec = word.end_seconds() + 0.3

        display_text = word.original if word.original else word.word
        print(f"\n[{start_idx + i}] '{display_text}' ({start_sec:.2f}s - {end_sec:.2f}s):")

        segment = audio[start_sec * 1000:end_sec * 1000]
        display(Audio(segment.get_array_of_samples(), rate=segment.frame_rate))


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
