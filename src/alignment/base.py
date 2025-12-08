"""
Base classes and data structures for alignment.

Design Philosophy:
- User-facing classes use SECONDS for all timestamps
- Internal processing uses frames, converted at boundaries
- One representation for each concept (no duplication)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union, Iterator
import json
import torch


# =============================================================================
# Internal Token (frame-based, used during alignment)
# =============================================================================

@dataclass
class AlignedToken:
    """
    A single aligned token (phone/character/subword).

    This is an INTERNAL class used during the alignment process.
    Timestamps are in frames (not seconds).

    Attributes:
        token_id: Token identifier (string or int)
        timestamp: Frame index in the audio
        score: Confidence score (log probability)
        attr: Additional attributes (word index, etc.)
    """
    token_id: Union[str, int]
    timestamp: int  # Frame index
    score: float = 0.0
    attr: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# User-facing Word (seconds-based)
# =============================================================================

@dataclass
class AlignedWord:
    """
    A word with its alignment information.

    This is the PRIMARY user-facing class. All times are in SECONDS.

    Attributes:
        word: The word text (normalized form)
        start: Start time in seconds
        end: End time in seconds
        score: Confidence score (0-1, higher is better)
        original: Original word form before normalization (if different)
        index: Word index in the original transcript

    Example:
        >>> word = result.words[0]
        >>> print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
        hello: 0.52s - 0.78s
    """
    word: str
    start: float  # seconds
    end: float    # seconds
    score: float = 0.0
    original: Optional[str] = None
    index: int = -1

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
        }
        if self.original and self.original != self.word:
            d["original"] = self.original
        if self.score > 0:
            d["score"] = round(self.score, 3)
        if self.index >= 0:
            d["index"] = self.index
        return d

    def __repr__(self):
        if self.original and self.original != self.word:
            return f"AlignedWord('{self.original}' ({self.word}), {self.start:.2f}s-{self.end:.2f}s)"
        return f"AlignedWord('{self.word}', {self.start:.2f}s-{self.end:.2f}s)"


# =============================================================================
# Alignment Result (user-facing)
# =============================================================================

@dataclass
class AlignmentResult:
    """
    Complete alignment result for an audio file.

    This is the PRIMARY result class. Provides simple, intuitive access
    to alignment results with all times in seconds.

    Attributes:
        words: List of aligned words (sorted by time)
        unaligned_regions: List of (start_idx, end_idx) for unaligned text
        metadata: Additional info (duration, model, etc.)

    Example:
        >>> result = align_long_audio("audio.mp3", "transcript.txt")
        >>> for word in result:
        ...     print(f"{word.word}: {word.start:.2f}s")
        >>> result.save_audacity_labels("labels.txt")
    """
    words: List[AlignedWord] = field(default_factory=list)
    unaligned_regions: List[Tuple[int, int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Core properties
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of aligned words."""
        return len(self.words)

    def __iter__(self) -> Iterator[AlignedWord]:
        """Iterate over aligned words."""
        return iter(self.words)

    def __getitem__(self, idx: int) -> AlignedWord:
        """Get word by index."""
        return self.words[idx]

    @property
    def text(self) -> str:
        """Get the aligned transcript as a string."""
        return " ".join(w.word for w in self.words)

    @property
    def duration(self) -> float:
        """Total duration covered by alignment (seconds)."""
        if not self.words:
            return 0.0
        return self.words[-1].end - self.words[0].start

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def get_word_at_time(self, time: float) -> Optional[AlignedWord]:
        """Find the word at a given time (seconds)."""
        for word in self.words:
            if word.start <= time <= word.end:
                return word
        return None

    def get_words_in_range(self, start: float, end: float) -> List[AlignedWord]:
        """Get all words within a time range (seconds)."""
        return [w for w in self.words if start <= w.start <= end]

    # -------------------------------------------------------------------------
    # Export methods
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-serializable)."""
        return {
            "words": [w.to_dict() for w in self.words],
            "unaligned_regions": self.unaligned_regions,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> str:
        """Save alignment as JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return path

    def to_audacity_labels(self) -> str:
        """Export as Audacity label format."""
        lines = []
        for word in self.words:
            lines.append(f"{word.start:.6f}\t{word.end:.6f}\t{word.word}")
        return "\n".join(lines)

    def save_audacity_labels(self, path: str) -> str:
        """Save as Audacity label file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_audacity_labels())
        return path

    def to_srt(self, words_per_subtitle: int = 10) -> str:
        """Export as SRT subtitle format."""
        lines = []
        idx = 1
        for i in range(0, len(self.words), words_per_subtitle):
            chunk = self.words[i:i + words_per_subtitle]
            if not chunk:
                continue
            start = chunk[0].start
            end = chunk[-1].end
            text = " ".join(w.word for w in chunk)

            # Format: HH:MM:SS,mmm
            start_str = f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d}"
            end_str = f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}"

            lines.append(f"{idx}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(text)
            lines.append("")
            idx += 1

        return "\n".join(lines)

    def save_srt(self, path: str, words_per_subtitle: int = 10) -> str:
        """Save as SRT subtitle file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_srt(words_per_subtitle))
        return path

    def to_textgrid(self) -> str:
        """Export as Praat TextGrid format."""
        if not self.words:
            return ""

        xmin = self.words[0].start
        xmax = self.words[-1].end

        lines = [
            'File type = "ooTextFile"',
            'Object class = "TextGrid"',
            '',
            f'xmin = {xmin}',
            f'xmax = {xmax}',
            'tiers? <exists>',
            'size = 1',
            'item []:',
            '    item [1]:',
            '        class = "IntervalTier"',
            '        name = "words"',
            f'        xmin = {xmin}',
            f'        xmax = {xmax}',
            f'        intervals: size = {len(self.words)}',
        ]

        for i, word in enumerate(self.words, 1):
            lines.extend([
                f'        intervals [{i}]:',
                f'            xmin = {word.start}',
                f'            xmax = {word.end}',
                f'            text = "{word.word}"',
            ])

        return "\n".join(lines)

    def save_textgrid(self, path: str) -> str:
        """Save as Praat TextGrid file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_textgrid())
        return path

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Get a summary string."""
        lines = [
            f"Alignment Result:",
            f"  Words aligned: {len(self.words)}",
        ]
        if self.unaligned_regions:
            lines.append(f"  Unaligned regions: {len(self.unaligned_regions)}")
        if self.words:
            lines.append(f"  Time range: {self.words[0].start:.2f}s - {self.words[-1].end:.2f}s")
        if "total_words" in self.metadata:
            coverage = 100.0 * len(self.words) / self.metadata["total_words"]
            lines.append(f"  Coverage: {coverage:.1f}%")
        if "audio_duration" in self.metadata:
            lines.append(f"  Audio duration: {self.metadata['audio_duration']:.1f}s")
        return "\n".join(lines)

    def __repr__(self):
        return f"AlignmentResult({len(self.words)} words)"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AlignmentConfig:
    """
    Configuration for alignment.

    Attributes:
        language: ISO 639-3 language code (e.g., "eng", "deu", "cmn")
        sample_rate: Audio sample rate (default 16000)
        segment_size: Segment size in seconds for long audio
        overlap: Overlap between segments in seconds
        batch_size: Batch size for neural network inference
        device: Device to run on ("cuda", "cpu", "auto")
    """
    language: str = "eng"
    sample_rate: int = 16000
    segment_size: float = 15.0
    overlap: float = 2.0
    batch_size: int = 32
    device: str = "auto"

    # Internal parameters (advanced users only)
    frame_duration: float = 0.02
    skip_penalty: float = -0.5
    return_penalty: float = -18.0
    blank_penalty: float = 0.0
    neighborhood_size: int = 5

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Backend base class (for extensibility)
# =============================================================================

class AlignerBackend(ABC):
    """
    Abstract base class for alignment backends.

    Implement this to add new alignment methods (MFA, Gentle, etc.)
    """

    BACKEND_NAME: str = "base"

    def __init__(self, config: AlignmentConfig):
        self.config = config
        self._loaded = False

    @abstractmethod
    def align(self, waveform: torch.Tensor, text: str, **kwargs) -> AlignmentResult:
        """Perform alignment. Returns AlignmentResult with times in seconds."""
        raise NotImplementedError

    def load(self) -> None:
        """Load models. Override in subclasses."""
        self._loaded = True

    def unload(self) -> None:
        """Free resources. Override in subclasses."""
        self._loaded = False

    def __enter__(self):
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, *args):
        self.unload()
        return False
