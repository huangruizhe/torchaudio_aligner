"""
Base classes and data structures for alignment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import torch


@dataclass
class AlignedToken:
    """
    A single aligned token (phone/character/subword).

    Attributes:
        token_id: Token identifier (string or int)
        timestamp: Frame index in the audio
        score: Confidence score (log probability)
        attr: Additional attributes (word index, etc.)
    """
    token_id: Union[str, int]
    timestamp: int
    score: float = 0.0
    attr: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignedWord:
    """
    A word with its alignment information.

    Attributes:
        word: The word text
        start_time: Start frame index
        end_time: End frame index (None if not set)
        phones: List of phone-level alignments
        score: Word-level confidence score
    """
    word: str
    start_time: int
    end_time: Optional[int] = None
    phones: List[AlignedToken] = field(default_factory=list)
    score: float = 0.0

    @property
    def start_seconds(self) -> float:
        """Start time in seconds (assuming 20ms frames)."""
        return self.start_time * 0.02

    @property
    def end_seconds(self) -> Optional[float]:
        """End time in seconds (assuming 20ms frames)."""
        return self.end_time * 0.02 if self.end_time is not None else None

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        if self.end_time is not None:
            return (self.end_time - self.start_time) * 0.02
        return None


@dataclass
class AlignmentResult:
    """
    Complete alignment result for an audio file.

    Attributes:
        word_alignments: Dict mapping word index to AlignedWord
        unaligned_indices: List of (start, end) word indices that couldn't be aligned
        token_alignments: Raw token-level alignments (optional)
        metadata: Additional metadata (model used, parameters, etc.)
    """
    word_alignments: Dict[int, AlignedWord]
    unaligned_indices: List[Tuple[int, int]] = field(default_factory=list)
    token_alignments: List[AlignedToken] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_aligned_words(self) -> int:
        """Number of successfully aligned words."""
        return len(self.word_alignments)

    @property
    def aligned_text(self) -> str:
        """Get the aligned transcript as a string."""
        words = [w.word for _, w in sorted(self.word_alignments.items())]
        return " ".join(words)

    def get_word_at_time(self, time_seconds: float) -> Optional[AlignedWord]:
        """Find the word at a given time."""
        frame = int(time_seconds / 0.02)
        for word in self.word_alignments.values():
            if word.start_time <= frame:
                if word.end_time is None or frame <= word.end_time:
                    return word
        return None

    def to_audacity_labels(self, frame_duration: float = 0.02) -> str:
        """
        Export alignment as Audacity labels format.

        Returns:
            String with tab-separated labels (start, end, label)
        """
        lines = []
        for _, word in sorted(self.word_alignments.items()):
            t = word.start_time * frame_duration
            lines.append(f"{t:.2f}\t{t:.2f}\t{word.word}")
        return "\n".join(lines)

    def to_textgrid(self) -> str:
        """Export alignment as Praat TextGrid format."""
        # Basic TextGrid format
        lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '']

        if not self.word_alignments:
            return "\n".join(lines)

        sorted_words = sorted(self.word_alignments.items())
        xmin = sorted_words[0][1].start_time * 0.02
        xmax = sorted_words[-1][1].end_time * 0.02 if sorted_words[-1][1].end_time else xmin + 1.0

        lines.extend([
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
            f'        intervals: size = {len(sorted_words)}',
        ])

        for i, (_, word) in enumerate(sorted_words, 1):
            start = word.start_time * 0.02
            end = word.end_time * 0.02 if word.end_time else start + 0.1
            lines.extend([
                f'        intervals [{i}]:',
                f'            xmin = {start}',
                f'            xmax = {end}',
                f'            text = "{word.word}"',
            ])

        return "\n".join(lines)


@dataclass
class AlignmentConfig:
    """
    Configuration for alignment.

    Attributes:
        backend: Alignment backend ("wfst", "mfa", "gentle")
        language: ISO 639-3 language code
        sample_rate: Audio sample rate (default 16000)
        segment_size: Segment size in seconds for long audio
        overlap: Overlap between segments in seconds
        batch_size: Batch size for neural network inference
        device: Device to run on ("cuda", "cpu", "mps")

        # WFST-specific
        skip_penalty: Penalty for skipping words (default -0.5)
        return_penalty: Penalty for return arcs (default -18.0)
        blank_penalty: Penalty for blank tokens (default 0)

        # Quality thresholds
        per_frame_score_threshold: Min score per frame (default 0.5)
        skip_percentage_threshold: Max skip percentage (default 0.2)
        neighborhood_size: Neighborhood for LIS filtering (default 5)
    """
    backend: str = "wfst"
    language: Optional[str] = None
    sample_rate: int = 16000
    segment_size: float = 15.0
    overlap: float = 2.0
    shortest_segment_size: float = 0.2
    batch_size: int = 32
    device: Optional[str] = None
    frame_duration: float = 0.02

    # WFST-specific parameters
    skip_penalty: float = -0.5
    return_penalty: float = -18.0
    blank_penalty: float = 0.0

    # Quality thresholds
    per_frame_score_threshold: float = 0.5
    skip_percentage_threshold: float = 0.2
    return_arcs_num_threshold: int = 3
    neighborhood_size: int = 5

    # Extra options
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class AlignerBackend(ABC):
    """
    Abstract base class for alignment backends.

    All alignment backends must implement:
    - align(): Perform alignment
    - supports_language(): Check language support

    Optional:
    - load(): Load any required models
    - unload(): Free resources
    """

    BACKEND_NAME: str = "base"
    SUPPORTED_LANGUAGES: List[str] = []

    def __init__(self, config: AlignmentConfig):
        self.config = config
        self._loaded = False

    @property
    def name(self) -> str:
        """Backend name for identification."""
        return self.BACKEND_NAME

    @property
    def is_loaded(self) -> bool:
        """Whether any required models are loaded."""
        return self._loaded

    def load(self) -> None:
        """Load any required models. Override in subclasses."""
        self._loaded = True

    def unload(self) -> None:
        """Free resources. Override in subclasses."""
        self._loaded = False

    @abstractmethod
    def align(
        self,
        waveform: torch.Tensor,
        text: str,
        **kwargs,
    ) -> AlignmentResult:
        """
        Perform speech-to-text alignment.

        Args:
            waveform: Audio tensor of shape (1, T) or (T,)
            text: Text to align
            **kwargs: Additional backend-specific arguments

        Returns:
            AlignmentResult with word-level alignments
        """
        raise NotImplementedError

    def supports_language(self, language: str) -> bool:
        """
        Check if the backend supports a given language.

        Args:
            language: ISO 639-3 language code

        Returns:
            True if supported
        """
        if not self.SUPPORTED_LANGUAGES:
            return True  # Empty = all languages
        return language.lower() in [l.lower() for l in self.SUPPORTED_LANGUAGES]

    def __enter__(self):
        """Context manager entry."""
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loaded={self._loaded})"
