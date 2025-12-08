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
# User-facing Character/Token (seconds-based)
# =============================================================================

@dataclass
class AlignedChar:
    """
    A single aligned character/token with timestamp.

    This provides character-level (or token-level) alignment for users
    who need finer granularity than word-level.

    Attributes:
        char: The character/token text
        start: Start time in seconds
        end: End time in seconds
        score: Confidence score (log probability from CTC)
        word_index: Index of the word this char belongs to

    Example:
        >>> for char in word.chars:
        ...     print(f"{char.char}: {char.start:.3f}s")
    """
    char: str
    start: float  # seconds
    end: float    # seconds
    score: float = 0.0
    word_index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "char": self.char,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "score": round(self.score, 3) if self.score != 0 else 0,
        }

    def __repr__(self):
        return f"AlignedChar('{self.char}', {self.start:.3f}s-{self.end:.3f}s)"


# =============================================================================
# User-facing Word (seconds-based)
# =============================================================================

# Default frame duration (20ms per frame, 50 frames/second)
# This is set by the model when loaded. Most CTC models use 20ms.
DEFAULT_FRAME_DURATION = 0.02


def set_frame_duration(duration: float) -> None:
    """
    Set the global frame duration used for frame-to-seconds conversion.

    This is typically called by the model loader when a model is loaded,
    as different models may have different frame rates.

    Args:
        duration: Frame duration in seconds (e.g., 0.02 for 20ms frames)

    Example:
        >>> from alignment.base import set_frame_duration
        >>> set_frame_duration(0.02)  # 20ms frames (MMS, Wav2Vec2)
        >>> set_frame_duration(0.01)  # 10ms frames (some NeMo models)
    """
    global DEFAULT_FRAME_DURATION
    DEFAULT_FRAME_DURATION = duration


def get_frame_duration() -> float:
    """Get the current global frame duration."""
    return DEFAULT_FRAME_DURATION


@dataclass
class AlignedWord:
    """
    A word with its alignment information.

    Primary attributes are in FRAMES. Call start_seconds()/end_seconds() for seconds.

    Attributes:
        word: The word text (normalized form)
        start_frame: Start frame index
        end_frame: End frame index
        score: Confidence score (average of char scores, higher is better)
        original: Original word form before normalization (if different)
        index: Word index in the original transcript
        chars: List of character-level alignments (optional)

    Example:
        >>> word = result.words[0]
        >>> print(f"{word.word}: {word.start_seconds():.2f}s - {word.end_seconds():.2f}s")
        >>> print(f"Frames: {word.start_frame} - {word.end_frame}")
    """
    word: str
    start_frame: int
    end_frame: int
    score: float = 0.0
    original: Optional[str] = None
    index: int = -1
    chars: List[AlignedChar] = field(default_factory=list)

    def start_seconds(self, frame_duration: float = None) -> float:
        """Get start time in seconds.

        Args:
            frame_duration: Optional override. If None, uses global DEFAULT_FRAME_DURATION.
        """
        if frame_duration is None:
            frame_duration = DEFAULT_FRAME_DURATION
        return self.start_frame * frame_duration

    def end_seconds(self, frame_duration: float = None) -> float:
        """Get end time in seconds.

        Args:
            frame_duration: Optional override. If None, uses global DEFAULT_FRAME_DURATION.
        """
        if frame_duration is None:
            frame_duration = DEFAULT_FRAME_DURATION
        return self.end_frame * frame_duration

    def duration_seconds(self, frame_duration: float = None) -> float:
        """Get duration in seconds.

        Args:
            frame_duration: Optional override. If None, uses global DEFAULT_FRAME_DURATION.
        """
        if frame_duration is None:
            frame_duration = DEFAULT_FRAME_DURATION
        return (self.end_frame - self.start_frame) * frame_duration

    @property
    def duration_frames(self) -> int:
        """Duration in frames."""
        return self.end_frame - self.start_frame

    @property
    def confidence(self) -> float:
        """Alias for score (confidence score 0-1)."""
        return self.score

    @property
    def display_text(self) -> str:
        """
        User-friendly display text showing original form when available.

        Returns original word form, with normalized form in parentheses
        if they differ. This is the recommended way to display words to users.

        Examples:
            - Normal word: "hello"
            - With original: "Meta's (metas)"
            - Unknown word: "你好 (*)"

        Returns:
            Formatted string for display to users
        """
        if self.original and self.original != self.word:
            return f"{self.original} ({self.word})"
        return self.word

    def to_dict(self, include_chars: bool = False, frame_duration: float = None) -> Dict[str, Any]:
        """Convert to dictionary (times in seconds for JSON export).

        Args:
            include_chars: Include character-level alignments
            frame_duration: Optional override. If None, uses global DEFAULT_FRAME_DURATION.
        """
        d = {
            "word": self.word,
            "start": round(self.start_seconds(frame_duration), 3),
            "end": round(self.end_seconds(frame_duration), 3),
        }
        if self.original and self.original != self.word:
            d["original"] = self.original
        if self.score > 0:
            d["score"] = round(self.score, 3)
        if self.index >= 0:
            d["index"] = self.index
        if include_chars and self.chars:
            d["chars"] = [c.to_dict() for c in self.chars]
        return d

    def __repr__(self):
        score_str = f", score={self.score:.2f}" if self.score > 0 else ""
        start_s = self.start_seconds()
        end_s = self.end_seconds()
        if self.original and self.original != self.word:
            return f"AlignedWord('{self.original}' ({self.word}), {start_s:.2f}s-{end_s:.2f}s{score_str})"
        return f"AlignedWord('{self.word}', {start_s:.2f}s-{end_s:.2f}s{score_str})"


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
        chars: List of all aligned characters (sorted by time)
        unaligned_regions: List of (start_idx, end_idx) for unaligned text
        metadata: Additional info (duration, model, etc.)

    Example:
        >>> result = align_long_audio("audio.mp3", "transcript.txt")
        >>> for word in result:
        ...     print(f"{word.word}: {word.start_seconds():.2f}s")
        >>> result.save_audacity_labels("labels.txt")
        >>> print(result.statistics())
    """
    words: List[AlignedWord] = field(default_factory=list)
    chars: List[AlignedChar] = field(default_factory=list)
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
        return self.words[-1].end_seconds() - self.words[0].start_seconds()

    @property
    def num_words(self) -> int:
        """Number of aligned words."""
        return len(self.words)

    @property
    def word_alignments(self) -> Dict[int, AlignedWord]:
        """Get word alignments as dict keyed by word index.

        Useful for detecting gaps in alignment (unaligned words).
        """
        return {w.index: w for w in self.words if w.index >= 0}

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def get_word_at_time(self, time: float) -> Optional[AlignedWord]:
        """Find the word at a given time (seconds)."""
        for word in self.words:
            if word.start_seconds() <= time <= word.end_seconds():
                return word
        return None

    def get_words_in_range(self, start: float, end: float) -> List[AlignedWord]:
        """Get all words within a time range (seconds)."""
        return [w for w in self.words if start <= w.start_seconds() <= end]

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
            lines.append(f"{word.start_seconds():.6f}\t{word.end_seconds():.6f}\t{word.word}")
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
            start = chunk[0].start_seconds()
            end = chunk[-1].end_seconds()
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

        xmin = self.words[0].start_seconds()
        xmax = self.words[-1].end_seconds()

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
                f'            xmin = {word.start_seconds()}',
                f'            xmax = {word.end_seconds()}',
                f'            text = "{word.word}"',
            ])

        return "\n".join(lines)

    def save_textgrid(self, path: str) -> str:
        """Save as Praat TextGrid file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_textgrid())
        return path

    def to_ctm(self, audio_id: str = "audio") -> str:
        """
        Export as CTM (time-marked conversation) format.

        CTM is a standard format for speech recognition output.
        Format: <audio_id> <channel> <start> <duration> <word> [<confidence>]

        Args:
            audio_id: Identifier for the audio file

        Returns:
            CTM format string
        """
        lines = []
        for word in self.words:
            duration = word.duration_seconds()
            conf_str = f" {word.score:.3f}" if word.score > 0 else ""
            lines.append(f"{audio_id} 1 {word.start_seconds():.3f} {duration:.3f} {word.word}{conf_str}")
        return "\n".join(lines)

    def save_ctm(self, path: str, audio_id: str = "audio") -> str:
        """Save as CTM file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_ctm(audio_id))
        return path

    def to_ass(
        self,
        words_per_line: int = 8,
        style_name: str = "Default",
        font_name: str = "Arial",
        font_size: int = 48,
        primary_color: str = "&H00FFFFFF",  # White
        highlight_color: str = "&H0000FFFF",  # Yellow
        outline_color: str = "&H00000000",  # Black
        vertical_margin: int = 50,
    ) -> str:
        """
        Export as ASS (Advanced SubStation Alpha) subtitle format.

        ASS supports word-by-word highlighting for karaoke-style subtitles.

        Args:
            words_per_line: Number of words per subtitle line
            style_name: Name of the style
            font_name: Font family name
            font_size: Font size in points
            primary_color: Default text color (ASS format: &HAABBGGRR)
            highlight_color: Highlighted word color
            outline_color: Text outline color
            vertical_margin: Bottom margin in pixels

        Returns:
            ASS format string
        """
        def time_to_ass(seconds: float) -> str:
            """Convert seconds to ASS timestamp format (H:MM:SS.cc)"""
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h}:{m:02d}:{s:05.2f}"

        # ASS header
        header = f"""[Script Info]
Title: TorchAudio Aligner Output
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},{font_name},{font_size},{primary_color},{highlight_color},{outline_color},&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,{vertical_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        lines = [header.strip()]

        # Group words into subtitle lines
        for i in range(0, len(self.words), words_per_line):
            chunk = self.words[i:i + words_per_line]
            if not chunk:
                continue

            line_start = chunk[0].start_seconds()
            line_end = chunk[-1].end_seconds()

            # Build karaoke text with timing tags
            # {\k<duration in centiseconds>}word
            karaoke_parts = []
            for j, word in enumerate(chunk):
                # Duration in centiseconds (100ths of a second)
                if j < len(chunk) - 1:
                    duration_cs = int((chunk[j + 1].start_seconds() - word.start_seconds()) * 100)
                else:
                    duration_cs = int(word.duration_seconds() * 100)
                karaoke_parts.append(f"{{\\kf{duration_cs}}}{word.word}")

            text = " ".join(karaoke_parts)
            start_str = time_to_ass(line_start)
            end_str = time_to_ass(line_end)

            lines.append(f"Dialogue: 0,{start_str},{end_str},{style_name},,0,0,0,,{text}")

        return "\n".join(lines)

    def save_ass(self, path: str, **kwargs) -> str:
        """Save as ASS subtitle file."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_ass(**kwargs))
        return path

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Compute alignment statistics.

        Returns:
            Dictionary with various statistics about the alignment.
        """
        if not self.words:
            return {"error": "No words aligned"}

        # Basic counts
        stats = {
            "num_words": len(self.words),
            "num_chars": len(self.chars) if self.chars else sum(len(w.chars) for w in self.words),
            "num_unaligned_regions": len(self.unaligned_regions),
        }

        # Coverage
        if "total_words" in self.metadata:
            stats["total_words_in_text"] = self.metadata["total_words"]
            stats["coverage_percent"] = 100.0 * len(self.words) / self.metadata["total_words"]

        # Time statistics
        stats["time_range"] = {
            "start": self.words[0].start_seconds(),
            "end": self.words[-1].end_seconds(),
            "duration": self.words[-1].end_seconds() - self.words[0].start_seconds(),
        }

        if "audio_duration" in self.metadata:
            stats["audio_duration"] = self.metadata["audio_duration"]
            aligned_duration = self.words[-1].end_seconds() - self.words[0].start_seconds()
            stats["aligned_time_percent"] = 100.0 * aligned_duration / self.metadata["audio_duration"]

        # Word duration statistics
        durations = [w.duration_seconds() for w in self.words]
        stats["word_duration"] = {
            "mean": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "median": sorted(durations)[len(durations) // 2],
        }

        # Confidence statistics (if available)
        scores = [w.score for w in self.words if w.score > 0]
        if scores:
            stats["confidence"] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "words_with_score": len(scores),
            }

        # Gap statistics (time between words)
        gaps = []
        for i in range(1, len(self.words)):
            gap = self.words[i].start_seconds() - self.words[i - 1].end_seconds()
            if gap > 0:
                gaps.append(gap)
        if gaps:
            stats["gaps"] = {
                "count": len(gaps),
                "total": sum(gaps),
                "mean": sum(gaps) / len(gaps),
                "max": max(gaps),
            }

        return stats

    def print_statistics(self) -> None:
        """Print alignment statistics in a readable format."""
        stats = self.statistics()

        print("=" * 60)
        print("Alignment Statistics")
        print("=" * 60)

        print(f"\nWords aligned: {stats['num_words']}")
        if "total_words_in_text" in stats:
            print(f"Total words in text: {stats['total_words_in_text']}")
            print(f"Coverage: {stats['coverage_percent']:.1f}%")

        print(f"Unaligned regions: {stats['num_unaligned_regions']}")

        tr = stats["time_range"]
        print(f"\nTime range: {tr['start']:.2f}s - {tr['end']:.2f}s ({tr['duration']:.1f}s)")
        if "audio_duration" in stats:
            print(f"Audio duration: {stats['audio_duration']:.1f}s")
            print(f"Aligned time: {stats.get('aligned_time_percent', 0):.1f}%")

        wd = stats["word_duration"]
        print("\nWord duration:")
        print(f"  Mean: {wd['mean']*1000:.0f}ms")
        print(f"  Min: {wd['min']*1000:.0f}ms")
        print(f"  Max: {wd['max']*1000:.0f}ms")
        print(f"  Median: {wd['median']*1000:.0f}ms")

        if "confidence" in stats:
            conf = stats["confidence"]
            print("\nConfidence scores:")
            print(f"  Mean: {conf['mean']:.3f}")
            print(f"  Min: {conf['min']:.3f}")
            print(f"  Max: {conf['max']:.3f}")

        if "gaps" in stats:
            g = stats["gaps"]
            print("\nGaps between words:")
            print(f"  Count: {g['count']}")
            print(f"  Total: {g['total']*1000:.0f}ms")
            print(f"  Mean: {g['mean']*1000:.0f}ms")
            print(f"  Max: {g['max']*1000:.0f}ms")

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
            lines.append(f"  Time range: {self.words[0].start_seconds():.2f}s - {self.words[-1].end_seconds():.2f}s")
        if "total_words" in self.metadata:
            coverage = 100.0 * len(self.words) / self.metadata["total_words"]
            lines.append(f"  Coverage: {coverage:.1f}%")
        if "audio_duration" in self.metadata:
            lines.append(f"  Audio duration: {self.metadata['audio_duration']:.1f}s")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Interactive Demo
    # -------------------------------------------------------------------------

    def create_interactive_demo(self, output_dir: str, title: str = None) -> str:
        """
        Create an interactive HTML demo with clickable word audio clips.

        This creates a self-contained demo directory with:
        - index.html: Interactive visualization with segment selector
        - audio_clips/: Pre-extracted WAV files for each aligned word

        Features:
        - Segment selector: Choose start/end index or click "Random 50 Words"
        - Click on any word to play its audio clip (works offline)
        - Modern UI with statistics

        Args:
            output_dir: Directory to save demo files
            title: Optional title for the demo page

        Returns:
            Path to the index.html file

        Example:
            >>> result = align_long_audio("audio.mp3", "text.txt", language="eng")
            >>> result.create_interactive_demo("demo_output/")
            >>> # Open demo_output/index.html in browser

        Raises:
            ValueError: If audio_file not in metadata (need to use align_long_audio)
        """
        # Check required metadata
        audio_file = self.metadata.get("audio_file")
        if not audio_file:
            raise ValueError(
                "audio_file not found in metadata. "
                "Use align_long_audio() to get results with this method enabled."
            )

        # Build title if not provided
        if title is None:
            title = "TorchAudio Aligner - Interactive Demo"
            if "language" in self.metadata:
                title = f"{title} ({self.metadata['language']})"

        # Use original_text if available, otherwise reconstruct from words
        text = self.metadata.get("original_text", " ".join(w.word for w in self.words))

        # Build word_alignment dict from words list
        word_alignment = {w.index: w for w in self.words}

        # Import and call the actual implementation
        from visualization_utils.gentle import create_interactive_demo as _create_demo
        return _create_demo(
            word_alignment=word_alignment,
            text=text,
            audio_file=audio_file,
            output_dir=output_dir,
            frame_duration=DEFAULT_FRAME_DURATION,
            title=title,
        )

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
