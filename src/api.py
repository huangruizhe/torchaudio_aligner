"""
High-level API for TorchAudio Long-Form Aligner.

This module provides a unified interface that combines:
- Audio frontend (loading, preprocessing, segmentation)
- Text frontend (loading, normalization, tokenization)
- Labeling utils (acoustic model inference)
- Alignment (WFST-based flexible alignment)
- Stitching utils (LIS-based segment concatenation)

Usage:
    from torchaudio_aligner import align_long_audio

    result = align_long_audio(
        audio="path/to/audio.mp3",
        text="path/to/transcript.pdf",
    )

    # Simple access - times are in seconds, ready to use
    for word in result.words:
        print(f"{word.text}: {word.start:.2f}s - {word.end:.2f}s")

    # Save outputs
    result.save_audacity_labels("labels.txt")
    result.save_gentle_html("visualization.html", audio_file="audio.mp3")
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """
    A simple word with start/end timestamps in seconds.

    This is the user-facing class for easy access to alignment results.
    No frame conversion needed - times are in seconds.

    Attributes:
        text: The word text (normalized form used for alignment)
        start: Start time in seconds
        end: End time in seconds
        index: Word index in the original text
        original: Original word form before normalization (if different)
    """
    text: str
    start: float
    end: float
    index: int = -1
    original: Optional[str] = None

    def __repr__(self):
        return f"WordTimestamp('{self.text}', {self.start:.2f}s-{self.end:.2f}s)"

# Configure logging format
logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)


@dataclass
class LongFormAlignmentResult:
    """
    Result of long-form alignment.

    Attributes:
        word_alignments: Dict mapping word index to AlignedWord (internal, frame-based)
        unaligned_indices: List of (start_idx, end_idx) for unaligned text regions
        token_alignments: List of aligned tokens (optional)
        metadata: Additional info (duration, segments, etc.)

    Properties:
        words: List of WordTimestamp objects with times in seconds (recommended for users)
    """
    word_alignments: Dict[int, Any]  # word_idx -> AlignedWord
    unaligned_indices: List[Tuple[int, int]]
    token_alignments: Optional[List[Any]] = None
    text_words: Optional[List[str]] = None
    original_text_words: Optional[List[str]] = None  # Non-normalized words
    metadata: Optional[Dict[str, Any]] = None
    _words_cache: Optional[List[WordTimestamp]] = field(default=None, repr=False)

    @property
    def words(self) -> List[WordTimestamp]:
        """
        Get aligned words as a simple list with times in seconds.

        This is the recommended way to access alignment results.
        Returns WordTimestamp objects sorted by start time.

        Example:
            for word in result.words:
                print(f"{word.text}: {word.start:.2f}s - {word.end:.2f}s")
        """
        if self._words_cache is not None:
            return self._words_cache

        frame_dur = self.frame_duration
        words_list = []

        for idx, aligned_word in sorted(self.word_alignments.items()):
            # Skip None words (end-of-text markers)
            if aligned_word.word is None:
                continue

            # Convert frames to seconds
            start_sec = aligned_word.start_time * frame_dur
            if aligned_word.end_time is not None:
                end_sec = aligned_word.end_time * frame_dur
            else:
                end_sec = start_sec + 0.5  # Fallback (should rarely happen)

            # Get original word form if available
            original = None
            if self.original_text_words and idx < len(self.original_text_words):
                orig_word = self.original_text_words[idx]
                if orig_word.lower() != aligned_word.word.lower():
                    original = orig_word

            words_list.append(WordTimestamp(
                text=aligned_word.word,
                start=start_sec,
                end=end_sec,
                index=idx,
                original=original,
            ))

        # Cache and return
        object.__setattr__(self, '_words_cache', words_list)
        return words_list

    @property
    def num_aligned_words(self) -> int:
        return len(self.word_alignments)

    @property
    def num_unaligned_regions(self) -> int:
        return len(self.unaligned_indices)

    def get_word(self, word_idx: int) -> Optional[Any]:
        """Get aligned word by index."""
        return self.word_alignments.get(word_idx)

    def get_words_in_range(self, start_sec: float, end_sec: float) -> List[WordTimestamp]:
        """Get all aligned words within a time range (in seconds)."""
        return [w for w in self.words if start_sec <= w.start <= end_sec]

    def to_audacity_labels(self) -> str:
        """Export to Audacity label format (times in seconds)."""
        lines = []
        for word in self.words:
            lines.append(f"{word.start:.6f}\t{word.end:.6f}\t{word.text}")
        return "\n".join(lines)

    def save_audacity_labels(self, output_path: Union[str, Path]) -> str:
        """Save alignment as Audacity label file."""
        output_path = Path(output_path)
        labels = self.to_audacity_labels()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(labels)
        return str(output_path)

    def to_gentle_html(
        self,
        audio_file: Optional[Union[str, Path]] = None,
        title: str = "TorchAudio Aligner - Alignment Visualization",
    ) -> str:
        """Generate Gentle-style HTML visualization."""
        from visualization_utils.gentle import get_gentle_visualization_from_words

        return get_gentle_visualization_from_words(
            self.words,
            audio_file=audio_file,
            title=title,
        )

    def save_gentle_html(
        self,
        output_path: Union[str, Path],
        audio_file: Optional[Union[str, Path]] = None,
        title: str = "TorchAudio Aligner - Alignment Visualization",
    ) -> str:
        """Save Gentle-style HTML visualization to file."""
        output_path = Path(output_path)
        html = self.to_gentle_html(audio_file, title)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return str(output_path)

    @property
    def frame_duration(self) -> float:
        """Get frame duration from metadata, default 0.02 (20ms)."""
        if self.metadata and "frame_duration" in self.metadata:
            return self.metadata["frame_duration"]
        return 0.02

    def coverage_percent(self) -> float:
        """Get percentage of words aligned."""
        if not self.text_words:
            return 0.0
        return 100.0 * len(self.word_alignments) / len(self.text_words)

    def summary(self) -> str:
        """Get a summary string of the alignment result."""
        lines = [
            f"Alignment Summary:",
            f"  Aligned words: {self.num_aligned_words}",
            f"  Unaligned regions: {self.num_unaligned_regions}",
        ]
        if self.text_words:
            lines.append(f"  Total words: {len(self.text_words)}")
            lines.append(f"  Coverage: {self.coverage_percent():.1f}%")
        if self.metadata:
            if "audio_duration" in self.metadata:
                lines.append(f"  Audio duration: {self.metadata['audio_duration']:.1f}s")
            if "num_segments" in self.metadata:
                lines.append(f"  Segments: {self.metadata['num_segments']}")
        return "\n".join(lines)


def align_long_audio(
    audio: Union[str, Path, torch.Tensor],
    text: Union[str, Path],
    language: str = "eng",
    model: Optional[Any] = None,
    model_name: str = "mms-fa",
    # Audio parameters
    segment_size: float = 15.0,
    overlap: float = 2.0,
    target_sample_rate: int = 16000,
    # Alignment parameters
    skip_penalty: float = -0.5,
    return_penalty: float = -18.0,
    batch_size: int = 32,
    # Text parameters
    expand_numbers: bool = True,
    romanize: bool = False,
    romanize_language: Optional[str] = None,
    # Device
    device: Optional[str] = None,
    # Options
    verbose: bool = True,
) -> LongFormAlignmentResult:
    """
    Align long-form audio with potentially noisy text.

    This is the main entry point for the library. It handles:
    1. Audio loading and segmentation
    2. Text loading and normalization
    3. Acoustic model inference
    4. WFST-based flexible alignment
    5. LIS-based segment stitching

    Args:
        audio: Path to audio file or waveform tensor
        text: Path to text file (txt/pdf/url) or text string
        language: ISO 639-3 language code (default: "eng")
        model: Pre-loaded model backend (optional)
        model_name: Model to load if model not provided
        segment_size: Segment size in seconds
        overlap: Overlap between segments in seconds
        target_sample_rate: Target sample rate for audio
        skip_penalty: Penalty for skipping words in WFST
        return_penalty: Penalty for returning to previous words
        batch_size: Batch size for model inference
        expand_numbers: Expand numbers to spoken form
        romanize: Romanize non-Latin text
        romanize_language: Language code for romanization
        device: Device for inference ('cuda' or 'cpu')
        verbose: Print progress messages

    Returns:
        LongFormAlignmentResult with word alignments and metadata
    """
    import torchaudio

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if verbose:
        logger.info(f"Device: {device}")

    # =========================================================================
    # Step 1: Load and process audio
    # =========================================================================
    if verbose:
        logger.info("Step 1: Loading and processing audio...")

    from audio_frontend import AudioFrontend, SegmentationResult

    audio_frontend = AudioFrontend(
        target_sample_rate=target_sample_rate,
        mono=True,
        normalize=False,
    )

    if isinstance(audio, (str, Path)):
        waveform, orig_sr = audio_frontend.load(str(audio))
        waveform = audio_frontend.resample(waveform, orig_sr)
    else:
        waveform = audio
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

    waveform = audio_frontend.to_mono(waveform)

    # Segment the audio
    segmentation = audio_frontend.segment(
        waveform,
        sample_rate=target_sample_rate,
        segment_size=segment_size,
        overlap=overlap,
    )

    if verbose:
        logger.info(f"  Duration: {segmentation.original_duration_seconds:.2f}s")
        logger.info(f"  Segments: {segmentation.num_segments}")

    # =========================================================================
    # Step 2: Load and process text
    # =========================================================================
    if verbose:
        logger.info("Step 2: Loading and processing text...")

    from text_frontend import (
        load_text_from_file,
        load_text_from_url,
        load_text_from_pdf,
        normalize_for_mms,
        romanize_text,
        CharTokenizer,
    )

    # Load text
    if isinstance(text, Path):
        text = str(text)

    if isinstance(text, str):
        if text.startswith("http"):
            text_content = load_text_from_url(text)
        elif text.endswith(".pdf"):
            text_content = load_text_from_pdf(text)
        elif Path(text).exists():
            text_content = load_text_from_file(text)
        else:
            text_content = text  # Assume it's the text itself
    else:
        text_content = text

    # Normalize text
    if romanize and romanize_language:
        text_content = romanize_text(text_content, language=romanize_language)

    # Keep original words before normalization for display
    original_text_words = text_content.split()

    text_normalized = normalize_for_mms(
        text_content,
        expand_numbers=expand_numbers,
    )

    text_words = text_normalized.split()

    if verbose:
        logger.info(f"  Words: {len(text_words)}")
        logger.info(f"  Preview: {' '.join(text_words[:10])}...")

    # =========================================================================
    # Step 3: Load acoustic model
    # =========================================================================
    if verbose:
        logger.info("Step 3: Loading acoustic model...")

    if model is None:
        from labeling_utils import load_model
        model = load_model(model_name)

    # Get vocabulary info
    vocab = model.get_vocab_info()

    # Create tokenizer
    from text_frontend import create_tokenizer_from_labels
    tokenizer = create_tokenizer_from_labels(
        tuple(vocab.labels),
        blank_token=vocab.blank_token,
        unk_token=vocab.unk_token,
    )

    # Tokenize text
    text_tokenized = tokenizer.encode(text_normalized)

    if verbose:
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Vocab size: {len(vocab.labels)}")

    # =========================================================================
    # Step 4: Build WFST decoding graph
    # =========================================================================
    if verbose:
        logger.info("Step 4: Building WFST decoding graph...")

    from alignment.wfst.factor_transducer import make_factor_transducer_word_level_index_with_skip

    decoding_graph, word_index_sym_tab, token_sym_tab = \
        make_factor_transducer_word_level_index_with_skip(
            text_tokenized,
            blank_penalty=0,
            skip_penalty=skip_penalty,
            return_penalty=return_penalty,
        )
    decoding_graph = decoding_graph.to(device)

    if verbose:
        logger.info(f"  Nodes: {decoding_graph.shape[0]}, Arcs: {decoding_graph.num_arcs}")

    # =========================================================================
    # Step 5: Align segments (following Tutorial.py pattern)
    # =========================================================================
    if verbose:
        logger.info("Step 5: Aligning segments...")

    from alignment.wfst.k2_utils import align_segments
    from tqdm import tqdm

    waveforms_batched, lengths = segmentation.get_waveforms_batched()
    frame_duration = 0.02  # 20ms for MMS
    offsets = segmentation.get_offsets_in_frames(frame_duration)

    alignment_results = []

    iterator = range(0, segmentation.num_segments, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Aligning")

    for i in iterator:
        batch_waveforms = waveforms_batched[i:i + batch_size].to(device)
        batch_lengths = lengths[i:i + batch_size].to(device)
        batch_offsets = offsets[i:i + batch_size]

        # Get emissions from model
        with torch.inference_mode():
            emissions, emission_lengths = model.get_emissions(batch_waveforms, batch_lengths)

        # Add star dimension if needed (MMS)
        if emissions.size(-1) == len(vocab.labels) - 1:
            star_dim = torch.full(
                (emissions.size(0), emissions.size(1), 1),
                -5.0,
                device=emissions.device,
            )
            emissions = torch.cat((emissions, star_dim), dim=-1)

        # Align segments using the tested API (matches Tutorial.py)
        batch_results = align_segments(
            emissions,
            decoding_graph,
            emission_lengths,
        )

        # Add frame offsets and word indices to tokens
        for aligned_tokens, offset in zip(batch_results, batch_offsets):
            for token in aligned_tokens:
                token.timestamp += offset.item()
                if token.token_id == tokenizer.blk_id:
                    continue
                if token.token_id in word_index_sym_tab:
                    token.attr["wid"] = word_index_sym_tab[token.token_id]
                if token.token_id in token_sym_tab:
                    token.attr["tk"] = token_sym_tab[token.token_id]
            alignment_results.append(aligned_tokens)

    if verbose:
        logger.info(f"  Aligned {len(alignment_results)} segments")

    # =========================================================================
    # Step 6: Concatenate alignments using LIS (following Tutorial.py pattern)
    # =========================================================================
    if verbose:
        logger.info("Step 6: Concatenating alignments with LIS...")

    from alignment.wfst.k2_utils import concat_alignments

    # Use concat_alignments which does LIS, outlier removal, and isolated word removal
    resolved_alignment_results, unaligned_indices = concat_alignments(
        alignment_results,
        neighborhood_size=5,
    )

    if verbose:
        logger.info(f"  Aligned tokens: {len(resolved_alignment_results)}")
        logger.info(f"  Unaligned regions: {len(unaligned_indices) if unaligned_indices else 0}")

    # =========================================================================
    # Step 7: Build word-level alignment (following Tutorial.py pattern)
    # =========================================================================
    if verbose:
        logger.info("Step 7: Building word-level alignment...")

    from alignment.wfst.k2_utils import get_final_word_alignment

    word_alignments = get_final_word_alignment(
        resolved_alignment_results,
        text_normalized,
        tokenizer,
    )

    if verbose:
        logger.info(f"  Aligned words: {len(word_alignments)} / {len(text_words)} ({100*len(word_alignments)/len(text_words):.1f}%)")

    # =========================================================================
    # Step 8: Placeholder for second-pass refinement
    # =========================================================================
    # TODO: Implement second-pass forced alignment for unaligned regions
    # This would re-align the "holes" using standard forced alignment
    # to catch words that were missed in the fuzzy alignment pass.

    if verbose and unaligned_indices:
        logger.info("Step 8: Second-pass refinement (TODO)")
        logger.info(f"  Skipped - {len(unaligned_indices)} unaligned regions")

    # =========================================================================
    # Build result
    # =========================================================================
    metadata = {
        "audio_duration": segmentation.original_duration_seconds,
        "num_segments": segmentation.num_segments,
        "num_words": len(text_words),
        "model": model_name,
        "language": language,
        "frame_duration": frame_duration,
    }

    result = LongFormAlignmentResult(
        word_alignments=word_alignments,
        unaligned_indices=unaligned_indices if unaligned_indices else [],
        token_alignments=resolved_alignment_results,
        text_words=text_words,
        original_text_words=original_text_words,
        metadata=metadata,
    )

    if verbose:
        logger.info("Alignment complete!")
        logger.info(f"  Result: {result.num_aligned_words} words aligned")

    return result


def second_pass_refinement(
    result: LongFormAlignmentResult,
    waveform: torch.Tensor,
    model: Any,
    sample_rate: int = 16000,
    frame_duration: float = 0.02,
) -> LongFormAlignmentResult:
    """
    Placeholder for second-pass refinement.

    This would re-align unaligned regions using standard forced alignment
    to improve recall for missed words.

    TODO: Implement this following Tutorial.py's second_pass_fa approach.
    """
    # TODO: Implement second-pass forced alignment for unaligned regions
    logger.warning("second_pass_refinement is not yet implemented")
    return result


# =============================================================================
# Fluent/Builder API: Aligner class
# =============================================================================

class Aligner:
    """
    Fluent API for long-form audio alignment.

    This provides a stateful interface for alignment, allowing you to
    configure the aligner once and run multiple alignments.

    Example:
        >>> aligner = Aligner(language="eng")
        >>> result = aligner.align(audio="file.mp3", text="transcript.pdf")
        >>> print(result.summary())

        # Or step by step:
        >>> aligner = Aligner(language="eng")
        >>> aligner.load_audio("file.mp3")
        >>> aligner.load_text("transcript.pdf")
        >>> result = aligner.run()

        # Reuse for another file:
        >>> aligner.load_audio("another.mp3")
        >>> result2 = aligner.run()
    """

    def __init__(
        self,
        language: str = "eng",
        model: Optional[str] = "mms-fa",
        device: Optional[str] = None,
        segment_size: float = 15.0,
        overlap: float = 2.0,
        batch_size: int = 8,
        verbose: bool = False,
    ):
        """
        Initialize the Aligner.

        Args:
            language: Language code (e.g., "eng", "deu", "fra")
            model: Model name ("mms-fa" or path to custom model)
            device: Device to use ("cuda", "cpu", or None for auto)
            segment_size: Segment size in seconds
            overlap: Overlap between segments in seconds
            batch_size: Batch size for inference
            verbose: Whether to print progress
        """
        self.language = language
        self.model_name = model
        self.device = device
        self.segment_size = segment_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.verbose = verbose

        # State
        self._model = None
        self._audio_path = None
        self._text_path = None
        self._waveform = None
        self._text = None

    def load_audio(self, audio: Union[str, Path, torch.Tensor]) -> "Aligner":
        """
        Load audio file or waveform.

        Args:
            audio: Path to audio file or waveform tensor

        Returns:
            self (for chaining)
        """
        if isinstance(audio, (str, Path)):
            self._audio_path = str(audio)
            self._waveform = None  # Will be loaded lazily
        else:
            self._waveform = audio
            self._audio_path = None
        return self

    def load_text(self, text: Union[str, Path]) -> "Aligner":
        """
        Load text from file or string.

        Args:
            text: Path to text/PDF file or text string

        Returns:
            self (for chaining)
        """
        if isinstance(text, Path):
            self._text_path = str(text)
            self._text = None
        elif isinstance(text, str) and (text.endswith('.pdf') or text.endswith('.txt') or '/' in text):
            # Looks like a file path
            self._text_path = text
            self._text = None
        else:
            # Raw text
            self._text = text
            self._text_path = None
        return self

    def align(
        self,
        audio: Optional[Union[str, Path, torch.Tensor]] = None,
        text: Optional[Union[str, Path]] = None,
    ) -> LongFormAlignmentResult:
        """
        Run alignment on audio and text.

        Args:
            audio: Audio file/tensor (optional if already loaded)
            text: Text file/string (optional if already loaded)

        Returns:
            LongFormAlignmentResult
        """
        if audio is not None:
            self.load_audio(audio)
        if text is not None:
            self.load_text(text)

        return self.run()

    def run(self) -> LongFormAlignmentResult:
        """
        Run alignment with currently loaded audio and text.

        Returns:
            LongFormAlignmentResult
        """
        # Determine audio source
        if self._audio_path:
            audio_source = self._audio_path
        elif self._waveform is not None:
            audio_source = self._waveform
        else:
            raise ValueError("No audio loaded. Call load_audio() first or pass audio to align().")

        # Determine text source
        if self._text_path:
            text_source = self._text_path
        elif self._text:
            text_source = self._text
        else:
            raise ValueError("No text loaded. Call load_text() first or pass text to align().")

        # Use the functional API
        return align_long_audio(
            audio=audio_source,
            text=text_source,
            language=self.language,
            model=self._model,
            device=self.device,
            segment_size=self.segment_size,
            overlap=self.overlap,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

    def __repr__(self) -> str:
        return (
            f"Aligner(language='{self.language}', model='{self.model_name}', "
            f"segment_size={self.segment_size}, overlap={self.overlap})"
        )
