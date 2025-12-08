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
        language="eng",
    )

    # Access results
    for word_idx, word in result.word_alignments.items():
        print(f"{word.word}: {word.start_seconds:.2f}s - {word.end_seconds:.2f}s")
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import torch

logger = logging.getLogger(__name__)


@dataclass
class LongFormAlignmentResult:
    """
    Result of long-form alignment.

    Attributes:
        word_alignments: Dict mapping word index to AlignedWord
        unaligned_indices: List of (start_idx, end_idx) for unaligned text regions
        token_alignments: List of aligned tokens (optional)
        metadata: Additional info (duration, segments, etc.)
    """
    word_alignments: Dict[int, Any]  # word_idx -> AlignedWord
    unaligned_indices: List[Tuple[int, int]]
    token_alignments: Optional[List[Any]] = None
    text_words: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def num_aligned_words(self) -> int:
        return len(self.word_alignments)

    @property
    def num_unaligned_regions(self) -> int:
        return len(self.unaligned_indices)

    def get_word(self, word_idx: int) -> Optional[Any]:
        """Get aligned word by index."""
        return self.word_alignments.get(word_idx)

    def get_words_in_range(self, start_sec: float, end_sec: float) -> List[Any]:
        """Get all aligned words within a time range."""
        result = []
        for word in self.word_alignments.values():
            if hasattr(word, 'start_seconds'):
                word_start = word.start_seconds
            else:
                word_start = word.start_time * 0.02  # Assume 20ms frame
            if start_sec <= word_start <= end_sec:
                result.append(word)
        return result

    def to_audacity_labels(self, frame_duration: float = 0.02) -> str:
        """Export to Audacity label format."""
        lines = []
        sorted_words = sorted(self.word_alignments.items())
        for i, (idx, word) in enumerate(sorted_words):
            start = word.start_time * frame_duration
            if i + 1 < len(sorted_words):
                end = sorted_words[i + 1][1].start_time * frame_duration
            else:
                end = start + 0.5  # Default duration for last word
            lines.append(f"{start:.6f}\t{end:.6f}\t{word.word}")
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
    # Step 5: Align segments
    # =========================================================================
    if verbose:
        logger.info("Step 5: Aligning segments...")

    from alignment.wfst.k2_utils import get_best_paths, get_texts_with_timestamp
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

        # Get best paths
        lattice = get_best_paths(emissions, decoding_graph, emission_lengths)
        batch_results = get_texts_with_timestamp(lattice)

        # Add frame offsets and word indices
        for tokens, offset in zip(batch_results, batch_offsets):
            for token in tokens:
                token.timestamp += offset.item()
                if token.token_id in word_index_sym_tab:
                    token.attr["wid"] = word_index_sym_tab[token.token_id]
                if token.token_id in token_sym_tab:
                    token.attr["tk"] = token_sym_tab[token.token_id]
            alignment_results.append(tokens)

    if verbose:
        logger.info(f"  Aligned {len(alignment_results)} segments")

    # =========================================================================
    # Step 6: Stitch segments using LIS
    # =========================================================================
    if verbose:
        logger.info("Step 6: Stitching segments with LIS...")

    from stitching_utils.lis import (
        compute_lis,
        remove_outliers,
        remove_isolated_words,
        find_unaligned_regions,
    )

    # Flatten all tokens
    all_tokens = []
    for tokens in alignment_results:
        all_tokens.extend(tokens)

    # Extract word indices
    word_indices = [t.attr.get("wid", -1) for t in all_tokens]

    # Compute LIS
    lis_indices = compute_lis(word_indices)

    # Remove outliers
    lis_indices = remove_outliers(word_indices, lis_indices, neighborhood_size=5)

    # Remove isolated words
    lis_indices = remove_isolated_words(word_indices, lis_indices, min_neighbors=2)

    # Get resolved tokens
    resolved_tokens = [all_tokens[i] for i in lis_indices]

    # Find unaligned regions
    aligned_word_indices = [t.attr["wid"] for t in resolved_tokens if "wid" in t.attr]
    unaligned_indices = find_unaligned_regions(aligned_word_indices, len(text_words))

    if verbose:
        logger.info(f"  Aligned tokens: {len(resolved_tokens)}")
        logger.info(f"  Unaligned regions: {len(unaligned_indices)}")

    # =========================================================================
    # Step 7: Build word-level alignment
    # =========================================================================
    if verbose:
        logger.info("Step 7: Building word-level alignment...")

    from alignment.base import AlignedWord

    word_alignments = {}

    for token in resolved_tokens:
        if "wid" not in token.attr:
            continue

        word_idx = token.attr["wid"]
        if word_idx < 0 or word_idx >= len(text_words):
            continue

        if word_idx not in word_alignments:
            word_alignments[word_idx] = AlignedWord(
                word=text_words[word_idx],
                start_time=token.timestamp,
                end_time=None,
            )
        else:
            # Update end time to include this token
            pass

    # Compute end times
    sorted_indices = sorted(word_alignments.keys())
    for i, idx in enumerate(sorted_indices[:-1]):
        next_idx = sorted_indices[i + 1]
        word_alignments[idx].end_time = word_alignments[next_idx].start_time

    # Last word gets a default duration
    if sorted_indices:
        last_idx = sorted_indices[-1]
        word_alignments[last_idx].end_time = word_alignments[last_idx].start_time + 25  # ~0.5s

    if verbose:
        logger.info(f"  Aligned words: {len(word_alignments)}")

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
        unaligned_indices=unaligned_indices,
        token_alignments=resolved_tokens,
        text_words=text_words,
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
