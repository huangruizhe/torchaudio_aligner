"""
High-level API for TorchAudio Long-Form Aligner.

This module provides a simple, unified interface for forced alignment.

Usage:
    from torchaudio_aligner import align_long_audio

    result = align_long_audio(
        audio="path/to/audio.mp3",
        text="path/to/transcript.pdf",
    )

    # Iterate over aligned words (times in seconds)
    for word in result:
        print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")

    # Export to various formats
    result.save_audacity_labels("labels.txt")
    result.save_srt("subtitles.srt")
    result.save_json("alignment.json")
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from pathlib import Path
import logging
import torch

# Import from base - these are the primary user-facing classes
from alignment.base import AlignmentResult, AlignedWord, AlignedChar, AlignedToken, AlignmentConfig

logger = logging.getLogger(__name__)

# Configure logging format
logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=logfmt)

# Re-export for convenience
__all__ = [
    "align_long_audio",
    "AlignmentResult",
    "AlignedWord",
    "AlignedChar",
    "AlignmentConfig",
]


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
    batch_size: int = 16,  # Reduced from 32 to avoid OOM on smaller GPUs
    # Text parameters
    expand_numbers: bool = True,
    romanize: bool = False,
    romanize_language: Optional[str] = None,
    # Device
    device: Optional[str] = None,
    # Options
    verbose: bool = True,
) -> AlignmentResult:
    """
    Align audio with text transcript.

    This is the main entry point for the library. It handles:
    - Loading audio from file or tensor
    - Loading text from file (PDF, TXT) or string
    - Normalization and tokenization
    - WFST-based flexible alignment
    - LIS-based stitching for long audio

    Args:
        audio: Audio file path or waveform tensor
        text: Text file path (PDF, TXT) or text string
        language: ISO 639-3 language code (default "eng")
        segment_size: Segment size in seconds (default 15.0)
        overlap: Overlap between segments in seconds (default 2.0)
        batch_size: Batch size for inference (default 32)
        verbose: Print progress (default True)

    Returns:
        AlignmentResult with aligned words (times in seconds)

    Example:
        >>> result = align_long_audio("audio.mp3", "transcript.pdf")
        >>> for word in result:
        ...     print(f"{word.word}: {word.start:.2f}s - {word.end:.2f}s")
        >>> result.save_audacity_labels("labels.txt")
    """
    import torchaudio

    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Clear GPU cache before starting (helps when running multiple alignments)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    frame_duration = 0.02  # 20ms frames (standard for MMS-FA)

    if verbose:
        logger.info("=" * 60)
        logger.info("TorchAudio Long-Form Aligner")
        logger.info("=" * 60)
        logger.info(f"Device: {device}")

    # =========================================================================
    # Step 1: Load and preprocess audio
    # =========================================================================
    if verbose:
        logger.info("Step 1: Loading audio...")

    if isinstance(audio, (str, Path)):
        waveform, sample_rate = torchaudio.load(str(audio))
        if verbose:
            logger.info(f"  File: {audio}")
            logger.info(f"  Original: {waveform.shape}, {sample_rate}Hz")
    else:
        waveform = audio
        sample_rate = target_sample_rate
        if verbose:
            logger.info(f"  Tensor: {waveform.shape}")

    # Resample if needed
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    # Convert to mono
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    duration_sec = waveform.size(1) / sample_rate

    if verbose:
        logger.info(f"  Processed: {waveform.shape}, {sample_rate}Hz, {duration_sec:.1f}s")

    # Segment audio
    from audio_frontend import segment_waveform

    segmentation = segment_waveform(
        waveform.squeeze(0),
        sample_rate=sample_rate,
        segment_size=segment_size,
        overlap=overlap,
    )

    if verbose:
        logger.info(f"  Segments: {segmentation.num_segments}")

    # =========================================================================
    # Step 2: Load and preprocess text
    # =========================================================================
    if verbose:
        logger.info("Step 2: Loading text...")

    from text_frontend import load_text_from_pdf, normalize_for_mms, romanize_text

    if isinstance(text, Path):
        text = str(text)

    if isinstance(text, str):
        if text.endswith('.pdf'):
            text_content = load_text_from_pdf(text)
            if verbose:
                logger.info(f"  PDF: {text}")
        elif text.endswith('.txt'):
            with open(text, "r", encoding="utf-8") as f:
                text_content = f.read()
            if verbose:
                logger.info(f"  TXT: {text}")
        elif len(text) > 500 or '\n' in text:
            text_content = text
        else:
            # Could be a short filename that doesn't exist
            try:
                with open(text, "r", encoding="utf-8") as f:
                    text_content = f.read()
                if verbose:
                    logger.info(f"  File: {text}")
            except FileNotFoundError:
                text_content = text
    else:
        text_content = text

    # Keep original words before normalization
    original_text_words = text_content.split()

    # Normalize
    if romanize and romanize_language:
        text_content = romanize_text(text_content, language=romanize_language)

    text_normalized = normalize_for_mms(text_content, expand_numbers=expand_numbers)
    text_words = text_normalized.split()

    if verbose:
        logger.info(f"  Words: {len(text_words)}")

    # =========================================================================
    # Step 3: Load acoustic model
    # =========================================================================
    if verbose:
        logger.info("Step 3: Loading acoustic model...")

    if model is None:
        from labeling_utils import load_model
        model = load_model(model_name)

    vocab = model.get_vocab_info()
    if verbose:
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Vocab: {len(vocab.labels)} tokens")

    # =========================================================================
    # Step 4: Create tokenizer and build WFST
    # =========================================================================
    if verbose:
        logger.info("Step 4: Building decoding graph...")

    from text_frontend import create_tokenizer_from_labels
    from alignment.wfst.factor_transducer import make_factor_transducer_word_level_index_with_skip

    tokenizer = create_tokenizer_from_labels(
        tuple(vocab.labels),
        blank_token=vocab.blank_token,
        unk_token=vocab.unk_token,
    )

    text_tokenized = tokenizer.encode(text_normalized)

    decoding_graph, word_index_sym_tab, token_sym_tab = \
        make_factor_transducer_word_level_index_with_skip(
            text_tokenized,
            blank_penalty=0,
            skip_penalty=skip_penalty,
            return_penalty=return_penalty,
        )

    decoding_graph = decoding_graph.to(device)

    if verbose:
        logger.info(f"  States: {decoding_graph.shape[0]}, Arcs: {decoding_graph.num_arcs}")

    # =========================================================================
    # Step 5: Align segments
    # =========================================================================
    if verbose:
        logger.info("Step 5: Aligning segments...")

    from tqdm import tqdm
    from alignment.wfst.k2_utils import align_segments

    num_segments = segmentation.num_segments
    segment_offsets_frames = segmentation.get_offsets_in_frames(frame_duration)

    alignment_results = []

    for i in tqdm(range(0, num_segments, batch_size), disable=not verbose):
        batch_end = min(i + batch_size, num_segments)
        batch_segments = segmentation.segments[i:batch_end]
        batch_offsets = segment_offsets_frames[i:batch_end]

        # Prepare batch
        max_len = max(seg.waveform.shape[-1] for seg in batch_segments)
        batch_waveforms = torch.zeros(len(batch_segments), max_len)
        batch_lengths = torch.zeros(len(batch_segments), dtype=torch.long)

        for j, seg in enumerate(batch_segments):
            length = seg.waveform.shape[-1]
            batch_waveforms[j, :length] = seg.waveform
            batch_lengths[j] = length

        batch_waveforms = batch_waveforms.to(device)
        batch_lengths = batch_lengths.to(device)

        # Get emissions
        with torch.inference_mode():
            emissions, emission_lengths = model.get_emissions(batch_waveforms, batch_lengths)

        # Align
        batch_results = align_segments(emissions, decoding_graph, emission_lengths)

        # Add offsets and word indices
        for aligned_tokens, offset in zip(batch_results, batch_offsets):
            offset_val = offset.item()
            for token in aligned_tokens:
                token.timestamp += offset_val
                if token.token_id == tokenizer.blk_id:
                    continue
                if token.token_id in word_index_sym_tab:
                    token.attr["wid"] = word_index_sym_tab[token.token_id]
                if token.token_id in token_sym_tab:
                    token.attr["tk"] = token_sym_tab[token.token_id]

        alignment_results.extend(batch_results)

        # Clean up
        del batch_waveforms, batch_lengths, emissions, emission_lengths
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if verbose:
        logger.info(f"  Aligned {len(alignment_results)} segments")

    # =========================================================================
    # Step 6: Concatenate alignments using LIS
    # =========================================================================
    if verbose:
        logger.info("Step 6: Concatenating alignments...")

    from alignment.wfst.k2_utils import concat_alignments

    resolved_tokens, unaligned_indices = concat_alignments(
        alignment_results,
        neighborhood_size=5,
    )

    if verbose:
        logger.info(f"  Tokens: {len(resolved_tokens)}")
        logger.info(f"  Unaligned regions: {len(unaligned_indices) if unaligned_indices else 0}")

    # =========================================================================
    # Step 7: Build word-level alignment (convert to seconds here!)
    # =========================================================================
    if verbose:
        logger.info("Step 7: Building word alignment...")

    from alignment.wfst.k2_utils import get_final_word_alignment_seconds

    words, chars = get_final_word_alignment_seconds(
        resolved_tokens,
        text_normalized,
        original_text_words,
        tokenizer,
        frame_duration=frame_duration,
    )

    if verbose:
        coverage = 100.0 * len(words) / len(text_words) if text_words else 0
        logger.info(f"  Aligned: {len(words)} / {len(text_words)} words ({coverage:.1f}%)")
        logger.info(f"  Characters: {len(chars)}")

    # =========================================================================
    # Build result
    # =========================================================================
    metadata = {
        "audio_duration": segmentation.original_duration_seconds,
        "num_segments": segmentation.num_segments,
        "total_words": len(text_words),
        "total_chars": len(chars),
        "model": model_name,
        "language": language,
    }

    result = AlignmentResult(
        words=words,
        chars=chars,
        unaligned_regions=unaligned_indices if unaligned_indices else [],
        metadata=metadata,
    )

    if verbose:
        logger.info("Alignment complete!")
        logger.info(f"  {len(result)} words aligned")

    return result


# =============================================================================
# Convenience alias for backwards compatibility
# =============================================================================

# Keep LongFormAlignmentResult as alias for backwards compatibility
LongFormAlignmentResult = AlignmentResult
