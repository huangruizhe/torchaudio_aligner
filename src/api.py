"""
High-level API for TorchAudio Long-Form Aligner.

This module provides a simple, unified interface for forced alignment.

Usage:
    from torchaudio_aligner import align_long_audio

    result = align_long_audio(
        audio="path/to/audio.mp3",
        text="path/to/transcript.pdf",
    )

    # Iterate over aligned words (times in frames, call methods for seconds)
    for word in result:
        print(f"{word.word}: {word.start_seconds():.2f}s - {word.end_seconds():.2f}s")

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
    batch_size: int = 32,
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
        AlignmentResult with aligned words (times in frames, use start_seconds()/end_seconds())

    Example:
        >>> result = align_long_audio("audio.mp3", "transcript.pdf")
        >>> for word in result:
        ...     print(f"{word.word}: {word.start_seconds():.2f}s - {word.end_seconds():.2f}s")
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

    # Get file names for display
    audio_name = str(audio) if isinstance(audio, (str, Path)) else "<tensor>"
    text_name = str(text) if isinstance(text, (str, Path)) else "<string>"

    if verbose:
        logger.info("=" * 60)
        logger.info("TorchAudio Long-Form Aligner")
        logger.info("=" * 60)
        logger.info(f"Audio: {audio_name}")
        logger.info(f"Text: {text_name}")
        logger.info(f"Language: {language}")
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
    duration_min = duration_sec / 60

    if verbose:
        logger.info(f"  Duration: {duration_sec:.1f}s ({duration_min:.1f} min)")

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

    from text_frontend import load_text_from_pdf, load_text_from_pdf_ocr, normalize_for_mms, romanize_text_aligned
    from text_frontend.romanization import preprocess_cjk

    # Map language codes to EasyOCR language codes for OCR fallback
    LANG_TO_OCR = {
        "hin": ["hi", "en"],  # Hindi
        "ara": ["ar", "en"],  # Arabic
        "tha": ["th", "en"],  # Thai
        "jpn": ["ja", "en"], "ja": ["ja", "en"],  # Japanese
        "kor": ["ko", "en"], "ko": ["ko", "en"],  # Korean
        "cmn": ["ch_sim", "en"], "zh": ["ch_sim", "en"],  # Chinese Simplified
        "yue": ["ch_tra", "en"],  # Chinese Traditional (Cantonese)
        "rus": ["ru", "en"],  # Russian
        "ben": ["bn", "en"],  # Bengali
        "tam": ["ta", "en"],  # Tamil
        "tel": ["te", "en"],  # Telugu
        "kan": ["kn", "en"],  # Kannada
        "mal": ["ml", "en"],  # Malayalam
    }

    if isinstance(text, Path):
        text = str(text)

    if isinstance(text, str):
        if text.endswith('.pdf'):
            text_content = load_text_from_pdf(text)
            # Check if PDF extraction returned meaningful text
            if len(text_content.strip()) < 100 or len(text_content.split()) < 20:
                # Try OCR fallback
                ocr_langs = LANG_TO_OCR.get(language, ["en"])
                if verbose:
                    logger.info(f"  PDF text extraction returned minimal content, trying OCR with languages: {ocr_langs}")
                try:
                    text_content = load_text_from_pdf_ocr(text, languages=ocr_langs, fallback_to_text=False)
                except ImportError as e:
                    logger.warning(f"  OCR not available: {e}")
                    logger.warning(f"  Install with: pip install easyocr pdf2image && apt install poppler-utils")
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

    # Determine if language needs special handling
    is_cjk = language in ("cmn", "zh", "jpn", "ja", "kor", "ko", "yue")
    is_non_latin = language in (
        "cmn", "zh", "jpn", "ja", "kor", "ko", "yue",  # CJK
        "hin", "tha", "ara", "heb", "rus", "ukr", "bul",  # Non-Latin
        "ell", "tam", "tel", "kan", "mal", "ben", "guj",  # More scripts
    )

    # CJK: split into individual characters
    if is_cjk:
        text_content = preprocess_cjk(text_content)
        if verbose:
            logger.info(f"  CJK preprocessing applied")

    # Keep original words before normalization
    original_text_words = text_content.split()

    # Auto-romanize for non-Latin scripts (unless explicitly disabled)
    if is_non_latin and not romanize:
        romanize = True
        romanize_language = language
        if verbose:
            logger.info(f"  Auto-romanization enabled for {language}")

    # Normalize
    if romanize and romanize_language:
        text_content = romanize_text_aligned(text_content, language=romanize_language)

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
    # Step 5: Align segments (following Tutorial.py pattern)
    # =========================================================================
    if verbose:
        logger.info("Step 5: Aligning segments...")

    from alignment.wfst.k2_utils import align_segments
    from tqdm import tqdm

    # Pre-batch all waveforms once on CPU (memory efficient pattern from Tutorial.py)
    waveforms_batched, lengths = segmentation.get_waveforms_batched()
    offsets = segmentation.get_offsets_in_frames(frame_duration)

    alignment_results = []

    iterator = range(0, segmentation.num_segments, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Aligning")

    for i in iterator:
        # Only move the current batch slice to GPU
        batch_waveforms = waveforms_batched[i:i + batch_size].to(device)
        batch_lengths = lengths[i:i + batch_size].to(device)
        batch_offsets = offsets[i:i + batch_size]

        # Get emissions from model
        with torch.inference_mode():
            emissions, emission_lengths = model.get_emissions(batch_waveforms, batch_lengths)

        # Add star dimension if needed (MMS model compatibility)
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
    # Step 7: Build word-level alignment (frames-based, call start_seconds()/end_seconds() for seconds)
    # =========================================================================
    if verbose:
        logger.info("Step 7: Building word alignment...")

    from alignment.wfst.k2_utils import get_final_word_alignment

    word_alignment_dict = get_final_word_alignment(
        resolved_tokens,
        text_normalized,
        original_text_words,
        tokenizer,
    )

    # Convert dict to sorted list
    words = [word_alignment_dict[idx] for idx in sorted(word_alignment_dict.keys())]

    if verbose:
        coverage = 100.0 * len(words) / len(text_words) if text_words else 0
        logger.info(f"  Aligned: {len(words)} / {len(text_words)} words ({coverage:.1f}%)")

    # =========================================================================
    # Build result
    # =========================================================================
    # Store audio_file path for methods like create_interactive_demo()
    audio_file_path = str(audio) if isinstance(audio, (str, Path)) else None

    metadata = {
        "audio_duration": segmentation.original_duration_seconds,
        "num_segments": segmentation.num_segments,
        "total_words": len(text_words),
        "model": model_name,
        "language": language,
        "audio_file": audio_file_path,
    }

    result = AlignmentResult(
        words=words,
        chars=[],  # Character-level not built in this path
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
