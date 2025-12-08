"""
WFST/k2-based aligner for long-form fuzzy alignment.

This is the main alignment backend using weighted finite state transducers
with factor transducer for segment-wise alignment.

Architecture note:
- This module produces SEGMENT-WISE alignment results (each segment independent)
- For global alignment, use stitching_utils to combine segments
- The align() method can optionally do stitching for convenience, but
  align_segments() returns raw segment results for use with stitching_utils

Usage:
    # Option 1: Full pipeline (alignment + stitching)
    result = aligner.align(waveform, text)

    # Option 2: Segment-wise only (for use with stitching_utils)
    segment_results = aligner.align_segments(waveform, text)
    # Then use stitching_utils to combine:
    from stitching_utils import stitch_alignments
    final_result = stitch_alignments(segment_results, method="lis")
"""

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import math
import logging

import torch
from tqdm import tqdm

from ..base import (
    AlignerBackend,
    AlignmentConfig,
    AlignmentResult,
    AlignedWord,
    AlignedToken,
)
from .factor_transducer import (
    make_factor_transducer_word_level_index_with_skip,
    flatten_list,
)
from .k2_utils import (
    get_best_paths,
    get_texts_with_timestamp,
    concat_alignments,
    get_final_word_alignment_seconds,
)
from .lis_utils import (
    compute_lis,
    remove_outliers,
    remove_isolated_words,
    find_unaligned_regions,
    get_lis_alignment,
)

# Import from unified frontends (absolute imports for compatibility)
from text_frontend import TokenizerInterface, create_tokenizer_from_labels
from audio_frontend import segment_waveform, SegmentationResult

logger = logging.getLogger(__name__)


@dataclass
class SegmentAlignmentResult:
    """
    Alignment result for a single segment.

    This is the output of segment-wise alignment, before stitching.
    Compatible with stitching_utils.SegmentAlignment.

    Attributes:
        tokens: List of aligned tokens for this segment
        segment_index: Index of this segment (0-based)
        frame_offset: Frame offset from audio start
        rejected: Whether this segment was rejected by heuristics
        score: Alignment score for this segment
        metadata: Additional information
    """
    tokens: List[AlignedToken]
    segment_index: int
    frame_offset: int = 0
    rejected: bool = False
    score: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_word_indices(self) -> List[int]:
        """Get word indices from tokens."""
        return [t.attr["wid"] for t in self.tokens if "wid" in t.attr]

    def __len__(self):
        return len(self.tokens)


class WFSTAligner(AlignerBackend):
    """
    WFST/k2-based aligner for fuzzy long-form alignment.

    This aligner uses:
    - Factor transducer with skip/return arcs for flexible matching
    - Uniform segmentation with overlap for long audio
    - LIS algorithm for robust segment concatenation
    - Two-pass alignment for improved recall

    Features:
    - Handles long audio (hours)
    - Tolerates noisy transcripts (insertions, deletions, substitutions)
    - Supports any CTC model backend
    - Multilingual (1100+ languages with MMS)

    Example:
        >>> from labeling_utils import load_model
        >>> from alignment.wfst import WFSTAligner

        >>> # Load acoustic model
        >>> model_backend = load_model("mms-fa")

        >>> # Create aligner
        >>> config = AlignmentConfig(
        ...     segment_size=15.0,
        ...     overlap=2.0,
        ...     skip_penalty=-0.5,
        ...     return_penalty=-18.0,
        ... )
        >>> aligner = WFSTAligner(config)

        >>> # Set model and tokenizer
        >>> aligner.set_model(model_backend)

        >>> # Align
        >>> result = aligner.align(waveform, text)
    """

    BACKEND_NAME = "wfst"

    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def set_model(self, model_backend, tokenizer: Optional[TokenizerInterface] = None):
        """
        Set the CTC model backend for emission extraction.

        Args:
            model_backend: CTCModelBackend instance from labeling_utils
            tokenizer: Optional tokenizer (auto-created if not provided)
        """
        self._model = model_backend

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            # Auto-create tokenizer from model labels
            vocab = model_backend.get_vocab_info()
            self._tokenizer = create_tokenizer_from_labels(
                tuple(vocab.labels),
                blank_token=vocab.blank_token,
                unk_token=vocab.unk_token,
            )

        self._loaded = True

    def load(self):
        """Load is handled by set_model()."""
        if self._model is None:
            raise RuntimeError(
                "Call set_model() with a CTCModelBackend before using align()"
            )
        self._loaded = True

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            self._model.unload()
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def align_segments(
        self,
        waveform: torch.Tensor,
        text: str,
        **kwargs,
    ) -> List[SegmentAlignmentResult]:
        """
        Align audio to text and return SEGMENT-WISE results (no stitching).

        This is the primary method for use with stitching_utils.
        Each segment is aligned independently; use stitching_utils to combine.

        Args:
            waveform: Audio tensor (1, T) or (T,)
            text: Text to align (can be noisy/imperfect)
            **kwargs: Override config parameters

        Returns:
            List of SegmentAlignmentResult, one per segment

        Example:
            >>> segment_results = aligner.align_segments(waveform, text)
            >>> # Convert to stitching_utils format
            >>> from stitching_utils import SegmentAlignment, stitch_alignments
            >>> stitch_input = [
            ...     SegmentAlignment(
            ...         tokens=seg.tokens,
            ...         segment_index=seg.segment_index,
            ...         frame_offset=seg.frame_offset,
            ...         rejected=seg.rejected,
            ...     )
            ...     for seg in segment_results
            ... ]
            >>> final = stitch_alignments(stitch_input, method="lis")
        """
        if not self._loaded:
            raise RuntimeError("Aligner not loaded. Call load() or set_model() first.")

        # Ensure proper shape: (1, T, 1)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(-1)

        config = self.config
        device = config.device

        logger.info(f"Input waveform shape: {waveform.shape}")

        # Step 0: Tokenize text
        logger.info("Step 0: Tokenizing text")
        text_normalized = self._tokenizer.text_normalize(text)
        tokenized_text = self._tokenizer.encode(text_normalized)
        logger.info(
            f"Tokenized: {len(tokenized_text)} words, "
            f"{sum(len(w) for w in tokenized_text)} tokens"
        )

        # Step 1: Build decoding graph
        logger.info("Step 1: Building decoding graph")
        decoding_graph, word_index_sym_tab, token_sym_tab = \
            make_factor_transducer_word_level_index_with_skip(
                tokenized_text,
                blank_penalty=config.blank_penalty,
                skip_penalty=config.skip_penalty,
                return_penalty=config.return_penalty,
            )
        decoding_graph = decoding_graph.to(device)
        logger.info(
            f"Graph: {decoding_graph.shape[0]} states, "
            f"{decoding_graph.num_arcs} arcs"
        )

        # Step 2: Segment audio using audio_frontend
        logger.info("Step 2: Segmenting audio")
        waveform_2d = waveform.squeeze(-1) if waveform.dim() == 3 else waveform

        segmentation_result = segment_waveform(
            waveform_2d,
            sample_rate=config.sample_rate,
            segment_size=config.segment_size,
            overlap=config.overlap,
            min_segment_size=config.shortest_segment_size,
        )

        segments, segment_lengths = segmentation_result.get_waveforms_batched()
        segment_offsets = torch.tensor([seg.offset_samples for seg in segmentation_result.segments])

        logger.info(
            f"Created {segments.shape[0]} segments of {config.segment_size}s"
        )

        # Step 3: Align segments in batches
        logger.info("Step 3: Aligning segments (independently)")
        output_frames_offset = segment_offsets // (
            config.sample_rate * config.frame_duration
        )

        segment_results = []
        segment_idx = 0

        for i in tqdm(range(0, segments.size(0), config.batch_size)):
            batch_segments = segments[i:i + config.batch_size].to(device)
            batch_lengths = segment_lengths[i:i + config.batch_size]
            batch_offsets = output_frames_offset[i:i + config.batch_size]

            # Get emissions from model
            with torch.inference_mode():
                batch_emissions, batch_emission_lengths = self._model.get_emissions(
                    batch_segments,
                    batch_lengths.to(device),
                )

            # Attach star dimension if needed
            vocab = self._model.get_vocab_info()
            if vocab.unk_id is None or self.config.extra_options.get("add_star", True):
                star_dim = torch.full(
                    (batch_emissions.size(0), batch_emissions.size(1), 1),
                    -5.0,
                    device=batch_emissions.device,
                    dtype=batch_emissions.dtype,
                )
                batch_emissions = torch.cat([batch_emissions, star_dim], dim=2)

            # Align batch - returns (tokens, scores, rejected)
            batch_results, batch_scores, batch_rejected = self._align_segments_batch(
                batch_emissions,
                decoding_graph,
                batch_emission_lengths,
            )

            # Build SegmentAlignmentResult for each segment
            for aligned_tokens, offset, score, rejected in zip(
                batch_results, batch_offsets, batch_scores, batch_rejected
            ):
                # Add offsets and symbol mappings to tokens
                for token in aligned_tokens:
                    token.timestamp += offset.item()
                    if token.token_id == self._tokenizer.blk_id:
                        continue
                    if token.token_id in word_index_sym_tab:
                        token.attr["wid"] = word_index_sym_tab[token.token_id]
                    if token.token_id in token_sym_tab:
                        token.attr["tk"] = token_sym_tab[token.token_id]

                segment_results.append(SegmentAlignmentResult(
                    tokens=aligned_tokens,
                    segment_index=segment_idx,
                    frame_offset=offset.item(),
                    rejected=rejected,
                    score=score,
                    metadata={
                        "num_tokens": len(aligned_tokens),
                        "word_indices": [t.attr["wid"] for t in aligned_tokens if "wid" in t.attr],
                    },
                ))
                segment_idx += 1

        logger.info(
            f"Segment-wise alignment complete: {len(segment_results)} segments, "
            f"{sum(1 for s in segment_results if not s.rejected)} accepted"
        )

        return segment_results

    def align(
        self,
        waveform: torch.Tensor,
        text: str,
        stitch: bool = True,
        **kwargs,
    ) -> AlignmentResult:
        """
        Align audio to text using WFST-based alignment.

        Args:
            waveform: Audio tensor (1, T) or (T,)
            text: Text to align (can be noisy/imperfect)
            stitch: Whether to stitch segments (default True for backward compat)
            **kwargs: Override config parameters

        Returns:
            AlignmentResult with word-level alignments

        Note:
            For more control over stitching, use align_segments() and
            then stitching_utils separately.
        """
        # Get segment-wise results
        segment_results = self.align_segments(waveform, text, **kwargs)

        if not stitch:
            # Return without stitching - just combine all tokens
            all_tokens = []
            for seg in segment_results:
                if not seg.rejected:
                    all_tokens.extend(seg.tokens)

            return AlignmentResult(
                word_alignments={},
                unaligned_indices=[],
                token_alignments=all_tokens,
                metadata={
                    "backend": self.BACKEND_NAME,
                    "num_segments": len(segment_results),
                    "stitched": False,
                },
            )

        # Convert to list format for _concat_alignments
        alignment_results = [
            seg.tokens if not seg.rejected else []
            for seg in segment_results
        ]

        config = self.config

        # Step 4: Concatenate alignments using LIS
        logger.info("Step 4: Concatenating alignments with LIS")
        resolved_results, unaligned_indices = self._concat_alignments(
            alignment_results,
            neighborhood_size=config.neighborhood_size,
        )
        logger.info(
            f"Aligned {len(resolved_results)} tokens, "
            f"{len(unaligned_indices)} unaligned regions"
        )

        # Step 5: Build word alignments using the existing working function
        logger.info("Step 5: Building word alignments")
        if not resolved_results:
            return AlignmentResult(
                word_alignments={},
                unaligned_indices=unaligned_indices,
                token_alignments=[],
                metadata={"backend": self.BACKEND_NAME},
            )

        # Use the existing tested function from k2_utils
        text_normalized = self._tokenizer.text_normalize(text)
        original_text_words = text.split()

        words, chars = get_final_word_alignment_seconds(
            resolved_results,
            text_normalized,
            original_text_words,
            self._tokenizer,
            frame_duration=config.frame_duration,
        )

        # Convert list to dict for AlignmentResult compatibility
        word_alignment = {word.index: word for word in words}
        logger.info(f"Aligned {len(word_alignment)} words")

        return AlignmentResult(
            word_alignments=word_alignment,
            unaligned_indices=unaligned_indices,
            token_alignments=resolved_results,
            metadata={
                "backend": self.BACKEND_NAME,
                "num_segments": len(segment_results),
                "stitched": True,
                "config": {
                    "skip_penalty": config.skip_penalty,
                    "return_penalty": config.return_penalty,
                },
            },
        )

    def _align_segments_batch(
        self,
        emissions: torch.Tensor,
        decoding_graph,
        segment_lengths: torch.Tensor,
    ) -> Tuple[List[List[AlignedToken]], List[float], List[bool]]:
        """
        Align a batch of segments using k2.

        Args:
            emissions: Log posteriors (L, T, C)
            decoding_graph: k2.Fsa
            segment_lengths: Tensor of frame lengths

        Returns:
            Tuple of:
            - List of AlignedToken lists (one per segment)
            - List of segment scores
            - List of rejection flags
        """
        config = self.config
        device = decoding_graph.device if hasattr(decoding_graph, 'device') else \
            (decoding_graph[0].device if isinstance(decoding_graph, list) else 'cpu')

        emissions = emissions.to(device)

        # Get best paths
        best_paths = get_best_paths(emissions, segment_lengths, decoding_graph)
        best_paths = best_paths.detach().to('cpu')

        # Get skip/return IDs
        if isinstance(decoding_graph, list):
            skip_id = decoding_graph[0].skip_id
            return_id = decoding_graph[0].return_id
        else:
            skip_id = decoding_graph.skip_id
            return_id = decoding_graph.return_id

        # Extract results
        decoding_results = get_texts_with_timestamp(
            best_paths,
            skip_id=skip_id,
            return_id=return_id,
        )
        hyps = decoding_results["hyps"]
        timestamps = decoding_results["timestamps"]
        conf_scores = decoding_results["conf_scores"]

        # Compute segment scores
        segment_scores = best_paths.get_tot_scores(
            use_double_scores=True, log_semiring=True
        )
        segment_scores_per_frame = segment_scores / segment_lengths.cpu()

        # Rejection heuristics
        condition1 = (
            segment_scores_per_frame <
            math.log(config.per_frame_score_threshold)
        ).tolist()
        condition2 = [
            len([_ for _ in h if _ == skip_id]) >
            config.skip_percentage_threshold * len(h)
            for h in hyps
        ]
        condition3 = [
            len([_ for _ in h if _ > skip_id]) >= config.return_arcs_num_threshold
            for h in hyps
        ]
        reject_segments = [
            c1 or c2 or c3
            for c1, c2, c3 in zip(condition1, condition2, condition3)
        ]

        # Remove skip/return IDs from results
        timestamps = [
            [ts for tid, ts in zip(h, t) if tid < skip_id]
            for h, t in zip(hyps, timestamps)
        ]
        hyps = [[tid for tid in h if tid < skip_id] for h in hyps]

        # Build results - include ALL tokens even for rejected segments
        # (rejection is handled by the caller)
        segment_results = []
        for hyp, timestamp, score in zip(hyps, timestamps, conf_scores):
            aligned_tokens = []
            for tid, ts, s in zip(hyp, timestamp, score):
                aligned_tokens.append(AlignedToken(tid, ts, s, {}))
            segment_results.append(aligned_tokens)

        return segment_results, segment_scores_per_frame.tolist(), reject_segments

    def _concat_alignments(
        self,
        alignment_results: List[List[AlignedToken]],
        neighborhood_size: int = 5,
    ) -> Tuple[List[AlignedToken], List[Tuple[int, int]]]:
        """
        Concatenate segment alignments using LIS.

        Args:
            alignment_results: List of segment-level alignments
            neighborhood_size: For isolated word removal

        Returns:
            (resolved_tokens, unaligned_indices)
        """
        # Extract word indices
        hyps = [
            [token.attr["wid"] for token in tokens if "wid" in token.attr]
            for tokens in alignment_results
        ]

        # Flatten and compute LIS
        hyp_list = [i for hyp in hyps for i in hyp]
        if not hyp_list:
            return [], []

        lis_results = compute_lis(hyp_list)

        # Post-process LIS
        lis_results = remove_outliers(lis_results)
        if not lis_results:
            return [], []

        lis_results = remove_isolated_words(lis_results, neighborhood_size)
        if not lis_results:
            return [], []

        # Align LIS to original results
        alignment_results = get_lis_alignment(lis_results, alignment_results)

        # Keep only LIS tokens
        resolved_results = []
        for aligned_tokens in alignment_results:
            word_start_flag = False
            for token in aligned_tokens:
                if token.attr.get("lis", False):
                    resolved_results.append(token)
                    word_start_flag = True
                elif "wid" in token.attr:
                    word_start_flag = False
                elif word_start_flag:
                    resolved_results.append(token)

        # Find unaligned regions
        rg_min = lis_results[0]
        rg_max = lis_results[-1]
        set_lis = set(lis_results)
        unaligned_indices = find_unaligned_regions(rg_min, rg_max, set_lis)

        return resolved_results, unaligned_indices
