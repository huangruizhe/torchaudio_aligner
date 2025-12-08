"""
k2 utilities for WFST-based alignment.

Provides functions for:
- Lattice generation
- Best path extraction
- Timestamp and confidence score extraction
"""

from typing import Dict, List, Tuple, Optional
import torch
import logging

logger = logging.getLogger(__name__)


def get_lattice(
    ctc_output: torch.Tensor,
    target_lengths: torch.Tensor,
    decoding_graph,
    search_beam: float = 20,
    output_beam: float = 8,
    min_active_states: int = 30,
    max_active_states: int = 10000,
    subsampling_factor: int = 1,
):
    """
    Create decoding lattice from CTC output and decoding graph.

    Args:
        ctc_output: Neural network output (N, T, C)
        target_lengths: Length of each utterance in frames
        decoding_graph: k2.Fsa decoding graph
        search_beam: Beam for lattice pruning
        output_beam: Beam for output pruning
        min_active_states: Minimum active states per frame
        max_active_states: Maximum active states per frame
        subsampling_factor: Subsampling factor of the model

    Returns:
        lattice: k2.Fsa lattice
        indices: Sort indices for restoring original order
    """
    try:
        import k2
    except ImportError:
        raise ImportError(
            "k2 is required for WFST alignment. "
            "Install with: pip install k2 -f https://k2-fsa.github.io/k2/cpu.html"
        )

    batch_size = ctc_output.size(0)

    # Build supervision segments
    supervision_segments = torch.stack(
        (
            torch.arange(batch_size),
            torch.zeros(batch_size),
            target_lengths.cpu(),
        ),
        1,
    ).to(torch.int32)

    # Sort by length (descending) for efficient batching
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    # Handle list of graphs (one per utterance)
    if isinstance(decoding_graph, list):
        if len(decoding_graph) > 1:
            decoding_graph = [decoding_graph[i] for i in indices.tolist()]
            decoding_graph = k2.create_fsa_vec(decoding_graph)
        else:
            decoding_graph = decoding_graph[0]
        decoding_graph = k2.arc_sort(decoding_graph)
        decoding_graph = decoding_graph.to(ctc_output.device)

    # Create dense FSA vector
    dense_fsa_vec = k2.DenseFsaVec(
        ctc_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    # Intersect with decoding graph
    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice, indices


def get_best_paths(
    ctc_output: torch.Tensor,
    target_lengths: torch.Tensor,
    decoding_graph,
    **kwargs,
):
    """
    Get best alignment paths from CTC output.

    Args:
        ctc_output: Neural network output (N, T, C)
        target_lengths: Length of each utterance in frames
        decoding_graph: k2.Fsa decoding graph
        **kwargs: Additional arguments for get_lattice

    Returns:
        best_paths: k2.Fsa with best paths (in original batch order)
    """
    try:
        import k2
    except ImportError:
        raise ImportError(
            "k2 is required for WFST alignment. "
            "Install with: pip install k2 -f https://k2-fsa.github.io/k2/cpu.html"
        )

    lattice, indices = get_lattice(
        ctc_output, target_lengths, decoding_graph, **kwargs
    )

    # Find shortest paths
    best_paths = k2.shortest_path(lattice, use_double_scores=True)

    # Restore original order
    _indices = {i_old: i_new for i_new, i_old in enumerate(indices.tolist())}
    best_paths = [best_paths[_indices[i]] for i in range(len(_indices))]
    best_paths = k2.create_fsa_vec(best_paths)

    return best_paths


def get_alignments(
    best_paths,
    kind: str,
    return_ragged: bool = False,
):
    """
    Extract labels or aux_labels from best paths.

    Args:
        best_paths: k2.Fsa with best paths
        kind: "labels" or "aux_labels"
        return_ragged: If True, return k2.RaggedTensor

    Returns:
        List of token sequences or k2.RaggedTensor
    """
    try:
        import k2
    except ImportError:
        raise ImportError("k2 required")

    assert kind in ("labels", "aux_labels")

    token_shape = best_paths.arcs.shape().remove_axis(1)
    tokens = k2.RaggedTensor(token_shape, getattr(best_paths, kind).contiguous())

    if return_ragged:
        return tokens
    else:
        tokens = tokens.remove_values_eq(-1)
        return tokens.tolist()


def get_texts_with_timestamp(
    best_paths,
    skip_id: int,
    return_id: int,
) -> Dict[str, List]:
    """
    Extract texts, timestamps, and confidence scores from best paths.

    This function parses the best alignment paths to extract:
    - Word indices (hyps)
    - Frame timestamps for each word
    - Confidence scores (mean pooled log probabilities)

    Args:
        best_paths: k2.Fsa with best alignment paths
        skip_id: Skip symbol ID
        return_id: Return symbol ID

    Returns:
        Dict with keys:
        - "hyps": Word indices (aux_labels with special symbols removed)
        - "timestamps": Frame indices for each word
        - "conf_scores": Confidence scores for each word
        - "all_labels": Raw labels (includes repeats)
        - "all_aux_labels": Raw aux_labels
        - "all_conf_scores": Frame-level scores
    """
    try:
        import k2
    except ImportError:
        raise ImportError("k2 required")

    assert skip_id < return_id

    # Get labels
    labels = get_alignments(best_paths, kind="labels", return_ragged=False)

    # Get aux labels
    all_aux_labels = get_alignments(best_paths, kind="aux_labels", return_ragged=True)
    all_aux_labels = all_aux_labels.remove_values_eq(-1)
    aux_labels = all_aux_labels.remove_values_leq(0)
    assert aux_labels.num_axes == 2
    all_aux_labels = all_aux_labels.tolist()
    aux_labels = aux_labels.tolist()

    # Get timestamps
    timestamps = []
    for l in all_aux_labels:
        time = [i for i, v in enumerate(l) if v > 0]
        timestamps.append(time)

    # Get confidence scores
    token_shape = best_paths.arcs.shape().remove_axis(1)
    all_conf_scores = k2.RaggedTensor(
        token_shape, best_paths.scores.contiguous()
    ).tolist()
    all_conf_scores = [l[:-1] for l in all_conf_scores]

    # Pool scores for each aux_label
    conf_scores = []
    pooling_func = lambda ll: sum(ll) / len(ll)  # Mean pooling

    for i, (_labels, _aux_labels, _scores) in enumerate(
        zip(labels, all_aux_labels, all_conf_scores)
    ):
        _conf_scores = []
        cur_token_scores = None
        skip_return_cnt = 0

        for l, l_aux, s in zip(_labels, _aux_labels, _scores):
            if l_aux == 0:
                continue
            if 0 < l_aux < skip_id:
                if cur_token_scores is not None:
                    assert len(cur_token_scores) > 0
                    _conf_scores.append(pooling_func(cur_token_scores))
                cur_token_scores = []
            if l_aux >= skip_id:
                skip_return_cnt += 1
            assert cur_token_scores is not None
            cur_token_scores.append(s)

        if cur_token_scores is not None:
            assert len(cur_token_scores) > 0
            _conf_scores.append(pooling_func(cur_token_scores))

        assert len(_conf_scores) == len(aux_labels[i]) - skip_return_cnt
        conf_scores.append(_conf_scores)

    return {
        "timestamps": timestamps,
        "hyps": aux_labels,
        "conf_scores": conf_scores,
        "all_labels": labels,
        "all_aux_labels": all_aux_labels,
        "all_conf_scores": all_conf_scores,
    }


# =============================================================================
# Data classes for alignment results (matching Tutorial.py)
# =============================================================================

from dataclasses import dataclass, field
from typing import Union


@dataclass
class AlignedToken:
    """
    A single aligned token from the alignment path.

    Attributes:
        token_id: Token ID from the decoding graph
        timestamp: Frame index in the audio
        attr: Dictionary of attributes (wid=word index, tk=token symbol, lis=in LIS)
        score: Confidence score
    """
    token_id: Union[str, int]
    timestamp: int
    attr: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class AlignedWord:
    """
    A word-level alignment result.

    Attributes:
        word: The word text
        start_time: Start frame index
        end_time: End frame index (or None)
        phones: List of phone-level alignments
    """
    word: str
    start_time: int
    end_time: Optional[int] = None
    phones: List = field(default_factory=list)


# =============================================================================
# Segment alignment (following Tutorial.py align_segments)
# =============================================================================

import math


def align_segments(
    emissions: torch.Tensor,
    decoding_graph,
    segment_lengths: torch.Tensor,
    per_frame_score_threshold: float = 0.5,
    skip_percentage_threshold: float = 0.2,
    return_arcs_num_threshold: int = 3,
) -> List[List[AlignedToken]]:
    """
    Align segments using k2 library (follows Tutorial.py pattern).

    This function does alignment for a batch of segments using k2's
    best path algorithm with the provided decoding graph.

    The function includes heuristics to reject unreliable alignments:
    - Low per-frame confidence score
    - Too many skip tokens (indicating misalignment)
    - Too many return arcs (indicating backtracking)

    Args:
        emissions: Neural network output (N, T, C)
        decoding_graph: k2.Fsa decoding graph with skip_id/return_id attributes
        segment_lengths: Length of each segment in frames
        per_frame_score_threshold: Threshold for per-frame confidence (default 0.5)
        skip_percentage_threshold: Max fraction of skip tokens allowed (default 0.2)
        return_arcs_num_threshold: Max number of return arcs allowed (default 3)

    Returns:
        List of AlignedToken lists, one per segment. Empty list for rejected segments.

    Raises:
        ImportError: If k2 is not installed
        AttributeError: If decoding_graph lacks skip_id/return_id attributes
    """
    try:
        import k2
    except ImportError:
        raise ImportError(
            "k2 required for alignment. "
            "Install with: pip install k2 -f https://k2-fsa.github.io/k2/cpu.html"
        )

    # Use the graph's device
    if isinstance(decoding_graph, list):
        device = decoding_graph[0].device
    else:
        device = decoding_graph.device
    emissions = emissions.to(device)

    # Get best paths
    best_paths = get_best_paths(emissions, segment_lengths, decoding_graph)
    best_paths = best_paths.detach().to('cpu')

    # Get skip/return IDs from the graph
    try:
        if isinstance(decoding_graph, list):
            skip_id = decoding_graph[0].skip_id
            return_id = decoding_graph[0].return_id
        else:
            skip_id = decoding_graph.skip_id
            return_id = decoding_graph.return_id
    except AttributeError:
        raise AttributeError(
            "decoding_graph must have 'skip_id' and 'return_id' attributes. "
            "Use make_factor_transducer_word_level_index_with_skip() to create it."
        )

    # Validate skip_id < return_id (required for filtering logic)
    if skip_id >= return_id:
        raise ValueError(f"skip_id ({skip_id}) must be less than return_id ({return_id})")

    # Extract alignment results
    decoding_results = get_texts_with_timestamp(
        best_paths,
        skip_id=skip_id,
        return_id=return_id,
    )
    hyps = decoding_results["hyps"]
    timestamps = decoding_results["timestamps"]
    conf_scores = decoding_results["conf_scores"]

    # Get segment-level confidence scores
    segment_scores = best_paths.get_tot_scores(use_double_scores=True, log_semiring=True)
    segment_scores_per_frame = segment_scores / segment_lengths.cpu()

    # Apply heuristics to reject unreliable segments
    # Condition 1: Low per-frame score (poor acoustic match)
    condition1 = (segment_scores_per_frame < math.log(per_frame_score_threshold)).tolist()

    # Condition 2: Too many skip tokens (text doesn't match audio)
    # Guard against division by zero
    condition2 = []
    for i in range(len(hyps)):
        if len(hyps[i]) == 0:
            condition2.append(False)  # Empty hypothesis - don't reject on this condition
        else:
            skip_count = sum(1 for tid in hyps[i] if tid == skip_id)
            condition2.append(skip_count > skip_percentage_threshold * len(hyps[i]))

    # Condition 3: Too many return arcs (excessive backtracking)
    condition3 = [
        sum(1 for tid in hyps[i] if tid > skip_id) >= return_arcs_num_threshold
        for i in range(len(hyps))
    ]

    reject_segments = [c1 or c2 or c3 for c1, c2, c3 in zip(condition1, condition2, condition3)]

    # Log rejection statistics for debugging
    num_rejected = sum(reject_segments)
    if num_rejected > 0:
        logger.debug(f"Rejected {num_rejected}/{len(reject_segments)} segments")

    # Remove skip/return IDs from the symbols (keep only actual tokens)
    timestamps = [
        [ts for tid, ts in zip(hyp, timestamp) if tid < skip_id]
        for hyp, timestamp in zip(hyps, timestamps)
    ]
    hyps = [[tid for tid in hyp if tid < skip_id] for hyp in hyps]

    # Build segment results
    segment_results = []
    for i, (hyp, timestamp, score, reject) in enumerate(zip(hyps, timestamps, conf_scores, reject_segments)):
        # Validate consistency
        if len(hyp) != len(timestamp):
            logger.warning(f"Segment {i}: hyp length ({len(hyp)}) != timestamp length ({len(timestamp)})")
            segment_results.append([])
            continue

        aligned_tokens = []
        if not reject:
            # Ensure we have matching scores (may be truncated due to skip removal)
            scores_to_use = score[:len(hyp)] if len(score) >= len(hyp) else score + [0.0] * (len(hyp) - len(score))
            for tid, ts, s in zip(hyp, timestamp, scores_to_use):
                aligned_tokens.append(AlignedToken(tid, ts, {}, s))

        segment_results.append(aligned_tokens)

    return segment_results


# =============================================================================
# Alignment concatenation (following Tutorial.py concat_alignments)
# =============================================================================

def concat_alignments(
    alignment_results: List[List[AlignedToken]],
    neighborhood_size: int = 5,
    neighbor_threshold: float = 0.4,
    outlier_scan_range: int = 100,
    outlier_threshold: int = 60,
) -> Tuple[List[AlignedToken], Optional[List[Tuple[int, int]]]]:
    """
    Concatenate segment alignments using LIS algorithm (follows Tutorial.py pattern).

    This finds the longest increasing subsequence of word indices
    across all segments, removes outliers and isolated words, and
    returns the resolved alignment.

    The algorithm:
    1. Extract word indices from all aligned tokens
    2. Compute LIS to find monotonically increasing alignment
    3. Remove outliers (words with large gaps from neighbors)
    4. Remove isolated words (words without sufficient aligned neighbors)
    5. Mark LIS tokens and collect resolved alignment

    Args:
        alignment_results: List of AlignedToken lists from align_segments
        neighborhood_size: Window size for isolated word detection (default 5)
        neighbor_threshold: Min fraction of neighbors that must be aligned (default 0.4)
        outlier_scan_range: How far to scan for outliers at boundaries (default 100)
        outlier_threshold: Gap size to consider as outlier (default 60)

    Returns:
        Tuple of (resolved_alignment_results, unaligned_text_indices)
        - resolved_alignment_results: List of AlignedToken in LIS order
        - unaligned_text_indices: List of (start, end) tuples for unaligned regions,
          or None if alignment failed completely
    """
    from .lis_utils import (
        compute_lis,
        remove_outliers,
        find_unaligned_regions,
        get_lis_alignment,
    )

    # Extract word indices from all segments
    hyps = [
        [token.attr["wid"] for token in aligned_tokens if "wid" in token.attr]
        for aligned_tokens in alignment_results
    ]

    # Flatten to single list
    hyp_list = [i for hyp in hyps for i in hyp]

    if not hyp_list:
        logger.warning("No word indices found in alignment results")
        return [], None

    # Compute LIS
    lis_results = compute_lis(hyp_list)
    logger.debug(f"LIS computed: {len(lis_results)} words from {len(hyp_list)} total")

    # Post-process: remove outliers at boundaries
    lis_results = remove_outliers(lis_results, scan_range=outlier_scan_range, outlier_threshold=outlier_threshold)
    if not lis_results:
        logger.warning("All LIS results removed as outliers")
        return [], None

    # Post-process: remove isolated words (insufficient aligned neighbors)
    rg_min = lis_results[0]
    rg_max = lis_results[-1]
    set_lis_results = set(lis_results)
    removed_isolated = 0

    for i in range(rg_min, rg_max + 1):
        if i in set_lis_results:
            left_neighbors = [j for j in range(i - neighborhood_size, i) if j in set_lis_results]
            right_neighbors = [j for j in range(i + 1, i + neighborhood_size + 1) if j in set_lis_results]
            num_left = i - max(i - neighborhood_size, rg_min)
            num_right = min(i + neighborhood_size, rg_max) - i

            # Remove if isolated on both sides
            left_sparse = num_left > 0 and len(left_neighbors) < neighbor_threshold * num_left
            right_sparse = num_right > 0 and len(right_neighbors) < neighbor_threshold * num_right
            if left_sparse and right_sparse:
                set_lis_results.remove(i)
                removed_isolated += 1

    if removed_isolated > 0:
        logger.debug(f"Removed {removed_isolated} isolated words")

    lis_results = [i for i in lis_results if i in set_lis_results]

    if not lis_results:
        logger.warning("All LIS results removed as isolated")
        return [], None

    # Align LIS results to original tokens (marks tokens with "lis" attribute)
    alignment_results = get_lis_alignment(lis_results, alignment_results)

    # Keep only LIS tokens (and continuation tokens within words)
    resolved_alignment_results = []
    for aligned_tokens in alignment_results:
        word_start_flag = False
        for token in aligned_tokens:
            if token.attr.get("lis", False):
                resolved_alignment_results.append(token)
                word_start_flag = True
            elif "wid" in token.attr:
                word_start_flag = False
            elif word_start_flag:
                # Continuation token within a word
                resolved_alignment_results.append(token)

    # Find unaligned transcript regions (holes)
    unaligned_text_indices = find_unaligned_regions(
        rg_min, rg_max, set_lis_results, merge_threshold=3
    )

    logger.debug(f"Resolved {len(resolved_alignment_results)} tokens, {len(unaligned_text_indices)} unaligned regions")

    return resolved_alignment_results, unaligned_text_indices


# =============================================================================
# Final word alignment (following Tutorial.py get_final_word_alignment)
# =============================================================================

def get_final_word_alignment(
    alignment_results: List[AlignedToken],
    text: str,
    tokenizer,
) -> Dict[int, AlignedWord]:
    """
    Convert token-level alignment to word-level alignment (follows Tutorial.py pattern).

    Groups consecutive tokens by word index and creates AlignedWord objects
    with start times and phone-level details.

    Args:
        alignment_results: List of AlignedToken from concat_alignments
        text: Original text (space-separated words)
        tokenizer: Tokenizer with id2token mapping (or similar interface)

    Returns:
        Dict mapping word index (int) to AlignedWord object.
        Keys are sorted in ascending order.

    Note:
        End times are not computed here - they should be derived from
        the start time of the next word or added separately.
    """
    if not alignment_results:
        return {}

    text_splitted = text.split()
    num_words = len(text_splitted)

    word_alignment = {}
    aligned_word = None
    word_idx = None

    for aligned_token in alignment_results:
        if "wid" in aligned_token.attr:
            # Save previous word before starting new one
            if aligned_word is not None and word_idx is not None:
                word_alignment[word_idx] = aligned_word

            word_idx = aligned_token.attr['wid']

            # Handle end-of-text placeholder (word_idx == num_words)
            if word_idx >= num_words:
                # This is the end marker, not a real word
                word = None
            else:
                word = text_splitted[word_idx]

            aligned_word = AlignedWord(
                word=word,
                start_time=aligned_token.timestamp,
                end_time=None,
                phones=[],
            )

        # Add phone-level alignment if token info available
        if 'tk' not in aligned_token.attr:
            continue

        if aligned_word is not None:
            # Get token symbol from tokenizer if available
            token_symbol = aligned_token.attr['tk']
            if hasattr(tokenizer, 'id2token') and token_symbol in tokenizer.id2token:
                token_symbol = tokenizer.id2token[token_symbol]

            aligned_word.phones.append(
                AlignedToken(
                    token_id=token_symbol,
                    timestamp=aligned_token.timestamp,
                    attr={},  # Use empty dict instead of None for consistency
                    score=aligned_token.score,
                )
            )

    # Save last word
    if aligned_word is not None and word_idx is not None:
        word_alignment[word_idx] = aligned_word

    # Sort by word index for consistent output
    word_alignment = {k: word_alignment[k] for k in sorted(word_alignment.keys())}

    # Compute end_time for each word from the next word's start_time
    sorted_indices = sorted(word_alignment.keys())
    for i, idx in enumerate(sorted_indices):
        if i + 1 < len(sorted_indices):
            next_idx = sorted_indices[i + 1]
            word_alignment[idx].end_time = word_alignment[next_idx].start_time
        # Last word: end_time remains None (or could estimate from last phone)

    logger.debug(f"Built word alignment: {len(word_alignment)} words")

    return word_alignment
