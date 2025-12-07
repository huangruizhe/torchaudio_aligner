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
