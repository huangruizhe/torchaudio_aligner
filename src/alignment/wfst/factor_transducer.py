"""
Factor transducer construction for CTC alignment.

This module provides functions to build weighted finite state transducers
for flexible speech-to-text alignment with support for:
- Word-level factor transducers
- Skip arcs for handling insertions/deletions
- Return arcs for error recovery
- Word index output labels for LIS-based concatenation
"""

from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


def flatten_list(lst):
    """
    Flatten a nested list of token IDs.

    Args:
        lst: List of lists (word -> tokens)

    Returns:
        Flattened list of tokens
    """
    return [item for w in lst for item in (w[0] if isinstance(w[0], list) else w)]


def make_ctc_graph(word_id_list, return_str=False):
    """
    Create a standard CTC graph (no skip/return arcs).

    Args:
        word_id_list: Tokenized text as list of lists [[tok1], [tok2, tok3], ...]
        return_str: If True, return string representation

    Returns:
        k2.Fsa or string representation of the CTC graph
    """
    try:
        import k2
    except ImportError:
        raise ImportError(
            "k2 is required for WFST alignment. "
            "Install with: pip install k2 -f https://k2-fsa.github.io/k2/cpu.html"
        )

    word_id_list_flattened = flatten_list(word_id_list)

    arcs = []
    start_state = 0
    final_state = 2 * len(word_id_list_flattened) + 1

    eps = 0
    arcs.append((start_state, start_state, eps, eps, 0))

    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0

    for w in word_id_list:
        if isinstance(w[0], list):
            w = w[0]

        for p in w:
            cur_non_blk_state = 2 * token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1

            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, p, 0))

            if prev_blk_state is not None:
                arcs.append((prev_blk_state, cur_non_blk_state, p, p, 0))

            arcs.append((cur_non_blk_state, cur_non_blk_state, p, eps, 0))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p
            token_cnt += 1

    arcs.append((prev_non_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_state, -1, -1, 0))
    arcs.append((final_state,))

    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        return fst


def make_factor_transducer_word_level_index_with_skip(
    word_id_list: List[List[int]],
    return_str: bool = False,
    blank_penalty: float = 0,
    skip_penalty: float = -0.5,
    return_penalty: Optional[float] = -18.0,
    noneps_bonus: float = 0.0,
    skip_id: Optional[int] = None,
) -> Tuple:
    """
    Create a factor transducer with word indices and skip/return arcs.

    This is the main function for building the decoding graph used in
    long-form fuzzy alignment. Key features:

    - **Word-level factors**: Words come as whole units (enter at word-start,
      exit at word-end)
    - **Word indices as output**: Instead of word symbols, output labels are
      word indices in the transcript. This enables LIS-based concatenation.
    - **Skip arcs**: Allow skipping tokens within a word (handles slight
      misalignments)
    - **Return arcs**: Allow returning to start state (handles gross
      misalignments)

    Args:
        word_id_list: Tokenized text as list of lists
            E.g., [[75], [47], [7], [629, 218]] for 4 words
        return_str: If True, return string representation instead of k2.Fsa
        blank_penalty: Penalty for blank tokens at graph boundaries (negative)
        skip_penalty: Penalty for skip arcs (default -0.5, negative = discouraged)
        return_penalty: Penalty for return arcs (default -18.0, None = no return arcs)
        noneps_bonus: Bonus for non-blank arcs (prior probability)
        skip_id: Custom skip symbol ID (auto-computed if None)

    Returns:
        If return_str=False:
            (fst, word_index_sym_tab, token_sym_tab)
            - fst: k2.Fsa with skip_id and return_id attributes
            - word_index_sym_tab: Dict mapping output label ID -> word index
            - token_sym_tab: Dict mapping output label ID -> token ID
        If return_str=True:
            (arc_string, word_index_sym_tab, token_sym_tab)

    Example:
        >>> text_tokenized = tokenizer.encode("hello world")
        >>> # [[h,e,l,l,o], [w,o,r,l,d]]
        >>> graph, word_sym, token_sym = make_factor_transducer_word_level_index_with_skip(
        ...     text_tokenized,
        ...     skip_penalty=-0.5,
        ...     return_penalty=-18.0
        ... )
        >>> graph = graph.to(device)
    """
    try:
        import k2
    except ImportError:
        raise ImportError(
            "k2 is required for WFST alignment. "
            "Install with: pip install k2 -f https://k2-fsa.github.io/k2/cpu.html"
        )

    word_id_list_flattened = flatten_list(word_id_list)

    # Check for empty input
    if not word_id_list or not word_id_list_flattened:
        raise ValueError(
            "word_id_list is empty after tokenization. "
            "This usually means text normalization failed for this language. "
            "Check if the text contains valid characters for the target language."
        )

    if skip_id is None:
        skip_id = 2 * len(word_id_list_flattened) + 1
    return_id = skip_id + 1

    arcs = []
    start_state = 0
    final_blk_state = 2 * len(word_id_list_flattened) + 1
    final_state = final_blk_state + 1

    eps = 0
    arcs.append((start_state, start_state, eps, eps, blank_penalty))

    # Symbol tables for decoding
    word_index_sym_tab = {eps: eps, 1: 0}  # Handle corner cases
    token_sym_tab = {eps: eps}
    sym_id = 1

    prev_blk_state = start_state
    prev_non_blk_state = None
    prev_p = None
    token_cnt = 0

    blank_states_for_skip = []  # Use -1 to separate words

    for i_w, w in enumerate(word_id_list):
        if isinstance(w[0], list):
            w = w[0]

        len_w = len(w)
        for i_p, p in enumerate(w):
            cur_non_blk_state = 2 * token_cnt + 1
            cur_blk_state = cur_non_blk_state + 1

            token_sym_tab[sym_id] = p

            if i_p == 0:
                blank_states_for_skip.append(-1)
            blank_states_for_skip.append(cur_blk_state)

            # Link to existing graph
            if prev_non_blk_state is not None:
                if p != prev_p:
                    arcs.append((prev_non_blk_state, cur_non_blk_state, p, sym_id, 0 + noneps_bonus))
                    if i_p == 0:
                        word_index_sym_tab[sym_id] = i_w

            if prev_blk_state is not None and prev_blk_state > 0:
                arcs.append((prev_blk_state, cur_non_blk_state, p, sym_id, 0 + noneps_bonus))
                if i_p == 0:
                    word_index_sym_tab[sym_id] = i_w

            # Token graph
            arcs.append((cur_non_blk_state, cur_non_blk_state, p, eps, 0 + noneps_bonus))
            arcs.append((cur_non_blk_state, cur_blk_state, eps, eps, 0))
            arcs.append((cur_blk_state, cur_blk_state, eps, eps, 0))

            # Factor arcs
            if i_p == 0:  # word-start
                arcs.append((start_state, cur_non_blk_state, p, sym_id, 0 + noneps_bonus))
            if i_p == len_w - 1:  # word-end
                arcs.append((cur_non_blk_state, final_blk_state, eps, sym_id + 1, 0))
                arcs.append((cur_non_blk_state, final_blk_state, p, sym_id + 1, 0 + noneps_bonus))

                # Return arcs
                if return_penalty is not None:
                    arcs.append((cur_non_blk_state, start_state, eps, return_id, return_penalty))

            prev_non_blk_state = cur_non_blk_state
            prev_blk_state = cur_blk_state
            prev_p = p

            token_cnt += 1
            sym_id += 1

    # Skip arcs (within words only, no cross-word leakage)
    for s1, s2 in zip(blank_states_for_skip, blank_states_for_skip[1:-1]):
        if s1 == -1 or s2 == -1:  # word boundary
            continue
        arcs.append((s1, s2, eps, skip_id, skip_penalty))

    # Final arcs
    arcs.append((final_blk_state, final_blk_state, eps, eps, blank_penalty))
    arcs.append((final_blk_state, final_state, -1, -1, 0))
    arcs.append((prev_blk_state, final_blk_state, eps, sym_id, 0))
    arcs.append((final_state,))
    word_index_sym_tab[sym_id] = i_w + 1

    # Validate skip_id
    assert skip_id > sym_id, f"skip_id={skip_id} must be > {sym_id}"
    assert skip_id > i_w + 1, f"skip_id={skip_id} must be > {i_w + 1}"

    # Build arc string
    new_arcs = sorted(arcs, key=lambda arc: arc[0])
    new_arcs = [" ".join(map(str, arc)) for arc in new_arcs]
    new_arcs = "\n".join(new_arcs)

    if return_str:
        return new_arcs, word_index_sym_tab, token_sym_tab
    else:
        fst = k2.Fsa.from_str(new_arcs, acceptor=False)
        fst = k2.arc_sort(fst)
        fst.skip_id = int(skip_id)
        fst.return_id = int(return_id)
        return fst, word_index_sym_tab, token_sym_tab
