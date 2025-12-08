"""
WFST/k2-based alignment with factor transducer.

This module provides robust alignment for long-form audio and noisy text
using weighted finite state transducers (WFST) with k2 library.

Key features:
- Factor transducer with skip/return arcs for fuzzy text matching
- Longest Increasing Subsequence (LIS) for multi-segment alignment
- Support for any CTC model backend
- Multilingual support (1100+ languages with MMS)

Note: Tokenization and segmentation are handled by the unified frontends:
- text_frontend: TokenizerInterface, CharTokenizer, create_tokenizer_from_labels
- audio_frontend: segment_waveform, SegmentationResult
"""

from .aligner import WFSTAligner
from .factor_transducer import (
    make_factor_transducer_word_level_index_with_skip,
    make_ctc_graph,
    flatten_list,
)
from .k2_utils import (
    get_best_paths,
    get_texts_with_timestamp,
    # Tutorial.py-style functions
    align_segments,
    concat_alignments,
    get_final_word_alignment,
    AlignedToken,
    AlignedWord,
)
from .lis_utils import (
    compute_lis,
    remove_outliers,
    remove_isolated_words,
    find_unaligned_regions,
    get_lis_alignment,
)

__all__ = [
    # Main aligner
    "WFSTAligner",
    # Factor transducer
    "make_factor_transducer_word_level_index_with_skip",
    "make_ctc_graph",
    "flatten_list",
    # k2 utilities
    "get_best_paths",
    "get_texts_with_timestamp",
    # Tutorial.py-style functions
    "align_segments",
    "concat_alignments",
    "get_final_word_alignment",
    "AlignedToken",
    "AlignedWord",
    # LIS utilities
    "compute_lis",
    "remove_outliers",
    "remove_isolated_words",
    "find_unaligned_regions",
    "get_lis_alignment",
]
