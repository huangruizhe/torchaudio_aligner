"""
Text Frontend Module for TorchAudio Long-Form Aligner

This module handles text loading, preprocessing, normalization, and tokenization
for speech-to-text alignment.

Main functionality:
- Load text from various sources (file, URL, PDF)
- Text normalization and cleanup
- Romanization for non-Latin scripts (via uroman)
- Tokenization for CTC models
"""

# Text loading
from .loaders import (
    load_text_from_file,
    load_text_from_url,
    load_text_from_pdf,
    load_text_from_pdf_ocr,
    get_available_loaders,
)

# Text normalization
from .normalization import (
    normalize_text_basic,
    normalize_for_mms,
    apply_text_normalization,
    expand_number,
    expand_numbers_in_text,
    get_available_normalizers,
)

# Romanization
from .romanization import (
    romanize_text,
    romanize_text_aligned,
    align_romanized_to_original,
    romanize_japanese_morphemes,
    romanize_japanese_morphemes_aligned,
    preprocess_cjk,
    get_available_romanizers,
)

# Tokenizers
from .tokenizers import (
    TokenizerInterface,
    CharTokenizer,
    BPETokenizer,
    PhonemeTokenizer,
    TokenizerConfig,
    create_tokenizer,
    create_tokenizer_from_labels,
)

# Main frontend class
from .frontend import (
    TextFrontend,
    PreparedText,
    prepare_for_alignment,
)

__all__ = [
    # Loaders
    "load_text_from_file",
    "load_text_from_url",
    "load_text_from_pdf",
    "load_text_from_pdf_ocr",
    "get_available_loaders",
    # Normalization
    "normalize_text_basic",
    "normalize_for_mms",
    "apply_text_normalization",
    "expand_number",
    "expand_numbers_in_text",
    "get_available_normalizers",
    # Romanization
    "romanize_text",
    "romanize_text_aligned",
    "align_romanized_to_original",
    "romanize_japanese_morphemes",
    "romanize_japanese_morphemes_aligned",
    "preprocess_cjk",
    "get_available_romanizers",
    # Tokenizers
    "TokenizerInterface",
    "CharTokenizer",
    "BPETokenizer",
    "PhonemeTokenizer",
    "TokenizerConfig",
    "create_tokenizer",
    "create_tokenizer_from_labels",
    # Frontend
    "TextFrontend",
    "PreparedText",
    "prepare_for_alignment",
]
