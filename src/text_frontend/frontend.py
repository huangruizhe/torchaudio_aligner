"""
Text Frontend Class

Main class for text loading, preprocessing, and tokenization.
"""

from dataclasses import dataclass
from typing import Optional, List, Union, Literal
from pathlib import Path
import logging

from .loaders import (
    load_text_from_file,
    load_text_from_url,
    load_text_from_pdf,
    load_text_from_pdf_ocr,
)
from .normalization import (
    normalize_text_basic,
    normalize_for_mms,
    apply_text_normalization,
)
from .romanization import (
    romanize_text,
    romanize_text_aligned,
    romanize_japanese_morphemes_aligned,
    preprocess_cjk,
)
from .tokenizers import (
    CharTokenizer,
    BPETokenizer,
    PhonemeTokenizer,
)

logger = logging.getLogger(__name__)

# Check for cutlet availability
_CUTLET_AVAILABLE = False
try:
    import cutlet
    _CUTLET_AVAILABLE = True
except ImportError:
    pass


class TextFrontend:
    """
    Text frontend for loading, preprocessing, and tokenizing text.

    Example:
        >>> frontend = TextFrontend()
        >>> text = frontend.load("document.pdf")
        >>> normalized = frontend.normalize(text)
        >>> tokens = frontend.tokenize(normalized, tokenizer)
    """

    def __init__(self):
        """Initialize text frontend."""
        pass

    def load(
        self,
        source: Union[str, Path],
        source_type: Literal["auto", "file", "url", "pdf"] = "auto",
        encoding: str = "utf-8",
    ) -> str:
        """
        Load text from various sources.

        Args:
            source: File path, URL, or PDF path
            source_type: Type of source ("auto" detects from source)
            encoding: File encoding for text files

        Returns:
            Loaded text
        """
        source_str = str(source)

        # Auto-detect source type
        if source_type == "auto":
            if source_str.startswith(("http://", "https://")):
                source_type = "url"
            elif source_str.lower().endswith(".pdf"):
                source_type = "pdf"
            else:
                source_type = "file"

        logger.info(f"Loading text from {source_type}: {source}")

        if source_type == "url":
            return load_text_from_url(source_str)
        elif source_type == "pdf":
            return load_text_from_pdf(source)
        else:
            return load_text_from_file(source, encoding=encoding)

    def normalize(
        self,
        text: str,
        lowercase: bool = True,
        romanize: bool = False,
        language: Optional[str] = None,
        cjk_split: bool = False,
        for_mms: bool = True,
        expand_numbers: bool = False,
        tn_language: str = "en",
    ) -> str:
        """
        Normalize text for alignment.

        Args:
            text: Input text
            lowercase: Convert to lowercase
            romanize: Apply romanization (requires uroman)
            language: Language code for romanization
            cjk_split: Split CJK characters into space-separated
            for_mms: Apply MMS-style normalization
            expand_numbers: Expand numbers to spoken form
            tn_language: Language code for number expansion

        Returns:
            Normalized text
        """
        # CJK preprocessing (before romanization)
        if cjk_split:
            text = preprocess_cjk(text)

        # Romanization
        if romanize:
            text = romanize_text(text, language=language)

        # Basic normalization
        if lowercase:
            text = text.lower()

        # MMS-style normalization
        if for_mms:
            text = normalize_for_mms(text, expand_numbers=expand_numbers, tn_language=tn_language)
        else:
            text = normalize_text_basic(text)

        return text

    def tokenize(
        self,
        text: str,
        tokenizer: CharTokenizer,
    ) -> List[List[int]]:
        """
        Tokenize normalized text.

        Args:
            text: Normalized text
            tokenizer: CharTokenizer instance

        Returns:
            List of token ID lists (one per word)
        """
        return tokenizer.encode(text)


@dataclass
class PreparedText:
    """Result of prepare_for_alignment()."""
    original_text: str          # Raw text as loaded
    normalized_text: str        # Normalized text (word count preserved!)
    original_words: List[str]   # Original words (for recovery)
    normalized_words: List[str] # Normalized words (for alignment)
    word_count: int             # Number of words
    tokens: Optional[List[List[int]]] = None  # Token IDs if tokenizer provided

    def recover_original(self, word_indices: List[int]) -> List[str]:
        """
        Recover original words from alignment word indices.

        Args:
            word_indices: List of word indices from alignment output

        Returns:
            List of original words at those indices
        """
        return [self.original_words[i] for i in word_indices if i < len(self.original_words)]


def prepare_for_alignment(
    source: Union[str, Path],
    language: str = "eng",
    tokenizer: Optional[Union[CharTokenizer, BPETokenizer, PhonemeTokenizer]] = None,
    expand_numbers: bool = True,
    use_ocr: bool = False,
    ocr_languages: List[str] = None,
) -> PreparedText:
    """
    One-liner to prepare text for forced alignment.

    This combines:
    1. Loading text from file/URL/PDF (with optional OCR)
    2. Language-specific preprocessing (CJK split, romanization)
    3. Text normalization (TN for numbers, MMS normalization)
    4. Optional tokenization

    The key invariant is preserved: word count stays the same.

    Args:
        source: Path to file, URL, or PDF
        language: Language code (ISO 639-3: "eng", "cmn", "jpn", etc.)
        tokenizer: Optional tokenizer
        expand_numbers: Expand numbers to spoken form
        use_ocr: Use OCR for scanned PDFs
        ocr_languages: Language codes for OCR

    Returns:
        PreparedText object

    Example:
        >>> result = prepare_for_alignment("transcript.pdf", language="en")
        >>> print(f"Loaded {result.word_count} words")
    """
    # Normalize language code
    lang_map = {
        "en": "eng", "zh": "cmn", "ja": "jpn", "hi": "hin",
        "ko": "kor", "tl": "tgl", "pt": "por", "de": "deu",
        "fr": "fra", "es": "spa", "ru": "rus", "ar": "ara",
        "zhuang": None, "za": None,
    }

    # Determine processing flags
    lang_code = lang_map.get(language, language)
    is_cjk = language in ("cmn", "zh", "jpn", "ja", "kor", "ko")
    is_japanese = language in ("jpn", "ja")
    needs_romanization = lang_code is not None and language not in ("eng", "en", "zhuang", "za")

    # TN language (num2words uses ISO 639-1)
    tn_lang_map = {"eng": "en", "cmn": "zh", "jpn": "ja", "deu": "de", "fra": "fr", "por": "pt"}
    tn_language = tn_lang_map.get(lang_code, "en")

    # Step 1: Load text
    source_str = str(source)
    if use_ocr and source_str.lower().endswith(".pdf"):
        original_text = load_text_from_pdf_ocr(source, languages=ocr_languages or [tn_language])
    elif source_str.startswith(("http://", "https://")):
        original_text = load_text_from_url(source_str)
    elif source_str.lower().endswith(".pdf"):
        original_text = load_text_from_pdf(source)
    else:
        original_text = load_text_from_file(source)

    # Clean up whitespace
    original_text = " ".join(original_text.split())

    # Step 2: Preprocess
    processed_text = original_text

    # CJK: split into individual characters
    if is_cjk:
        processed_text = preprocess_cjk(processed_text)

    # Store original words (after CJK split if applicable)
    original_words = processed_text.split()

    # Step 3: Romanize (for non-Latin scripts)
    if needs_romanization:
        if is_japanese and _CUTLET_AVAILABLE:
            processed_text = romanize_japanese_morphemes_aligned(processed_text)
        else:
            processed_text = romanize_text_aligned(processed_text, language=lang_code)

    # Step 4: Text Normalization (expand numbers)
    if expand_numbers:
        processed_text = apply_text_normalization(processed_text, language=tn_language, word_joiner="")

    # Step 5: MMS normalization (final cleanup)
    normalized_text = normalize_for_mms(processed_text, expand_numbers=False)
    normalized_words = normalized_text.split()

    # Verify word count preservation
    assert len(original_words) == len(normalized_words), (
        f"Word count changed! original={len(original_words)}, normalized={len(normalized_words)}. "
        "This is a bug in the text frontend."
    )

    # Step 6: Tokenize (optional)
    tokens = None
    if tokenizer is not None:
        tokens = tokenizer.encode(normalized_text)

    return PreparedText(
        original_text=original_text,
        normalized_text=normalized_text,
        original_words=original_words,
        normalized_words=normalized_words,
        word_count=len(original_words),
        tokens=tokens,
    )
