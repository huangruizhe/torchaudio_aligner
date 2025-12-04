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

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Literal
from pathlib import Path
import logging
import re
import string

logger = logging.getLogger(__name__)

# Optional dependencies
_REQUESTS_AVAILABLE = False
_BS4_AVAILABLE = False
_PYPDF_AVAILABLE = False
_UROMAN_AVAILABLE = False
_NUM2WORDS_AVAILABLE = False
_WETEXT_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None

try:
    import uroman
    _UROMAN_AVAILABLE = True
except ImportError:
    uroman = None

try:
    import num2words as _num2words_module
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _num2words_module = None

try:
    from wetext import Normalizer as _WeTextNormalizer
    _WETEXT_AVAILABLE = True
except ImportError:
    _WeTextNormalizer = None

# Placeholder for whisper_normalizer - TODO: implement when needed
# try:
#     from whisper_normalizer.english import EnglishTextNormalizer as _WhisperNormalizer
#     _WHISPER_NORMALIZER_AVAILABLE = True
# except ImportError:
#     _WhisperNormalizer = None
_WHISPER_NORMALIZER_AVAILABLE = False
_WhisperNormalizer = None

# Placeholder for nemo_text_processing - TODO: implement when needed
# NeMo provides comprehensive TN/ITN for many languages
# try:
#     from nemo_text_processing.text_normalization.normalize import Normalizer as _NemoNormalizer
#     _NEMO_AVAILABLE = True
# except ImportError:
#     _NemoNormalizer = None
_NEMO_AVAILABLE = False
_NemoNormalizer = None

# Optional: OCR for scanned PDFs
_OCR_AVAILABLE = False
_PDF2IMAGE_AVAILABLE = False
try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    easyocr = None

try:
    from pdf2image import convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_path = None

# Optional: Japanese morphological analysis with cutlet
_CUTLET_AVAILABLE = False
try:
    import cutlet
    _CUTLET_AVAILABLE = True
except ImportError:
    cutlet = None


def get_available_text_backends() -> dict:
    """Return availability of optional text processing backends."""
    return {
        "requests": _REQUESTS_AVAILABLE,
        "beautifulsoup": _BS4_AVAILABLE,
        "pypdf": _PYPDF_AVAILABLE,
        "uroman": _UROMAN_AVAILABLE,
        "num2words": _NUM2WORDS_AVAILABLE,
        "wetext": _WETEXT_AVAILABLE,
        "whisper_normalizer": _WHISPER_NORMALIZER_AVAILABLE,  # placeholder
        "nemo_text_processing": _NEMO_AVAILABLE,  # placeholder
        "ocr": _OCR_AVAILABLE,
        "pdf2image": _PDF2IMAGE_AVAILABLE,
        "cutlet": _CUTLET_AVAILABLE,
    }


# =============================================================================
# Text Loading Utilities
# =============================================================================

def load_text_from_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Load text from a local file.

    Args:
        file_path: Path to text file
        encoding: File encoding (default utf-8)

    Returns:
        Text content as string
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    logger.info(f"Loading text from file: {file_path}")
    with open(file_path, "r", encoding=encoding) as f:
        text = f.read()

    logger.info(f"Loaded {len(text)} characters, {len(text.split())} words")
    return text


def load_text_from_url(url: str, timeout: int = 30) -> str:
    """
    Load text from a URL (HTML page).

    Requires: pip install requests beautifulsoup4

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Extracted text content
    """
    if not _REQUESTS_AVAILABLE:
        raise ImportError("requests is required. Install with: pip install requests")
    if not _BS4_AVAILABLE:
        raise ImportError("beautifulsoup4 is required. Install with: pip install beautifulsoup4")

    logger.info(f"Fetching text from URL: {url}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    text = text.replace("\r\n", "\n")

    logger.info(f"Loaded {len(text)} characters, {len(text.split())} words")
    return text


def load_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from a PDF file.

    Requires: pip install pypdf

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content
    """
    if not _PYPDF_AVAILABLE:
        raise ImportError("pypdf is required. Install with: pip install pypdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(str(pdf_path))

    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())
    text = " ".join(text_parts)

    logger.info(f"Extracted {len(text)} characters, {len(text.split())} words from {len(reader.pages)} pages")
    return text


def load_text_from_pdf_ocr(
    pdf_path: Union[str, Path],
    languages: List[str] = None,
    dpi: int = 300,
    fallback_to_text: bool = True,
) -> str:
    """
    Extract text from a scanned/image-based PDF using OCR.

    This is useful for PDFs that contain scanned images instead of text
    (common for Hindi, historical documents, etc.).

    Requires: pip install easyocr pdf2image
    Also requires poppler-utils system package for pdf2image.

    Args:
        pdf_path: Path to PDF file
        languages: List of language codes for EasyOCR (e.g., ['en'], ['hi', 'en'])
            Default: ['en']
        dpi: Resolution for PDF to image conversion (higher = better quality but slower)
        fallback_to_text: If True, try text extraction first, use OCR only if empty

    Returns:
        Extracted text content

    Example:
        >>> # For Hindi scanned PDF
        >>> text = load_text_from_pdf_ocr("hindi_document.pdf", languages=['hi', 'en'])

        >>> # For English scanned PDF
        >>> text = load_text_from_pdf_ocr("scanned_book.pdf", languages=['en'])
    """
    if not _PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required for OCR. Install with: pip install pdf2image\n"
            "Also install poppler-utils: brew install poppler (macOS) or apt install poppler-utils (Linux)"
        )
    if not _OCR_AVAILABLE:
        raise ImportError("easyocr is required for OCR. Install with: pip install easyocr")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    languages = languages or ['en']

    # Try text extraction first if fallback enabled
    if fallback_to_text:
        try:
            text = load_text_from_pdf(pdf_path)
            # Check if we got meaningful text (not just whitespace/noise)
            if len(text.strip()) > 100 and len(text.split()) > 20:
                logger.info("PDF contains extractable text, skipping OCR")
                return text
        except Exception:
            pass
        logger.info("PDF appears to be scanned/image-based, using OCR")

    logger.info(f"Converting PDF to images (dpi={dpi}): {pdf_path}")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info(f"Converted {len(images)} pages to images")

    # Initialize EasyOCR reader
    logger.info(f"Initializing EasyOCR with languages: {languages}")
    reader = easyocr.Reader(languages, gpu=False)  # CPU by default for compatibility

    text_parts = []
    for i, image in enumerate(images):
        logger.info(f"OCR processing page {i+1}/{len(images)}")
        # EasyOCR expects numpy array or file path
        import numpy as np
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)  # detail=0 returns text only
        page_text = " ".join(results)
        text_parts.append(page_text)

    text = " ".join(text_parts)
    logger.info(f"OCR extracted {len(text)} characters, {len(text.split())} words from {len(images)} pages")
    return text


# =============================================================================
# Text Normalization (TN) - Written to Spoken Form
# =============================================================================
#
# TN Pipeline Summary:
# ====================
# Goal: Convert written form (e.g., "66", "$7.30", "1st") to spoken form
#       (e.g., "sixty-six", "seven dollars thirty cents", "first")
#       while PRESERVING WORD COUNT for alignment recovery.
#
# Cascading Fallback Strategy:
# ----------------------------
# 1. wetext (if available) - Comprehensive TN for EN/ZH/JA
#    - Handles: numbers, dates, times, currency, measurements, abbreviations
#    - Lightweight, no Pynini dependency
#
# 2. whisper_normalizer (placeholder) - OpenAI Whisper's text normalizer
#    - Well-tested, handles many edge cases
#    - pip install whisper-normalizer
#    - TODO: Implement when needed
#
# 3. num2words (if available) - Number expansion for 60+ languages
#    - Handles: integers, decimals, ordinals, comma-separated numbers
#    - Currency support: $, €, £, ¥, ₹ (e.g., "$7.50" -> "seven dollars fifty cents")
#    - Output joined with word_joiner to preserve word count
#
# 4. No-op fallback - Return original text unchanged
#    - Numbers will become "*" after MMS normalization
#
# Key Insight: Use word_joiner="" to merge multi-word outputs into single word
#              e.g., "$66" -> "sixty-six dollars" -> "sixtysixdollars" (1 word!)
#              This preserves word count for alignment recovery!
#
# =============================================================================


# Currency symbols: (symbol, singular, plural, cent_singular, cent_plural)
_CURRENCY_SYMBOLS = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "pence", "pence"),
    "¥": ("yen", "yen", "sen", "sen"),
    "₹": ("rupee", "rupees", "paisa", "paise"),
}

# Regex patterns (compiled once for efficiency)
_RE_DECIMAL = re.compile(r'^\d+\.\d+$')
_RE_ORDINAL = re.compile(r'^(\d+)(st|nd|rd|th)$', re.IGNORECASE)
_RE_COMMA_NUM = re.compile(r'^[\d,]+$')


def _num2words_safe(num, lang="en", to="cardinal"):
    """Wrapper for num2words with error handling."""
    try:
        return _num2words_module.num2words(num, lang=lang, to=to)
    except (ValueError, NotImplementedError, OverflowError):
        return None


def _expand_currency(word: str, symbol: str, names: tuple, lang: str) -> Optional[str]:
    """Expand currency: $66 -> sixty-six dollars, $7.50 -> seven dollars fifty cents."""
    num_part = word[len(symbol):].strip(string.punctuation.replace(".", ""))
    singular, plural, cent_sg, cent_pl = names

    if "." in num_part:
        parts = num_part.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            main = int(parts[0])
            cents = int(parts[1].ljust(2, "0")[:2])
            main_text = _num2words_safe(main, lang)
            if main_text is None:
                return None
            unit = singular if main == 1 else plural
            if cents > 0:
                cent_text = _num2words_safe(cents, lang)
                cent_unit = cent_sg if cents == 1 else cent_pl
                return f"{main_text} {unit} {cent_text} {cent_unit}"
            return f"{main_text} {unit}"
    elif num_part.replace(",", "").isdigit():
        num = int(num_part.replace(",", ""))
        num_text = _num2words_safe(num, lang)
        if num_text is None:
            return None
        return f"{num_text} {singular if num == 1 else plural}"
    return None


def _expand_percentage(word: str, lang: str) -> Optional[str]:
    """Expand percentage: 50% -> fifty percent."""
    num_part = word[:-1].strip(string.punctuation.replace(".", ""))
    if num_part.isdigit():
        num_text = _num2words_safe(int(num_part), lang)
    elif _RE_DECIMAL.match(num_part):
        num_text = _num2words_safe(float(num_part), lang)
    else:
        return None
    return f"{num_text} percent" if num_text else None


def _expand_mixed(word: str, lang: str) -> Optional[str]:
    """Expand mixed letter-number: COVID19 -> covid nineteen, B2B -> b two b."""
    segments = re.findall(r'[a-zA-Z]+|\d+', word)
    result = []
    for seg in segments:
        if seg.isdigit():
            text = _num2words_safe(int(seg), lang)
            if text is None:
                return None
            result.append(text)
        else:
            result.append(seg.lower())
    return " ".join(result)


def _expand_number_with_num2words(
    word: str,
    language: str = "en",
    word_joiner: Optional[str] = "",
) -> str:
    """
    Expand a single word containing numbers to spoken form.

    Handles: integers, decimals, ordinals (1st), currency ($66),
    percentage (50%), mixed (COVID19), comma-separated (1,000).

    Args:
        word: Input word
        language: Language code for num2words
        word_joiner: Join multi-word outputs ("" preserves word count, None keeps spaces)

    Returns:
        Expanded word or original if not a number
    """
    if not _NUM2WORDS_AVAILABLE:
        return word

    stripped = word.strip(string.punctuation)
    if not stripped:
        return word

    expanded = None

    # 1. Currency ($66, €7.50)
    for symbol, names in _CURRENCY_SYMBOLS.items():
        if word.startswith(symbol):
            expanded = _expand_currency(word, symbol, names, language)
            break

    # 2. Percentage (50%, 3.5%)
    if expanded is None and word.endswith('%'):
        expanded = _expand_percentage(word, language)

    # 3. Integer (66)
    if expanded is None and stripped.isdigit():
        expanded = _num2words_safe(int(stripped), language)

    # 4. Decimal (3.14)
    if expanded is None and _RE_DECIMAL.match(stripped):
        expanded = _num2words_safe(float(stripped), language)

    # 5. Ordinal (1st, 2nd, 3rd)
    if expanded is None:
        m = _RE_ORDINAL.match(stripped)
        if m:
            expanded = _num2words_safe(int(m.group(1)), language, to='ordinal')

    # 6. Comma-separated (1,000)
    if expanded is None and _RE_COMMA_NUM.match(stripped) and ',' in stripped:
        expanded = _num2words_safe(int(stripped.replace(',', '')), language)

    # 7. Mixed letter-number (COVID19, B2B)
    if expanded is None and re.search(r'\d', word) and re.search(r'[a-zA-Z]', word):
        expanded = _expand_mixed(word, language)

    if expanded is None:
        return word

    # Join multi-word outputs to preserve word count
    if word_joiner is not None:
        expanded = expanded.replace(" ", word_joiner).replace("-", word_joiner)

    return expanded


def _expand_numbers_with_num2words(
    text: str,
    language: str = "en",
    word_joiner: Optional[str] = "",
) -> str:
    """
    Expand all numbers in text using num2words library.

    Args:
        text: Input text
        language: Language code for num2words
        word_joiner: String to join multi-word outputs (default "" to preserve word count)

    Returns:
        Text with numbers expanded to spoken form
    """
    if not _NUM2WORDS_AVAILABLE:
        return text

    words = text.split()
    expanded_words = [_expand_number_with_num2words(w, language, word_joiner) for w in words]
    return " ".join(expanded_words)


def _normalize_with_wetext(text: str, language: str = "en") -> str:
    """
    Apply wetext normalization (comprehensive TN).

    Wetext handles:
    - Numbers (cardinal, ordinal)
    - Dates and times
    - Currency (e.g., "$7.30" -> "seven dollars thirty cents")
    - Measurements
    - Abbreviations

    Supported languages: "en" (English), "zh" (Chinese), "ja" (Japanese)

    Args:
        text: Input text
        language: Language code ("en", "zh", "ja")

    Returns:
        Normalized text, or None if wetext fails/unavailable
    """
    if not _WETEXT_AVAILABLE:
        return None  # Signal to try next fallback

    try:
        normalizer = _WeTextNormalizer(language)
        return normalizer.normalize(text)
    except Exception as e:
        logger.debug(f"wetext normalization failed: {e}")
        return None  # Signal to try next fallback


def _normalize_with_whisper(text: str, language: str = "en") -> str:
    """
    Apply Whisper's text normalizer (placeholder - not yet implemented).

    Whisper normalizer handles:
    - Numbers and ordinals
    - Currency and percentages
    - Abbreviations and contractions
    - Well-tested on diverse real-world data

    Install: pip install whisper-normalizer

    Args:
        text: Input text
        language: Language code (currently only "en" supported)

    Returns:
        Normalized text, or None if unavailable/fails
    """
    if not _WHISPER_NORMALIZER_AVAILABLE:
        return None  # Signal to try next fallback

    # TODO: Implement when needed
    # try:
    #     normalizer = _WhisperNormalizer()
    #     return normalizer(text)
    # except Exception as e:
    #     logger.debug(f"whisper normalizer failed: {e}")
    #     return None

    return None  # Placeholder - not yet implemented


def apply_text_normalization(
    text: str,
    language: str = "en",
    word_joiner: Optional[str] = "",
) -> str:
    """
    Apply text normalization with cascading fallback strategy.

    Tries in order:
    1. wetext - Comprehensive TN (EN/ZH/JA only)
    2. whisper_normalizer - Well-tested normalizer (EN only, placeholder)
    3. num2words - Number expansion (60+ languages)
    4. No-op - Return original text

    Args:
        text: Input text
        language: Language code (e.g., "en", "zh", "ja", "de", "fr")
        word_joiner: String to join multi-word TN outputs into single word.
            - "" (empty, default): "123" -> "onehundredandtwentythree" (preserves word count!)
            - "-": "123" -> "one-hundred-and-twenty-three"
            - "_": "123" -> "one_hundred_and_twenty_three"
            - None: Keep as-is, e.g., "123" -> "one hundred and twenty-three" (breaks word count!)

    Returns:
        Text with numbers/symbols converted to spoken form

    Example:
        >>> apply_text_normalization("I have 123 apples", word_joiner="")
        "I have onehundredandtwentythree apples"  # 4 words -> 4 words (preserved!)

        >>> apply_text_normalization("I have 123 apples", word_joiner=None)
        "I have one hundred and twenty-three apples"  # 4 words -> 7 words (broken!)
    """
    # Strategy 1: Try wetext (comprehensive, but limited language support)
    # Note: wetext may return multi-word output, so we apply word_joiner after
    wetext_langs = {"en", "zh", "ja"}
    if language in wetext_langs:
        result = _normalize_with_wetext(text, language)
        if result is not None:
            logger.debug(f"TN: used wetext for language={language}")
            # wetext doesn't have word_joiner support, so we handle it here
            # This is a heuristic: we can't perfectly know which words were expanded
            # For now, return as-is (wetext is comprehensive so should be fine)
            return result

    # Strategy 2: Try whisper_normalizer (placeholder - EN only)
    if language == "en":
        result = _normalize_with_whisper(text, language)
        if result is not None:
            logger.debug(f"TN: used whisper_normalizer for language={language}")
            return result

    # Strategy 3: Try num2words (numbers only, but 60+ languages)
    if _NUM2WORDS_AVAILABLE:
        result = _expand_numbers_with_num2words(text, language, word_joiner)
        logger.debug(f"TN: used num2words for language={language}")
        return result

    # Strategy 4: No-op fallback
    logger.warning(f"TN: no library available for language={language}, returning original text")
    return text


# Legacy function names for backward compatibility
def expand_number(word: str, language: str = "en", word_joiner: Optional[str] = "") -> str:
    """Expand a single number to spoken form. (Legacy wrapper)"""
    return _expand_number_with_num2words(word, language, word_joiner)


def expand_numbers_in_text(text: str, language: str = "en", word_joiner: Optional[str] = "") -> str:
    """Expand all numbers in text to spoken form. (Legacy wrapper)"""
    return apply_text_normalization(text, language, word_joiner)


def apply_wetext_normalization(text: str, language: str = "en") -> str:
    """Apply wetext normalization with num2words fallback. (Legacy wrapper)"""
    return apply_text_normalization(text, language)


# =============================================================================
# Basic Text Normalization
# =============================================================================

def normalize_text_basic(text: str) -> str:
    """
    Basic text normalization: lowercase, collapse whitespace.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    text = text.lower()
    text = text.replace("'", "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Punctuation to remove (keep apostrophe for English contractions)
_MMS_PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'


def _normalize_word_for_mms(word: str, unk_token: str = "*") -> str:
    """
    Normalize a single word for MMS model.

    - Remove punctuation (except apostrophe)
    - Remove hyphens (to merge "sixty-six" -> "sixtysix")
    - Convert to lowercase
    - If word becomes empty, return unk_token

    This preserves word count which is critical for alignment.
    """
    # Remove punctuation (keep apostrophe)
    word = word.translate(str.maketrans("", "", _MMS_PUNCTUATION))
    # Lowercase
    word = word.lower()
    # Normalize apostrophe variants
    word = word.replace("'", "'")
    # Remove hyphens (merge hyphenated words like "sixty-six" -> "sixtysix")
    word = word.replace("-", "")
    # If empty, return unknown token
    if len(word) == 0:
        return unk_token
    # Check if word contains only valid MMS characters (a-z and ')
    if not all(c in "abcdefghijklmnopqrstuvwxyz'" for c in word):
        return unk_token
    return word


def normalize_for_mms(
    text: str,
    unk_token: str = "*",
    expand_numbers: bool = False,
    tn_language: str = "en",
    word_joiner: Optional[str] = "",
) -> str:
    """
    Normalize text for MMS model.

    This normalizes each word individually and preserves word count:
    - Optionally expand numbers to spoken form (e.g., "123" -> "onehundredandtwentythree")
    - Remove punctuation (except apostrophe)
    - Remove hyphens (merge "sixty-six" -> "sixtysix")
    - Convert to lowercase
    - Words that become empty or contain non-ASCII are replaced with unk_token

    Example (without TN):
        "Hello, World! 2025 你好" -> "hello world * *"

    Example (with TN, word_joiner=""):
        "Hello, World! 2025 你好" -> "hello world twothousandandtwentyfive *"

    Args:
        text: Input text
        unk_token: Token for unknown/invalid words (default "*")
        expand_numbers: Whether to expand numbers to spoken form
        tn_language: Language code for number expansion (default "en")
        word_joiner: String to join multi-word TN outputs (default "" to preserve word count)

    Returns:
        Normalized text with same word count as input
    """
    # Optionally expand numbers first (with word_joiner to preserve word count)
    if expand_numbers:
        text = expand_numbers_in_text(text, language=tn_language, word_joiner=word_joiner)

    words = text.split()
    normalized_words = [_normalize_word_for_mms(w, unk_token) for w in words]
    return " ".join(normalized_words)


def romanize_text(
    text: str,
    language: Optional[str] = None,
) -> str:
    """
    Romanize non-Latin script text using uroman.

    Requires: pip install uroman-python

    Args:
        text: Input text (can be any script)
        language: ISO-639-3 language code (e.g., "cmn" for Chinese, "jpn" for Japanese)
            See: https://iso639-3.sil.org/sites/iso639-3/files/downloads/iso-639-3_Latin1.tab

    Returns:
        Romanized text
    """
    if not _UROMAN_AVAILABLE:
        raise ImportError("uroman-python is required. Install with: pip install uroman-python")

    logger.info(f"Romanizing text with language={language}")
    if language:
        romanized = uroman.uroman(text, language=language)
    else:
        romanized = uroman.uroman(text)

    return romanized


def align_romanized_to_original(
    original_words: List[str],
    romanized_words: List[str],
    unk_token: str = "*",
) -> List[str]:
    """
    Align romanized words to original words, preserving word count.

    uroman sometimes merges or splits characters during romanization, causing
    word count mismatches. This function uses sequence alignment (difflib) to
    align the romanized output back to the original word count.

    Args:
        original_words: List of original words (e.g., CJK characters)
        romanized_words: List of romanized words from uroman
        unk_token: Token to use for unaligned positions

    Returns:
        List of romanized words with same length as original_words

    Example:
        >>> orig = ["私", "は", "猫", "です"]  # 4 words
        >>> roman = ["watashi", "ha", "nekodesu"]  # uroman merged last 2
        >>> align_romanized_to_original(orig, roman)
        ["watashi", "ha", "neko", "desu"]  # or ["watashi", "ha", "nekodesu", "*"]
    """
    import difflib

    if len(original_words) == len(romanized_words):
        return romanized_words  # No alignment needed

    # Use SequenceMatcher to find matching blocks
    # We match based on position ratio (since content is different scripts)
    n_orig = len(original_words)
    n_roman = len(romanized_words)

    if n_roman == 0:
        return [unk_token] * n_orig

    # Simple approach: distribute romanized words proportionally
    # and handle insertions/deletions with unk_token
    result = []

    if n_roman > n_orig:
        # More romanized words than original - merge some
        # Calculate how many romanized words per original word
        ratio = n_roman / n_orig
        for i in range(n_orig):
            start_idx = int(i * ratio)
            end_idx = int((i + 1) * ratio)
            # Merge romanized words for this position
            merged = "".join(romanized_words[start_idx:end_idx])
            result.append(merged if merged else unk_token)
    else:
        # Fewer romanized words than original - insert unk_token
        # Calculate positions where we need to insert
        ratio = n_orig / n_roman
        roman_idx = 0
        for i in range(n_orig):
            expected_roman_idx = int(i / ratio)
            if expected_roman_idx < n_roman and roman_idx <= expected_roman_idx:
                result.append(romanized_words[roman_idx])
                roman_idx += 1
            else:
                result.append(unk_token)

        # If we still have romanized words left, append them to last position
        if roman_idx < n_roman:
            remaining = "".join(romanized_words[roman_idx:])
            result[-1] = result[-1] + remaining if result[-1] != unk_token else remaining

    assert len(result) == n_orig, f"Alignment failed: {len(result)} != {n_orig}"
    return result


def romanize_text_aligned(
    text: str,
    language: Optional[str] = None,
    unk_token: str = "*",
) -> str:
    """
    Romanize text and align output to preserve word count.

    This is a wrapper around romanize_text() that ensures the romanized output
    has the same word count as the input text. This is critical for alignment
    recovery via word indices.

    Args:
        text: Input text (space-separated words)
        language: ISO-639-3 language code
        unk_token: Token to use for unaligned positions

    Returns:
        Romanized text with same word count as input
    """
    original_words = text.split()
    romanized = romanize_text(text, language)
    romanized_words = romanized.split()

    if len(original_words) == len(romanized_words):
        return romanized

    # Align to preserve word count
    logger.warning(
        f"uroman word count mismatch: {len(original_words)} -> {len(romanized_words)}, "
        "applying alignment correction"
    )
    aligned_words = align_romanized_to_original(original_words, romanized_words, unk_token)
    return " ".join(aligned_words)


def romanize_japanese_morphemes(
    text: str,
    system: str = "hepburn",
    use_foreign_spelling: bool = False,
) -> str:
    """
    Romanize Japanese text using morphological analysis with cutlet.

    This provides higher quality romanization than uroman by:
    1. Performing morphological analysis (word segmentation)
    2. Using proper readings for kanji based on context
    3. Following standard romanization systems (Hepburn, Kunrei, etc.)

    Requires: pip install cutlet

    Args:
        text: Japanese text to romanize
        system: Romanization system - "hepburn" (default), "kunrei", "nihon"
        use_foreign_spelling: If True, keep foreign words in original spelling

    Returns:
        Romanized text (space-separated morphemes)

    Example:
        >>> romanize_japanese_morphemes("風立ちぬ、いざ生きめやも")
        "kaze tachinu, iza ikime ya mo"

        >>> romanize_japanese_morphemes("東京は日本の首都です")
        "toukyou wa nihon no shuto desu"
    """
    if not _CUTLET_AVAILABLE:
        raise ImportError(
            "cutlet is required for Japanese morphological romanization. "
            "Install with: pip install cutlet\n"
            "Note: cutlet requires fugashi and unidic, which will be installed automatically."
        )

    logger.info(f"Romanizing Japanese with cutlet (system={system})")

    # Initialize cutlet with specified romanization system
    katsu = cutlet.Cutlet(system)
    katsu.use_foreign_spelling = use_foreign_spelling

    # Romanize the text
    romanized = katsu.romaji(text)

    logger.info(f"Romanized {len(text)} chars to {len(romanized)} chars")
    return romanized


def romanize_japanese_morphemes_aligned(
    text: str,
    system: str = "hepburn",
    unk_token: str = "*",
) -> str:
    """
    Romanize Japanese text with word count preservation.

    This combines cutlet's morphological analysis with alignment correction
    to ensure the output has the same word count as the input.

    Args:
        text: Japanese text (space-separated, e.g., from preprocess_cjk)
        system: Romanization system
        unk_token: Token for unaligned positions

    Returns:
        Romanized text with same word count as input
    """
    if not _CUTLET_AVAILABLE:
        # Fallback to uroman if cutlet not available
        logger.warning("cutlet not available, falling back to uroman for Japanese")
        return romanize_text_aligned(text, language="jpn", unk_token=unk_token)

    original_words = text.split()

    # Romanize each character/word separately to preserve count
    romanized_words = []
    katsu = cutlet.Cutlet(system)

    for word in original_words:
        try:
            rom = katsu.romaji(word).strip()
            romanized_words.append(rom if rom else unk_token)
        except Exception:
            romanized_words.append(unk_token)

    assert len(romanized_words) == len(original_words), \
        f"Word count mismatch: {len(romanized_words)} != {len(original_words)}"

    return " ".join(romanized_words)


# =============================================================================
# Language-Specific Preprocessing
# =============================================================================

def preprocess_cjk(text: str, punctuation_chars: Optional[str] = None) -> str:
    """
    Preprocess CJK (Chinese, Japanese, Korean) text:
    - Remove punctuation
    - Split into individual characters (space-separated)

    Args:
        text: Input CJK text
        punctuation_chars: Additional punctuation characters to remove

    Returns:
        Space-separated characters
    """
    # Default CJK and ASCII punctuation
    try:
        import zhon
        default_punct = zhon.hanzi.punctuation + string.punctuation
    except ImportError:
        # Fallback if zhon not available
        default_punct = string.punctuation + "。，！？、；：""''（）【】《》"

    punct_set = set(default_punct)
    if punctuation_chars:
        punct_set.update(punctuation_chars)

    # Remove whitespace and punctuation
    text = "".join(text.split())
    text = "".join(c for c in text if c not in punct_set)

    # Split into characters
    return " ".join(list(text))


# =============================================================================
# Tokenizer
# =============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for text tokenizer."""
    blank_token: str = "-"
    unk_token: str = "*"
    vocab: Optional[List[str]] = None


class CharTokenizer:
    """
    Character-level tokenizer for CTC models.

    Converts text to a list of token lists (one per word).
    """

    def __init__(
        self,
        token2id: dict,
        blank_token: str = "-",
        unk_token: str = "*",
    ):
        """
        Initialize tokenizer.

        Args:
            token2id: Mapping from token string to token ID
            blank_token: CTC blank token
            unk_token: Unknown token for out-of-vocabulary characters
        """
        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.blank_id = token2id.get(blank_token)
        self.unk_id = token2id.get(unk_token)

    def normalize(self, text: str) -> str:
        """Normalize text for this tokenizer (MMS-style)."""
        return normalize_for_mms(text)

    def encode_word(self, word: str) -> List[int]:
        """
        Encode a single word to token IDs.

        Args:
            word: Input word

        Returns:
            List of token IDs
        """
        tokens = []
        for char in word:
            if char in self.token2id:
                tokens.append(self.token2id[char])
            elif self.unk_id is not None:
                tokens.append(self.unk_id)
        return tokens

    def encode(self, text: str) -> List[List[int]]:
        """
        Encode text to token IDs (word-level grouping).

        Args:
            text: Input text (should be normalized)

        Returns:
            List of token ID lists, one per word
        """
        words = text.split()
        return [self.encode_word(word) for word in words]

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        """
        Decode token IDs back to words.

        Args:
            token_ids: List of token ID lists

        Returns:
            List of words
        """
        words = []
        for word_tokens in token_ids:
            word = "".join(self.id2token.get(t, self.unk_token) for t in word_tokens)
            words.append(word)
        return words


# =============================================================================
# BPE Tokenizer (SentencePiece)
# =============================================================================

class BPETokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer using SentencePiece.

    Converts text to subword tokens while preserving word boundaries.
    Useful for models trained with BPE vocabulary (e.g., Conformer CTC).

    Requires: pip install sentencepiece
    """

    def __init__(
        self,
        sp_model_path: str,
        blank_token: str = "<s>",
        unk_token: str = "<unk>",
    ):
        """
        Initialize BPE tokenizer.

        Args:
            sp_model_path: Path to SentencePiece model file (.model)
            blank_token: CTC blank token (must exist in SP model)
            unk_token: Unknown token (must exist in SP model)
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece is required. Install with: pip install sentencepiece")

        self.sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.blank_id = self.sp_model.piece_to_id(blank_token)
        self.unk_id = self.sp_model.piece_to_id(unk_token)

        # Build token<->id mappings
        self.token2id = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.vocab_size())}
        self.id2token = {v: k for k, v in self.token2id.items()}

        # Tokens that start a new word (have ▁ prefix)
        self._word_start_ids = {
            i for i in range(self.sp_model.vocab_size())
            if self.sp_model.id_to_piece(i).startswith("▁")
        }

    def normalize(self, text: str) -> str:
        """Normalize text for this tokenizer."""
        # BPE typically handles casing internally
        return text.strip()

    def _get_word_boundaries(self, token_ids: List[int]) -> List[List[int]]:
        """Split flat token list into word-level groups based on ▁ prefix."""
        result = []
        current_word = []

        for tid in token_ids:
            if tid in self._word_start_ids and current_word:
                result.append(current_word)
                current_word = []
            current_word.append(tid)

        if current_word:
            result.append(current_word)

        return result

    def encode_word(self, word: str) -> List[int]:
        """Encode a single word to token IDs."""
        return self.sp_model.encode(word, out_type=int)

    def encode(self, text: str) -> List[List[int]]:
        """
        Encode text to token IDs (word-level grouping).

        Args:
            text: Input text

        Returns:
            List of token ID lists, one per word
        """
        text = text.strip()
        token_ids = self.sp_model.encode(text, out_type=int)
        return self._get_word_boundaries(token_ids)

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        """Decode token IDs back to words."""
        return [
            [self.sp_model.id_to_piece(t) for t in word_tokens]
            for word_tokens in token_ids
        ]

    def decode_flat(self, token_ids: List[int]) -> str:
        """Decode flat token list to text."""
        return self.sp_model.decode(token_ids)


# =============================================================================
# Phoneme Tokenizer (CMUDict + G2P)
# =============================================================================

class PhonemeTokenizer:
    """
    Phoneme-level tokenizer using CMUDict and G2P fallback.

    Converts words to phoneme sequences for phone-level alignment.
    Supports multiple pronunciations per word.

    Requires: pip install cmudict g2p_en
    """

    def __init__(
        self,
        blank_token: str = "<blk>",
        unk_token: str = "<unk>",
    ):
        """
        Initialize phoneme tokenizer.

        Args:
            blank_token: CTC blank token
            unk_token: Unknown token for OOV words
        """
        try:
            import cmudict
            from g2p_en import G2p
        except ImportError:
            raise ImportError(
                "cmudict and g2p_en are required. Install with: pip install cmudict g2p_en"
            )

        self.cmu = cmudict.dict()
        self.g2p = G2p()

        # Build phoneme vocabulary from CMUDict
        self.token2id = {p: i + 1 for i, (p, _) in enumerate(cmudict.phones())}
        self.token2id[blank_token] = 0
        self.blank_id = 0
        self.unk_token = unk_token
        self.unk_id = len(self.token2id)
        self.token2id[unk_token] = self.unk_id

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.blank_token = blank_token

    def normalize(self, text: str) -> str:
        """Normalize text for this tokenizer."""
        return text.strip().lower()

    def _get_word_pronunciations(
        self, word: str, num_prons: Optional[int] = None
    ) -> List[List[str]]:
        """
        Get phoneme pronunciations for a word.

        Args:
            word: Input word
            num_prons: Maximum number of pronunciations to return

        Returns:
            List of pronunciations, each a list of phonemes
        """
        import re

        if word in self.cmu:
            prons = self.cmu[word][:num_prons]
        else:
            # Fallback to G2P for OOV words
            pron = self.g2p(word.replace("'", ""))
            if len(pron) == 0:
                pron = [self.unk_token]
            prons = [pron]

        # Remove stress markers (numbers)
        prons = [tuple(re.sub(r'\d', '', p) for p in pron) for pron in prons]
        prons = list(set(prons))  # Remove duplicates
        return [list(p) for p in prons]

    def encode_word(self, word: str, num_prons: Optional[int] = None) -> List[List[int]]:
        """
        Encode a single word to phoneme IDs.

        Args:
            word: Input word
            num_prons: Maximum number of pronunciations

        Returns:
            List of pronunciations, each a list of phoneme IDs
        """
        prons = self._get_word_pronunciations(word.lower(), num_prons)
        return [
            [self.token2id.get(p, self.unk_id) for p in pron]
            for pron in prons
        ]

    def encode(
        self, text: str, num_prons: Optional[int] = None
    ) -> List[List[List[int]]]:
        """
        Encode text to phoneme IDs (word-level grouping with multiple prons).

        Args:
            text: Input text
            num_prons: Maximum pronunciations per word

        Returns:
            List of word encodings, each containing list of possible pronunciations,
            each pronunciation being a list of phoneme IDs
        """
        text = text.strip().lower()
        return [self.encode_word(word, num_prons) for word in text.split()]

    def encode_flat(self, text: str) -> List[int]:
        """
        Encode text to flat phoneme sequence (first pronunciation only).

        Args:
            text: Input text

        Returns:
            Flat list of phoneme IDs
        """
        encoded = self.encode(text, num_prons=1)
        return [pid for word_prons in encoded for pid in word_prons[0]]

    def decode(self, token_ids: List[List[List[int]]]) -> List[List[List[str]]]:
        """Decode phoneme IDs back to phoneme strings."""
        return [
            [[self.id2token.get(p, self.unk_token) for p in pron] for pron in word_prons]
            for word_prons in token_ids
        ]


# =============================================================================
# Tokenizer Factory
# =============================================================================

def create_tokenizer(
    tokenizer_type: str = "char",
    **kwargs
) -> Union[CharTokenizer, BPETokenizer, PhonemeTokenizer]:
    """
    Factory function to create tokenizers.

    Args:
        tokenizer_type: One of "char", "bpe", "phoneme"
        **kwargs: Arguments passed to tokenizer constructor

    Returns:
        Tokenizer instance

    Examples:
        >>> # MMS character tokenizer
        >>> tok = create_tokenizer("char", token2id=mms_vocab, blank_token="-", unk_token="*")

        >>> # BPE tokenizer
        >>> tok = create_tokenizer("bpe", sp_model_path="model.spm")

        >>> # Phoneme tokenizer
        >>> tok = create_tokenizer("phoneme")
    """
    if tokenizer_type == "char":
        return CharTokenizer(**kwargs)
    elif tokenizer_type == "bpe":
        return BPETokenizer(**kwargs)
    elif tokenizer_type == "phoneme":
        return PhonemeTokenizer(**kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. Use 'char', 'bpe', or 'phoneme'.")


# =============================================================================
# Main Text Frontend Class
# =============================================================================

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
            for_mms: Apply MMS-style normalization (keep only a-z, ', space)
            expand_numbers: Expand numbers to spoken form (e.g., "66" -> "sixty-six")
            tn_language: Language code for number expansion (default "en")

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


# =============================================================================
# Convenience Functions
# =============================================================================

def load_text(
    source: Union[str, Path],
    source_type: Literal["auto", "file", "url", "pdf"] = "auto",
) -> str:
    """
    Convenience function to load text from various sources.

    Args:
        source: File path, URL, or PDF path
        source_type: Type of source

    Returns:
        Loaded text
    """
    frontend = TextFrontend()
    return frontend.load(source, source_type=source_type)


def normalize_text(
    text: str,
    romanize: bool = False,
    language: Optional[str] = None,
    cjk_split: bool = False,
    expand_numbers: bool = False,
    tn_language: str = "en",
) -> str:
    """
    Convenience function to normalize text for alignment.

    Args:
        text: Input text
        romanize: Apply romanization
        language: Language code for romanization
        cjk_split: Split CJK characters
        expand_numbers: Expand numbers to spoken form
        tn_language: Language code for number expansion

    Returns:
        Normalized text
    """
    frontend = TextFrontend()
    return frontend.normalize(
        text,
        romanize=romanize,
        language=language,
        cjk_split=cjk_split,
        expand_numbers=expand_numbers,
        tn_language=tn_language,
    )


# =============================================================================
# One-Liner Convenience Function for Alignment
# =============================================================================

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

    This is the main convenience function that combines:
    1. Loading text from file/URL/PDF (with optional OCR)
    2. Language-specific preprocessing (CJK split, romanization)
    3. Text normalization (TN for numbers, MMS normalization)
    4. Optional tokenization

    The key invariant is preserved: word count stays the same through all transforms,
    enabling lossless recovery of original words via word indices.

    Args:
        source: Path to file, URL, or PDF
        language: Language code. Supported formats:
            - ISO 639-3: "eng", "cmn", "jpn", "hin", "kor", "tgl", "por"
            - ISO 639-1: "en", "zh", "ja", "hi", "ko", "tl", "pt"
            - Special: "zhuang" or "za" for Zhuang
        tokenizer: Optional tokenizer (CharTokenizer, BPETokenizer, or PhonemeTokenizer)
        expand_numbers: Expand numbers to spoken form (e.g., "$123" -> "onehundredtwentythreedollars")
        use_ocr: Use OCR for scanned PDFs (requires easyocr, pdf2image)
        ocr_languages: Language codes for OCR (e.g., ['hi', 'en'] for Hindi)

    Returns:
        PreparedText object with:
            - original_text: Raw loaded text
            - normalized_text: Alignment-ready text
            - original_words: List of original words
            - normalized_words: List of normalized words
            - word_count: Number of words (same for original and normalized!)
            - tokens: Token IDs if tokenizer provided

    Example:
        >>> from torchaudio_aligner import prepare_for_alignment
        >>>
        >>> # English PDF
        >>> result = prepare_for_alignment("transcript.pdf", language="en")
        >>> print(f"Loaded {result.word_count} words")
        >>>
        >>> # Chinese text with romanization
        >>> result = prepare_for_alignment("chinese.txt", language="cmn")
        >>>
        >>> # Japanese with cutlet morphological analysis
        >>> result = prepare_for_alignment("japanese.txt", language="jpn")
        >>>
        >>> # Hindi scanned PDF with OCR
        >>> result = prepare_for_alignment("hindi.pdf", language="hi", use_ocr=True)
        >>>
        >>> # With tokenizer for immediate alignment
        >>> tokenizer = create_tokenizer("char", token2id=mms_vocab)
        >>> result = prepare_for_alignment("audio.txt", tokenizer=tokenizer)
        >>> # result.tokens is ready for alignment!
    """
    # Normalize language code
    lang_map = {
        # ISO 639-1 to ISO 639-3
        "en": "eng", "zh": "cmn", "ja": "jpn", "hi": "hin",
        "ko": "kor", "tl": "tgl", "pt": "por", "de": "deu",
        "fr": "fra", "es": "spa", "ru": "rus", "ar": "ara",
        # Special cases
        "zhuang": None, "za": None,  # Latin script, no romanization needed
    }

    # Determine processing flags based on language
    lang_code = lang_map.get(language, language)  # ISO 639-3 code for uroman
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
            # Use cutlet for better Japanese romanization
            processed_text = romanize_japanese_morphemes_aligned(processed_text)
        else:
            # Use uroman with alignment correction
            processed_text = romanize_text_aligned(processed_text, language=lang_code)

    # Step 4: Text Normalization (expand numbers)
    if expand_numbers:
        processed_text = apply_text_normalization(processed_text, language=tn_language, word_joiner="")

    # Step 5: MMS normalization (final cleanup)
    normalized_text = normalize_for_mms(processed_text, expand_numbers=False)  # Already expanded
    normalized_words = normalized_text.split()

    # Verify word count preservation (critical invariant!)
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


# =============================================================================
# Quick Start Examples (for documentation)
# =============================================================================

def _example_usage():
    """
    Example usage of torchaudio_aligner text frontend.

    This function is for documentation only and is not meant to be called.
    """
    # Example 1: Simple English text
    result = prepare_for_alignment("transcript.txt", language="en")
    print(f"Loaded {result.word_count} words")
    print(f"Normalized: {result.normalized_text[:100]}...")

    # Example 2: Chinese text (will be CJK-split and romanized)
    result = prepare_for_alignment("chinese.pdf", language="cmn")
    print(f"Chinese: {result.word_count} characters")

    # Example 3: Japanese with morphological analysis
    result = prepare_for_alignment("japanese.txt", language="jpn")
    print(f"Japanese: {result.word_count} morphemes")

    # Example 4: Hindi scanned PDF with OCR
    result = prepare_for_alignment(
        "hindi_scanned.pdf",
        language="hi",
        use_ocr=True,
        ocr_languages=["hi", "en"]
    )
    print(f"Hindi (OCR): {result.word_count} words")

    # Example 5: With tokenizer for immediate alignment
    mms_vocab = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'-*")}
    tokenizer = create_tokenizer("char", token2id=mms_vocab, blank_token="-", unk_token="*")
    result = prepare_for_alignment("audio_transcript.txt", tokenizer=tokenizer)
    print(f"Tokens ready: {len(result.tokens)} word groups")

    # Example 6: Recover original words after alignment
    # Suppose alignment gives us word indices [0, 5, 10, 15]
    aligned_indices = [0, 5, 10, 15]
    original_words = result.recover_original(aligned_indices)
    print(f"Aligned words: {original_words}")
