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
