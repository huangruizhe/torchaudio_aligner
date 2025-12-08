"""
Text Normalization (TN) Module

Converts written form to spoken form for alignment:
- Numbers: "123" -> "one hundred twenty three"
- Currency: "$7.50" -> "seven dollars fifty cents"
- Percentages: "50%" -> "fifty percent"
- Ordinals: "1st" -> "first"

Uses cascading fallback strategy:
1. wetext (comprehensive, EN/ZH/JA)
2. whisper_normalizer (placeholder)
3. num2words (60+ languages)
4. No-op fallback
"""

from typing import Optional
import re
import string
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
_NUM2WORDS_AVAILABLE = False
_WETEXT_AVAILABLE = False
_WHISPER_NORMALIZER_AVAILABLE = False
_UNIDECODE_AVAILABLE = False

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

try:
    from unidecode import unidecode as _unidecode
    _UNIDECODE_AVAILABLE = True
except ImportError:
    _unidecode = None

# Placeholder for whisper_normalizer
_WhisperNormalizer = None


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


def expand_number(
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

    # 3. Integer (66) - use regex to match only ASCII digits (isdigit() matches Unicode like ²)
    if expanded is None and re.fullmatch(r'[0-9]+', stripped):
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
    # Only match ASCII digits to avoid Unicode superscripts like ²
    if expanded is None and re.search(r'[0-9]', word) and re.search(r'[a-zA-Z]', word):
        try:
            expanded = _expand_mixed(word, language)
        except (ValueError, TypeError):
            expanded = None

    if expanded is None:
        return word

    # Join multi-word outputs to preserve word count
    if word_joiner is not None:
        expanded = expanded.replace(" ", word_joiner).replace("-", word_joiner)

    return expanded


def expand_numbers_in_text(
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
    expanded_words = [expand_number(w, language, word_joiner) for w in words]
    return " ".join(expanded_words)


def _normalize_with_wetext(text: str, language: str = "en") -> Optional[str]:
    """Apply wetext normalization (comprehensive TN)."""
    if not _WETEXT_AVAILABLE:
        return None

    try:
        normalizer = _WeTextNormalizer(language)
        return normalizer.normalize(text)
    except Exception as e:
        logger.debug(f"wetext normalization failed: {e}")
        return None


def _normalize_with_whisper(text: str, language: str = "en") -> Optional[str]:
    """Apply Whisper's text normalizer (placeholder)."""
    if not _WHISPER_NORMALIZER_AVAILABLE:
        return None
    return None  # Placeholder


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
            - "" (empty, default): "123" -> "onehundredandtwentythree"
            - "-": "123" -> "one-hundred-and-twenty-three"
            - None: Keep as-is "123" -> "one hundred and twenty-three"

    Returns:
        Text with numbers/symbols converted to spoken form
    """
    # Strategy 1: Try wetext
    wetext_langs = {"en", "zh", "ja"}
    if language in wetext_langs:
        result = _normalize_with_wetext(text, language)
        if result is not None:
            logger.debug(f"TN: used wetext for language={language}")
            return result

    # Strategy 2: Try whisper_normalizer (placeholder)
    if language == "en":
        result = _normalize_with_whisper(text, language)
        if result is not None:
            logger.debug(f"TN: used whisper_normalizer for language={language}")
            return result

    # Strategy 3: Try num2words
    if _NUM2WORDS_AVAILABLE:
        result = expand_numbers_in_text(text, language, word_joiner)
        logger.debug(f"TN: used num2words for language={language}")
        return result

    # Strategy 4: No-op fallback
    logger.warning(f"TN: no library available for language={language}")
    return text


# Basic normalization
def normalize_text_basic(text: str) -> str:
    """Basic text normalization: lowercase, collapse whitespace."""
    text = text.lower()
    text = text.replace("'", "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# MMS-style normalization
_MMS_PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'
_MMS_VALID_CHARS = set("abcdefghijklmnopqrstuvwxyz'")


def _normalize_word_for_mms(word: str, unk_token: str = "*") -> str:
    """
    Normalize a single word for MMS model.

    Uses unidecode for robust Unicode handling when available:
    - Smart quotes (') → straight apostrophe (')
    - Accented chars (café) → ASCII (cafe)
    - Em-dashes, special punctuation → removed or simplified

    Falls back to manual normalization if unidecode unavailable.
    """
    # Step 1: ASCII transliteration (handles all Unicode edge cases)
    if _UNIDECODE_AVAILABLE:
        word = _unidecode(word)
    else:
        # Fallback: manual apostrophe normalization
        word = word.replace("'", "'")  # Right single quotation mark
        word = word.replace("'", "'")  # Left single quotation mark
        word = word.replace("`", "'")  # Backtick
        word = word.replace("´", "'")  # Acute accent

    # Step 2: Remove punctuation and lowercase
    word = word.translate(str.maketrans("", "", _MMS_PUNCTUATION))
    word = word.lower()
    word = word.replace("-", "")

    # Step 3: Filter to valid MMS characters
    result = ''.join(c for c in word if c in _MMS_VALID_CHARS)

    return result if result else unk_token


def normalize_for_mms(
    text: str,
    unk_token: str = "*",
    expand_numbers: bool = False,
    tn_language: str = "en",
    word_joiner: Optional[str] = "",
) -> str:
    """
    Normalize text for MMS model.

    Preserves word count:
    - Optionally expand numbers to spoken form
    - Remove punctuation (except apostrophe)
    - Remove hyphens
    - Convert to lowercase
    - Words that become empty are replaced with unk_token

    Args:
        text: Input text
        unk_token: Token for unknown/invalid words (default "*")
        expand_numbers: Whether to expand numbers to spoken form
        tn_language: Language code for number expansion
        word_joiner: String to join multi-word TN outputs

    Returns:
        Normalized text with same word count as input
    """
    if expand_numbers:
        text = expand_numbers_in_text(text, language=tn_language, word_joiner=word_joiner)

    words = text.split()
    normalized_words = [_normalize_word_for_mms(w, unk_token) for w in words]
    return " ".join(normalized_words)


def get_available_normalizers() -> dict:
    """Return availability of normalization backends."""
    return {
        "num2words": _NUM2WORDS_AVAILABLE,
        "wetext": _WETEXT_AVAILABLE,
        "whisper_normalizer": _WHISPER_NORMALIZER_AVAILABLE,
        "unidecode": _UNIDECODE_AVAILABLE,
    }
