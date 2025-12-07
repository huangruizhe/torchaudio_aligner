"""
Romanization Module

Converts non-Latin scripts to Latin characters for alignment:
- uroman: Universal romanization (Arabic, Hindi, Russian, etc.)
- cutlet: Japanese morphological romanization (higher quality)
- CJK preprocessing: Character-level splitting
"""

from typing import Optional, List
import string
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
_UROMAN_AVAILABLE = False
_CUTLET_AVAILABLE = False

try:
    import uroman
    _UROMAN_AVAILABLE = True
except ImportError:
    uroman = None

try:
    import cutlet
    _CUTLET_AVAILABLE = True
except ImportError:
    cutlet = None


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

    uroman sometimes merges or splits characters during romanization.
    This function aligns the romanized output back to the original word count.

    Args:
        original_words: List of original words
        romanized_words: List of romanized words from uroman
        unk_token: Token to use for unaligned positions

    Returns:
        List of romanized words with same length as original_words
    """
    if len(original_words) == len(romanized_words):
        return romanized_words

    n_orig = len(original_words)
    n_roman = len(romanized_words)

    if n_roman == 0:
        return [unk_token] * n_orig

    result = []

    if n_roman > n_orig:
        # More romanized words than original - merge some
        ratio = n_roman / n_orig
        for i in range(n_orig):
            start_idx = int(i * ratio)
            end_idx = int((i + 1) * ratio)
            merged = "".join(romanized_words[start_idx:end_idx])
            result.append(merged if merged else unk_token)
    else:
        # Fewer romanized words than original - insert unk_token
        ratio = n_orig / n_roman
        roman_idx = 0
        for i in range(n_orig):
            expected_roman_idx = int(i / ratio)
            if expected_roman_idx < n_roman and roman_idx <= expected_roman_idx:
                result.append(romanized_words[roman_idx])
                roman_idx += 1
            else:
                result.append(unk_token)

        # Append remaining romanized words to last position
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
        >>> romanize_japanese_morphemes("東京は日本の首都です")
        "toukyou wa nihon no shuto desu"
    """
    if not _CUTLET_AVAILABLE:
        raise ImportError(
            "cutlet is required for Japanese morphological romanization. "
            "Install with: pip install cutlet"
        )

    logger.info(f"Romanizing Japanese with cutlet (system={system})")

    katsu = cutlet.Cutlet(system)
    katsu.use_foreign_spelling = use_foreign_spelling

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

    Args:
        text: Japanese text (space-separated, e.g., from preprocess_cjk)
        system: Romanization system
        unk_token: Token for unaligned positions

    Returns:
        Romanized text with same word count as input
    """
    if not _CUTLET_AVAILABLE:
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
        default_punct = string.punctuation + "。，！？、；：""''（）【】《》"

    punct_set = set(default_punct)
    if punctuation_chars:
        punct_set.update(punctuation_chars)

    # Remove whitespace and punctuation
    text = "".join(text.split())
    text = "".join(c for c in text if c not in punct_set)

    # Split into characters
    return " ".join(list(text))


def get_available_romanizers() -> dict:
    """Return availability of romanization backends."""
    return {
        "uroman": _UROMAN_AVAILABLE,
        "cutlet": _CUTLET_AVAILABLE,
    }
