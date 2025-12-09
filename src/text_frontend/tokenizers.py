"""
Tokenizer Module

Tokenizers for converting text to token IDs for alignment:
- CharTokenizer: Character-level (for MMS/CTC models)
- BPETokenizer: Subword-level (SentencePiece, for Conformer)
- PhonemeTokenizer: Phoneme-level (CMUDict + G2P)

Token format for alignment:
- Input: "this is a sentence"
- Output: [[75], [47], [7], [629, 218]]
  (list of lists, each inner list = tokens for one word)

Each tokenizer handles its own vocabulary-specific normalization:
- Knows what characters it supports
- Normalizes unsupported chars (smart quotes, accents, etc.) appropriately
- Uses unidecode for robust ASCII transliteration when needed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Set
import re
import logging
import unicodedata

from .normalization import normalize_for_mms

# Optional dependency for robust Unicode normalization
try:
    from unidecode import unidecode as _unidecode
    _UNIDECODE_AVAILABLE = True
except ImportError:
    _unidecode = None
    _UNIDECODE_AVAILABLE = False

logger = logging.getLogger(__name__)


# ===========================================================================
# Punctuation Detection Utilities
# ===========================================================================

# Characters that are pronounced (keep as content, not blank)
DEFAULT_PRONOUNCED_PUNCT = {
    "$", "%", "&", "@", "#", "+", "=", "/", "-",
}

# Extra characters to treat as unpronounced (zero-width, formatting)
DEFAULT_EXTRA_UNPRONOUNCED = {
    "\u200b",  # ZERO WIDTH SPACE
    "\u200c",  # ZERO WIDTH NON-JOINER
    "\u200d",  # ZERO WIDTH JOINER
    "\u2060",  # WORD JOINER
}


def is_unpronounced_punct(
    ch: str,
    pronounced_punct: Set[str] = None,
    extra_unpronounced: Set[str] = None,
) -> bool:
    """
    Check if a character is punctuation that is NOT pronounced.

    Args:
        ch: Single character to check
        pronounced_punct: Characters to keep even if they're punctuation
        extra_unpronounced: Additional characters to treat as unpronounced

    Returns:
        True if the character should be treated as silent/blank
    """
    if not ch:
        return False

    if pronounced_punct is None:
        pronounced_punct = DEFAULT_PRONOUNCED_PUNCT
    if extra_unpronounced is None:
        extra_unpronounced = DEFAULT_EXTRA_UNPRONOUNCED

    # User explicitly wants this character pronounced
    if ch in pronounced_punct:
        return False

    # Unicode punctuation categories (P*)
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True

    # Extra invisible/formatting characters
    if ch in extra_unpronounced:
        return True

    return False


def is_only_unpronounced_punct(word: str) -> bool:
    """Check if a word consists entirely of unpronounced punctuation."""
    if not word:
        return True
    return all(is_unpronounced_punct(ch) for ch in word)


# ===========================================================================
# Abstract Base Class for Alignment Tokenizers
# ===========================================================================

class TokenizerInterface(ABC):
    """
    Abstract tokenizer interface for alignment.

    Tokenizers must implement:
    - encode(): Convert text to token IDs (word-level grouping)
    - text_normalize(): Normalize text before tokenization
    - get_supported_chars(): Return set of characters this tokenizer supports

    Token format:
    - Input: "this is a sentence"
    - Output: [[75], [47], [7], [629, 218]]
      (list of lists, each inner list = tokens for one word)

    Special tokens:
    - blk_id: Blank token ID (usually 0)
    - unk_id: Unknown token ID

    Attributes:
        token2id: Dict mapping token strings to IDs
        id2token: Dict mapping IDs to token strings
        blk_id: Blank token ID
        unk_id: Unknown token ID
    """

    token2id: Dict[str, int]
    id2token: Dict[int, str]
    blk_id: int
    unk_id: int

    def get_supported_chars(self) -> Set[str]:
        """
        Return set of characters this tokenizer supports.

        Default implementation returns all single-char tokens from vocabulary.
        Subclasses can override for more specific behavior.
        """
        return {k for k in self.token2id.keys() if len(k) == 1}

    def normalize_for_vocab(self, word: str, unk_token: str = "*") -> str:
        """
        Normalize a word to fit this tokenizer's vocabulary.

        Uses unidecode for robust Unicode->ASCII transliteration when available.
        This handles smart quotes, accented characters, etc.

        Args:
            word: Input word
            unk_token: Token to return if word becomes empty

        Returns:
            Normalized word containing only supported characters
        """
        supported = self.get_supported_chars()

        # Step 1: Try ASCII transliteration if unidecode available
        if _UNIDECODE_AVAILABLE:
            word = _unidecode(word)

        # Step 2: Lowercase and filter to supported chars
        result = []
        for char in word.lower():
            if char in supported:
                result.append(char)

        return ''.join(result) if result else unk_token

    @abstractmethod
    def encode(self, sentence: str, out_type=int) -> List[List[int]]:
        """
        Encode text to token IDs.

        Args:
            sentence: Input text
            out_type: Output type (int or str)

        Returns:
            List of lists: [[word1_tokens], [word2_tokens], ...]
        """
        raise NotImplementedError

    def encode_flatten(self, sentence: str, out_type=int) -> List[int]:
        """Encode and flatten to a single list of tokens."""
        tokens = self.encode(sentence, out_type=out_type)

        # Handle phoneme tokenizer which returns [[[pron1], [pron2]], ...]
        if tokens and tokens[0] and isinstance(tokens[0][0], (list, tuple)):
            tokens = [t for w_prons in tokens for t in w_prons[0]]
        else:
            tokens = [t for w_tokens in tokens for t in w_tokens]

        return tokens

    def decode_flatten(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs back to token strings."""
        if isinstance(token_ids[0], list):
            return [[self.id2token[t] for t in utt] for utt in token_ids]
        return [self.id2token[t] for t in token_ids]

    @abstractmethod
    def text_normalize(self, text: str) -> str:
        """Normalize text before tokenization."""
        raise NotImplementedError


@dataclass
class TokenizerConfig:
    """Configuration for text tokenizer."""
    blank_token: str = "-"
    unk_token: str = "*"
    vocab: Optional[List[str]] = None


class CharTokenizer(TokenizerInterface):
    """
    Character-level tokenizer for CTC models.

    Converts text to a list of token lists (one per word).
    Used with character-based CTC models like MMS_FA and Wav2Vec2.

    Supports robust Unicode normalization via unidecode:
    - Smart quotes (') → straight apostrophe (')
    - Accented chars (café) → ASCII (cafe)
    - Em-dashes (—) → hyphens (-) or removed

    Example:
        >>> labels = ('-', 'a', 'i', 'e', ..., '*')
        >>> tokenizer = CharTokenizer(
        ...     token2id={c: i for i, c in enumerate(labels)},
        ...     blank_token="-",
        ...     unk_token="*",
        ... )
        >>> tokenizer.encode("hello")
        [[7, 4, 11, 11, 14]]  # h, e, l, l, o
        >>> tokenizer.normalize_for_vocab("Meta's café")  # Smart quote + accent
        "meta's cafe"
    """

    # Punctuation to remove during normalization (fallback if unidecode unavailable)
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

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
        self.token2id = token2id.copy()
        self.id2token = {v: k for k, v in token2id.items()}
        self.blank_token = blank_token
        self.unk_token = unk_token
        self.blk_id = token2id.get(blank_token, 0)
        self.unk_id = token2id.get(unk_token)

        # Cache supported characters for efficient lookup
        self._supported_chars = self.get_supported_chars()

        # Verify blank is at position 0 (k2 requirement)
        if self.blk_id != 0:
            logger.warning(
                f"Blank token '{blank_token}' has ID {self.blk_id}, expected 0. "
                "This may cause issues with k2 alignment."
            )

    def get_supported_chars(self) -> Set[str]:
        """Return set of characters this tokenizer supports."""
        return {k for k in self.token2id.keys() if len(k) == 1}

    def normalize_for_vocab(self, word: str, unk_token: str = None) -> str:
        """
        Normalize a word to fit this tokenizer's vocabulary.

        Uses unidecode for robust Unicode handling:
        - Smart quotes → straight apostrophe
        - Accented chars → ASCII equivalents
        - Other Unicode → best ASCII approximation

        Special handling for punctuation:
        - Words that are ONLY unpronounced punctuation (., !, ?) -> blank_token
        - Words with real content that becomes empty -> unk_token

        Args:
            word: Input word (may contain Unicode)
            unk_token: Token for empty result (default: self.unk_token)

        Returns:
            Normalized word with only supported characters
        """
        if unk_token is None:
            unk_token = self.unk_token

        # Check if original word is only unpronounced punctuation BEFORE normalization
        # If so, use blank (silence) instead of unk (unknown content)
        original_is_punct_only = is_only_unpronounced_punct(word)

        # Step 1: ASCII transliteration (handles smart quotes, accents, etc.)
        if _UNIDECODE_AVAILABLE:
            word = _unidecode(word)
        else:
            # Fallback: manual punctuation removal
            word = word.translate(str.maketrans("", "", self.punctuation))

        # Step 2: Lowercase
        word = word.lower()

        # Step 3: Keep only supported characters
        result = ''.join(c for c in word if c in self._supported_chars)

        if result:
            return result
        elif original_is_punct_only:
            # Pure punctuation -> blank (silence, no acoustic content)
            return self.blank_token
        else:
            # Had real content but became empty -> unk (unknown)
            return unk_token

    def _normalize_word(self, word: str) -> str:
        """Normalize a single word. Uses normalize_for_vocab internally."""
        return self.normalize_for_vocab(word)

    def text_normalize(self, text: str) -> str:
        """
        Normalize text, preserving word count.

        ┌─────────────────────────────────────────────────────────────────────┐
        │ CRITICAL: Word count MUST be preserved for alignment recovery!      │
        │                                                                     │
        │   len(text.split()) == len(self.text_normalize(text).split())      │
        │                                                                     │
        │ This allows mapping alignment output indices back to original text. │
        └─────────────────────────────────────────────────────────────────────┘

        Words that become empty after normalization are replaced with unk_token.
        """
        words = [self.normalize_for_vocab(w) for w in text.split()]
        return " ".join(words)

    def normalize(self, text: str) -> str:
        """Alias for text_normalize (backward compatibility)."""
        return self.text_normalize(text)

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

    def encode(self, text: str, out_type=int) -> List[List[int]]:
        """
        Encode text to token IDs (word-level grouping).

        Args:
            text: Input text (should be normalized)
            out_type: Output type (int for IDs, str for tokens)

        Returns:
            List of token ID lists, one per word
        """
        words = text.split()
        if out_type == int:
            return [self.encode_word(word) for word in words]
        else:
            return [[c for c in word] for word in words]

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


class BPETokenizer(TokenizerInterface):
    """
    BPE (Byte Pair Encoding) tokenizer using SentencePiece.

    Converts text to subword tokens while preserving word boundaries.
    Word boundaries are detected by the "▁" prefix in SentencePiece tokens.

    Requires: pip install sentencepiece

    Example:
        >>> import sentencepiece as spm
        >>> sp_model = spm.SentencePieceProcessor(model_file="model.spm")
        >>> tokenizer = BPETokenizer(sp_model_path="model.spm")
    """

    # Punctuation to remove during normalization
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

    def __init__(
        self,
        sp_model_path: str = None,
        sp_model=None,
        blank_token: str = "<s>",
        unk_token: str = "<unk>",
    ):
        """
        Initialize BPE tokenizer.

        Args:
            sp_model_path: Path to SentencePiece model file (.model)
            sp_model: Pre-loaded SentencePiece model (alternative to path)
            blank_token: CTC blank token
            unk_token: Unknown token
        """
        if sp_model is not None:
            self.sp_model = sp_model
        elif sp_model_path is not None:
            try:
                import sentencepiece as spm
            except ImportError:
                raise ImportError("sentencepiece is required. Install with: pip install sentencepiece")
            self.sp_model = spm.SentencePieceProcessor(model_file=str(sp_model_path))
        else:
            raise ValueError("Either sp_model_path or sp_model must be provided")

        self.blank_token = blank_token
        self.unk_token = unk_token
        self.blk_id = self.sp_model.piece_to_id(blank_token)
        self.unk_id = self.sp_model.piece_to_id(unk_token)

        # Build token<->id mappings
        self.token2id = {self.sp_model.id_to_piece(i): i for i in range(self.sp_model.vocab_size())}
        self.id2token = {v: k for k, v in self.token2id.items()}

        # Tokens that start a new word (have ▁ prefix)
        self._word_start_ids = {
            i for i in range(self.sp_model.vocab_size())
            if self.sp_model.id_to_piece(i).startswith("▁")
        }

        # Verify blank is at position 0
        if self.blk_id != 0:
            logger.warning(
                f"Blank token '{blank_token}' has ID {self.blk_id}, expected 0. "
                "This may cause issues with k2 alignment."
            )

    def _normalize_word(self, word: str) -> str:
        """Normalize a single word."""
        word = word.translate(str.maketrans("", "", self.punctuation))
        word = word.lower()
        if len(word) == 0:
            return "*"
        return word

    def text_normalize(self, text: str) -> str:
        """Normalize text, preserving word count."""
        words = [self._normalize_word(w) for w in text.split()]
        return " ".join(words)

    def normalize(self, text: str) -> str:
        """Alias for text_normalize."""
        return self.text_normalize(text)

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

        # Remove empty first group if present
        if result and not result[0]:
            result = result[1:]

        return result

    def encode_word(self, word: str) -> List[int]:
        """Encode a single word to token IDs."""
        return self.sp_model.encode(word, out_type=int)

    def encode(self, text: str, out_type=int) -> List[List[int]]:
        """
        Encode text to token IDs (word-level grouping).

        Args:
            text: Input text
            out_type: Output type (int or str)

        Returns:
            List of token ID lists, one per word
        """
        text = text.strip()
        token_ids = self.sp_model.encode(text, out_type=out_type)
        if out_type == int:
            return self._get_word_boundaries(token_ids)
        else:
            # For string output, split by word
            return [[t] for t in token_ids]

    def decode(self, token_ids: List[List[int]]) -> List[str]:
        """Decode token IDs back to words."""
        return [
            [self.sp_model.id_to_piece(t) for t in word_tokens]
            for word_tokens in token_ids
        ]

    def decode_flat(self, token_ids: List[int]) -> str:
        """Decode flat token list to text."""
        return self.sp_model.decode(token_ids)


class PhonemeTokenizer(TokenizerInterface):
    """
    Phoneme-level tokenizer using CMUDict and G2P fallback.

    Converts words to phoneme sequences for phone-level alignment.
    Supports multiple pronunciations per word.

    Requires: pip install cmudict g2p_en

    Example:
        >>> tokenizer = PhonemeTokenizer()
        >>> tokenizer.encode("hello world")
        [[[HH, AH, L, OW]], [[W, ER, L, D]]]
    """

    # Punctuation to remove during normalization
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

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
        self.blk_id = 0
        self.unk_token = unk_token
        self.unk_id = len(self.token2id)
        self.token2id[unk_token] = self.unk_id

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.blank_token = blank_token

    def _normalize_word(self, word: str) -> str:
        """Normalize a single word."""
        word = word.translate(str.maketrans("", "", self.punctuation))
        word = word.lower()
        if len(word) == 0:
            return self.unk_token
        return word

    def text_normalize(self, text: str) -> str:
        """Normalize text, preserving word count."""
        words = [self._normalize_word(w) for w in text.split()]
        return " ".join(words)

    def normalize(self, text: str) -> str:
        """Alias for text_normalize."""
        return self.text_normalize(text)

    def _get_word_pronunciations(
        self, word: str, num_prons: Optional[int] = None
    ) -> List[List[str]]:
        """Get phoneme pronunciations for a word."""
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
        self, text: str, num_prons: Optional[int] = None, out_type=int
    ) -> List[List[List[int]]]:
        """
        Encode text to phoneme IDs (word-level grouping with multiple prons).

        Args:
            text: Input text
            num_prons: Maximum pronunciations per word
            out_type: Output type (int or str)

        Returns:
            List of word encodings, each containing list of possible pronunciations
        """
        text = text.strip().lower()
        if out_type == int:
            return [self.encode_word(word, num_prons) for word in text.split()]
        else:
            return [self._get_word_pronunciations(word, num_prons) for word in text.split()]

    def encode_flat(self, text: str, out_type=int) -> List[int]:
        """
        Encode text to flat phoneme sequence (first pronunciation only).

        Args:
            text: Input text
            out_type: Output type

        Returns:
            Flat list of phoneme IDs
        """
        encoded = self.encode(text, num_prons=1, out_type=out_type)
        return [pid for word_prons in encoded for pid in word_prons[0]]

    def decode(self, token_ids: List[List[List[int]]]) -> List[List[List[str]]]:
        """Decode phoneme IDs back to phoneme strings."""
        return [
            [[self.id2token.get(p, self.unk_token) for p in pron] for pron in word_prons]
            for word_prons in token_ids
        ]


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
        >>> tok = create_tokenizer("char", token2id=mms_vocab, blank_token="-")

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


def create_tokenizer_from_labels(
    labels: tuple,
    blank_token: str = "-",
    unk_token: str = "*",
) -> CharTokenizer:
    """
    Create a character tokenizer from a label tuple.

    This is a convenience function for creating tokenizers that match
    TorchAudio/HuggingFace model vocabularies.

    Args:
        labels: Tuple of label strings from model.get_labels() or vocab_info.labels
        blank_token: Blank token (should be labels[0])
        unk_token: Unknown token

    Returns:
        CharTokenizer instance

    Example:
        >>> import torchaudio
        >>> bundle = torchaudio.pipelines.MMS_FA
        >>> labels = bundle.get_labels(star="*")
        >>> tokenizer = create_tokenizer_from_labels(labels)

        >>> # Or from labeling_utils
        >>> from labeling_utils import load_model
        >>> model = load_model("mms-fa")
        >>> vocab = model.get_vocab_info()
        >>> tokenizer = create_tokenizer_from_labels(
        ...     tuple(vocab.labels),
        ...     blank_token=vocab.blank_token,
        ...     unk_token=vocab.unk_token,
        ... )
    """
    token2id = {c: i for i, c in enumerate(labels)}
    return CharTokenizer(
        token2id=token2id,
        blank_token=blank_token,
        unk_token=unk_token,
    )
