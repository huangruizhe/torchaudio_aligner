"""
Tokenizer Module

Tokenizers for converting text to token IDs for alignment:
- CharTokenizer: Character-level (for MMS/CTC models)
- BPETokenizer: Subword-level (SentencePiece, for Conformer)
- PhonemeTokenizer: Phoneme-level (CMUDict + G2P)
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import re
import logging

from .normalization import normalize_for_mms

logger = logging.getLogger(__name__)


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


class BPETokenizer:
    """
    BPE (Byte Pair Encoding) tokenizer using SentencePiece.

    Converts text to subword tokens while preserving word boundaries.

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
            blank_token: CTC blank token
            unk_token: Unknown token
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
        self, text: str, num_prons: Optional[int] = None
    ) -> List[List[List[int]]]:
        """
        Encode text to phoneme IDs (word-level grouping with multiple prons).

        Args:
            text: Input text
            num_prons: Maximum pronunciations per word

        Returns:
            List of word encodings, each containing list of possible pronunciations
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
