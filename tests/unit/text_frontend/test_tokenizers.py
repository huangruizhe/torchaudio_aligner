"""
Tests for tokenizer module.

Based on test_text_frontend.ipynb Test 5.

Tests cover:
- CharTokenizer
- TokenizerInterface
- Word boundary preservation
- encode/encode_flatten methods
- Vocabulary normalization
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for text_frontend imports"
)


# Common MMS-style vocabulary for tests
MMS_VOCAB = {c: i for i, c in enumerate(
    ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
     'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
     't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '*']
)}


class TestCharTokenizerCreation:
    """Tests for CharTokenizer creation."""

    def test_create_from_vocab(self):
        """Test creating CharTokenizer from vocabulary dict."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(
            token2id=MMS_VOCAB,
            blank_token="-",
            unk_token="*",
        )

        assert tokenizer.blk_id == 0  # '-' is at index 0
        assert tokenizer.unk_token == "*"

    def test_token2id_mapping(self):
        """Test token to ID mapping."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        assert tokenizer.token2id["a"] == 1
        assert tokenizer.token2id["z"] == 26
        assert tokenizer.token2id["'"] == 27

    def test_id2token_mapping(self):
        """Test ID to token mapping."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        assert tokenizer.id2token[0] == "-"
        assert tokenizer.id2token[1] == "a"

    def test_supported_chars(self):
        """Test get_supported_chars returns single-char tokens."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        supported = tokenizer.get_supported_chars()
        assert "a" in supported
        assert "'" in supported
        assert len(supported) == 29  # All single chars in vocab


class TestCharTokenizerEncode:
    """Tests for CharTokenizer encoding."""

    def test_encode_single_word(self):
        """Test encoding a single word."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        result = tokenizer.encode("hello")
        assert len(result) == 1  # One word
        assert isinstance(result[0], list)  # List of token IDs

    def test_encode_multiple_words(self):
        """Test encoding multiple words."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        result = tokenizer.encode("hello world")
        assert len(result) == 2  # Two words

    def test_encode_preserves_word_boundaries(self):
        """Test that encode returns List[List[int]] preserving word boundaries."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        text = "hello world test"
        encoded = tokenizer.encode(text)

        # Should return one sublist per word
        assert len(encoded) == 3
        assert all(isinstance(word_tokens, list) for word_tokens in encoded)

    def test_encode_word_character_mapping(self):
        """Test that each character maps to correct token."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        result = tokenizer.encode("hi")
        # "hi" should be [[h_id, i_id]]
        assert len(result) == 1
        assert result[0] == [MMS_VOCAB["h"], MMS_VOCAB["i"]]

    def test_encode_flatten(self):
        """Test encode_flatten returns flat list."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        flattened = tokenizer.encode_flatten("hi")

        # Should be flat list
        assert isinstance(flattened, list)
        assert all(isinstance(t, int) for t in flattened)
        assert len(flattened) == 2  # h, i

    def test_encode_out_type_str(self):
        """Test encode with out_type=str."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        result = tokenizer.encode("hi", out_type=str)
        assert result == [["h", "i"]]


class TestCharTokenizerNormalization:
    """Tests for CharTokenizer normalization."""

    def test_text_normalize_preserves_word_count(self):
        """Test that text_normalize preserves word count."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        text = "Hello World! This is a TEST."
        normalized = tokenizer.text_normalize(text)

        orig_words = text.split()
        norm_words = normalized.split()

        assert len(orig_words) == len(norm_words)

    def test_normalize_lowercases(self):
        """Test that normalization lowercases text."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        normalized = tokenizer.text_normalize("HELLO")
        assert normalized == "hello"

    def test_normalize_for_vocab_handles_smart_quotes(self):
        """Test normalize_for_vocab handles smart quotes."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        # Smart quote (curly apostrophe) should become straight apostrophe
        result = tokenizer.normalize_for_vocab("Meta's")
        assert result == "meta's"

    def test_normalize_for_vocab_handles_accents(self):
        """Test normalize_for_vocab handles accented characters."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        result = tokenizer.normalize_for_vocab("café")
        assert result == "cafe"

    def test_normalize_empty_becomes_unk(self):
        """Test that empty result becomes unk token."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        # Pure punctuation should become *
        result = tokenizer.normalize_for_vocab("!!!")
        assert result == "*"


class TestCharTokenizerDecode:
    """Tests for CharTokenizer decoding."""

    def test_decode_round_trip(self):
        """Test encode/decode round trip."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        original = "hello world"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)

        assert decoded == ["hello", "world"]

    def test_decode_flatten(self):
        """Test decode_flatten."""
        from text_frontend.tokenizers import CharTokenizer

        tokenizer = CharTokenizer(token2id=MMS_VOCAB, blank_token="-", unk_token="*")

        token_ids = [MMS_VOCAB["h"], MMS_VOCAB["i"]]
        decoded = tokenizer.decode_flatten(token_ids)

        assert decoded == ["h", "i"]


class TestCreateTokenizerFactory:
    """Tests for tokenizer factory functions."""

    def test_create_tokenizer_char(self):
        """Test create_tokenizer with 'char' type."""
        from text_frontend.tokenizers import create_tokenizer, CharTokenizer

        tokenizer = create_tokenizer(
            "char",
            token2id=MMS_VOCAB,
            blank_token="-",
            unk_token="*",
        )

        assert isinstance(tokenizer, CharTokenizer)

    def test_create_tokenizer_invalid_type(self):
        """Test create_tokenizer with invalid type."""
        from text_frontend.tokenizers import create_tokenizer

        with pytest.raises(ValueError):
            create_tokenizer("invalid_type")

    def test_create_tokenizer_from_labels(self):
        """Test create_tokenizer_from_labels convenience function."""
        from text_frontend.tokenizers import create_tokenizer_from_labels, CharTokenizer

        labels = tuple(['-', 'a', 'b', 'c', '*'])
        tokenizer = create_tokenizer_from_labels(labels, blank_token="-", unk_token="*")

        assert isinstance(tokenizer, CharTokenizer)
        assert tokenizer.blk_id == 0
        assert tokenizer.token2id["a"] == 1


class TestBPETokenizer:
    """Tests for BPE tokenizer (if sentencepiece available)."""

    def test_bpe_requires_model(self):
        """Test BPE tokenizer requires model path or model object."""
        from text_frontend.tokenizers import BPETokenizer

        with pytest.raises(ValueError):
            BPETokenizer()  # No model provided


class TestPhonemeTokenizer:
    """Tests for Phoneme tokenizer (if cmudict and g2p_en available)."""

    def test_phoneme_tokenizer_creation(self):
        """Test creating phoneme tokenizer."""
        try:
            from text_frontend.tokenizers import PhonemeTokenizer

            tokenizer = PhonemeTokenizer()
            assert tokenizer.blk_id == 0
        except ImportError:
            pytest.skip("cmudict/g2p_en not available")

    def test_phoneme_encode(self):
        """Test phoneme tokenizer encoding."""
        try:
            from text_frontend.tokenizers import PhonemeTokenizer

            tokenizer = PhonemeTokenizer()
            result = tokenizer.encode("hello")

            # Should return list of pronunciations
            assert len(result) == 1  # One word
            assert len(result[0]) >= 1  # At least one pronunciation
        except ImportError:
            pytest.skip("cmudict/g2p_en not available")
