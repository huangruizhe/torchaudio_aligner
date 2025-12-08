"""
Tests for word index mapping and recovery.

Based on test_text_frontend.ipynb and test_alignment.ipynb test cases.

Tests cover:
- AlignedWord.index correctly maps to original text position
- AlignedWord.original preserves pre-normalization form
- Round-trip: can reconstruct original text from alignment
- display_text property shows original when different

Key concept from Tutorial.py:
- All text processing uses [fun(w) for w in words] pattern
- Empty results become '*' (unk token)
- Word count MUST be preserved through all transforms for alignment recovery
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for alignment.base imports"
)


class TestAlignedWordIndexMapping:
    """Tests for AlignedWord.index property."""

    def test_index_attribute_default(self):
        """Test default index value is -1."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=0, end_frame=50)
        assert word.index == -1

    def test_index_attribute_set(self):
        """Test index can be set."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=0, end_frame=50, index=5)
        assert word.index == 5

    def test_index_maps_to_original_text(self):
        """Test that word.index maps back to original text position."""
        from alignment.base import AlignedWord, AlignmentResult

        original_text = "Hello World This Is A Test"
        original_words = original_text.split()

        # Simulate alignment result with indices
        aligned_words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, index=0),
            AlignedWord(word="world", start_frame=60, end_frame=110, index=1),
            AlignedWord(word="this", start_frame=120, end_frame=160, index=2),
        ]

        # Can recover original words using index
        for aw in aligned_words:
            recovered = original_words[aw.index]
            assert recovered.lower() == aw.word.lower()


class TestAlignedWordOriginalForm:
    """Tests for AlignedWord.original property."""

    def test_original_default_none(self):
        """Test original is None by default."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=0, end_frame=50)
        assert word.original is None

    def test_original_preserves_punctuation(self):
        """Test original preserves punctuation."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="hello",  # normalized
            start_frame=0,
            end_frame=50,
            original="Hello!",  # with punctuation and case
        )
        assert word.original == "Hello!"

    def test_original_preserves_case(self):
        """Test original preserves case."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="hello",
            start_frame=0,
            end_frame=50,
            original="HELLO",
        )
        assert word.original == "HELLO"

    def test_original_preserves_unicode(self):
        """Test original preserves unicode characters."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="cafe",  # normalized (no accent)
            start_frame=0,
            end_frame=50,
            original="café",  # with accent
        )
        assert word.original == "café"

    def test_original_preserves_numbers(self):
        """Test original preserves numbers before expansion."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="sixtysixdollars",  # expanded
            start_frame=0,
            end_frame=50,
            original="$66",  # original currency
        )
        assert word.original == "$66"


class TestDisplayTextProperty:
    """Tests for AlignedWord.display_text property."""

    def test_display_text_no_original(self):
        """Test display_text returns word when no original."""
        from alignment.base import AlignedWord

        word = AlignedWord(word="hello", start_frame=0, end_frame=50)
        assert word.display_text == "hello"

    def test_display_text_same_as_word(self):
        """Test display_text when original equals word."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="hello",
            start_frame=0,
            end_frame=50,
            original="hello",
        )
        assert word.display_text == "hello"

    def test_display_text_shows_difference(self):
        """Test display_text shows both when different."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="hello",
            start_frame=0,
            end_frame=50,
            original="Hello!",
        )
        # Should show "Hello! (hello)" format
        assert word.display_text == "Hello! (hello)"

    def test_display_text_case_difference(self):
        """Test display_text with case difference."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="hello",
            start_frame=0,
            end_frame=50,
            original="HELLO",
        )
        assert word.display_text == "HELLO (hello)"

    def test_display_text_punctuation(self):
        """Test display_text with punctuation."""
        from alignment.base import AlignedWord

        word = AlignedWord(
            word="dont",
            start_frame=0,
            end_frame=50,
            original="Don't",
        )
        assert word.display_text == "Don't (dont)"


class TestAlignmentResultRecovery:
    """Tests for recovering original text from AlignmentResult."""

    def test_recover_original_via_indices(self):
        """Test recovering original text using word indices."""
        from alignment.base import AlignedWord, AlignmentResult

        original_text = "Hello, World! This is a Test."
        original_words = original_text.split()

        # Create aligned words with indices
        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, index=0),
            AlignedWord(word="world", start_frame=60, end_frame=110, index=1),
            AlignedWord(word="this", start_frame=120, end_frame=160, index=2),
            AlignedWord(word="is", start_frame=170, end_frame=200, index=3),
            AlignedWord(word="a", start_frame=210, end_frame=230, index=4),
            AlignedWord(word="test", start_frame=240, end_frame=300, index=5),
        ]
        result = AlignmentResult(words=words)

        # Recover original words
        recovered = [original_words[w.index] for w in result.words]
        assert recovered == ["Hello,", "World!", "This", "is", "a", "Test."]

    def test_recover_original_via_original_field(self):
        """Test recovering original using AlignedWord.original field."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, original="Hello,"),
            AlignedWord(word="world", start_frame=60, end_frame=110, original="World!"),
        ]
        result = AlignmentResult(words=words)

        # Recover using original field
        recovered = [w.original for w in result.words]
        assert recovered == ["Hello,", "World!"]

    def test_text_property_uses_normalized(self):
        """Test AlignmentResult.text returns normalized words."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, original="Hello,"),
            AlignedWord(word="world", start_frame=60, end_frame=110, original="World!"),
        ]
        result = AlignmentResult(words=words)

        # text property returns normalized (word field)
        assert result.text == "hello world"


class TestWordAlignmentsDictionary:
    """Tests for AlignmentResult.word_alignments property."""

    def test_word_alignments_keyed_by_index(self):
        """Test word_alignments dict is keyed by word index."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, index=0),
            AlignedWord(word="world", start_frame=60, end_frame=110, index=1),
            AlignedWord(word="test", start_frame=120, end_frame=170, index=5),  # gap
        ]
        result = AlignmentResult(words=words)

        alignments = result.word_alignments
        assert 0 in alignments
        assert 1 in alignments
        assert 5 in alignments
        assert alignments[0].word == "hello"
        assert alignments[5].word == "test"

    def test_word_alignments_allows_gap_detection(self):
        """Test that gaps in indices can be detected."""
        from alignment.base import AlignedWord, AlignmentResult

        words = [
            AlignedWord(word="hello", start_frame=0, end_frame=50, index=0),
            AlignedWord(word="world", start_frame=60, end_frame=110, index=1),
            # index 2, 3, 4 missing (unaligned)
            AlignedWord(word="test", start_frame=120, end_frame=170, index=5),
        ]
        result = AlignmentResult(words=words)

        alignments = result.word_alignments
        aligned_indices = set(alignments.keys())

        # Can detect which words are unaligned
        total_words = 6  # indices 0-5
        all_indices = set(range(total_words))
        unaligned_indices = all_indices - aligned_indices

        assert unaligned_indices == {2, 3, 4}


class TestTokenizerWordBoundaries:
    """Tests for tokenizer word boundary preservation (Test 5 from notebook)."""

    def test_char_tokenizer_preserves_word_boundaries(self):
        """Test CharTokenizer returns List[List[int]] preserving word boundaries."""
        from text_frontend.tokenizers import CharTokenizer

        # MMS-style vocabulary
        mms_vocab = {c: i for i, c in enumerate(
            ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
             't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '*']
        )}

        tokenizer = CharTokenizer(token2id=mms_vocab, unk_token='*')

        text = "hello world test"
        encoded = tokenizer.encode(text)

        # Should return List[List[int]] - one sublist per word
        assert len(encoded) == 3  # 3 words
        assert isinstance(encoded[0], list)

    def test_encode_flatten_returns_flat_list(self):
        """Test encode_flatten returns flat token list."""
        from text_frontend.tokenizers import CharTokenizer

        mms_vocab = {c: i for i, c in enumerate(
            ['-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
             't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '*']
        )}

        tokenizer = CharTokenizer(token2id=mms_vocab, unk_token='*')

        text = "hi"
        flattened = tokenizer.encode_flatten(text)

        # Should be flat list
        assert isinstance(flattened, list)
        assert not isinstance(flattened[0], list)


class TestEndToEndIndexRecovery:
    """End-to-end tests for word index recovery."""

    def test_normalization_to_alignment_recovery(self):
        """Test full pipeline: normalize -> align -> recover original."""
        from text_frontend.normalization import normalize_for_mms
        from alignment.base import AlignedWord, AlignmentResult

        # Original text with punctuation, case, numbers
        original_text = "Hello, World! I have $50."
        original_words = original_text.split()

        # Normalize (word count must be preserved!)
        normalized = normalize_for_mms(original_text, expand_numbers=True, word_joiner="")
        normalized_words = normalized.split()

        assert len(original_words) == len(normalized_words), \
            "Word count not preserved!"

        # Simulate alignment result
        aligned_words = []
        for i, (orig, norm) in enumerate(zip(original_words, normalized_words)):
            aligned_words.append(AlignedWord(
                word=norm,
                start_frame=i * 50,
                end_frame=(i + 1) * 50,
                original=orig,
                index=i,
            ))

        result = AlignmentResult(words=aligned_words)

        # Can recover original text
        recovered_via_index = [original_words[w.index] for w in result.words]
        recovered_via_original = [w.original for w in result.words]

        assert recovered_via_index == original_words
        assert recovered_via_original == original_words

    def test_partial_alignment_recovery(self):
        """Test recovery when only some words are aligned."""
        from alignment.base import AlignedWord, AlignmentResult

        original_text = "The quick brown fox jumps over"
        original_words = original_text.split()

        # Only some words aligned (indices 1, 2, 4 - "quick", "brown", "jumps")
        aligned_words = [
            AlignedWord(word="quick", start_frame=50, end_frame=100, index=1, original="quick"),
            AlignedWord(word="brown", start_frame=110, end_frame=160, index=2, original="brown"),
            AlignedWord(word="jumps", start_frame=220, end_frame=280, index=4, original="jumps"),
        ]

        result = AlignmentResult(words=aligned_words)

        # Can recover aligned portion
        recovered = [original_words[w.index] for w in result.words]
        assert recovered == ["quick", "brown", "jumps"]

        # Can identify unaligned words
        aligned_indices = {w.index for w in result.words}
        unaligned = [original_words[i] for i in range(len(original_words))
                     if i not in aligned_indices]
        assert unaligned == ["The", "fox", "over"]
