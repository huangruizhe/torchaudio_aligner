"""
Tests for AlignmentResult class.

Tests cover:
- Constructor and attributes
- Properties (text, duration, num_words, word_alignments)
- Query methods (get_word_at_time, get_words_in_range)
- Export methods (to_audacity_labels, to_srt, to_textgrid, to_ctm, to_ass, to_json, to_dict)
- Statistics
- Iteration
- __repr__() and summary()

Note: Tests import alignment.base lazily to allow pytest collection
even if torch is not available. Tests skip at runtime if torch is missing.
"""

import pytest
import json

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for alignment.base imports"
)


class TestAlignmentResultConstructor:
    """Tests for AlignmentResult constructor and basic attributes."""

    def test_default_constructor(self):
        """Test default AlignmentResult."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        assert result.words == []
        assert result.chars == []
        assert result.unaligned_regions == []
        assert result.metadata == {}

    def test_with_words(self, aligned_words_list):
        """Test AlignmentResult with words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult(words=aligned_words_list)
        assert len(result.words) == 6
        assert result.words[0].word == "hello"

    def test_with_unaligned_regions(self, aligned_words_list):
        """Test AlignmentResult with unaligned regions."""
        from alignment.base import AlignmentResult
        result = AlignmentResult(
            words=aligned_words_list,
            unaligned_regions=[(10, 12), (20, 25)],
        )
        assert result.unaligned_regions == [(10, 12), (20, 25)]

    def test_with_metadata(self, aligned_words_list):
        """Test AlignmentResult with metadata."""
        from alignment.base import AlignmentResult
        result = AlignmentResult(
            words=aligned_words_list,
            metadata={"audio_duration": 10.0, "model": "mms-fa"},
        )
        assert result.metadata["audio_duration"] == 10.0
        assert result.metadata["model"] == "mms-fa"


class TestAlignmentResultProperties:
    """Tests for AlignmentResult properties."""

    def test_text_property(self, alignment_result_simple):
        """Test text property returns joined words."""
        assert alignment_result_simple.text == "hello world this is a test"

    def test_text_empty(self):
        """Test text property with no words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        assert result.text == ""

    def test_num_words_property(self, alignment_result_simple):
        """Test num_words property."""
        assert alignment_result_simple.num_words == 6

    def test_num_words_empty(self):
        """Test num_words with no words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        assert result.num_words == 0

    def test_duration_property(self, alignment_result_simple):
        """Test duration property."""
        # First word starts at frame 100 (2.0s), last ends at frame 400 (8.0s)
        # Duration = 8.0 - 2.0 = 6.0s
        assert alignment_result_simple.duration == 6.0

    def test_duration_empty(self):
        """Test duration with no words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        assert result.duration == 0.0

    def test_word_alignments_property(self, alignment_result_simple):
        """Test word_alignments property returns dict."""
        alignments = alignment_result_simple.word_alignments
        assert isinstance(alignments, dict)
        assert len(alignments) == 6
        assert 0 in alignments
        assert alignments[0].word == "hello"


class TestAlignmentResultIteration:
    """Tests for iteration over AlignmentResult."""

    def test_iter(self, alignment_result_simple):
        """Test iterating over result."""
        words = list(alignment_result_simple)
        assert len(words) == 6
        assert words[0].word == "hello"

    def test_len(self, alignment_result_simple):
        """Test len() on result."""
        assert len(alignment_result_simple) == 6

    def test_getitem(self, alignment_result_simple):
        """Test indexing result."""
        assert alignment_result_simple[0].word == "hello"
        assert alignment_result_simple[-1].word == "test"

    def test_slice(self, alignment_result_simple):
        """Test slicing result."""
        subset = alignment_result_simple[1:3]
        assert len(subset) == 2


class TestAlignmentResultQueryMethods:
    """Tests for query methods."""

    def test_get_word_at_time_found(self, alignment_result_simple):
        """Test get_word_at_time finds word."""
        # "hello" is at frames 100-150 (2.0s-3.0s)
        word = alignment_result_simple.get_word_at_time(2.5)
        assert word is not None
        assert word.word == "hello"

    def test_get_word_at_time_not_found(self, alignment_result_simple):
        """Test get_word_at_time returns None when not found."""
        # No word at time 0.0
        word = alignment_result_simple.get_word_at_time(0.5)
        assert word is None

    def test_get_word_at_time_boundary(self, alignment_result_simple):
        """Test get_word_at_time at word boundary."""
        # Start of "hello" at 2.0s
        word = alignment_result_simple.get_word_at_time(2.0)
        assert word is not None
        assert word.word == "hello"

    def test_get_words_in_range(self, alignment_result_simple):
        """Test get_words_in_range."""
        # Words starting between 2.0s and 6.0s
        words = alignment_result_simple.get_words_in_range(2.0, 6.0)
        # "hello" (2.0), "world" (3.2), "this" (4.6), "is" (5.6)
        assert len(words) >= 3

    def test_get_words_in_range_empty(self, alignment_result_simple):
        """Test get_words_in_range with no matches."""
        words = alignment_result_simple.get_words_in_range(0.0, 1.0)
        assert words == []


class TestAlignmentResultExportAudacity:
    """Tests for Audacity label export."""

    def test_to_audacity_labels(self, alignment_result_simple):
        """Test to_audacity_labels output format."""
        labels = alignment_result_simple.to_audacity_labels()
        lines = labels.strip().split("\n")
        assert len(lines) == 6

        # Check first line format: start\tend\tlabel
        parts = lines[0].split("\t")
        assert len(parts) == 3
        assert parts[2] == "hello"
        assert float(parts[0]) == 2.0  # start
        assert float(parts[1]) == 3.0  # end

    def test_to_audacity_labels_empty(self):
        """Test to_audacity_labels with no words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        labels = result.to_audacity_labels()
        assert labels == ""


class TestAlignmentResultExportSRT:
    """Tests for SRT subtitle export."""

    def test_to_srt_format(self, alignment_result_simple):
        """Test to_srt output format."""
        srt = alignment_result_simple.to_srt(words_per_subtitle=3)
        assert "1" in srt  # Subtitle number
        assert "-->" in srt  # Time separator
        assert "hello world this" in srt.lower()

    def test_to_srt_timestamp_format(self, alignment_result_simple):
        """Test SRT timestamp format (HH:MM:SS,mmm)."""
        srt = alignment_result_simple.to_srt()
        # Should contain timestamp like 00:00:02,000
        assert "00:00:0" in srt


class TestAlignmentResultExportTextGrid:
    """Tests for Praat TextGrid export."""

    def test_to_textgrid_format(self, alignment_result_simple):
        """Test to_textgrid output format."""
        tg = alignment_result_simple.to_textgrid()
        assert 'File type = "ooTextFile"' in tg
        assert 'Object class = "TextGrid"' in tg
        assert 'name = "words"' in tg
        assert "hello" in tg

    def test_to_textgrid_empty(self):
        """Test to_textgrid with no words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        tg = result.to_textgrid()
        assert tg == ""


class TestAlignmentResultExportCTM:
    """Tests for CTM format export."""

    def test_to_ctm_format(self, alignment_result_simple):
        """Test to_ctm output format."""
        ctm = alignment_result_simple.to_ctm(audio_id="test_audio")
        lines = ctm.strip().split("\n")
        assert len(lines) == 6

        # CTM format: <audio_id> <channel> <start> <duration> <word> [<confidence>]
        parts = lines[0].split()
        assert parts[0] == "test_audio"
        assert parts[1] == "1"  # channel
        assert parts[4] == "hello"

    def test_to_ctm_with_scores(self):
        """Test to_ctm includes confidence when available."""
        from alignment.base import AlignmentResult, AlignedWord
        word = AlignedWord(word="test", start_frame=0, end_frame=50, score=0.95)
        result = AlignmentResult(words=[word])
        ctm = result.to_ctm()
        assert "0.95" in ctm or "0.950" in ctm


class TestAlignmentResultExportASS:
    """Tests for ASS subtitle export."""

    def test_to_ass_format(self, alignment_result_simple):
        """Test to_ass output format."""
        ass = alignment_result_simple.to_ass()
        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "Dialogue:" in ass

    def test_to_ass_karaoke_tags(self, alignment_result_simple):
        """Test to_ass contains karaoke timing tags."""
        ass = alignment_result_simple.to_ass()
        # Should contain karaoke tags like {\kf100}
        assert "\\kf" in ass


class TestAlignmentResultExportJSON:
    """Tests for JSON export."""

    def test_to_json(self, alignment_result_simple):
        """Test to_json output."""
        json_str = alignment_result_simple.to_json()
        data = json.loads(json_str)
        assert "words" in data
        assert len(data["words"]) == 6
        assert data["words"][0]["word"] == "hello"

    def test_to_dict(self, alignment_result_simple):
        """Test to_dict output."""
        d = alignment_result_simple.to_dict()
        assert "words" in d
        assert "metadata" in d
        assert "unaligned_regions" in d


class TestAlignmentResultStatistics:
    """Tests for statistics() method."""

    def test_statistics_basic(self, alignment_result_simple):
        """Test basic statistics."""
        stats = alignment_result_simple.statistics()
        assert stats["num_words"] == 6
        assert "time_range" in stats
        assert "word_duration" in stats

    def test_statistics_time_range(self, alignment_result_simple):
        """Test time range in statistics."""
        stats = alignment_result_simple.statistics()
        assert stats["time_range"]["start"] == 2.0
        assert stats["time_range"]["end"] == 8.0

    def test_statistics_word_duration(self, alignment_result_simple):
        """Test word duration statistics."""
        stats = alignment_result_simple.statistics()
        assert "mean" in stats["word_duration"]
        assert "min" in stats["word_duration"]
        assert "max" in stats["word_duration"]
        assert "median" in stats["word_duration"]

    def test_statistics_empty(self):
        """Test statistics with no words."""
        from alignment.base import AlignmentResult
        result = AlignmentResult()
        stats = result.statistics()
        assert "error" in stats

    def test_statistics_with_coverage(self):
        """Test statistics includes coverage when total_words in metadata."""
        from alignment.base import AlignmentResult, AlignedWord
        words = [AlignedWord(word="test", start_frame=0, end_frame=50, index=0)]
        result = AlignmentResult(words=words, metadata={"total_words": 10})
        stats = result.statistics()
        assert stats["coverage_percent"] == 10.0


class TestAlignmentResultRepr:
    """Tests for __repr__() and summary()."""

    def test_repr(self, alignment_result_simple):
        """Test __repr__ output."""
        r = repr(alignment_result_simple)
        assert "AlignmentResult" in r
        assert "6 words" in r

    def test_summary(self, alignment_result_simple):
        """Test summary() output."""
        summary = alignment_result_simple.summary()
        assert "6" in summary  # word count
        assert "Time range" in summary or "time" in summary.lower()


class TestAlignmentResultEdgeCases:
    """Edge case tests."""

    def test_single_word(self):
        """Test result with single word."""
        from alignment.base import AlignmentResult, AlignedWord
        word = AlignedWord(word="only", start_frame=0, end_frame=50)
        result = AlignmentResult(words=[word])
        assert len(result) == 1
        assert result.text == "only"
        assert result.duration == 1.0  # 50 * 0.02

    def test_words_with_gaps(self):
        """Test result with gaps between words."""
        from alignment.base import AlignmentResult, AlignedWord
        words = [
            AlignedWord(word="first", start_frame=0, end_frame=50),
            AlignedWord(word="second", start_frame=200, end_frame=250),  # gap
        ]
        result = AlignmentResult(words=words)
        stats = result.statistics()
        assert "gaps" in stats

    def test_unaligned_regions_in_statistics(self, alignment_result_with_unaligned):
        """Test statistics includes unaligned region count."""
        stats = alignment_result_with_unaligned.statistics()
        assert stats["num_unaligned_regions"] == 2
