"""
Integration tests for export functionality.

Tests cover:
- Audacity label export
- SRT subtitle export
- TextGrid export
- CTM export
- ASS karaoke subtitle export
- JSON export
"""

import pytest
import json
import tempfile
from pathlib import Path

from test_utils import requires_torch


class TestAudacityExport:
    """Tests for Audacity label export."""

    def test_to_audacity_labels_basic(self, alignment_result_simple):
        """Test basic Audacity label generation."""
        labels = alignment_result_simple.to_audacity_labels()
        lines = labels.strip().split("\n")

        assert len(lines) == 6
        assert "hello" in lines[0]

    def test_to_audacity_labels_format(self, alignment_result_simple):
        """Test Audacity label format (tab-separated)."""
        labels = alignment_result_simple.to_audacity_labels()
        lines = labels.strip().split("\n")

        # Each line should have 3 tab-separated fields
        for line in lines:
            parts = line.split("\t")
            assert len(parts) == 3

            # First two should be floats
            start = float(parts[0])
            end = float(parts[1])
            assert start < end

    def test_to_audacity_labels_times(self, alignment_result_simple):
        """Test Audacity label times are correct."""
        labels = alignment_result_simple.to_audacity_labels()
        lines = labels.strip().split("\n")

        # First word "hello" at frame 100-150, should be 2.0-3.0s
        parts = lines[0].split("\t")
        assert float(parts[0]) == 2.0
        assert float(parts[1]) == 3.0

    def test_to_audacity_labels_empty(self):
        """Test Audacity labels with empty result."""
        from alignment.base import AlignmentResult

        result = AlignmentResult()
        labels = result.to_audacity_labels()
        assert labels == ""

    def test_save_audacity_labels(self, alignment_result_simple):
        """Test saving Audacity labels to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            saved_path = alignment_result_simple.save_audacity_labels(path)
            assert saved_path == path
            assert Path(path).exists()

            content = Path(path).read_text()
            assert "hello" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestAudacityUtilsFunctions:
    """Tests for audacity.py utility functions."""

    def test_get_audacity_labels_from_dict(self, aligned_words_list):
        """Test get_audacity_labels with word dictionary."""
        from visualization_utils.audacity import get_audacity_labels

        # Create word alignment dict
        word_alignment = {i: w for i, w in enumerate(aligned_words_list)}

        labels = get_audacity_labels(word_alignment)
        lines = labels.strip().split("\n")

        assert len(lines) == 6

    def test_get_audacity_labels_custom_frame_duration(self, aligned_words_list):
        """Test get_audacity_labels with custom frame duration."""
        from visualization_utils.audacity import get_audacity_labels

        word_alignment = {0: aligned_words_list[0]}

        # With frame_duration=0.01, frame 100 = 1.0s
        labels = get_audacity_labels(word_alignment, frame_duration=0.01)
        parts = labels.strip().split("\t")

        assert float(parts[0]) == 1.0  # 100 * 0.01

    def test_save_audacity_labels_function(self, aligned_words_list):
        """Test save_audacity_labels utility function."""
        from visualization_utils.audacity import save_audacity_labels

        word_alignment = {i: w for i, w in enumerate(aligned_words_list)}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            saved_path = save_audacity_labels(word_alignment, path)
            assert Path(saved_path).exists()
        finally:
            Path(path).unlink(missing_ok=True)


class TestSRTExport:
    """Tests for SRT subtitle export."""

    def test_to_srt_basic(self, alignment_result_simple):
        """Test basic SRT generation."""
        srt = alignment_result_simple.to_srt(words_per_subtitle=3)

        assert "1" in srt  # Subtitle number
        assert "-->" in srt  # Time separator
        assert "hello" in srt.lower()

    def test_to_srt_format(self, alignment_result_simple):
        """Test SRT timestamp format."""
        srt = alignment_result_simple.to_srt(words_per_subtitle=6)
        lines = srt.strip().split("\n")

        # First line should be subtitle number
        assert lines[0] == "1"

        # Second line should have timestamp
        assert "-->" in lines[1]

        # Timestamp format: HH:MM:SS,mmm
        parts = lines[1].split(" --> ")
        assert len(parts) == 2
        assert ":" in parts[0]
        assert "," in parts[0]

    def test_to_srt_words_per_subtitle(self, alignment_result_simple):
        """Test SRT with different words_per_subtitle."""
        srt1 = alignment_result_simple.to_srt(words_per_subtitle=2)
        srt2 = alignment_result_simple.to_srt(words_per_subtitle=6)

        # More subtitles with fewer words per subtitle
        count1 = srt1.count("-->")
        count2 = srt2.count("-->")
        assert count1 >= count2

    def test_save_srt(self, alignment_result_simple):
        """Test saving SRT to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            path = f.name

        try:
            saved_path = alignment_result_simple.save_srt(path)
            assert saved_path == path
            assert Path(path).exists()

            content = Path(path).read_text()
            assert "-->" in content
        finally:
            Path(path).unlink(missing_ok=True)


class TestTextGridExport:
    """Tests for Praat TextGrid export."""

    def test_to_textgrid_basic(self, alignment_result_simple):
        """Test basic TextGrid generation."""
        tg = alignment_result_simple.to_textgrid()

        assert 'File type = "ooTextFile"' in tg
        assert 'Object class = "TextGrid"' in tg
        assert 'name = "words"' in tg

    def test_to_textgrid_contains_words(self, alignment_result_simple):
        """Test TextGrid contains aligned words."""
        tg = alignment_result_simple.to_textgrid()

        assert "hello" in tg
        assert "world" in tg
        assert "test" in tg

    def test_to_textgrid_interval_format(self, alignment_result_simple):
        """Test TextGrid interval format."""
        tg = alignment_result_simple.to_textgrid()

        assert "xmin" in tg
        assert "xmax" in tg
        assert "intervals" in tg

    def test_to_textgrid_empty(self):
        """Test TextGrid with empty result."""
        from alignment.base import AlignmentResult

        result = AlignmentResult()
        tg = result.to_textgrid()
        assert tg == ""

    def test_save_textgrid(self, alignment_result_simple):
        """Test saving TextGrid to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".TextGrid", delete=False) as f:
            path = f.name

        try:
            saved_path = alignment_result_simple.save_textgrid(path)
            assert saved_path == path
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)


class TestCTMExport:
    """Tests for CTM format export."""

    def test_to_ctm_basic(self, alignment_result_simple):
        """Test basic CTM generation."""
        ctm = alignment_result_simple.to_ctm(audio_id="test_audio")
        lines = ctm.strip().split("\n")

        assert len(lines) == 6
        assert "test_audio" in lines[0]

    def test_to_ctm_format(self, alignment_result_simple):
        """Test CTM format: <audio_id> <channel> <start> <duration> <word>."""
        ctm = alignment_result_simple.to_ctm()
        lines = ctm.strip().split("\n")

        parts = lines[0].split()
        assert len(parts) >= 5  # May have optional confidence

        # Check fields
        assert parts[1] == "1"  # channel
        float(parts[2])  # start (should be parseable as float)
        float(parts[3])  # duration (should be parseable as float)
        assert parts[4] == "hello"

    def test_to_ctm_with_scores(self):
        """Test CTM includes confidence when available."""
        from alignment.base import AlignmentResult, AlignedWord

        word = AlignedWord(word="test", start_frame=0, end_frame=50, score=0.95)
        result = AlignmentResult(words=[word])

        ctm = result.to_ctm()
        # Should contain the confidence score
        assert "0.95" in ctm

    def test_to_ctm_custom_audio_id(self, alignment_result_simple):
        """Test CTM with custom audio ID."""
        ctm = alignment_result_simple.to_ctm(audio_id="my_recording_001")
        assert "my_recording_001" in ctm

    def test_save_ctm(self, alignment_result_simple):
        """Test saving CTM to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ctm", delete=False) as f:
            path = f.name

        try:
            saved_path = alignment_result_simple.save_ctm(path)
            assert saved_path == path
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)


class TestASSExport:
    """Tests for ASS karaoke subtitle export."""

    def test_to_ass_basic(self, alignment_result_simple):
        """Test basic ASS generation."""
        ass = alignment_result_simple.to_ass()

        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass
        assert "Dialogue:" in ass

    def test_to_ass_karaoke_tags(self, alignment_result_simple):
        """Test ASS karaoke timing tags."""
        ass = alignment_result_simple.to_ass()

        # Should contain karaoke tags like {\kf100}
        assert "\\kf" in ass

    def test_to_ass_style_parameters(self, alignment_result_simple):
        """Test ASS custom style parameters."""
        ass = alignment_result_simple.to_ass(
            font_name="Times New Roman",
            font_size=36,
        )

        assert "Times New Roman" in ass
        assert "36" in ass

    def test_to_ass_words_per_line(self, alignment_result_simple):
        """Test ASS with different words per line."""
        ass1 = alignment_result_simple.to_ass(words_per_line=2)
        ass2 = alignment_result_simple.to_ass(words_per_line=6)

        # More dialogue lines with fewer words
        count1 = ass1.count("Dialogue:")
        count2 = ass2.count("Dialogue:")
        assert count1 >= count2

    def test_save_ass(self, alignment_result_simple):
        """Test saving ASS to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ass", delete=False) as f:
            path = f.name

        try:
            saved_path = alignment_result_simple.save_ass(path)
            assert saved_path == path
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)


class TestJSONExport:
    """Tests for JSON export."""

    def test_to_json_basic(self, alignment_result_simple):
        """Test basic JSON generation."""
        json_str = alignment_result_simple.to_json()
        data = json.loads(json_str)

        assert "words" in data
        assert len(data["words"]) == 6

    def test_to_json_word_fields(self, alignment_result_simple):
        """Test JSON word fields."""
        json_str = alignment_result_simple.to_json()
        data = json.loads(json_str)

        word = data["words"][0]
        assert "word" in word
        assert "start" in word
        assert "end" in word
        assert word["word"] == "hello"

    def test_to_dict(self, alignment_result_simple):
        """Test to_dict method."""
        d = alignment_result_simple.to_dict()

        assert "words" in d
        assert "metadata" in d
        assert "unaligned_regions" in d

    def test_save_json(self, alignment_result_simple):
        """Test saving JSON to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            saved_path = alignment_result_simple.save_json(path)
            assert saved_path == path
            assert Path(path).exists()

            with open(path) as f:
                data = json.load(f)
            assert "words" in data
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_json_indentation(self, alignment_result_simple):
        """Test JSON indentation."""
        json_str = alignment_result_simple.to_json(indent=4)
        lines = json_str.split("\n")

        # Indented JSON should have multiple lines
        assert len(lines) > 1


class TestExportRoundTrip:
    """Tests for export/import round trips."""

    def test_json_roundtrip_preserves_words(self, alignment_result_simple):
        """Test JSON export/import preserves word data."""
        json_str = alignment_result_simple.to_json()
        data = json.loads(json_str)

        # Verify all words preserved
        words = [w["word"] for w in data["words"]]
        assert words == ["hello", "world", "this", "is", "a", "test"]

    def test_json_roundtrip_preserves_times(self, alignment_result_simple):
        """Test JSON export preserves time data."""
        json_str = alignment_result_simple.to_json()
        data = json.loads(json_str)

        # First word at 2.0-3.0s
        assert data["words"][0]["start"] == 2.0
        assert data["words"][0]["end"] == 3.0
