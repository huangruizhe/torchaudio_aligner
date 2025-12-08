"""
Tests for text loading utilities.

Tests cover:
- load_text_from_file
- load_text_from_url (if available)
- load_text_from_pdf (if available)
- load_text convenience function
- get_available_loaders
"""

import pytest

# Import markers from conftest
from conftest import TORCH_AVAILABLE

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch required for text_frontend imports"
)


class TestGetAvailableLoaders:
    """Tests for get_available_loaders function."""

    def test_returns_dict(self):
        """Test that get_available_loaders returns a dict."""
        from text_frontend.loaders import get_available_loaders

        loaders = get_available_loaders()
        assert isinstance(loaders, dict)

    def test_file_always_available(self):
        """Test that file loader is always available."""
        from text_frontend.loaders import get_available_loaders

        loaders = get_available_loaders()
        assert loaders["file"] is True

    def test_contains_expected_keys(self):
        """Test that result contains expected loader keys."""
        from text_frontend.loaders import get_available_loaders

        loaders = get_available_loaders()
        expected_keys = ["file", "url", "pdf", "ocr"]

        for key in expected_keys:
            assert key in loaders


class TestLoadTextFromFile:
    """Tests for load_text_from_file function."""

    def test_load_text_file(self, tmp_path):
        """Test loading text from a simple text file."""
        from text_frontend.loaders import load_text_from_file

        # Create test file
        test_file = tmp_path / "test.txt"
        test_content = "Hello World\nThis is a test."
        test_file.write_text(test_content)

        # Load and verify
        loaded = load_text_from_file(test_file)
        assert loaded == test_content

    def test_load_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent file."""
        from text_frontend.loaders import load_text_from_file

        fake_path = tmp_path / "nonexistent.txt"
        with pytest.raises(FileNotFoundError):
            load_text_from_file(fake_path)

    def test_load_unicode_content(self, tmp_path):
        """Test loading file with Unicode content."""
        from text_frontend.loaders import load_text_from_file

        test_file = tmp_path / "unicode.txt"
        test_content = "Café résumé naïve 日本語 한국어"
        test_file.write_text(test_content, encoding="utf-8")

        loaded = load_text_from_file(test_file, encoding="utf-8")
        assert loaded == test_content

    def test_load_empty_file(self, tmp_path):
        """Test loading empty file."""
        from text_frontend.loaders import load_text_from_file

        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        loaded = load_text_from_file(test_file)
        assert loaded == ""

    def test_load_with_path_object(self, tmp_path):
        """Test load accepts Path objects."""
        from pathlib import Path
        from text_frontend.loaders import load_text_from_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        loaded = load_text_from_file(Path(test_file))
        assert loaded == "test content"

    def test_load_with_string_path(self, tmp_path):
        """Test load accepts string paths."""
        from text_frontend.loaders import load_text_from_file

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        loaded = load_text_from_file(str(test_file))
        assert loaded == "test content"


class TestLoadText:
    """Tests for load_text convenience function."""

    def test_load_text_autodetects_txt(self, tmp_path):
        """Test load_text auto-detects text files."""
        from text_frontend.loaders import load_text

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        loaded = load_text(test_file)
        assert loaded == "Hello World"

    def test_load_text_with_pdf_extension(self, tmp_path):
        """Test load_text detects PDF extension (may fail without pypdf)."""
        from text_frontend.loaders import load_text, get_available_loaders

        if not get_available_loaders()["pdf"]:
            pytest.skip("pypdf not available")

        # This would need an actual PDF file to test properly
        # Just verify it tries the PDF loader
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"")  # Empty file

        with pytest.raises(Exception):  # Will fail on empty/invalid PDF
            load_text(fake_pdf)


class TestLoadTextFromUrl:
    """Tests for URL loading (if available)."""

    def test_url_loader_availability(self):
        """Test checking URL loader availability."""
        from text_frontend.loaders import get_available_loaders

        loaders = get_available_loaders()
        # Just check we can query it
        assert "url" in loaders

    def test_load_from_url_import_error(self):
        """Test that load_text_from_url raises ImportError if dependencies missing."""
        from text_frontend.loaders import load_text_from_url, get_available_loaders

        if get_available_loaders()["url"]:
            pytest.skip("URL loader is available, cannot test import error")

        with pytest.raises(ImportError):
            load_text_from_url("https://example.com")


class TestLoadTextFromPdf:
    """Tests for PDF loading (if available)."""

    def test_pdf_loader_availability(self):
        """Test checking PDF loader availability."""
        from text_frontend.loaders import get_available_loaders

        loaders = get_available_loaders()
        assert "pdf" in loaders

    def test_load_from_pdf_file_not_found(self, tmp_path):
        """Test FileNotFoundError for non-existent PDF."""
        from text_frontend.loaders import load_text_from_pdf, get_available_loaders

        if not get_available_loaders()["pdf"]:
            pytest.skip("pypdf not available")

        fake_path = tmp_path / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError):
            load_text_from_pdf(fake_path)
