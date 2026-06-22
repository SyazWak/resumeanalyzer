"""Tests for TextExtractor module."""

import pytest

from utils.text_extractor import TextExtractor


@pytest.fixture
def extractor():
    """Create a TextExtractor instance."""
    return TextExtractor()


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a sample TXT file."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("John Doe\nSoftware Engineer\nPython, JavaScript")
    return file_path


@pytest.fixture
def sample_empty_file(tmp_path):
    """Create an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    return file_path


class TestTextExtractorInit:
    def test_default_preferred_engine(self, extractor):
        assert extractor.preferred_pdf_engine == "pymupdf"

    def test_supported_formats(self, extractor):
        assert ".pdf" in extractor.supported_formats
        assert ".txt" in extractor.supported_formats
        assert ".docx" in extractor.supported_formats


class TestExtractFromTxt:
    def test_extract_text(self, extractor, sample_txt_file):
        text = extractor.extract_text(sample_txt_file)
        assert "John Doe" in text
        assert "Software Engineer" in text

    def test_extract_empty_file(self, extractor, sample_empty_file):
        text = extractor.extract_text(sample_empty_file)
        assert text == ""


class TestValidateFile:
    def test_valid_file(self, extractor, sample_txt_file):
        is_valid, error = extractor.validate_file(sample_txt_file)
        assert is_valid is True
        assert error == "File is valid"

    def test_nonexistent_file(self, extractor):
        is_valid, error = extractor.validate_file("/nonexistent/file.txt")
        assert is_valid is False
        assert "does not exist" in error

    def test_empty_file(self, extractor, sample_empty_file):
        is_valid, error = extractor.validate_file(sample_empty_file)
        assert is_valid is False
        assert "empty" in error

    def test_unsupported_format(self, tmp_path):
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content")
        extractor = TextExtractor()
        is_valid, error = extractor.validate_file(file_path)
        assert is_valid is False
        assert "Unsupported" in error


class TestSupportedFormats:
    def test_get_supported_formats(self, extractor):
        formats = extractor.get_supported_formats()
        assert ".pdf" in formats
        assert ".txt" in formats
        assert ".docx" in formats
