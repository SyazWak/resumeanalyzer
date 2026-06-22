"""Tests for ResumeWriter module."""

import json
from unittest.mock import MagicMock

import pytest

from utils.resume_writer import ResumeWriter, SectionRewrite


@pytest.fixture
def mock_analyzer():
    """Create a mocked AdvancedAIAnalyzer."""
    analyzer = MagicMock()
    return analyzer


@pytest.fixture
def writer(mock_analyzer):
    """Create a ResumeWriter with mocked analyzer."""
    return ResumeWriter(ai_analyzer=mock_analyzer)


SAMPLE_RESUME = """John Doe
Software Engineer

Summary
Experienced developer with 5 years of Python experience.

Experience
ABC Company
Software Engineer
- Built web applications using Python and Django
- Improved system performance by 30%

Skills
Python, JavaScript, Django, React, SQL

Education
BS Computer Science, State University
"""


class TestParseResumeSections:
    def test_parse_with_headings(self, writer):
        sections = writer.parse_resume_sections(SAMPLE_RESUME)
        names = [s["name"] for s in sections]
        assert "Summary" in names
        assert "Experience" in names
        assert "Skills" in names
        assert "Education" in names

    def test_parse_no_headings(self, writer):
        text = "Just some text without any headings at all."
        sections = writer.parse_resume_sections(text)
        assert len(sections) == 1
        assert sections[0]["name"] == "Full Resume"

    def test_parse_empty_text(self, writer):
        sections = writer.parse_resume_sections("")
        assert len(sections) == 1
        assert sections[0]["name"] == "Full Resume"


class TestRewriteSection:
    def test_successful_rewrite(self, writer, mock_analyzer):
        mock_analyzer._make_api_request.return_value = json.dumps(
            {
                "improved_text": "Results-driven developer with 5+ years of Python expertise.",
                "explanation": "Added action verb and quantified experience.",
            }
        )

        result = writer._rewrite_section(
            "Summary",
            "Experienced developer with 5 years of Python experience.",
            "Looking for Python developer",
            ["React", "AWS"],
            ["Python", "Django"],
        )

        assert result.section_name == "Summary"
        assert result.improved_text == "Results-driven developer with 5+ years of Python expertise."
        assert result.accepted is True

    def test_rewrite_failure(self, writer, mock_analyzer):
        mock_analyzer._make_api_request.side_effect = Exception("API error")

        result = writer._rewrite_section("Summary", "Some text", "Job description", [], [])

        assert result.accepted is False
        assert "failed" in result.explanation.lower()


class TestGenerateFinalVersion:
    def test_all_accepted(self, writer):
        sections = [
            SectionRewrite("Summary", "Original", "Improved", "Changed", True),
            SectionRewrite("Skills", "Original", "Improved", "Changed", True),
        ]
        result = writer.generate_final_version(sections)
        assert "Improved" in result
        assert "Original" not in result

    def test_mixed_accepted(self, writer):
        sections = [
            SectionRewrite("Summary", "Original", "Improved", "Changed", True),
            SectionRewrite("Skills", "Original", "Improved", "Changed", False),
        ]
        result = writer.generate_final_version(sections)
        assert "Improved" in result  # Summary was accepted
        assert "Original" in result  # Skills was rejected


class TestExportPDF:
    def test_export_pdf(self, writer, tmp_path):
        filepath = str(tmp_path / "test_resume.pdf")
        result = writer.export_pdf("John Doe\nSoftware Engineer", filepath)
        assert result is True
        assert tmp_path.joinpath("test_resume.pdf").exists()


class TestExportDOCX:
    def test_export_docx(self, writer, tmp_path):
        filepath = str(tmp_path / "test_resume.docx")
        result = writer.export_docx("John Doe\nSoftware Engineer", filepath)
        assert result is True
        assert tmp_path.joinpath("test_resume.docx").exists()
