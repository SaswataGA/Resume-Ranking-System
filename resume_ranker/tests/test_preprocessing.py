"""Unit tests for preprocessing module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTextCleaner:
    def test_clean_text_basic(self):
        from preprocessing.text_cleaner import clean_text
        result = clean_text("Hello,  World!  This is a TEST.")
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == result.lower()

    def test_clean_text_removes_email(self):
        from preprocessing.text_cleaner import clean_text
        text = "Contact me at john.doe@example.com for more info"
        result = clean_text(text)
        assert "@" not in result or "EMAIL" in result

    def test_clean_text_removes_urls(self):
        from preprocessing.text_cleaner import clean_text
        text = "Visit https://github.com/user/repo for code"
        result = clean_text(text)
        assert "https://" not in result

    def test_tokenize(self):
        from preprocessing.text_cleaner import tokenize
        tokens = tokenize("Python machine learning engineer")
        assert isinstance(tokens, list)
        assert len(tokens) >= 3
        assert "python" in tokens or "Python" in tokens

    def test_remove_stopwords(self):
        from preprocessing.text_cleaner import remove_stopwords
        tokens = ["this", "is", "a", "python", "developer"]
        result = remove_stopwords(tokens)
        assert "python" in result or "developer" in result
        # "this", "is", "a" are stopwords
        assert "this" not in result

    def test_lemmatize_tokens(self):
        from preprocessing.text_cleaner import lemmatize_tokens
        tokens = ["running", "models", "engineers"]
        result = lemmatize_tokens(tokens)
        assert isinstance(result, list)
        assert len(result) == len(tokens)

    def test_preprocess_text_returns_string(self):
        from preprocessing.text_cleaner import preprocess_text
        text = "Experienced Python developer with machine learning skills"
        result = preprocess_text(text, return_tokens=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_preprocess_text_returns_tokens(self):
        from preprocessing.text_cleaner import preprocess_text
        text = "Python machine learning deep learning"
        result = preprocess_text(text, return_tokens=True)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_preprocess_empty_text(self):
        from preprocessing.text_cleaner import preprocess_text
        assert preprocess_text("") == ""
        assert preprocess_text("", return_tokens=True) == []

    def test_preprocess_for_tfidf(self):
        from preprocessing.text_cleaner import preprocess_for_tfidf
        result = preprocess_for_tfidf("Python developer with 5 years experience")
        assert isinstance(result, str)

    def test_preprocess_for_bert(self):
        from preprocessing.text_cleaner import preprocess_for_bert
        result = preprocess_for_bert("Python developer with 5 years experience")
        assert isinstance(result, str)
        assert len(result) > 0


class TestSectionExtractor:
    def test_extract_sections_basic(self):
        from preprocessing.section_extractor import extract_sections
        text = """SKILLS
Python, Java, SQL

EXPERIENCE
Software Engineer at TechCorp 2020-2023

EDUCATION
BS Computer Science, MIT 2020
"""
        sections = extract_sections(text)
        assert isinstance(sections, dict)
        assert "full_text" in sections
        assert sections["full_text"] == text

    def test_extract_sections_has_keys(self):
        from preprocessing.section_extractor import extract_sections
        sections = extract_sections("Some resume text")
        required_keys = ["skills", "experience", "education", "certifications"]
        for key in required_keys:
            assert key in sections

    def test_extract_sections_empty_text(self):
        from preprocessing.section_extractor import extract_sections
        sections = extract_sections("")
        assert sections["skills"] == ""
        assert sections["experience"] == ""

    def test_get_section_text(self):
        from preprocessing.section_extractor import get_section_text
        sections = {"skills": "Python, Java", "experience": "5 years"}
        assert get_section_text(sections, "skills") == "Python, Java"
        assert get_section_text(sections, "missing") == ""


class TestPDFParser:
    def test_parse_txt_file(self, tmp_path):
        from preprocessing.pdf_parser import parse_resume_file
        # Create a temp txt file
        txt_file = tmp_path / "test_resume.txt"
        txt_file.write_text("Python Developer\nSkills: Python, SQL\nExperience: 3 years")
        result = parse_resume_file(txt_file)
        assert "Python" in result
        assert "SQL" in result

    def test_parse_nonexistent_file(self):
        from preprocessing.pdf_parser import parse_resume_file
        result = parse_resume_file(Path("/nonexistent/file.txt"))
        assert result == ""

    def test_parse_unsupported_format(self, tmp_path):
        from preprocessing.pdf_parser import parse_resume_file
        weird_file = tmp_path / "test.xyz"
        weird_file.write_text("content")
        result = parse_resume_file(weird_file)
        assert result == ""
