"""
Robust PDF and TXT resume parser.
Supports pdfplumber (primary) and PyMuPDF (fallback).
"""

import re
from pathlib import Path
from typing import Optional

from utils.logger import log
import config


def parse_pdf_pdfplumber(path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
        return "\n".join(text_parts)
    except Exception as e:
        log.warning(f"pdfplumber failed for {path}: {e}")
        return ""


def parse_pdf_pymupdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        log.warning(f"PyMuPDF failed for {path}: {e}")
        return ""


def parse_txt(path: Path) -> str:
    """Read plain text file."""
    from utils.helpers import read_text_file
    return read_text_file(path)


def parse_docx(path: Path) -> str:
    """Extract text from .docx file."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs]
        return "\n".join(paragraphs)
    except Exception as e:
        log.warning(f"docx parse failed for {path}: {e}")
        return ""


def parse_resume_file(path: Path) -> str:
    """
    Parse resume from any supported format.
    Returns raw text string.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if not path.exists():
        log.error(f"File not found: {path}")
        return ""

    log.info(f"Parsing resume: {path.name}")

    if suffix == ".pdf":
        parser = config.PDF_PARSER
        if parser == "pdfplumber":
            text = parse_pdf_pdfplumber(path)
            if not text.strip():
                log.info(f"Falling back to PyMuPDF for {path.name}")
                text = parse_pdf_pymupdf(path)
        else:
            text = parse_pdf_pymupdf(path)
            if not text.strip():
                text = parse_pdf_pdfplumber(path)
    elif suffix == ".txt":
        text = parse_txt(path)
    elif suffix == ".docx":
        text = parse_docx(path)
    else:
        log.warning(f"Unsupported file format: {suffix}")
        return ""

    if not text.strip():
        log.warning(f"No text extracted from {path.name}")
        return ""

    # Basic post-processing
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    log.debug(f"Extracted {len(text)} chars from {path.name}")
    return text
