"""
Extract structured sections from raw resume text:
  - Skills
  - Experience
  - Education
  - Certifications
  - Summary / Objective
"""

import re
from typing import Dict, List, Optional

from utils.logger import log


# ─── Section Header Patterns ──────────────────────────────────────────────────

SECTION_PATTERNS: Dict[str, List[str]] = {
    "skills": [
        r"(technical\s*skills?|skills?|core\s*competenc(?:y|ies)|technologies|tools?\s*&?\s*technologies?|"
        r"programming\s*languages?|key\s*skills?|expertise|proficiencies|competencies)",
    ],
    "experience": [
        r"(work\s*experience|professional\s*experience|employment\s*history|career\s*history|"
        r"work\s*history|experience|positions?\s*held|relevant\s*experience)",
    ],
    "education": [
        r"(education(?:al)?\s*(?:background|qualifications?)?|academic\s*(?:background|qualifications?)?|"
        r"degrees?|schooling|qualifications?)",
    ],
    "certifications": [
        r"(certifications?|certificates?|licenses?|professional\s*development|accreditations?|"
        r"credentials?|courses?\s*&?\s*certifications?)",
    ],
    "summary": [
        r"(summary|objective|profile|about\s*me|professional\s*summary|career\s*objective|"
        r"executive\s*summary|personal\s*statement)",
    ],
    "projects": [
        r"(projects?|personal\s*projects?|side\s*projects?|open\s*source|portfolio)",
    ],
    "awards": [
        r"(awards?|honors?|achievements?|accomplishments?|recognition)",
    ],
}

# Build combined pattern
_ALL_HEADERS = "|".join(
    "|".join(patterns)
    for patterns in SECTION_PATTERNS.values()
)

_SECTION_SPLIT_RE = re.compile(
    r"(?m)^(?P<header>(?:" + _ALL_HEADERS + r"))\s*[:\-]?\s*$",
    re.IGNORECASE,
)


def _detect_section_name(header_text: str) -> Optional[str]:
    """Map a detected header text to a canonical section name."""
    h = header_text.lower().strip()
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, h, re.IGNORECASE):
                return section
    return None


def extract_sections(raw_text: str) -> Dict[str, str]:
    """
    Extract structured sections from resume text.
    
    Returns:
        dict with keys: skills, experience, education, certifications,
                        summary, projects, awards, other
    """
    sections: Dict[str, str] = {
        "skills": "",
        "experience": "",
        "education": "",
        "certifications": "",
        "summary": "",
        "projects": "",
        "awards": "",
        "other": "",
        "full_text": raw_text,
    }

    if not raw_text:
        return sections

    lines = raw_text.split("\n")
    current_section = "other"
    current_lines: List[str] = []
    section_content: Dict[str, List[str]] = {s: [] for s in sections}

    # Heuristic: detect lines that look like section headers
    def _is_header(line: str) -> Optional[str]:
        stripped = line.strip()
        if len(stripped) < 3 or len(stripped) > 80:
            return None
        # Check if line is a standalone header (ends with optional colon)
        candidate = re.sub(r"[:.\-_\s]+$", "", stripped)
        detected = _detect_section_name(candidate)
        return detected

    for line in lines:
        detected = _is_header(line)
        if detected is not None:
            # Save current section
            if current_lines:
                section_content[current_section].extend(current_lines)
            current_section = detected
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        section_content[current_section].extend(current_lines)

    # Join sections
    for sec, lines_list in section_content.items():
        text = "\n".join(lines_list).strip()
        if text:
            sections[sec] = text

    # If skills section is empty, try to find inline skill lists
    if not sections["skills"]:
        sections["skills"] = _extract_inline_skills(raw_text)

    # If experience is empty but full text has dates, use full text
    if not sections["experience"]:
        sections["experience"] = _extract_experience_fallback(raw_text)

    return sections


def _extract_inline_skills(text: str) -> str:
    """Try to find skill-like content even without a clear header."""
    # Look for lines with comma-separated tech terms
    skill_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Lines with multiple comma-separated items likely are skill lists
        items = [i.strip() for i in stripped.split(",")]
        if len(items) >= 3:
            skill_lines.append(stripped)
    return "\n".join(skill_lines)


def _extract_experience_fallback(text: str) -> str:
    """Extract lines with date patterns as experience indicator."""
    date_pattern = re.compile(
        r"\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(19|20)?\d{2}\b",
        re.IGNORECASE,
    )
    experience_lines = []
    in_block = False
    for line in text.split("\n"):
        if date_pattern.search(line):
            in_block = True
        if in_block:
            experience_lines.append(line)
            if not line.strip():
                in_block = False
    return "\n".join(experience_lines)


def get_section_text(sections: Dict[str, str], section_name: str) -> str:
    """Safely get section text."""
    return sections.get(section_name, "")
