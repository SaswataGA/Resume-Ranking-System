"""
Skill extraction and normalization module.
Maps raw skill mentions to canonical skill names using a skill ontology.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from utils.logger import log
import config


class SkillOntology:
    """Loads and provides lookup into the skill ontology."""

    def __init__(self, ontology_path: Path = None):
        path = ontology_path or config.SKILL_ONTOLOGY_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.canonical_skills: Dict[str, List[str]] = data["canonical_skills"]
        self.skill_categories: Dict[str, List[str]] = data["skill_categories"]

        # Build reverse lookup: alias -> canonical
        self._alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.canonical_skills.items():
            self._alias_to_canonical[canonical.lower()] = canonical
            for alias in aliases:
                self._alias_to_canonical[alias.lower()] = canonical

        # All canonical skill names (lowercased set)
        self.all_canonical = set(self.canonical_skills.keys())

        log.info(f"Loaded skill ontology: {len(self.canonical_skills)} canonical skills, "
                 f"{len(self._alias_to_canonical)} total aliases")

    def normalize(self, skill: str) -> Optional[str]:
        """Normalize a skill string to its canonical form."""
        key = skill.lower().strip()
        return self._alias_to_canonical.get(key)

    def get_category(self, canonical_skill: str) -> Optional[str]:
        """Get category for a canonical skill."""
        for cat, skills in self.skill_categories.items():
            if canonical_skill.lower() in [s.lower() for s in skills]:
                return cat
        return None

    def is_skill(self, text: str) -> bool:
        """Check if text matches any known skill."""
        return self.normalize(text) is not None


class SkillExtractor:
    """
    Extracts and normalizes skills from text.
    Uses regex-based matching against the skill ontology.
    """

    def __init__(self, ontology: SkillOntology = None):
        self.ontology = ontology or SkillOntology()
        self._build_patterns()

    def _build_patterns(self):
        """Build regex patterns for all skills and aliases."""
        self._patterns: List[Tuple[re.Pattern, str]] = []

        for canonical, aliases in self.ontology.canonical_skills.items():
            all_variants = [canonical] + aliases
            # Sort by length descending (match longer first)
            all_variants.sort(key=len, reverse=True)
            # Escape and build pattern
            escaped = [re.escape(v) for v in all_variants]
            pattern = re.compile(
                r"(?<![a-z0-9\-])(" + "|".join(escaped) + r")(?![a-z0-9\-])",
                re.IGNORECASE,
            )
            self._patterns.append((pattern, canonical))

        log.debug(f"Built {len(self._patterns)} skill patterns")

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract canonical skill names from text.
        
        Returns:
            Sorted list of unique canonical skill names
        """
        if not text:
            return []

        found: Set[str] = set()
        text_lower = text.lower()

        for pattern, canonical in self._patterns:
            if pattern.search(text_lower):
                found.add(canonical)

        return sorted(found)

    def extract_skills_with_context(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills and return with surrounding context.
        
        Returns:
            dict mapping canonical skill to list of matching contexts
        """
        if not text:
            return {}

        result: Dict[str, List[str]] = {}

        for pattern, canonical in self._patterns:
            matches = pattern.finditer(text)
            contexts = []
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                contexts.append(text[start:end].strip())
            if contexts:
                result[canonical] = contexts

        return result

    def skill_overlap(
        self,
        skills_a: List[str],
        skills_b: List[str],
    ) -> Dict[str, float]:
        """
        Compute skill overlap metrics between two skill sets.
        
        Returns:
            dict with: overlap_count, jaccard, precision, recall, f1
        """
        set_a = set(skills_a)
        set_b = set(skills_b)

        if not set_a and not set_b:
            return {"overlap_count": 0, "jaccard": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": []}

        intersection = set_a & set_b
        union = set_a | set_b

        overlap_count = len(intersection)
        jaccard = len(intersection) / len(union) if union else 0.0
        precision = len(intersection) / len(set_b) if set_b else 0.0  # of job skills, how many candidate has
        recall = len(intersection) / len(set_a) if set_a else 0.0    # of candidate skills, how many match job
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "overlap_count": overlap_count,
            "jaccard": jaccard,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched": sorted(intersection),
            "missing": sorted(set_b - set_a),  # skills in job not in candidate
            "extra": sorted(set_a - set_b),    # skills in candidate not in job
        }

    def categorize_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """Group skills by category."""
        categorized: Dict[str, List[str]] = {}
        for skill in skills:
            cat = self.ontology.get_category(skill) or "other"
            categorized.setdefault(cat, []).append(skill)
        return categorized


# Module-level singleton
_ontology: Optional[SkillOntology] = None
_extractor: Optional[SkillExtractor] = None


def get_skill_extractor() -> SkillExtractor:
    """Get or create singleton SkillExtractor."""
    global _ontology, _extractor
    if _extractor is None:
        _ontology = SkillOntology()
        _extractor = SkillExtractor(_ontology)
    return _extractor
