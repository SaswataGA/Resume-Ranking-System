"""General utility helpers for the resume ranking system."""

import re
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from utils.logger import log


def save_joblib(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    log.info(f"Saved object to {path}")


def load_joblib(path: Union[str, Path]) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)
    log.info(f"Loaded object from {path}")
    return obj


def save_json(data: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Saved JSON to {path}")


def load_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    log.info(f"Saved pickle to {path}")


def load_pickle(path: Union[str, Path]) -> Any:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s\-\+\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def flatten_list(nested: List[List]) -> List:
    return [item for sublist in nested for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize an array of scores to [0, 1]."""
    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:
        return np.ones_like(scores) * 0.5
    return (scores - min_s) / (max_s - min_s)


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Return 1-based ranks (1 = best) from scores array."""
    order = np.argsort(scores)[::-1]
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def truncate_text(text: str, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def read_text_file(path: Union[str, Path]) -> str:
    path = Path(path)
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read file with known encodings: {path}")
