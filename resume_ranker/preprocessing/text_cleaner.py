"""
Text cleaning, tokenization, stopword removal, and lemmatization pipeline.
"""

import re
import string
from typing import List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from utils.logger import log
import config


def _ensure_nltk_resources():
    """Download required NLTK resources if missing."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            log.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name, quiet=True)


_ensure_nltk_resources()
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words(config.NLTK_STOPWORDS_LANG))

# Keep important technical stopwords
_KEEP_WORDS = {
    "c", "r", "go", "no", "not", "java", "sql", "aws", "gcp", "api",
    "ml", "ai", "dl", "cv", "nlp", "ci", "cd", "db",
}
_FILTERED_STOP_WORDS = _stop_words - _KEEP_WORDS


def remove_emails(text: str) -> str:
    return re.sub(r"\S+@\S+", " EMAIL ", text)


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+|linkedin\S*|github\S*", " URL ", text)


def remove_phone_numbers(text: str) -> str:
    return re.sub(r"\b[\+\(]?[\d\s\-\(\)\.]{7,15}\d\b", " PHONE ", text)


def remove_special_chars(text: str, keep_hyphens: bool = True) -> str:
    if keep_hyphens:
        text = re.sub(r"[^\w\s\-\+\#\.]", " ", text)
    else:
        text = re.sub(r"[^\w\s]", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_emails_flag: bool = True,
    remove_urls_flag: bool = True,
    remove_phones: bool = True,
) -> str:
    """Full text cleaning pipeline."""
    if not text:
        return ""

    if remove_emails_flag:
        text = remove_emails(text)
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_phones:
        text = remove_phone_numbers(text)

    if lowercase:
        text = text.lower()

    text = remove_special_chars(text)
    text = normalize_whitespace(text)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords while keeping technical terms."""
    return [t for t in tokens if t not in _FILTERED_STOP_WORDS]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Lemmatize tokens."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def filter_tokens(tokens: List[str], min_length: int = None) -> List[str]:
    """Remove short tokens and pure punctuation."""
    min_len = min_length or config.MIN_TOKEN_LENGTH
    return [
        t for t in tokens
        if len(t) >= min_len and not all(c in string.punctuation for c in t)
    ]


def preprocess_text(
    text: str,
    remove_stops: bool = None,
    lemmatize: bool = None,
    return_tokens: bool = False,
) -> str | List[str]:
    """
    Full preprocessing pipeline.
    
    Args:
        text: Raw input text
        remove_stops: Whether to remove stopwords (default from config)
        lemmatize: Whether to lemmatize (default from config)
        return_tokens: If True, return token list; else return joined string
    
    Returns:
        Preprocessed string or list of tokens
    """
    if not text:
        return [] if return_tokens else ""

    do_remove_stops = remove_stops if remove_stops is not None else config.REMOVE_STOPWORDS
    do_lemmatize = lemmatize if lemmatize is not None else config.LEMMATIZE

    text = clean_text(text)
    tokens = tokenize(text)
    tokens = filter_tokens(tokens)

    if do_remove_stops:
        tokens = remove_stopwords(tokens)

    if do_lemmatize:
        tokens = lemmatize_tokens(tokens)

    tokens = filter_tokens(tokens)

    if return_tokens:
        return tokens

    return " ".join(tokens)


def preprocess_for_tfidf(text: str) -> str:
    """Preprocessing tuned for TF-IDF (keep more tokens)."""
    return preprocess_text(text, remove_stops=True, lemmatize=True, return_tokens=False)


def preprocess_for_bert(text: str) -> str:
    """Preprocessing tuned for BERT (light cleaning, preserve meaning)."""
    if not text:
        return ""
    text = remove_emails(text)
    text = remove_urls(text)
    text = remove_phone_numbers(text)
    text = normalize_whitespace(text)
    # Keep original case for BERT
    if len(text) > config.BERT_MAX_SEQ_LENGTH * 4:
        text = text[:config.BERT_MAX_SEQ_LENGTH * 4]
    return text
