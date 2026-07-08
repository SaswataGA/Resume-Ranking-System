"""
BERT / Sentence Transformer model for semantic resume ranking.
Uses sentence-transformers for dense embeddings + cosine similarity.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.logger import log
from utils.helpers import save_pickle, load_pickle
import config


class BERTSimilarityModel:
    """
    Semantic similarity using Sentence Transformers (BERT-based).
    Produces dense embeddings for job descriptions and resumes,
    then ranks via cosine similarity.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.BERT_MODEL_NAME
        self._model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0

    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                log.info(f"Loading Sentence Transformer: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                log.info("Sentence Transformer loaded successfully")
            except Exception as e:
                log.error(f"Failed to load Sentence Transformer: {e}")
                raise

    def encode(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        Uses cache to avoid recomputing identical texts.
        """
        self._load_model()
        batch_size = batch_size or config.BERT_BATCH_SIZE

        if use_cache:
            embeddings = []
            texts_to_encode = []
            indices_to_encode = []

            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    embeddings.append((i, self._embedding_cache[text]))
                    self._cache_hits += 1
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)

            if texts_to_encode:
                from sentence_transformers import SentenceTransformer
                new_embeddings = self._model.encode(
                    texts_to_encode,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True,
                )
                for text, emb in zip(texts_to_encode, new_embeddings):
                    self._embedding_cache[text] = emb

                for idx, emb in zip(indices_to_encode, new_embeddings):
                    embeddings.append((idx, emb))

            embeddings.sort(key=lambda x: x[0])
            return np.array([e for _, e in embeddings])
        else:
            return self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

    def score(self, job_text: str, resume_texts: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between job description and resumes.
        
        Returns:
            Array of similarity scores [0, 1]
        """
        log.info(f"Computing BERT similarity for {len(resume_texts)} resumes")
        all_texts = [job_text] + resume_texts
        embeddings = self.encode(all_texts, use_cache=True)

        job_emb = embeddings[0:1]
        resume_embs = embeddings[1:]

        similarities = cosine_similarity(job_emb, resume_embs).flatten()

        # Normalize from [-1, 1] to [0, 1]
        similarities = (similarities + 1) / 2

        log.debug(f"BERT scores: min={similarities.min():.4f}, max={similarities.max():.4f}")
        return similarities

    def score_sections(
        self,
        job_sections: Dict[str, str],
        resume_sections: Dict[str, str],
        weights: Dict[str, float] = None,
    ) -> float:
        """
        Compute weighted section-by-section semantic similarity.
        
        Args:
            job_sections: Dict of section_name -> text for job
            resume_sections: Dict of section_name -> text for resume
            weights: Section weights (defaults to config)
            
        Returns:
            Weighted similarity score
        """
        weights = weights or config.SECTION_WEIGHTS
        total_score = 0.0
        total_weight = 0.0

        for section, weight in weights.items():
            job_text = job_sections.get(section, "").strip()
            resume_text = resume_sections.get(section, "").strip()

            if not job_text or not resume_text:
                continue

            sim = self.score(job_text, [resume_text])[0]
            total_score += sim * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_score / total_weight

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        return self.encode([text])[0]

    def save_cache(self, path: Path = None) -> None:
        path = path or config.BERT_EMBEDDINGS_CACHE
        save_pickle(self._embedding_cache, path)
        log.info(f"Saved {len(self._embedding_cache)} cached embeddings")

    def load_cache(self, path: Path = None) -> "BERTSimilarityModel":
        path = path or config.BERT_EMBEDDINGS_CACHE
        if path.exists():
            self._embedding_cache = load_pickle(path)
            log.info(f"Loaded {len(self._embedding_cache)} cached embeddings")
        return self
