"""
Sentence-transformer embedding wrapper for the Teacher RAG pipeline.

Loads the embedding model once via ``@st.cache_resource`` and exposes
single-text, batched, and similarity utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@st.cache_resource(show_spinner="Loading embedding model …")
def _load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. ``all-MiniLM-L6-v2``).
    device:
        Torch device string (``cpu`` or ``cuda``).

    Returns
    -------
    SentenceTransformer
        The loaded model instance.
    """
    logger.info("Loading embedding model '%s' on %s …", model_name, device)
    model = SentenceTransformer(model_name, device=device)
    logger.info("Embedding model loaded — dimension=%d", model.get_sentence_embedding_dimension())
    return model


class TextEmbedder:
    """Generate dense vector embeddings for text passages.

    Parameters
    ----------
    config : dict
        Full application configuration; the ``embedding`` section is used.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        emb_cfg = config.get("embedding", {})
        self.model_name: str = emb_cfg.get("embedding_model", "all-MiniLM-L6-v2")
        self.device: str = emb_cfg.get("embedding_device", "cpu")
        self.batch_size: int = emb_cfg.get("batch_size", 32)
        self.model: SentenceTransformer = _load_embedding_model(
            self.model_name, self.device
        )

    # ------------------------------------------------------------------ #
    #  Core API                                                           #
    # ------------------------------------------------------------------ #

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Parameters
        ----------
        text:
            The input sentence or passage.

        Returns
        -------
        np.ndarray
            1-D float array of shape ``(embedding_dim,)``.
        """
        return self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts in memory-efficient batches.

        Parameters
        ----------
        texts:
            List of input strings.

        Returns
        -------
        np.ndarray
            2-D float array of shape ``(len(texts), embedding_dim)``.
        """
        logger.info(
            "Embedding %d texts in batches of %d …", len(texts), self.batch_size
        )
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Parameters
        ----------
        vec1:
            First embedding vector.
        vec2:
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity in ``[-1, 1]``.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get_model_info(self) -> Dict[str, Any]:
        """Return descriptive information about the loaded model.

        Returns
        -------
        Dict[str, Any]
            ``{model_name, embedding_dim, device}``
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": int(self.model.get_sentence_embedding_dimension()),
            "device": self.device,
        }
