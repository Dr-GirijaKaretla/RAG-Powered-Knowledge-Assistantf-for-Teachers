"""
Semantic retrieval with optional cross-encoder re-ranking.

Wraps the embedding + vector-store search flow and adds
post-retrieval filtering, re-ranking, and context formatting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from sentence_transformers import CrossEncoder

from src.embedding.embedder import TextEmbedder
from src.utils.logger import setup_logger
from src.vectorstore.store import VectorStore

logger = setup_logger(__name__)


@st.cache_resource(show_spinner="Loading re-ranker model …")
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load and cache the cross-encoder re-ranker.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the cross-encoder.

    Returns
    -------
    CrossEncoder
        The loaded CrossEncoder model.
    """
    logger.info("Loading cross-encoder '%s' …", model_name)
    model = CrossEncoder(model_name)
    logger.info("Cross-encoder loaded.")
    return model


class Retriever:
    """Top-K semantic retrieval with optional cross-encoder re-ranking.

    Parameters
    ----------
    config : dict
        Full application config; the ``retrieval`` section is used.
    embedder : TextEmbedder
        The shared embedding model instance.
    vectorstore : VectorStore
        The shared vector store instance.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        embedder: TextEmbedder,
        vectorstore: VectorStore,
    ) -> None:
        ret_cfg = config.get("retrieval", {})
        self.top_k: int = ret_cfg.get("top_k", 5)
        self.use_rerank: bool = ret_cfg.get("rerank", True)
        self.reranker_model_name: str = ret_cfg.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.similarity_threshold: float = ret_cfg.get("similarity_threshold", 0.3)

        self.embedder = embedder
        self.vectorstore = vectorstore

        self._cross_encoder: Optional[CrossEncoder] = None
        if self.use_rerank:
            self._cross_encoder = _load_cross_encoder(self.reranker_model_name)

    # ------------------------------------------------------------------ #
    #  Main retrieval method                                              #
    # ------------------------------------------------------------------ #

    def retrieve(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for *query*.

        Steps:
        1. Embed the query.
        2. Search for ``top_k * 2`` candidates from vector store.
        3. (Optional) Re-rank with cross-encoder.
        4. Filter by similarity threshold.
        5. Return final ``top_k`` results.

        Parameters
        ----------
        query:
            The user's natural-language question.
        top_k:
            Override for the default top-K value.

        Returns
        -------
        List[Dict[str, Any]]
            Ranked result dicts with ``chunk_id``, ``text``, ``source``,
            ``page``, ``similarity_score``, and optionally ``rerank_score``.
        """
        k = top_k or self.top_k
        query_embedding = self.embedder.embed_text(query)

        # Fetch 2× candidates to give re-ranker room
        candidates = self.vectorstore.search(query_embedding, top_k=k * 2)

        # Convert ChromaDB distances to similarity scores (cosine: similarity = 1 - distance)
        for c in candidates:
            c["similarity_score"] = round(1.0 - c.get("distance", 0.0), 4)
            c["rerank_score"] = None

        # Optional re-ranking
        if self.use_rerank and self._cross_encoder is not None and candidates:
            candidates = self.rerank(query, candidates)

        # Filter by threshold
        candidates = self.filter_by_threshold(candidates, self.similarity_threshold)

        # Return top-k
        final = candidates[:k]
        logger.info(
            "Retrieved %d chunks for query (candidates=%d, rerank=%s)",
            len(final),
            len(candidates),
            self.use_rerank,
        )
        return final

    # ------------------------------------------------------------------ #
    #  Re-ranking                                                        #
    # ------------------------------------------------------------------ #

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank *candidates* using the cross-encoder model.

        Parameters
        ----------
        query:
            Original user query.
        candidates:
            Initial retrieval candidates.

        Returns
        -------
        List[Dict[str, Any]]
            Candidates sorted by cross-encoder score (descending).
        """
        if not self._cross_encoder or not candidates:
            return candidates

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = round(float(score), 4)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------ #
    #  Filtering                                                         #
    # ------------------------------------------------------------------ #

    def filter_by_threshold(
        self,
        results: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Remove results below a similarity threshold.

        Parameters
        ----------
        results:
            Scored result dicts.
        threshold:
            Minimum similarity score to keep.

        Returns
        -------
        List[Dict[str, Any]]
            Filtered results.
        """
        thr = threshold if threshold is not None else self.similarity_threshold
        return [r for r in results if r.get("similarity_score", 0) >= thr]

    # ------------------------------------------------------------------ #
    #  Context formatting                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_context(results: List[Dict[str, Any]]) -> str:
        """Concatenate retrieved chunks into a single context string.

        Each chunk is followed by an inline source citation in the form:
        ``[Source: filename, Page N]``

        Parameters
        ----------
        results:
            Retrieved and ranked chunk dicts.

        Returns
        -------
        str
            Formatted context ready for the prompt builder.
        """
        if not results:
            return ""

        parts: List[str] = []
        for r in results:
            citation = f"[Source: {r['source']}, Page {r['page']}]"
            parts.append(f"{r['text']}\n{citation}")
        return "\n\n---\n\n".join(parts)
