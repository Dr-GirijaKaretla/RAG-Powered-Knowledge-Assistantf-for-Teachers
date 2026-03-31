"""
Evaluation metrics for retrieval quality and answer faithfulness.

Provides Recall@K, Precision@K, answer-relevance (cosine),
context-faithfulness (token overlap), and plotting helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import plotly.graph_objects as go

from src.embedding.embedder import TextEmbedder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationMetrics:
    """Compute retrieval and generation quality metrics.

    All plotting methods return Plotly ``Figure`` objects suitable for
    ``st.plotly_chart(fig, use_container_width=True)``.
    """

    # ------------------------------------------------------------------ #
    #  Retrieval metrics                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Recall@K.

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved chunk IDs.
        relevant_ids:
            Set of truly relevant chunk IDs.
        k:
            Cut-off rank.

        Returns
        -------
        float
            Recall score in [0, 1].
        """
        if not relevant_ids:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return len(top_k & relevant_ids) / len(relevant_ids)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int,
    ) -> float:
        """Compute Precision@K.

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved chunk IDs.
        relevant_ids:
            Set of truly relevant chunk IDs.
        k:
            Cut-off rank.

        Returns
        -------
        float
            Precision score in [0, 1].
        """
        if k == 0:
            return 0.0
        top_k = set(retrieved_ids[:k])
        return len(top_k & relevant_ids) / k

    # ------------------------------------------------------------------ #
    #  Answer quality                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def answer_relevance_score(
        question: str,
        answer: str,
        embedder: TextEmbedder,
    ) -> float:
        """Estimate answer relevance via cosine similarity of embeddings.

        Parameters
        ----------
        question:
            The user's question.
        answer:
            The generated answer.
        embedder:
            Embedding model for encoding both strings.

        Returns
        -------
        float
            Cosine similarity in [0, 1] (clamped).
        """
        q_vec = embedder.embed_text(question)
        a_vec = embedder.embed_text(answer)
        sim = float(np.dot(q_vec, a_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(a_vec) + 1e-10))
        return max(0.0, min(1.0, sim))

    @staticmethod
    def context_faithfulness(
        answer: str,
        context: str,
        embedder: Optional[TextEmbedder] = None,
    ) -> float:
        """Check how much of the answer is grounded in the context.

        Uses word-level overlap as a lightweight faithfulness proxy.

        Parameters
        ----------
        answer:
            The generated answer.
        context:
            The retrieved context string.
        embedder:
            Unused here but kept for API compatibility.

        Returns
        -------
        float
            Overlap score in [0, 1].
        """
        if not answer or not context:
            return 0.0

        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())

        # Remove very short tokens (noise)
        answer_tokens = {t for t in answer_tokens if len(t) > 2}
        context_tokens = {t for t in context_tokens if len(t) > 2}

        if not answer_tokens:
            return 0.0

        overlap = answer_tokens & context_tokens
        return len(overlap) / len(answer_tokens)

    # ------------------------------------------------------------------ #
    #  Plotting helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def display_retrieval_metrics(results: List[Dict[str, Any]]) -> go.Figure:
        """Create a horizontal bar chart of per-chunk similarity scores.

        Parameters
        ----------
        results:
            Retrieval results with ``chunk_id`` and ``similarity_score``.

        Returns
        -------
        go.Figure
            A Plotly horizontal bar chart.
        """
        if not results:
            fig = go.Figure()
            fig.add_annotation(text="No retrieval results.", showarrow=False)
            return fig

        labels = [r.get("chunk_id", f"chunk_{i}") for i, r in enumerate(results)]
        scores = [r.get("similarity_score", 0) for r in results]

        # Reverse for top-down display
        labels = labels[::-1]
        scores = scores[::-1]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                marker_color="#6366F1",
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Chunk Similarity Scores",
            xaxis_title="Cosine Similarity",
            yaxis_title="Chunk ID",
            height=max(300, len(results) * 40),
            margin=dict(l=200),
            template="plotly_white",
        )
        return fig

    @staticmethod
    def display_answer_stats(qa_result: Dict[str, Any]) -> go.Figure:
        """Create a pie chart of timing breakdown for a QA result.

        Parameters
        ----------
        qa_result:
            The result dict from :meth:`RAGPipeline.ask`.

        Returns
        -------
        go.Figure
            A Plotly pie chart.
        """
        retrieval_ms = qa_result.get("retrieval_time_ms", 0)
        generation_ms = qa_result.get("generation_time_ms", 0)

        fig = go.Figure(
            go.Pie(
                labels=["Retrieval", "Generation"],
                values=[retrieval_ms, generation_ms],
                marker_colors=["#6366F1", "#8B5CF6"],
                textinfo="label+percent",
                hole=0.4,
            )
        )
        fig.update_layout(
            title="Response Timing Breakdown",
            template="plotly_white",
            height=350,
        )
        return fig
