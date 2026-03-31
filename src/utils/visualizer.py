"""
Plotly-based visualisation helpers for the Streamlit dashboard.

All methods return :class:`plotly.graph_objects.Figure` instances ready
for ``st.plotly_chart(fig, use_container_width=True)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go

from src.utils.logger import setup_logger
from src.vectorstore.store import VectorStore

logger = setup_logger(__name__)


class Visualizer:
    """Dashboard chart builders for the Teacher RAG application."""

    # ------------------------------------------------------------------ #
    #  Chunk-level charts                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def chunk_size_distribution(chunks: List[Dict[str, Any]]) -> go.Figure:
        """Histogram of chunk character counts.

        Parameters
        ----------
        chunks:
            List of chunk dicts (must have ``char_count``).

        Returns
        -------
        go.Figure
            Plotly histogram.
        """
        sizes = [c.get("char_count", 0) for c in chunks]

        fig = go.Figure(
            go.Histogram(
                x=sizes,
                nbinsx=20,
                marker_color="#6366F1",
                opacity=0.85,
            )
        )
        fig.update_layout(
            title="Chunk Size Distribution",
            xaxis_title="Character Count",
            yaxis_title="Frequency",
            template="plotly_white",
            height=350,
        )
        return fig

    @staticmethod
    def document_chunk_map(vectorstore: VectorStore) -> go.Figure:
        """Bar chart of chunks per document.

        Parameters
        ----------
        vectorstore:
            The vector store instance.

        Returns
        -------
        go.Figure
            Plotly bar chart.
        """
        documents = vectorstore.list_documents()
        counts: List[int] = []
        for doc in documents:
            chunks = vectorstore.get_document_chunks(doc)
            counts.append(len(chunks))

        fig = go.Figure(
            go.Bar(
                x=documents,
                y=counts,
                marker_color="#8B5CF6",
                text=counts,
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Chunks per Document",
            xaxis_title="Document",
            yaxis_title="Chunk Count",
            template="plotly_white",
            height=400,
        )
        return fig

    @staticmethod
    def similarity_scores_chart(results: List[Dict[str, Any]]) -> go.Figure:
        """Horizontal bar chart of retrieval similarity scores.

        Parameters
        ----------
        results:
            Retrieval results with ``chunk_id`` and ``similarity_score``.

        Returns
        -------
        go.Figure
            Plotly horizontal bar chart.
        """
        if not results:
            fig = go.Figure()
            fig.add_annotation(text="No results to display.", showarrow=False)
            return fig

        labels = [
            f"{r.get('source', '?')} (p{r.get('page', '?')})"
            for r in results
        ]
        scores = [r.get("similarity_score", 0) for r in results]
        rerank = [r.get("rerank_score") for r in results]

        labels = labels[::-1]
        scores = scores[::-1]
        rerank = rerank[::-1]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                name="Similarity",
                marker_color="#6366F1",
                text=[f"{s:.3f}" for s in scores],
                textposition="outside",
            )
        )
        # Add rerank scores if available
        if any(r is not None for r in rerank):
            rerank_clean = [r if r is not None else 0 for r in rerank]
            fig.add_trace(
                go.Bar(
                    x=rerank_clean,
                    y=labels,
                    orientation="h",
                    name="Re-rank",
                    marker_color="#10B981",
                    text=[f"{s:.3f}" if s else "" for s in rerank_clean],
                    textposition="outside",
                )
            )

        fig.update_layout(
            title="Retrieval Scores",
            xaxis_title="Score",
            barmode="group",
            template="plotly_white",
            height=max(300, len(results) * 50),
            margin=dict(l=200),
        )
        return fig

    @staticmethod
    def retrieval_timeline(qa_result: Dict[str, Any]) -> go.Figure:
        """Waterfall chart showing retrieval vs generation time.

        Parameters
        ----------
        qa_result:
            Result dict from :meth:`RAGPipeline.ask`.

        Returns
        -------
        go.Figure
            Plotly waterfall chart.
        """
        retrieval_ms = qa_result.get("retrieval_time_ms", 0)
        generation_ms = qa_result.get("generation_time_ms", 0)
        total_ms = retrieval_ms + generation_ms

        fig = go.Figure(
            go.Waterfall(
                name="Pipeline",
                orientation="v",
                x=["Retrieval", "Generation", "Total"],
                y=[retrieval_ms, generation_ms, total_ms],
                measure=["relative", "relative", "total"],
                connector={"line": {"color": "#6366F1"}},
                decreasing={"marker": {"color": "#EF4444"}},
                increasing={"marker": {"color": "#10B981"}},
                totals={"marker": {"color": "#8B5CF6"}},
                text=[f"{retrieval_ms:.0f}ms", f"{generation_ms:.0f}ms", f"{total_ms:.0f}ms"],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Response Time Breakdown",
            yaxis_title="Time (ms)",
            template="plotly_white",
            height=350,
        )
        return fig

    @staticmethod
    def knowledge_base_summary_card(stats: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for ``st.metric`` display cards.

        Parameters
        ----------
        stats:
            Dict from :meth:`VectorStore.get_collection_stats`.

        Returns
        -------
        Dict[str, Any]
            ``{total_documents, total_chunks, collection_name}``
        """
        return {
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "collection_name": stats.get("collection_name", ""),
        }
