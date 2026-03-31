"""
Document summarisation feature using map-reduce for long documents.

For short documents (< 2 000 chars) the full text is summarised in
one pass.  Longer documents are chunked, summarised individually,
then a meta-summary is produced.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.generation.generator import LLMGenerator
from src.retrieval.retriever import Retriever
from src.utils.logger import setup_logger
from src.vectorstore.store import VectorStore

logger = setup_logger(__name__)

_SHORT_DOC_THRESHOLD = 2000  # characters


class DocumentSummarizer:
    """Generate concise summaries for uploaded documents.

    Parameters
    ----------
    generator : LLMGenerator
        The shared LLM generation wrapper.
    retriever : Retriever
        The shared retrieval component (unused directly but kept for extension).
    vectorstore : VectorStore
        The shared vector store for fetching document chunks.
    """

    def __init__(
        self,
        generator: LLMGenerator,
        retriever: Retriever,
        vectorstore: VectorStore,
    ) -> None:
        self.generator = generator
        self.retriever = retriever
        self.vectorstore = vectorstore

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def summarize_document(self, source_filename: str) -> Dict[str, Any]:
        """Summarise the document identified by *source_filename*.

        Strategy:
        - If total text < 2 000 chars → single-pass summary.
        - Otherwise → map-reduce (per-chunk summaries → meta-summary).

        Parameters
        ----------
        source_filename:
            The ``source`` metadata value stored in the vector store.

        Returns
        -------
        Dict[str, Any]
            ``{source, summary, word_count_original, word_count_summary,
              compression_ratio}``
        """
        chunks = self.vectorstore.get_document_chunks(source_filename)
        if not chunks:
            return {
                "source": source_filename,
                "summary": "No content found for this document.",
                "word_count_original": 0,
                "word_count_summary": 0,
                "compression_ratio": 0.0,
            }

        # Sort chunks by page number to preserve reading order
        chunks.sort(key=lambda c: (c.get("page", 0), c.get("chunk_id", "")))

        full_text = " ".join(c["text"] for c in chunks)
        original_word_count = len(full_text.split())

        if len(full_text) < _SHORT_DOC_THRESHOLD:
            summary = self._single_pass(full_text)
        else:
            summary = self._map_reduce(chunks)

        summary_word_count = len(summary.split())
        compression = (
            round(original_word_count / summary_word_count, 2)
            if summary_word_count > 0
            else 0.0
        )

        logger.info(
            "Summarised '%s': %d → %d words (%.1f× compression)",
            source_filename,
            original_word_count,
            summary_word_count,
            compression,
        )

        return {
            "source": source_filename,
            "summary": summary,
            "word_count_original": original_word_count,
            "word_count_summary": summary_word_count,
            "compression_ratio": compression,
        }

    # ------------------------------------------------------------------ #
    #  Strategies                                                        #
    # ------------------------------------------------------------------ #

    def _single_pass(self, text: str) -> str:
        """Summarise short text in a single LLM call.

        Parameters
        ----------
        text:
            Full document text.

        Returns
        -------
        str
            The generated summary.
        """
        return self.generator.generate_summary(text)

    def _map_reduce(self, chunks: List[Dict[str, Any]]) -> str:
        """Summarise a long document via map-reduce.

        Steps:
        1. Summarise each chunk individually (*map*).
        2. Concatenate chunk summaries.
        3. Generate a final meta-summary (*reduce*).

        Parameters
        ----------
        chunks:
            Ordered list of chunk dicts.

        Returns
        -------
        str
            Final meta-summary.
        """
        chunk_summaries: List[str] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text.strip():
                s = self.generator.generate_summary(text)
                chunk_summaries.append(s)

        combined = " ".join(chunk_summaries)
        meta_summary = self.generator.generate_summary(combined)
        return meta_summary
