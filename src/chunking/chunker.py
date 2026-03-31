"""
Document chunking strategies for the Teacher RAG pipeline.

Supports fixed-size and recursive character splitting via
``langchain_text_splitters``.  Each chunk is returned as a
richly-annotated dictionary suitable for embedding and storage.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentChunker:
    """Split parsed pages into overlapping text chunks.

    Parameters
    ----------
    config : dict
        Full application config; the ``chunking`` section is used.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        chunking_cfg = config.get("chunking", {})
        self.chunk_size: int = chunking_cfg.get("chunk_size", 512)
        self.chunk_overlap: int = chunking_cfg.get("chunk_overlap", 64)
        self.strategy: str = chunking_cfg.get("chunking_strategy", "recursive")

        # Pre-build splitters
        self._fixed_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

        self._recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------ #
    #  Strategy implementations                                           #
    # ------------------------------------------------------------------ #

    def fixed_chunk(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split *text* into fixed-size character chunks.

        Parameters
        ----------
        text:
            The full text to chunk.
        metadata:
            Page-level metadata (``source``, ``page``).

        Returns
        -------
        List[Dict[str, Any]]
            Annotated chunk dictionaries.
        """
        splits = self._fixed_splitter.split_text(text)
        return self._build_chunk_dicts(splits, metadata)

    def recursive_chunk(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split *text* using recursive character boundaries.

        Parameters
        ----------
        text:
            The full text to chunk.
        metadata:
            Page-level metadata.

        Returns
        -------
        List[Dict[str, Any]]
            Annotated chunk dictionaries.
        """
        splits = self._recursive_splitter.split_text(text)
        return self._build_chunk_dicts(splits, metadata)

    # ------------------------------------------------------------------ #
    #  Public router                                                      #
    # ------------------------------------------------------------------ #

    def chunk(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of parsed pages using the configured strategy.

        Parameters
        ----------
        pages:
            List of ``{"text": str, "source": str, "page": int}`` dicts.

        Returns
        -------
        List[Dict[str, Any]]
            Flat list of annotated chunk dicts across all pages.
        """
        all_chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for page in pages:
            text = page.get("text", "")
            if not text.strip():
                continue

            metadata = {
                "source": page.get("source", "unknown"),
                "page": page.get("page", 1),
            }

            if self.strategy == "fixed":
                page_chunks = self.fixed_chunk(text, metadata)
            else:
                # Default to recursive
                page_chunks = self.recursive_chunk(text, metadata)

            # Assign globally unique chunk IDs
            for chunk in page_chunks:
                source_stem = metadata["source"].rsplit(".", 1)[0]
                chunk["chunk_id"] = f"{source_stem}_chunk_{global_idx:04d}"
                global_idx += 1

            all_chunks.extend(page_chunks)

        logger.info(
            "Chunked %d pages → %d chunks (strategy=%s, size=%d, overlap=%d)",
            len(pages),
            len(all_chunks),
            self.strategy,
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    # ------------------------------------------------------------------ #
    #  Statistics                                                         #
    # ------------------------------------------------------------------ #

    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute descriptive statistics over a collection of chunks.

        Parameters
        ----------
        chunks:
            The list of chunk dicts.

        Returns
        -------
        Dict[str, Any]
            ``{total_chunks, avg_chars, min_chars, max_chars,
              avg_words, total_words}``
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chars": 0,
                "min_chars": 0,
                "max_chars": 0,
                "avg_words": 0,
                "total_words": 0,
            }

        char_counts = [c["char_count"] for c in chunks]
        word_counts = [c["word_count"] for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chars": round(sum(char_counts) / len(char_counts), 1),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_words": round(sum(word_counts) / len(word_counts), 1),
            "total_words": sum(word_counts),
        }

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_chunk_dicts(
        splits: List[str], metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert raw text splits into richly-annotated chunk dicts.

        Parameters
        ----------
        splits:
            Plain text segments produced by a text splitter.
        metadata:
            Shared metadata for every chunk (source, page).

        Returns
        -------
        List[Dict[str, Any]]
            Chunk dicts with ``chunk_id`` left as placeholder (set later).
        """
        chunks: List[Dict[str, Any]] = []
        for text in splits:
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": "",  # set by self.chunk()
                    "text": text,
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            )
        return chunks
