"""
ChromaDB vector store wrapper for persistent document storage.

Provides add / search / delete / list operations with full metadata
filtering and collection statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:
    """Persistent ChromaDB vector store for chunked documents.

    Parameters
    ----------
    config : dict
        Full application configuration; the ``vectorstore`` section is used.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        vs_cfg = config.get("vectorstore", {})
        self.persist_path: str = vs_cfg.get("vectorstore_path", "./vectorstore_data")
        self.collection_name: str = vs_cfg.get("collection_name", "teacher_knowledge_base")

        # Ensure directory exists
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection='%s', path='%s', docs=%d",
            self.collection_name,
            self.persist_path,
            self._collection.count(),
        )

    # ------------------------------------------------------------------ #
    #  Write operations                                                   #
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert chunks and their embeddings into the collection.

        Parameters
        ----------
        chunks:
            Annotated chunk dicts (must contain ``chunk_id``, ``text``,
            ``source``, ``page``, ``word_count``, ``char_count``).
        embeddings:
            2-D numpy array of shape ``(len(chunks), dim)``.

        Returns
        -------
        int
            Number of chunks stored.
        """
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        embedding_list: List[List[float]] = []

        for idx, chunk in enumerate(chunks):
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])
            metadatas.append(
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "word_count": chunk["word_count"],
                    "char_count": chunk["char_count"],
                }
            )
            embedding_list.append(embeddings[idx].tolist())

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embedding_list,
        )
        logger.info("Upserted %d chunks into collection '%s'", len(ids), self.collection_name)
        return len(ids)

    # ------------------------------------------------------------------ #
    #  Search                                                             #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the *top_k* most similar chunks to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            1-D numpy array for the query.
        top_k:
            Number of results to return.

        Returns
        -------
        List[Dict[str, Any]]
            Each dict contains ``chunk_id``, ``text``, ``source``,
            ``page``, ``distance``.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            output.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "word_count": meta.get("word_count", 0),
                    "char_count": meta.get("char_count", 0),
                    "distance": results["distances"][0][i],
                }
            )
        return output

    # ------------------------------------------------------------------ #
    #  Delete                                                             #
    # ------------------------------------------------------------------ #

    def delete_document(self, source_filename: str) -> int:
        """Remove all chunks belonging to *source_filename*.

        Parameters
        ----------
        source_filename:
            The ``source`` metadata value to match.

        Returns
        -------
        int
            Number of chunks removed.
        """
        existing = self._collection.get(
            where={"source": source_filename},
            include=[],
        )
        ids_to_delete = existing["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        logger.info("Deleted %d chunks for source='%s'", len(ids_to_delete), source_filename)
        return len(ids_to_delete)

    # ------------------------------------------------------------------ #
    #  Query helpers                                                      #
    # ------------------------------------------------------------------ #

    def list_documents(self) -> List[str]:
        """Return a sorted list of unique source filenames in the store.

        Returns
        -------
        List[str]
            Deduplicated source filenames.
        """
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])
        sources = sorted({m.get("source", "") for m in all_meta["metadatas"]})
        return sources

    def get_document_chunks(self, source: str) -> List[Dict[str, Any]]:
        """Return all chunks for a given *source* filename.

        Parameters
        ----------
        source:
            The source filename to filter on.

        Returns
        -------
        List[Dict[str, Any]]
            Chunk dicts with ``chunk_id``, ``text``, ``source``, ``page``.
        """
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        chunks: List[Dict[str, Any]] = []
        for i, cid in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            chunks.append(
                {
                    "chunk_id": cid,
                    "text": results["documents"][i],
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 0),
                    "word_count": meta.get("word_count", 0),
                    "char_count": meta.get("char_count", 0),
                }
            )
        return chunks

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the current collection.

        Returns
        -------
        Dict[str, Any]
            ``{total_chunks, total_documents, collection_name}``
        """
        total_chunks = self._collection.count()
        documents = self.list_documents()
        return {
            "total_chunks": total_chunks,
            "total_documents": len(documents),
            "collection_name": self.collection_name,
        }

    def reset_collection(self, confirm: bool = False) -> bool:
        """Wipe all data from the collection.

        Parameters
        ----------
        confirm:
            Safety flag — must be ``True`` to proceed.

        Returns
        -------
        bool
            ``True`` if the collection was reset.
        """
        if not confirm:
            logger.warning("reset_collection called without confirmation — aborting.")
            return False
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' has been reset.", self.collection_name)
        return True
