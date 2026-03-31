"""
Unified RAG orchestrator that wires every component together.

The :class:`RAGPipeline` is the single entry-point used by
``app.py``.  It owns the lifecycle of parsing → cleaning →
chunking → embedding → storing → retrieving → generating.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.chunking.chunker import DocumentChunker
from src.embedding.embedder import TextEmbedder
from src.features.quiz_generator import QuizGenerator
from src.features.summarizer import DocumentSummarizer
from src.generation.generator import LLMGenerator
from src.ingestion.cleaner import TextCleaner
from src.ingestion.parser import DocumentParser
from src.retrieval.retriever import Retriever
from src.utils.logger import setup_logger
from src.vectorstore.store import VectorStore

logger = setup_logger(__name__)


class RAGPipeline:
    """Master orchestrator for the Teacher RAG application.

    Parameters
    ----------
    config_path : str
        Filesystem path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        logger.info("Initialising RAGPipeline from '%s' …", config_path)
        self.config = self._load_config(config_path)

        # ── Component initialisation ──────────────────────────
        logger.info("  → DocumentParser")
        self.parser = DocumentParser(self.config)

        logger.info("  → TextCleaner")
        self.cleaner = TextCleaner()

        logger.info("  → DocumentChunker")
        self.chunker = DocumentChunker(self.config)

        logger.info("  → TextEmbedder")
        self.embedder = TextEmbedder(self.config)

        logger.info("  → VectorStore")
        self.vectorstore = VectorStore(self.config)

        logger.info("  → Retriever")
        self.retriever = Retriever(self.config, self.embedder, self.vectorstore)

        logger.info("  → LLMGenerator")
        self.generator = LLMGenerator(self.config)

        logger.info("  → DocumentSummarizer")
        self.summarizer = DocumentSummarizer(
            self.generator, self.retriever, self.vectorstore
        )

        logger.info("  → QuizGenerator")
        self.quiz_generator = QuizGenerator(
            self.retriever, self.generator, self.config
        )

        logger.info("RAGPipeline initialisation complete ✓")

    # ------------------------------------------------------------------ #
    #  Config loader                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Read and return the YAML configuration file.

        Parameters
        ----------
        config_path:
            Path to ``config.yaml``.

        Returns
        -------
        Dict[str, Any]
            Parsed configuration dictionary.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    # ------------------------------------------------------------------ #
    #  Document ingestion                                                #
    # ------------------------------------------------------------------ #

    def ingest_document(
        self,
        file: Any,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the full ingestion pipeline on an uploaded file.

        Steps:
        1. Parse the file into pages.
        2. Clean each page's text.
        3. Chunk into overlapping segments.
        4. Filter out non-meaningful chunks.
        5. Embed all chunks in batch.
        6. Store chunks + embeddings in the vector store.

        Parameters
        ----------
        file:
            An uploaded file-like object (e.g. ``st.UploadedFile``).
        progress_callback:
            Optional callable ``(fraction: float, message: str)`` for UI
            progress updates.  If ``None``, progress is silently logged.

        Returns
        -------
        Dict[str, Any]
            Ingestion report with ``filename``, ``pages_parsed``,
            ``chunks_created``, ``chunks_stored``, ``ingestion_time_ms``.
        """
        start = time.perf_counter()

        def _progress(frac: float, msg: str) -> None:
            logger.info("Ingestion %.0f%%: %s", frac * 100, msg)
            if progress_callback:
                progress_callback(frac, msg)

        filename = getattr(file, "name", "unknown")

        # 1. Parse
        _progress(0.10, "Parsing document …")
        pages = self.parser.parse(file)

        # 2. Clean
        _progress(0.30, "Cleaning text …")
        for page in pages:
            page["text"] = self.cleaner.clean(page.get("text", ""))

        # 3. Chunk
        _progress(0.50, "Creating chunks …")
        chunks = self.chunker.chunk(pages)

        # 4. Filter
        meaningful_chunks = [
            c for c in chunks if self.cleaner.is_meaningful(c["text"])
        ]

        # 5. Embed
        _progress(0.70, "Generating embeddings …")
        texts = [c["text"] for c in meaningful_chunks]
        embeddings = self.embedder.embed_batch(texts) if texts else []

        # 6. Store
        _progress(0.90, "Storing in knowledge base …")
        stored = 0
        if len(meaningful_chunks) > 0:
            stored = self.vectorstore.add_documents(meaningful_chunks, embeddings)

        elapsed = (time.perf_counter() - start) * 1000
        _progress(1.0, "Done ✓")

        report = {
            "filename": filename,
            "pages_parsed": len(pages),
            "chunks_created": len(chunks),
            "chunks_stored": stored,
            "ingestion_time_ms": round(elapsed, 1),
        }
        logger.info("Ingestion report: %s", report)
        return report

    # ------------------------------------------------------------------ #
    #  Question answering                                                #
    # ------------------------------------------------------------------ #

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Answer a natural-language question using the RAG pipeline.

        Parameters
        ----------
        question:
            The teacher's question.
        chat_history:
            Previous chat messages for conversational context.
        top_k:
            Override for the number of chunks to retrieve.

        Returns
        -------
        Dict[str, Any]
            ``{question, answer, sources, context_used,
              retrieval_time_ms, generation_time_ms}``
        """
        # 1. Retrieve
        t0 = time.perf_counter()
        results = self.retriever.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context = self.retriever.format_context(results)

        # 2. Generate
        t1 = time.perf_counter()
        if chat_history:
            from src.generation.prompt_builder import PromptBuilder

            prompt = PromptBuilder.follow_up_prompt(question, context, chat_history)
            answer = self.generator.generate(prompt)
        else:
            answer = self.generator.generate_answer(question, context)
        generation_ms = (time.perf_counter() - t1) * 1000

        sources = [
            {
                "source": r["source"],
                "page": r["page"],
                "score": r.get("rerank_score") or r.get("similarity_score", 0),
            }
            for r in results
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_used": context,
            "retrieval_time_ms": round(retrieval_ms, 1),
            "generation_time_ms": round(generation_ms, 1),
            "retrieval_results": results,
        }

    # ------------------------------------------------------------------ #
    #  Delegate helpers                                                  #
    # ------------------------------------------------------------------ #

    def summarize(self, source_filename: str) -> Dict[str, Any]:
        """Summarise a document by filename.

        Parameters
        ----------
        source_filename:
            Document source name.

        Returns
        -------
        Dict[str, Any]
            Summary result dict.
        """
        return self.summarizer.summarize_document(source_filename)

    def generate_quiz(
        self,
        topic: Optional[str] = None,
        source_filename: Optional[str] = None,
        num_questions: int = 5,
        difficulty: str = "medium",
    ) -> List[Dict[str, Any]]:
        """Generate a quiz by topic or by document.

        Parameters
        ----------
        topic:
            Free-text topic (used if *source_filename* is ``None``).
        source_filename:
            Document source name (takes priority over *topic*).
        num_questions:
            Number of quiz questions.
        difficulty:
            ``easy``, ``medium``, or ``hard``.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed quiz items.
        """
        if source_filename:
            return self.quiz_generator.generate_quiz_from_document(
                source_filename, num_questions, difficulty
            )
        elif topic:
            return self.quiz_generator.generate_quiz_from_topic(
                topic, num_questions, difficulty
            )
        return []

    def list_documents(self) -> List[str]:
        """Return the list of documents in the knowledge base.

        Returns
        -------
        List[str]
            Unique source filenames.
        """
        return self.vectorstore.list_documents()

    def delete_document(self, source: str) -> int:
        """Delete a document from the knowledge base.

        Parameters
        ----------
        source:
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        return self.vectorstore.delete_document(source)

    def get_system_stats(self) -> Dict[str, Any]:
        """Return a comprehensive system status dictionary.

        Returns
        -------
        Dict[str, Any]
            Combined stats from vector store, embedder, and generator.
        """
        vs_stats = self.vectorstore.get_collection_stats()
        emb_info = self.embedder.get_model_info()
        gen_info = self.generator.get_model_info()

        return {
            "vectorstore": vs_stats,
            "embedding_model": emb_info,
            "generation_model": gen_info,
            "documents": self.list_documents(),
        }

    # ------------------------------------------------------------------ #
    #  Demo content loader                                               #
    # ------------------------------------------------------------------ #

    def ingest_demo_content(self) -> List[Dict[str, Any]]:
        """Ingest the three built-in demo text snippets.

        Returns
        -------
        List[Dict[str, Any]]
            Ingestion report for each demo document.
        """
        demos = {
            "demo_biology_photosynthesis.txt": (
                "Photosynthesis is the process by which green plants and some other "
                "organisms use sunlight to synthesize nutrients from carbon dioxide "
                "and water. It occurs primarily in chloroplasts, using chlorophyll "
                "to capture light energy. The light-dependent reactions take place "
                "in the thylakoid membrane, producing ATP and NADPH. The Calvin "
                "cycle uses these energy carriers to fix carbon dioxide into glucose. "
                "Photosynthesis is fundamental to life on Earth as it produces "
                "oxygen and forms the base of most food chains. The overall equation "
                "is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Factors affecting "
                "the rate of photosynthesis include light intensity, carbon dioxide "
                "concentration, and temperature. C4 and CAM plants have evolved "
                "alternative carbon fixation pathways to minimize photorespiration "
                "in hot, arid environments."
            ),
            "demo_history_wwii.txt": (
                "World War II lasted from 1939 to 1945 and involved most of the "
                "world's nations. It began with Germany's invasion of Poland on "
                "September 1, 1939. The Allied Powers, including the United States, "
                "United Kingdom, and Soviet Union, opposed the Axis Powers of "
                "Germany, Italy, and Japan. The Holocaust resulted in the systematic "
                "murder of six million Jews. The war ended in Europe on May 8, 1945 "
                "(V-E Day) and in the Pacific on September 2, 1945 (V-J Day) after "
                "atomic bombs were dropped on Hiroshima and Nagasaki. Key battles "
                "included Stalingrad, Midway, D-Day at Normandy, and the Battle of "
                "the Bulge. The war led to the creation of the United Nations and "
                "marked the beginning of the Cold War era."
            ),
            "demo_math_calculus.txt": (
                "Calculus is the mathematical study of continuous change. It has "
                "two major branches: differential calculus and integral calculus. "
                "Differential calculus concerns instantaneous rates of change and "
                "slopes of curves. Integral calculus concerns accumulation of "
                "quantities and areas under curves. The Fundamental Theorem of "
                "Calculus links the two branches. Derivatives measure how a function "
                "changes as its input changes. Integrals compute the accumulated "
                "change of a function over an interval. Calculus was independently "
                "developed by Isaac Newton and Gottfried Wilhelm Leibniz. "
                "Applications include physics (motion, forces), engineering "
                "(optimization), economics (marginal analysis), and biology "
                "(population models). Key concepts include limits, continuity, "
                "the chain rule, integration by parts, and Taylor series."
            ),
        }

        reports: List[Dict[str, Any]] = []
        for filename, content in demos.items():
            # Check if already loaded
            if filename in self.list_documents():
                logger.info("Demo '%s' already loaded — skipping.", filename)
                continue

            # Create an in-memory file-like object
            fake_file = io.BytesIO(content.encode("utf-8"))
            fake_file.name = filename
            report = self.ingest_document(fake_file)
            reports.append(report)

        return reports
