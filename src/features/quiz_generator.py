"""
Automatic quiz generation from uploaded educational content.

Supports generation by free-text topic (semantic retrieval) or
by specific uploaded document.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.generator import LLMGenerator
from src.retrieval.retriever import Retriever
from src.utils.logger import setup_logger
from src.vectorstore.store import VectorStore

logger = setup_logger(__name__)


class QuizGenerator:
    """Create multiple-choice quizzes from the knowledge base.

    Parameters
    ----------
    retriever : Retriever
        Shared retrieval component.
    generator : LLMGenerator
        Shared LLM generation wrapper.
    config : dict
        Full application configuration; the ``quiz`` section is used.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: LLMGenerator,
        config: Dict[str, Any],
    ) -> None:
        self.retriever = retriever
        self.generator = generator

        quiz_cfg = config.get("quiz", {})
        self.default_num_questions: int = quiz_cfg.get("quiz_num_questions", 5)
        self.default_difficulty: str = quiz_cfg.get("quiz_difficulty", "medium")

    # ------------------------------------------------------------------ #
    #  By topic (semantic retrieval)                                     #
    # ------------------------------------------------------------------ #

    def generate_quiz_from_topic(
        self,
        topic: str,
        num_questions: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a quiz on *topic* by retrieving relevant chunks.

        Parameters
        ----------
        topic:
            Free-text topic, e.g. ``"photosynthesis"``.
        num_questions:
            Override for the default question count.
        difficulty:
            Override for the default difficulty level.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed quiz items from :meth:`LLMGenerator.generate_quiz`.
        """
        n = num_questions or self.default_num_questions
        diff = difficulty or self.default_difficulty

        results = self.retriever.retrieve(topic, top_k=5)
        context = self.retriever.format_context(results)

        if not context.strip():
            logger.warning("No content found for topic '%s'", topic)
            return []

        quiz = self.generator.generate_quiz(context, n, diff)
        logger.info("Generated %d quiz questions for topic '%s'", len(quiz), topic)
        return quiz

    # ------------------------------------------------------------------ #
    #  By document                                                       #
    # ------------------------------------------------------------------ #

    def generate_quiz_from_document(
        self,
        source_filename: str,
        num_questions: Optional[int] = None,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a quiz from a specific uploaded document.

        Parameters
        ----------
        source_filename:
            The ``source`` metadata value in the vector store.
        num_questions:
            Override for the default question count.
        difficulty:
            Override for the default difficulty level.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed quiz items.
        """
        n = num_questions or self.default_num_questions
        diff = difficulty or self.default_difficulty

        # Fetch all chunks for this document directly
        vs: VectorStore = self.retriever.vectorstore
        chunks = vs.get_document_chunks(source_filename)

        if not chunks:
            logger.warning("No chunks found for document '%s'", source_filename)
            return []

        # Sort and concatenate
        chunks.sort(key=lambda c: (c.get("page", 0), c.get("chunk_id", "")))
        context = " ".join(c["text"] for c in chunks)

        # Truncate if extremely long to stay within model limits
        max_context_chars = 3000
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        quiz = self.generator.generate_quiz(context, n, diff)
        logger.info(
            "Generated %d quiz questions from document '%s'",
            len(quiz),
            source_filename,
        )
        return quiz

    # ------------------------------------------------------------------ #
    #  Display / export helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_quiz_for_display(quiz_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare quiz items for Streamlit rendering.

        Adds a 1-based ``number`` key and ensures all fields exist.

        Parameters
        ----------
        quiz_list:
            Raw parsed quiz items.

        Returns
        -------
        List[Dict[str, Any]]
            Display-ready quiz items.
        """
        display: List[Dict[str, Any]] = []
        for idx, item in enumerate(quiz_list, start=1):
            display.append(
                {
                    "number": idx,
                    "question": item.get("question", f"Question {idx}"),
                    "options": item.get("options", {"A": "", "B": "", "C": "", "D": ""}),
                    "answer": item.get("answer", ""),
                    "explanation": item.get("explanation", ""),
                }
            )
        return display

    @staticmethod
    def export_quiz_as_text(quiz_list: List[Dict[str, Any]]) -> str:
        """Convert quiz items to a downloadable plain-text string.

        Parameters
        ----------
        quiz_list:
            Parsed quiz items.

        Returns
        -------
        str
            Formatted plain-text quiz.
        """
        lines: List[str] = []
        for idx, item in enumerate(quiz_list, start=1):
            lines.append(f"Q{idx}: {item.get('question', '')}")
            opts = item.get("options", {})
            for letter in ("A", "B", "C", "D"):
                lines.append(f"  {letter}) {opts.get(letter, '')}")
            lines.append(f"  Answer: {item.get('answer', '')}")
            explanation = item.get("explanation", "")
            if explanation:
                lines.append(f"  Explanation: {explanation}")
            lines.append("")
        return "\n".join(lines)
