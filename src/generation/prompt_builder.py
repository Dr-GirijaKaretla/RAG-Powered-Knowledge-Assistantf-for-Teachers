"""
Prompt templates for every generation task in the RAG pipeline.

All prompts are plain-text templates formatted with Python ``str.format``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PromptBuilder:
    """Build task-specific prompts for the LLM generation stage.

    All methods return a fully-formatted prompt string ready to be
    passed directly to the generator.
    """

    # ------------------------------------------------------------------ #
    #  QA Prompt                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def qa_prompt(question: str, context: str) -> str:
        """Build a grounded question-answering prompt.

        Parameters
        ----------
        question:
            The teacher's question.
        context:
            Retrieved context string with inline source citations.

        Returns
        -------
        str
            Formatted QA prompt.
        """
        return (
            "You are a knowledgeable teaching assistant helping a teacher. "
            "Answer the question using ONLY the information in the context below. "
            "If the answer is not in the context, say: 'I could not find this "
            "in your uploaded materials.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    # ------------------------------------------------------------------ #
    #  Summarisation Prompt                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def summary_prompt(text: str) -> str:
        """Build a document summarisation prompt.

        Parameters
        ----------
        text:
            The text to summarise.

        Returns
        -------
        str
            Formatted summary prompt.
        """
        return (
            "Summarize the following educational text clearly and concisely. "
            "Highlight key concepts, important definitions, and main ideas.\n\n"
            f"Text:\n{text}\n\n"
            "Summary:"
        )

    # ------------------------------------------------------------------ #
    #  Quiz Prompt                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def quiz_prompt(context: str, num_questions: int, difficulty: str) -> str:
        """Build a multiple-choice quiz generation prompt.

        Parameters
        ----------
        context:
            Source material for quiz questions.
        num_questions:
            Number of questions to generate.
        difficulty:
            One of ``easy``, ``medium``, ``hard``.

        Returns
        -------
        str
            Formatted quiz prompt.
        """
        return (
            f"You are an expert educator. Create {num_questions} {difficulty}-level "
            "multiple choice questions based on the context below. "
            "Each question must have 4 options (A, B, C, D) and a correct answer. "
            "Format each question as:\n"
            "Q[n]: [question text]\n"
            "A) ... B) ... C) ... D) ...\n"
            "Answer: [letter]\n"
            "Explanation: [brief explanation]\n\n"
            f"Context:\n{context}"
        )

    # ------------------------------------------------------------------ #
    #  Follow-up / Conversational Prompt                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def follow_up_prompt(
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a conversational follow-up prompt.

        Injects the last 3 turns of chat history so the LLM can
        maintain conversational continuity.

        Parameters
        ----------
        question:
            The latest user question.
        context:
            Retrieved context string.
        chat_history:
            List of ``{"role": str, "content": str}`` dicts.

        Returns
        -------
        str
            Formatted conversational prompt.
        """
        history_block = ""
        if chat_history:
            recent = chat_history[-6:]  # last 3 turns = 6 messages (user+assistant)
            lines = []
            for msg in recent:
                role_label = "Teacher" if msg.get("role") == "user" else "Assistant"
                lines.append(f"{role_label}: {msg.get('content', '')}")
            history_block = "\n".join(lines) + "\n\n"

        return (
            "You are a knowledgeable teaching assistant. Continue the conversation "
            "below and answer the latest question using ONLY the provided context. "
            "If the answer is not in the context, say: 'I could not find this "
            "in your uploaded materials.'\n\n"
            f"Conversation so far:\n{history_block}"
            f"Context:\n{context}\n\n"
            f"Teacher: {question}\n\n"
            "Assistant:"
        )
