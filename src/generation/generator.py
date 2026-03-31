"""
LLM generation wrapper using HuggingFace ``transformers`` pipeline.

Loads the generative model once via ``@st.cache_resource`` and
provides task-specific generation helpers that delegate prompt
construction to :class:`PromptBuilder`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline

from src.generation.prompt_builder import PromptBuilder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@st.cache_resource(show_spinner="Loading language model …")
def _load_generation_pipeline(
    model_name: str,
    fallback_model: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> Any:
    """Load and cache the HuggingFace text2text-generation pipeline.

    Tries *model_name* first; if loading fails, falls back to
    *fallback_model*.

    Parameters
    ----------
    model_name:
        Primary HuggingFace model identifier.
    fallback_model:
        Backup model identifier if the primary fails.
    max_new_tokens:
        Maximum tokens to generate per call.
    temperature:
        Sampling temperature.
    do_sample:
        Whether to use sampling (vs greedy decoding).

    Returns
    -------
    transformers.Pipeline
        A ``text2text-generation`` pipeline.
    """
    for name in (model_name, fallback_model):
        try:
            logger.info("Attempting to load model '%s' …", name)
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            pipe = hf_pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            logger.info("Model '%s' loaded successfully.", name)
            return pipe
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load '%s': %s — trying fallback …", name, exc)

    raise RuntimeError(
        f"Could not load either '{model_name}' or '{fallback_model}'. "
        "Check your internet connection or model availability."
    )


class LLMGenerator:
    """Generate answers, summaries, and quizzes using a seq2seq LLM.

    Parameters
    ----------
    config : dict
        Full application configuration; the ``generation`` section is used.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        gen_cfg = config.get("generation", {})
        self.model_name: str = gen_cfg.get("generation_model", "google/flan-t5-large")
        self.fallback_model: str = gen_cfg.get("fallback_model", "facebook/bart-large-cnn")
        self.max_new_tokens: int = gen_cfg.get("max_new_tokens", 512)
        self.temperature: float = gen_cfg.get("temperature", 0.3)
        self.do_sample: bool = gen_cfg.get("do_sample", False)

        self._pipe = _load_generation_pipeline(
            self.model_name,
            self.fallback_model,
            self.max_new_tokens,
            self.temperature,
            self.do_sample,
        )
        self._prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------ #
    #  Core generation                                                   #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        """Run inference on a formatted prompt string.

        Parameters
        ----------
        prompt:
            The fully-constructed prompt.

        Returns
        -------
        str
            Generated text.
        """
        outputs = self._pipe(prompt)
        text: str = outputs[0].get("generated_text", "").strip()
        return text

    # ------------------------------------------------------------------ #
    #  Task-specific helpers                                             #
    # ------------------------------------------------------------------ #

    def generate_answer(self, question: str, context: str) -> str:
        """Generate a grounded answer to *question* given *context*.

        Parameters
        ----------
        question:
            User question.
        context:
            Retrieved & formatted context.

        Returns
        -------
        str
            LLM-generated answer.
        """
        prompt = self._prompt_builder.qa_prompt(question, context)
        return self.generate(prompt)

    def generate_summary(self, text: str) -> str:
        """Generate a concise summary of *text*.

        Parameters
        ----------
        text:
            Document text to summarise.

        Returns
        -------
        str
            Summary text.
        """
        prompt = self._prompt_builder.summary_prompt(text)
        return self.generate(prompt)

    def generate_quiz(
        self,
        context: str,
        num_questions: int = 5,
        difficulty: str = "medium",
    ) -> List[Dict[str, Any]]:
        """Generate and parse a multiple-choice quiz.

        Parameters
        ----------
        context:
            Source material for the quiz.
        num_questions:
            Number of questions to create.
        difficulty:
            ``easy``, ``medium``, or ``hard``.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed quiz items.
        """
        prompt = self._prompt_builder.quiz_prompt(context, num_questions, difficulty)
        raw = self.generate(prompt)
        return self.parse_quiz_output(raw)

    # ------------------------------------------------------------------ #
    #  Quiz parser                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parse_quiz_output(raw_text: str) -> List[Dict[str, Any]]:
        """Parse raw LLM quiz output into structured data.

        Expected format per question::

            Q1: ...
            A) ... B) ... C) ... D) ...
            Answer: B
            Explanation: ...

        Parameters
        ----------
        raw_text:
            Raw generated text.

        Returns
        -------
        List[Dict[str, Any]]
            Each item: ``{"question", "options", "answer", "explanation"}``.
        """
        questions: List[Dict[str, Any]] = []

        # Split on question headers  Q1:, Q2:, etc.
        q_blocks = re.split(r"Q\d+[:.]\s*", raw_text)
        q_blocks = [b.strip() for b in q_blocks if b.strip()]

        for block in q_blocks:
            question_text = ""
            options: Dict[str, str] = {"A": "", "B": "", "C": "", "D": ""}
            answer = ""
            explanation = ""

            lines = block.split("\n")
            # First non-empty line is the question
            remaining_lines: List[str] = []
            found_question = False
            for line in lines:
                line_s = line.strip()
                if not found_question and line_s:
                    question_text = line_s
                    found_question = True
                else:
                    remaining_lines.append(line_s)

            body = " ".join(remaining_lines)

            # Parse options
            opt_matches = re.findall(r"([A-D])\)\s*([^A-D)]+?)(?=(?:[A-D]\)|Answer:|Explanation:|$))", body)
            for letter, text in opt_matches:
                options[letter] = text.strip()

            # Parse answer
            ans_match = re.search(r"Answer:\s*([A-D])", body, re.IGNORECASE)
            if ans_match:
                answer = ans_match.group(1).upper()

            # Parse explanation
            exp_match = re.search(r"Explanation:\s*(.+)", body, re.IGNORECASE)
            if exp_match:
                explanation = exp_match.group(1).strip()

            if question_text:
                questions.append(
                    {
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "explanation": explanation,
                    }
                )

        return questions

    # ------------------------------------------------------------------ #
    #  Model info                                                        #
    # ------------------------------------------------------------------ #

    def get_model_info(self) -> Dict[str, Any]:
        """Return descriptive information about the loaded model.

        Returns
        -------
        Dict[str, Any]
            ``{model_name, max_new_tokens, temperature, do_sample}``
        """
        # Determine the actual model name loaded
        actual_name = self.model_name
        try:
            actual_name = self._pipe.model.config._name_or_path
        except Exception:
            pass

        return {
            "model_name": actual_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
