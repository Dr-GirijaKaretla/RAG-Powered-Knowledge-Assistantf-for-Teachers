"""
Text cleaning and normalisation utilities for ingested documents.

The :class:`TextCleaner` pipeline strips HTML artefacts, collapses
whitespace, removes noisy special characters, and filters out
meaningless micro-chunks before they reach the chunking stage.
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextCleaner:
    """Clean and normalise raw text extracted from documents.

    Usage
    -----
    >>> cleaner = TextCleaner()
    >>> clean_text = cleaner.clean(raw_text)
    """

    # Regex patterns compiled once for performance.
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _SPECIAL_CHAR_RE = re.compile(r"[^\w\s.,;:!?'\"()\-/&%$#@\[\]{}+=*<>°²³]")
    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _HEADER_FOOTER_RE = re.compile(
        r"(?i)^(page\s*\d+|confidential|draft|©.*|all rights reserved.*)$",
        re.MULTILINE,
    )

    # ------------------------------------------------------------------ #
    #  Individual cleaning steps                                          #
    # ------------------------------------------------------------------ #

    def remove_html(self, text: str) -> str:
        """Strip HTML / XML tags from *text*.

        Parameters
        ----------
        text:
            Raw text potentially containing HTML markup.

        Returns
        -------
        str
            Text with all ``<...>`` tags removed.
        """
        return self._HTML_TAG_RE.sub("", text)

    def remove_special_chars(self, text: str) -> str:
        """Remove unusual special characters while preserving punctuation.

        Parameters
        ----------
        text:
            Partially cleaned text.

        Returns
        -------
        str
            Text with only common punctuation retained.
        """
        return self._SPECIAL_CHAR_RE.sub("", text)

    def normalize_whitespace(self, text: str) -> str:
        """Collapse consecutive spaces / tabs and excessive newlines.

        Parameters
        ----------
        text:
            Text with potentially irregular whitespace.

        Returns
        -------
        str
            Whitespace-normalised text.
        """
        text = self._MULTI_SPACE_RE.sub(" ", text)
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()

    def remove_headers_footers(self, text: str) -> str:
        """Remove common repeated page headers and footers.

        Parameters
        ----------
        text:
            Text that may contain repeated header/footer lines.

        Returns
        -------
        str
            Text with recognised boilerplate lines removed.
        """
        return self._HEADER_FOOTER_RE.sub("", text)

    # ------------------------------------------------------------------ #
    #  Master pipeline                                                    #
    # ------------------------------------------------------------------ #

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on *text*.

        Pipeline order:
        1. Remove HTML tags
        2. Remove headers / footers
        3. Remove noisy special characters
        4. Normalise whitespace

        Parameters
        ----------
        text:
            Raw extracted text.

        Returns
        -------
        str
            Cleaned, normalised text ready for chunking.
        """
        text = self.remove_html(text)
        text = self.remove_headers_footers(text)
        text = self.remove_special_chars(text)
        text = self.normalize_whitespace(text)
        return text

    # ------------------------------------------------------------------ #
    #  Quality gate                                                       #
    # ------------------------------------------------------------------ #

    def is_meaningful(self, text: str, min_words: int = 10) -> bool:
        """Decide whether *text* contains enough substance to be kept.

        Parameters
        ----------
        text:
            A text chunk candidate.
        min_words:
            Minimum word count required (default ``10``).

        Returns
        -------
        bool
            ``True`` if the chunk has at least *min_words* words.
        """
        word_count = len(text.split())
        return word_count >= min_words

    # ------------------------------------------------------------------ #
    #  Batch helpers                                                      #
    # ------------------------------------------------------------------ #

    def clean_pages(self, pages: List[dict]) -> List[dict]:
        """Clean the ``"text"`` field of every page dict in *pages*.

        Parameters
        ----------
        pages:
            List of page dicts as returned by :class:`DocumentParser`.

        Returns
        -------
        List[dict]
            Same list with cleaned text in each dict.
        """
        cleaned: List[dict] = []
        for page in pages:
            clean_text = self.clean(page.get("text", ""))
            if clean_text:
                cleaned.append({**page, "text": clean_text})
        logger.info("Cleaned %d / %d pages", len(cleaned), len(pages))
        return cleaned
