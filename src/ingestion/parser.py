"""
Document parser for PDF, DOCX, and TXT file formats.

Extracts page-level text from uploaded files and provides
validation and metadata utilities.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pdfplumber
from docx import Document as DocxDocument

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentParser:
    """Parse uploaded educational documents into a list of page dicts.

    Each page dict has the schema::

        {"page": int, "text": str, "source": str}

    Parameters
    ----------
    config : dict
        The application configuration dictionary (``ingestion`` section
        is used for validation limits).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        ingestion_cfg = config.get("ingestion", {})
        self.supported_formats: List[str] = ingestion_cfg.get(
            "supported_formats", ["pdf", "docx", "txt"]
        )
        self.max_file_size_mb: float = ingestion_cfg.get("max_file_size_mb", 20)

    # ------------------------------------------------------------------ #
    #  Format-specific parsers                                            #
    # ------------------------------------------------------------------ #

    def parse_pdf(self, file: Any) -> List[Dict[str, Any]]:
        """Extract text from a PDF file using *pdfplumber*.

        Parameters
        ----------
        file:
            A file-like object (e.g. ``st.UploadedFile``).

        Returns
        -------
        List[Dict[str, Any]]
            List of ``{"page": int, "text": str, "source": str}`` dicts.
        """
        filename = getattr(file, "name", "unknown.pdf")
        pages: List[Dict[str, Any]] = []
        file.seek(0)
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for idx, pdf_page in enumerate(pdf.pages, start=1):
                text = pdf_page.extract_text() or ""
                pages.append({"page": idx, "text": text, "source": filename})
        logger.info("Parsed PDF '%s': %d pages", filename, len(pages))
        return pages

    def parse_docx(self, file: Any) -> List[Dict[str, Any]]:
        """Extract text from a DOCX file using *python-docx*.

        Parameters
        ----------
        file:
            A file-like object (e.g. ``st.UploadedFile``).

        Returns
        -------
        List[Dict[str, Any]]
            Each paragraph group is treated as a "page".
        """
        filename = getattr(file, "name", "unknown.docx")
        file.seek(0)
        doc = DocxDocument(io.BytesIO(file.read()))
        full_text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())

        # DOCX has no native page concept → treat the whole document as page 1
        pages = [{"page": 1, "text": full_text, "source": filename}]
        logger.info("Parsed DOCX '%s': %d characters", filename, len(full_text))
        return pages

    def parse_txt(self, file: Any) -> List[Dict[str, Any]]:
        """Read a plain-text file.

        Parameters
        ----------
        file:
            A file-like object.

        Returns
        -------
        List[Dict[str, Any]]
            Single-page representation of the text file.
        """
        filename = getattr(file, "name", "unknown.txt")
        file.seek(0)
        raw_bytes = file.read()
        text = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else raw_bytes
        pages = [{"page": 1, "text": text, "source": filename}]
        logger.info("Parsed TXT '%s': %d characters", filename, len(text))
        return pages

    # ------------------------------------------------------------------ #
    #  Auto-routing                                                       #
    # ------------------------------------------------------------------ #

    def parse(self, file: Any) -> List[Dict[str, Any]]:
        """Detect file format and delegate to the appropriate parser.

        Parameters
        ----------
        file:
            An uploaded file-like object with a ``.name`` attribute.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed page dicts.

        Raises
        ------
        ValueError
            If the file extension is not in the supported formats list.
        """
        filename: str = getattr(file, "name", "")
        ext = Path(filename).suffix.lower().lstrip(".")

        if ext == "pdf":
            return self.parse_pdf(file)
        elif ext == "docx":
            return self.parse_docx(file)
        elif ext == "txt":
            return self.parse_txt(file)
        else:
            raise ValueError(
                f"Unsupported file format '.{ext}'. "
                f"Accepted formats: {self.supported_formats}"
            )

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #

    def validate(self, file: Any) -> Dict[str, Any]:
        """Check whether *file* is within format and size limits.

        Parameters
        ----------
        file:
            An uploaded file-like object.

        Returns
        -------
        Dict[str, Any]
            ``{"valid": bool, "error": Optional[str]}``
        """
        filename: str = getattr(file, "name", "")
        ext = Path(filename).suffix.lower().lstrip(".")

        if ext not in self.supported_formats:
            return {
                "valid": False,
                "error": f"Unsupported format '.{ext}'. Accepted: {self.supported_formats}",
            }

        # Check file size
        file.seek(0, 2)  # seek to end
        size_bytes = file.tell()
        file.seek(0)
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > self.max_file_size_mb:
            return {
                "valid": False,
                "error": (
                    f"File size ({size_mb:.1f} MB) exceeds the "
                    f"{self.max_file_size_mb} MB limit."
                ),
            }

        return {"valid": True, "error": None}

    # ------------------------------------------------------------------ #
    #  Metadata                                                           #
    # ------------------------------------------------------------------ #

    def get_document_metadata(self, file: Any, pages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Return descriptive metadata for *file*.

        Parameters
        ----------
        file:
            The uploaded file-like object.
        pages:
            Pre-parsed pages (avoids re-parsing).

        Returns
        -------
        Dict[str, Any]
            ``{filename, format, size_kb, page_count, total_chars}``
        """
        filename: str = getattr(file, "name", "unknown")
        ext = Path(filename).suffix.lower().lstrip(".")

        file.seek(0, 2)
        size_kb = file.tell() / 1024
        file.seek(0)

        if pages is None:
            pages = self.parse(file)

        total_chars = sum(len(p.get("text", "")) for p in pages)

        return {
            "filename": filename,
            "format": ext,
            "size_kb": round(size_kb, 2),
            "page_count": len(pages),
            "total_chars": total_chars,
        }
