"""Shared exception types.

Core/parser modules raise these instead of importing FastAPI directly, so
extraction/parsing logic can be unit-tested with zero web-framework
dependency. The API layer (app/main.py) is the only place that translates
them into HTTPException.
"""

from __future__ import annotations


class DocumentParserError(Exception):
    """Base class for all expected/user-facing errors in this project."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentExtractionError(DocumentParserError):
    """Raised when a PDF's text/OCR content could not be extracted."""


class LLMBackendError(DocumentParserError):
    """Raised when the configured LLM backend (Ollama) fails or is unreachable."""
