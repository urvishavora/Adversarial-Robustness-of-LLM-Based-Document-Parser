"""Output schemas (as plain dict templates) shared across parser modules."""

from __future__ import annotations

import json
from typing import Any

# Canonical document categories this project supports. "Form" covers
# application forms, whether printed, handwritten, or a mix of both -- the
# handwriting layer is an extraction detail, not a different document type.
DOCUMENT_TYPES = {
    "Resume",
    "Invoice",
    "Receipt",
    "Contract",
    "Report",
    "Form",
    "Other",
    "Unknown",
}

EMPTY_RESUME: dict[str, Any] = {
    "file_name": "",
    "name": {"full_name": None, "first_name": None, "last_name": None},
    "job_title": None,
    "contact": {
        "phone": None,
        "email": None,
        "address": None,
        "linkedin": None,
    },
    "summary": None,
    "education": [],
    "experience": [],
    "skills": [],
    "languages": [],
    "references": [],
    "certifications": [],
    "awards": [],
    "activities": [],
    "data_quality_warnings": [],
}

# Generic schema used by every non-resume document type (invoice, receipt,
# contract, report, form). Each parser module fills `fields` with the labels
# that are actually meaningful for that document type; the top-level
# dates/amounts/line_items/parties arrays are cross-cutting conveniences that
# every downstream consumer can rely on regardless of document type.
EMPTY_GENERIC: dict[str, Any] = {
    "document_type": "Unknown",
    "summary": None,
    "fields": {},
    "dates": [],
    "amounts": [],
    "line_items": [],
    "parties": [],
    "security_notes": {
        "possible_prompt_injection": False,
        "suspicious_content": [],
    },
    "data_quality_warnings": [],
}

EMPTY_HANDWRITING: dict[str, Any] = {
    "entries": [],
    "confidence": 0.0,
    "review_required": True,
    "source": "ollama_vision",
}


def deep_copy_json(value: Any) -> Any:
    """Deep-copy a JSON-serializable value (dict/list/scalar tree)."""
    return json.loads(json.dumps(value))


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
