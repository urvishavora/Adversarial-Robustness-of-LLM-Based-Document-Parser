"""Top-level document parsing orchestration: classify -> dispatch -> parse.

Kept separate from app/main.py (the FastAPI wiring) so the whole pipeline
can be exercised in tests without going through HTTP at all.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app import config
from app.classification import infer_document_type
from app.parsers.application_form import parse_application_form
from app.parsers.contract import parse_contract
from app.parsers.generic import parse_generic_document
from app.parsers.invoice import parse_invoice
from app.parsers.receipt import parse_receipt
from app.parsers.report import parse_report
from app.parsers.resume import parse_resume

logger = logging.getLogger("document_parser")


def load_ground_truth() -> list[dict[str, Any]]:
    """Load ground-truth resumes used to build few-shot examples.

    Returns an empty list (rather than raising) when the file is missing --
    the resume parser degrades gracefully to zero examples in that case.
    """
    if not config.GROUND_TRUTH_PATH.exists():
        logger.warning("Ground-truth file does not exist: %s", config.GROUND_TRUTH_PATH)
        return []
    try:
        payload = json.loads(config.GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to load ground truth: %s", exc)
        return []
    resumes = payload.get("resumes", []) if isinstance(payload, dict) else []
    return [item for item in resumes if isinstance(item, dict)]


GROUND_TRUTH_RESUMES = load_ground_truth()


def parse_document(text: str, filename: str) -> tuple[str, dict[str, Any], list[str]]:
    """Classify a document and run it through the matching parser module.

    Returns (predicted_type, parsed_output, validation_issues). Only the
    resume path currently produces non-empty validation_issues (it's the
    only parser with a deterministic post-hoc validation/repair step); other
    types return an empty list.
    """
    predicted_type = infer_document_type(text, filename)

    if predicted_type == "Resume":
        result, issues = parse_resume(text, filename, GROUND_TRUTH_RESUMES)
        return predicted_type, result, issues

    if predicted_type == "Invoice":
        return predicted_type, parse_invoice(text, filename), []

    if predicted_type == "Receipt":
        return predicted_type, parse_receipt(text, filename), []

    if predicted_type == "Contract":
        return predicted_type, parse_contract(text, filename), []

    if predicted_type == "Report":
        return predicted_type, parse_report(text, filename), []

    if predicted_type == "Form":
        return predicted_type, parse_application_form(text, filename), []

    # "Other" / "Unknown": fall back to the plain generic extractor with no
    # type-specific guidance or enrichment.
    return predicted_type, parse_generic_document(text, filename, predicted_type), []
