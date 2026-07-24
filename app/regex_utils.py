"""Shared deterministic (non-LLM) extraction helpers for structured documents.

Invoices, receipts, contracts, and reports are template-driven enough that a
label -> nearby-value regex can extract identifiers, dates, and money
amounts with effectively 100% precision whenever the pattern matches --
unlike an LLM, a regex match either is or isn't the literal source text, so
there is no hallucination risk for what it does catch. These functions are
combined with the LLM output in each parser module: deterministic values win
when present, the LLM fills in everything else.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from app.json_utils import dedupe_strings

_DATE_PATTERN = re.compile(
    r"(?<!\d)(?:"
    r"(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)\d{2}"  # DD/MM/YYYY or DD-MM-YYYY
    r"|(?:0?[1-9]|1[0-2])[./-](?:0?[1-9]|[12]\d|3[01])[./-](?:19|20)\d{2}"  # MM/DD/YYYY
    r"|(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])"  # ISO YYYY-MM-DD
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+(?:19|20)\d{2}"  # Month DD, YYYY
    r")(?!\d)",
    re.IGNORECASE,
)

_AMOUNT_PATTERN = re.compile(
    r"(?<![\w.])"
    r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
    r"|\b\d{1,3}(?:,\d{3})*\.\d{2}\b"
)


class LabeledAmount(NamedTuple):
    label: str
    value: str


def extract_dates(text: str) -> list[str]:
    """Return unique displayed dates in their original source format."""
    return dedupe_strings(match.group(0) for match in _DATE_PATTERN.finditer(text))


def extract_amounts(text: str) -> list[str]:
    """Return unique displayed currency amounts in their original source format."""
    return dedupe_strings(match.group(0) for match in _AMOUNT_PATTERN.finditer(text))


def extract_labeled_value(
    text: str, label_patterns: tuple[str, ...], *, max_chars: int = 120
) -> str | None:
    """Return the short value that follows one of the given field labels,
    e.g. label_patterns=(r"invoice\\s*(?:#|no\\.?|number)",) matches
    "Invoice #: INV-2024-0091" -> "INV-2024-0091".
    """
    combined = "|".join(f"(?:{pattern})" for pattern in label_patterns)
    match = re.search(rf"(?im)\b(?:{combined})\b\s*[:#-]?\s*(.+)", text)
    if not match:
        return None
    candidate = match.group(1)[:max_chars].strip()
    candidate = re.split(r"\s{2,}|\t|\n", candidate, maxsplit=1)[0].strip(" :;,.|-_")
    return candidate or None


def extract_labeled_amount(text: str, label_patterns: tuple[str, ...]) -> str | None:
    """Return the first currency amount found near one of the given labels
    (e.g. "Total", "Subtotal", "Amount Due", "Tax") on the same or next line.
    """
    combined = "|".join(f"(?:{pattern})" for pattern in label_patterns)
    match = re.search(rf"(?im)\b(?:{combined})\b.{{0,40}}", text)
    if not match:
        return None
    window = text[match.start() : match.end() + 40]
    amount_match = _AMOUNT_PATTERN.search(window)
    return amount_match.group(0) if amount_match else None
