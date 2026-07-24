"""Report parsing: LLM extraction + light deterministic enrichment.

Reports (business reports, research/technical reports, audits) are the most
free-form of the six document types, so this module leans almost entirely
on the LLM pass; the deterministic layer just backstops document-wide dates
and any dollar/numeric figures mentioned in the body so they're always
present in the top-level `amounts`/`dates` arrays even if the model omits
one from `fields`.
"""

from __future__ import annotations

from typing import Any

from app.json_utils import dedupe_strings
from app.parsers.generic import parse_generic_document
from app.regex_utils import extract_amounts, extract_dates, extract_labeled_value

_GUIDANCE = """
REPORT-SPECIFIC GUIDANCE:
- Include under fields: title, author(s), organization, reporting_period, report_date,
  executive_summary, methodology, findings (array), metrics (array of {name, value,
  unit} objects when the report presents quantified results), conclusions, and
  recommendations (array).
- Preserve the report's own section headings as nested keys under fields rather than
  flattening everything into a single summary.
- Keep tables of figures as line_items (one object per row, with every visible column).
"""

_REPORT_DATE_LABELS = (r"report\s*date", r"date\s*of\s*(?:this\s*)?report")
_AUTHOR_LABELS = (r"prepared\s*by", r"author(?:\(s\))?", r"submitted\s*by")


def _enrich(result: dict[str, Any], text: str) -> None:
    fields = result.setdefault("fields", {})

    report_date = extract_labeled_value(text, _REPORT_DATE_LABELS)
    if report_date and not fields.get("report_date"):
        fields["report_date"] = report_date

    author = extract_labeled_value(text, _AUTHOR_LABELS)
    if author and not fields.get("author"):
        fields["author"] = author

    # The model sometimes returns amounts/dates as numbers or mixed types
    # rather than the displayed string (e.g. 134.45 instead of "$134.45").
    # dedupe_strings coerces everything to a comparable string before
    # deduplicating, which also avoids a set/sort TypeError on mixed types.
    result["dates"] = dedupe_strings([*result.get("dates", []), *extract_dates(text)])
    result["amounts"] = dedupe_strings([*result.get("amounts", []), *extract_amounts(text)])


def parse_report(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(
        text, filename, "Report", extra_guidance=_GUIDANCE, enrich=_enrich
    )
