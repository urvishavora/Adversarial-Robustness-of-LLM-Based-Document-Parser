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
from app.parsers.sections import extract_clauses
from app.regex_utils import extract_amounts, extract_dates, extract_labeled_value

_GUIDANCE = """
REPORT-SPECIFIC GUIDANCE:
- Include under fields: title, author(s), organization, reporting_period, report_date,
  executive_summary, methodology, findings (array), metrics (array of {name, value,
  unit} objects when the report presents quantified results), conclusions, and
  recommendations (array).

- findings and recommendations MUST carry the report's actual content, not just its
  section headings. Every entry is an object with BOTH keys:
      {"heading": "<the section heading>", "content": "<the text under that heading>"}
  A heading on its own is not an extraction -- it discards the substance of the
  report. If a section has several paragraphs, include them all in content.

  Correct:
      {"heading": "Problem Description",
       "content": "The customer found a sharp burr on the oil-channel cross hole of
                   7 machined housings from lot AH-260418..."}
  Wrong (content dropped):
      {"title": "Problem Description"}

- executive_summary, methodology and conclusions must be the report's own prose for
  those sections, copied across. Leave a field null only when that section genuinely
  does not exist in the document -- never because it is long.
- Preserve the report's own section headings as nested keys under fields rather than
  flattening everything into a single summary.
- Keep tables of figures as line_items (one object per row, with every visible column).
"""

_REPORT_DATE_LABELS = (r"report\s*date", r"date\s*of\s*(?:this\s*)?report")
_AUTHOR_LABELS = (r"prepared\s*by", r"author(?:\(s\))?", r"submitted\s*by")


_CONTENT_KEYS = ("content", "value", "text", "description", "detail", "details", "body", "summary")
_HEADING_KEYS = ("heading", "title", "label", "name", "section")


def _entry_has_content(entry: Any) -> bool:
    """True when a findings/recommendations entry carries substance.

    An entry consisting only of a heading is not an extraction -- it names a
    section and throws away what the section said. This is easy to miss
    because such output is structurally valid and every heading really does
    appear in the document, so it scores well on any check that only asks
    "is this value present in the source?".
    """
    if isinstance(entry, str):
        return len(entry.split()) > 4
    if not isinstance(entry, dict):
        return False
    for key, value in entry.items():
        if key.lower() in _HEADING_KEYS:
            continue
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, dict)) and value:
            return True
        if isinstance(value, (int, float)):
            return True
    # Only heading-like keys present -> no content.
    return any(
        key.lower() in _CONTENT_KEYS and str(value).strip() for key, value in entry.items()
    )


def _warn_on_heading_only_sections(result: dict[str, Any]) -> None:
    fields = result.get("fields")
    if not isinstance(fields, dict):
        return
    warnings = result.setdefault("data_quality_warnings", [])
    if not isinstance(warnings, list):
        return
    for section in ("findings", "recommendations"):
        entries = fields.get(section)
        if not isinstance(entries, list) or not entries:
            continue
        empty = sum(1 for entry in entries if not _entry_has_content(entry))
        if empty:
            warnings.append(
                f"{empty} of {len(entries)} {section} entries contain only a heading "
                f"with no supporting text from the report."
            )


def _enrich(result: dict[str, Any], text: str) -> None:
    fields = result.setdefault("fields", {})

    # Reports lose even more content than contracts: measured recall against
    # full-text ground truth was 3.5%, because the model returns section
    # headings and drops the text beneath them. The same deterministic
    # segmentation used for contract clauses recovers the body here -- a
    # report's sections are a layout property too. Stored under `sections`
    # so it does not collide with the model's own findings/recommendations.
    if not fields.get("sections"):
        sections = extract_clauses(text)
        if sections:
            fields["sections"] = [
                {"heading": s_["heading"], "number": s_["clause_number"], "text": s_["text"]}
                for s_ in sections
            ]

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
    _warn_on_heading_only_sections(result)

    result["dates"] = dedupe_strings([*result.get("dates", []), *extract_dates(text)])
    result["amounts"] = dedupe_strings([*result.get("amounts", []), *extract_amounts(text)])


def parse_report(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(
        text, filename, "Report", extra_guidance=_GUIDANCE, enrich=_enrich
    )
