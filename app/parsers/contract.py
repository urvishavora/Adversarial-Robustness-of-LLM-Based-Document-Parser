"""Contract parsing: LLM extraction + deterministic regex enrichment.

Contracts are free-form prose more often than a strict table/form layout, so
the LLM pass carries most of the weight here. The deterministic layer only
picks off the handful of near-universally labeled facts: an effective date,
a governing-law clause, and any explicitly numbered/dollar terms.
"""

from __future__ import annotations

from typing import Any

from app.json_utils import dedupe_strings
from app.parsers.generic import parse_generic_document
from app.regex_utils import extract_amounts, extract_dates, extract_labeled_value

_GUIDANCE = """
CONTRACT-SPECIFIC GUIDANCE:
- Include under fields: agreement_title, effective_date, execution_date, term_length,
  renewal_terms, termination_clause, governing_law, jurisdiction, payment_terms,
  confidentiality_clause, and signatures (an array of {name, role/title, date, page}
  for each signature block found).
- parties must list every named party to the agreement, each as an object with at
  least name and role (e.g. "Client", "Contractor", "Landlord", "Tenant").
- Preserve section/clause numbering (e.g. "3.2") when the source uses it, as a
  clause_number field alongside each extracted clause.
- Quote defined terms and obligations close to verbatim; do not paraphrase legal language.
"""

_EFFECTIVE_DATE_LABELS = (r"effective\s*date", r"date\s*of\s*(?:this\s*)?agreement")
_GOVERNING_LAW_LABELS = (r"governing\s*law",)


def _enrich(result: dict[str, Any], text: str) -> None:
    fields = result.setdefault("fields", {})

    effective_date = extract_labeled_value(text, _EFFECTIVE_DATE_LABELS)
    if effective_date and not fields.get("effective_date"):
        fields["effective_date"] = effective_date

    governing_law = extract_labeled_value(text, _GOVERNING_LAW_LABELS, max_chars=200)
    if governing_law and not fields.get("governing_law"):
        fields["governing_law"] = governing_law

    # The model sometimes returns amounts/dates as numbers or mixed types
    # rather than the displayed string (e.g. 134.45 instead of "$134.45").
    # dedupe_strings coerces everything to a comparable string before
    # deduplicating, which also avoids a set/sort TypeError on mixed types.
    result["dates"] = dedupe_strings([*result.get("dates", []), *extract_dates(text)])
    result["amounts"] = dedupe_strings([*result.get("amounts", []), *extract_amounts(text)])


def parse_contract(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(
        text, filename, "Contract", extra_guidance=_GUIDANCE, enrich=_enrich
    )
