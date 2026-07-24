"""Invoice parsing: LLM extraction + deterministic regex enrichment.

Invoice numbers, dates, and totals are the fields a user is most likely to
depend on being exactly right, and they are also the fields a plain regex
can find with effectively no error rate whenever a recognizable label is
present. This module always tries the deterministic pass first and only
lets the LLM value stand when no deterministic match was found, so a typo'd
or hallucinated total can't silently override a value that was read
verbatim off the page.
"""

from __future__ import annotations

from typing import Any

from app.json_utils import dedupe_strings
from app.parsers.generic import parse_generic_document
from app.regex_utils import extract_amounts, extract_dates, extract_labeled_amount, extract_labeled_value

_GUIDANCE = """
INVOICE-SPECIFIC GUIDANCE:
- Include under fields: invoice_number, invoice_date, due_date, purchase_order_number,
  vendor (name, address, email, phone), customer/bill_to (name, address, email, phone),
  ship_to (if different from bill_to), currency, subtotal, tax, discount, shipping,
  total, amount_due, payment_terms, and payment_instructions when present.
- line_items must be an array of objects, each with description, quantity, unit_price,
  and amount (or whatever subset of these columns the invoice actually shows).
- Do not compute or correct totals yourself -- copy the printed subtotal/tax/total
  values exactly even if they appear not to add up; a mismatch is source data, not
  something to silently fix.
"""

_INVOICE_NUMBER_LABELS = (r"invoice\s*(?:#|no\.?|number|id)", r"inv\s*#")
_INVOICE_DATE_LABELS = (r"invoice\s*date", r"date\s*of\s*invoice", r"\bdate\b")
_DUE_DATE_LABELS = (r"due\s*date", r"payment\s*due")
_PO_NUMBER_LABELS = (r"purchase\s*order\s*(?:#|no\.?|number)?", r"\bpo\s*(?:#|no\.?|number)\b")
_SUBTOTAL_LABELS = (r"sub\s*-?\s*total",)
_TAX_LABELS = (r"tax(?:\s*\(\d+%?\))?", r"vat", r"gst")
_TOTAL_LABELS = (r"total\s*due", r"amount\s*due", r"grand\s*total", r"balance\s*due", r"\btotal\b")


def _enrich(result: dict[str, Any], text: str) -> None:
    fields = result.setdefault("fields", {})

    invoice_number = extract_labeled_value(text, _INVOICE_NUMBER_LABELS)
    if invoice_number and not fields.get("invoice_number"):
        fields["invoice_number"] = invoice_number

    po_number = extract_labeled_value(text, _PO_NUMBER_LABELS)
    if po_number and not fields.get("purchase_order_number"):
        fields["purchase_order_number"] = po_number

    invoice_date = extract_labeled_value(text, _INVOICE_DATE_LABELS)
    if invoice_date and not fields.get("invoice_date"):
        fields["invoice_date"] = invoice_date

    due_date = extract_labeled_value(text, _DUE_DATE_LABELS)
    if due_date and not fields.get("due_date"):
        fields["due_date"] = due_date

    subtotal = extract_labeled_amount(text, _SUBTOTAL_LABELS)
    if subtotal and not fields.get("subtotal"):
        fields["subtotal"] = subtotal

    tax = extract_labeled_amount(text, _TAX_LABELS)
    if tax and not fields.get("tax"):
        fields["tax"] = tax

    total = extract_labeled_amount(text, _TOTAL_LABELS)
    if total and not (fields.get("total") or fields.get("amount_due")):
        fields["total"] = total

    # The model sometimes returns amounts/dates as numbers or mixed types
    # rather than the displayed string (e.g. 134.45 instead of "$134.45").
    # dedupe_strings coerces everything to a comparable string before
    # deduplicating, which also avoids a set/sort TypeError on mixed types.
    result["dates"] = dedupe_strings([*result.get("dates", []), *extract_dates(text)])
    result["amounts"] = dedupe_strings([*result.get("amounts", []), *extract_amounts(text)])


def parse_invoice(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(
        text, filename, "Invoice", extra_guidance=_GUIDANCE, enrich=_enrich
    )
