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
from app.regex_utils import (
    extract_amounts,
    extract_currency,
    extract_dates,
    extract_labeled_amount,
    extract_labeled_amount_in_order,
    extract_labeled_value,
    check_totals_consistency,
)

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
# Named sales taxes from several jurisdictions, most specific first, so a
# document's own tax label is matched rather than relying on the model.
# Not tuned to any one country: CGST/SGST/IGST (India), HST/QST/PST
# (Canada), GST (several), VAT (Europe and beyond), plus a generic "tax".
_TAX_LABELS = (
    r"cgst", r"sgst", r"igst", r"utgst",
    r"hst", r"qst", r"pst", r"gst",
    r"vat", r"sales\s*tax", r"tax(?:\s*\(\d+%?\))?",
)
# Priority order, honoured by extract_labeled_amount_in_order. A bare
# "total" is listed last so a more specific balance/amount-due label wins
# when the invoice prints both.
_TOTAL_LABELS = (
    r"balance\s*due",
    r"amount\s*due",
    r"total\s*due",
    r"grand\s*total",
    r"\btotal\b",
)


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

    # Monetary fields prefer the deterministic value over the model's. A
    # label->amount match is literal source text; the model was observed
    # transposing digits (5350.00 for a printed 5530.00) and omitting
    # amount_due entirely.
    subtotal = extract_labeled_amount_in_order(text, _SUBTOTAL_LABELS)
    if subtotal:
        fields["subtotal"] = subtotal

    tax = extract_labeled_amount_in_order(text, _TAX_LABELS)
    if tax:
        fields["tax"] = tax

    total = extract_labeled_amount_in_order(text, _TOTAL_LABELS)
    if total:
        fields["total"] = total
        if not fields.get("amount_due"):
            fields["amount_due"] = total

    # "$" alone is shared by many currencies and was consistently read as
    # USD on Canadian invoices; a printed ISO code (or an HST/QST line) is
    # unambiguous.
    currency = extract_currency(text)
    if currency:
        fields["currency"] = currency

    # The model sometimes returns amounts/dates as numbers or mixed types
    # rather than the displayed string (e.g. 134.45 instead of "$134.45").
    # dedupe_strings coerces everything to a comparable string before
    # deduplicating, which also avoids a set/sort TypeError on mixed types.
    # Arithmetic cross-check. Catches tampering with monetary values
    # regardless of the technique used, including a deleted label that
    # silently redirects extraction to the wrong figure.
    discrepancy = check_totals_consistency(fields)
    if discrepancy:
        warnings = result.setdefault("data_quality_warnings", [])
        if isinstance(warnings, list):
            warnings.append(discrepancy)

    result["dates"] = dedupe_strings([*result.get("dates", []), *extract_dates(text)])
    result["amounts"] = dedupe_strings([*result.get("amounts", []), *extract_amounts(text)])


def parse_invoice(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(
        text, filename, "Invoice", extra_guidance=_GUIDANCE, enrich=_enrich
    )
