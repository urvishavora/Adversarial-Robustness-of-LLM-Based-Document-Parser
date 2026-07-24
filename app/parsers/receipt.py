"""Receipt parsing: LLM extraction + deterministic regex enrichment.

Receipts are shorter and more variable in layout than invoices (retail POS
tapes, restaurant checks, ride-share/e-commerce email-style receipts), so
the deterministic layer here focuses on the handful of fields that are
almost always present in a recognizable form: date/time, payment method,
and the final total.
"""

from __future__ import annotations

from typing import Any

from app.json_utils import dedupe_strings
from app.parsers.generic import parse_generic_document
from app.regex_utils import extract_amounts, extract_dates, extract_labeled_amount, extract_labeled_value

_GUIDANCE = """
RECEIPT-SPECIFIC GUIDANCE:
- Include under fields: merchant_name, merchant_address, receipt_number/transaction_id,
  date, time, payment_method (cash/card/etc.), card_last4, cashier, subtotal, tax,
  tip/gratuity, discount, total, amount_tendered, and change_due when present.
- line_items must be an array of objects, each with description/name, quantity, unit_price,
  and amount (or whatever subset of these columns the receipt actually shows).
- Do not compute or correct totals yourself -- copy the printed values exactly.
"""

_RECEIPT_NUMBER_LABELS = (r"receipt\s*(?:#|no\.?|number)", r"transaction\s*(?:#|id|no\.?)", r"order\s*(?:#|no\.?|number)")
_SUBTOTAL_LABELS = (r"sub\s*-?\s*total",)
_TAX_LABELS = (r"tax(?:\s*\(\d+%?\))?", r"vat", r"gst")
_TIP_LABELS = (r"tip", r"gratuity")
_TOTAL_LABELS = (r"total\s*due", r"grand\s*total", r"balance\s*due", r"\btotal\b", r"amount\s*paid")
_CHANGE_LABELS = (r"change\s*(?:due)?",)
_PAYMENT_METHOD_LABELS = (r"payment\s*method", r"paid\s*(?:by|via)", r"tender(?:ed)?\s*type")


def _enrich(result: dict[str, Any], text: str) -> None:
    fields = result.setdefault("fields", {})

    receipt_number = extract_labeled_value(text, _RECEIPT_NUMBER_LABELS)
    if receipt_number and not fields.get("receipt_number"):
        fields["receipt_number"] = receipt_number

    payment_method = extract_labeled_value(text, _PAYMENT_METHOD_LABELS)
    if payment_method and not fields.get("payment_method"):
        fields["payment_method"] = payment_method

    subtotal = extract_labeled_amount(text, _SUBTOTAL_LABELS)
    if subtotal and not fields.get("subtotal"):
        fields["subtotal"] = subtotal

    tax = extract_labeled_amount(text, _TAX_LABELS)
    if tax and not fields.get("tax"):
        fields["tax"] = tax

    tip = extract_labeled_amount(text, _TIP_LABELS)
    if tip and not fields.get("tip"):
        fields["tip"] = tip

    total = extract_labeled_amount(text, _TOTAL_LABELS)
    if total and not fields.get("total"):
        fields["total"] = total

    change_due = extract_labeled_amount(text, _CHANGE_LABELS)
    if change_due and not fields.get("change_due"):
        fields["change_due"] = change_due

    # The model sometimes returns amounts/dates as numbers or mixed types
    # rather than the displayed string (e.g. 134.45 instead of "$134.45").
    # dedupe_strings coerces everything to a comparable string before
    # deduplicating, which also avoids a set/sort TypeError on mixed types.
    result["dates"] = dedupe_strings([*result.get("dates", []), *extract_dates(text)])
    result["amounts"] = dedupe_strings([*result.get("amounts", []), *extract_amounts(text)])


def parse_receipt(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(
        text, filename, "Receipt", extra_guidance=_GUIDANCE, enrich=_enrich
    )
