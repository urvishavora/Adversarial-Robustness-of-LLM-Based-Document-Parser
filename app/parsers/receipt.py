"""Receipt parsing: LLM extraction + deterministic regex enrichment.

Receipts are shorter and more variable in layout than invoices (retail POS
tapes, restaurant checks, ride-share/e-commerce email-style receipts), so
the deterministic layer here focuses on the handful of fields that are
almost always present in a recognizable form: date/time, payment method,
and the final total.
"""

from __future__ import annotations

import re
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
)

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
# Named sales taxes from several jurisdictions, most specific first, so a
# document's own tax label is matched rather than relying on the model.
# Not tuned to any one country: CGST/SGST/IGST (India), HST/QST/PST
# (Canada), GST (several), VAT (Europe and beyond), plus a generic "tax".
_TAX_LABELS = (
    r"cgst", r"sgst", r"igst", r"utgst",
    r"hst", r"qst", r"pst", r"gst",
    r"vat", r"sales\s*tax", r"tax(?:\s*\(\d+%?\))?",
)
_TIP_LABELS = (r"tip", r"gratuity")
# Priority order matters here and is honoured by extract_labeled_amount_in_order.
# A restaurant bill prints "Total CAD $151.93" *above* "Final amount CAD
# $171.93"; the amount actually paid is the latter, so the more specific
# post-tip labels must be tried before a bare "total".
_TOTAL_LABELS = (
    r"final\s*amount",
    r"amount\s*paid",
    r"grand\s*total",
    r"total\s*due",
    r"balance\s*due",
    r"\btotal\b",
)
_CHANGE_LABELS = (r"change\s*(?:due)?",)
_PAYMENT_METHOD_LABELS = (r"payment\s*method", r"paid\s*(?:by|via)", r"tender(?:ed)?\s*type")

# A card payment is printed as its own line naming the method and the masked
# card number: "Visa Contactless **** 1842", "Mastercard **** 6031". That
# line is unambiguous, which matters because the surrounding text can easily
# mislead a model into guessing: a grocery receipt reading
# "Cashier 06 / Terminal 03" was extracted as payment_method "Cash" even
# though the payment line plainly said "Visa Contactless".
_CARD_PAYMENT_LINE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z ./&'-]{1,40}?)[ \t]+\*{2,}[ \t]*(\d{4})[ \t]*$",
    re.MULTILINE,
)
# Word-boundary matched so "Cashier" is not read as "Cash".
_CASH_WORD = re.compile(r"\bcash\b", re.IGNORECASE)


def extract_payment_details(text: str) -> tuple[str | None, str | None]:
    """Read the payment method and masked card digits literally off the receipt.

    Returns ``(method, card_last4)``, either of which may be None. A matched
    card line is strong evidence -- the method is named right next to the
    masked number -- so callers should prefer it over a model's inference.
    The cash fallback is weaker (it only looks for the word anywhere in the
    document) and is treated as a gap-filler rather than an override.
    """
    match = _CARD_PAYMENT_LINE.search(text)
    if match:
        return match.group(1).strip(), match.group(2)
    if _CASH_WORD.search(text):
        return "Cash", None
    return None, None


def _enrich(result: dict[str, Any], text: str) -> None:
    fields = result.setdefault("fields", {})

    receipt_number = extract_labeled_value(text, _RECEIPT_NUMBER_LABELS)
    if receipt_number and not fields.get("receipt_number"):
        fields["receipt_number"] = receipt_number

    # Deterministic payment details take precedence over the model's answer
    # when the receipt actually prints a card line: the printed line is the
    # document's own statement of how it was paid, whereas the model's value
    # is an inference that can be pulled off course by nearby wording.
    card_method, card_last4 = extract_payment_details(text)
    if card_last4:
        fields["payment_method"] = card_method
        if not fields.get("card_last4"):
            fields["card_last4"] = card_last4
    elif card_method and not fields.get("payment_method"):
        fields["payment_method"] = card_method

    payment_method = extract_labeled_value(text, _PAYMENT_METHOD_LABELS)
    if payment_method and not fields.get("payment_method"):
        fields["payment_method"] = payment_method

    subtotal = extract_labeled_amount_in_order(text, _SUBTOTAL_LABELS)
    if subtotal:
        fields["subtotal"] = subtotal

    # Monetary fields use the deterministic value in preference to the
    # model's: a label->amount match is literal source text, whereas the
    # model has been observed reporting a tax *rate* as the tax amount, and
    # picking the pre-tip total on a restaurant bill.
    tax = extract_labeled_amount_in_order(text, _TAX_LABELS)
    if tax:
        fields["tax"] = tax

    tip = extract_labeled_amount(text, _TIP_LABELS)
    if tip and not fields.get("tip"):
        fields["tip"] = tip

    total = extract_labeled_amount_in_order(text, _TOTAL_LABELS)
    if total:
        fields["total"] = total

    currency = extract_currency(text)
    if currency:
        fields["currency"] = currency

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
