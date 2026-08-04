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
from collections import Counter
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

# Currency symbols, longest-first so "R$" is tried before a bare "$".
# Deliberately not limited to $/EUR/GBP: an invoice may be issued in any
# currency, and a symbol the pattern doesn't know gets silently dropped
# from the extracted value.
_CURRENCY_SYMBOL = r"(?:R\$|RM|Rs\.?|₹|[$€£¥₩₪₺฿₫₦₱]|CHF|kr)"

# Digit grouping is not universally in threes. Indian numbering groups the
# last three digits and then in twos -- 1,41,760.00 is one lakh forty-one
# thousand. A `(?:,\d{3})*` pattern cannot match that, and because the
# regex still matched the *tail*, "₹1,41,760.00" was extracted as
# "41,760.00": a silently wrong number, off by a factor of ~3.4. Allowing
# 2- or 3-digit groups covers both conventions.
_GROUPED_NUMBER = r"\d{1,3}(?:,\d{2,3})*(?:\.\d{1,2})?"
_PLAIN_NUMBER = r"\d+(?:\.\d{1,2})?"
# Same grouping, but decimals required -- used for amounts written without a
# currency symbol, so quantities, years and reference numbers aren't
# mistaken for money.
_GROUPED_NUMBER_WITH_DECIMALS = r"\d{1,3}(?:,\d{2,3})*\.\d{2}"

# The trailing (?!\d) guards against a partial match winning: without it,
# the grouped-number branch matches the first three digits of "5000" and
# stops, extracting "¥500". The guard forces backtracking to the branch
# that consumes the whole number.
_AMOUNT_PATTERN = re.compile(
    r"(?<![\w.])(?:"
    # Symbol-prefixed: the symbol is part of the value, so it is captured
    # rather than dropped.
    + _CURRENCY_SYMBOL + r"\s?(?:" + _GROUPED_NUMBER + r"|" + _PLAIN_NUMBER + r")(?!\d)"
    # A number with no currency symbol must not be followed by "%", or a tax
    # *rate* gets captured as the tax *amount*: "Sales Tax 8.25%" yielded
    # 8.25 instead of the $82.50 printed on the next line.
    + r"|" + _GROUPED_NUMBER_WITH_DECIMALS + r"(?![\d%])"
    + r"|\b\d+\.\d{2}\b(?!\s*%)"
    + r")"
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


# ISO 4217 codes that appear verbatim on documents ("Total CAD $65.14",
# "Approved | CAD 65.14"). A printed code is unambiguous, unlike a bare "$"
# which is shared by many currencies -- a Canadian receipt showing "$" was
# repeatedly reported as USD.
_CURRENCY_CODE_PATTERN = re.compile(
    r"\b(CAD|USD|EUR|GBP|AUD|NZD|INR|JPY|CHF|SEK|NOK|DKK|MXN|BRL|ZAR|SGD|HKD|CNY|AED)\b"
)


# Symbols that map to exactly one currency. "$" is absent on purpose -- it
# is shared by many currencies, so it carries no information on its own.
_UNAMBIGUOUS_SYMBOLS = {
    "₹": "INR",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₩": "KRW",
    "₪": "ILS",
    "₺": "TRY",
    "฿": "THB",
    "₫": "VND",
    "₦": "NGN",
    "₱": "PHP",
    "R$": "BRL",
}

# Sales taxes unique to a single country, which therefore identify the
# currency when no code or unambiguous symbol is printed. Deliberately
# excludes bare "GST" (Australia, India, Singapore, New Zealand, Canada all
# use it) and "VAT" (used across Europe and beyond).
_JURISDICTION_TAXES = (
    (r"\b(HST|QST)\b", "CAD"),          # Harmonized / Quebec sales tax: Canada only
    (r"\b(CGST|SGST|IGST|UTGST)\b", "INR"),  # India's split GST components
)

# A tax name that several countries share, paired with an address format
# unique to one of them. "GST" alone is used by Canada, Australia, India,
# Singapore and New Zealand, so it cannot identify a currency on its own --
# but a Canadian postal code (letter-digit-letter digit-letter-digit, e.g.
# "V6C 1V5") is not used anywhere else, and the two together are decisive.
# This matters in practice: Ontario invoices print HST and were already
# handled, while GST invoices from BC, Alberta and Manitoba were not.
_CANADIAN_POSTAL_CODE = r"\b[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d\b"
_AMBIGUOUS_TAX_WITH_ADDRESS = (
    (r"\bGST\b", _CANADIAN_POSTAL_CODE, "CAD"),
)


def extract_currency(text: str) -> str | None:
    """Return the currency code printed on the document, if any.

    Uses the most frequently occurring code rather than the first, so a
    single stray mention doesn't outweigh the document's actual currency.
    Returns None when no explicit code appears -- callers should leave the
    existing value alone in that case rather than guessing from a symbol.
    """
    codes = _CURRENCY_CODE_PATTERN.findall(text.upper())
    if codes:
        return Counter(codes).most_common(1)[0][0]

    # No printed code. Fall back to an unambiguous currency symbol -- one
    # that maps to exactly one currency. "$" is deliberately excluded: it is
    # shared by the US, Canada, Australia, Singapore, Hong Kong, Mexico and
    # others, so guessing from it is how Canadian documents got reported as
    # USD in the first place.
    for symbol, code in _UNAMBIGUOUS_SYMBOLS.items():
        if symbol in text:
            return code

    # Still nothing. A jurisdiction-specific tax name identifies the country,
    # and therefore the currency, even when only "$" is printed. Only taxes
    # unique to one country are listed: bare "GST" is excluded because it is
    # also used in Australia, India, Singapore and New Zealand, and bare
    # "VAT" is used across Europe and beyond.
    for pattern, code in _JURISDICTION_TAXES:
        if re.search(pattern, text, re.IGNORECASE):
            return code

    # A shared tax name plus a country-unique address format.
    for tax_pattern, address_pattern, code in _AMBIGUOUS_TAX_WITH_ADDRESS:
        if re.search(tax_pattern, text, re.IGNORECASE) and re.search(address_pattern, text):
            return code

    # No evidence. Returning None is deliberate -- callers leave the existing
    # value untouched rather than defaulting to a currency the document never
    # mentions.
    return None


def extract_labeled_amount_in_order(text: str, label_patterns: tuple[str, ...]) -> str | None:
    """Like `extract_labeled_amount`, but honours label priority.

    `extract_labeled_amount` ORs every label into one alternation, so it
    returns whichever label happens to appear *earliest in the document*,
    regardless of the order they were listed in. That silently defeats the
    intent of an ordered tuple. On a restaurant bill printing both

        Total CAD          $151.93
        Tip                 $20.00
        Final amount CAD   $171.93

    the alternation matches "Total" first and reports the pre-tip amount as
    the total. Trying patterns in the given order instead lets a caller say
    "prefer 'final amount' over a bare 'total'".
    """
    for pattern in label_patterns:
        match = re.search(rf"(?im)\b(?:{pattern})\b.{{0,40}}", text)
        if not match:
            continue
        window = text[match.start() : match.end() + 40]
        amount_match = _AMOUNT_PATTERN.search(window)
        if amount_match:
            return amount_match.group(0)
    return None


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


def _money_to_float(value: Any) -> float | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if not cleaned or cleaned in {"-", "."}:
        return None
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return None


# Rounding on a printed receipt is to the cent; a couple of cents of slack
# absorbs legitimate rounding without hiding a real discrepancy.
_TOTAL_TOLERANCE = 0.05


def check_totals_consistency(fields: dict[str, Any]) -> str | None:
    """Verify that subtotal + tax (+ tip) reconciles to the stated total.

    This is the strongest available defence against tampering with money,
    because it does not depend on recognising *how* the document was
    manipulated. One sample attack simply deleted the word "Total" from the
    line "Total CAD", so the label no longer matched and extraction picked
    up a different figure -- reporting $55.80 on a receipt that actually
    totalled $60.79. No amount of phrase matching catches that, but the
    arithmetic does: 53.80 + 6.99 does not equal 55.80.

    Returns a description of the discrepancy, or None when the figures
    reconcile or there is not enough information to judge.
    """
    subtotal = _money_to_float(fields.get("subtotal"))
    total = _money_to_float(fields.get("total") or fields.get("amount_due"))
    if subtotal is None or total is None:
        return None

    tax = _money_to_float(fields.get("tax")) or 0.0
    tip = _money_to_float(fields.get("tip") or fields.get("tip/gratuity")) or 0.0
    discount = _money_to_float(fields.get("discount")) or 0.0

    expected = round(subtotal + tax + tip - abs(discount), 2)
    if abs(expected - total) <= _TOTAL_TOLERANCE:
        return None

    # A discount the parser didn't capture would also explain a total lower
    # than the sum, so say what was compared rather than asserting fraud.
    return (
        f"Totals do not reconcile: subtotal {subtotal:.2f} + tax {tax:.2f}"
        + (f" + tip {tip:.2f}" if tip else "")
        + (f" - discount {abs(discount):.2f}" if discount else "")
        + f" = {expected:.2f}, but the extracted total is {total:.2f}."
    )
