from __future__ import annotations

from app.regex_utils import extract_amounts, extract_dates, extract_labeled_amount, extract_labeled_value


def test_extract_dates_multiple_formats():
    text = "Signed on 03/14/2024. ISO date 2024-03-14. Written as March 14, 2024."
    dates = extract_dates(text)
    assert "03/14/2024" in dates
    assert "2024-03-14" in dates
    assert "March 14, 2024" in dates


def test_extract_amounts_dollar_and_plain():
    text = "Subtotal: $1,250.00 Tax: $100.00 Total 1350.00"
    amounts = extract_amounts(text)
    assert "$1,250.00" in amounts
    assert "$100.00" in amounts


def test_extract_labeled_value_finds_nearest_match():
    text = "Invoice Number: INV-2024-0091\nDue Date: 04/13/2024"
    value = extract_labeled_value(text, (r"invoice\s*(?:#|no\.?|number)",))
    assert value == "INV-2024-0091"


def test_extract_labeled_value_returns_none_when_absent():
    assert extract_labeled_value("no relevant labels here", (r"invoice\s*number",)) is None


def test_extract_labeled_amount_finds_nearby_currency():
    text = "Grand Total: $118.80 after tax"
    assert extract_labeled_amount(text, (r"grand\s*total",)) == "$118.80"
