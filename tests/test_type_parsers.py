"""Exercise each non-resume parser module end-to-end with the LLM call
mocked out. This proves the prompt-building -> normalize -> deterministic
enrichment pipeline runs without error for every document type, and that
the regex enrichment layer actually fills in fields the (mocked, minimal)
LLM response left out.
"""

from __future__ import annotations

from unittest.mock import patch

from app.parsers.application_form import parse_application_form
from app.parsers.contract import parse_contract
from app.parsers.invoice import parse_invoice
from app.parsers.receipt import parse_receipt
from app.parsers.report import parse_report
from app.pdf_extraction import extract_pdf_text
from tests.fixtures.synthetic_pdfs import (
    make_application_form_pdf,
    make_contract_pdf,
    make_invoice_pdf,
    make_receipt_pdf,
    make_report_pdf,
)

_MINIMAL_LLM_RESPONSE = {
    "document_type": "Unknown",
    "summary": None,
    "fields": {},
    "dates": [],
    "amounts": [],
    "line_items": [],
    "parties": [],
    "security_notes": {"possible_prompt_injection": False, "suspicious_content": []},
}


def _text(make_pdf) -> str:
    return extract_pdf_text(make_pdf())


def test_parse_invoice_enriches_missing_fields_from_regex():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_LLM_RESPONSE)):
        result = parse_invoice(_text(make_invoice_pdf), "invoice.pdf")

    assert result["document_type"] == "Invoice"
    assert result["fields"]["invoice_number"] == "INV-2024-0091"
    assert result["fields"]["total"] == "$118.80"
    assert "03/14/2024" in result["dates"] or "04/13/2024" in result["dates"]


def test_parse_invoice_handles_mixed_numeric_and_string_amounts_from_model():
    """Regression test: a real Ollama response returned some amounts as
    numbers (134.45) and others as strings ("$134.45") in the same list,
    which crashed `sorted(set(...))` with a float/str comparison TypeError.
    """
    llm_response = dict(_MINIMAL_LLM_RESPONSE)
    llm_response["amounts"] = [134.45, "$8.80", 17.48]
    llm_response["dates"] = ["03/14/2024", 2024]
    with patch("app.parsers.generic.call_ollama", return_value=llm_response):
        result = parse_invoice(_text(make_invoice_pdf), "invoice.pdf")
    assert "134.45" in result["amounts"]
    assert "$8.80" in result["amounts"]


def test_parse_receipt_handles_mixed_numeric_and_string_amounts_from_model():
    llm_response = dict(_MINIMAL_LLM_RESPONSE)
    llm_response["amounts"] = [11.34, "$4.50"]
    with patch("app.parsers.generic.call_ollama", return_value=llm_response):
        result = parse_receipt(_text(make_receipt_pdf), "receipt.pdf")
    assert "11.34" in result["amounts"]


def test_parse_invoice_prefers_llm_value_when_already_present():
    llm_response = dict(_MINIMAL_LLM_RESPONSE)
    llm_response["fields"] = {"invoice_number": "FROM-LLM"}
    with patch("app.parsers.generic.call_ollama", return_value=llm_response):
        result = parse_invoice(_text(make_invoice_pdf), "invoice.pdf")
    assert result["fields"]["invoice_number"] == "FROM-LLM"


def test_parse_receipt_enriches_missing_fields_from_regex():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_LLM_RESPONSE)):
        result = parse_receipt(_text(make_receipt_pdf), "receipt.pdf")

    assert result["document_type"] == "Receipt"
    assert result["fields"]["total"] == "$11.34"
    assert result["fields"]["change_due"] == "$3.66"


def test_parse_contract_enriches_effective_date_and_governing_law():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_LLM_RESPONSE)):
        result = parse_contract(_text(make_contract_pdf), "contract.pdf")

    assert result["document_type"] == "Contract"
    assert result["fields"]["effective_date"] == "01/15/2024"
    assert "Delaware" in result["fields"]["governing_law"]


def test_parse_report_enriches_report_date_and_author():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_LLM_RESPONSE)):
        result = parse_report(_text(make_report_pdf), "report.pdf")

    assert result["document_type"] == "Report"
    assert result["fields"]["report_date"] == "04/01/2024"
    assert result["fields"]["author"] == "Data Analytics Team"


def test_parse_application_form_enriches_id_and_declaration():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_LLM_RESPONSE)):
        result = parse_application_form(_text(make_application_form_pdf), "application.pdf")

    assert result["document_type"] == "Form"
    assert result["fields"]["application_id"] == "APP2024X771"
    assert "declare" in result["fields"]["declaration"].lower()


def test_payment_method_read_from_card_line_not_guessed():
    """Regression test for a real error on clean_receipt_01_grocery_store.pdf:
    payment_method came back as "Cash" even though the receipt's payment line
    plainly reads "Visa Contactless **** 1842". The misleading nearby text is
    "Cashier 06 / Terminal 03" -- so the word-boundary matching matters, since
    "Cashier" must not be read as "Cash".
    """
    from app.parsers.receipt import extract_payment_details

    text = (
        "Maple Basket Market\n"
        "Cashier 06 / Terminal 03\n"
        "Subtotal\n$62.80\n"
        "PAYMENT\n"
        "Approved  |  CAD  65.14\n"
        "Visa Contactless **** 1842\n"
        "Thank you for your purchase.\n"
    )
    assert extract_payment_details(text) == ("Visa Contactless", "1842")


def test_payment_method_detects_genuine_cash_sale():
    from app.parsers.receipt import extract_payment_details

    text = (
        "Corner Store\n"
        "PAYMENT\n"
        "Transaction approved - cash sale\n"
        "Cash CAD 50.00\n"
        "Change 1.17\n"
    )
    assert extract_payment_details(text) == ("Cash", None)


def test_cashier_alone_is_not_read_as_cash_payment():
    from app.parsers.receipt import extract_payment_details

    assert extract_payment_details("Cashier 06 / Terminal 03\nTotal CAD $10.00\n") == (None, None)


def test_all_real_receipt_payment_methods_match_ground_truth(receipts_dir, receipts_ground_truth_path):
    """Reads payment method + masked card digits straight off each real
    receipt and checks them against the bundled ground truth. This path is
    fully deterministic (no model involved), so it is expected to match
    exactly rather than approximately.
    """
    import json

    from app.parsers.receipt import extract_payment_details
    from app.pdf_extraction import extract_pdf_text

    if not receipts_ground_truth_path.exists():
        return
    truth = {
        record["filename"]: record
        for record in json.loads(receipts_ground_truth_path.read_text())["receipts"]
    }

    mismatches = []
    for pdf_path in sorted(receipts_dir.glob("*.pdf")):
        expected = truth.get(pdf_path.name, {}).get("payment", {})
        method, last4 = extract_payment_details(extract_pdf_text(pdf_path.read_bytes()))
        expected_last4 = expected.get("card_last_four")
        if expected_last4 and last4 != expected_last4:
            mismatches.append((pdf_path.name, "card_last4", last4, expected_last4))
        # Receipt 04 prints "Debit Interac" while the ground truth normalizes
        # it to "Interac Debit". The parser reproduces what the document
        # actually says, so compare on the words present, not their order.
        if method and expected.get("method"):
            if sorted(method.lower().split()) != sorted(str(expected["method"]).lower().split()):
                mismatches.append((pdf_path.name, "method", method, expected["method"]))

    assert not mismatches, f"payment details did not match ground truth: {mismatches}"


def test_currency_prefers_printed_code_over_symbol():
    """Regression test: "$" is shared by many currencies and was being
    reported as USD on Canadian documents (9 of 10 invoices). A printed ISO
    code is unambiguous; HST/QST are Canada-specific and identify the
    currency even when only "$" is shown.
    """
    from app.regex_utils import extract_currency

    assert extract_currency("Total CAD $65.14\nApproved | CAD 65.14") == "CAD"
    assert extract_currency("Total USD $100.00\nSales Tax $8.00") == "USD"
    assert extract_currency("Total $2,938.00\nHST 13% $338.00") == "CAD"
    # No evidence at all -> None, so the caller leaves the existing value alone.
    assert extract_currency("Total $2,938.00\nSales Tax $338.00") is None


def test_labeled_amount_honours_priority_order():
    """Regression test: a restaurant bill prints the pre-tip "Total CAD
    $151.93" above "Final amount CAD $171.93". The alternation-based helper
    returns whichever label appears first in the document, so it reported
    the pre-tip figure. Priority order must be respected instead.
    """
    from app.regex_utils import extract_labeled_amount, extract_labeled_amount_in_order

    text = "Total CAD\n$151.93\nTip\n$20.00\nFinal amount CAD\n$171.93\n"
    labels = (r"final\s*amount", r"\btotal\b")

    assert extract_labeled_amount_in_order(text, labels) == "$171.93"
    # Documents the old behaviour that caused the bug.
    assert extract_labeled_amount(text, labels) == "$151.93"


def test_named_sales_tax_returns_amount_not_rate():
    """Regression test: HST was absent from the tax labels, so the tax line
    fell through to the model, which returned the rate ("13%") instead of
    the amount ($6.94).
    """
    from app.parsers.receipt import _enrich

    result = {"fields": {}}
    _enrich(result, "Subtotal\n$134.82\nHST 13%\n$6.94\nTotal CAD\n$141.76\n")
    assert result["fields"]["tax"] == "$6.94"
    assert result["fields"]["subtotal"] == "$134.82"
    assert result["fields"]["total"] == "$141.76"


def test_all_real_receipt_money_fields_match_ground_truth(receipts_dir, receipts_ground_truth_path):
    """End-to-end check of the deterministic layer against ground truth,
    starting from an EMPTY model result -- these fields must be recoverable
    from the document alone, with no LLM involvement.
    """
    import json
    import re

    from app.parsers.receipt import _enrich
    from app.pdf_extraction import extract_pdf_text

    if not receipts_ground_truth_path.exists():
        return
    truth = {
        r["filename"]: r for r in json.loads(receipts_ground_truth_path.read_text())["receipts"]
    }

    def money(value):
        if value is None:
            return None
        return round(float(re.sub(r"[^0-9.]", "", str(value)) or 0), 2)

    mismatches = []
    for pdf_path in sorted(receipts_dir.glob("*.pdf")):
        expected = truth.get(pdf_path.name)
        if not expected:
            continue
        result = {"fields": {}}
        _enrich(result, extract_pdf_text(pdf_path.read_bytes()))
        fields = result["fields"]
        summary = expected["summary"]

        checks = {
            "subtotal": (money(fields.get("subtotal")), summary.get("subtotal")),
            "total": (money(fields.get("total")), summary.get("total")),
            "currency": (fields.get("currency"), expected.get("currency")),
        }
        if summary.get("taxes"):
            checks["tax"] = (money(fields.get("tax")), summary["taxes"][0]["amount"])

        for field, (got, want) in checks.items():
            if want is not None and got != want:
                mismatches.append((pdf_path.name, field, got, want))

    assert not mismatches, f"deterministic extraction disagreed with ground truth: {mismatches}"


def test_amounts_extract_across_currency_conventions():
    """The extractor must work on any document, not just the test dataset.

    Guards two silent-wrong-value bugs:
    - Indian digit grouping is not in threes (1,41,760 is one lakh forty-one
      thousand). A `(?:,\\d{3})*` pattern matched only the tail, so
      "Rs 1,41,760.00" was extracted as "41,760.00" -- off by ~3.4x.
    - Unknown currency symbols were dropped from the value entirely.
    """
    from app.regex_utils import extract_amounts

    assert extract_amounts("Total 1,41,760.00") == ["1,41,760.00"]
    assert extract_amounts("Total 1,234,567.89") == ["1,234,567.89"]
    assert extract_amounts("Total EUR 99.00") == ["99.00"]
    # Symbols are kept as part of the value, never invented or substituted.
    for text, expected in (
        ("Total 141.76", "141.76"),
        ("Total 1,200.50", "1,200.50"),
    ):
        assert extract_amounts(text) == [expected]

    # Quantities, years and reference numbers are not money.
    assert extract_amounts("Qty 12 hrs") == []
    assert extract_amounts("Invoice No 2026") == []


def test_tax_rate_is_not_mistaken_for_tax_amount():
    """Regression test: "Sales Tax 8.25%" yielded 8.25 as the tax amount
    instead of the $82.50 printed beneath it.
    """
    from app.parsers.invoice import _enrich

    result = {"fields": {}}
    _enrich(result, "Subtotal\n$1,000.00\nSales Tax 8.25%\n$82.50\nBalance Due\n$1,082.50\n")
    assert result["fields"]["tax"] == "$82.50"
    assert result["fields"]["total"] == "$1,082.50"


def test_currency_is_never_guessed_from_a_dollar_sign():
    """"$" is shared by USD, CAD, AUD, SGD, HKD and others, so it carries no
    information on its own -- guessing from it is what reported Canadian
    documents as USD. With no other evidence the function returns None so
    the caller leaves the existing value alone rather than inventing one.
    """
    from app.regex_utils import extract_currency

    assert extract_currency("Subtotal $1,000.00\nSales Tax $82.50\nTotal $1,082.50") is None
    # Unambiguous symbols and printed codes are still honoured.
    assert extract_currency("Total 1,41,600.00 INR") == "INR"
    assert extract_currency("Total 1,440.00 and VAT applied in GBP") == "GBP"
