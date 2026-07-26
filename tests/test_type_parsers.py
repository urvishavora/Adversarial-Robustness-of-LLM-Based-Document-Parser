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
