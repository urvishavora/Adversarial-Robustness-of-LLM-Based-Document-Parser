from __future__ import annotations

from app.classification import infer_document_type
from app.pdf_extraction import extract_pdf_text
from tests.fixtures.synthetic_pdfs import (
    make_application_form_pdf,
    make_contract_pdf,
    make_invoice_pdf,
    make_receipt_pdf,
    make_report_pdf,
)


def _extract(make_pdf) -> str:
    return extract_pdf_text(make_pdf())


def test_classifies_invoice():
    assert infer_document_type(_extract(make_invoice_pdf), "invoice.pdf") == "Invoice"


def test_classifies_receipt():
    assert infer_document_type(_extract(make_receipt_pdf), "receipt.pdf") == "Receipt"


def test_classifies_contract():
    assert infer_document_type(_extract(make_contract_pdf), "contract.pdf") == "Contract"


def test_classifies_report():
    assert infer_document_type(_extract(make_report_pdf), "report.pdf") == "Report"


def test_classifies_application_form():
    assert infer_document_type(_extract(make_application_form_pdf), "application.pdf") == "Form"


def test_classifies_resume(resumes_dir):
    pdf_files = sorted(resumes_dir.glob("*.pdf"))
    assert pdf_files
    text = extract_pdf_text(pdf_files[0].read_bytes())
    assert infer_document_type(text, pdf_files[0].name) == "Resume"


def test_unknown_when_no_keywords_match():
    assert infer_document_type("lorem ipsum dolor", "file.pdf") == "Unknown"
