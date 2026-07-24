"""End-to-end smoke tests through the actual FastAPI app for all six document
types, with the LLM backend mocked out. This is the strongest guarantee this
project's test suite can offer without a live Ollama server: every route,
every classification branch, every parser module's prompt-build ->
normalize -> enrich pipeline, and the handwriting merge step all get
exercised for real, end to end, and must not raise.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures.synthetic_pdfs import (
    make_application_form_pdf,
    make_contract_pdf,
    make_invoice_pdf,
    make_receipt_pdf,
    make_report_pdf,
)

client = TestClient(app)

_MINIMAL_GENERIC_RESPONSE = {
    "document_type": "Unknown",
    "summary": None,
    "fields": {},
    "dates": [],
    "amounts": [],
    "line_items": [],
    "parties": [],
    "security_notes": {"possible_prompt_injection": False, "suspicious_content": []},
}

_MINIMAL_RESUME_RESPONSE = {
    "file_name": "",
    "name": {"full_name": "Jamie Rivera", "first_name": "Jamie", "last_name": "Rivera"},
    "job_title": "Software Engineer",
    "contact": {"phone": "555-0100", "email": "jamie@example.com", "address": None, "linkedin": None},
    "summary": "Experienced engineer.",
    "education": [{"institution": "State University", "degree": "BSc Computer Science"}],
    "experience": [{"company": "Acme", "job_title": "Engineer", "date": "2020-2023"}],
    "skills": ["Python", "SQL"],
    "languages": [],
    "references": [],
    "certifications": [],
    "awards": [],
    "activities": [],
}

_MINIMAL_HANDWRITING_RESPONSE = {"entries": []}


def _upload(filename: str, content: bytes):
    return client.post("/upload", files={"file": (filename, content, "application/pdf")})


def test_home_and_health_endpoints_respond():
    assert client.get("/").status_code == 200
    health = client.get("/health").json()
    assert health["fastapi"] == "running"


def test_upload_rejects_non_pdf():
    response = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_rejects_empty_file():
    response = client.post("/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_upload_invoice_end_to_end():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_GENERIC_RESPONSE)):
        response = _upload("invoice.pdf", make_invoice_pdf())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_document_type"] == "Invoice"
    assert body["parsed_output"]["fields"]["invoice_number"] == "INV-2024-0091"


def test_upload_receipt_end_to_end():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_GENERIC_RESPONSE)):
        response = _upload("receipt.pdf", make_receipt_pdf())
    assert response.status_code == 200
    assert response.json()["predicted_document_type"] == "Receipt"


def test_upload_contract_end_to_end():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_GENERIC_RESPONSE)):
        response = _upload("contract.pdf", make_contract_pdf())
    assert response.status_code == 200
    assert response.json()["predicted_document_type"] == "Contract"


def test_upload_report_end_to_end():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_GENERIC_RESPONSE)):
        response = _upload("report.pdf", make_report_pdf())
    assert response.status_code == 200
    assert response.json()["predicted_document_type"] == "Report"


def test_upload_application_form_end_to_end_with_handwriting_layer():
    with patch("app.parsers.generic.call_ollama", return_value=dict(_MINIMAL_GENERIC_RESPONSE)), patch(
        "app.main.extract_handwritten_application_fields",
        return_value={"entries": [], "confidence": 0.0, "review_required": True, "source": "ollama_vision"},
    ):
        response = _upload("application.pdf", make_application_form_pdf())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_document_type"] == "Form"
    assert "handwritten_fields" in body["parsed_output"]["fields"]


def test_upload_resume_end_to_end(resumes_dir):
    pdf_files = sorted(resumes_dir.glob("*.pdf"))
    assert pdf_files
    with patch("app.parsers.resume.call_ollama", return_value=dict(_MINIMAL_RESUME_RESPONSE)):
        response = _upload(pdf_files[0].name, pdf_files[0].read_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_document_type"] == "Resume"
    assert body["parsed_output"]["name"]["full_name"] == "Jamie Rivera"
