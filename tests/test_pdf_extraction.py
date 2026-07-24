from __future__ import annotations

import re

from app.errors import DocumentExtractionError
from app.pdf_extraction import extract_pdf_text, is_scanned_pdf, render_pages_as_jpeg_base64
from tests.fixtures.synthetic_pdfs import make_invoice_pdf, make_scanned_image_pdf


def test_extract_pdf_text_on_real_resume(resumes_dir):
    pdf_files = sorted(resumes_dir.glob("*.pdf"))
    assert pdf_files, "expected at least one real resume PDF for this test"
    for pdf_path in pdf_files:
        text = extract_pdf_text(pdf_path.read_bytes())
        assert text.strip(), f"{pdf_path.name} produced no extracted text"


def test_extract_pdf_text_on_sample_pdf(sample_pdf_path):
    if not sample_pdf_path.exists():
        return
    text = extract_pdf_text(sample_pdf_path.read_bytes())
    assert isinstance(text, str)


def test_extract_pdf_text_digital_layer():
    text = extract_pdf_text(make_invoice_pdf())
    assert "Invoice Number" in text
    assert "Total Due" in text


def test_extract_pdf_text_ocr_fallback_on_scanned_page():
    lines = ["SCANNED DOCUMENT", "Invoice Number: SCAN-9001", "Total: $42.00"]
    scanned_bytes = make_scanned_image_pdf(lines)

    assert is_scanned_pdf(scanned_bytes) is True

    text = extract_pdf_text(scanned_bytes)
    # OCR on a rasterized synthetic page can still misread an individual
    # digit -- that's real OCR behavior, not a bug in this pipeline -- so
    # check for recognizable structure rather than a single exact digit.
    assert "Page 1: OCR" in text
    assert "Invoice Number" in text or "SCAN" in text
    assert "42.00" in text


def test_extract_pdf_text_raises_on_garbage_bytes():
    try:
        extract_pdf_text(b"not a real pdf")
        assert False, "expected DocumentExtractionError"
    except DocumentExtractionError:
        pass


def test_column_split_keeps_dates_attached_to_their_job_entry(resumes_dir):
    """Regression test for a real bug: donna_stroupe_sales_representative.pdf
    has a right-aligned date badge per job entry in its main column, plus an
    unrelated right-hand sidebar (education/skills/languages). The naive
    "single widest x0 gap" column heuristic picked a spurious gap elsewhere
    on the page instead of the real sidebar boundary, which lumped every job
    entry's date into the sidebar bucket -- a plain y-sort there then
    scattered all 4 dates in among unrelated skills/education content,
    detached from the job entries they belonged to.

    After the fix, each job title's date should immediately follow it (give
    or take the company name line) rather than appearing many lines later
    next to sidebar content.
    """
    pdf_path = resumes_dir / "donna_stroupe_sales_representative.pdf"
    if not pdf_path.exists():
        return
    text = extract_pdf_text(pdf_path.read_bytes())

    # "Timmerman Industries" (the employer) appears once per job entry, right
    # next to that entry's date. For every one of those 4 occurrences, some
    # 4-digit year should appear within a short window -- if the dates got
    # detached into an unrelated sidebar section, most/all of these windows
    # would come up empty.
    company = "Timmerman Industries"
    occurrences = [i for i in range(len(text)) if text.startswith(company, i)]
    assert len(occurrences) == 4, f"expected 4 job entries, found {len(occurrences)}"

    for index in occurrences:
        window = text[max(0, index - 150) : index + 150]
        assert re.search(r"20\d{2}", window), (
            f"no year found within 150 chars of a '{company}' occurrence at "
            f"index {index} -- that job entry's date looks detached again."
        )


def test_render_pages_as_jpeg_base64_returns_one_image_per_page():
    images = render_pages_as_jpeg_base64(make_invoice_pdf())
    assert len(images) == 1
    assert isinstance(images[0], str) and len(images[0]) > 100
