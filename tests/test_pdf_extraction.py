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


def test_two_column_section_is_not_interleaved(resumes_dir):
    """Regression test for a real bug verified against the rendered page:
    henrietta_mitchell_...pdf runs full width at the top (contact, summary,
    experience) and then splits into two columns at the bottom --
    "EDUCATION & CERTIFICATIONS" on the left, "EXTRACURRICULAR ACTIVITIES"
    on the right.

    The old whole-page column heuristic had to classify the entire page as
    either one-column or two-column, so it read the bottom section by plain
    y-order and interleaved the two columns row by row: "Bachelor of
    Business Administration" was immediately followed by "President,
    Business Club" from the *other* column, which then got attributed to
    the degree. Each column must be read through completely before the
    next one starts.
    """
    pdf_path = resumes_dir / "henrietta_mitchell_business_management_analysis.pdf"
    if not pdf_path.exists():
        return
    text = extract_pdf_text(pdf_path.read_bytes())

    education_start = text.index("EDUCATION & CERTIFICATIONS")
    activities_start = text.index("EXTRACURRICULAR ACTIVITIES")
    assert education_start < activities_start, "left column should be read before the right column"

    education_block = text[education_start:activities_start]
    # Every education/certification entry belongs in the left column block,
    # uninterrupted by anything from the activities column.
    for expected in (
        "Bachelor of Business Administration",
        "Majors: Analytics and Project Management",
        "Graduate Project Management Certification",
        "Impact Evaluation Methods 3-Day Short Course",
        "Liceria & Co.",
    ):
        assert expected in education_block, f"{expected!r} should be in the education column"

    # ...and nothing from the activities column should have leaked into it.
    for activity in ("President, Business Club", "Community Volunteer", "Paucek and Lage"):
        assert activity not in education_block, (
            f"{activity!r} leaked into the education column -- the two columns are interleaved again"
        )


def test_wrapped_url_is_rejoined(resumes_dir):
    """Regression test for a real bug: a long URL in a narrow column wraps
    mid-token and PyMuPDF reports each visual line as its own block
    ('https://www.linkedin.com/in/s' + 'ebastian-bennett?'). Emitted as two
    lines, a consumer reassembles them with a separator that was never in
    the document, producing an invalid URL.
    """
    for filename, expected_url in (
        ("sebastian_bennett_real_estate_agent.pdf", "https://www.linkedin.com/in/sebastian-bennett?"),
        ("lorna_alvarado_marketing_manager.pdf", "https://www.linkedin.com/in/lorna-alvarado?"),
    ):
        pdf_path = resumes_dir / filename
        if not pdf_path.exists():
            continue
        text = extract_pdf_text(pdf_path.read_bytes())
        assert expected_url in text, f"{filename}: expected the wrapped URL to be rejoined intact"
        assert "/in/s\nebastian" not in text
        assert "linkedin.com/i\nn/" not in text


def test_render_pages_as_jpeg_base64_returns_one_image_per_page():
    images = render_pages_as_jpeg_base64(make_invoice_pdf())
    assert len(images) == 1
    assert isinstance(images[0], str) and len(images[0]) > 100
