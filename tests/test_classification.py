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


def test_classifies_every_real_resume(resumes_dir):
    """Regression test: test_classifies_resume above only ever checked the
    first resume alphabetically, which let a real bug slip through
    undetected -- several of the real sample resumes render section headers
    with deliberate letter-spacing ("E D U C A T I O N"), which a plain
    string-equality header check doesn't recognize as "education". Checking
    every sample file (not just one) is what actually catches that class of
    bug.
    """
    pdf_files = sorted(resumes_dir.glob("*.pdf"))
    assert pdf_files
    misclassified = []
    for pdf_path in pdf_files:
        text = extract_pdf_text(pdf_path.read_bytes())
        predicted = infer_document_type(text, pdf_path.name)
        if predicted != "Resume":
            misclassified.append((pdf_path.name, predicted))
    assert not misclassified, f"expected all real resumes to classify as Resume: {misclassified}"


def test_unknown_when_no_keywords_match():
    assert infer_document_type("lorem ipsum dolor", "file.pdf") == "Unknown"


def test_academic_report_not_misclassified_as_resume():
    """Regression test for a real bug: an academic-style report whose body
    text happens to mention generic resume words ("education", "experience",
    "skills") inline -- e.g. describing a dataset's demographic columns or a
    method's track record -- was outscoring an actual Report on keyword
    count alone, because those single generic words were counted as plain
    substring hits anywhere in the text. A real resume uses those words as
    short section headings, not buried in sentences, so only header-style
    occurrences should count toward the Resume score.
    """
    text = (
        "Abstract\n"
        "This report presents findings from a study of patient education, "
        "clinical experience, and provider skills across several hospital "
        "sites. The dataset includes patient education level as one of "
        "several demographic attributes used in the analysis.\n"
        "Methodology\n"
        "Data was collected from health records covering employment status, "
        "education level, and years of clinical experience.\n"
        "Findings\n"
        "The methodology identified several significant correlations.\n"
        "Recommendations\n"
        "We recommend further study of these factors.\n"
        "Conclusion\n"
        "This report's findings support the proposed recommendations."
    )
    assert infer_document_type(text, "study_report.pdf") == "Report"


def test_resume_with_plain_section_headings_still_classifies_correctly():
    text = (
        "Jordan Lee\n"
        "Experience\n"
        "Senior Analyst at Example Corp, 2020-2024\n"
        "Education\n"
        "B.S. Computer Science, State University\n"
        "Skills\n"
        "Python, SQL, data visualization\n"
    )
    assert infer_document_type(text, "jordan_lee_resume.pdf") == "Resume"


def test_resume_with_letter_spaced_headings_still_classifies_correctly():
    """Regression test: some real resume templates render section headers
    with per-letter spacing for visual style ("E D U C A T I O N"), which
    PyMuPDF extracts as literal space characters between every letter. The
    header check needs to recognize that as "education" too, not just an
    exact, unspaced match.
    """
    text = (
        "J o r d a n   L e e\n"
        "W O R K   E X P E R I E N C E\n"
        "Senior Analyst at Example Corp, 2020-2024\n"
        "E D U C A T I O N\n"
        "B.S. Computer Science, State University\n"
        "S K I L L S\n"
        "Python, SQL, data visualization\n"
    )
    assert infer_document_type(text, "jordan_lee_resume.pdf") == "Resume"


def test_form_mentioning_work_experience_field_not_misclassified_as_resume():
    """Regression test: a loan/job application form that asks for a
    numeric "Total Work Experience" or has an "Employment Details" section
    is not itself a resume just because it uses that phrase as a field
    label. Only "curriculum vitae"/"professional summary"-style phrases
    that are essentially resume-exclusive should count as plain substring
    hits; everyday phrases like "work experience" need the header-grouping
    or checkbox signal to actually distinguish a resume from a form.
    """
    text = (
        "HOME LOAN APPLICATION FORM\n"
        "FORM - B (EMPLOYMENT DETAILS)\n"
        "Organization Type [ ] Public Sector [ ] Private [ ] Government\n"
        "Employment Status [ ] Permanent [ ] Contract\n"
        "Total Work Experience Years [ ][ ] Months [ ][ ]\n"
        "Is there a break in service beyond 3 months? [ ] Yes [ ] No\n"
        "Date of Birth [ ][ ] / [ ][ ] / [ ][ ][ ][ ]\n"
    )
    assert infer_document_type(text, "home_loan_application.pdf") == "Form"


def test_receipt_titled_sales_invoice_still_classifies_as_receipt():
    """Regression test for a real bug found on a real sample
    (clean_receipt_05_electronics_store.pdf): a store receipt whose header
    reads "SALES INVOICE" tied 2-2 against Receipt, and the tie was
    silently resolved by dict insertion order -- Invoice happened to be
    declared first -- so it came out Invoice.

    Two things were wrong. "subtotal" was scored as an Invoice signal even
    though it appears on essentially every receipt too (10/10 of the real
    sample receipts), so it could only ever add noise. And ties resolved on
    declaration order rather than on evidence. The real distinction is
    purpose: an invoice requests payment not yet made, a receipt documents
    payment already completed -- this document shows a completed card
    payment and no amount due.
    """
    text = (
        "Northstar Circuit Depot\n"
        "25 Vector Park Drive\n"
        "SALES INVOICE\n"
        "Receipt: NCD-INV-726104\n"
        "2 USB-C Cable 2 m $18.99 $37.98\n"
        "Subtotal $361.39\n"
        "HST 13% $46.98\n"
        "Total CAD $408.37\n"
        "PAYMENT\n"
        "Visa **** 4480\n"
        "Thank you for your purchase.\n"
    )
    assert infer_document_type(text, "clean_receipt_05_electronics_store.pdf") == "Receipt"


def test_real_invoice_still_classifies_as_invoice():
    """The receipt-vs-invoice fix must not swing the other way: a genuine
    invoice -- one requesting payment that hasn't happened yet -- must
    still be Invoice even though it shares subtotal/tax/total vocabulary
    with receipts.
    """
    text = (
        "Acme Supplies Inc.\n"
        "INVOICE\n"
        "Invoice Number: INV-2024-0091\n"
        "Bill To: Contoso LLC\n"
        "Due Date: 04/13/2024\n"
        "Subtotal: $110.00\n"
        "Tax: $8.80\n"
        "Amount Due: $118.80\n"
        "Payment Terms: Net 30\n"
    )
    assert infer_document_type(text, "invoice_2024_0091.pdf") == "Invoice"


def test_report_with_approval_block_not_misclassified_as_receipt():
    """Regression test: "Approved" reads like card-authorization language,
    but approval/sign-off blocks are just as common in reports, contracts,
    and forms. Scoring a bare "approved" as a Receipt signal mislabeled an
    8D quality report as a Receipt, so it must not count on its own.
    """
    text = (
        "Automotive 8D Report - CNC Housing Burr\n"
        "Northstar Automotive Systems\n"
        "Problem Description\n"
        "The customer found a sharp burr on the oil-channel cross hole.\n"
        "Corrective Action\n"
        "Set brush replacement at 7,500 cycles.\n"
        "Prepared by: Maria Chen, Quality Manager\n"
        "Approved by: Plant Quality Director\n"
    )
    assert infer_document_type(text, "automotive-8d-demo.pdf") == "Report"


def test_all_real_receipts_classify_as_receipt(receipts_dir):
    """Checks every real receipt in the bundled baseline dataset, not just
    one -- the "SALES INVOICE" receipt above was the only one of the ten
    that failed, so a single-file check would have missed it entirely.
    """
    pdf_files = sorted(receipts_dir.glob("*.pdf"))
    assert pdf_files, "expected the bundled receipt sample PDFs"

    misclassified = []
    for pdf_path in pdf_files:
        predicted = infer_document_type(extract_pdf_text(pdf_path.read_bytes()), pdf_path.name)
        if predicted != "Receipt":
            misclassified.append((pdf_path.name, predicted))
    assert not misclassified, f"expected all real receipts to classify as Receipt: {misclassified}"


def test_checkbox_signal_not_confused_by_numeric_citations():
    """Regression test: the checkbox-density signal that helps classify
    heavily-OCR'd, checkbox-dense forms must not fire on an academic
    paper's numeric bracket citations (e.g. "[1]", "[12]"), which look
    superficially similar but are not checkboxes.
    """
    text = (
        "Abstract\n"
        "Prior work [1] has explored this problem extensively [2]. Building "
        "on [3] and [4], this report proposes a new method [5][6].\n"
        "Methodology\n"
        "We extend the approach described in [7].\n"
        "Findings\n"
        "Results improve on the baseline reported in [1] by 12%.\n"
        "Recommendations\n"
        "Future work should consider [8].\n"
    )
    assert infer_document_type(text, "citation_heavy_report.pdf") == "Report"
