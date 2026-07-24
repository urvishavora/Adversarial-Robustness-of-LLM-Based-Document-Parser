"""Synthetic PDF generators used for testing.

These stand in for real-world samples of invoice/receipt/contract/report/
application-form documents so the extraction + classification + regex
enrichment layers can be exercised with real PDF bytes (not just raw
strings) even before real user-supplied samples are available. They are
deliberately generic layouts, not modeled on any specific real document.
"""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def _render_lines(lines: list[str]) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    pdf.setFont("Helvetica", 11)
    for line in lines:
        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 11)
            y = height - 50
        pdf.drawString(50, y, line)
        y -= 16
    pdf.save()
    return buffer.getvalue()


def make_invoice_pdf() -> bytes:
    return _render_lines(
        [
            "Acme Supplies Inc.",
            "123 Market Street, Springfield, IL",
            "",
            "INVOICE",
            "Invoice Number: INV-2024-0091",
            "Invoice Date: 03/14/2024",
            "Due Date: 04/13/2024",
            "Purchase Order Number: PO-5521",
            "",
            "Bill To: Contoso LLC, 45 Industrial Way, Chicago, IL",
            "",
            "Description          Quantity   Unit Price   Amount",
            "Widget A             10         $5.00        $50.00",
            "Widget B             5          $12.00       $60.00",
            "",
            "Subtotal: $110.00",
            "Tax: $8.80",
            "Total Due: $118.80",
            "Payment Terms: Net 30",
        ]
    )


def make_receipt_pdf() -> bytes:
    return _render_lines(
        [
            "Corner Cafe",
            "88 Baker Street",
            "Receipt Number: R-778812",
            "Date: 07/02/2026  Time: 14:32",
            "",
            "Item                 Qty   Price   Amount",
            "Latte                1     $4.50   $4.50",
            "Bagel                2     $3.00   $6.00",
            "",
            "Subtotal: $10.50",
            "Tax: $0.84",
            "Total: $11.34",
            "Payment Method: Visa ending 4242",
            "Amount Tendered: $15.00",
            "Change Due: $3.66",
            "Thank you for your purchase!",
        ]
    )


def make_contract_pdf() -> bytes:
    return _render_lines(
        [
            "SERVICE AGREEMENT",
            "",
            "This Agreement is entered into as of Effective Date: 01/15/2024,",
            "by and between Northwind Traders (\"Client\") and Fabrikam Consulting",
            "(\"Contractor\"), collectively the parties.",
            "",
            "1. TERM",
            "This Agreement shall remain in effect for twelve (12) months.",
            "",
            "2. PAYMENT TERMS",
            "Client shall pay Contractor $5,000.00 per month.",
            "",
            "3. GOVERNING LAW",
            "Governing Law: State of Delaware.",
            "",
            "4. TERMINATION",
            "Either party may terminate this agreement with 30 days written notice.",
            "",
            "IN WITNESS WHEREOF, the parties have executed this Agreement.",
            "Signature: ______________  Name: Jane Doe  Title: CEO  Date: 01/15/2024",
            "Signature: ______________  Name: John Smith  Title: Director  Date: 01/15/2024",
        ]
    )


def make_report_pdf() -> bytes:
    return _render_lines(
        [
            "QUARTERLY PERFORMANCE REPORT",
            "Prepared By: Data Analytics Team",
            "Report Date: 04/01/2024",
            "Reporting Period: Q1 2024",
            "",
            "EXECUTIVE SUMMARY",
            "Revenue grew 12% quarter over quarter, driven by new customer acquisition.",
            "",
            "FINDINGS",
            "- Customer churn decreased from 5.2% to 4.1%",
            "- Average deal size increased to $8,200",
            "",
            "METHODOLOGY",
            "Data was collected from the CRM and finance systems for Jan-Mar 2024.",
            "",
            "RECOMMENDATIONS",
            "- Increase investment in customer success",
            "- Expand the sales team by two headcount",
            "",
            "CONCLUSION",
            "The quarter showed strong, sustainable growth across all key metrics.",
        ]
    )


def make_application_form_pdf() -> bytes:
    return _render_lines(
        [
            "UNIVERSITY OF SPRINGFIELD",
            "APPLICATION FORM",
            "Application ID: APP2024X771",
            "",
            "PERSONAL INFORMATION",
            "Full Name: Alex Morgan",
            "Date of Birth: 22/05/2001",
            "Email: alex.morgan@example.com",
            "Mobile: +1-555-0102",
            "",
            "PROGRAM INFORMATION",
            "Program Applied For: Computer Science",
            "School / Faculty: School of Engineering",
            "",
            "ACADEMIC QUALIFICATION",
            "Class 10 (High School)   2016   88%",
            "Class 12 (10+2)          2018   91%",
            "",
            "DECLARATION",
            "I hereby declare that the information provided above is true and",
            "correct to the best of my knowledge.",
            "",
            "Signature of Applicant: ______________",
            "Date: 01/06/2024",
        ]
    )


def make_scanned_image_pdf(lines: list[str]) -> bytes:
    """Render text into a PNG image (no embedded text layer) and wrap it in a
    single-page PDF, to exercise the OCR fallback path the same way a real
    flatbed scan of a printed page would.
    """
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=36)
    except TypeError:
        # Older Pillow: load_default() takes no size argument.
        font = ImageFont.load_default()

    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    y = 80
    for line in lines:
        draw.text((80, y), line, fill="black", font=font)
        y += 60

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    from reportlab.lib.utils import ImageReader

    pdf.drawImage(ImageReader(io.BytesIO(png_bytes)), 0, 0, width=letter[0], height=letter[1])
    pdf.save()
    return pdf_buffer.getvalue()
