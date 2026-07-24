"""PDF text extraction, layout handling, and OCR.

Three extraction paths, used depending on what a page actually contains:

1. Digital text layer (normal PDFs -- resumes, invoices, receipts, contracts,
   reports, printed forms exported from a word processor). Blocks are put
   back into reading order with simple two-column detection so a sidebar
   doesn't get spliced into the middle of unrelated text.
2. Scanned/image-only pages (no embedded text layer at all). A full-page
   Tesseract OCR pass recovers the printed text.
3. A per-page render-to-base64-JPEG path used only for the handwriting
   vision model, which reads the page as an image rather than as text.
"""

from __future__ import annotations

import io
import logging
import re

import fitz
import pytesseract
from PIL import Image, ImageFilter, ImageOps

from app import config
from app.errors import DocumentExtractionError

logger = logging.getLogger("document_parser")

config.configure_tesseract()


# --- Reading-order / column handling ----------------------------------------

def get_largest_font_line(page: "fitz.Page") -> str:
    """Return the text of the largest-font line on a page.

    Resume/report/contract headers near-universally render the most
    important line (name, title) in the largest font on the page -- this is
    a layout convention, not something that depends on the words used, so it
    sidesteps word-pattern heuristics entirely.
    """
    try:
        raw = page.get_text("dict")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read font metadata: %s", exc)
        return ""

    best_size = 0.0
    best_text = ""
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            parts = []
            line_size = 0.0
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if span_text.strip():
                    parts.append(span_text)
                    line_size = max(line_size, span.get("size", 0))
            line_text = "".join(parts).strip()
            if line_text and line_size > best_size:
                best_size = line_size
                best_text = line_text
    return best_text


def _looks_like_header_is_present(page: "fitz.Page", first_page_text: str) -> bool:
    """Does the largest-font text on the page survive intact in the primary
    block extraction? If not, that header is worth recovering with OCR.
    """
    largest_line = get_largest_font_line(page)
    if not largest_line:
        return True

    normalized_largest = re.sub(r"\s+", " ", largest_line).strip().lower()
    normalized_primary = re.sub(r"\s+", " ", first_page_text).lower()
    return normalized_largest in normalized_primary


def _split_into_columns(blocks: list[tuple]) -> list[list[tuple]] | None:
    """Detect a genuine two-column layout and split blocks accordingly.

    Returns None (meaning: don't split, use a plain vertical sort) unless a
    genuine column boundary is found: a wide x-gap where both sides have a
    substantial, roughly balanced number of blocks, AND no block's content
    actually crosses that boundary.

    That last check matters: naively picking the single widest gap in the
    page's x0 values is unreliable. A resume can have a wide *spurious* gap
    that has nothing to do with the real column layout -- e.g. a second
    reference entry or a right-aligned date badge starting well to the right
    of the main column's left margin, purely because nothing else happens to
    print in between. Picking that gap as the split point then lumps a
    right-aligned date badge (which visually belongs to the row/entry in the
    main column, just aligned to its right edge) into the sidebar bucket,
    and a plain y-sort of the sidebar scatters that date among unrelated
    sidebar content. The fix: a real column boundary is one where blocks
    don't straddle it -- a left-side block's right edge (x1) shouldn't
    extend deep past the split line, since two side-by-side columns by
    definition don't overlap in x. A right-aligned date badge within the
    main column's row *does* end near (not deep past) that column's own
    right margin, so it still passes; a genuinely different, wider block
    (like a paragraph) that happens to start left of a spurious gap but end
    far to its right fails this check and correctly rules the gap out.
    """
    if len(blocks) < 12:
        return None

    xs = sorted(block[0] for block in blocks)
    page_width = max(block[2] for block in blocks) - min(block[0] for block in blocks)
    if page_width <= 0:
        return None

    gap_threshold = page_width * 0.15
    overhang_tolerance = page_width * 0.2

    candidates = sorted(
        (
            (xs[i] - xs[i - 1], (xs[i - 1] + xs[i]) / 2)
            for i in range(1, len(xs))
            if xs[i] - xs[i - 1] >= gap_threshold
        ),
        reverse=True,
    )

    # PyMuPDF occasionally merges two same-row, side-by-side section headers
    # from different columns into a single wide block (e.g. "SUMMARY" and
    # "EDUCATION" printed on the same line in different columns become one
    # "SUMMARY   EDUCATION" block spanning almost the full page width).
    # Such near-full-width blocks are a merge artifact, not evidence that a
    # candidate split is wrong -- they're excluded from the overhang check
    # below (though still assigned to a side by x0, same as any block).
    full_width_threshold = page_width * 0.6

    for _gap, split_x in candidates:
        left = [block for block in blocks if block[0] < split_x]
        right = [block for block in blocks if block[0] >= split_x]

        smaller, larger = sorted((len(left), len(right)))
        if smaller < 8 or smaller / larger < 0.35:
            continue

        max_overhang = max(
            (
                block[2] - split_x
                for block in left
                if (block[2] - block[0]) < full_width_threshold
            ),
            default=0.0,
        )
        if max_overhang > overhang_tolerance:
            continue

        return [left, right]

    return None


def _order_blocks_reading_order(blocks: list[tuple]) -> list[tuple]:
    columns = _split_into_columns(blocks)
    if not columns:
        return sorted(blocks, key=lambda block: (block[1], block[0]))

    ordered: list[tuple] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda block: (block[1], block[0])))
    return ordered


# --- OCR ---------------------------------------------------------------------

def _prepare_ocr_image(page: "fitz.Page", zoom: float = 2.5) -> Image.Image:
    """Render a page at high resolution and improve contrast for OCR."""
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    return image.filter(ImageFilter.SHARPEN)


def ocr_page_text(page: "fitz.Page", zoom: float = 2.5) -> str:
    """OCR a single page. Used for header recovery on otherwise-digital pages."""
    try:
        image = _prepare_ocr_image(page, zoom=zoom)
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6 -l eng").strip()
    except pytesseract.TesseractNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OCR pass failed: %s", exc)
        return ""


def extract_scanned_pdf_text(document: "fitz.Document") -> str:
    """Full-document OCR pass for image-only (scanned) PDFs."""
    pages: list[str] = []
    try:
        for page_number, page in enumerate(document, start=1):
            page_text = ocr_page_text(page)
            if page_text:
                pages.append(f"--- Page {page_number}: OCR ---\n{page_text}")
    except pytesseract.TesseractNotFoundError as exc:
        raise DocumentExtractionError(
            "Tesseract OCR was not found. Install Tesseract and, on Windows, "
            "set the TESSERACT_CMD environment variable to the full path of "
            "tesseract.exe.",
            status_code=500,
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise DocumentExtractionError(f"OCR failed: {exc}", status_code=500) from exc

    return "\n\n".join(pages).strip()


# --- Public entry point -------------------------------------------------------

def extract_pdf_text(file_bytes: bytes) -> str:
    """Extract text from a PDF, falling back to OCR when needed.

    - Digital PDFs: reading-order-aware block extraction (column-safe).
    - Scanned/image-only PDFs: full-page OCR.
    - Digital PDFs whose largest-font header (name/title) got mangled by the
      primary extraction: an OCR pass recovers just that header and it is
      appended, clearly labeled, rather than replacing the primary text.
    """
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentExtractionError(f"Unable to open PDF: {exc}", status_code=400) from exc

    try:
        if document.page_count == 0:
            raise DocumentExtractionError("The PDF has no pages.", status_code=400)

        pages: list[str] = []
        first_page_text = ""
        for page_number, page in enumerate(document, start=1):
            raw_blocks = [
                block for block in page.get_text("blocks") if len(block) > 4 and str(block[4]).strip()
            ]
            ordered_blocks = _order_blocks_reading_order(raw_blocks)
            page_text = "\n".join(str(block[4]).strip() for block in ordered_blocks)
            if page_number == 1:
                first_page_text = page_text
            if page_text:
                pages.append(f"--- Page {page_number} ---\n{page_text}")

        text = "\n\n".join(pages).strip()

        if not text:
            logger.info("No embedded PDF text found; falling back to full OCR")
            text = extract_scanned_pdf_text(document)
            if not text:
                raise DocumentExtractionError(
                    "No readable text was found after PDF extraction and OCR. "
                    "The PDF may be blank, corrupted, or a scan too low-quality "
                    "for OCR.",
                    status_code=400,
                )
            return text

        if not _looks_like_header_is_present(document[0], first_page_text):
            header_ocr = ocr_page_text(document[0])
            if header_ocr:
                text = (
                    f"{text}\n\n--- OCR-RECOVERED HEADER TEXT (Page 1, the "
                    f"primary extraction above appears to be missing a "
                    f"name/title header -- this OCR pass may recover it) ---"
                    f"\n{header_ocr}"
                )

        return text
    finally:
        document.close()


def is_scanned_pdf(file_bytes: bytes) -> bool:
    """True when a PDF has no meaningful embedded text layer on any page."""
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return False
    try:
        for page in document:
            if page.get_text("text").strip():
                return False
        return True
    finally:
        document.close()


def render_pages_as_jpeg_base64(
    file_bytes: bytes, *, zoom: float = 1.4, max_dimension: int = 900, quality: int = 55
) -> list[str]:
    """Render each PDF page as a compact base64 JPEG for a vision model.

    Used only for the handwriting extraction layer. Kept deliberately small
    (thumbnailed + moderate JPEG quality) to keep vision-model prompt/context
    usage and upload size down while still keeping handwriting readable.
    """
    import base64

    encoded_images: list[str] = []
    document = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page in document:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    finally:
        document.close()
    return encoded_images
