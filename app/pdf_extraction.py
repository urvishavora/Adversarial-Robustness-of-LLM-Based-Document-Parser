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
import statistics

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


def _cut_groups(blocks: list[tuple], axis: int, min_gap: float) -> list[list[tuple]]:
    """Split blocks into groups separated by a clean gap along one axis.

    A gap only counts when *no* block spans it: walking the blocks in order
    of their leading edge, a new group starts whenever the next block
    begins at least ``min_gap`` past the furthest trailing edge seen so
    far. That "furthest trailing edge" bookkeeping is what makes the cut
    safe -- a wide block straddling the gap keeps everything in one group,
    which is exactly the desired behaviour for a full-width heading sitting
    above two columns.
    """
    start_index, end_index = (0, 2) if axis == 0 else (1, 3)
    ordered = sorted(blocks, key=lambda block: block[start_index])

    groups: list[list[tuple]] = [[ordered[0]]]
    running_edge = ordered[0][end_index]
    for block in ordered[1:]:
        if block[start_index] - running_edge >= min_gap:
            groups.append([block])
            running_edge = block[end_index]
        else:
            groups[-1].append(block)
            running_edge = max(running_edge, block[end_index])
    return groups


# A column gap is a large fraction of page width; a section break is a much
# smaller fraction of page height (it only has to exceed normal line/paragraph
# spacing, which is why the row threshold is the tighter of the two).
_COLUMN_GAP_RATIO = 0.05
_ROW_GAP_RATIO = 0.02
_MAX_CUT_DEPTH = 8


def _groups_are_row_aligned(groups: list[list[tuple]]) -> bool:
    """True when a vertical split would cut a table apart row-wise.

    A wide x-gap does not always mean "independent columns". A summary
    table puts its labels in one x-band and its amounts in another:

        Subtotal            $62.80
        HST 13%              $2.34
        Total CAD           $65.14

    Those are two x-bands separated by a clean gap, so a naive column split
    fires and emits every label followed by every amount. The label/amount
    pairing then survives only as position, and one slip misreads the
    subtotal as the line above it -- which is exactly what happened.

    The distinguishing signal is row alignment. In a table, each band has
    the same number of rows and they sit at the same y positions. In a
    genuine multi-column layout the columns flow independently, so their
    rows do not line up one-for-one. When the rows do align, the vertical
    split is refused and the caller falls back to reading in row order,
    which keeps each label next to its own value.
    """
    if len(groups) < 2:
        return False
    counts = [len(group) for group in groups]
    if min(counts) < 2 or max(counts) - min(counts) > 1:
        return False

    def centers(group: list[tuple]) -> list[float]:
        return sorted((block[1] + block[3]) / 2 for block in group)

    heights = [block[3] - block[1] for group in groups for block in group]
    tolerance = max(statistics.median(heights) * 0.6, 1.0) if heights else 1.0

    reference = centers(max(groups, key=len))
    for group in groups:
        if group is max(groups, key=len):
            continue
        group_centers = centers(group)
        aligned = sum(
            1 for c in group_centers if any(abs(c - r) <= tolerance for r in reference)
        )
        if aligned / len(group_centers) < 0.8:
            return False
    return True


def _xy_cut(blocks: list[tuple], page_width: float, page_height: float, depth: int) -> list[tuple]:
    """Recursive XY-cut: recover reading order for mixed-layout pages.

    The previous approach tried to classify the *whole page* as either
    one-column or two-column. Real documents aren't that uniform: a resume
    routinely runs full width at the top (contact line, summary, experience)
    and then splits into two columns lower down (education | activities).
    Forcing one global decision gets that page wrong either way -- a
    whole-page column split mangles the full-width part, and no split at all
    interleaves the two-column part line by line, so "Bachelor of Business
    Administration" ends up followed by "President, Business Club" from the
    neighbouring column.

    XY-cut handles this by deciding locally instead of globally. At each
    step it looks for a clean vertical gap (a column boundary) and, failing
    that, a clean horizontal gap (a section break), then recurses into each
    resulting group. Columns are tried first because a vertical split is the
    stronger structural claim: it only succeeds when genuinely nothing spans
    the gap, whereas almost any page can be sliced into horizontal bands.
    Full-width content therefore blocks a bogus column split automatically,
    and a region that really is two columns gets read down one column before
    the other.
    """
    if len(blocks) <= 1 or depth >= _MAX_CUT_DEPTH:
        return sorted(blocks, key=lambda block: (block[1], block[0]))

    column_groups = _cut_groups(blocks, axis=0, min_gap=page_width * _COLUMN_GAP_RATIO)
    if len(column_groups) > 1 and _groups_are_row_aligned(column_groups):
        # A table, not independent columns -- read it in row order so each
        # label stays next to its own value.
        return sorted(blocks, key=lambda block: (block[1], block[0]))
    if len(column_groups) > 1:
        ordered: list[tuple] = []
        for group in sorted(column_groups, key=lambda g: min(b[0] for b in g)):
            ordered.extend(_xy_cut(group, page_width, page_height, depth + 1))
        return ordered

    row_groups = _cut_groups(blocks, axis=1, min_gap=page_height * _ROW_GAP_RATIO)
    if len(row_groups) > 1:
        ordered = []
        for group in sorted(row_groups, key=lambda g: min(b[1] for b in g)):
            ordered.extend(_xy_cut(group, page_width, page_height, depth + 1))
        return ordered

    return sorted(blocks, key=lambda block: (block[1], block[0]))


def _extract_layout_blocks(page: "fitz.Page") -> list[tuple]:
    """Build layout units from a page, splitting blocks that span columns.

    PyMuPDF sometimes emits a single block containing lines from two
    different columns -- two side-by-side section headers on the same row
    ("EDUCATION & CERTIFICATIONS" and "EXTRACURRICULAR ACTIVITIES") come
    back as one block, as do the first rows beneath them. No amount of
    block *reordering* can fix that, because the columns are already fused
    inside one unit. So any block whose own lines separate cleanly along x
    is split into one unit per column before ordering happens.
    """
    try:
        data = page.get_text("dict")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read page layout: %s", exc)
        return [block for block in page.get_text("blocks") if len(block) > 4 and str(block[4]).strip()]

    page_width = max(float(page.rect.width), 1.0)
    units: list[tuple] = []

    for block in data.get("blocks", []):
        lines = block.get("lines")
        if not lines:
            continue

        line_items: list[tuple] = []
        for line in lines:
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if text.strip():
                bbox = line["bbox"]
                line_items.append((bbox[0], bbox[1], bbox[2], bbox[3], text.strip()))
        if not line_items:
            continue

        groups = _cut_groups(line_items, axis=0, min_gap=page_width * _COLUMN_GAP_RATIO)
        for group in groups:
            group_sorted = sorted(group, key=lambda item: (item[1], item[0]))
            units.append(
                (
                    min(item[0] for item in group),
                    min(item[1] for item in group),
                    max(item[2] for item in group),
                    max(item[3] for item in group),
                    "\n".join(item[4] for item in group_sorted),
                )
            )

    return units


_URL_TAIL_PATTERN = re.compile(r"(https?://|www\.)\S*$", re.IGNORECASE)


def _merge_wrapped_url_blocks(blocks: list[tuple]) -> list[str]:
    """Rejoin a URL that a narrow column split across separate blocks.

    A long URL in a narrow sidebar wraps mid-token, and PyMuPDF reports
    each visual line as its own block:

        'https://www.linkedin.com/in/s'
        'ebastian-bennett?'

    Emitted as two lines, a downstream consumer has no way to know they
    were one token: the model reading this text reassembles it as
    ``https://www.linkedin.com/in/s/ebastian-bennett?`` -- inserting a
    separator that was never in the document and producing a URL that
    doesn't resolve.

    Joining is deliberately narrow, so ordinary text is never glued
    together. All must hold: the first block ends mid-URL, the next block
    contains no whitespace at all (a wrapped URL fragment is a single
    token, whereas ordinary following text almost always has spaces), the
    two share a left edge (same column), and they are vertically adjacent
    (within ~1.5 line heights). Chained continuations are supported for a
    URL wrapped across three or more lines.
    """
    texts: list[str] = []
    consumed: set[int] = set()

    for index, block in enumerate(blocks):
        if index in consumed:
            continue

        text = str(block[4]).strip()
        current = block
        next_index = index + 1

        while next_index < len(blocks) and _URL_TAIL_PATTERN.search(text):
            candidate = blocks[next_index]
            candidate_text = str(candidate[4]).strip()

            if not candidate_text or re.search(r"\s", candidate_text):
                break
            if abs(candidate[0] - current[0]) > 2.0:
                break

            line_height = max(current[3] - current[1], 1.0)
            vertical_gap = candidate[1] - current[3]
            if vertical_gap < -1.0 or vertical_gap > line_height * 1.5:
                break

            text += candidate_text
            consumed.add(next_index)
            current = candidate
            next_index += 1

        texts.append(text)

    return texts


def _order_blocks_reading_order(blocks: list[tuple]) -> list[tuple]:
    if not blocks:
        return []

    page_width = max(max(b[2] for b in blocks) - min(b[0] for b in blocks), 1.0)
    page_height = max(max(b[3] for b in blocks) - min(b[1] for b in blocks), 1.0)
    return _xy_cut(blocks, page_width, page_height, 0)


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
            raw_blocks = _extract_layout_blocks(page)
            ordered_blocks = _order_blocks_reading_order(raw_blocks)
            page_text = "\n".join(_merge_wrapped_url_blocks(ordered_blocks))
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
