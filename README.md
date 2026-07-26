# LLM PDF Parser

Dynamic, schema-agnostic PDF extraction backed by a local Ollama model, with
a separate parser module per document type: resume, invoice, receipt,
contract, report, and application form (printed and handwritten).

## What changed from the old `main_*.py` files

The project used to be five overlapping monolithic scripts
(`main.py`, `main_old.py`, `main_resume.py`, `main_updated_ocr2.py`,
`main_hand_ocr.py`). They're preserved for reference under `legacy/`, but are
no longer the entry point. Notably, **`main_hand_ocr.py` had a real bug**:
a mis-indented `try/except` in its confidence-parsing code raised a hard
`IndentationError` on import, so that file could never actually run. You can
confirm this yourself:

```bash
python3 -m py_compile legacy/main_hand_ocr.py   # -> IndentationError, line 1054
```

Everything now lives under `app/`, split by responsibility:

```
app/
  config.py              env-var configuration (Ollama URLs/models, limits)
  errors.py              shared exception types
  schemas.py             output schema templates (EMPTY_RESUME, EMPTY_GENERIC, ...)
  json_utils.py          JSON repair + generic value normalization (unit-tested, no LLM needed)
  pdf_extraction.py       text extraction, column detection, OCR fallback, page-to-image rendering
  classification.py      keyword-based document-type routing (no LLM call)
  regex_utils.py          shared deterministic date/amount/labeled-value extraction
  llm_client.py            all Ollama HTTP calls (text + vision), retries, health check
  document_service.py      classify -> dispatch to the right parser module
  main.py                  FastAPI routes only
  parsers/
    resume.py              prompt + normalization + validate/repair pass
    generic.py             shared LLM-pass + normalize helper for the 5 non-resume types
    invoice.py              + deterministic invoice-number/dates/totals extraction
    receipt.py               + deterministic receipt-number/totals/payment-method extraction
    contract.py              + deterministic effective-date/governing-law extraction
    report.py                + deterministic report-date/author extraction
    application_form.py      + deterministic application-ID/dates/declaration extraction
    handwriting.py            vision-model handwritten field extraction (the fixed module)
tests/                       unit + integration tests (see "Testing" below)
scripts/
  parse_pdf.py                parse any single PDF from the command line
  score_resumes.py             accuracy report for resumes/ vs ground_truth_resumes.json
```

Nothing in `app/` is hardcoded to any specific sample file. `fields` on every
non-resume document is built dynamically by the model from whatever labels
actually appear in that document; the regex "enrichment" layer only matches
generic, universal labels (e.g. "invoice number", "total", "governing law",
"date of birth") that apply across arbitrary documents of that type, not
anything specific to the files this was tested against.

## Setup

### 1. Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt   # includes requirements.txt + pytest/reportlab
```

### 2. Ollama (the LLM backend, runs entirely on your machine, free)

Install from https://ollama.com, then pull the models this project uses:

```bash
ollama pull llama3.1            # text extraction (resumes, invoices, receipts, contracts, reports, forms)
ollama pull llama3.2-vision     # only needed for handwritten form fields
ollama serve                    # start the server (usually already running as a service)
```

`llama3.1` is an 8B model -- accurate but needs a reasonably capable machine
(8GB+ RAM free). If your hardware struggles, `MODEL_NAME=llama3.2:3b` is a
much lighter, faster alternative at some accuracy cost (set it in `.env`).

### 3. Tesseract OCR (for scanned/image-only pages)

- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki, or set
  `TESSERACT_CMD` in `.env` if it's not on PATH.
- macOS: `brew install tesseract`
- Linux: `apt install tesseract-ocr`

### 4. Configuration

Copy `.env.example` to `.env` and adjust if needed -- every setting has a
sensible default and the app runs with zero configuration against a
default local Ollama install.

## Running the server

```bash
uvicorn app.main:app --reload
```

Then check `http://localhost:8000/health` -- it reports whether Ollama is
reachable and whether the configured text/vision models are actually
installed, which is the fastest way to catch a setup problem before
uploading anything. Interactive API docs are at `http://localhost:8000/docs`.

Upload a PDF:

```bash
curl -F "file=@resumes/daniel_gallego_ux_designer.pdf" http://localhost:8000/upload
```

Or skip the HTTP server entirely and parse a file directly:

```bash
python scripts/parse_pdf.py resumes/daniel_gallego_ux_designer.pdf
```

## Testing

```bash
pytest                          # everything that doesn't need a live Ollama (69 tests)
```

These cover PDF text/OCR extraction, document classification, JSON
repair/normalization, every parser module's regex-enrichment logic against
synthetic invoice/receipt/contract/report/form PDFs, the handwriting
mapping/merge logic (including a regression test for the old
`IndentationError`), and full FastAPI request/response cycles for all six
document types with the LLM call mocked out.

Two more tests need your local Ollama and are skipped automatically if it
isn't reachable:

```bash
pytest tests/test_accuracy_resumes_live.py -v -s   # per-field accuracy vs ground_truth_resumes.json
```

or, for a plainer report:

```bash
python scripts/score_resumes.py
```

This prints, per resume, every field that scored under 100% against
`ground_truth_resumes.json`, plus the overall average -- this is how you
reproduce and improve on the old "92.5%" figure, and see exactly which
fields are still wrong rather than just one number. The scoring rule itself
is documented in `app/accuracy.py`; there's no single universally-agreed way
to grade this, so treat it as this project's own metric, not an absolute
truth.

## Adding real samples for invoice/receipt/contract/report/application forms

Only resumes shipped with real samples + ground truth in this project.
`tests/fixtures/synthetic_pdfs.py` generates simple stand-in PDFs for the
other five types so their pipelines are exercised by the test suite, but
those aren't real-world documents. To validate and tune accuracy on your
own files:

```bash
python scripts/parse_pdf.py path/to/your/real_invoice.pdf
```

Inspect the output, and if a field the document clearly shows didn't come
through, that's the signal to tighten the relevant parser's prompt guidance
or add a label pattern to its regex enrichment (e.g. `app/parsers/invoice.py`
-> `_INVOICE_NUMBER_LABELS`), not something to hardcode per file.

## Honest limitations

- **"100% on every document" isn't a guarantee this architecture -- or any
  LLM-based one -- can make.** What this project does provide: a
  deterministic regex layer that gets identifiers/dates/totals exactly
  right whenever a recognizable label is present (no hallucination risk for
  what it catches), a validate+repair pass for resumes that catches and
  fixes the specific failure modes seen before (missing sections, section
  leakage), and a full test suite proving every code path runs without
  error. The remaining gap between that and "100%" is genuine LLM extraction
  quality on free-form content, which depends on the model you run and the
  document's own legibility/layout.
- **Handwriting accuracy depends heavily on the vision model and the
  handwriting's own legibility.** `HANDWRITING_MIN_CONFIDENCE` controls when
  a result is flagged `review_required` instead of trusted outright.
- This sandbox could not run Ollama itself (no internet access to Hugging
  Face/Ollama's registry, and 2 CPU/3.8GB RAM is too small for an 8B model
  regardless). Everything not dependent on a live model was tested for
  real here; the LLM-dependent paths need to be run against your own Ollama
  to get real accuracy numbers.
- **Real numeric accuracy scores only exist for resumes**, because that's
  the only type with a ground-truth file (`ground_truth_resumes.json`).
  Accuracy is "does the output match a known-correct answer" -- for
  invoices/receipts/contracts/reports/forms there's no equivalent
  known-correct reference shipped with this project, so all that can
  honestly be checked for those types is: did it error, was the document
  type classified correctly, and do the extracted fields look complete and
  plausible against the source PDF. If you want real scores for those
  types too, the same pattern as `ground_truth_resumes.json` /
  `scripts/score_resumes.py` can be extended once you have (or hand-label)
  a few correct-answer files for them.
- **Classification fix (2026-07-26):** `app/classification.py` originally
  counted a few single generic words ("experience", "education", "skills")
  as a Resume signal anywhere they appeared in the text. That
  misclassified some academic-paper-style reports as Resume, because a
  paper can mention "patient education" or "clinical experience" inline
  without being a resume. Fixed by only counting those words when they
  appear as their own short heading line (the way an actual resume uses
  them), not buried in a sentence -- a structural fix, not a per-file one.
  Covered by `tests/test_classification.py::test_academic_report_not_misclassified_as_resume`.
- **Follow-up classification fixes (same day), found by testing against 15
  real report/contract/form PDFs and all 10 real resumes directly (no LLM
  needed -- classification is deterministic):**
  - Several real resume templates render section headers with deliberate
    letter-spacing ("E D U C A T I O N"), which the header-only check above
    didn't recognize as "education". Fixed by comparing whitespace-collapsed
    forms on both sides instead of exact strings.
  - The header check alone was too loose: a single "Education" heading on
    an otherwise-unrelated document (a loan/job application form legitimately
    asking about schooling) was enough to tip it to Resume. Fixed by
    requiring at least two distinct resume-style headers together
    (experience/education/skills/employment), which is how real resumes
    actually use them and forms don't.
  - "work experience" and "employment history" were removed from the plain
    substring keyword list -- real bank/loan forms use those exact phrases
    as field labels too (e.g. "Total Work Experience Years"), so they
    weren't resume-exclusive.
  - Added a checkbox-density signal for Form: heavily-OCR'd scanned forms
    are dense with short bracketed checkbox glyphs (`[ ]`, `[_]`, `[J]`)
    that survive OCR far more reliably than exact label text does, which
    matters when OCR garbles labels like "Date of Birth" into
    "DateofBirth". Numeric bracket citations (`[1]`, `[12]`) are excluded
    from this count so a citation-heavy academic Report isn't mistaken for
    a checkbox-heavy Form.
  - All 10 real sample resumes and 15 real report/contract/form PDFs the
    user supplied now classify correctly; covered by 5 new tests in
    `tests/test_classification.py`.
- **Reading order rewritten as a recursive XY-cut.** The old logic had to
  classify a whole page as either one-column or two-column. Real pages
  aren't uniform: `henrietta_mitchell_...pdf` runs full width at the top
  (contact, summary, experience) and splits into two columns at the bottom
  ("EDUCATION & CERTIFICATIONS" | "EXTRACURRICULAR ACTIVITIES"). One
  global decision gets that page wrong either way, and it was being read
  by plain y-order, interleaving the two columns row by row -- "Bachelor of
  Business Administration" immediately followed by "President, Business
  Club" from the *other* column, which then got attributed to the degree.
  `_xy_cut` now decides locally: at each step it looks for a clean vertical
  gap (column boundary), falling back to a clean horizontal gap (section
  break), and recurses. Columns are tried first because a vertical split
  only succeeds when genuinely nothing spans the gap, so full-width content
  automatically prevents a bogus column split.
  - PyMuPDF also sometimes emits a *single block* containing lines from two
    different columns, which no amount of reordering can fix because the
    columns are already fused inside one unit. `_extract_layout_blocks`
    now splits any block whose own lines separate cleanly along x.
  - Verified against the rendered page image, not just the text dump.
    Regression test: `test_two_column_section_is_not_interleaved`.
- **Wrapped URLs are rejoined.** A long URL in a narrow column wraps
  mid-token and PyMuPDF reports each visual line as its own block
  (`https://www.linkedin.com/in/s` + `ebastian-bennett?`). Emitted as two
  lines, the model reassembled them with a separator that was never in the
  document, producing `.../in/s/ebastian-bennett?` -- a URL that doesn't
  resolve. Joining is deliberately narrow (same left edge, vertically
  adjacent, continuation contains no whitespace) so ordinary text is never
  glued together. Affected Sebastian Bennett and Lorna Alvarado.
- **Cross-field validation added.** The existing checks only fired when an
  entire section went missing. These catch the quieter failure where a
  section is present but an entry is missing a field the document plainly
  shows: experience entries with no employer, education entries with no
  institution or degree, and contact URLs/emails that don't appear
  anywhere in the source document. That last check is what catches a
  mangled URL -- `.../in/s/ebastian-bennett?` is perfectly well-formed, so
  no structural check can flag it, but it isn't in the document.
- **Receipt-vs-Invoice fix.** `clean_receipt_05_electronics_store.pdf` --
  a store receipt whose header reads "SALES INVOICE" -- was classified
  Invoice. Root cause was two separate flaws:
  - `subtotal` was scored as an *Invoice* signal, but it appears on 10/10
    of the real sample receipts too. Shared billing vocabulary can't
    separate the two types, so it only ever added noise. The Invoice and
    Receipt keyword lists are now restricted to what actually
    distinguishes them: an invoice *requests* payment not yet made
    ("amount due", "due date", "net 30", "remit to"), a receipt
    *documents* payment already completed ("tendered", "auth code",
    "paid in full", "thank you for your purchase").
  - The two types then tied 2-2, and `max()` silently resolved the tie by
    dict insertion order -- Invoice simply happened to be declared first.
    Ties now break on the most specific evidence (longest matched phrase),
    since a hit on a long distinctive phrase is much stronger evidence
    than a hit on a short common word, with the type name as a final
    stable tiebreak.
  - Note: a bare `approved` was briefly added as a Receipt signal and
    immediately reverted -- it mislabeled the 8D quality report as a
    Receipt, because approval/sign-off blocks are just as common in
    reports, contracts, and forms. Covered by
    `test_report_with_approval_block_not_misclassified_as_receipt`.
  - All 10 real receipts now classify correctly, and a real invoice still
    classifies as Invoice. The 10 receipt sample PDFs and
    `ground_truth_receipts.json` are now bundled in `receipts/` so
    `test_all_real_receipts_classify_as_receipt` checks every one of them
    rather than a single file.
