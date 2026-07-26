"""Document-type classification.

Deliberately simple and deterministic (keyword scoring, no LLM call) so it's
fast, free, fully unit-testable, and never itself a source of "flaky" output.
The downstream parser modules are where document-specific accuracy work
happens; this just routes the document to the right one.
"""

from __future__ import annotations

import re

# Substring terms, matched anywhere in the sample. These are all
# multi-word or otherwise distinctive phrases -- generic enough to apply to
# any document of that type, but specific enough that they rarely show up
# by coincidence in an unrelated document.
_KEYWORD_SCORES: dict[str, tuple[str, ...]] = {
    "Resume": (
        "curriculum vitae", "professional summary", "career objective",
        "references available", "professional experience", "technical skills",
    ),
    # Invoice vs Receipt is the one genuinely confusable pair here, because
    # they share almost all of their surface vocabulary (merchant, line
    # items, subtotal, tax, total). The real distinction is what the
    # document is *for*: an invoice REQUESTS payment that hasn't happened
    # yet, while a receipt DOCUMENTS a payment that already completed. So
    # the terms below are deliberately restricted to that distinction --
    # request-for-payment language on one side, proof-of-payment language
    # on the other. Shared billing vocabulary like "subtotal", "total", or
    # "tax" is intentionally absent from both: it appears on essentially
    # every document of either type, so it can only add noise, never
    # separate them.
    "Invoice": (
        "invoice", "invoice number", "invoice no", "bill to", "amount due",
        "balance due", "remit to", "purchase order", "net 30", "due date",
        "payment terms", "please pay", "pay by",
    ),
    "Receipt": (
        "receipt", "cashier", "change due", "payment method", "thank you for your purchase",
        "cash tendered", "card ending", "transaction id", "sold to",
        # Proof-of-payment phrases. Deliberately *not* included here: a bare
        # "approved". It reads like card-authorization language, but
        # approval/sign-off blocks are just as common in reports, contracts
        # and forms ("Approved by: ..."), so on its own it separates
        # nothing -- it only mislabeled an 8D quality report as a Receipt.
        "amount tendered", "tendered", "auth code", "sale complete",
        "paid in full", "total paid", "approval code",
    ),
    "Contract": (
        "agreement", "whereas", "party", "parties", "terms and conditions",
        "governing law", "hereinafter referred to as", "in witness whereof",
        "effective date", "termination", "indemnif",
    ),
    "Report": (
        "executive summary", "findings", "methodology", "recommendations",
        "report", "abstract", "conclusion", "table of contents", "appendix",
    ),
    "Form": (
        "application form", "application id", "program applied for",
        "academic qualification", "emergency contact", "application status",
        "please complete", "signature of applicant", "date of birth",
        "office use only", "tick one", "please check the appropriate box",
    ),
}

# These single, generic words are common resume *section headers*, but they
# are also ordinary English words that show up constantly in unrelated
# running text and even in other documents' own section headers -- e.g. an
# academic paper's dataset description mentioning a patient's "education"
# level, or a loan/job application form that -- entirely reasonably -- has
# its own standalone "Education" section asking the applicant to list their
# schooling. Counting a single such header as a Resume signal let documents
# like that outscore the real Form/Report on the strength of one
# coincidental heading. A real resume, though, essentially always groups
# *several* of these as headers together (experience + education [+ skills
# [+ employment]]), whereas a form or paper with an incidental "Education"
# section on its own does not also have "Experience"/"Skills" as headers.
# Requiring at least two distinct header-style matches keeps this a
# structural signal about resumes specifically, not a wording rule tied to
# any one document.
_HEADER_ONLY_TERMS: dict[str, tuple[str, ...]] = {
    "Resume": ("experience", "education", "skills", "employment"),
}
_HEADER_TERM_WEIGHT = 2
_MIN_DISTINCT_HEADER_TERMS = 2


def _condense(s: str) -> str:
    """Collapse all whitespace out of a string.

    Several real resume templates render section headers with deliberate
    letter-spacing for visual style -- "E D U C A T I O N" rather than
    "EDUCATION" -- which PyMuPDF preserves as literal space characters
    between every letter. A plain equality/substring check against
    "education" would never match that. Comparing condensed forms (all
    whitespace removed on both sides) recognizes the same heading either
    way without weakening the check to a loose substring match.
    """
    return re.sub(r"\s+", "", s)


# A fillable form -- especially once run through OCR -- is dense with short
# bracketed checkbox glyphs ("[ ]", "[_]", "[J]", the OCR's best guess at a
# checkbox or tick-box) that essentially never appear in a resume, receipt,
# or the prose sections of a contract. This is a layout signal (how the
# page is laid out), not a wording one, so it still generalizes to any
# fillable form regardless of what it's collecting -- which matters because
# heavily-OCR'd forms often garble the exact label text ("Date of Birth"
# run together as "DateofBirth") enough that phrase matching alone misses
# them, while the checkbox layout survives OCR much more reliably.
# Bracketed *numeric* citation markers ("[1]", "[12]") are excluded -- an
# academic-paper-style Report is often dense with exactly those and would
# otherwise look identical to a checkbox-heavy form.
_CHECKBOX_PATTERN = re.compile(r"\[([^\]\n]{0,3})\]")
_CHECKBOX_COUNT_FOR_FULL_CREDIT = 6
_CHECKBOX_SIGNAL_WEIGHT = 3


def _checkbox_hit_count(sample_text: str) -> int:
    return sum(
        1 for match in _CHECKBOX_PATTERN.findall(sample_text) if not match.strip().isdigit()
    )


def _header_line_hits(lines: list[str], terms: tuple[str, ...]) -> set[str]:
    condensed_terms = {term: _condense(term) for term in terms}
    matched: set[str] = set()
    for line in lines:
        stripped = line.strip().rstrip(":").strip().lower()
        if len(stripped) > 40:
            continue  # a heading is short; a sentence/table row is not
        condensed_line = _condense(stripped)
        for term, condensed_term in condensed_terms.items():
            if condensed_line == condensed_term:
                matched.add(term)
    return matched


def infer_document_type(text: str, filename: str) -> str:
    """Score keyword hits per category and return the best match.

    Filename gets included in the sample since users often name files in a
    way that telegraphs type (e.g. "invoice_2024_03.pdf"), which is a free,
    reliable signal alongside document body content.
    """
    sample_text = text[:5000]
    sample = f"{filename}\n{sample_text}".lower()
    sample_lines = sample_text.lower().splitlines()

    matched_terms = {
        doc_type: [term for term in terms if term in sample]
        for doc_type, terms in _KEYWORD_SCORES.items()
    }
    scores = {doc_type: float(len(hits)) for doc_type, hits in matched_terms.items()}

    for doc_type, header_terms in _HEADER_ONLY_TERMS.items():
        matched = _header_line_hits(sample_lines, header_terms)
        if len(matched) >= _MIN_DISTINCT_HEADER_TERMS:
            scores[doc_type] = scores.get(doc_type, 0) + len(matched) * _HEADER_TERM_WEIGHT

    checkbox_hits = _checkbox_hit_count(sample_text)
    if checkbox_hits:
        credit = min(checkbox_hits, _CHECKBOX_COUNT_FOR_FULL_CREDIT) / _CHECKBOX_COUNT_FOR_FULL_CREDIT
        scores["Form"] = scores.get("Form", 0) + credit * _CHECKBOX_SIGNAL_WEIGHT

    # Tie-break on the most specific evidence rather than on dict insertion
    # order. Plain `max()` over the score dict silently resolves ties by
    # whichever type happens to be declared first, which is not a signal at
    # all -- that's exactly how an electronics-store receipt titled "SALES
    # INVOICE" got labeled Invoice off a 2-2 tie. When two types score
    # equally, prefer the one whose longest matched term is longer: a hit
    # on a long, distinctive phrase ("thank you for your purchase") is much
    # stronger evidence than a hit on a short common word ("party"), since
    # long phrases are far less likely to appear by coincidence. The final
    # `doc_type` key keeps the result stable and reproducible if even that
    # ties.
    def _rank(item: tuple[str, float]) -> tuple[float, int, str]:
        doc_type, score = item
        longest_match = max((len(term) for term in matched_terms.get(doc_type, [])), default=0)
        return (score, longest_match, doc_type)

    best_type, best_score = max(scores.items(), key=_rank)
    return best_type if best_score > 0 else "Unknown"
