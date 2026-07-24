"""Document-type classification.

Deliberately simple and deterministic (keyword scoring, no LLM call) so it's
fast, free, fully unit-testable, and never itself a source of "flaky" output.
The downstream parser modules are where document-specific accuracy work
happens; this just routes the document to the right one.
"""

from __future__ import annotations

_KEYWORD_SCORES: dict[str, tuple[str, ...]] = {
    "Resume": (
        "experience", "education", "skills", "employment", "curriculum vitae",
        "professional summary", "objective", "references available",
    ),
    "Invoice": (
        "invoice", "invoice number", "invoice no", "bill to", "amount due",
        "subtotal", "remit to", "purchase order", "net 30", "due date",
    ),
    "Receipt": (
        "receipt", "cashier", "change due", "payment method", "thank you for your purchase",
        "cash tendered", "card ending", "transaction id", "sold to",
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


def infer_document_type(text: str, filename: str) -> str:
    """Score keyword hits per category and return the best match.

    Filename gets included in the sample since users often name files in a
    way that telegraphs type (e.g. "invoice_2024_03.pdf"), which is a free,
    reliable signal alongside document body content.
    """
    sample = f"{filename}\n{text[:5000]}".lower()
    scores = {
        doc_type: sum(term in sample for term in terms)
        for doc_type, terms in _KEYWORD_SCORES.items()
    }
    best_type, best_score = max(scores.items(), key=lambda pair: pair[1])
    return best_type if best_score > 0 else "Unknown"
