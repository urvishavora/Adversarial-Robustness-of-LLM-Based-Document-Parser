"""Deterministic segmentation of prose documents into their sections.

Shared by contracts (clauses) and reports (sections). Section structure
is a layout property -- numbering and heading lines -- so it can be
recovered without the model. That matters because the model reliably
returns section headings and drops the text beneath them: measured
content recall against full-text ground truth was ~18% for contracts
and 3.5% for reports before this.
"""

from __future__ import annotations

import re
from typing import Any


# --- Deterministic clause segmentation -------------------------------------
#
# A contract's substance lives in its clauses, but the model reliably returns
# only the handful of named fields (effective_date, governing_law, ...) and
# leaves the body behind -- measured content recall against full-text ground
# truth was ~18%. Clause structure is a layout property, not a semantic one,
# so it can be recovered deterministically and does not depend on the model
# noticing anything.
#
# Two shapes have to work, because real agreements use both:
#   - numbered:   "1. This Car Rental Agreement is made...", "3.2 ...",
#                 "ARTICLE IV", "Section 5"
#   - unnumbered: a short heading line ("Scope of Use") above its paragraphs,
#                 or pure prose with no headings at all (many NDAs)
_CLAUSE_NUMBER_RE = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d+(?:\.\d+)*)[.)]\s+"          # 1.  2)  3.1
    r"|(?P<article>ARTICLE\s+[IVXLC\d]+)\b"   # ARTICLE IV
    r"|(?P<section>SECTION\s+\d+(?:\.\d+)*)\b"  # SECTION 5
    r")",
    re.IGNORECASE,
)
_PAGE_MARKER_RE = re.compile(r"^---\s*Page\s+\d+.*?---$", re.IGNORECASE)


def _looks_like_heading(line: str) -> bool:
    """A short standalone line introducing the paragraphs beneath it."""
    stripped = line.strip()
    if not (3 <= len(stripped) <= 60):
        return False
    if stripped.endswith((".", ";", ",", ":")) and not stripped.endswith(":"):
        return False
    if _CLAUSE_NUMBER_RE.match(stripped):
        return False
    words = stripped.split()
    if len(words) > 8:
        return False
    # Title Case or ALL CAPS, which is how headings are set in practice.
    letters = [w for w in words if w[:1].isalpha()]
    if not letters:
        return False
    return all(w[:1].isupper() for w in letters)


def extract_clauses(text: str) -> list[dict[str, Any]]:
    """Segment an agreement into {clause_number, heading, text} records.

    Falls back progressively: numbered markers first, then heading lines,
    then paragraph blocks -- so a contract with no numbering at all (common
    for NDAs) still yields its body rather than nothing.
    """
    lines = [ln for ln in text.splitlines() if not _PAGE_MARKER_RE.match(ln.strip())]

    clauses: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_heading: str | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        body = " ".join(current["_body"]).strip()
        if body or current["heading"]:
            clauses.append(
                {
                    "clause_number": current["clause_number"],
                    "heading": current["heading"],
                    "text": body,
                }
            )
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        match = _CLAUSE_NUMBER_RE.match(line)
        if match:
            flush()
            number = match.group("num") or match.group("article") or match.group("section")
            current = {
                "clause_number": number,
                "heading": pending_heading,
                "_body": [line[match.end():].strip()],
            }
            pending_heading = None
            continue

        if _looks_like_heading(line):
            # A heading closes the previous clause and labels the next one.
            flush()
            pending_heading = line.rstrip(":")
            continue

        if current is None:
            current = {"clause_number": None, "heading": pending_heading, "_body": []}
            pending_heading = None
        current["_body"].append(line)

    flush()

    # Drop fragments that carry no real text -- a heading with nothing under
    # it is not a clause.
    return [c for c in clauses if len((c["text"] or "").split()) >= 3]


