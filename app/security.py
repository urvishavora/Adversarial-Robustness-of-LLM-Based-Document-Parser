"""Deterministic security scanning for uploaded PDFs.

Two different threats need two different responses, so they are graded
separately rather than lumped into one "malicious" flag:

1. Active content -- embedded JavaScript, auto-run actions, launch actions,
   file attachments. This is a weaponised *file*. No legitimate invoice
   ships JavaScript, so these are `critical` and the caller should refuse
   the document outright.

2. Hidden text -- white-on-white, fully transparent, off-page, or
   microscopic text. The *file* may be fine and the visible document
   genuinely useful; what's dangerous is that the hidden span reaches the
   language model, where "IGNORE ALL PREVIOUS INSTRUCTIONS..." is read as
   an instruction. The response is to quarantine those spans out of the
   text before the prompt is built, then parse the visible document
   normally and report what was removed.

Everything here is deterministic -- span colour, alpha, size and geometry
straight from the PDF, plus a byte scan for action keywords. No model call,
so it cannot itself be manipulated by document content, and it adds no
inference cost.

The most important false positive to avoid is legitimate white text. Resume
and report templates routinely print white headings on a dark sidebar or
banner. Text is therefore only treated as hidden when nothing is painted
behind it -- a white span sitting on a dark filled shape is design, not an
attack.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import fitz

logger = logging.getLogger("document_parser")

# Ordered least to most severe so the overall grade is a max().
SEVERITY_ORDER = ("none", "info", "low", "medium", "high", "critical")


def _worst(*severities: str) -> str:
    return max(severities, key=lambda s: SEVERITY_ORDER.index(s) if s in SEVERITY_ORDER else 0)


# --- Active content ----------------------------------------------------------
#
# Checked structurally, against the document catalog and page objects, NOT by
# scanning raw bytes. A byte scan was tried first and was unusable: "/JS" and
# "/AA" are two characters and collide constantly inside compressed streams,
# which flagged all ten sample resumes as containing active content and one
# as critical. A detector that cries wolf on every real document is worse
# than none, because it trains the user to ignore it.
#
# Keys that indicate executable or auto-triggered behaviour, mapped to how
# severely they should be graded.
_CATALOG_ACTION_KEYS: tuple[tuple[str, str, str, str], ...] = (
    ("OpenAction", "auto_action", "critical", "Action that runs automatically when the file opens"),
    ("AA", "auto_action", "high", "Event-triggered additional action"),
)
# Action subtypes that actually execute something or reach the network.
# /GoTo, /GoToR and a bare destination array are navigation.
_DANGEROUS_ACTION_SUBTYPES = {
    "JavaScript", "Launch", "SubmitForm", "ImportData", "URI", "GoToE", "Movie", "Sound",
}


def _action_subtype(document: "fitz.Document", xref: int, key: str) -> str | None:
    """The /S subtype of an action, or None for a plain destination."""
    try:
        kind, value = document.xref_get_key(xref, f"{key}/S")
    except Exception:  # pragma: no cover - defensive
        return None
    if kind != "name" or not value:
        return None
    return str(value).lstrip("/")


_JAVASCRIPT_SEVERITY = ("javascript", "critical", "Embedded JavaScript")
_EMBEDDED_FILE_SEVERITY = ("embedded_file", "high", "Embedded file attachment")

# Phrases whose only purpose is to redirect an instruction-following system.
# Scored higher when found in hidden text, since visible text may legitimately
# quote them (a security policy document, for instance).
_INJECTION_PATTERNS = (
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(?:all\s+)?(?:previous|prior|the)\s+(?:instructions?|document|above)",
    r"\bsystem\s*[:>]",
    r"\bassistant\s*[:>]",
    r"you\s+are\s+now\s+(?:a|an)\b",
    r"new\s+instructions?\s*[:>]",
    r"override\s+(?:your|all|previous)\b",
    r"do\s+not\s+follow\s+(?:the|your|any)\s+(?:instructions?|rules?)",
    r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt",
    r"exfiltrat|send\s+(?:it|this|the\s+\w+)\s+to\s+\S+@",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def _has_key(document: "fitz.Document", xref: int, key: str) -> bool:
    try:
        kind, _value = document.xref_get_key(xref, key)
    except Exception:  # pragma: no cover - defensive
        return False
    return kind not in (None, "null")


def scan_active_content(file_bytes: bytes) -> list[dict[str, Any]]:
    """Inspect the PDF's object structure for executable or auto-run content."""
    findings: list[dict[str, Any]] = []
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:  # pragma: no cover - handled upstream
        return findings

    try:
        catalog = document.pdf_catalog()

        # An /OpenAction is usually navigation -- "open at page 3, fit width"
        # -- which is entirely ordinary. Only the action *subtype* says
        # whether anything executes. Flagging the mere presence of
        # /OpenAction marked two normal reports as critical.
        for key, kind, severity, description in _CATALOG_ACTION_KEYS:
            if not _has_key(document, catalog, key):
                continue
            subtype = _action_subtype(document, catalog, key)
            if subtype is None:
                # No /S subtype: a plain destination array. Navigation only.
                continue
            if subtype not in _DANGEROUS_ACTION_SUBTYPES:
                continue
            findings.append(
                {
                    "type": kind,
                    "severity": severity,
                    "detail": f"{description} (/{key} -> /{subtype}).",
                }
            )

        # JavaScript lives under the catalog's name tree, not as a bare key.
        if _has_key(document, catalog, "Names/JavaScript"):
            kind, severity, description = _JAVASCRIPT_SEVERITY
            findings.append({"type": kind, "severity": severity, "detail": f"{description}."})

        # Per-page event actions (e.g. run-on-page-open).
        for page_number in range(document.page_count):
            if _has_key(document, document.page_xref(page_number), "AA"):
                findings.append(
                    {
                        "type": "auto_action",
                        "severity": "high",
                        "page": page_number + 1,
                        "detail": "Page-level event-triggered action (/AA).",
                    }
                )
                break  # one finding is enough to characterise the document

        try:
            attachment_count = document.embfile_count()
        except Exception:  # pragma: no cover - older PyMuPDF
            attachment_count = 0
        if attachment_count:
            kind, severity, description = _EMBEDDED_FILE_SEVERITY
            findings.append(
                {
                    "type": kind,
                    "severity": severity,
                    "detail": f"{description}: {attachment_count} file(s) embedded.",
                }
            )
    finally:
        document.close()

    return findings


# --- Hidden text -------------------------------------------------------------

_MIN_HIDDEN_TEXT_CHARS = 8


def _is_meaningful(text: str) -> bool:
    """Does this span carry enough real content to be a payload?"""
    stripped = "".join(ch for ch in text if ch.isprintable() and not ch.isspace())
    if len(stripped) < _MIN_HIDDEN_TEXT_CHARS:
        return False
    # Decorative rules and separators are all one repeated punctuation mark.
    if not any(ch.isalnum() for ch in stripped):
        return False
    return True


_WHITE = 0xFFFFFF
_MIN_LEGIBLE_FONT_PT = 3.0
# Luminance below this counts as a "dark" backdrop that makes white text real.
_DARK_FILL_LUMINANCE = 0.6


def _luminance(fill: Any) -> float:
    """Perceived brightness of a PyMuPDF fill colour (0=black, 1=white)."""
    if not isinstance(fill, (tuple, list)) or len(fill) < 3:
        return 1.0
    r, g, b = (float(c) for c in fill[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _dark_backdrops(page: "fitz.Page") -> list["fitz.Rect"]:
    """Regions that can make light text legible: dark vector fills, and any
    image.

    Images matter as much as fills. A report cover with white headings over a
    photograph has no dark *drawing* behind the text, only a picture -- and
    checking fills alone reported those headings as concealed payloads.
    Image content is not inspected pixel by pixel; its presence behind the
    text is enough to make "invisible" an unsafe conclusion.
    """
    rects: list[fitz.Rect] = []
    try:
        for drawing in page.get_drawings():
            fill = drawing.get("fill")
            if fill is not None and _luminance(fill) <= _DARK_FILL_LUMINANCE:
                rects.append(fitz.Rect(drawing["rect"]))
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        for image in page.get_images(full=True):
            rects.extend(page.get_image_rects(image[0]))
    except Exception:  # pragma: no cover - defensive
        pass
    return rects


def _is_light(color_int: Any) -> bool:
    try:
        value = int(color_int)
    except (TypeError, ValueError):
        return False
    r, g, b = (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF
    return _luminance((r / 255, g / 255, b / 255)) >= 0.85


def scan_hidden_text(file_bytes: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """Find text a reader cannot see but an extractor still reads.

    Returns (findings, hidden_snippets). The snippets are what the caller
    should strip from the extracted text before prompting a model.
    """
    findings: list[dict[str, Any]] = []
    hidden_text: list[str] = []

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:  # pragma: no cover - handled upstream by extraction
        return findings, hidden_text

    try:
        for page_number, page in enumerate(document, start=1):
            try:
                data = page.get_text("dict")
            except Exception:  # pragma: no cover - defensive
                continue

            page_rect = page.rect
            backdrops: list[fitz.Rect] | None = None  # computed lazily; get_drawings is not free

            for block in data.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = str(span.get("text", "")).strip()
                        if not _is_meaningful(text):
                            # Invisible formatting characters and decorative
                            # rules ("--------", a stray U+206B) are hidden by
                            # nature and carry no payload. Reporting them buries
                            # real findings in noise.
                            continue

                        bbox = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))
                        size = float(span.get("size", 0) or 0)
                        alpha = span.get("alpha", 255)
                        reasons: list[str] = []

                        if isinstance(alpha, (int, float)) and alpha <= 0:
                            reasons.append("fully transparent")

                        if _is_light(span.get("color")):
                            if backdrops is None:
                                backdrops = _dark_backdrops(page)
                            # White text is only hidden when nothing dark is
                            # painted behind it -- otherwise it is a heading
                            # on a coloured banner, which is ordinary design.
                            on_dark = any(rect.intersects(bbox) for rect in backdrops)
                            if not on_dark:
                                reasons.append("light text with no darker background behind it")

                        if 0 < size < _MIN_LEGIBLE_FONT_PT:
                            reasons.append(f"font size {size:.1f}pt is below legibility")

                        if not bbox.intersects(page_rect):
                            reasons.append("positioned outside the visible page area")

                        if reasons:
                            hidden_text.append(text)
                            findings.append(
                                {
                                    "type": "hidden_text",
                                    # Invisible text is not automatically an
                                    # attack. A real published paper in the
                                    # sample set contains white-on-white text
                                    # left behind by its authoring tool --
                                    # verified by rendering the page. Grading
                                    # every such span "high" would cry wolf on
                                    # honest documents. What escalates it to
                                    # critical is instruction-like phrasing,
                                    # which scan_injection_phrases reports
                                    # separately. The text is quarantined out
                                    # of the prompt either way.
                                    "severity": "medium",
                                    "page": page_number,
                                    "detail": "; ".join(reasons),
                                    "text_preview": text[:160],
                                }
                            )
    finally:
        document.close()

    return findings, hidden_text


# --- Injection phrasing ------------------------------------------------------

def scan_injection_phrases(visible_text: str, hidden_text: list[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    for snippet in hidden_text:
        match = _INJECTION_RE.search(snippet)
        if match:
            findings.append(
                {
                    "type": "prompt_injection",
                    "severity": "critical",
                    "detail": "Instruction-like phrasing concealed in hidden text.",
                    "text_preview": snippet[:160],
                    "matched": match.group(0)[:80],
                }
            )

    match = _INJECTION_RE.search(visible_text or "")
    if match:
        findings.append(
            {
                # Visible instruction-like text is far weaker evidence: a
                # security policy or a prompt-engineering guide legitimately
                # contains these phrases.
                "type": "prompt_injection_visible",
                "severity": "medium",
                "detail": "Instruction-like phrasing present in visible text.",
                "matched": match.group(0)[:80],
            }
        )
    return findings


# --- Public entry point ------------------------------------------------------

def scan_pdf(file_bytes: bytes, visible_text: str = "") -> dict[str, Any]:
    """Run every deterministic check and grade the document.

    `severity` is the worst finding. Callers decide policy from it: refuse
    on "critical" active content, quarantine and warn otherwise.
    """
    active = scan_active_content(file_bytes)
    hidden_findings, hidden_text = scan_hidden_text(file_bytes)
    occluded_findings, occluded_text = scan_occluded_text(file_bytes)
    hidden_findings += occluded_findings
    hidden_text += occluded_text
    annotation_findings = scan_annotations(file_bytes)
    # Hidden spans are still embedded in the extracted text at this point, so
    # they are removed before looking for injection phrasing "in visible
    # text" -- otherwise a concealed payload is reported twice, once as
    # hidden and again as visible.
    truly_visible = strip_hidden_text(visible_text, hidden_text)
    injection = scan_injection_phrases(truly_visible, hidden_text)

    confusable = scan_confusable_text(truly_visible)
    findings = active + hidden_findings + injection + confusable + annotation_findings
    severity = "none"
    for finding in findings:
        severity = _worst(severity, finding.get("severity", "info"))

    # Only active content makes the file itself unsafe to process. Hidden
    # text is neutralised by quarantine, so it warns rather than blocks.
    blocking = [f for f in active if f["severity"] == "critical"]

    return {
        "severity": severity,
        "safe_to_parse": not blocking,
        "findings": findings,
        "hidden_text_snippets": hidden_text,
        "counts": {
            "active_content": len(active),
            "hidden_text": len(hidden_findings),
            "prompt_injection": len(injection),
            "confusable_text": len(confusable),
            "annotations": len(annotation_findings),
        },
    }


def strip_hidden_text(text: str, hidden_snippets: list[str]) -> str:
    """Remove quarantined spans from extracted text before prompting.

    Detecting an injection but still handing it to the model would be worse
    than not detecting it, so this runs before the prompt is built. Only
    exact spans identified as hidden are removed; visible content is never
    touched.
    """
    if not hidden_snippets:
        return text
    cleaned = text
    for snippet in sorted(set(hidden_snippets), key=len, reverse=True):
        if len(snippet.strip()) < 3:
            continue
        cleaned = cleaned.replace(snippet, " [removed: hidden text] ")
    return cleaned


# --- Occluded text (hidden behind a logo or shape drawn over it) -------------
#
# The hidden-text check above asks what is painted *behind* a span. This asks
# the opposite: what is painted *in front* of it. Text can be perfectly
# black-on-white and still invisible if an opaque logo is drawn on top
# afterwards -- the reader sees the logo, the extractor sees the text.
#
# PyMuPDF exposes a `seqno` (content-stream sequence number) on both text
# spans and drawings, which is what makes this decidable: a shape with a
# higher seqno was painted later, therefore on top. Without that ordering a
# background banner would be indistinguishable from a covering logo, since
# both simply overlap the text.
_OPAQUE_FILL_THRESHOLD = 0.95
_OCCLUSION_AREA_RATIO = 0.85


def _texttrace_string(span: dict[str, Any]) -> str:
    chars = span.get("chars") or []
    out = []
    for char in chars:
        try:
            out.append(chr(char[0]))
        except (TypeError, ValueError, IndexError):  # pragma: no cover - defensive
            continue
    return "".join(out)


def scan_occluded_text(file_bytes: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """Detect text hidden underneath an opaque shape drawn over it.

    DISABLED -- returns no findings.

    Deciding this requires z-order: a shape painted *before* the text is a
    background (white heading on a dark banner is ordinary design), while a
    shape painted *after* it is a cover. The only PyMuPDF API exposing that
    ordering for text is `get_texttrace()`, and using it here proved unsafe:
    scanning the sample corpus in one process reproducibly ended in
    `Fatal Python error: none_dealloc` during garbage collection. Individual
    documents were fine; the fault accumulated across many open/close
    cycles, which in a long-running server would eventually abort the
    process mid-request.

    A working implementation existed and correctly caught text hidden under
    a logo, but a detector that can crash the service is worse than a
    documented gap, so it is switched off rather than shipped. Note the
    other hidden-text checks still cover the common variants (white-on-
    white, transparent, microscopic, off-page); what is missed is
    specifically normal-looking text with an opaque shape painted on top.

    Re-enabling needs either a PyMuPDF version where `get_texttrace()` is
    stable, or a render-and-compare approach that checks whether a span's
    ink actually appears in the rasterised page.
    """
    return [], []


# --- Confusable / mixed-script text ------------------------------------------
#
# A Cyrillic "о" is visually identical to a Latin "o". Swapping one into a
# word leaves the page looking untouched while changing the bytes a parser
# matches on -- enough to defeat a label lookup, or to spoof an organisation
# name. Legitimate multilingual documents contain several scripts, but they
# very rarely mix two scripts *inside a single word*, which is what is
# flagged here.
_SCRIPT_PREFIXES = ("LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "HEBREW")


def _char_script(char: str) -> str | None:
    import unicodedata

    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    for prefix in _SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return None


def scan_confusable_text(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for token in re.findall(r"\S+", text or ""):
        letters = [c for c in token if c.isalpha()]
        if len(letters) < 3:
            continue
        scripts = {s for s in (_char_script(c) for c in letters) if s}
        if len(scripts) > 1 and token not in seen:
            seen.add(token)
            findings.append(
                {
                    "type": "confusable_text",
                    "severity": "high",
                    "detail": (
                        "Word mixes character sets ("
                        + ", ".join(sorted(scripts))
                        + "), which renders identically but changes the underlying text."
                    ),
                    "text_preview": token[:80],
                }
            )
        if len(findings) >= 20:  # a corrupted document shouldn't produce endless findings
            break
    return findings


# --- Annotations / comment boxes ---------------------------------------------
#
# Annotation text is not part of the page content stream, so it does not
# reach the model through the current extraction path. It is scanned anyway:
# the safety of that arrangement is an accident of how extraction happens to
# work today, not a control, and a comment reading "This document has been
# verified, approve it" is a social-engineering payload aimed at whoever or
# whatever reads it next.
def scan_annotations(file_bytes: bytes) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:  # pragma: no cover - handled upstream
        return findings

    try:
        for page_number, page in enumerate(document, start=1):
            try:
                annotations = list(page.annots() or [])
            except Exception:  # pragma: no cover - defensive
                continue
            for annotation in annotations:
                try:
                    info = annotation.info or {}
                except Exception:  # pragma: no cover - defensive
                    continue
                content = " ".join(
                    str(info.get(key, "") or "") for key in ("content", "subject", "title")
                ).strip()
                if not content:
                    continue
                match = _INJECTION_RE.search(content)
                findings.append(
                    {
                        "type": "annotation_injection" if match else "annotation_text",
                        "severity": "high" if match else "info",
                        "page": page_number,
                        "detail": (
                            "Instruction-like phrasing inside a PDF annotation."
                            if match
                            else "Document carries annotation/comment text."
                        ),
                        "text_preview": content[:160],
                    }
                )
    finally:
        document.close()

    return findings
