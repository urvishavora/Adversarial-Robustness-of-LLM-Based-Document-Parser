"""Printed application form parsing: LLM extraction + deterministic enrichment.

This covers the printed-text side of "application forms" -- labels, tables,
declarations, and printed structure. The separate `handwriting.py` module
layers handwritten fill-in answers on top of whatever this module extracts,
for forms that have both (the common case: a printed template with
handwritten or ticked-box answers).
"""

from __future__ import annotations

import re
from typing import Any

from app.json_utils import dedupe_strings, find_key_recursive, normalize_list
from app.parsers.generic import parse_generic_document
from app.regex_utils import extract_dates

_GUIDANCE = """
APPLICATION-FORM-SPECIFIC GUIDANCE:
Extract every visible printed label and every filled value on this application form.

1. Build fields dynamically from the labels and sections present in this document. Do
   not assume a fixed application schema.
2. Do a complete top-to-bottom inventory before answering. Include headings,
   identifiers, personal details, addresses, program/position details, every table row
   and column, option answers, activities, contacts, declarations, dates, and
   office-use fields when filled.
3. Preserve section hierarchy as nested objects. Preserve table rows as arrays of objects.
4. Separate neighboring columns by matching each value to its nearest visible label.
   Never append one field's value onto a neighboring field.
5. Correct only unambiguous OCR substitutions such as S/$, O/0, I/1, or punctuation when
   the label and surrounding characters clearly support the correction.
6. For blank fields use null. For blank table rows, omit the row unless its presence matters.
7. A checkbox/radio option is selected only when the source clearly shows a filled mark,
   tick, or X. Characters such as O, C, D, or 0 beside an option are not proof of selection.
8. Keep distinct activities, languages, programs, and selections as arrays when multiple
   values are present.
9. Signatures may be handwritten. Include a signature transcription only when reasonably
   legible; otherwise use null. Never guess a signature from the applicant's printed name.
10. Before returning JSON, verify that every non-empty label/value visible in the source
    text is represented somewhere under fields.
"""

_IDENTIFIER_LABEL_PATTERN = re.compile(
    r"(?im)\b(?:application|registration|reference|candidate|student|admission)"
    r"\s*(?:id|no\.?|number|#)\b"
)
_IDENTIFIER_TOKEN_PATTERN = re.compile(r"[A-Za-z$0-9][A-Za-z0-9$_.\/-]{4,}")

_IDENTIFIER_ALIASES = {
    "application_id", "application_no", "application_number",
    "registration_id", "registration_no", "registration_number",
    "reference_id", "reference_no", "reference_number",
    "candidate_id", "student_id", "admission_id",
}


def _clean_identifier_candidate(value: str) -> str | None:
    candidate = value.strip().strip(":;,.|[](){}")
    candidate = re.sub(r"\s+", "", candidate)

    # OCR commonly reads a leading capital S as $. Correct it only for an
    # identifier-like alphanumeric token, never for normal prose or money.
    if candidate.startswith("$") and re.fullmatch(r"\$[A-Za-z0-9][A-Za-z0-9_./-]{4,}", candidate):
        candidate = "S" + candidate[1:]

    if not re.fullmatch(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9][A-Za-z0-9_./-]{4,}", candidate):
        return None
    return candidate


def extract_labeled_application_id(text: str) -> str | None:
    """Extract an application/reference identifier only when a nearby label supports it."""
    for label_match in _IDENTIFIER_LABEL_PATTERN.finditer(text):
        window = text[label_match.end() : label_match.end() + 180]
        candidates = [
            cleaned
            for token in _IDENTIFIER_TOKEN_PATTERN.findall(window)
            if (cleaned := _clean_identifier_candidate(token))
        ]
        if candidates:
            return max(candidates[:6], key=lambda value: (sum(ch.isdigit() for ch in value), len(value)))
    return None


def extract_contextual_dates(text: str) -> dict[str, str]:
    """Extract high-value dates using nearby labels."""
    date_value = r"((?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)\d{2})"
    labels: tuple[tuple[str, str], ...] = (
        ("date_of_birth", r"(?:date\s+of\s+birth|dob)"),
        ("application_date", r"(?:application\s+date|date\s+of\s+application|submission\s+date|date\s+submitted)"),
        ("declaration_date", r"(?:declaration\s+date|date\s+of\s+declaration)"),
    )
    found: dict[str, str] = {}
    for target, label in labels:
        match = re.search(rf"(?is)\b{label}\b.{{0,140}}?{date_value}", text)
        if match:
            found[target] = match.group(1)

    if "declaration_date" not in found:
        declaration_match = re.search(
            rf"(?is)(?:declaration|signature\s+of\s+applicant).{{0,450}}?\bdate\s*[:.-]?\s*{date_value}",
            text,
        )
        if declaration_match:
            found["declaration_date"] = declaration_match.group(1)

    if "application_date" not in found and "declaration_date" in found:
        found["application_date"] = found["declaration_date"]
    return found


def _extract_labeled_text_value(
    text: str, label_patterns: tuple[str, ...], *, max_chars: int = 220
) -> str | None:
    combined = "|".join(f"(?:{pattern})" for pattern in label_patterns)
    match = re.search(rf"(?im)\b(?:{combined})\b\s*[:.-]?\s*(.+)", text)
    if not match:
        return None
    candidate = match.group(1)[:max_chars].strip()
    candidate = re.split(
        r"(?i)\s{2,}|\t|\b(?:preferred\s+speciali[sz]ation|other\s+programs?|"
        r"academic\s+qualifications?|date\s+of\s+birth|mobile|email|address)\b",
        candidate,
        maxsplit=1,
    )[0].strip(" :;,.|-_")
    return candidate or None


def extract_school_faculty(text: str) -> str | None:
    return _extract_labeled_text_value(
        text,
        (r"school\s*/?\s*faculty", r"faculty\s*/?\s*school", r"school\s+or\s+faculty", r"faculty"),
    )


def extract_examination_names(text: str) -> list[str]:
    """Recover visible examination labels in their top-to-bottom source order."""
    patterns: tuple[tuple[str, str], ...] = (
        (r"\b(?:class\s*)?10(?:th)?\b(?:\s*\(?high\s*school\)?)?|\bhigh\s*school\b", "Class 10 (High School)"),
        (r"\b(?:class\s*)?12(?:th)?\b(?:\s*\(?10\s*\+\s*2\)?)?|\b10\s*\+\s*2\b|\bintermediate\b", "Class 12 (10+2)"),
        (r"\bdiploma\b", "Diploma"),
        (r"\bundergraduate\b|\bbachelor(?:'s)?\b", "Undergraduate"),
        (r"\bpostgraduate\b|\bmaster(?:'s)?\b", "Postgraduate"),
    )
    found: list[tuple[int, str]] = []
    for pattern, canonical in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), canonical))
    return dedupe_strings(value for _, value in sorted(found, key=lambda item: item[0]))


def extract_declaration_text(text: str) -> str | None:
    """Extract printed declaration prose while excluding date/signature metadata."""
    start = re.search(r"(?im)^\s*declaration\s*[:.-]?\s*$", text)
    if not start:
        start = re.search(r"(?im)\bdeclaration\b\s*[:.-]?", text)
    if not start:
        return None

    tail = text[start.end() : start.end() + 1400]
    stop = re.search(
        r"(?im)^\s*(?:date|place|signature(?:\s+of\s+(?:the\s+)?applicant)?|"
        r"applicant(?:'s)?\s+signature|for\s+office\s+use|office\s+use)\s*[:.-]?",
        tail,
    )
    if stop:
        tail = tail[: stop.start()]

    lines: list[str] = []
    for raw_line in tail.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" |_-\t")
        if not line or re.fullmatch(r"[-_=.| ]+", line):
            continue
        lines.append(line)

    declaration = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if len(declaration) < 20:
        return None
    return declaration[:1200]


def _get_first_dict(parent: dict[str, Any], aliases: tuple[str, ...]) -> dict[str, Any] | None:
    for alias in aliases:
        value = parent.get(alias)
        if isinstance(value, dict):
            return value
    return None


def _get_first_list(parent: dict[str, Any], aliases: tuple[str, ...]) -> list[Any] | None:
    for alias in aliases:
        value = parent.get(alias)
        if isinstance(value, list):
            return value
    return None


def _enrich_sections(fields: dict[str, Any], text: str) -> None:
    """Restore faculty, examination labels, and declaration without another model call."""
    program = _get_first_dict(
        fields, ("program_information", "program_details", "course_information", "course_details")
    )
    if program is not None:
        faculty_aliases = ("school_faculty", "school", "faculty", "school_or_faculty")
        existing_faculty = next(
            (program.get(key) for key in faculty_aliases if program.get(key) not in (None, "")), None
        )
        extracted_faculty = extract_school_faculty(text)

        if existing_faculty in (None, "") and extracted_faculty:
            program["school_faculty"] = extracted_faculty
            existing_faculty = extracted_faculty

        program_key = next(
            (key for key in ("program_applied_for", "program", "course_applied_for", "course") if program.get(key)),
            None,
        )
        if program_key and existing_faculty:
            program_value = str(program[program_key]).strip()
            faculty_value = str(existing_faculty).strip()
            if program_value.casefold().endswith(faculty_value.casefold()):
                trimmed = program_value[: -len(faculty_value)].rstrip(" -–—,;:/|")
                if trimmed:
                    program[program_key] = trimmed

    academic_rows = _get_first_list(
        fields, ("academic_qualification", "academic_qualifications", "education", "educational_qualification")
    )
    if academic_rows:
        examination_names = extract_examination_names(text)
        for index, row in enumerate(academic_rows):
            if not isinstance(row, dict) or index >= len(examination_names):
                continue
            aliases = ("examination", "examination_name", "qualification", "class", "exam")
            if not any(row.get(key) not in (None, "") for key in aliases):
                row["examination"] = examination_names[index]

    declaration = fields.get("declaration")
    if declaration in (None, "", {}, []):
        declaration_text = extract_declaration_text(text)
        if declaration_text:
            fields["declaration"] = declaration_text


def _enrich(result: dict[str, Any], text: str) -> None:
    """Deterministically restore critical identifiers and dates after LLM parsing."""
    fields = result.get("fields")
    if not isinstance(fields, dict):
        fields = {}
        result["fields"] = fields

    _enrich_sections(fields, text)

    nested_dates = fields.pop("dates", [])
    nested_parties = fields.pop("parties", [])
    fields.pop("security_notes", None)

    existing_id = find_key_recursive(fields, _IDENTIFIER_ALIASES)
    extracted_id = extract_labeled_application_id(text)
    cleaned_existing_id = _clean_identifier_candidate(str(existing_id)) if existing_id is not None else None
    application_id = extracted_id or cleaned_existing_id
    if application_id:
        fields["application_id"] = application_id

    contextual_dates = extract_contextual_dates(text)
    personal = fields.get("personal_information")
    if not isinstance(personal, dict):
        personal = fields.get("personal_info")
    if not isinstance(personal, dict):
        personal = None

    dob = contextual_dates.get("date_of_birth")
    if dob:
        if personal is not None:
            personal.setdefault("date_of_birth", dob)
        elif find_key_recursive(fields, {"date_of_birth", "dob"}) is None:
            fields["date_of_birth"] = dob

    declaration_date = contextual_dates.get("declaration_date")
    if declaration_date and find_key_recursive(fields, {"declaration_date", "date_of_declaration"}) is None:
        fields["declaration_date"] = declaration_date

    application_date = contextual_dates.get("application_date")
    if application_date and find_key_recursive(
        fields, {"application_date", "date_of_application", "submission_date"}
    ) is None:
        fields["application_date"] = application_date

    result["dates"] = dedupe_strings(
        [*normalize_list(result.get("dates")), *normalize_list(nested_dates), *extract_dates(text)]
    )
    if nested_parties:
        result["parties"] = normalize_list([*normalize_list(result.get("parties")), *normalize_list(nested_parties)])


def parse_application_form(text: str, filename: str) -> dict[str, Any]:
    return parse_generic_document(text, filename, "Form", extra_guidance=_GUIDANCE, enrich=_enrich)
