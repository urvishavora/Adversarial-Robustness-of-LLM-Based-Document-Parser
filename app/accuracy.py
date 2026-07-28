"""Resume extraction accuracy scoring.

There is no single universally-agreed way to turn "how close is this parsed
resume to the ground truth" into one percentage, especially once list
fields (education, experience, skills...) can differ in order, count, and
which optional sub-fields are present. This module implements one
reasonable, fully-deterministic definition so the number is at least
reproducible and improvable over time -- treat the resulting percentage as
"this project's accuracy metric", not an absolute truth.

Scoring approach, per resume:
- Scalar fields (name.*, job_title, contact.*) score 1.0 on an exact
  case/whitespace-insensitive match, else 0.0. contact.address and URL
  fields are compared on alphanumerics only, since commas, case and line
  breaks are transcription style rather than a difference in the value.
- summary scores on token-overlap similarity (0..1), since minor
  whitespace/punctuation differences shouldn't zero out an otherwise-correct
  summary.
- education / experience score via greedy best-overlap matching between
  expected and predicted entries, then per-matched-pair field agreement.
  An unmatched expected entry scores 0.
- skills / languages / certifications / awards / activities / references
  score via a flattened bag-of-strings recall: what fraction of the ground
  truth's leaf string values (regardless of exact shape -- list, dict, or
  dict-of-lists) also appear somewhere in the predicted output.

The overall score is the unweighted average of every field group that the
ground truth actually has content for (a group with nothing expected is
excluded from the average rather than forced to 0 or 1).
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from app.json_utils import flatten_strings

_LIST_OF_DICT_FIELDS = ("education", "experience")
_BAG_OF_STRINGS_FIELDS = ("skills", "languages", "certifications", "awards", "activities", "references")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _normalize_loosely(value: Any) -> str:
    """Compare on content, ignoring formatting the document didn't fix.

    Postal addresses and URLs are the same value whether written
    "43-589 Beechwood Dr, Waterloo, ON N2T 2K9" or
    "43-589 BEECHWOOD DR  WATERLOO ON N2T 2K9" -- case, commas, line breaks
    and a trailing "?" on a URL carry no information. Grading those by exact
    string equality reported three correctly-extracted addresses as 0%,
    which measures transcription style rather than whether the field was
    found.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _scalar_match(expected: Any, predicted: Any, *, loose: bool = False) -> float | None:
    """Returns None when there's nothing to grade (expected is empty)."""
    if expected in (None, ""):
        return None
    if loose:
        return 1.0 if _normalize_loosely(expected) == _normalize_loosely(predicted) else 0.0
    return 1.0 if _normalize_text(expected) == _normalize_text(predicted) else 0.0


def _text_similarity(expected: Any, predicted: Any) -> float | None:
    if expected in (None, ""):
        return None
    a, b = _normalize_text(expected), _normalize_text(predicted)
    if not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _bag_of_strings(value: Any) -> set[str]:
    return {_normalize_text(item) for item in flatten_strings(value) if str(item).strip()}


def _bag_recall(expected: Any, predicted: Any) -> float | None:
    expected_bag = _bag_of_strings(expected)
    if not expected_bag:
        return None
    predicted_bag = _bag_of_strings(predicted)
    return len(expected_bag & predicted_bag) / len(expected_bag)


def _entry_overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    bag_a, bag_b = _bag_of_strings(a), _bag_of_strings(b)
    if not bag_a or not bag_b:
        return 0.0
    return len(bag_a & bag_b) / max(len(bag_a), len(bag_b))


def _entry_field_agreement(expected: dict[str, Any], predicted: dict[str, Any]) -> float:
    graded_keys = [key for key in expected if expected.get(key) not in (None, "", [], {})]
    if not graded_keys:
        return 1.0
    matches = sum(
        1
        for key in graded_keys
        if _normalize_text(expected.get(key)) == _normalize_text(predicted.get(key))
    )
    return matches / len(graded_keys)


def _score_list_of_dicts(expected: list[Any], predicted: list[Any]) -> float | None:
    expected_dicts = [item for item in expected if isinstance(item, dict)]
    if not expected_dicts:
        return None
    predicted_dicts = [item for item in predicted if isinstance(item, dict)] if isinstance(predicted, list) else []

    remaining = list(predicted_dicts)
    total = 0.0
    for expected_entry in expected_dicts:
        if not remaining:
            continue  # no predicted entries left -> this entry scores 0
        best_index = max(range(len(remaining)), key=lambda i: _entry_overlap(expected_entry, remaining[i]))
        best_match = remaining[best_index]
        if _entry_overlap(expected_entry, best_match) > 0:
            total += _entry_field_agreement(expected_entry, best_match)
            remaining.pop(best_index)
    return total / len(expected_dicts)


def score_resume(expected: dict[str, Any], predicted: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Return (overall_score_0_to_1, per_field_scores) comparing a predicted
    resume against its ground-truth entry. Fields with nothing to grade in
    the ground truth are omitted from both the per-field dict and the
    overall average.
    """
    scores: dict[str, float] = {}

    expected_name = expected.get("name", {}) if isinstance(expected.get("name"), dict) else {}
    predicted_name = predicted.get("name", {}) if isinstance(predicted.get("name"), dict) else {}
    for sub in ("full_name", "first_name", "last_name"):
        result = _scalar_match(expected_name.get(sub), predicted_name.get(sub))
        if result is not None:
            scores[f"name.{sub}"] = result

    job_title_score = _scalar_match(expected.get("job_title"), predicted.get("job_title"))
    if job_title_score is not None:
        scores["job_title"] = job_title_score

    expected_contact = expected.get("contact", {}) if isinstance(expected.get("contact"), dict) else {}
    predicted_contact = predicted.get("contact", {}) if isinstance(predicted.get("contact"), dict) else {}
    # address and URLs are graded loosely: punctuation, case and line breaks
    # differ between how a document prints them and how they are transcribed,
    # and none of that changes whether the field was correctly found.
    _LOOSE_CONTACT = {"address", "linkedin", "website", "portfolio"}
    for sub in ("phone", "email", "address", "linkedin"):
        result = _scalar_match(
            expected_contact.get(sub), predicted_contact.get(sub), loose=sub in _LOOSE_CONTACT
        )
        if result is not None:
            scores[f"contact.{sub}"] = result

    summary_score = _text_similarity(expected.get("summary"), predicted.get("summary"))
    if summary_score is not None:
        scores["summary"] = summary_score

    for field in _LIST_OF_DICT_FIELDS:
        expected_value = expected.get(field)
        if isinstance(expected_value, list):
            result = _score_list_of_dicts(expected_value, predicted.get(field, []))
            if result is not None:
                scores[field] = result

    for field in _BAG_OF_STRINGS_FIELDS:
        result = _bag_recall(expected.get(field), predicted.get(field))
        if result is not None:
            scores[field] = result

    overall = sum(scores.values()) / len(scores) if scores else 1.0
    return overall, scores
