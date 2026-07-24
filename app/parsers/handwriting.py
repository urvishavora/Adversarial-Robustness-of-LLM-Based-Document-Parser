"""Handwritten form-field extraction using an Ollama vision model.

Requires a multimodal model pulled locally, e.g. `ollama pull llama3.2-vision`
(configurable via the VISION_MODEL_NAME env var). This layer is additive: it
reads handwritten answers/ticked boxes off a rendered page image and merges
them into an application form's `fields.handwritten_fields`, on top of
whatever `application_form.py` already extracted from the printed text/OCR
layer. Runs only for documents classified as "Form" and only when
ENABLE_HANDWRITING is on.

NOTE ON A FIXED BUG: an earlier version of this file had a mis-indented
try/except around the confidence-parsing line in what is now
`_clean_confidence`, which raised a hard `IndentationError` on import (the
module could never even load). That block is rewritten below with
consistent indentation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app import config
from app.json_utils import clean_scalar, normalize_list
from app.llm_client import call_ollama_vision
from app.pdf_extraction import render_pages_as_jpeg_base64

logger = logging.getLogger("document_parser")

_HANDWRITING_PROMPT = """
You are reading one page of an application form.

Extract only:
- handwritten words
- handwritten numbers
- handwritten dates
- handwritten signatures when readable
- visibly checked, ticked, filled, or selected options

Printed text is context only. Do not return printed labels as answers unless the
printed option is visibly selected.

Return exactly one JSON object:

{
  "entries": [
    {
      "field": "nearest printed field label",
      "value": "handwritten or selected value",
      "confidence": 0.0
    }
  ]
}

Rules:
1. Each entry must represent one visible handwritten answer or selected option.
2. Use the nearest printed label as field.
3. Preserve spelling, punctuation, phone numbers, dates, IDs, and email addresses.
4. For a checked option, use the option label as value.
5. Do not include empty fields or unselected options.
6. Do not guess unreadable writing.
7. confidence must be between 0 and 1.
8. Return valid JSON only.
""".strip()


def _empty_handwriting_result() -> dict[str, Any]:
    return {"entries": [], "confidence": 0.0, "review_required": True, "source": "ollama_vision"}


def _clean_confidence(raw: Any) -> float:
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        confidence = 0.0
    return max(0.0, min(1.0, confidence))


def _clean_handwriting_result(value: Any) -> dict[str, Any]:
    result = _empty_handwriting_result()
    if not isinstance(value, dict):
        return result

    entries = value.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    cleaned = []
    for item in entries:
        if not isinstance(item, dict):
            continue

        field = clean_scalar(item.get("field"))
        entry_value = clean_scalar(item.get("value"))
        confidence = _clean_confidence(item.get("confidence", 0.0))

        if field and entry_value:
            cleaned_entry: dict[str, Any] = {
                "field": str(field),
                "value": str(entry_value),
                "confidence": round(confidence, 3),
            }
            page = item.get("page")
            if isinstance(page, int) and page > 0:
                cleaned_entry["page"] = page
            cleaned.append(cleaned_entry)

    result["entries"] = cleaned
    if cleaned:
        result["confidence"] = round(sum(e["confidence"] for e in cleaned) / len(cleaned), 3)
    result["review_required"] = result["confidence"] < config.HANDWRITING_MIN_CONFIDENCE
    return result


def _normalize_handwriting_field_name(value: Any) -> str:
    """'First Name' -> 'first_name', 'Mobile No.' -> 'mobile_no'."""
    text = str(value or "").strip().lower()
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _merge_handwriting_page_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge page-level handwriting entries and remove duplicates."""
    merged = _empty_handwriting_result()
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}

    for page_result in results:
        if not isinstance(page_result, dict):
            continue
        entries = page_result.get("entries")
        if not isinstance(entries, list):
            continue

        for item in entries:
            if not isinstance(item, dict):
                continue
            field = clean_scalar(item.get("field"))
            entry_value = clean_scalar(item.get("value"))
            if field is None or entry_value is None:
                continue

            confidence = _clean_confidence(item.get("confidence", 0.0))
            cleaned_entry: dict[str, Any] = {
                "field": str(field),
                "value": str(entry_value),
                "confidence": round(confidence, 3),
            }
            page = item.get("page")
            if isinstance(page, int) and page > 0:
                cleaned_entry["page"] = page

            key = (_normalize_handwriting_field_name(str(field)), str(entry_value).strip().casefold())
            existing = deduplicated.get(key)
            if existing is None or confidence > float(existing.get("confidence", 0.0)):
                deduplicated[key] = cleaned_entry

    merged["entries"] = list(deduplicated.values())
    confidences = [
        float(item["confidence"]) for item in merged["entries"] if isinstance(item.get("confidence"), (int, float))
    ]
    if confidences:
        merged["confidence"] = round(sum(confidences) / len(confidences), 3)
    merged["review_required"] = not merged["entries"] or merged["confidence"] < config.HANDWRITING_MIN_CONFIDENCE
    return merged


def extract_handwritten_application_fields(file_bytes: bytes) -> dict[str, Any]:
    """Extract handwritten entries page by page using a compact generic schema."""
    images = render_pages_as_jpeg_base64(file_bytes)

    if not images:
        result = _empty_handwriting_result()
        result.update(
            {
                "error": "no_rendered_pages",
                "error_detail": "The PDF did not produce any renderable pages.",
                "vision_model": config.VISION_MODEL_NAME,
            }
        )
        return result

    page_results: list[dict[str, Any]] = []
    page_errors: list[dict[str, Any]] = []

    for page_number, image_base64 in enumerate(images, start=1):
        try:
            raw_result = call_ollama_vision(image_base64, _HANDWRITING_PROMPT)
            page_result = _clean_handwriting_result(raw_result)
            entries = page_result.get("entries")
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        entry.setdefault("page", page_number)
            page_results.append(page_result)
        except Exception as exc:
            logger.exception("Handwriting extraction failed for page %s", page_number)
            page_errors.append({"page": page_number, "error": type(exc).__name__, "detail": str(exc)})

    if not page_results:
        result = _empty_handwriting_result()
        result.update(
            {
                "error": "handwriting_extraction_unavailable",
                "error_detail": "Vision extraction failed for every rendered page.",
                "page_errors": page_errors,
                "vision_model": config.VISION_MODEL_NAME,
            }
        )
        return result

    result = _merge_handwriting_page_results(page_results)
    result["vision_model"] = config.VISION_MODEL_NAME
    result["pages_processed"] = len(page_results)
    result["pages_failed"] = len(page_errors)
    if page_errors:
        result["page_errors"] = page_errors
        result["review_required"] = True
    return result


# --- Mapping raw entries into a stable, field-named schema -------------------

_SCALAR_ALIASES: dict[str, set[str]] = {
    "position_applied_for": {"position_applied_for", "position", "job_applied_for", "role_applied_for", "vacancy_applied_for", "job_title"},
    "first_name": {"first_name", "forename", "given_name"},
    "last_name": {"last_name", "surname", "family_name"},
    "national_insurance_number": {"national_insurance_number", "national_insurance_no", "ni_number", "ni_no", "nino"},
    "address": {"address", "home_address", "residential_address", "postal_address"},
    "email": {"email", "email_address", "e_mail"},
    "mobile": {"mobile", "mobile_number", "mobile_no", "cell", "cell_phone", "cellphone"},
    "home_phone": {"home_phone", "home_phone_number", "telephone", "telephone_number", "landline"},
    "age_bracket": {"age_bracket", "age_group", "age_range"},
    "preferred_work_hours": {"preferred_work_hours", "preferred_hours", "hours_preferred", "preferred_shift", "preferred_shifts"},
    "employment_duration": {"employment_duration", "length_of_employment", "duration_of_employment", "how_long_can_you_work", "availability_duration"},
    "career_goals": {"career_goals", "career_goal", "future_goals", "career_aspirations"},
    "handwritten_date": {"date", "signed_date", "signature_date", "application_date", "handwritten_date"},
}

_DAY_ALIASES: dict[str, tuple[str, ...]] = {
    "monday": ("monday", "mon"),
    "tuesday": ("tuesday", "tue", "tues"),
    "wednesday": ("wednesday", "wed"),
    "thursday": ("thursday", "thu", "thur", "thurs"),
    "friday": ("friday", "fri"),
    "saturday": ("saturday", "sat"),
    "sunday": ("sunday", "sun"),
}

_SELECTED_VALUES = {"true", "yes", "selected", "checked", "ticked", "tick", "x", "✓", "✔", "filled"}


def _handwriting_value_is_selected(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in _SELECTED_VALUES


def _append_unique(target: list[Any], value: Any) -> None:
    cleaned = clean_scalar(value)
    if cleaned is None:
        return
    candidate = str(cleaned)
    existing = {str(item).strip().casefold() for item in target if item not in (None, "")}
    if candidate.strip().casefold() not in existing:
        target.append(candidate)


def _map_handwriting_entries(handwriting: dict[str, Any]) -> dict[str, Any]:
    """Convert compact vision-model entries into a stable, named schema."""
    mapped: dict[str, Any] = {
        "position_applied_for": None,
        "first_name": None,
        "last_name": None,
        "national_insurance_number": None,
        "address": None,
        "email": None,
        "mobile": None,
        "home_phone": None,
        "age_bracket": None,
        "eligibility_to_work": {},
        "preferred_work_hours": None,
        "weekly_availability": {},
        "employment_duration": None,
        "top_qualities": [],
        "customer_service_responses": [],
        "career_goals": None,
        "signature_present": False,
        "signature_name": None,
        "handwritten_date": None,
        "other_handwritten_entries": [],
        "entries": normalize_list(handwriting.get("entries")),
        "confidence": handwriting.get("confidence", 0.0),
        "review_required": bool(handwriting.get("review_required", True)),
        "source": handwriting.get("source", "ollama_vision"),
    }

    alias_to_target: dict[str, str] = {
        alias: target for target, aliases in _SCALAR_ALIASES.items() for alias in aliases
    }

    entries = handwriting.get("entries")
    if not isinstance(entries, list):
        entries = []

    for item in entries:
        if not isinstance(item, dict):
            continue

        original_field = clean_scalar(item.get("field"))
        entry_value = clean_scalar(item.get("value"))
        if original_field is None or entry_value is None:
            continue

        field_key = _normalize_handwriting_field_name(original_field)
        value_text = str(entry_value).strip()

        target_field = alias_to_target.get(field_key)
        if target_field:
            if mapped[target_field] in (None, ""):
                mapped[target_field] = value_text
            continue

        matched_scalar = False
        for alias, target in alias_to_target.items():
            if len(alias) >= 5 and (field_key.startswith(f"{alias}_") or field_key.endswith(f"_{alias}")):
                if mapped[target] in (None, ""):
                    mapped[target] = value_text
                matched_scalar = True
                break
        if matched_scalar:
            continue

        matched_day = False
        for canonical_day, aliases in _DAY_ALIASES.items():
            if any(
                field_key == alias
                or field_key.startswith(f"{alias}_")
                or field_key.endswith(f"_{alias}")
                or f"_{alias}_" in f"_{field_key}_"
                for alias in aliases
            ):
                mapped["weekly_availability"][canonical_day] = value_text
                matched_day = True
                break
        if matched_day:
            continue

        if any(
            phrase in field_key
            for phrase in (
                "eligible_to_work", "eligibility_to_work", "right_to_work",
                "legally_entitled_to_work", "permission_to_work",
                "work_authorisation", "work_authorization",
            )
        ):
            option_key = field_key
            for prefix in (
                "eligible_to_work_", "eligibility_to_work_", "right_to_work_",
                "legally_entitled_to_work_", "permission_to_work_",
                "work_authorisation_", "work_authorization_",
            ):
                option_key = option_key.removeprefix(prefix)
            if option_key in {"", "eligible", "eligibility"}:
                option_key = "answer"
            mapped["eligibility_to_work"][option_key] = (
                True if _handwriting_value_is_selected(entry_value) else value_text
            )
            continue

        if "signature" in field_key or field_key in {"signed_by", "applicant_signature"}:
            mapped["signature_present"] = True
            if value_text.casefold() not in {"true", "yes", "selected", "checked", "present", "signed", "x", "✓", "✔"}:
                mapped["signature_name"] = value_text
            continue

        if any(phrase in field_key for phrase in ("quality", "qualities", "strength", "strengths", "best_attribute", "top_attribute")):
            _append_unique(mapped["top_qualities"], value_text)
            continue

        if any(phrase in field_key for phrase in ("customer_service", "customer_experience", "customer_question", "dealing_with_customers")):
            mapped["customer_service_responses"].append(
                {"field": str(original_field), "response": value_text, "confidence": item.get("confidence", 0.0)}
            )
            continue

        mapped["other_handwritten_entries"].append(
            {"field": str(original_field), "value": value_text, "confidence": item.get("confidence", 0.0)}
        )

    if mapped["signature_name"]:
        mapped["signature_present"] = True

    confidence = _clean_confidence(mapped["confidence"])
    mapped["confidence"] = round(confidence, 3)
    mapped["review_required"] = mapped["confidence"] < config.HANDWRITING_MIN_CONFIDENCE

    # Do not publish a guessed signature transcription when the overall
    # vision result is below the configured confidence threshold.
    if mapped["review_required"]:
        mapped["signature_name"] = None

    for metadata_key in ("vision_model", "error", "error_detail"):
        if metadata_key in handwriting:
            mapped[metadata_key] = handwriting[metadata_key]

    return mapped


def merge_handwritten_fields(parsed_output: dict[str, Any], handwriting: dict[str, Any]) -> None:
    """Map vision entries into the stable handwriting schema and attach them
    to the parsed application form output under fields.handwritten_fields.
    """
    fields = parsed_output.get("fields")
    if not isinstance(fields, dict):
        fields = {}
        parsed_output["fields"] = fields
    fields["handwritten_fields"] = _map_handwriting_entries(handwriting)
