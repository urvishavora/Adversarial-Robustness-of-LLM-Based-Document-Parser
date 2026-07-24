"""JSON cleanup/repair and generic value normalization helpers.

These are pure functions with no I/O, which makes them fully unit-testable
without a working LLM backend -- most of this project's "does it actually
work" guarantee lives here.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

import json_repair

logger = logging.getLogger("document_parser")


def clean_llm_json(raw_output: str) -> dict[str, Any]:
    """Parse a model's raw text response into a JSON object.

    Handles the common failure modes of local LLM output:
    - Markdown code fences around the JSON.
    - Leading/trailing commentary outside the outermost braces.
    - Truncated JSON (the model hit its token limit mid-generation) -- in
      that case, `json_repair` is used to salvage whatever fields were
      actually finished rather than failing the whole request.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("Model returned an empty response")

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")

    end = cleaned.rfind("}")
    candidate = cleaned[start : end + 1] if end >= start else cleaned[start:]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning(
            "Model response was not valid JSON (likely truncated) -- "
            "attempting repair instead of failing the request."
        )
        repaired = json_repair.loads(cleaned[start:])
        if not isinstance(repaired, dict):
            raise ValueError("Model response is not a JSON object") from None
        return repaired

    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return parsed


def clean_scalar(value: Any) -> Any:
    """Normalize a single value: trim strings, drop empties down to None."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (int, float, bool)):
        return value
    return str(value).strip() or None


def normalize_list(value: Any) -> list[Any]:
    """Coerce any value into a clean list of scalars/dicts.

    - `None` -> `[]`
    - a bare scalar -> a one-item list
    - dict items are recursively cleaned and dropped if they end up empty
    - nested lists are flattened one level
    """
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    result: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            cleaned = {
                str(key): normalize_value(child)
                for key, child in item.items()
                if str(key).strip()
            }
            cleaned = {
                key: child for key, child in cleaned.items() if child not in (None, [], {})
            }
            if cleaned:
                result.append(cleaned)
        elif isinstance(item, list):
            result.extend(normalize_list(item))
        else:
            cleaned_item = clean_scalar(item)
            if cleaned_item is not None:
                result.append(cleaned_item)
    return result


def normalize_value(value: Any) -> Any:
    """Recursively normalize an arbitrary JSON value (dict/list/scalar)."""
    if isinstance(value, dict):
        return {
            str(key): normalize_value(child)
            for key, child in value.items()
            if str(key).strip()
        }
    if isinstance(value, list):
        return normalize_list(value)
    return clean_scalar(value)


def dedupe_strings(values: Iterable[Any]) -> list[str]:
    """Case-insensitive de-duplication that preserves first-seen order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_scalar(value)
        if cleaned is None:
            continue
        text = str(cleaned)
        key = text.casefold()
        if key not in seen:
            seen.add(key)
            result.append(text)
    return result


def flatten_strings(value: Any) -> Iterable[str]:
    """Yield every string found anywhere in a nested dict/list structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from flatten_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from flatten_strings(child)


def find_key_recursive(value: Any, aliases: set[str]) -> Any:
    """Depth-first search for the first value under any key in `aliases`.

    Key names are matched after lowercasing and collapsing non-alphanumeric
    runs to underscores, so "Application No.", "application_no", and
    "APPLICATION-NO" all match the same alias.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized_key in aliases and child not in (None, "", [], {}):
                return child
        for child in value.values():
            found = find_key_recursive(child, aliases)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_key_recursive(child, aliases)
            if found not in (None, "", [], {}):
                return found
    return None
