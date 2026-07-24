from __future__ import annotations

import pytest

from app.json_utils import (
    clean_llm_json,
    clean_scalar,
    dedupe_strings,
    find_key_recursive,
    normalize_list,
    normalize_value,
)


def test_clean_llm_json_plain_object():
    assert clean_llm_json('{"a": 1}') == {"a": 1}


def test_clean_llm_json_strips_markdown_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert clean_llm_json(raw) == {"a": 1}


def test_clean_llm_json_strips_leading_commentary():
    raw = 'Sure, here is the JSON:\n{"a": 1}\nLet me know if you need anything else.'
    assert clean_llm_json(raw) == {"a": 1}


def test_clean_llm_json_repairs_truncated_output():
    # Missing closing braces/brackets, as if generation was cut off mid-stream.
    raw = '{"a": 1, "b": [1, 2, 3'
    result = clean_llm_json(raw)
    assert result["a"] == 1


def test_clean_llm_json_rejects_non_object():
    with pytest.raises(ValueError):
        clean_llm_json("[1, 2, 3]")


def test_clean_llm_json_rejects_empty():
    with pytest.raises(ValueError):
        clean_llm_json("")


def test_clean_scalar_trims_and_empties_to_none():
    assert clean_scalar("  hello  ") == "hello"
    assert clean_scalar("   ") is None
    assert clean_scalar(None) is None
    assert clean_scalar(42) == 42
    assert clean_scalar(True) is True


def test_normalize_list_flattens_and_drops_empty_dicts():
    value = [{"a": " x ", "b": None}, {}, None, "  ", "keep"]
    assert normalize_list(value) == [{"a": "x"}, "keep"]


def test_normalize_list_wraps_scalar():
    assert normalize_list("solo") == ["solo"]
    assert normalize_list(None) == []


def test_normalize_value_recurses():
    value = {"outer": [{"inner": " y "}]}
    assert normalize_value(value) == {"outer": [{"inner": "y"}]}


def test_dedupe_strings_case_insensitive_preserves_order():
    assert dedupe_strings(["Foo", "bar", "FOO", "Bar", "baz"]) == ["Foo", "bar", "baz"]


def test_find_key_recursive_matches_normalized_alias():
    data = {"Application No.": "APP123", "nested": {"other": "value"}}
    assert find_key_recursive(data, {"application_no", "application_number"}) == "APP123"


def test_find_key_recursive_returns_none_when_absent():
    assert find_key_recursive({"a": {"b": "c"}}, {"missing_key"}) is None
