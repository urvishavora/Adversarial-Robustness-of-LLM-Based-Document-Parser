"""Tests for the handwriting mapping/merge logic (pure functions -- no vision
model call needed). These specifically guard against the IndentationError
bug that made the original main_hand_ocr.py fail to even import: the
confidence-parsing helper is exercised directly here with malformed input.
"""

from __future__ import annotations

from app.parsers.handwriting import (
    _clean_confidence,
    _clean_handwriting_result,
    _map_handwriting_entries,
    _merge_handwriting_page_results,
    merge_handwritten_fields,
)


def test_clean_confidence_handles_bad_input_without_raising():
    assert _clean_confidence("not a number") == 0.0
    assert _clean_confidence(None) == 0.0
    assert _clean_confidence(1.5) == 1.0
    assert _clean_confidence(-0.5) == 0.0
    assert _clean_confidence(0.42) == 0.42


def test_clean_handwriting_result_drops_incomplete_entries():
    raw = {
        "entries": [
            {"field": "First Name", "value": "John", "confidence": 0.9},
            {"field": "", "value": "should be dropped"},
            {"field": "Only field, no value"},
        ]
    }
    result = _clean_handwriting_result(raw)
    assert len(result["entries"]) == 1
    assert result["entries"][0]["field"] == "First Name"
    assert result["confidence"] == 0.9
    assert result["review_required"] is False


def test_clean_handwriting_result_non_dict_returns_empty():
    result = _clean_handwriting_result("not a dict")
    assert result["entries"] == []
    assert result["review_required"] is True


def test_merge_handwriting_page_results_dedupes_across_pages():
    page_1 = {"entries": [{"field": "First Name", "value": "John", "confidence": 0.8, "page": 1}]}
    page_2 = {"entries": [{"field": "first_name", "value": "John", "confidence": 0.95, "page": 2}]}
    merged = _merge_handwriting_page_results([page_1, page_2])
    assert len(merged["entries"]) == 1
    assert merged["entries"][0]["confidence"] == 0.95  # higher-confidence duplicate wins


def test_map_handwriting_entries_scalar_and_day_and_signature():
    handwriting = {
        "entries": [
            {"field": "First Name", "value": "Jamie", "confidence": 0.9},
            {"field": "Monday", "value": "9am-5pm", "confidence": 0.9},
            {"field": "Signature", "value": "Jamie Fox", "confidence": 0.9},
            {"field": "Top Quality", "value": "Reliable", "confidence": 0.9},
        ],
        "confidence": 0.9,
        "review_required": False,
    }
    mapped = _map_handwriting_entries(handwriting)
    assert mapped["first_name"] == "Jamie"
    assert mapped["weekly_availability"]["monday"] == "9am-5pm"
    assert mapped["signature_present"] is True
    assert mapped["signature_name"] == "Jamie Fox"
    assert "Reliable" in mapped["top_qualities"]


def test_map_handwriting_entries_low_confidence_suppresses_signature_name():
    handwriting = {
        "entries": [{"field": "Signature", "value": "Illegible Guess", "confidence": 0.2}],
        "confidence": 0.2,
        "review_required": True,
    }
    mapped = _map_handwriting_entries(handwriting)
    assert mapped["signature_present"] is True
    assert mapped["signature_name"] is None
    assert mapped["review_required"] is True


def test_merge_handwritten_fields_attaches_to_parsed_output():
    parsed_output: dict = {"fields": {"existing": "value"}}
    handwriting = {"entries": [{"field": "First Name", "value": "Sam", "confidence": 0.9}], "confidence": 0.9}
    merge_handwritten_fields(parsed_output, handwriting)
    assert parsed_output["fields"]["existing"] == "value"
    assert parsed_output["fields"]["handwritten_fields"]["first_name"] == "Sam"


def test_prompt_echo_and_repetition_loops_are_discarded():
    """Regression test from a real granite3.2-vision run. Two failure modes
    appeared in the output and both would have been published as data:

    - the model echoed the prompt's own schema placeholder, producing
      {"field": "handwritten or selected value"};
    - it fell into a repetition loop, returning a "national insurance
      number" of 419 followed by ~700 twos.

    Note the second is why overall character variety is the wrong test:
    that string still contains four distinct characters. A long run of one
    repeated character is the actual signal.
    """
    from app.parsers.handwriting import _clean_handwriting_result, _is_degenerate_value

    assert _is_degenerate_value("419" + "2" * 700) is True
    # Genuine values must survive, including long ones and spaced-out IDs.
    assert _is_degenerate_value("4 9 K L M T P 5") is False
    assert _is_degenerate_value("B-104, GREEN PARK SOCIETY, SECTOR 21, NOIDA - 201301") is False

    result = _clean_handwriting_result(
        {
            "entries": [
                {"field": "handwritten or selected value", "value": "Doe", "confidence": 0.5},
                {"field": "national insurance number", "value": "419" + "2" * 700, "confidence": 0.5},
                {"field": "Surname", "value": "Okafor", "confidence": 0.9},
                {"field": "Occupation", "value": "Taxi Driver", "confidence": 0.8},
            ]
        }
    )
    kept = [(e["field"], e["value"]) for e in result["entries"]]
    assert kept == [("Surname", "Okafor"), ("Occupation", "Taxi Driver")]


def test_canadian_gst_invoice_resolves_currency():
    """Regression test: GST invoices from BC/AB/MB reported currency as "$"
    or USD while Ontario HST invoices resolved correctly. "GST" alone is
    ambiguous (Australia, India, Singapore, NZ all use it), but a Canadian
    postal code is not used anywhere else -- the pair is decisive.
    """
    from app.regex_utils import extract_currency

    assert extract_currency("Vancouver, BC V6C 1V5\nGST 5%\n$180.00") == "CAD"
    assert extract_currency("Calgary, AB T2C 4S1\nGST 5%\n$95.00") == "CAD"
    # GST without a Canadian address stays ambiguous rather than guessing.
    assert extract_currency("12 George St, Sydney NSW 2000\nGST 10%\n$100.00") is None
