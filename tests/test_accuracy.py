"""Tests for the accuracy scorer itself (app/accuracy.py) -- pure functions,
no LLM/network involved. These prove the scorer behaves sanely before
anyone relies on the number it produces against real model output.
"""

from __future__ import annotations

import json

from app.accuracy import score_resume


def test_perfect_match_scores_1(ground_truth_path):
    payload = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    for resume in payload["resumes"]:
        overall, _ = score_resume(resume, resume)
        assert overall == 1.0, f"{resume['file_name']} did not self-match at 1.0"


def test_completely_wrong_output_scores_low():
    expected = {
        "name": {"full_name": "Jane Doe", "first_name": "Jane", "last_name": "Doe"},
        "job_title": "Engineer",
        "contact": {"phone": "555-1234", "email": "jane@example.com", "address": None, "linkedin": None},
        "summary": "Experienced backend engineer with 5 years in distributed systems.",
        "education": [{"institution": "State University", "degree": "BSc"}],
        "experience": [{"company": "Acme", "job_title": "Engineer"}],
        "skills": ["Python", "SQL"],
    }
    predicted = {
        "name": {"full_name": None, "first_name": None, "last_name": None},
        "job_title": None,
        "contact": {"phone": None, "email": None, "address": None, "linkedin": None},
        "summary": None,
        "education": [],
        "experience": [],
        "skills": [],
    }
    overall, _ = score_resume(expected, predicted)
    assert overall < 0.2


def test_partial_match_scores_between_zero_and_one():
    expected = {
        "name": {"full_name": "Jane Doe", "first_name": "Jane", "last_name": "Doe"},
        "job_title": "Engineer",
        "contact": {"phone": None, "email": None, "address": None, "linkedin": None},
        "skills": ["Python", "SQL", "Docker"],
    }
    predicted = {
        "name": {"full_name": "Jane Doe", "first_name": "Jane", "last_name": "Doe"},
        "job_title": "Software Engineer",  # close but not exact
        "contact": {},
        "skills": ["Python", "SQL"],  # missing "Docker"
    }
    overall, per_field = score_resume(expected, predicted)
    assert 0.5 < overall < 1.0
    assert per_field["job_title"] == 0.0  # exact-match field, not fuzzy
    assert 0.5 < per_field["skills"] < 1.0


def test_fields_absent_from_ground_truth_are_excluded_not_penalized():
    expected = {"name": {"full_name": "Jane Doe"}, "job_title": None, "contact": {}}
    predicted = {"name": {"full_name": "Jane Doe"}, "job_title": "Made Up Title", "contact": {}}
    overall, per_field = score_resume(expected, predicted)
    assert "job_title" not in per_field
    assert overall == 1.0
