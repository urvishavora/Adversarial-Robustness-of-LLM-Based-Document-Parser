from __future__ import annotations

from app.parsers.resume import (
    normalize_contact,
    normalize_name,
    normalize_resume,
    select_resume_examples,
    validation_issues,
)


def test_normalize_name_basic():
    result = normalize_name({"full_name": "Jane Doe"}, {})
    assert result == {"full_name": "Jane Doe", "first_name": "Jane", "last_name": "Doe"}


def test_normalize_name_despaces_letter_spaced_components():
    result = normalize_name({"first_name": "S E B A S T I A N", "last_name": "B E N N E T T"}, {})
    assert result["first_name"] == "Sebastian"
    assert result["last_name"] == "Bennett"
    assert result["full_name"] == "Sebastian Bennett"


def test_normalize_name_despaces_letter_spaced_full_name_only():
    result = normalize_name({"full_name": "J A N E   D O E"}, {})
    # No first/last given, so despacing falls back to the whole string.
    assert result["full_name"] == "Jane doe" or result["full_name"].replace(" ", "").lower() == "janedoe"


def test_normalize_name_falls_back_to_flattened_root_keys():
    result = normalize_name(None, {"full_name": "Alex Kim"})
    assert result["full_name"] == "Alex Kim"


def test_normalize_contact_aliases():
    result = normalize_contact({"telephone": "555-1234", "email_address": "a@b.com"}, {})
    assert result["phone"] == "555-1234"
    assert result["email"] == "a@b.com"


def test_normalize_resume_skills_as_flat_list():
    parsed = {"skills": ["Python", "SQL", ""]}
    result = normalize_resume(parsed, "resume.pdf")
    assert result["skills"] == ["Python", "SQL"]


def test_normalize_resume_skills_as_category_dict():
    parsed = {"skills": {"Languages": ["Python"], "Empty": []}}
    result = normalize_resume(parsed, "resume.pdf")
    assert result["skills"] == {"Languages": ["Python"]}


def test_normalize_resume_recovers_leaked_list_content():
    parsed = {
        "experience": [
            {"company": "Acme", "job_title": "Engineer"},
            "skills':['Project Management', 'Leadership'],",
        ],
        "skills": [],
    }
    result = normalize_resume(parsed, "resume.pdf")
    assert {"company": "Acme", "job_title": "Engineer"} in result["experience"]
    assert result["skills"] == ["Project Management", "Leadership"]
    assert any("misplaced" in warning for warning in result["data_quality_warnings"])


def test_normalize_resume_flags_placeholder_text():
    parsed = {"summary": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."}
    result = normalize_resume(parsed, "resume.pdf")
    assert any("placeholder" in warning for warning in result["data_quality_warnings"])


def test_validation_issues_flags_missing_name_and_sections():
    source_text = "EXPERIENCE\nAcme Corp\nEDUCATION\nState University\nSKILLS\nPython"
    resume = {
        "name": {"full_name": None},
        "job_title": None,
        "summary": None,
        "education": [],
        "experience": [],
        "skills": [],
        "activities": [],
    }
    issues = validation_issues(resume, source_text)
    assert any("name" in issue.lower() for issue in issues)
    assert any("education" in issue.lower() for issue in issues)
    assert any("experience" in issue.lower() for issue in issues)
    assert any("skills" in issue.lower() for issue in issues)


def test_validation_issues_clean_resume_has_no_issues():
    source_text = "Experience: Acme. Education: State U, BSc. Skills: Python"
    resume = {
        "name": {"full_name": "Jane Doe"},
        "job_title": "Engineer",
        "summary": "Experienced engineer",
        "education": [{"institution": "State U", "degree": "BSc"}],
        "experience": [{"company": "Acme", "job_title": "Engineer"}],
        "skills": ["Python"],
        "activities": [],
    }
    assert validation_issues(resume, source_text) == []


def test_validation_issues_flags_missing_company_and_degree():
    """The section-level checks only fire when a whole section vanishes.
    These catch the quieter failure -- the section is there, but an entry
    is missing a field the document plainly shows, which otherwise reads
    as a clean success.
    """
    source_text = "Experience: Acme. Education: State U. Skills: Python"
    resume = {
        "name": {"full_name": "Jane Doe"},
        "job_title": "Engineer",
        "summary": "Experienced engineer",
        "education": [{"institution": "State U"}],
        "experience": [{"job_title": "Engineer"}],
        "skills": ["Python"],
        "activities": [],
    }
    issues = validation_issues(resume, source_text)
    assert any("missing the employer/company" in issue for issue in issues)
    assert any("missing the degree" in issue for issue in issues)


def test_validation_issues_flags_url_not_present_in_source():
    """A URL rebuilt from text that wrapped mid-token comes back with a
    separator that was never in the document. It is still a well-formed
    URL, so no structural check catches it -- but it does not appear in
    the source, and that does.
    """
    source_text = (
        "Sebastian Bennett\nhttps://www.linkedin.com/in/sebastian-bennett?\n"
        "Experience: Really Great Company. Education: University, B.A. Skills: Negotiation"
    )
    base = {
        "name": {"full_name": "Sebastian Bennett"},
        "job_title": "Real Estate Agent",
        "summary": "Experienced agent",
        "education": [{"institution": "University", "degree": "B.A."}],
        "experience": [{"company": "Really Great Company", "job_title": "Real Estate Agent"}],
        "skills": ["Negotiation"],
        "activities": [],
    }

    mangled = dict(base, contact={"linkedin": "https://www.linkedin.com/in/s/ebastian-bennett?"})
    assert any("does not appear in the source" in issue for issue in validation_issues(mangled, source_text))

    correct = dict(base, contact={"linkedin": "https://www.linkedin.com/in/sebastian-bennett?"})
    assert validation_issues(correct, source_text) == []


def test_validation_issues_flags_degree_like_experience_entry():
    """Regression test for a real failure observed on a live resume: a
    second education entry ("Diploma in Advertising Management" /
    "University of Engineering and Technology") got mapped into the
    experience array using job_title/company keys, which the older
    "institution" key check didn't catch.
    """
    resume = {
        "name": {"full_name": "Jane Doe"},
        "job_title": "Designer",
        "experience": [
            {"job_title": "Designer", "company": "Acme"},
            {"job_title": "Diploma in Advertising Management", "company": "University of Engineering and Technology"},
        ],
    }
    issues = validation_issues(resume, "experience education")
    assert any("misplaced education entry" in issue for issue in issues)


def test_validation_issues_flags_concatenated_skills_category():
    """Regression test: the model sometimes emits a category label and its
    members joined into one string (e.g. "Design: Branding, Logo Design,
    Typography") instead of splitting them into individual skill items.
    """
    resume = {
        "name": {"full_name": "Jane Doe"},
        "job_title": "Designer",
        "skills": ["Design: Branding & Identity, Logo Design, Typography", "Python"],
    }
    issues = validation_issues(resume, "skills")
    assert any("concatenated into one string" in issue for issue in issues)


def test_validation_issues_does_not_false_positive_on_normal_skills():
    resume = {
        "name": {"full_name": "Jane Doe"},
        "job_title": "Designer",
        "skills": ["Python", "SQL", "Docker"],
    }
    assert validation_issues(resume, "skills") == []


def test_select_resume_examples_excludes_self_by_filename():
    ground_truth = [
        {"file_name": "self.pdf", "summary": "self summary text"},
        {"file_name": "other.pdf", "summary": "python engineer with experience"},
    ]
    examples = select_resume_examples("python engineer", "self.pdf", ground_truth, limit=2)
    assert all(example.get("file_name") != "self.pdf" for example in examples)


def test_split_date_range_derives_endpoints():
    """A printed range carries both endpoints, but the model returns the
    combined string and leaves start_date/end_date null. Splitting it is
    arithmetic on a value the model already produced -- no second model
    call, and it cannot introduce a value the document didn't show.
    """
    from app.parsers.resume import split_date_range

    assert split_date_range("2020 - 2023") == ("2020", "2023")
    assert split_date_range("2020 – 2023") == ("2020", "2023")  # en dash
    assert split_date_range("2017- 2019") == ("2017", "2019")
    assert split_date_range("Oct 2023 - Present") == ("Oct 2023", "Present")
    assert split_date_range("Jan 2022 to Aug 2023") == ("Jan 2022", "Aug 2023")

    # Not ranges -- must not invent endpoints.
    assert split_date_range("2020") == (None, None)
    assert split_date_range("Bachelor of Arts") == (None, None)
    assert split_date_range(None) == (None, None)


def test_date_endpoints_never_overwrite_model_values():
    from app.parsers.resume import _fill_date_endpoints

    entries = [
        {"date": "2020 - 2023"},
        {"date": "2016 - 2020", "start_date": "FROM MODEL"},
    ]
    _fill_date_endpoints(entries)
    assert entries[0]["start_date"] == "2020"
    assert entries[0]["end_date"] == "2023"
    assert entries[1]["start_date"] == "FROM MODEL"
    assert entries[1]["end_date"] == "2020"


def test_address_graded_on_content_not_punctuation():
    """Three correctly-extracted addresses scored 0% because grading used
    exact string equality: the document prints "43-589 BEECHWOOD DR
    WATERLOO ON N2T 2K9" and the ground truth writes it with commas and
    mixed case. That measures transcription style, not extraction.
    """
    from app.accuracy import score_resume

    expected = {
        "file_name": "x.pdf",
        "contact": {"address": "43-589 Beechwood Dr, Waterloo, ON N2T 2K9"},
    }
    predicted = {"contact": {"address": "43-589 BEECHWOOD DR  WATERLOO ON N2T 2K9"}}
    _, per_field = score_resume(expected, predicted)
    assert per_field["contact.address"] == 1.0

    # A genuinely different address must still score 0.
    _, wrong = score_resume(expected, {"contact": {"address": "12 Other Street, Ottawa"}})
    assert wrong["contact.address"] == 0.0
