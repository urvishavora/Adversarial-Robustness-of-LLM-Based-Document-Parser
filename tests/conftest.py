from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def resumes_dir() -> Path:
    return PROJECT_ROOT / "resumes"


@pytest.fixture(scope="session")
def ground_truth_path() -> Path:
    return PROJECT_ROOT / "ground_truth_resumes.json"


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    return PROJECT_ROOT / "sample.pdf"
