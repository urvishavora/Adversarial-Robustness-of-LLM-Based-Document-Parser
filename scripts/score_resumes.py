#!/usr/bin/env python3
"""Standalone accuracy report for the resumes/ folder against
ground_truth_resumes.json, run directly against your local Ollama --
no pytest required.

Usage (from the project root, with your virtualenv active and
`ollama serve` running with MODEL_NAME pulled):

    python scripts/score_resumes.py
    python scripts/score_resumes.py --resumes-dir path/to/other/resumes --ground-truth path/to/gt.json

Prints a per-file breakdown of any field that scored under 100%, plus the
overall average, so you can see exactly what's still wrong rather than just
a single percentage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.accuracy import score_resume  # noqa: E402
from app.llm_client import get_installed_ollama_models  # noqa: E402
from app.parsers.resume import parse_resume  # noqa: E402
from app.pdf_extraction import extract_pdf_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumes-dir", type=Path, default=PROJECT_ROOT / "resumes")
    parser.add_argument("--ground-truth", type=Path, default=PROJECT_ROOT / "ground_truth_resumes.json")
    args = parser.parse_args()

    reachable, installed_models, error = get_installed_ollama_models()
    if not reachable:
        print(f"Ollama is not reachable: {error}")
        print("Start it with `ollama serve` and make sure MODEL_NAME is pulled, then try again.")
        return 1

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))["resumes"]
    by_filename = {item["file_name"]: item for item in ground_truth}

    pdf_files = sorted(args.resumes_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {args.resumes_dir}")
        return 1

    scores: list[float] = []
    for pdf_path in pdf_files:
        expected = by_filename.get(pdf_path.name)
        if expected is None:
            print(f"[skip] {pdf_path.name}: no ground-truth entry")
            continue

        text = extract_pdf_text(pdf_path.read_bytes())
        predicted, issues = parse_resume(text, pdf_path.name, ground_truth)
        overall, per_field = score_resume(expected, predicted)
        scores.append(overall)

        print(f"\n{pdf_path.name}: {overall:.1%}")
        if issues:
            print(f"  remaining validation issues: {issues}")
        for field, score in sorted(per_field.items()):
            if score < 0.999:
                print(f"    {field}: {score:.1%}")

    if scores:
        print(f"\n{'=' * 50}")
        print(f"Overall average across {len(scores)} resumes: {sum(scores) / len(scores):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
