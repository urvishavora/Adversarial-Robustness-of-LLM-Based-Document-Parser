#!/usr/bin/env python3
"""Parse a single PDF through the full pipeline and print the JSON result,
without needing the FastAPI server running. Handy for quickly testing any
new sample file (invoice, receipt, contract, report, resume, or application
form) against your local Ollama.

Usage:
    python scripts/parse_pdf.py path/to/file.pdf
    python scripts/parse_pdf.py path/to/file.pdf --pretty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.document_service import parse_document  # noqa: E402
from app.errors import DocumentParserError  # noqa: E402
from app.parsers.handwriting import extract_handwritten_application_fields, merge_handwritten_fields  # noqa: E402
from app.pdf_extraction import extract_pdf_text  # noqa: E402
from app import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--no-handwriting", action="store_true", help="Skip the handwriting vision layer")
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"File not found: {args.pdf_path}")
        return 1

    try:
        file_bytes = args.pdf_path.read_bytes()
        text = extract_pdf_text(file_bytes)
        predicted_type, result, issues = parse_document(text, args.pdf_path.name)

        if predicted_type == "Form" and config.ENABLE_HANDWRITING and not args.no_handwriting:
            handwriting = extract_handwritten_application_fields(file_bytes)
            merge_handwritten_fields(result, handwriting)
    except DocumentParserError as exc:
        print(f"Error: {exc.message}")
        return 1

    print(f"Predicted type: {predicted_type}")
    if issues:
        print(f"Validation issues: {issues}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
