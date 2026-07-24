"""FastAPI application: HTTP wiring only.

All actual logic (extraction, classification, parsing, normalization) lives
in app/pdf_extraction.py, app/classification.py, app/document_service.py,
and app/parsers/*. This module's only job is: accept an upload, call the
pipeline, translate internal errors into HTTP responses.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from app import config
from app.document_service import GROUND_TRUTH_RESUMES, parse_document
from app.errors import DocumentParserError
from app.llm_client import get_installed_ollama_models, normalize_ollama_model_name
from app.parsers.handwriting import extract_handwritten_application_fields, merge_handwritten_fields
from app.pdf_extraction import extract_pdf_text

app = FastAPI(
    title="Modular Document Parser",
    description=(
        "Dynamic, schema-agnostic PDF extraction with per-document-type "
        "parser modules (resume, invoice, receipt, contract, report, form) "
        "backed by a local Ollama model."
    ),
    version="5.0.0",
)


@app.get("/")
def home() -> dict[str, Any]:
    return {
        "message": "Modular document parser API is running.",
        "version": "5.0.0",
        "text_model": config.MODEL_NAME,
        "vision_model": config.VISION_MODEL_NAME,
        "handwriting_enabled": config.ENABLE_HANDWRITING,
        "repair_pass_enabled": config.ENABLE_REPAIR_PASS,
        "ground_truth_examples": len(GROUND_TRUTH_RESUMES),
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    ollama_running, installed_models, ollama_error = get_installed_ollama_models()
    installed_normalized = {normalize_ollama_model_name(name) for name in installed_models}

    return {
        "fastapi": "running",
        "ollama": "running" if ollama_running else "not reachable",
        "text_model": config.MODEL_NAME,
        "text_model_installed": normalize_ollama_model_name(config.MODEL_NAME) in installed_normalized,
        "vision_model": config.VISION_MODEL_NAME,
        "vision_model_installed": normalize_ollama_model_name(config.VISION_MODEL_NAME) in installed_normalized,
        "handwriting_enabled": config.ENABLE_HANDWRITING,
        "installed_models": installed_models,
        "ollama_error": ollama_error,
        "ground_truth_loaded": len(GROUND_TRUTH_RESUMES),
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "unnamed.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if file.content_type and file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Uploaded file is not a PDF")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")
    if len(file_bytes) > config.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="PDF exceeds maximum allowed size")

    total_start = time.perf_counter()

    try:
        extraction_start = time.perf_counter()
        extracted_text = extract_pdf_text(file_bytes)
        extraction_seconds = time.perf_counter() - extraction_start

        parsing_start = time.perf_counter()
        predicted_type, parsed_output, validation_issues = parse_document(extracted_text, filename)
        parsing_seconds = time.perf_counter() - parsing_start

        handwriting_seconds = 0.0
        if predicted_type == "Form" and config.ENABLE_HANDWRITING:
            handwriting_start = time.perf_counter()
            handwriting = extract_handwritten_application_fields(file_bytes)
            merge_handwritten_fields(parsed_output, handwriting)
            handwriting_seconds = time.perf_counter() - handwriting_start
    except DocumentParserError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return {
        "filename": filename,
        "file_size_bytes": len(file_bytes),
        "text_length": len(extracted_text),
        "predicted_document_type": predicted_type,
        "parsed_output": parsed_output,
        "validation_issues": validation_issues,
        "performance": {
            "pdf_extraction_seconds": round(extraction_seconds, 3),
            "llm_processing_seconds": round(parsing_seconds, 3),
            "handwriting_processing_seconds": round(handwriting_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_start, 3),
        },
    }
