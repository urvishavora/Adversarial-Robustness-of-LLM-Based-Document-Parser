"""Central configuration for the document parser.

Every setting is overridable via an environment variable so the same code can
run against a local Ollama install (the default, free/offline path) or a
hosted API, without touching any parser module.
"""

from __future__ import annotations

import os
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --- LLM backend (Ollama by default) ---------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://localhost:11434/api/chat")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", "http://localhost:11434/api/tags")

# Text extraction model. llama3.1 is the most accurate option this project has
# been tuned against; llama3.2:3b is a much faster/lighter alternative.
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1")

# Vision model used only for handwritten form content. Requires a multimodal
# Ollama model, e.g. `ollama pull llama3.2-vision`.
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "llama3.2-vision")
ENABLE_HANDWRITING = _bool_env("ENABLE_HANDWRITING", True)
HANDWRITING_MIN_CONFIDENCE = float(os.getenv("HANDWRITING_MIN_CONFIDENCE", "0.65"))
VISION_TIMEOUT_SECONDS = int(os.getenv("VISION_TIMEOUT_SECONDS", "900"))

# --- Ground truth (used only to build few-shot resume examples) ------------
GROUND_TRUTH_PATH = Path(
    os.getenv("GROUND_TRUTH_PATH", str(Path(__file__).resolve().parent.parent / "ground_truth_resumes.json"))
)

# --- Request/extraction limits ----------------------------------------------
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))
MAX_PROMPT_CHARACTERS = int(os.getenv("MAX_PROMPT_CHARACTERS", "24000"))
MAX_EXAMPLES = int(os.getenv("MAX_EXAMPLES", "2"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "480"))

# Context window requested from Ollama. This is the single biggest lever on
# memory use and therefore on speed: the KV cache scales linearly with it,
# and for an 8B model it costs very roughly 128 KB per token -- about 4 GB
# at 32768 tokens versus 1 GB at 8192. If that pushes the machine into
# swap, generation slows by an order of magnitude.
#
# Nothing here needs a large window. Prompts are capped at
# MAX_PROMPT_CHARACTERS (24000 chars, so ~6000 tokens even in the worst
# case; a typical resume prompt measures ~2800) and generation is capped at
# num_predict (6000). 16384 leaves comfortable headroom above that
# worst-case ~12000 while halving the KV cache versus the previous
# hardcoded 32768. Lower it to 8192 on a memory-constrained machine --
# still above what a resume needs -- but do not set it below
# MAX_PROMPT_CHARACTERS/4 + num_predict, because Ollama silently truncates
# the prompt to fit, which loses document content rather than erroring.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
ENABLE_REPAIR_PASS = _bool_env("ENABLE_REPAIR_PASS", True)

# --- OCR ---------------------------------------------------------------------
# Windows default install path. Override with TESSERACT_CMD if needed; on
# Linux/macOS with tesseract on PATH, this is simply left unset.
_DEFAULT_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", _DEFAULT_WINDOWS_TESSERACT)


def configure_tesseract() -> None:
    """Point pytesseract at a custom tesseract binary if one is configured."""
    import pytesseract

    if os.name == "nt" and Path(TESSERACT_CMD).exists():
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    elif os.getenv("TESSERACT_CMD") and Path(TESSERACT_CMD).exists():
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
