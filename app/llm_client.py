"""Ollama client: text generation, vision (handwriting), and health checks.

This is the only module that talks to the LLM backend over HTTP. Every
parser module calls through `call_ollama` / `call_ollama_vision` so swapping
backends later (e.g. a hosted API) only means changing this one file.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from app import config
from app.errors import LLMBackendError
from app.json_utils import clean_llm_json

logger = logging.getLogger("document_parser")


def call_ollama(prompt: str, *, max_retries: int | None = None, num_predict: int = 6000) -> dict[str, Any]:
    """Call the text-generation endpoint and return a parsed JSON object.

    Retries transient failures (timeouts, connection resets) up to
    `max_retries` times with a short linear backoff before giving up. A
    non-transient failure (bad JSON shape, HTTP 4xx) is not retried.
    """
    retries = config.OLLAMA_MAX_RETRIES if max_retries is None else max_retries
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                config.OLLAMA_URL,
                json={
                    "model": config.MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": "10m",
                    "options": {
                        "temperature": 0,
                        "num_predict": num_predict,
                        "num_ctx": 32768,
                        "repeat_penalty": 1.05,
                    },
                },
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            raw_output = payload.get("response")
            if not isinstance(raw_output, str):
                raise ValueError("Ollama response does not contain a string response")
            return clean_llm_json(raw_output)

        except requests.exceptions.ConnectionError as exc:
            last_exc = LLMBackendError(
                f"Cannot connect to Ollama at {config.OLLAMA_URL}. Is `ollama serve` "
                f"running and is `{config.MODEL_NAME}` pulled?",
                status_code=503,
            )
        except requests.exceptions.Timeout as exc:
            last_exc = LLMBackendError("Ollama request timed out.", status_code=504)
        except requests.exceptions.HTTPError as exc:
            # HTTP errors (e.g. model not found -> 404) are not transient.
            raise LLMBackendError(f"Ollama HTTP error: {exc}", status_code=502) from exc
        except requests.exceptions.RequestException as exc:
            last_exc = LLMBackendError(f"Ollama request failed: {exc}", status_code=502)
        except (ValueError, json.JSONDecodeError) as exc:
            last_exc = LLMBackendError(f"Invalid JSON from Ollama: {exc}", status_code=502)

        if attempt < retries:
            logger.warning(
                "Ollama call failed (attempt %d/%d): %s -- retrying",
                attempt + 1,
                retries + 1,
                last_exc,
            )
            time.sleep(1.5 * (attempt + 1))

    assert last_exc is not None
    raise last_exc


def call_ollama_vision(image_base64: str, prompt: str) -> dict[str, Any]:
    """Call the chat/vision endpoint with one image, streaming the response.

    Streaming (rather than `stream=False`) avoids a false read-timeout while
    a vision model is still generating: each received chunk resets the
    socket's read timer, and the text chunks are joined before JSON parsing.
    """
    request_payload = {
        "model": config.VISION_MODEL_NAME,
        "stream": True,
        "format": "json",
        "keep_alive": "15m",
        "messages": [
            {"role": "user", "content": prompt, "images": [image_base64]},
        ],
        "options": {
            "temperature": 0,
            "num_predict": 800,
            "num_ctx": 16384,
            "repeat_penalty": 1.05,
        },
    }

    try:
        response = requests.post(
            config.OLLAMA_CHAT_URL,
            json=request_payload,
            stream=True,
            timeout=(15, config.VISION_TIMEOUT_SECONDS),
        )
    except requests.RequestException as exc:
        raise LLMBackendError(
            f"Unable to connect to Ollama vision endpoint: {exc}", status_code=503
        ) from exc

    if not response.ok:
        detail = response.text.strip()
        raise LLMBackendError(
            f"Ollama vision request failed with HTTP {response.status_code}: {detail[:1500]}",
            status_code=502,
        )

    content_parts: list[str] = []
    final_error: str | None = None

    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed Ollama stream chunk: %r", line[:300])
                continue

            if chunk.get("error"):
                final_error = str(chunk["error"])
                break

            message = chunk.get("message")
            if isinstance(message, dict):
                piece = message.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)

            if chunk.get("done") is True:
                break
    finally:
        response.close()

    if final_error:
        raise LLMBackendError(f"Ollama vision stream failed: {final_error}", status_code=502)

    raw_output = "".join(content_parts).strip()
    if not raw_output:
        raise LLMBackendError("Vision model returned no generated content.", status_code=502)

    return clean_llm_json(raw_output)


def normalize_ollama_model_name(name: str) -> str:
    """Normalize model names so `name` and `name:latest` compare equally."""
    return name.strip().removesuffix(":latest")


def get_installed_ollama_models() -> tuple[bool, list[str], str | None]:
    """Return (reachable, installed_model_names, error_message_if_any)."""
    try:
        response = requests.get(config.OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = [
            str(item.get("name", "")).strip()
            for item in payload.get("models", [])
            if isinstance(item, dict) and item.get("name")
        ]
        return True, models, None
    except (requests.exceptions.RequestException, ValueError) as exc:
        return False, [], str(exc)
