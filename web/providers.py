"""LLM provider helpers: routing, text generation, and extraction calls."""

from __future__ import annotations

import base64
import re

import openai


def extract_provider_message(exc: Exception) -> str:
    """Pull the human-readable message out of a provider exception.

    Provider SDKs wrap the upstream HTTP error body in their exception class,
    but ``str(exc)`` returns a noisy ``"Error code: 401 - {'error': {...}}"``
    blob.  This helper digs out the inner ``error.message`` when present and
    falls back to the exception's own ``.message`` / ``str()`` otherwise.

    Examples of cleanups:
      ``Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-...'}}``
        → ``Incorrect API key provided: sk-...``
      ``BadRequestError - {'error': {'message': 'max_tokens is too large: ...'}}``
        → ``max_tokens is too large: ...``
    """
    # OpenAI-style exceptions expose the parsed body via .body / .response
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if msg:
                return str(msg)
        elif isinstance(err, str) and err:
            return err
    # Gemini-style: errors carry a .message attribute directly
    msg = getattr(exc, "message", None)
    if msg and msg != str(exc):
        return str(msg)
    # Last resort: stringify the exception
    return str(exc)


def get_provider(model: str, base_url: str | None = None) -> str:
    """Infer the provider from the model name prefix, or 'vllm' if a custom base URL is set."""
    if base_url:
        return "vllm"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("deepseek"):
        return "deepseek"
    return "openai"


def _openai_compat_client(api_key: str, base_url: str | None = None) -> openai.OpenAI:
    """Create an OpenAI-compatible client, optionally with a custom base URL (e.g. vLLM).

    vLLM requires a non-empty API key even when auth is disabled — any string works.
    The base URL is normalised to end in /v1 so the OpenAI SDK appends paths correctly.
    """
    effective_key = api_key or "dummy-key"
    if base_url:
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        return openai.OpenAI(api_key=effective_key, base_url=url)
    return openai.OpenAI(api_key=effective_key)


def generate_text(
    model: str,
    api_key: str,
    prompt: str,
    temperature: float = 0.3,
    base_url: str | None = None,
) -> str:
    """Single-turn text-only generation. Returns the response string."""
    provider = get_provider(model, base_url)

    if provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=4096,
            ),
        )
        return response.text.strip()

    if provider == "deepseek":
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    # OpenAI or vLLM (OpenAI-compatible)
    client = _openai_compat_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


_EMPTY_USAGE: dict = {"prompt": 0, "completion": 0, "total": 0}


def _openai_usage(response) -> dict:
    """Extract token counts from an OpenAI (or OpenAI-compatible) response."""
    u = getattr(response, "usage", None)
    if u is None:
        return _EMPTY_USAGE
    return {
        "prompt":     getattr(u, "prompt_tokens",     0) or 0,
        "completion": getattr(u, "completion_tokens", 0) or 0,
        "total":      getattr(u, "total_tokens",      0) or 0,
    }


def _gemini_usage(response) -> dict:
    """Extract token counts from a Gemini response."""
    try:
        um = response.usage_metadata
        return {
            "prompt":     getattr(um, "prompt_token_count",     0) or 0,
            "completion": getattr(um, "candidates_token_count", 0) or 0,
            "total":      getattr(um, "total_token_count",      0) or 0,
        }
    except Exception:
        return _EMPTY_USAGE


def extract_with_images(
    model: str,
    api_key: str,
    content_blocks: list,         # OpenAI-format content list (text + image_url blocks)
    extraction_images: list[str], # base64 PNGs, used by the Gemini path
    prompt: str,
    page_instruction: str,
    n: int,
    base_url: str | None = None,
) -> tuple[str, str, dict]:
    """Run vision-based extraction. Returns (result_text, finish_reason, token_usage)."""
    provider = get_provider(model, base_url)

    if provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        parts: list = [types.Part.from_text(text=prompt + page_instruction)]
        for i, b64 in enumerate(extraction_images):
            img_bytes = base64.b64decode(b64)
            parts.append(types.Part.from_text(text=f"PDF page {i + 1} of {n}:"))
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=16000),
        )
        text = response.text or ""
        try:
            finish = response.candidates[0].finish_reason.name.lower()
        except Exception:
            finish = "stop"
        return text.strip(), finish, _gemini_usage(response)

    # OpenAI or vLLM (OpenAI-compatible)
    client = _openai_compat_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content_blocks}],
        max_tokens=16000,
    )
    choice = response.choices[0]
    return choice.message.content.strip(), (choice.finish_reason or "stop"), _openai_usage(response)


# DeepSeek's output token limit (max_tokens must be ≤ 8192 per their API)
_DEEPSEEK_MAX_OUTPUT_TOKENS = 8000

# Approximate input token budget: context window minus output reservation.
# deepseek-chat context = 64k tokens; we reserve 8k for output → ~56k for input.
# Using chars as a proxy: ~3.5 chars/token → 56k tokens ≈ 196k chars.
_DEEPSEEK_INPUT_CHAR_LIMIT = 180_000


def _split_markdown_pages(markdown_text: str) -> list[str]:
    """Split page-labelled markdown back into individual page strings."""
    return re.split(r"(?=--- PDF page \d+ of \d+ ---)", markdown_text)


def extract_with_text(
    model: str,
    api_key: str,
    markdown_text: str,
    prompt: str,
    page_instruction: str,
    base_url: str | None = None,
) -> tuple[str, str, dict]:
    """Run text-based extraction (PDF text layer as markdown input).

    Works for all providers.  DeepSeek auto-chunks long documents because of its
    smaller context window; OpenAI and Gemini handle long documents in a single call.
    Returns (result_text, finish_reason, token_usage).
    """
    provider = get_provider(model, base_url)
    full_prompt = f"{prompt}{page_instruction}\n\n{markdown_text}"

    # ── Gemini ────────────────────────────────────────────────────────────────
    if provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=16000),
        )
        text = response.text or ""
        try:
            finish = response.candidates[0].finish_reason.name.lower()
        except Exception:
            finish = "stop"
        return text.strip(), finish, _gemini_usage(response)

    # ── OpenAI or vLLM (OpenAI-compatible) ───────────────────────────────────
    if provider in ("openai", "vllm"):
        client = _openai_compat_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=16000,
        )
        choice = response.choices[0]
        return choice.message.content.strip(), (choice.finish_reason or "stop"), _openai_usage(response)

    # ── DeepSeek (with auto-chunking for long documents) ─────────────────────
    import json as _json

    header = f"{prompt}{page_instruction}\n\n"
    full_text = header + markdown_text
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    if len(full_text) <= _DEEPSEEK_INPUT_CHAR_LIMIT:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_text}],
            max_tokens=_DEEPSEEK_MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0]
        return choice.message.content.strip(), (choice.finish_reason or "stop"), _openai_usage(response)

    # Document is too long — chunk by page sections
    pages = _split_markdown_pages(markdown_text)
    chunks: list[str] = []
    current = ""
    for page_text in pages:
        candidate = current + page_text
        if len(header + candidate) > _DEEPSEEK_INPUT_CHAR_LIMIT and current:
            chunks.append(current)
            current = page_text
        else:
            current = candidate
    if current:
        chunks.append(current)

    print(
        f"[extract_with_text] document too long ({len(full_text):,} chars) — "
        f"splitting into {len(chunks)} chunks",
        flush=True,
    )

    all_results: list[str] = []
    last_finish = "stop"
    total_usage: dict = {"prompt": 0, "completion": 0, "total": 0}

    for i, chunk in enumerate(chunks):
        chunk_instruction = (
            f"{page_instruction}\n\nThis is chunk {i + 1} of {len(chunks)}. "
            "Extract all relevant data from the pages in this chunk only. "
            "Return valid JSON."
        )
        chunk_prompt = f"{prompt}{chunk_instruction}\n\n{chunk}"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": chunk_prompt}],
            max_tokens=_DEEPSEEK_MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0]
        all_results.append(choice.message.content.strip())
        last_finish = choice.finish_reason or "stop"
        u = _openai_usage(response)
        total_usage["prompt"]     += u["prompt"]
        total_usage["completion"] += u["completion"]
        total_usage["total"]      += u["total"]
        print(f"[extract_with_text] chunk {i + 1}/{len(chunks)}: finish_reason={last_finish!r}", flush=True)

    if len(all_results) == 1:
        return all_results[0], last_finish, total_usage

    # Merge JSON arrays/objects from all chunks.
    # Strategy: parse each chunk result, collect all top-level arrays into one.
    merged_samples: list = []
    merge_key: str | None = None

    for raw in all_results:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text)
        try:
            parsed = _json.loads(text)
        except _json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            merged_samples.extend(parsed)
        elif isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, list):
                    merged_samples.extend(v)
                    merge_key = merge_key or k
                    break

    if merged_samples:
        if merge_key:
            merged = _json.dumps({merge_key: merged_samples}, indent=2)
        else:
            merged = _json.dumps(merged_samples, indent=2)
        return merged, last_finish, total_usage

    # Fallback: concatenate raw responses separated by a comment
    return "\n\n".join(all_results), last_finish, total_usage
