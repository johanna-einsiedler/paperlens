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


# Provider-specific base URLs for the OpenAI-compatible chat-completions
# API.  DeepSeek and Mistral both speak the same wire format as OpenAI, so
# we reuse the openai SDK with a custom base_url instead of pulling in
# their official clients.
_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
_MISTRAL_BASE_URL  = "https://api.mistral.ai/v1"
# Anthropic exposes an OpenAI-compatible endpoint, so we drive Claude
# through the same openai SDK + a custom base_url (like DeepSeek/Mistral)
# rather than pulling in the anthropic package.  The Messages API
# requires an explicit output cap, so anthropic calls always pass
# max_tokens.
_ANTHROPIC_BASE_URL    = "https://api.anthropic.com/v1"
_ANTHROPIC_MAX_TOKENS  = 8192


def get_provider(model: str, base_url: str | None = None) -> str:
    """Infer the provider from the model name prefix, or 'vllm' if a custom base URL is set."""
    if base_url:
        return "vllm"
    if model.startswith("gemini"):
        return "google"
    if model.startswith("deepseek"):
        return "deepseek"
    if model.startswith("claude"):
        return "anthropic"
    # Mistral ships text-only ``mistral-*`` and vision-capable ``pixtral-*``
    # models.  Both live on the same Mistral endpoint.
    if model.startswith(("mistral", "pixtral")):
        return "mistral"
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
        client = openai.OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    if provider == "mistral":
        client = openai.OpenAI(api_key=api_key, base_url=_MISTRAL_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    if provider == "anthropic":
        client = openai.OpenAI(api_key=api_key, base_url=_ANTHROPIC_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    # OpenAI or vLLM (OpenAI-compatible)
    client = _openai_compat_client(api_key, base_url)
    kwargs: dict = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
    }
    # GPT-5 only accepts the default temperature (1); passing any custom
    # value (including 0.0 from our test-connection ping) raises a 400.
    # Vision-extraction paths already omit temperature, so this is the
    # only place we have to guard.
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


_EMPTY_USAGE: dict = {"prompt": 0, "completion": 0, "total": 0}


def _resolved_model(response, provider: str) -> str | None:
    """Extract the dated snapshot the provider's API actually served.

    Users pick an alias (``gpt-5``, ``gpt-4o``, ``gemini-2.5-pro`` ...) but
    every provider resolves that to a dated model id (``gpt-5-2025-09-15``,
    ``gemini-2.5-pro-002`` ...) in the response.  Capturing the resolved
    id makes extractions reproducible — without it, the same alias can
    silently roll forward to a new revision between runs and we'd have
    no audit trail.

    Returns the resolved id (str), or None if the response doesn't carry
    one (older SDKs, malformed responses).  Callers persist it next to
    the user-selected alias on the job row.
    """
    if response is None:
        return None
    if provider == "google":
        # google-genai exposes the served snapshot as ``.model_version``;
        # older revisions used ``.model``.  Try both.
        for attr in ("model_version", "model"):
            v = getattr(response, attr, None)
            if isinstance(v, str) and v:
                return v
        return None
    # OpenAI-compatible (openai, deepseek, mistral, vllm) — ``response.model``
    # is the dated snapshot.  Some self-hosted vLLM servers echo the alias
    # back unchanged; that's fine, we record what the server told us.
    v = getattr(response, "model", None)
    if isinstance(v, str) and v:
        return v
    return None


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
) -> tuple[str, str, dict, str | None]:
    """Run vision-based extraction.
    Returns ``(result_text, finish_reason, token_usage, resolved_model)``
    where ``resolved_model`` is the dated snapshot the provider actually
    served (e.g. ``gpt-5-2025-09-15``) or None if unavailable."""
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
            config=types.GenerateContentConfig(temperature=0),
        )
        text = response.text or ""
        try:
            finish = response.candidates[0].finish_reason.name.lower()
        except Exception:
            finish = "stop"
        return text.strip(), finish, _gemini_usage(response), _resolved_model(response, "google")

    if provider == "mistral":
        # Mistral's Pixtral models accept the same OpenAI-style content
        # blocks (text + image_url).  Plain ``mistral-*`` models will reject
        # the request — the user should pick a ``pixtral-*`` model for
        # vision, which our default model list already does.
        client = openai.OpenAI(api_key=api_key, base_url=_MISTRAL_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_blocks}],
        )
        choice = response.choices[0]
        return (
            choice.message.content.strip(),
            (choice.finish_reason or "stop"),
            _openai_usage(response),
            _resolved_model(response, "mistral"),
        )

    if provider == "anthropic":
        # Claude models accept the same OpenAI-style image_url blocks via
        # the compatibility endpoint.  max_tokens is mandatory.
        client = openai.OpenAI(api_key=api_key, base_url=_ANTHROPIC_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_blocks}],
            max_tokens=_ANTHROPIC_MAX_TOKENS,
        )
        choice = response.choices[0]
        return (
            choice.message.content.strip(),
            (choice.finish_reason or "stop"),
            _openai_usage(response),
            _resolved_model(response, "anthropic"),
        )

    # OpenAI or vLLM (OpenAI-compatible)
    client = _openai_compat_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content_blocks}],
    )
    choice = response.choices[0]
    return (
        choice.message.content.strip(),
        (choice.finish_reason or "stop"),
        _openai_usage(response),
        _resolved_model(response, provider),
    )


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
) -> tuple[str, str, dict, str | None]:
    """Run text-based extraction (PDF text layer as markdown input).

    Works for all providers.  DeepSeek auto-chunks long documents because of its
    smaller context window; OpenAI and Gemini handle long documents in a single call.
    Returns ``(result_text, finish_reason, token_usage, resolved_model)`` —
    ``resolved_model`` is the dated snapshot the provider actually served.
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
            config=types.GenerateContentConfig(temperature=0),
        )
        text = response.text or ""
        try:
            finish = response.candidates[0].finish_reason.name.lower()
        except Exception:
            finish = "stop"
        return text.strip(), finish, _gemini_usage(response), _resolved_model(response, "google")

    # ── Mistral (OpenAI-compatible, 128k context — no chunking) ───────────────
    if provider == "mistral":
        client = openai.OpenAI(api_key=api_key, base_url=_MISTRAL_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
        )
        choice = response.choices[0]
        return (
            choice.message.content.strip(),
            (choice.finish_reason or "stop"),
            _openai_usage(response),
            _resolved_model(response, "mistral"),
        )

    # ── Anthropic / Claude (OpenAI-compatible, large context — no chunking) ───
    if provider == "anthropic":
        client = openai.OpenAI(api_key=api_key, base_url=_ANTHROPIC_BASE_URL)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=_ANTHROPIC_MAX_TOKENS,
        )
        choice = response.choices[0]
        return (
            choice.message.content.strip(),
            (choice.finish_reason or "stop"),
            _openai_usage(response),
            _resolved_model(response, "anthropic"),
        )

    # ── OpenAI or vLLM (OpenAI-compatible) ───────────────────────────────────
    if provider in ("openai", "vllm"):
        client = _openai_compat_client(api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
        )
        choice = response.choices[0]
        return (
            choice.message.content.strip(),
            (choice.finish_reason or "stop"),
            _openai_usage(response),
            _resolved_model(response, provider),
        )

    # ── DeepSeek (with auto-chunking for long documents) ─────────────────────
    import json as _json

    header = f"{prompt}{page_instruction}\n\n"
    full_text = header + markdown_text
    client = openai.OpenAI(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)

    if len(full_text) <= _DEEPSEEK_INPUT_CHAR_LIMIT:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_text}],
            max_tokens=_DEEPSEEK_MAX_OUTPUT_TOKENS,
        )
        choice = response.choices[0]
        return (
            choice.message.content.strip(),
            (choice.finish_reason or "stop"),
            _openai_usage(response),
            _resolved_model(response, "deepseek"),
        )

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
    last_resolved: str | None = None

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
        last_resolved = _resolved_model(response, "deepseek") or last_resolved
        print(f"[extract_with_text] chunk {i + 1}/{len(chunks)}: finish_reason={last_finish!r}", flush=True)

    if len(all_results) == 1:
        return all_results[0], last_finish, total_usage, last_resolved

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
        return merged, last_finish, total_usage, last_resolved

    # Fallback: concatenate raw responses separated by a comment
    return "\n\n".join(all_results), last_finish, total_usage, last_resolved
