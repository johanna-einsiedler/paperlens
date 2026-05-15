"""Server-side helpers that don't belong on a route.

Split out of ``server.py`` to keep that file focused on app init +
HTTP wiring.  Functions here are pure-Python utilities (config
readers, request-side validators, provider-error translators) that
the route handlers compose.

Naming convention: underscore-prefixed names are the "internal use"
default.  They're public enough to be imported across modules but
not part of any external API contract.
"""
from __future__ import annotations

import os
import sys
import traceback

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from providers import extract_provider_message


# ── Environment-driven config readers ─────────────────────────────────────────
# Read at request time (not import time) so a deployment can tune them
# without a code change.  Defaults chosen for a typical OpenAI Tier-1
# key + 1 GB server RAM.

def _max_batch_papers() -> int:
    """Per-batch ceiling on number of papers.  Default: 20."""
    try:
        return max(1, int(os.environ.get("PAPERLENS_MAX_BATCH_PAPERS", "20")))
    except ValueError:
        return 20


def _max_pdf_bytes() -> int:
    """Per-file ceiling — should match the frontend's 50 MB enforcement."""
    try:
        return max(1, int(os.environ.get("PAPERLENS_MAX_PDF_BYTES", str(50 * 1024 * 1024))))
    except ValueError:
        return 50 * 1024 * 1024


def _maseminer_only() -> bool:
    """When truthy in the environment, the app boots in MASEMiner-only mode:
    ``/`` redirects to ``/maseminer``, the page title swaps to "MASEMiner",
    and the "Switch back to generic PaperLens" affordances are hidden.
    Used by the public ``maseminer`` local-run distribution; the hosted
    PaperLens app leaves this unset and serves both audiences."""
    return os.environ.get("PAPERLENS_MASEMINER_ONLY", "").strip().lower() in ("1", "true", "yes")


# ── Request-side validators ───────────────────────────────────────────────────

def _prompt_has_evidence_schema(prompt: str) -> bool:
    """≥3 of (evidence, snippet, page, source) → likely already requests evidence."""
    p = prompt.lower()
    return sum(tok in p for tok in ("evidence", "snippet", "page", "source")) >= 3


def _ascii_only(value: str, label: str) -> None:
    """Raise HTTPException 400 if a credential / URL contains non-ASCII characters.

    HTTP headers (Authorization) and URLs must be ASCII per the HTTP/1.1 spec.
    A common failure mode is users pasting an API key from a rich-text document
    that has auto-replaced a hyphen with an en-dash (U+2013) or em-dash (U+2014).
    """
    if not value:
        return
    try:
        value.encode("ascii")
    except UnicodeEncodeError as e:
        bad = value[e.start]
        raise HTTPException(
            status_code=400,
            detail=(
                f"{label} contains a non-ASCII character "
                f"(U+{ord(bad):04X} {bad!r}) at position {e.start}. "
                "This often happens when pasting from a document where hyphens "
                "have been auto-replaced with en/em-dashes. Please re-type or "
                "re-paste the value as plain text."
            ),
        )


# ── Provider-error translation ────────────────────────────────────────────────

def _provider_error_response(exc: Exception) -> JSONResponse:
    """Translate provider-side exceptions into clean user-facing JSON errors.

    The provider's actual reason (e.g. "Incorrect API key provided: sk-...",
    "Image is too large", "Rate limit reached for ...") is preserved verbatim
    so the user sees what's wrong, not a generic placeholder.
    """
    import openai
    # Always log the full traceback to stderr for operator-side diagnosis.
    try:
        traceback.print_exc(file=sys.stderr)
    except UnicodeEncodeError:
        sys.stderr.buffer.write(traceback.format_exc().encode("utf-8", errors="replace"))
    sys.stderr.flush()

    msg = extract_provider_message(exc)

    if isinstance(exc, openai.AuthenticationError):
        return JSONResponse(
            {"error": f"Authentication failed: {msg}"}, status_code=401,
        )
    if isinstance(exc, openai.RateLimitError):
        return JSONResponse(
            {"error": f"Rate limit / quota error: {msg}"}, status_code=429,
        )
    if isinstance(exc, openai.NotFoundError):
        return JSONResponse(
            {"error": f"Not found: {msg}"}, status_code=400,
        )
    if isinstance(exc, openai.BadRequestError):
        return JSONResponse(
            {"error": f"Request rejected by provider: {msg}"}, status_code=400,
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return JSONResponse(
            {"error": f"Permission denied: {msg}"}, status_code=403,
        )
    if isinstance(exc, openai.APIConnectionError):
        return JSONResponse(
            {"error": f"Could not reach the provider: {msg}"}, status_code=502,
        )
    if isinstance(exc, openai.APITimeoutError):
        return JSONResponse(
            {"error": f"Provider timed out: {msg}"}, status_code=504,
        )
    if isinstance(exc, openai.APIStatusError):
        # Catch-all for any other HTTP-level provider error (5xx, etc.)
        status = getattr(exc, "status_code", 502) or 502
        return JSONResponse(
            {"error": f"Provider error ({status}): {msg}"}, status_code=int(status),
        )
    if isinstance(exc, UnicodeEncodeError):
        return JSONResponse(
            {"error": "Server encoding error — set PYTHONIOENCODING=utf-8 or PYTHONUTF8=1 "
                      "in the environment. Original: " + msg},
            status_code=500,
        )
    return JSONResponse({"error": msg}, status_code=500)
