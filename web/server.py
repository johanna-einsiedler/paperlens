"""FastAPI backend for PaperLens.

Routes
------
GET  /                               static index.html
GET  /static/{path}                  static asset
POST /api/generate-prompt            AI-generated extraction/labeling prompt
POST /api/check-evidence-schema      heuristic check for prompt warnings
POST /api/adapt-prompt               LLM rewrite that injects evidence schema
POST /api/extract                    enqueue an extraction job, returns job_id
GET  /api/jobs/{job_id}              poll job status + result (no page images)
GET  /api/jobs/{job_id}/pages        page images for a finished job
POST /api/pages                      ad-hoc PDF→highlighted images (review flow)

Concurrency
-----------
Extraction runs as an asyncio task launched from /api/extract.  The route
returns immediately with a job id; the frontend polls /api/jobs/{id} until
status==done.  This way LLM calls (30–120 s) don't tie up an HTTP request and
many users can extract concurrently on a single uvicorn worker.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Force UTF-8 stdio so logging never crashes on non-ASCII content (e.g. en-dashes
# in user prompts).  Some hosting environments default to ASCII / C locale.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# Load web/.env (gitignored) before any module-level env reads.  The
# loader is its own module so standalone diagnostics can ``import
# dotenv_local; dotenv_local.load()`` without booting the FastAPI stack.
import dotenv_local
dotenv_local.load()

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import datasets as datasets_mod
import db
import donor
import jobs as jobs_mod
import zenodo
import presets_loader
from _helpers import (
    _ascii_only,
    _max_batch_papers,
    _max_pdf_bytes,
    _maseminer_only,
    _prompt_has_evidence_schema,
    _provider_error_response,
)
from pdf_utils import extract_evidence_snippets, pdf_to_highlighted_images
from prompt_builder import EVIDENCE_APPENDIX, build_meta_prompt
from providers import generate_text
from schemas import (
    AdaptPromptIn,
    BatchEmailIn,
    BuildPresetPromptIn,
    CheckSchemaIn,
    DonateIn,
    GeneratePromptIn,
    TestConnectionIn,
    VerifyPasswordIn,
)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    db.init()
    # Best-effort cleanup of week-old jobs on each boot
    try:
        db.cleanup_old_jobs(7 * 24 * 3600)
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="MetaPaperLens", docs_url=None, redoc_url=None, lifespan=_lifespan)


# ── Static serving ────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma":        "no-cache",
    "Expires":       "0",
}


@app.get("/", response_model=None)
def index() -> FileResponse | RedirectResponse:
    # In MASEMiner-only deployments the root URL bounces straight to the
    # dedicated landing — local researchers shouldn't see PaperLens chrome
    # they don't need.  307 (not 301) so a future rebuild can flip the
    # flag back off without browsers caching the redirect forever.
    if _maseminer_only():
        return RedirectResponse(url="/maseminer", status_code=307)
    # Prevent the browser from serving a stale index that points to an outdated
    # bundle.  Static assets are still mounted above with their own headers.
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE_HEADERS)


@app.get("/maseminer")
def maseminer_landing() -> FileResponse:
    """Dedicated entry point for MASEMiner.  Serves the same index.html;
    the frontend detects ``window.location.pathname === '/maseminer'`` and
    shows a custom hero landing instead of the generic mode cards."""
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE_HEADERS)


@app.get("/masemminer")
def masemminer_legacy() -> RedirectResponse:
    """Backwards-compat redirect — the canonical URL is now ``/maseminer``
    (single ``M`` to match the brand name MASEMiner).  Permanent so any
    bookmarked links and the old fly.io rewrites keep working."""
    return RedirectResponse(url="/maseminer", status_code=301)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Disable caching on /static/* so JS/CSS updates always reach the browser
    on a normal reload (no need for hard-refresh after each deploy)."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        for k, v in _NO_CACHE_HEADERS.items():
            response.headers[k] = v
    return response


# Pydantic request schemas live in ``schemas.py``; helpers
# (env-var readers, request validators, provider-error translation)
# live in ``_helpers.py``.  Both are imported at the top of this file.


# ── /api/generate-prompt ──────────────────────────────────────────────────────

@app.post("/api/generate-prompt")
async def generate_prompt(payload: GeneratePromptIn) -> Any:
    api_key  = payload.api_key.strip()
    question = payload.question.strip()
    context  = payload.context.strip()
    base_url = (payload.base_url or "").strip() or None

    if not api_key and not base_url:
        raise HTTPException(status_code=400, detail="API key is required.")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if payload.mode not in ("extraction", "labeling", "summarize"):
        raise HTTPException(status_code=400, detail="Invalid mode.")
    _ascii_only(api_key,  "API key")
    _ascii_only(base_url, "Server URL")

    meta_prompt = build_meta_prompt(payload.mode, question, context)
    try:
        generated = await asyncio.to_thread(
            generate_text, payload.model, api_key, meta_prompt, 0.3, base_url
        )
        return {"prompt": generated + EVIDENCE_APPENDIX, "model_used": payload.model}
    except Exception as exc:  # noqa: BLE001
        return _provider_error_response(exc)


# ── /api/check-evidence-schema ────────────────────────────────────────────────

@app.post("/api/check-evidence-schema")
def check_evidence_schema(payload: CheckSchemaIn) -> dict:
    return {"has_evidence_schema": _prompt_has_evidence_schema(payload.prompt)}


# ── /api/config ──────────────────────────────────────────────────────────────

@app.get("/api/config")
def get_config() -> dict:
    """Server-side limits + branding the frontend should respect.  Read at
    page load so the UI can render 'up to N papers per batch', refuse
    oversize uploads early, and swap the app title in MASEMiner-only
    deployments."""
    masem_only = _maseminer_only()
    return {
        "max_batch_papers": _max_batch_papers(),
        "max_pdf_bytes":    _max_pdf_bytes(),
        "maseminer_only":   masem_only,
        "app_title":        "MASEMiner" if masem_only else "MetaPaperLens",
        "app_tagline": (
            "Local extraction of factor loadings, correlations, and study metadata from PDFs."
            if masem_only else
            "AI-powered data extraction and labeling for academic papers"
        ),
        # Donation flow is feature-flagged; the frontend hides the modal
        # when donate.enabled is false.  ``live`` distinguishes dry-run from
        # the real-PR mode so the modal's success copy is honest.
        # ``zenodo`` is best-effort — when the token isn't set the donor
        # silently skips the Zenodo step, the PR still works.
        "donate": {
            "enabled": donor.is_enabled(),
            "live":    donor.is_live(),
            "zenodo":  zenodo.is_configured(),
        },
    }


# ── /api/donate ──────────────────────────────────────────────────────────────

# Rate limit: max donations per IP within the window below.  Configurable
# via the ``PAPERLENS_DONATE_RATE_LIMIT`` env var; a value of 0 disables
# the limit entirely (useful while bootstrapping and during testing).
# Loopback IPs (127.0.0.1 / ::1) always bypass the check — when you're
# testing on localhost you should not be blocked by a check designed for
# the public internet.
_DONATE_RATE_WINDOW_SEC = 24 * 3600


def _donate_rate_limit() -> int:
    """``PAPERLENS_DONATE_RATE_LIMIT`` parsed as a non-negative int.
    Returns ``3`` (the default) on missing / malformed values, ``0``
    when the operator explicitly disabled the limit."""
    raw = os.environ.get("PAPERLENS_DONATE_RATE_LIMIT", "")
    if not raw:
        return 3
    try:
        return max(0, int(raw))
    except ValueError:
        return 3


def _client_ip_from_request(request: Request) -> str:
    """Best-effort client IP.  Fly sets X-Forwarded-For; locally we fall
    back to the socket peer.  Pepper-hashed before storage so the raw IP
    never lives in the DB."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        # Left-most entry is the original client per the standard.
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def _is_loopback_ip(ip: str) -> bool:
    """Treat 127.x.x.x / ::1 as ``dev`` and skip the rate limit there.
    Avoids the standard library's ``ipaddress`` for a trivial check
    because the strings are well-known."""
    return ip in {"127.0.0.1", "::1", "localhost"} or ip.startswith("127.")


@app.post("/api/donate")
def donate_dataset(payload: DonateIn, request: Request) -> dict:
    """Build a citable dataset bundle from a finished batch and (in live mode)
    open a PR against the curated public repo.

    Feature-flagged behind ``PAPERLENS_DONATE_ENABLED``.  When disabled the
    endpoint 404s so it isn't an attack surface in deployments that haven't
    opted into the feature.  ``PAPERLENS_DONATE_LIVE`` controls dry-run vs.
    real PR creation (default: dry-run, writes the bundle to /tmp).
    """
    if not donor.is_enabled():
        raise HTTPException(status_code=404, detail="Donation flow is not enabled on this server.")

    # Consent gate — both checkboxes are required by the modal; reject any
    # request that arrives without them (defensive against tampered clients).
    if not (payload.consents.sharing_rights and payload.consents.license_cc_by_4):
        raise HTTPException(status_code=400, detail="Both consent checkboxes are required.")

    # Extension flow: title / description / visibility come from the existing
    # dataset, so they're not required on the incoming payload.  Only the
    # extends slug + attribution + consent matter.  Validate the slug shape
    # defensively (the path-traversal guard duplicates datasets._is_safe_slug
    # but the donate endpoint can't import that private helper).
    is_extension = bool(payload.extends)
    if is_extension:
        if not payload.extends or len(payload.extends) > 120 or "/" in payload.extends:
            raise HTTPException(status_code=400, detail="Invalid 'extends' dataset id.")
        title = ""   # ignored by donor when extends is set
    else:
        title = (payload.title or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Dataset title is required.")
        if len(title) > 200:
            raise HTTPException(status_code=400, detail="Dataset title must be 200 characters or fewer.")

        if payload.visibility.mode not in {"public", "gated"}:
            raise HTTPException(status_code=400, detail="Invalid visibility mode.")
        if payload.visibility.mode == "gated":
            pw = payload.visibility.password or ""
            if len(pw) < 8:
                raise HTTPException(status_code=400, detail="Gated datasets need a password of at least 8 characters.")

    if payload.attribution.mode not in {"anonymous", "attributed"}:
        raise HTTPException(status_code=400, detail="Invalid attribution mode.")
    if payload.attribution.mode == "attributed" and not payload.attribution.name.strip():
        raise HTTPException(status_code=400, detail="Attributed donations need a donor name.")

    # Per-IP rate-limit: peppered SHA-256, never stores the raw IP.
    # Loopback bypass — local dev iteration must not be rate-limited.
    # ``PAPERLENS_DONATE_RATE_LIMIT=0`` disables the check entirely.
    client_ip = _client_ip_from_request(request)
    try:
        ip_hash = donor.hash_ip(client_ip)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    limit = _donate_rate_limit()
    if limit > 0 and not _is_loopback_ip(client_ip):
        recent = db.count_donations_by_ip(ip_hash, _DONATE_RATE_WINDOW_SEC)
        if recent >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Donation rate limit reached ({limit} per 24h).  Try again tomorrow.",
            )

    req = donor.DonationRequest(
        batch_id                = payload.batch_id,
        title                   = title,
        description             = (payload.description or "").strip(),
        attribution_mode        = payload.attribution.mode,
        attribution_name        = payload.attribution.name.strip(),
        attribution_affiliation = payload.attribution.affiliation.strip(),
        visibility              = payload.visibility.mode,
        password                = payload.visibility.password,
        extends_dataset_id      = payload.extends,
        human_verified          = bool(payload.verification.human_verified),
        verification_notes      = (payload.verification.notes or "").strip()[:500],
    )
    try:
        return donor.donate(req, ip_hash=ip_hash)
    except ValueError as exc:
        # Validation surfaced from build_bundle (e.g. no done jobs).
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Misconfiguration (missing env vars in live mode).
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/datasets — community-donated datasets in metalens-datasets ─────────

@app.get("/api/datasets")
def list_datasets(refresh: int = 0) -> dict:
    """Listing of public datasets in the curated GitHub repo.  Powers
    the "Extend an existing dataset" picker on the landing screen.

    ``?refresh=1`` bypasses the in-process cache and forces a fresh
    fetch from GitHub — the picker UI exposes this as a Refresh
    button so users who just merged a PR can see it without waiting
    for the TTL to expire.

    Password hashes are stripped server-side — gated-ness is signalled
    only via a derived ``gated`` boolean per record so the frontend can
    render a 🔒 indicator without ever holding the hash itself.

    Response shape:
        {
          "datasets": [
            {
              "id":              "ncs-18-2026-06",
              "title":           "...",
              "description":     "...",
              "donor":           {...},
              "visibility":      "public" | "gated",
              "gated":           false,
              "schema_version":  "masem-v3",
              "github_url":      "https://github.com/owner/repo/tree/main/datasets/<id>",
              "zenodo_doi":      "10.5281/zenodo.123" | null,
              "created_at":      "2026-06-03T17:43:20Z",
              "paper_count":     2,
              "model_used":      ["gpt-4o"]
            },
            ...
          ],
          "fetched_at":    "2026-06-04T...",
          "cache_age_sec": 0
        }

    Cached in-process for 1 hour; donor.donate() invalidates the cache
    after a successful PR so freshly-merged datasets surface
    immediately on the next request.
    """
    try:
        return datasets_mod.list_datasets(force_refresh=bool(refresh))
    except RuntimeError as exc:
        # PAPERLENS_GH_REPO unset — surface a 503 so the frontend can
        # show a "feature not available here" message rather than a
        # generic 500.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        # Network / GitHub-side failure — degrade to an empty list
        # rather than blocking the rest of the page.
        return {"datasets": [], "fetched_at": None, "cache_age_sec": 0,
                "error": str(exc)[:200]}


@app.get("/api/datasets/{dataset_id}/full")
def get_dataset_full(dataset_id: str) -> dict:
    """Full dataset payload for the extend flow: metadata (without
    password_hash) + the verbatim prompt body.  The UI calls this
    after the user has picked a dataset (and cleared the password
    gate if gated) so it can pre-load state.generatedPrompt and the
    suggested model before sending the user to step 2.

    The underlying repo is public on GitHub so the prompt is already
    public information — this endpoint doesn't add a password gate
    of its own; the gate is on the EXTENSION submission side."""
    if not dataset_id or len(dataset_id) > 120:
        raise HTTPException(status_code=400, detail="Invalid dataset id.")
    payload = datasets_mod.get_dataset_full(dataset_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return payload


@app.post("/api/datasets/{dataset_id}/verify-password")
def verify_dataset_password(
    dataset_id: str,
    payload: VerifyPasswordIn,
    request: Request,
) -> dict:
    """Verify a user-supplied extension password against the dataset's
    stored bcrypt hash (which lives in metalens-datasets and is fetched
    server-side; the hash never leaves the server).

    Response shape: ``{"ok": bool}``.  Returns 200 in all valid cases
    — the boolean discriminates success from failure.  404 only when
    the dataset itself doesn't exist; 400 on a malformed slug or empty
    password; 429 if the requester has burned through their attempts.

    Rate-limit: ``_VERIFY_PER_IP_PER_DATASET`` attempts per dataset per
    IP, sliding 1-hour window.  In-memory only — single-process
    deployments are fine; multi-machine would need a shared store.
    Loopback IPs bypass the limiter (dev convenience).
    """
    # Defence in depth — FastAPI accepts the path param raw, validate
    # before the lookup hits GitHub.
    if not dataset_id or len(dataset_id) > 120:
        raise HTTPException(status_code=400, detail="Invalid dataset id.")

    password = payload.password or ""
    if not password:
        raise HTTPException(status_code=400, detail="Password is required.")

    # Per-IP, per-dataset rate limit — bcrypt is slow on purpose
    # (~250ms) so we don't need to be aggressive here, but a cap
    # prevents an attacker from grinding the hash from one IP.
    client_ip = _client_ip_from_request(request)
    if not _is_loopback_ip(client_ip):
        try:
            ip_hash = donor.hash_ip(client_ip)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        if _verify_attempts_too_many(ip_hash, dataset_id):
            raise HTTPException(
                status_code=429,
                detail="Too many password attempts for this dataset.  Try again in an hour.",
            )
        _verify_attempts_bump(ip_hash, dataset_id)

    metadata = datasets_mod.get_dataset_metadata(dataset_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    # Public datasets — no password gate, anyone can extend.  We
    # return ok=True for any non-empty password so the frontend
    # treats the verify step as cleared.  (The frontend shouldn't
    # actually call this for public datasets, but be lenient.)
    stored_hash = metadata.get("password_hash")
    if not stored_hash:
        return {"ok": True, "gated": False}

    ok = donor.verify_password(password, stored_hash)
    return {"ok": bool(ok), "gated": True}


# In-memory sliding-window attempt counter.  Keyed on (ip_hash,
# dataset_id) → list of attempt timestamps.  Cleaned lazily on each
# read to avoid an unbounded grow if traffic is bursty.
_VERIFY_ATTEMPTS: dict[tuple[str, str], list[float]] = {}
_VERIFY_PER_IP_PER_DATASET = 10
_VERIFY_WINDOW_SEC         = 3600


def _verify_attempts_too_many(ip_hash: str, dataset_id: str) -> bool:
    """True iff ``ip_hash`` has burned through its quota for
    ``dataset_id`` in the last hour."""
    import time as _time
    cutoff = _time.time() - _VERIFY_WINDOW_SEC
    key    = (ip_hash, dataset_id)
    recent = [t for t in _VERIFY_ATTEMPTS.get(key, []) if t >= cutoff]
    _VERIFY_ATTEMPTS[key] = recent
    return len(recent) >= _VERIFY_PER_IP_PER_DATASET


def _verify_attempts_bump(ip_hash: str, dataset_id: str) -> None:
    import time as _time
    key = (ip_hash, dataset_id)
    _VERIFY_ATTEMPTS.setdefault(key, []).append(_time.time())


# ── /api/presets — domain workflow presets (e.g. MASEMiner) ─────────────────

@app.get("/api/presets")
def list_presets() -> dict:
    """List the available presets with branding metadata only — keeps the
    response small for the workflow picker on the landing screen."""
    return {"presets": presets_loader.list_summaries()}


@app.get("/api/presets/{preset_id}")
def get_preset(preset_id: str) -> dict:
    """Full preset detail (including the full prompt body)."""
    preset = presets_loader.get(preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found.")
    return preset


@app.post("/api/build-preset-prompt")
def build_preset_prompt(payload: BuildPresetPromptIn) -> dict:
    """Re-render a preset's template with user-supplied ``template_params``
    (the guided MASEMiner builder form posts here on every change to keep
    the live prompt preview in sync, then again on "Build" to commit the
    final prompt + sub_views).

    Merges the form values on top of the preset's defaults so the user
    only has to send what they actually changed.  Returns the rendered
    prompt + the auto-generated ``sub_views`` so the result panel can
    pick up the right sub-tab set immediately."""
    preset = presets_loader.get(payload.preset_id)
    if preset is None:
        raise HTTPException(status_code=404, detail="Preset not found.")
    base_params = dict(preset.get("template_params") or {})
    base_params.update(payload.template_params or {})
    template = presets_loader.read_template_for(payload.preset_id)
    if template is None:
        raise HTTPException(
            status_code=400,
            detail="This preset does not use a parameterised template.",
        )
    prompt = presets_loader.render_template(template, base_params)
    # Use the preset's explicit sub_views (declared in its JSON) when
    # available AND the caller didn't ACTUALLY change data_sources —
    # preset authors stay in control of the result-panel layout in the
    # default case.  ``overrode_sources`` only fires when the payload's
    # data_sources DIFFERS from the preset's own defaults, not merely
    # when the key is present.  Without this guard the masem-builder
    # silently breaks the Direct preset's explicit sub_views: the
    # builder echoes the FULL template_params on every form change
    # (including the unchanged ``data_sources: ["records"]``), which
    # the previous "key in payload" check mis-classified as a user
    # override → fell into auto-generation → "records" isn't in
    # _SUB_VIEW_SPECS → returned an empty list → wiped the Effect-sizes
    # + Descriptives sub-tabs.  Comparing against the preset default
    # restores the intended behaviour: "override only when different".
    preset_sources = (preset.get("template_params") or {}).get("data_sources") or []
    payload_params = payload.template_params or {}
    overrode_sources = (
        "data_sources" in payload_params
        and (payload_params.get("data_sources") or []) != preset_sources
    )
    if not overrode_sources and preset.get("sub_views"):
        sub_views = preset["sub_views"]
    else:
        sub_views = presets_loader.build_sub_views(base_params.get("data_sources") or [])
    return {
        "prompt":          prompt,
        "sub_views":       sub_views,
        "template_params": base_params,
    }


# ── /api/test-connection ─────────────────────────────────────────────────────

@app.post("/api/test-connection")
async def test_connection(payload: TestConnectionIn) -> Any:
    """Quick credential check: send a 1-token completion to confirm the key /
    endpoint / model combination is reachable and authorised.  Saves the user
    from waiting 30-120 s on extraction only to discover their key is wrong."""
    api_key  = payload.api_key.strip()
    model    = payload.model.strip()
    base_url = (payload.base_url or "").strip() or None

    if not api_key and not base_url:
        raise HTTPException(status_code=400, detail="API key is required.")
    _ascii_only(api_key,  "API key")
    _ascii_only(base_url, "Server URL")

    try:
        # Smallest possible prompt — costs ~1 input + 1 output token
        await asyncio.to_thread(
            generate_text, model, api_key, "ping", 0.0, base_url,
        )
        return {"ok": True, "model": model}
    except Exception as exc:  # noqa: BLE001
        return _provider_error_response(exc)


# ── /api/adapt-prompt ─────────────────────────────────────────────────────────

@app.post("/api/adapt-prompt")
async def adapt_prompt(payload: AdaptPromptIn) -> Any:
    api_key  = payload.api_key.strip()
    prompt   = payload.prompt.strip()
    base_url = (payload.base_url or "").strip() or None

    if not api_key and not base_url:
        raise HTTPException(status_code=400, detail="API key is required.")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    _ascii_only(api_key,  "API key")
    _ascii_only(base_url, "Server URL")

    instruction = (
        "You will be given an extraction or labeling prompt for academic papers. "
        "Your task is to MINIMALLY MODIFY this prompt so that it instructs the AI "
        "to additionally output an 'evidence' array on each result/sample object, "
        "with this exact schema:\n\n"
        "  evidence: [{snippet: string (verbatim quote), page: integer (1-indexed PDF page), "
        "source: string|null (e.g. 'Table 2'), field: string (which extracted field this supports)}]\n\n"
        "Rules:\n"
        "1. Do NOT change anything else about the prompt — preserve all existing instructions, "
        "schema definitions, edge-case rules, and style.\n"
        "2. Add the evidence requirement clearly so the AI knows to output it on EVERY result object.\n"
        "3. Keep the evidence as a separate array — do NOT inline snippet/page next to numeric values.\n"
        "4. Return only the modified prompt — no preamble, no explanation, no markdown fences.\n\n"
        "Original prompt:\n------\n"
        f"{prompt}\n"
        "------\n\nReturn the modified prompt now."
    )
    try:
        adapted = await asyncio.to_thread(
            generate_text, payload.model, api_key, instruction, 0.1, base_url
        )
        return {"prompt": adapted}
    except Exception as exc:  # noqa: BLE001
        return _provider_error_response(exc)


# ── /api/extract — enqueue ───────────────────────────────────────────────────

@app.post("/api/check-pdf")
async def check_pdf(pdf: UploadFile = File(...)) -> dict:
    """Inspect an uploaded PDF's text layer so the client can warn the user
    upfront when a paper is scanned / image-only (vision extraction still
    works, but rect-based highlights won't be available)."""
    filename = pdf.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file (.pdf).")
    try:
        pdf_bytes = await pdf.read()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")
    try:
        from pdf_utils import probe_text_layer
        return probe_text_layer(pdf_bytes)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to inspect PDF: {e}")


@app.post("/api/extract")
async def extract(
    api_key: str = Form(""),
    model: str = Form("gpt-4o-mini"),
    prompt: str = Form(""),
    use_text_extraction: str = Form("0"),
    base_url: str = Form(""),
    batch_id: str = Form(""),       # generated client-side; same id for every paper in a batch
    notify_email: str = Form(""),   # optional — triggers email when the batch finishes
    pdf: UploadFile = File(...),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict:
    api_key      = api_key.strip()
    prompt       = prompt.strip()
    base_url     = base_url.strip() or None
    batch_id     = batch_id.strip() or None
    notify_email = notify_email.strip() or None

    if not api_key and not base_url:
        raise HTTPException(status_code=400, detail="API key is required.")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    _ascii_only(api_key,  "API key")
    _ascii_only(base_url, "Server URL")

    filename = pdf.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file (.pdf).")

    try:
        pdf_bytes = await pdf.read()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

    # Per-file byte limit (sanity check; the frontend already enforces 50 MB)
    pdf_bytes_size = len(pdf_bytes)
    if pdf_bytes_size > _max_pdf_bytes():
        raise HTTPException(
            status_code=400,
            detail=f"PDF exceeds the per-file limit ({pdf_bytes_size // (1024*1024)} MB > "
                   f"{_max_pdf_bytes() // (1024*1024)} MB).",
        )

    if batch_id:
        # Idempotent — only the first paper in the batch creates the row.
        # session_id scopes the History view to one browser.
        db.create_batch(batch_id, notify_email, session_id=x_session_id)
        # Per-batch cap on *uploaded files*.  Reruns of an already-uploaded
        # paper submit a new job under the same filename — we want those to
        # pass through freely, only newly-introduced filenames count toward
        # the cap.  Counting distinct filenames already in the batch +
        # whether THIS filename is novel gives the right semantic and stays
        # enforced even if the frontend misbehaves.
        existing = db.distinct_filenames_in_batch(batch_id)
        cap      = _max_batch_papers()
        is_rerun = filename in existing
        if not is_rerun and len(existing) >= cap:
            raise HTTPException(
                status_code=400,
                detail=f"Batch limit reached ({cap} papers per batch). "
                       f"Start a new extraction or split the upload into smaller batches.",
            )
    job_id = jobs_mod.new_job_id()
    db.create_job(job_id, filename, batch_id=batch_id, prompt=prompt, model=model)

    jobs_mod.submit(
        job_id,
        model=model,
        api_key=api_key,
        prompt=prompt,
        pdf_bytes=pdf_bytes,
        use_text_extraction=use_text_extraction == "1",
        base_url=base_url,
        batch_id=batch_id,
    )
    return {"job_id": job_id, "filename": filename, "status": "pending", "batch_id": batch_id}


# ── /api/jobs/{id} — poll ────────────────────────────────────────────────────

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    # Compute total evidence entries (with snippet, regardless of page) on
    # demand so the frontend can distinguish "no evidence array at all" from
    # "evidence returned but pages are missing/unmapped".
    from pdf_utils import count_evidence_entries
    result   = row.get("result") or ""
    ev_total = count_evidence_entries(result) if result else 0
    return {
        "job_id":          row["id"],
        "status":          row["status"],
        "phase":           row.get("phase"),
        "filename":        row.get("filename"),
        "batch_id":        row.get("batch_id"),
        "result":          result,
        "error":           row.get("error"),
        "pages_processed": row.get("pages_processed"),
        "evidence_count":  row.get("evidence_count"),    # entries actually highlighted
        "evidence_total":  ev_total,                     # entries with a snippet
        "finish_reason":   row.get("finish_reason"),
        "token_usage":     row.get("token_usage"),
        "model":           row.get("model"),             # alias the user selected
        "resolved_model":  row.get("resolved_model"),    # dated snapshot the provider served
    }


@app.get("/api/jobs/{job_id}/pages")
def get_job_pages(job_id: str) -> dict:
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    images        = jobs_mod.get_page_images(job_id) or []
    highlights    = jobs_mod.get_page_highlights(job_id) or []
    scanned_pages = jobs_mod.get_scanned_pages(job_id) or []
    # ``highlights`` is a list of {page, snippet, field, source, rects} —
    # the frontend overlays an SVG layer of yellow rectangles on top of
    # ``page_images``, filtered by the currently-selected sub-view (if any).
    # ``scanned_pages`` lists 1-indexed pages with no usable text layer; the
    # client uses this to explain WHY a given page can't be highlighted.
    return {
        "job_id":        job_id,
        "page_images":   images,
        "highlights":    highlights,
        "scanned_pages": scanned_pages,
    }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Set the cancel flag.  Worker thread checks it between phases and stops."""
    if not db.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    requested = db.request_cancel(job_id)
    return {"ok": True, "cancel_requested": requested}


# ── /api/batches — history + per-batch detail ────────────────────────────────

@app.get("/api/batches")
def list_batches(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> dict:
    """Recent batches for the current browser only.

    Each browser mints a UUID once and stores it in localStorage; that id is
    sent on every request as ``X-Session-Id``.  This endpoint scopes the
    History view to that id, so users no longer see other people's batches.
    Without a session id, returns an empty list.
    """
    return {"batches": db.list_batches(limit=50, session_id=x_session_id)}


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str) -> dict:
    """Full detail for one batch: jobs + the email that was attached to it."""
    batch = db.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    jobs_in_batch = db.list_jobs_in_batch(batch_id)
    # Strip the page-image blob — keep payload small for the history view
    return {"batch": batch, "jobs": jobs_in_batch}


@app.post("/api/batches/{batch_id}/cancel")
def cancel_batch(batch_id: str) -> dict:
    """Cancel every still-running job in the batch."""
    jobs_in_batch = db.list_jobs_in_batch(batch_id)
    n = sum(1 for j in jobs_in_batch if db.request_cancel(j["id"]))
    return {"ok": True, "cancelled": n}


@app.post("/api/batches/{batch_id}/email")
def set_batch_email(batch_id: str, payload: BatchEmailIn) -> dict:
    """Attach (or update) the notification email for a batch.

    The user can call this at any time during processing.  When the worker
    finishes the last job in the batch it'll pick up the address and send.
    If the batch is *already* finished (and not yet notified), we send
    immediately so the user still gets the email.
    """
    email = payload.email.strip()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    _ascii_only(email, "Email")
    if not db.update_batch_email(batch_id, email):
        raise HTTPException(status_code=404, detail="Batch not found.")

    # Already finished?  Send right now (atomic claim prevents duplicates).
    if db.all_batch_jobs_finished(batch_id) and db.claim_batch_notification(batch_id):
        import notifier
        notifier.send_batch_complete_async(
            to=email,
            batch_id=batch_id,
            jobs_in_batch=db.list_jobs_in_batch(batch_id),
        )
        return {"ok": True, "sent_now": True}
    return {"ok": True, "sent_now": False}


# ── /api/pages — review-only flow (unchanged) ────────────────────────────────

@app.post("/api/pages")
async def render_pages(
    pdf: UploadFile = File(...),
    result: str = Form(""),
) -> dict:
    filename = pdf.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file (.pdf).")
    pdf_bytes = await pdf.read()

    snippets_by_page = extract_evidence_snippets(result) if result else {}
    images = await asyncio.to_thread(pdf_to_highlighted_images, pdf_bytes, snippets_by_page)
    return {
        "filename":    filename,
        "page_images": [f"data:image/jpeg;base64,{b}" for b in images],
    }


# ── Local dev entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5001)),
        reload=True,
    )
