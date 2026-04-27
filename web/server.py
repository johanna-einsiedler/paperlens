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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import jobs as jobs_mod
from pdf_utils import extract_evidence_snippets, pdf_to_highlighted_images
from prompt_builder import EVIDENCE_APPENDIX, build_meta_prompt
from providers import extract_provider_message, generate_text

STATIC_DIR = Path(__file__).parent / "static"


# ── Batch limits ──────────────────────────────────────────────────────────────
# Read from env at request time so a deployment can tune them without a code
# change.  Defaults chosen for a typical OpenAI Tier-1 key + 1 GB server RAM.
import os as _os

def _max_batch_papers() -> int:
    try:
        return max(1, int(_os.environ.get("PAPERLENS_MAX_BATCH_PAPERS", "20")))
    except ValueError:
        return 20

def _max_pdf_bytes() -> int:
    """Per-file ceiling — should match the frontend's 50 MB enforcement."""
    try:
        return max(1, int(_os.environ.get("PAPERLENS_MAX_PDF_BYTES", str(50 * 1024 * 1024))))
    except ValueError:
        return 50 * 1024 * 1024


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    db.init()
    # Best-effort cleanup of week-old jobs on each boot
    try:
        db.cleanup_old_jobs(7 * 24 * 3600)
    except Exception:  # noqa: BLE001
        pass
    yield


app = FastAPI(title="PaperLens", docs_url=None, redoc_url=None, lifespan=_lifespan)


# ── Static serving ────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma":        "no-cache",
    "Expires":       "0",
}


@app.get("/")
def index() -> FileResponse:
    # Prevent the browser from serving a stale index that points to an outdated
    # bundle.  Static assets are still mounted above with their own headers.
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE_HEADERS)


@app.middleware("http")
async def _no_cache_static(request, call_next):
    """Disable caching on /static/* so JS/CSS updates always reach the browser
    on a normal reload (no need for hard-refresh after each deploy)."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        for k, v in _NO_CACHE_HEADERS.items():
            response.headers[k] = v
    return response


# ── Pydantic models ───────────────────────────────────────────────────────────

class GeneratePromptIn(BaseModel):
    api_key: str = ""
    model: str = "gpt-4o-mini"
    mode: str = "extraction"
    question: str = ""
    context: str = ""
    base_url: str | None = None


class CheckSchemaIn(BaseModel):
    prompt: str = ""


class TestConnectionIn(BaseModel):
    api_key:  str = ""
    model:    str = "gpt-4o-mini"
    base_url: str | None = None


class BatchEmailIn(BaseModel):
    email: str = ""


class AdaptPromptIn(BaseModel):
    api_key: str = ""
    model: str = "gpt-4o-mini"
    prompt: str = ""
    base_url: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _provider_error_response(exc: Exception) -> JSONResponse:
    """Translate provider-side exceptions into clean user-facing JSON errors.

    The provider's actual reason (e.g. "Incorrect API key provided: sk-...",
    "Image is too large", "Rate limit reached for ...") is preserved verbatim
    so the user sees what's wrong, not a generic placeholder.
    """
    import openai
    import sys
    import traceback
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
    if payload.mode not in ("extraction", "labeling"):
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
    """Server-side limits the frontend should respect.  Read at page load so the
    UI can render 'up to N papers per batch' and refuse oversize uploads early."""
    return {
        "max_batch_papers": _max_batch_papers(),
        "max_pdf_bytes":    _max_pdf_bytes(),
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
        # Idempotent — only the first paper in the batch creates the row
        db.create_batch(batch_id, notify_email)
        # Per-batch cap on paper count.  Counts existing jobs with this id so
        # the limit is enforced even if the frontend misbehaves or the same
        # batch_id is reused across requests.
        already = db.count_jobs_in_batch(batch_id)
        cap     = _max_batch_papers()
        if already >= cap:
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
    }


@app.get("/api/jobs/{job_id}/pages")
def get_job_pages(job_id: str) -> dict:
    row = db.get_job(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found.")
    images = jobs_mod.get_page_images(job_id) or []
    return {"job_id": job_id, "page_images": images}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Set the cancel flag.  Worker thread checks it between phases and stops."""
    if not db.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    requested = db.request_cancel(job_id)
    return {"ok": True, "cancel_requested": requested}


# ── /api/batches — history + per-batch detail ────────────────────────────────

@app.get("/api/batches")
def list_batches() -> dict:
    """Recent batches with aggregate counts — feeds the History view."""
    return {"batches": db.list_batches(limit=50)}


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
