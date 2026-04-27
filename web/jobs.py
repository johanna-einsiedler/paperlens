"""In-process job runner for extraction tasks.

Each job runs on a daemon thread.  Threads (rather than ``asyncio.create_task``)
are used because:
  * They survive across HTTP request boundaries — important for TestClient
    where the asyncio loop terminates after each request.
  * The LLM SDKs and PyMuPDF are sync; ``asyncio.to_thread`` would spawn a
    thread anyway, so we save one indirection.
  * Extraction is mostly I/O wait, so the GIL isn't a bottleneck.

State persistence:
  * Job status, result, phase, metadata → SQLite (``db.py``)
  * Page-image blobs → in-memory dict keyed by job id
"""

from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any

import db
import notifier
from pdf_utils import (
    EXTRACTION_DPI,
    count_evidence_entries,
    extract_evidence_snippets,
    merge_snippet_dicts,
    pdf_to_highlighted_images,
    pdf_to_images,
    pdf_to_markdown,
    recover_orphan_pages,
)
from providers import extract_provider_message, extract_with_images, extract_with_text, get_provider

# In-memory page-image cache keyed by job id.
_PAGE_IMAGES: dict[str, list[str]] = {}
_LOCK = threading.Lock()


def new_job_id() -> str:
    return str(uuid.uuid4())


def get_page_images(job_id: str) -> list[str] | None:
    with _LOCK:
        return _PAGE_IMAGES.get(job_id)


def _set_page_images(job_id: str, images: list[str]) -> None:
    with _LOCK:
        _PAGE_IMAGES[job_id] = images


def forget_page_images(job_id: str) -> None:
    with _LOCK:
        _PAGE_IMAGES.pop(job_id, None)


class _Cancelled(Exception):
    """Raised when a cancel was requested mid-flight."""


def _check_cancelled(job_id: str) -> None:
    if db.is_cancel_requested(job_id):
        raise _Cancelled()


def _set_phase(job_id: str, phase: str) -> None:
    """Update the job's phase label and check for cancellation in one step."""
    _check_cancelled(job_id)
    db.update_phase(job_id, phase)


def submit(
    job_id: str,
    *,
    model: str,
    api_key: str,
    prompt: str,
    pdf_bytes: bytes,
    use_text_extraction: bool,
    base_url: str | None = None,
    batch_id: str | None = None,
) -> None:
    """Spawn a daemon thread to run the extraction in the background."""
    t = threading.Thread(
        target=_run_extraction,
        kwargs={
            "job_id": job_id,
            "model": model,
            "api_key": api_key,
            "prompt": prompt,
            "pdf_bytes": pdf_bytes,
            "use_text_extraction": use_text_extraction,
            "base_url": base_url,
            "batch_id": batch_id,
        },
        daemon=True,
    )
    t.start()


def _run_extraction(
    job_id: str,
    *,
    model: str,
    api_key: str,
    prompt: str,
    pdf_bytes: bytes,
    use_text_extraction: bool,
    base_url: str | None,
    batch_id: str | None,
) -> None:
    """Background worker — runs the full extract → highlight pipeline."""
    db.update_status(job_id, "processing")
    try:
        provider   = get_provider(model, base_url)
        force_text = use_text_extraction or provider == "deepseek"

        if force_text:
            result, finish, n, usage = _run_text_path(job_id, model, api_key, prompt, pdf_bytes, base_url)
        else:
            result, finish, n, usage = _run_vision_path(job_id, model, api_key, prompt, pdf_bytes, base_url)

        _set_phase(job_id, "Highlighting evidence")
        # Snippets the model explicitly tagged with a page number
        snippets_by_page = extract_evidence_snippets(result)
        # Snippets the model returned WITHOUT a page — recover by searching
        # the PDF text.  Most models drop the page field at least sometimes;
        # this rescues those entries so highlighting still works.
        recovered        = recover_orphan_pages(result, pdf_bytes)
        snippets_by_page = merge_snippet_dicts(snippets_by_page, recovered)
        n_snippets       = sum(len(v) for v in snippets_by_page.values())
        page_images      = pdf_to_highlighted_images(pdf_bytes, snippets_by_page)
        _set_page_images(job_id, [f"data:image/jpeg;base64,{b}" for b in page_images])

        _check_cancelled(job_id)
        db.mark_done(
            job_id,
            result=result,
            pages_processed=n,
            evidence_count=n_snippets,
            finish_reason=finish,
            token_usage=usage,
        )
    except _Cancelled:
        db.mark_cancelled(job_id)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        db.mark_error(job_id, extract_provider_message(exc))
    finally:
        # If this was the last job in its batch, send the completion email.
        if batch_id:
            _maybe_notify_batch(batch_id)


def _maybe_notify_batch(batch_id: str) -> None:
    """If every job in this batch is in a terminal state and an email was
    requested, send the summary (but only once — claim_batch_notification is
    atomic against multiple workers finishing concurrently)."""
    if not db.all_batch_jobs_finished(batch_id):
        return
    batch = db.get_batch(batch_id)
    if not batch or not batch.get("notify_email"):
        return
    if not db.claim_batch_notification(batch_id):
        return  # another worker already sent it
    jobs_in_batch = db.list_jobs_in_batch(batch_id)
    notifier.send_batch_complete_async(
        to=batch["notify_email"],
        batch_id=batch_id,
        jobs_in_batch=jobs_in_batch,
    )


def _run_vision_path(
    job_id: str, model: str, api_key: str, prompt: str, pdf_bytes: bytes,
    base_url: str | None,
) -> tuple[str, str, int, dict]:
    _set_phase(job_id, "Rendering pages")
    images = pdf_to_images(pdf_bytes, dpi=EXTRACTION_DPI, fmt="png")
    if not images:
        raise ValueError("The PDF appears to be empty.")
    n = len(images)

    page_instruction = (
        f"\n\nThis document has been split into {n} page image{'s' if n != 1 else ''}. "
        "They are provided below in order, each labelled with its sequential PDF page number "
        "(1 = first page of the PDF). "
        "IMPORTANT: when citing evidence, always use this sequential PDF page number — "
        "NOT any journal or book page number that may be printed in the page itself."
    )
    content: list[Any] = [{"type": "text", "text": prompt + page_instruction}]
    for i, b64 in enumerate(images):
        content.append({"type": "text", "text": f"PDF page {i + 1} of {n}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    _set_phase(job_id, "Calling vision model")
    result, finish, usage = extract_with_images(
        model=model, api_key=api_key,
        content_blocks=content, extraction_images=images,
        prompt=prompt, page_instruction=page_instruction, n=n,
        base_url=base_url,
    )
    return result, finish, n, usage


def _run_text_path(
    job_id: str, model: str, api_key: str, prompt: str, pdf_bytes: bytes,
    base_url: str | None,
) -> tuple[str, str, int, dict]:
    _set_phase(job_id, "Extracting text layer")
    markdown_text, n = pdf_to_markdown(pdf_bytes)
    if not markdown_text.strip():
        raise ValueError(
            "No text layer found in this PDF. Text extraction requires a native text PDF — "
            "scanned (image-only) PDFs cannot be processed via this method."
        )
    page_instruction = (
        f"\n\nThe document text has been extracted and split into {n} labelled page sections "
        "below. IMPORTANT: when citing evidence, use the PDF page number shown in the section "
        "headers (e.g. '--- PDF page 4 of 12 ---') — NOT any journal or book page number "
        "printed in the document itself."
    )
    _set_phase(job_id, "Calling text model")
    result, finish, usage = extract_with_text(
        model=model, api_key=api_key,
        markdown_text=markdown_text, prompt=prompt, page_instruction=page_instruction,
        base_url=base_url,
    )
    return result, finish, n, usage
