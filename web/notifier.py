"""Email notifications when an extraction batch completes.

Configured via environment variables (read at send time, so changes take effect
without a restart):

    PAPERLENS_SMTP_HOST       e.g. "smtp.gmail.com"  (required to enable sending)
    PAPERLENS_SMTP_PORT       default 587
    PAPERLENS_SMTP_USER       SMTP login
    PAPERLENS_SMTP_PASS       SMTP password
    PAPERLENS_SMTP_USE_TLS    "1" to STARTTLS, "0" to skip (default 1)
    PAPERLENS_SMTP_FROM       From address (default = SMTP user)
    PAPERLENS_PUBLIC_URL      Optional — included as a "view results" link in the body

If ``PAPERLENS_SMTP_HOST`` is unset, the email is *logged to stderr* instead of
sent.  This makes local development painless: no real SMTP needed, and the body
of every notification is visible in the server log.
"""

from __future__ import annotations

import os
import smtplib
import sys
import threading
import traceback
from email.message import EmailMessage


def _is_configured() -> bool:
    return bool(os.environ.get("PAPERLENS_SMTP_HOST"))


def _build_message(
    *, to: str, subject: str, body: str, sender: str
) -> EmailMessage:
    msg = EmailMessage()
    msg["From"]    = sender
    msg["To"]      = to
    msg["Subject"] = subject
    msg.set_content(body)
    return msg


def _send_via_smtp(msg: EmailMessage) -> None:
    host = os.environ["PAPERLENS_SMTP_HOST"]
    port = int(os.environ.get("PAPERLENS_SMTP_PORT", "587"))
    user = os.environ.get("PAPERLENS_SMTP_USER", "")
    pwd  = os.environ.get("PAPERLENS_SMTP_PASS", "")
    use_tls = os.environ.get("PAPERLENS_SMTP_USE_TLS", "1") == "1"

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        if user:
            smtp.login(user, pwd)
        smtp.send_message(msg)


def _format_summary(
    *, batch_id: str, jobs_in_batch: list[dict], public_url: str | None,
) -> tuple[str, str]:
    """Build (subject, body) text for a batch notification."""
    n_total     = len(jobs_in_batch)
    n_done      = sum(1 for j in jobs_in_batch if j.get("status") == "done")
    n_error     = sum(1 for j in jobs_in_batch if j.get("status") == "error")
    n_cancelled = sum(1 for j in jobs_in_batch if j.get("status") == "cancelled")

    if n_done == n_total:
        subject = f"PaperLens — extraction complete ({n_total} paper{'s' if n_total != 1 else ''})"
    elif n_done == 0:
        subject = f"PaperLens — extraction failed ({n_total} paper{'s' if n_total != 1 else ''})"
    else:
        subject = f"PaperLens — extraction finished with {n_error + n_cancelled} issue(s)"

    lines = [
        "Your PaperLens extraction batch has finished.",
        "",
        f"Batch ID:   {batch_id}",
        f"Papers:     {n_total} total",
        f"Successful: {n_done}",
    ]
    if n_error:
        lines.append(f"Failed:     {n_error}")
    if n_cancelled:
        lines.append(f"Cancelled:  {n_cancelled}")

    lines.append("")
    lines.append("Per-paper status:")
    for j in jobs_in_batch:
        status = j.get("status", "?")
        name   = j.get("filename") or "(unknown)"
        marker = {"done": "✓", "error": "✕", "cancelled": "⊘"}.get(status, "·")
        line   = f"  {marker} {name} — {status}"
        if status == "error" and j.get("error"):
            line += f" ({j['error'][:120]})"
        lines.append(line)

    if public_url:
        lines.append("")
        lines.append(f"View results: {public_url.rstrip('/')}/?batch={batch_id}")

    lines.append("")
    lines.append("— PaperLens")

    return subject, "\n".join(lines)


def send_batch_complete(
    *, to: str, batch_id: str, jobs_in_batch: list[dict], public_url: str | None = None,
) -> None:
    """Send the batch-complete notification, or log it if SMTP isn't configured.

    Runs synchronously — caller should off-load to a thread if blocking matters.
    Never raises: failures are logged and swallowed so they can't break the worker.
    """
    sender   = os.environ.get("PAPERLENS_SMTP_FROM") or os.environ.get("PAPERLENS_SMTP_USER") or "noreply@paperlens.local"
    subject, body = _format_summary(
        batch_id=batch_id, jobs_in_batch=jobs_in_batch,
        public_url=public_url or os.environ.get("PAPERLENS_PUBLIC_URL"),
    )
    msg = _build_message(to=to, subject=subject, body=body, sender=sender)

    if not _is_configured():
        # Dev / unconfigured production: visible in stderr, no real send.
        print(
            "─── [notifier] SMTP not configured; would have sent: ───\n"
            f"To:      {to}\nSubject: {subject}\n\n{body}\n"
            "──────────────────────────────────────────────────────",
            file=sys.stderr, flush=True,
        )
        return

    try:
        _send_via_smtp(msg)
        print(f"[notifier] sent batch-complete email to {to} (batch={batch_id})",
              file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        print("[notifier] SMTP send failed:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def send_batch_complete_async(**kwargs) -> None:
    """Fire-and-forget wrapper — sending email shouldn't block the worker thread."""
    threading.Thread(target=send_batch_complete, kwargs=kwargs, daemon=True).start()
