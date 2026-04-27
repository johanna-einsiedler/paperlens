"""SQLite-backed job storage for asynchronous extraction.

Schema (current):

    jobs(
      id              TEXT PRIMARY KEY,    -- uuid4
      batch_id        TEXT,                -- groups papers from one upload
      status          TEXT NOT NULL,       -- pending|processing|done|error|cancelled
      phase           TEXT,                -- human-readable progress phase (e.g. "Rendering pages")
      filename        TEXT,
      result          TEXT,                -- model JSON output
      error           TEXT,
      pages_processed INTEGER,
      evidence_count  INTEGER,
      finish_reason   TEXT,
      token_usage     TEXT,                -- JSON {prompt, completion, total}
      cancel_requested INTEGER DEFAULT 0,  -- 1 = user asked us to stop
      prompt          TEXT,                -- the prompt used (for the History view)
      model           TEXT,                -- the model used
      created_at      REAL,
      updated_at      REAL
    )

    batches(
      id              TEXT PRIMARY KEY,
      notify_email    TEXT,
      notified_at     REAL,                -- non-null iff completion email was sent
      created_at      REAL
    )

Notes
-----
* Page images are intentionally NOT persisted — they can be re-rendered from
  the (uploaded) PDF on demand and they're 1–10 MB each.  We keep them in
  process memory keyed by job id (see ``jobs.py``).
* The DB lives at ``$PAPERLENS_DB_PATH`` if set, otherwise next to this file.
* ``init()`` runs ``ALTER TABLE`` for additive columns so existing databases
  upgrade in place.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path(__file__).parent / "paperlens.sqlite3"


def _db_path() -> Path:
    return Path(os.environ.get("PAPERLENS_DB_PATH", DEFAULT_DB_PATH))


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path(), isolation_level=None, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────

# Columns added after the original schema — we ALTER TABLE to add any that are
# missing so users on an older database upgrade in place without losing jobs.
_ADDITIVE_JOB_COLUMNS = [
    ("batch_id",         "TEXT"),
    ("phase",            "TEXT"),
    ("cancel_requested", "INTEGER DEFAULT 0"),
    ("prompt",           "TEXT"),
    ("model",            "TEXT"),
]


def init() -> None:
    """Create the schema if it does not exist; add new columns to old DBs."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
              id              TEXT PRIMARY KEY,
              status          TEXT NOT NULL,
              filename        TEXT,
              result          TEXT,
              error           TEXT,
              pages_processed INTEGER,
              evidence_count  INTEGER,
              finish_reason   TEXT,
              token_usage     TEXT,
              created_at      REAL,
              updated_at      REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
              id            TEXT PRIMARY KEY,
              notify_email  TEXT,
              notified_at   REAL,
              created_at    REAL
            )
        """)
        # Migrate any older "jobs" table missing newer columns
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
        for name, type_ in _ADDITIVE_JOB_COLUMNS:
            if name not in existing_cols:
                conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {type_}")
        conn.execute("CREATE INDEX IF NOT EXISTS jobs_batch_id_idx ON jobs(batch_id)")


# ── Job CRUD ──────────────────────────────────────────────────────────────────

def create_job(
    job_id: str,
    filename: str,
    *,
    batch_id: str | None = None,
    prompt: str | None = None,
    model: str | None = None,
) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO jobs (id, batch_id, status, filename, prompt, model, created_at, updated_at)
               VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)""",
            (job_id, batch_id, filename, prompt, model, now, now),
        )


def update_status(job_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status, time.time(), job_id),
        )


def update_phase(job_id: str, phase: str) -> None:
    """Set a human-readable phase label for an in-flight job (e.g. 'Calling LLM')."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET phase = ?, updated_at = ? WHERE id = ?",
            (phase, time.time(), job_id),
        )


def mark_done(
    job_id: str,
    *,
    result: str,
    pages_processed: int,
    evidence_count: int,
    finish_reason: str,
    token_usage: dict,
) -> None:
    with _connect() as conn:
        conn.execute(
            """UPDATE jobs SET
                 status = 'done',
                 result = ?,
                 pages_processed = ?,
                 evidence_count = ?,
                 finish_reason = ?,
                 token_usage = ?,
                 phase = NULL,
                 updated_at = ?
               WHERE id = ?""",
            (result, pages_processed, evidence_count, finish_reason,
             json.dumps(token_usage), time.time(), job_id),
        )


def mark_error(job_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'error', error = ?, phase = NULL, updated_at = ? WHERE id = ?",
            (error, time.time(), job_id),
        )


def mark_cancelled(job_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'cancelled', phase = NULL, updated_at = ? WHERE id = ?",
            (time.time(), job_id),
        )


def request_cancel(job_id: str) -> bool:
    """Set cancel_requested=1 if the job is still pending/processing.  Returns
    True if a cancel flag was actually set."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET cancel_requested = 1, updated_at = ? "
            "WHERE id = ? AND status IN ('pending','processing')",
            (time.time(), job_id),
        )
        return cur.rowcount > 0


def is_cancel_requested(job_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return bool(row and row["cancel_requested"])


def get_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    out = dict(row)
    if out.get("token_usage"):
        try:
            out["token_usage"] = json.loads(out["token_usage"])
        except json.JSONDecodeError:
            out["token_usage"] = None
    return out


def cleanup_old_jobs(older_than_seconds: int = 24 * 3600) -> int:
    """Delete jobs older than the given threshold. Returns the number deleted."""
    cutoff = time.time() - older_than_seconds
    with _connect() as conn:
        cur = conn.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
        return cur.rowcount


# ── Batch CRUD ────────────────────────────────────────────────────────────────

def create_batch(batch_id: str, notify_email: str | None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO batches (id, notify_email, created_at) VALUES (?, ?, ?)",
            (batch_id, notify_email, time.time()),
        )


def get_batch(batch_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    return dict(row) if row else None


def list_batches(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent batches with aggregate job counts — feeds the History view."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT
                 b.id, b.notify_email, b.notified_at, b.created_at,
                 COUNT(j.id)                                                  AS n_total,
                 SUM(CASE WHEN j.status = 'done'      THEN 1 ELSE 0 END)      AS n_done,
                 SUM(CASE WHEN j.status = 'error'     THEN 1 ELSE 0 END)      AS n_error,
                 SUM(CASE WHEN j.status = 'cancelled' THEN 1 ELSE 0 END)      AS n_cancelled,
                 SUM(CASE WHEN j.status IN ('pending','processing') THEN 1 ELSE 0 END) AS n_pending,
                 MIN(j.filename)                                              AS sample_filename,
                 MAX(j.model)                                                 AS model
               FROM batches b
               LEFT JOIN jobs j ON j.batch_id = b.id
               GROUP BY b.id
               ORDER BY b.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_jobs_in_batch(batch_id: str) -> int:
    """How many jobs share this batch_id? Used to enforce per-batch caps."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE batch_id = ?", (batch_id,),
        ).fetchone()
    return int(row["n"]) if row else 0


def list_jobs_in_batch(batch_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE batch_id = ? ORDER BY created_at",
            (batch_id,),
        ).fetchall()
    out = []
    for row in rows:
        d = dict(row)
        if d.get("token_usage"):
            try:
                d["token_usage"] = json.loads(d["token_usage"])
            except json.JSONDecodeError:
                d["token_usage"] = None
        out.append(d)
    return out


def all_batch_jobs_finished(batch_id: str) -> bool:
    """True iff every job in the batch is in a terminal state."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS pending FROM jobs "
            "WHERE batch_id = ? AND status IN ('pending','processing')",
            (batch_id,),
        ).fetchone()
    return (row["pending"] or 0) == 0


def update_batch_email(batch_id: str, email: str) -> bool:
    """Set the notification email for an existing batch.  Returns True if the
    batch existed and was updated."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE batches SET notify_email = ? WHERE id = ?", (email, batch_id),
        )
        return cur.rowcount > 0


def claim_batch_notification(batch_id: str) -> bool:
    """Atomic 'I am sending the notification email' claim.  Returns True iff
    this caller is the one that should send it (only one thread wins)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE batches SET notified_at = ? WHERE id = ? AND notified_at IS NULL",
            (time.time(), batch_id),
        )
        return cur.rowcount > 0
