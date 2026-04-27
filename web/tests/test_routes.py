"""Integration tests for the FastAPI routes.

LLM clients are mocked so we don't burn API credits.  The PDF route uses
real PyMuPDF on a generated test PDF.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Fresh TestClient with an isolated SQLite DB per test.

    db.py reads PAPERLENS_DB_PATH at call time (not import time), so a simple
    env var swap + db.init() is enough — no module reload needed (which would
    break unittest.mock.patch references)."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    import db
    import server
    db.init()
    return TestClient(server.app)


# ── /api/check-evidence-schema ───────────────────────────────────────────────

def test_test_connection_requires_credentials(client):
    r = client.post("/api/test-connection", json={"api_key": "", "model": "gpt-4o-mini"})
    assert r.status_code == 400


def test_test_connection_rejects_non_ascii_key(client):
    r = client.post("/api/test-connection",
                    json={"api_key": "sk–test", "model": "gpt-4o-mini"})
    assert r.status_code == 400
    assert "non-ASCII" in r.json()["detail"]


def test_test_connection_happy_path(client):
    """Connection test calls generate_text once with a tiny prompt and returns ok=True."""
    with patch("server.generate_text", return_value="pong"):
        r = client.post("/api/test-connection",
                        json={"api_key": "sk-real", "model": "gpt-4o-mini"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "gpt-4o-mini"


def test_test_connection_surfaces_provider_error(client):
    """If the provider rejects the key, the user sees the real reason."""
    import openai
    import httpx
    def raising(*args, **kwargs):
        response = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        raise openai.AuthenticationError(
            message="Error code: 401",
            response=response,
            body={"error": {"message": "Incorrect API key provided: sk-bad."}},
        )
    with patch("server.generate_text", side_effect=raising):
        r = client.post("/api/test-connection",
                        json={"api_key": "sk-bad", "model": "gpt-4o-mini"})
    assert r.status_code == 401
    assert "Incorrect API key" in r.json()["error"]


def test_check_evidence_schema_positive(client):
    r = client.post("/api/check-evidence-schema",
                    json={"prompt": "Quote the snippet from the page and cite source."})
    assert r.status_code == 200
    assert r.json() == {"has_evidence_schema": True}


def test_check_evidence_schema_negative(client):
    r = client.post("/api/check-evidence-schema", json={"prompt": "Just extract the data."})
    assert r.status_code == 200
    assert r.json() == {"has_evidence_schema": False}


# ── /api/generate-prompt ─────────────────────────────────────────────────────

def test_generate_prompt_requires_api_key(client):
    r = client.post("/api/generate-prompt",
                    json={"api_key": "", "model": "gpt-4o-mini",
                          "mode": "extraction", "question": "Get sample sizes."})
    assert r.status_code == 400


def test_generate_prompt_requires_question(client):
    r = client.post("/api/generate-prompt",
                    json={"api_key": "k", "model": "gpt-4o-mini",
                          "mode": "extraction", "question": ""})
    assert r.status_code == 400


def test_generate_prompt_rejects_invalid_mode(client):
    r = client.post("/api/generate-prompt",
                    json={"api_key": "k", "model": "gpt-4o-mini",
                          "mode": "bogus", "question": "x"})
    assert r.status_code == 400


def test_generate_prompt_happy_path(client):
    with patch("server.generate_text", return_value="GENERATED PROMPT BODY"):
        r = client.post("/api/generate-prompt",
                        json={"api_key": "k", "model": "gpt-4o-mini",
                              "mode": "extraction", "question": "Get sample sizes."})
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "gpt-4o-mini"
    # Server appends EVIDENCE_APPENDIX
    assert "GENERATED PROMPT BODY" in body["prompt"]
    assert "evidence" in body["prompt"].lower()


def test_generate_prompt_threads_base_url(client):
    captured = {}

    def fake_generate_text(model, api_key, prompt, temperature=0.3, base_url=None):
        captured["base_url"] = base_url
        return "ok"

    with patch("server.generate_text", side_effect=fake_generate_text):
        r = client.post("/api/generate-prompt",
                        json={"api_key": "any", "model": "llama3.2:3b",
                              "mode": "extraction", "question": "x",
                              "base_url": "http://localhost:11434"})
    assert r.status_code == 200
    assert captured["base_url"] == "http://localhost:11434"


# ── /api/adapt-prompt ────────────────────────────────────────────────────────

def test_adapt_prompt_rejects_non_ascii_api_key(client):
    """Catches the en-dash-from-smart-paste case before httpx blows up."""
    r = client.post("/api/adapt-prompt", json={
        "api_key": "sk–test",   # en-dash instead of hyphen
        "model":   "gpt-4o-mini",
        "prompt":  "extract things",
    })
    assert r.status_code == 400
    assert "non-ASCII" in r.json()["detail"]
    assert "U+2013" in r.json()["detail"]


def test_generate_prompt_rejects_non_ascii_base_url(client):
    r = client.post("/api/generate-prompt", json={
        "api_key":  "k",
        "model":    "x",
        "mode":     "extraction",
        "question": "y",
        "base_url": "http://localhost–weird:8000",
    })
    assert r.status_code == 400
    assert "non-ASCII" in r.json()["detail"]


def test_adapt_prompt_calls_llm_and_returns_output(client):
    with patch("server.generate_text", return_value="ADAPTED PROMPT") as mk:
        r = client.post("/api/adapt-prompt",
                        json={"api_key": "k", "model": "gpt-4o-mini",
                              "prompt": "extract things"})
    assert r.status_code == 200
    assert r.json() == {"prompt": "ADAPTED PROMPT"}
    # The instruction passed to the LLM must include the original prompt
    call_args = mk.call_args
    instruction = call_args.args[2] if len(call_args.args) >= 3 else call_args.kwargs.get("prompt", "")
    assert "extract things" in instruction
    assert "evidence" in instruction.lower()


# ── /api/extract → /api/jobs/{id} ────────────────────────────────────────────

def _make_pdf_bytes() -> bytes:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "N = 147 undergraduate students participated.", fontsize=12)
    out = doc.tobytes()
    doc.close()
    return out


def test_extract_rejects_non_pdf(client):
    r = client.post("/api/extract",
                    data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "p"},
                    files={"pdf": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_extract_returns_job_id_then_polls_to_done(client):
    """Submit → poll until status flips.  We mock the LLM call so the job
    completes near-instantly; the test then asserts the polling endpoint
    returns the structured result."""
    pdf = _make_pdf_bytes()

    # Mock both the vision and text extraction paths to return canned output.
    def fake_extract_with_images(**kwargs):
        return ('{"samples": [{"n": 147}]}', "stop", {"prompt": 100, "completion": 50, "total": 150})

    with patch("jobs.extract_with_images", side_effect=fake_extract_with_images):
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "Extract n.",
                  "use_text_extraction": "0"},
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "pending"
        job_id = body["job_id"]

        # Poll up to 5s for the job to finish
        deadline = time.time() + 5
        final = None
        while time.time() < deadline:
            poll = client.get(f"/api/jobs/{job_id}")
            assert poll.status_code == 200
            data = poll.json()
            if data["status"] in ("done", "error"):
                final = data
                break
            time.sleep(0.1)

    assert final is not None, "job did not finish in time"
    assert final["status"] == "done", f"expected done, got {final}"
    assert final["pages_processed"] == 1
    assert final["token_usage"] == {"prompt": 100, "completion": 50, "total": 150}
    assert "samples" in final["result"]


def test_extract_routes_deepseek_to_text_path(client):
    pdf = _make_pdf_bytes()

    def fake_extract_with_text(**kwargs):
        return ('[{"sample_id":"a"}]', "stop", {"prompt": 10, "completion": 5, "total": 15})

    with patch("jobs.extract_with_text", side_effect=fake_extract_with_text) as mk:
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "deepseek-chat", "prompt": "x",
                  "use_text_extraction": "0"},  # toggle off, but server should still pick text path for deepseek
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        deadline = time.time() + 5
        final = None
        while time.time() < deadline:
            poll = client.get(f"/api/jobs/{job_id}").json()
            if poll["status"] in ("done", "error"):
                final = poll
                break
            time.sleep(0.1)

    assert final["status"] == "done"
    assert mk.called, "deepseek should have used the text path"


def test_get_job_returns_evidence_total(client):
    """When the model emits evidence without page numbers, evidence_total
    counts them so the frontend can show the right warning."""
    pdf = _make_pdf_bytes()
    # Model output has 2 evidence entries: one with page, one without
    no_page_result = (
        '{"samples": [{"n": 147}],'
        ' "evidence": ['
        '   {"snippet": "with page", "page": 1, "source": null, "field": "samples[0].n"},'
        '   {"snippet": "without page", "field": "samples[0].n"}'
        ' ]}'
    )

    def fake_extract(**kwargs):
        return (no_page_result, "stop", {"prompt": 1, "completion": 1, "total": 2})

    with patch("jobs.extract_with_images", side_effect=fake_extract):
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x"},
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
        job_id = r.json()["job_id"]

        deadline = time.time() + 5
        final = None
        while time.time() < deadline:
            poll = client.get(f"/api/jobs/{job_id}").json()
            if poll["status"] in ("done", "error"):
                final = poll; break
            time.sleep(0.05)

    assert final is not None
    assert final["status"] == "done"
    # Total = 2 (both have a snippet); count = at least 1 (the one with a page;
    # the orphan may also be recovered if the snippet text appears in the PDF).
    assert final["evidence_total"] == 2
    assert final["evidence_count"] >= 1


def test_get_job_404_on_unknown_id(client):
    r = client.get("/api/jobs/nonexistent-uuid")
    assert r.status_code == 404


def test_extract_job_surfaces_clean_provider_error(client):
    """When the LLM raises with a structured body, the polled job result
    should carry the clean inner message, not the noisy str(exc) wrapper."""
    pdf = _make_pdf_bytes()

    import openai
    import httpx

    def raising_extract_with_images(**kwargs):
        # Build a real OpenAI AuthenticationError with a populated body
        response = httpx.Response(
            status_code=401,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        raise openai.AuthenticationError(
            message="Error code: 401",
            response=response,
            body={"error": {"message": "Incorrect API key provided: sk-test.",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key"}},
        )

    with patch("jobs.extract_with_images", side_effect=raising_extract_with_images):
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x",
                  "use_text_extraction": "0"},
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]

        deadline = time.time() + 5
        final = None
        while time.time() < deadline:
            poll = client.get(f"/api/jobs/{job_id}").json()
            if poll["status"] in ("done", "error"):
                final = poll
                break
            time.sleep(0.1)

    assert final is not None
    assert final["status"] == "error"
    # The clean inner message — not "Error code: 401 - {'error': {'message': ...}}"
    assert final["error"] == "Incorrect API key provided: sk-test."


def test_generate_prompt_surfaces_provider_message(client):
    """A bad request from the provider should reach the user with the actual
    reason ('max_tokens too large', 'image too big', etc.)."""
    import openai
    import httpx

    def raising_generate_text(model, api_key, prompt, temperature=0.3, base_url=None):
        response = httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        raise openai.BadRequestError(
            message="Error code: 400",
            response=response,
            body={"error": {"message": "max_tokens is too large: 999999",
                            "type": "invalid_request_error",
                            "param": "max_tokens"}},
        )

    with patch("server.generate_text", side_effect=raising_generate_text):
        r = client.post("/api/generate-prompt", json={
            "api_key": "k", "model": "gpt-4o-mini",
            "mode": "extraction", "question": "x",
        })
    assert r.status_code == 400
    assert "max_tokens is too large: 999999" in r.json()["error"]


# ── /api/pages (review-only flow) ────────────────────────────────────────────

def test_pages_renders_images_without_result(client):
    pdf = _make_pdf_bytes()
    r = client.post("/api/pages", files={"pdf": ("paper.pdf", pdf, "application/pdf")},
                    data={"result": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "paper.pdf"
    assert len(body["page_images"]) == 1
    assert body["page_images"][0].startswith("data:image/jpeg;base64,")


def test_pages_rejects_non_pdf(client):
    r = client.post("/api/pages", files={"pdf": ("notes.txt", b"hi", "text/plain")},
                    data={"result": ""})
    assert r.status_code == 400


# ── Batch + cancel endpoints ─────────────────────────────────────────────────

def test_extract_creates_batch_and_links_job(client):
    """When a batch_id is provided, the job is linked to it and the batch row is created."""
    pdf = _make_pdf_bytes()
    bid = "test-batch-1"
    with patch("jobs.extract_with_images",
               return_value=('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})):
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x",
                  "use_text_extraction": "0", "batch_id": bid},
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
    assert r.status_code == 200
    assert r.json()["batch_id"] == bid

    # Wait for the worker thread to finish
    deadline = time.time() + 5
    while time.time() < deadline:
        bjr = client.get(f"/api/batches/{bid}").json()
        if bjr["jobs"] and bjr["jobs"][0]["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    detail = client.get(f"/api/batches/{bid}").json()
    assert detail["batch"]["id"] == bid
    assert len(detail["jobs"]) == 1
    assert detail["jobs"][0]["batch_id"] == bid


def test_list_batches_aggregates_counts(client):
    """list_batches returns aggregate counts per batch — feeds the History view."""
    pdf = _make_pdf_bytes()
    with patch("jobs.extract_with_images",
               return_value=('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})):
        client.post("/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x", "batch_id": "b1"},
            files={"pdf": ("a.pdf", pdf, "application/pdf")},
        )
        # Wait for completion
        deadline = time.time() + 5
        while time.time() < deadline:
            j = client.get("/api/batches").json()["batches"]
            if j and (j[0]["n_done"] or 0) == 1:
                break
            time.sleep(0.05)

    listing = client.get("/api/batches").json()["batches"]
    assert len(listing) >= 1
    b = next(x for x in listing if x["id"] == "b1")
    assert b["n_total"] == 1
    assert b["n_done"]  == 1


def test_cancel_endpoint_requests_cancel(client):
    """POST /api/jobs/<id>/cancel sets the cancel flag."""
    import db as _db
    pdf = _make_pdf_bytes()

    # Slow-running mock — gives the cancel time to land before the job finishes
    def slow_extract(**kwargs):
        time.sleep(0.6)
        return ('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})

    with patch("jobs.extract_with_images", side_effect=slow_extract):
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x"},
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
        job_id = r.json()["job_id"]
        # Cancel immediately
        cancel = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel.status_code == 200
        assert cancel.json()["ok"] is True

        # Wait for the worker to settle into a terminal state
        deadline = time.time() + 5
        final = None
        while time.time() < deadline:
            poll = client.get(f"/api/jobs/{job_id}").json()
            if poll["status"] in ("done", "error", "cancelled"):
                final = poll; break
            time.sleep(0.05)
    assert final is not None
    # The status will be "cancelled" if the cancel landed before mark_done,
    # "done" otherwise — either way the cancel flag was accepted.
    assert final["status"] in ("cancelled", "done")


def test_cancel_unknown_job_404(client):
    r = client.post("/api/jobs/doesnt-exist/cancel")
    assert r.status_code == 404


# ── Email notifier ───────────────────────────────────────────────────────────

def test_notifier_logs_when_smtp_unconfigured(monkeypatch, capsys):
    """Without SMTP env vars, the notifier prints to stderr instead of sending."""
    import notifier
    monkeypatch.delenv("PAPERLENS_SMTP_HOST", raising=False)
    notifier.send_batch_complete(
        to="user@example.com",
        batch_id="b-test",
        jobs_in_batch=[{"filename": "a.pdf", "status": "done"},
                       {"filename": "b.pdf", "status": "error", "error": "bad key"}],
    )
    out = capsys.readouterr().err
    assert "user@example.com" in out
    assert "b-test" in out
    assert "a.pdf" in out
    assert "b.pdf" in out


def test_notifier_summary_subject_for_full_success():
    import notifier
    subject, body = notifier._format_summary(
        batch_id="b1",
        jobs_in_batch=[{"filename": "x.pdf", "status": "done"}],
        public_url=None,
    )
    assert "complete" in subject.lower()
    assert "x.pdf" in body


def test_config_endpoint_returns_limits(client):
    """The frontend reads this on load to render 'up to N papers per batch'."""
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["max_batch_papers"], int) and body["max_batch_papers"] >= 1
    assert isinstance(body["max_pdf_bytes"],    int) and body["max_pdf_bytes"]    >= 1024


def test_config_endpoint_respects_env_var(client, monkeypatch):
    monkeypatch.setenv("PAPERLENS_MAX_BATCH_PAPERS", "5")
    r = client.get("/api/config")
    assert r.json()["max_batch_papers"] == 5


def test_extract_rejects_batch_over_limit(client, monkeypatch):
    """Once the cap is hit, the next paper in the same batch_id is refused."""
    monkeypatch.setenv("PAPERLENS_MAX_BATCH_PAPERS", "2")
    pdf = _make_pdf_bytes()
    bid = "test-cap"

    # Two successful submits should be accepted
    with patch("jobs.extract_with_images",
               return_value=('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})):
        for _ in range(2):
            r = client.post(
                "/api/extract",
                data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x", "batch_id": bid},
                files={"pdf": ("paper.pdf", pdf, "application/pdf")},
            )
            assert r.status_code == 200

        # Third one is over the cap → 400
        r = client.post(
            "/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x", "batch_id": bid},
            files={"pdf": ("paper.pdf", pdf, "application/pdf")},
        )
    assert r.status_code == 400
    assert "Batch limit reached" in r.json()["detail"]


def test_set_batch_email_validates(client):
    """Bad email → 400, missing batch → 404."""
    # Set up a real batch first
    pdf = _make_pdf_bytes()
    with patch("jobs.extract_with_images",
               return_value=('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})):
        client.post("/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x", "batch_id": "b-em"},
            files={"pdf": ("a.pdf", pdf, "application/pdf")},
        )
    # Bad email
    r = client.post("/api/batches/b-em/email", json={"email": "not-an-email"})
    assert r.status_code == 400
    # Missing batch
    r = client.post("/api/batches/nonexistent/email", json={"email": "u@example.com"})
    assert r.status_code == 404


def test_set_batch_email_attaches_address_to_batch(client):
    """Happy path: address saves to the batch row, returns ok."""
    import db as _db
    pdf = _make_pdf_bytes()
    with patch("jobs.extract_with_images",
               return_value=('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})):
        client.post("/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x", "batch_id": "b-em2"},
            files={"pdf": ("a.pdf", pdf, "application/pdf")},
        )
    r = client.post("/api/batches/b-em2/email", json={"email": "alice@example.com"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # Verify the column was updated
    batch = _db.get_batch("b-em2")
    assert batch and batch["notify_email"] == "alice@example.com"


def test_set_batch_email_after_completion_sends_immediately(client, capsys):
    """If user adds email after the batch finished, fire the email right away."""
    pdf = _make_pdf_bytes()
    with patch("jobs.extract_with_images",
               return_value=('{"x":1}', "stop", {"prompt": 1, "completion": 1, "total": 2})):
        client.post("/api/extract",
            data={"api_key": "k", "model": "gpt-4o-mini", "prompt": "x", "batch_id": "b-em3"},
            files={"pdf": ("a.pdf", pdf, "application/pdf")},
        )
        # Wait for the worker to finish
        deadline = time.time() + 5
        while time.time() < deadline:
            poll = client.get("/api/batches/b-em3").json()["jobs"]
            if poll and poll[0]["status"] in ("done", "error"):
                break
            time.sleep(0.05)

    # Now attach the email after the fact
    r = client.post("/api/batches/b-em3/email", json={"email": "after@example.com"})
    assert r.status_code == 200
    assert r.json()["sent_now"] is True
    # The notifier dev-mode log should have hit stderr
    time.sleep(0.4)  # let the daemon thread flush
    out = capsys.readouterr().err
    assert "after@example.com" in out


def test_notifier_summary_subject_for_full_failure():
    import notifier
    subject, body = notifier._format_summary(
        batch_id="b1",
        jobs_in_batch=[{"filename": "x.pdf", "status": "error", "error": "401"}],
        public_url=None,
    )
    assert "failed" in subject.lower()
