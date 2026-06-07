"""Unit tests for datasets.py — the /api/datasets discovery endpoint.

Focus areas:
- ``password_hash`` is stripped from every returned record
- the cache honours its TTL and can be invalidated
- malformed metadata files are skipped, not 500ed
- the GitHub Contents API is correctly walked (folders → metadata.json)
"""
from __future__ import annotations

import base64
import json
from unittest.mock import patch

import pytest

import datasets as datasets_mod


# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64(payload: dict) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _stub_metadata(**overrides) -> dict:
    base = {
        "schema_version":  "metadata-v1",
        "dataset_id":      "ncs-18-2026-06",
        "title":           "NCS-18 factor loadings",
        "description":     "Pilot dataset",
        "donor":           {"mode": "anonymous"},
        "visibility":      "public",
        "extraction": {
            "schema_version": "masem-v3",
            "paper_count":    2,
            "model_used":     ["gpt-4o"],
            "prompt_sha256":  "abc123",
        },
        "created_at":      "2026-06-03T17:43:20Z",
    }
    base.update(overrides)
    return base


class _FakeResponse:
    """Minimal httpx-Response stand-in for the patched client."""
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class _FakeClient:
    """Patches in for httpx.Client; routes URLs to canned responses."""
    def __init__(self, routes: dict[str, _FakeResponse]):
        self._routes = routes
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, headers=None, **_):
        self.calls.append(url)
        return self._routes.get(url, _FakeResponse(404, {}))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch):
    """Each test starts with a cold cache + minimal env config."""
    datasets_mod.invalidate_cache()
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    monkeypatch.delenv("PAPERLENS_DONATE_ENABLED", raising=False)
    yield
    datasets_mod.invalidate_cache()


# ── Tests ────────────────────────────────────────────────────────────────────

def test_strip_password_hash_removes_hash_adds_gated_flag():
    full = {"title": "X", "password_hash": "$2b$12$...", "donor": {"mode": "anonymous"}}
    out = datasets_mod._strip_password_hash(full)
    assert "password_hash" not in out
    assert out["gated"] is True

    public = {"title": "Y", "donor": {"mode": "anonymous"}}
    out = datasets_mod._strip_password_hash(public)
    assert "password_hash" not in out
    assert out["gated"] is False


def test_parse_one_handles_malformed_blob():
    # Not valid base64 → None, not a crash
    assert datasets_mod._parse_one("not-base64-at-all!!") is None
    # Valid base64 but not JSON → None
    bad = base64.b64encode(b"<html>nope</html>").decode("ascii")
    assert datasets_mod._parse_one(bad) is None


def test_list_datasets_strips_password_hash_end_to_end(monkeypatch):
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [{"name": "ncs-18-2026-06", "type": "dir"}],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata(password_hash="$2b$12$SECRET-HASH-HERE"))},
        ),
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))
    result = datasets_mod.list_datasets(force_refresh=True)
    assert len(result["datasets"]) == 1
    ds = result["datasets"][0]
    assert "password_hash" not in ds
    assert ds["gated"] is True
    assert ds["title"] == "NCS-18 factor loadings"
    assert ds["github_url"].endswith("/datasets/ncs-18-2026-06")
    assert ds["paper_count"] == 2
    assert ds["schema_version"] == "masem-v3"
    assert ds["model_used"] == ["gpt-4o"]


def test_list_datasets_public_records_have_gated_false(monkeypatch):
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [{"name": "open-2026-06", "type": "dir"}],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/open-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata())},
        ),
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))
    result = datasets_mod.list_datasets(force_refresh=True)
    assert result["datasets"][0]["gated"] is False


def test_list_datasets_empty_when_folder_missing(monkeypatch):
    """A repo that exists but has no datasets/ folder yet should return
    an empty list, not error."""
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            404, {},
        ),
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))
    result = datasets_mod.list_datasets(force_refresh=True)
    assert result["datasets"] == []


def test_list_datasets_skips_malformed_metadata(monkeypatch):
    """One folder with bad metadata.json shouldn't drop the others."""
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [
                {"name": "good-2026-06", "type": "dir"},
                {"name": "broken-2026-06", "type": "dir"},
                {"name": "missing-2026-06", "type": "dir"},
            ],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/good-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata(dataset_id="good-2026-06"))},
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/broken-2026-06/metadata.json": _FakeResponse(
            200, {"content": "this-is-not-base64-json!!"},
        ),
        # missing-2026-06 has no metadata.json at all → default 404
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))
    result = datasets_mod.list_datasets(force_refresh=True)
    ids = [d["dataset_id"] for d in result["datasets"]]
    assert "good-2026-06" in ids
    assert "broken-2026-06" not in ids
    assert "missing-2026-06" not in ids


def test_list_datasets_sorted_newest_first(monkeypatch):
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [
                {"name": "old-2025-01", "type": "dir"},
                {"name": "new-2026-06", "type": "dir"},
            ],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/old-2025-01/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata(dataset_id="old-2025-01", created_at="2025-01-15T00:00:00Z"))},
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/new-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata(dataset_id="new-2026-06", created_at="2026-06-04T00:00:00Z"))},
        ),
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))
    result = datasets_mod.list_datasets(force_refresh=True)
    ids = [d["dataset_id"] for d in result["datasets"]]
    assert ids == ["new-2026-06", "old-2025-01"]


def test_cache_returns_same_payload_second_call(monkeypatch):
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [{"name": "ncs-18-2026-06", "type": "dir"}],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata())},
        ),
    }
    fake = _FakeClient(routes)
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: fake)

    first  = datasets_mod.list_datasets()
    assert first["cache_age_sec"] == 0
    first_call_count = len(fake.calls)

    second = datasets_mod.list_datasets()
    # Cached → no further HTTP calls.
    assert len(fake.calls) == first_call_count
    assert second["datasets"] == first["datasets"]
    assert second["cache_age_sec"] >= 0


def test_api_datasets_refresh_param_bypasses_cache(monkeypatch, tmp_path):
    """?refresh=1 should re-fetch from GitHub even when the cache is
    warm.  Without it the second call returns cached data; with it
    we expect another round of HTTP calls."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [{"name": "ncs-18-2026-06", "type": "dir"}],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata())},
        ),
    }
    fake = _FakeClient(routes)
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: fake)

    client = TestClient(server.app)

    # Warm cache.
    client.get("/api/datasets")
    n_warm = len(fake.calls)
    assert n_warm > 0

    # Cached fetch — no new HTTP calls.
    client.get("/api/datasets")
    assert len(fake.calls) == n_warm

    # Force refresh — new HTTP calls.
    r = client.get("/api/datasets?refresh=1")
    assert r.status_code == 200
    assert r.json()["cache_age_sec"] == 0
    assert len(fake.calls) > n_warm


def test_cache_invalidate_forces_refetch(monkeypatch):
    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [{"name": "ncs-18-2026-06", "type": "dir"}],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata())},
        ),
    }
    fake = _FakeClient(routes)
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: fake)

    datasets_mod.list_datasets()
    n_before = len(fake.calls)
    datasets_mod.invalidate_cache()
    datasets_mod.list_datasets()
    assert len(fake.calls) > n_before, "invalidate_cache should force a re-fetch"


def test_gh_repo_unset_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("PAPERLENS_GH_REPO", raising=False)
    with pytest.raises(RuntimeError, match="PAPERLENS_GH_REPO"):
        datasets_mod._gh_repo()


# ── Route-level integration test via the FastAPI TestClient ──────────────────

def test_api_datasets_endpoint(monkeypatch, tmp_path):
    """Confirm /api/datasets is wired correctly and password_hash never
    appears in the response body."""
    import os, tempfile
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    routes = {
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets": _FakeResponse(
            200, [{"name": "ncs-18-2026-06", "type": "dir"}],
        ),
        "https://api.github.com/repos/test-owner/test-repo/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata(password_hash="$2b$12$SECRET"))},
        ),
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))

    client = TestClient(server.app)
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    assert len(body["datasets"]) == 1
    ds = body["datasets"][0]
    # The hash must NEVER appear, even though the source metadata has one.
    assert "password_hash" not in ds
    assert ds["gated"] is True
    # The raw response body string must also not contain the hash literal.
    assert "$2b$12$SECRET" not in r.text


def test_api_datasets_handles_unconfigured_repo(monkeypatch, tmp_path):
    """When PAPERLENS_GH_REPO isn't set, the endpoint should return 503
    (so the frontend can degrade gracefully) instead of bubbling a 500."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.delenv("PAPERLENS_GH_REPO", raising=False)
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    client = TestClient(server.app)
    r = client.get("/api/datasets")
    assert r.status_code == 503
    assert "PAPERLENS_GH_REPO" in r.json()["detail"]


# ── /api/datasets/{id}/verify-password (Phase 3c) ───────────────────────────

def _verify_client(monkeypatch, tmp_path, *, metadata):
    """Build a TestClient with the env, DB, and GitHub mock pre-configured
    so the verify-password tests don't repeat 10 lines of setup each."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    monkeypatch.setenv("PAPERLENS_DONATE_IP_PEPPER", "test-pepper-verify")
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    # Reset the in-memory rate-limit table between tests so attempts
    # from a previous test don't leak in.
    server._VERIFY_ATTEMPTS.clear()

    if metadata is None:
        routes = {
            "https://api.github.com/repos/test-owner/test-repo"
            "/contents/datasets/missing-2026-06/metadata.json": _FakeResponse(404, {}),
        }
    else:
        slug = metadata.get("dataset_id", "ncs-18-2026-06")
        routes = {
            f"https://api.github.com/repos/test-owner/test-repo"
            f"/contents/datasets/{slug}/metadata.json": _FakeResponse(
                200, {"content": _b64(metadata)},
            ),
        }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))
    return TestClient(server.app)


def test_verify_password_correct_returns_ok_true(monkeypatch, tmp_path):
    """Happy path — a gated dataset with the right password verifies."""
    import donor
    hashed = donor.hash_password("correct horse battery staple")
    metadata = _stub_metadata(
        dataset_id="ncs-18-2026-06",
        password_hash=hashed,
        visibility="gated",
    )
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)
    r = client.post("/api/datasets/ncs-18-2026-06/verify-password",
                    json={"password": "correct horse battery staple"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["gated"] is True


def test_verify_password_wrong_returns_ok_false(monkeypatch, tmp_path):
    """Wrong password returns 200 with ok=false (not 401) — by design
    so the frontend can show 'incorrect password' inline without
    treating it as an auth error."""
    import donor
    hashed = donor.hash_password("the right one")
    metadata = _stub_metadata(
        dataset_id="ncs-18-2026-06",
        password_hash=hashed,
        visibility="gated",
    )
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)
    r = client.post("/api/datasets/ncs-18-2026-06/verify-password",
                    json={"password": "wrong guess"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["gated"] is True


def test_verify_password_public_dataset_returns_ok(monkeypatch, tmp_path):
    """A dataset without a password_hash is public — any non-empty
    password verifies (the frontend shouldn't actually call this for
    public datasets, but be lenient)."""
    metadata = _stub_metadata(
        dataset_id="open-2026-06",
        visibility="public",
    )
    # Important — no password_hash key in the metadata.
    metadata.pop("password_hash", None)
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)
    r = client.post("/api/datasets/open-2026-06/verify-password",
                    json={"password": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["gated"] is False


def test_verify_password_missing_dataset_returns_404(monkeypatch, tmp_path):
    client = _verify_client(monkeypatch, tmp_path, metadata=None)
    r = client.post("/api/datasets/missing-2026-06/verify-password",
                    json={"password": "guess"})
    assert r.status_code == 404


def test_verify_password_empty_password_returns_400(monkeypatch, tmp_path):
    metadata = _stub_metadata(dataset_id="ncs-18-2026-06", password_hash="$2b$12$x")
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)
    r = client.post("/api/datasets/ncs-18-2026-06/verify-password",
                    json={"password": ""})
    assert r.status_code == 400


def test_verify_password_invalid_slug_returns_400(monkeypatch, tmp_path):
    """Path-traversal-shaped slugs should be rejected before the lookup
    hits GitHub.  FastAPI's path param won't accept literal slashes,
    so we test with characters that are technically valid in a URL
    path but not in our slug format."""
    metadata = _stub_metadata()
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)
    # Empty-string slug isn't actually routable (the path becomes
    # /api/datasets//verify-password which 404s the route).  Test a
    # malformed slug instead (UPPERCASE characters aren't in our
    # _SAFE_SLUG pattern).
    r = client.post("/api/datasets/UPPERCASE-2026-06/verify-password",
                    json={"password": "x"})
    # The lookup returns None (slug rejected) → 404.
    assert r.status_code == 404


def test_verify_password_rate_limit_after_10_attempts(monkeypatch, tmp_path):
    """11th attempt within an hour should 429 even with the right
    password — the limit applies regardless of correctness."""
    import donor
    hashed = donor.hash_password("right")
    metadata = _stub_metadata(
        dataset_id="ncs-18-2026-06",
        password_hash=hashed,
        visibility="gated",
    )
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)

    # 10 wrong attempts — all 200 ok=false.  Pass an explicit
    # non-loopback IP so the limiter applies (loopback bypasses).
    headers = {"X-Forwarded-For": "203.0.113.7"}
    for i in range(10):
        r = client.post("/api/datasets/ncs-18-2026-06/verify-password",
                        json={"password": f"wrong{i}"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["ok"] is False

    # 11th — even with the RIGHT password, the limiter trips.
    r = client.post("/api/datasets/ncs-18-2026-06/verify-password",
                    json={"password": "right"}, headers=headers)
    assert r.status_code == 429
    assert "Too many" in r.json()["detail"]


def test_get_dataset_full_returns_metadata_and_prompt(monkeypatch, tmp_path):
    """Happy path: /api/datasets/{id}/full returns the sanitized
    metadata + the prompt.md body, no password_hash leakage."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    metadata = _stub_metadata(
        dataset_id="ncs-18-2026-06",
        password_hash="$2b$12$SECRET-NEVER-LEAKS",
    )
    routes = {
        "https://api.github.com/repos/test-owner/test-repo"
        "/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(metadata)},
        ),
        "https://api.github.com/repos/test-owner/test-repo"
        "/contents/datasets/ncs-18-2026-06/prompt.md": _FakeResponse(
            200, {"content": base64.b64encode(b"You are an expert...").decode("ascii")},
        ),
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))

    client = TestClient(server.app)
    r = client.get("/api/datasets/ncs-18-2026-06/full")
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == "You are an expert..."
    assert body["title"] == "NCS-18 factor loadings"
    assert body["schema_version"] == "masem-v3"
    assert body["model_used"] == ["gpt-4o"]
    assert "password_hash" not in body
    assert "$2b$12$SECRET-NEVER-LEAKS" not in r.text


def test_get_dataset_full_missing_returns_404(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    routes = {}  # default → 404 on every URL
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))

    client = TestClient(server.app)
    r = client.get("/api/datasets/missing-2026-06/full")
    assert r.status_code == 404


def test_get_dataset_full_handles_missing_prompt_file(monkeypatch, tmp_path):
    """If metadata exists but prompt.md doesn't, return the metadata
    with prompt='' rather than 404 — the user can still extend
    by re-entering the prompt manually."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("PAPERLENS_GH_REPO", "test-owner/test-repo")
    import db; db.init()
    from fastapi.testclient import TestClient
    import server

    routes = {
        "https://api.github.com/repos/test-owner/test-repo"
        "/contents/datasets/ncs-18-2026-06/metadata.json": _FakeResponse(
            200, {"content": _b64(_stub_metadata())},
        ),
        # No prompt.md → 404 by default.
    }
    monkeypatch.setattr(datasets_mod.httpx, "Client", lambda *a, **kw: _FakeClient(routes))

    client = TestClient(server.app)
    r = client.get("/api/datasets/ncs-18-2026-06/full")
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == ""
    assert body["title"] == "NCS-18 factor loadings"


def test_verify_password_loopback_bypasses_rate_limit(monkeypatch, tmp_path):
    """Localhost dev iteration must not be blocked by the limiter.
    TestClient sets request.client.host to 'testclient' (not 127.0.0.1)
    so we simulate a loopback peer via X-Forwarded-For — mirrors how
    a real local-dev request looks once Fly's proxy stamps the header
    in production."""
    import donor
    hashed = donor.hash_password("right")
    metadata = _stub_metadata(
        dataset_id="ncs-18-2026-06",
        password_hash=hashed,
        visibility="gated",
    )
    client = _verify_client(monkeypatch, tmp_path, metadata=metadata)

    headers = {"X-Forwarded-For": "127.0.0.1"}
    for _ in range(12):
        r = client.post("/api/datasets/ncs-18-2026-06/verify-password",
                        json={"password": "wrong"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["ok"] is False
