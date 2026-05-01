"""Tests for the domain-workflow preset system."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Fresh TestClient with a fresh DB.  Reuses the standard fixture pattern
    from test_routes.py so the preset endpoints get the same setup."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    import db
    import server
    db.init()
    return TestClient(server.app)


# ── Loader: fail-soft on bad inputs ──────────────────────────────────────────

def test_loader_skips_invalid_json(monkeypatch, tmp_path, capsys):
    """A malformed JSON file should not crash discovery."""
    import presets_loader
    bad = tmp_path / "bogus.json"
    bad.write_text("{not valid json")
    monkeypatch.setattr(presets_loader, "PRESETS_DIR", tmp_path)
    out = presets_loader.load_all()
    assert out == {}
    err = capsys.readouterr().err
    assert "skipping" in err


def test_loader_rejects_missing_required_keys(monkeypatch, tmp_path, capsys):
    import presets_loader
    p = tmp_path / "incomplete.json"
    p.write_text(json.dumps({"id": "x"}))   # missing title, tagline, mode, prompt
    monkeypatch.setattr(presets_loader, "PRESETS_DIR", tmp_path)
    assert presets_loader.load_all() == {}
    assert "missing required keys" in capsys.readouterr().err


def test_loader_resolves_prompt_file(monkeypatch, tmp_path):
    """A preset that references a sibling prompt file should have the body
    inlined into the returned dict under 'prompt'."""
    import presets_loader
    (tmp_path / "p.prompt.md").write_text("PROMPT BODY")
    (tmp_path / "p.json").write_text(json.dumps({
        "id":          "p",
        "title":       "P",
        "tagline":     "t",
        "mode":        "extraction",
        "prompt_file": "p.prompt.md",
    }))
    monkeypatch.setattr(presets_loader, "PRESETS_DIR", tmp_path)
    out = presets_loader.load_all()
    assert "p" in out
    assert out["p"]["prompt"] == "PROMPT BODY"
    assert "prompt_file" not in out["p"]


def test_loader_blocks_path_traversal_in_prompt_file(monkeypatch, tmp_path, capsys):
    """A prompt_file that escapes the presets dir should be rejected."""
    import presets_loader
    # Point PRESETS_DIR at a sub-directory; the prompt_file uses ../ to escape
    sub = tmp_path / "presets"
    sub.mkdir()
    (tmp_path / "secret.txt").write_text("escaped content")
    (sub / "evil.json").write_text(json.dumps({
        "id":          "evil",
        "title":       "Evil",
        "tagline":     "t",
        "mode":        "extraction",
        "prompt_file": "../secret.txt",
    }))
    monkeypatch.setattr(presets_loader, "PRESETS_DIR", sub)
    out = presets_loader.load_all()
    assert "evil" not in out
    err = capsys.readouterr().err
    assert "outside presets dir" in err or "empty" in err


# ── /api/presets — list endpoint ────────────────────────────────────────────

def test_list_presets_includes_masem(client):
    """The shipped masem.json should be discoverable."""
    r = client.get("/api/presets")
    assert r.status_code == 200
    ids = [p["id"] for p in r.json()["presets"]]
    assert "masem" in ids


def test_list_presets_excludes_prompt_body(client):
    """The list endpoint returns only branding/summary fields — keeps payload small."""
    r = client.get("/api/presets")
    for p in r.json()["presets"]:
        # Required summary fields
        for key in ("id", "title", "tagline"):
            assert key in p
        # The full prompt body must NOT be in the list response
        assert "prompt" not in p


# ── /api/presets/{id} — detail endpoint ─────────────────────────────────────

def test_get_preset_returns_full_dict(client):
    r = client.get("/api/presets/masem")
    assert r.status_code == 200
    body = r.json()
    # All of these are part of the contract the frontend relies on
    for key in ("id", "title", "tagline", "mode", "default_provider",
                "default_model", "prompt", "skip_to"):
        assert key in body, f"masem preset missing {key!r}"
    assert body["id"] == "masem"
    # The prompt body must include the evidence appendix content (since
    # presets bypass the LLM-generation step that normally appends it)
    prompt = body["prompt"]
    # Some preset shape that the renderer can table-ify — either the explicit
    # `_table` marker OR the dotted-key format (F1.1, R1.2, ...)
    assert ("_table" in prompt) or ("F1.1" in prompt) or ("R1.2" in prompt)
    assert "evidence" in prompt.lower()
    assert "snippet" in prompt.lower()
    assert "page" in prompt.lower()


def test_get_preset_404_for_unknown(client):
    r = client.get("/api/presets/does-not-exist")
    assert r.status_code == 404


def test_masem_preset_declares_sub_views(client):
    """The MASEM preset ships with three sub-tabs the frontend renders under
    the active paper card.  Verify the contract so the JS doesn't blow up."""
    r = client.get("/api/presets/masem")
    body = r.json()
    sub_views = body.get("sub_views")
    assert isinstance(sub_views, list)
    assert len(sub_views) == 3
    ids = [s["id"] for s in sub_views]
    assert ids == ["loadings", "correlations", "descriptives"]
    # Every sub-view has a label and either include_keys or exclude_keys
    for sv in sub_views:
        assert "label" in sv and sv["label"]
        assert ("include_keys" in sv) or ("exclude_keys" in sv)
    # Loadings restricts to factor_loadings; descriptives excludes both tables
    by_id = {s["id"]: s for s in sub_views}
    assert "factor_loadings"      in by_id["loadings"]["include_keys"]
    assert "factor_correlations"  in by_id["correlations"]["include_keys"]
    assert "factor_loadings"      in by_id["descriptives"]["exclude_keys"]
    assert "factor_correlations"  in by_id["descriptives"]["exclude_keys"]
    # Evidence-key narrowing: when on the Loadings sub-view, page-nav and
    # rect overlays should be scoped strictly to ``factor_loadings`` (not
    # also to ``sample_id`` / ``n``, which live in include_keys for data
    # display only).  Same for correlations.
    assert by_id["loadings"]["evidence_keys"]     == ["factor_loadings"]
    assert by_id["correlations"]["evidence_keys"] == ["factor_correlations"]
    # Descriptives doesn't need narrowing — exclude_keys already does the job.
    assert "evidence_keys" not in by_id["descriptives"]
