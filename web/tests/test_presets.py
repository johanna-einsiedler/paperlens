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
    """The MASEM preset ships with sub-tabs the frontend renders under
    the active paper card.  Auto-generated from ``data_sources`` — the
    umbrella ``masem`` preset is general MASEM, so its tabs are
    ``corrmatrix`` + ``singlecorrs`` (not the TAS-20 loadings tabs)."""
    r = client.get("/api/presets/masem")
    body = r.json()
    sub_views = body.get("sub_views")
    assert isinstance(sub_views, list)
    assert len(sub_views) == 3
    ids = [s["id"] for s in sub_views]
    assert ids == ["corrmatrix", "singlecorrs", "descriptives"]
    # Every sub-view has a label and either include_keys or exclude_keys
    for sv in sub_views:
        assert "label" in sv and sv["label"]
        assert ("include_keys" in sv) or ("exclude_keys" in sv)
    by_id = {s["id"]: s for s in sub_views}
    assert "correlation_matrix"   in by_id["corrmatrix"]["include_keys"]
    assert "single_correlations"  in by_id["singlecorrs"]["include_keys"]
    assert "correlation_matrix"   in by_id["descriptives"]["exclude_keys"]
    assert "single_correlations"  in by_id["descriptives"]["exclude_keys"]
    # Evidence-key narrowing: highlights for each sub-view scoped strictly
    # to the data source it's about.
    assert by_id["corrmatrix"]["evidence_keys"]   == ["correlation_matrix"]
    assert by_id["singlecorrs"]["evidence_keys"]  == ["single_correlations"]
    # Descriptives doesn't need narrowing — exclude_keys already does the job.
    assert "evidence_keys" not in by_id["descriptives"]


# ── Variants (TAS-20 example, hidden from landing) ────────────────────────

def test_landing_lists_only_umbrella_masem(client):
    """The landing-list endpoint should surface only the parent ``masem``
    preset; variant starters (e.g. ``masem-tas20``) are reachable via
    the in-app builder but should not clutter the landing-screen
    workflow picker."""
    r = client.get("/api/presets")
    ids = [p["id"] for p in r.json()["presets"]]
    assert "masem" in ids
    assert "masem-tas20" not in ids


def test_variant_preset_still_fetchable_by_id(client):
    """Hidden landing presets must still be loadable by id (the in-app
    builder posts to /api/build-preset-prompt with the variant id)."""
    r = client.get("/api/presets/masem-tas20")
    assert r.status_code == 200
    body = r.json()
    assert body.get("landing_hidden") is True


def test_umbrella_masem_is_general_scope(client):
    """The umbrella ``masem`` preset is now the general MASEM workflow
    (theoretical-constructs scope, correlation matrices + prose
    correlations, no factor loadings by default)."""
    r = client.get("/api/presets/masem")
    body = r.json()
    p = body["template_params"]
    assert p["content_scope"] == "theoretical_constructs"
    assert "correlation_matrix" in p["data_sources"]
    assert "single_correlations" in p["data_sources"]
    assert "factor_loadings" not in p["data_sources"]
    # Rendered prompt has the correlation sections, not the factor
    # loadings / correlations sections.
    prompt = body["prompt"]
    assert "## Correlation matrix" in prompt
    assert "## Single correlations from prose" in prompt
    assert "Factor loadings (`factor_loadings`)" not in prompt
    # Sub-views match the general data sources.
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["corrmatrix", "singlecorrs", "descriptives"]


def test_tas20_variant_renders_without_pre_baked_item_texts(client):
    """The TAS-20 sub-preset sets up the structural scaffold (factor
    loadings + factor correlations, DIF/DDF/EOT factor names, the
    standard CFA item-to-factor fallback) but does NOT ship the
    verbatim TAS-20 item content — those are copyrighted and users
    paste their own copy into section C of the builder if they want
    semantic-content matching."""
    r = client.get("/api/presets/masem-tas20")
    assert r.status_code == 200
    body = r.json()
    prompt = body["prompt"]
    # The TAS-20 SCAFFOLD is in the prompt
    assert "Toronto Alexithymia Scale (TAS-20, 20 items) factor-analytic data" in prompt
    assert "F1 = **DIF** (Difficulty Identifying Feelings)" in prompt
    assert "F1 = items 1, 3, 6, 7, 9, 13, 14" in prompt
    assert "Ignore any solution that includes items from measures other than the TAS-20." in prompt
    # But the verbatim copyrighted item content is OUT
    assert "I am often confused about what emotion I am feeling" not in prompt
    assert "Looking for hidden meanings in movies or plays distracts" not in prompt
    assert "Reference item texts" not in prompt
    # Sub-views match TAS-20 (factor loadings + correlations)
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["loadings", "correlations", "descriptives"]


def test_tas20_variant_includes_user_supplied_item_texts(client):
    """Users paste their own item texts into section C of the builder.
    When that posts back through /api/build-preset-prompt with the
    TAS-20 starter, the rendered prompt must surface those item texts
    in the reference block."""
    r = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem-tas20",
        "template_params": {
            "item_texts": [
                "User-pasted item 1",
                "User-pasted item 20",
            ],
            "include_item_texts": True,
        },
    })
    assert r.status_code == 200
    prompt = r.json()["prompt"]
    assert "Reference item texts" in prompt
    assert "User-pasted item 1" in prompt
    assert "User-pasted item 20" in prompt


def test_study_characteristics_block_renders_when_provided(client):
    """The new section D in the builder feeds free-form study context
    into the rendered prompt under "## About these studies".  Empty
    text → no block emitted."""
    # No D-text → no block
    r1 = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem", "template_params": {},
    })
    assert "## About these studies" not in r1.json()["prompt"]
    # With D-text → block appears
    r2 = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem",
        "template_params": {
            "study_characteristics_text":
                "Studies are mostly Korean and Japanese TAS-20 versions.",
        },
    })
    body = r2.json()
    assert "## About these studies" in body["prompt"]
    assert "Korean and Japanese TAS-20 versions" in body["prompt"]


# ── /api/build-preset-prompt — guided builder render route ─────────────────

def test_build_preset_prompt_default_matches_get(client):
    """Posting an empty ``template_params`` should re-render the preset's
    own defaults — i.e. produce the same prompt as ``GET /api/presets/<id>``."""
    direct = client.get("/api/presets/masem").json()
    built  = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem",
        "template_params": {},
    }).json()
    assert built["prompt"]    == direct["prompt"]
    assert built["sub_views"] == direct["sub_views"]


def test_build_preset_prompt_overrides_data_sources(client):
    """User-supplied ``data_sources`` overrides the preset's defaults so
    the form can flip data sources on/off without picking a different
    starter preset."""
    r = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem",
        "template_params": {
            "data_sources": ["factor_loadings", "factor_correlations"],
            "instrument_name":      "TAS-20",
            "instrument_name_long": "Toronto Alexithymia Scale",
            "n_items":               20,
            "n_factors":             5,
            "content_scope":         "concrete_items",
        },
    })
    assert r.status_code == 200
    body = r.json()
    assert "## Factor loadings" in body["prompt"]
    assert "## Factor correlations" in body["prompt"]
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["loadings", "correlations", "descriptives"]


def test_build_preset_prompt_404_for_unknown(client):
    r = client.post("/api/build-preset-prompt",
                    json={"preset_id": "does-not-exist", "template_params": {}})
    assert r.status_code == 404


def test_build_preset_prompt_400_for_inline_prompt_preset(client, tmp_path, monkeypatch):
    """Presets that ship an inline ``prompt`` (no template file) cannot be
    re-rendered via the builder — return 400 with a clear message."""
    # Drop a quick preset with no template into the loader's directory
    import presets_loader, server
    # Use the existing preset dir but add a temporary file
    pdir = presets_loader.PRESETS_DIR
    inline = pdir / "_test_inline.json"
    import json as _json
    inline.write_text(_json.dumps({
        "id":      "_test_inline",
        "title":   "T",
        "tagline": "t",
        "mode":    "extraction",
        "prompt":  "PROMPT BODY",
    }))
    try:
        r = client.post("/api/build-preset-prompt",
                        json={"preset_id": "_test_inline", "template_params": {}})
        assert r.status_code == 400
    finally:
        inline.unlink(missing_ok=True)
