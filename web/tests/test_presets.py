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
    """The umbrella ``masem`` preset is the Direct-information variant —
    it extracts correlations (matrices + prose) plus metadata, so its
    sub-tabs are ``correlations`` + ``descriptives``.  Declared
    explicitly in masem.json so the preset author controls the layout
    instead of relying on auto-generation from data_sources."""
    r = client.get("/api/presets/masem")
    body = r.json()
    sub_views = body.get("sub_views")
    assert isinstance(sub_views, list)
    assert len(sub_views) == 2
    ids = [s["id"] for s in sub_views]
    assert ids == ["correlations", "descriptives"]
    # Every sub-view has a label and either include_keys or exclude_keys
    for sv in sub_views:
        assert "label" in sv and sv["label"]
        assert ("include_keys" in sv) or ("exclude_keys" in sv)
    by_id = {s["id"]: s for s in sub_views}
    # The Correlations tab carries both correlation data sources so a
    # single tab covers prose- and table-reported correlations.
    assert "correlation_matrix"   in by_id["correlations"]["include_keys"]
    assert "single_correlations"  in by_id["correlations"]["include_keys"]
    assert "correlation_matrix"   in by_id["descriptives"]["exclude_keys"]
    assert "single_correlations"  in by_id["descriptives"]["exclude_keys"]
    # Evidence-key narrowing: highlights for the Correlations tab scoped
    # to both correlation sources.
    assert sorted(by_id["correlations"]["evidence_keys"]) == [
        "correlation_matrix",
        "single_correlations",
    ]
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


def test_umbrella_masem_is_blank_correlations_starter(client):
    """The umbrella ``masem`` preset is the Direct-information variant:
    it extracts correlations directly (matrix + prose) plus metadata.
    No factor-analysis fields are pre-baked — the matching Indirect-
    information starter (``masem-tas20``) covers that path."""
    r = client.get("/api/presets/masem")
    body = r.json()
    p = body["template_params"]
    assert "correlation_matrix"   in p["data_sources"]
    assert "single_correlations"  in p["data_sources"]
    # Factor-analysis fields stay blank in the Direct variant — those
    # are the Indirect variant's territory.
    assert p.get("factor_naming") in ([], None)
    assert p.get("cfa_item_assignment") in ({}, None)
    assert p.get("item_texts") in ([], None)
    # Rendered prompt uses the generic scale-name placeholder verbiage.
    prompt = body["prompt"]
    assert "the target scale" in prompt or "target scale" in prompt
    # Sub-views: Correlations + Descriptives.
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["correlations", "descriptives"]


def test_tas20_variant_renders_with_pre_baked_scaffold(client):
    """The TAS-20 sub-preset ships the TAS-20 scaffold inside the SCALE
    SPECIFICATION header: scale name, item count, max factors, the 20
    item texts, and the auto-generated factor_key_mapping.  The
    factor_labels block (DIF / DDF / EOT) was removed because it
    confused the model on papers using non-standard factor names."""
    r = client.get("/api/presets/masem-tas20")
    assert r.status_code == 200
    body = r.json()
    prompt = body["prompt"]
    # SCALE SPECIFICATION header + values
    assert "# SCALE SPECIFICATION" in prompt
    assert "[scale_name]: Toronto Alexithymia Scale (TAS-20)" in prompt
    assert "[n_items]: 20" in prompt
    assert "[n_factors_max]: 5" in prompt
    # factor_labels block must NOT be in the prompt anymore
    assert "[factor_labels]" not in prompt
    assert "F1 = DIF" not in prompt
    # Factor-key mapping auto-generated for the 5 factors — still there
    assert "[factor_key_mapping]" in prompt
    assert "F-I, FI, Factor I, Factor 1, Component 1 -> F1" in prompt
    assert "F-V, FV, Factor V, Factor 5, Component 5 -> F5" in prompt
    # Item-text list — first numbered line + last item
    assert "1: I am often confused about what emotion I am feeling." in prompt
    assert "20: Looking for hidden meanings in movies or plays distracts from their enjoyment." in prompt
    # The new template adds the confidence-self-assessment step
    assert "## STEP 9: Self-assess extraction confidence" in prompt
    assert '"extraction_confidence"' in prompt
    # Sub-views match TAS-20 (factor loadings + correlations)
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["loadings", "correlations", "descriptives"]


def test_tas20_variant_includes_user_supplied_item_texts(client):
    """Users paste their own item texts into the Item labels textarea.
    When that posts back through /api/build-preset-prompt with the
    TAS-20 starter, the rendered prompt must surface those item texts
    in the SCALE SPECIFICATION block."""
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
    assert "[item_labels]:" in prompt
    assert "1: User-pasted item 1" in prompt
    assert "2: User-pasted item 20" in prompt


def test_extraction_confidence_block_in_default_prompt(client):
    """The new default template instructs the model to self-assess its
    extraction confidence for loadings / correlations / metadata and
    emit an ``extraction_confidence`` object in every sample."""
    r = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem", "template_params": {},
    })
    body = r.json()
    prompt = body["prompt"]
    # Step 9 + the three required category keys appear in the prompt
    assert "## STEP 9: Self-assess extraction confidence" in prompt
    assert "``factor_loadings``" in prompt
    assert "``factor_correlations``" in prompt
    assert "``metadata``" in prompt
    # The output-schema example carries the extraction_confidence block
    assert '"extraction_confidence"' in prompt
    # The three category-value strings the model must use
    assert '"high"' in prompt
    assert '"medium"' in prompt
    assert '"low"' in prompt


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
            "scale_name":            "Toronto Alexithymia Scale (TAS-20)",
            "instrument_name":       "Toronto Alexithymia Scale (TAS-20)",
            "instrument_name_long":  "Toronto Alexithymia Scale (TAS-20)",
            "n_items":               20,
            "n_factors":             5,
            "content_scope":         "concrete_items",
        },
    })
    assert r.status_code == 200
    body = r.json()
    prompt = body["prompt"]
    # The new template structures loadings + correlations as Step 5 + 6.
    assert "## STEP 5: Extract factor loadings" in prompt
    assert "## STEP 6: Extract factor correlations" in prompt
    # Scale-specification values flow through to the prompt body.
    assert "[scale_name]: Toronto Alexithymia Scale (TAS-20)" in prompt
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
