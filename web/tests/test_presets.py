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
    # `_table` marker, the dotted-key format (F1.1, R1.2), or the new Direct
    # ``records[]`` array of objects.
    assert ("_table" in prompt) or ("F1.1" in prompt) or ("R1.2" in prompt) or ("records" in prompt)
    assert "evidence" in prompt.lower()
    assert "snippet" in prompt.lower()
    assert "page" in prompt.lower()


def test_get_preset_404_for_unknown(client):
    r = client.get("/api/presets/does-not-exist")
    assert r.status_code == 404


def test_masem_preset_declares_sub_views(client):
    """The umbrella ``masem`` preset (Direct-information variant) ships
    two sub-tabs: Effect sizes (the per-sample ``records[]`` array) and
    Descriptives (everything else).  Reliability and instrument fields
    live inline on each record (``rel1``, ``rel2``, ``instr1``,
    ``instr2``) rather than in a separate top-level array, so there is
    no longer a separate Reliabilities tab.  Each sub-view declares
    its ``confidence_keys`` so the confidence-badge row above the data
    panel only shows the ratings that apply to the active tab
    (Effect sizes → effect_sizes + reliabilities; Descriptives →
    metadata).  Declared explicitly in masem.json."""
    r = client.get("/api/presets/masem")
    body = r.json()
    sub_views = body.get("sub_views")
    assert isinstance(sub_views, list)
    assert len(sub_views) == 2
    ids = [s["id"] for s in sub_views]
    assert ids == ["effectsizes", "descriptives"]
    for sv in sub_views:
        assert "label" in sv and sv["label"]
        assert ("include_keys" in sv) or ("exclude_keys" in sv)
    by_id = {s["id"]: s for s in sub_views}
    assert "records" in by_id["effectsizes"]["include_keys"]
    assert "records" in by_id["descriptives"]["exclude_keys"]
    assert by_id["effectsizes"]["evidence_keys"] == ["records"]
    assert "evidence_keys" not in by_id["descriptives"]
    # Per-sub-view confidence scoping — Effect sizes tab owns the
    # effect_sizes + reliabilities ratings; Descriptives owns metadata.
    assert by_id["effectsizes"]["confidence_keys"] == ["effect_sizes", "reliabilities"]
    assert by_id["descriptives"]["confidence_keys"] == ["metadata"]
    assert by_id["effectsizes"]["label"] == "Effect sizes"


# ── Variants (NCS-18 example, hidden from landing) ────────────────────────

def test_landing_lists_only_umbrella_masem(client):
    """The landing-list endpoint should surface only the parent ``masem``
    preset; variant starters (e.g. ``masem-ncs18``) are reachable via
    the in-app builder but should not clutter the landing-screen
    workflow picker."""
    r = client.get("/api/presets")
    ids = [p["id"] for p in r.json()["presets"]]
    assert "masem" in ids
    assert "masem-ncs18" not in ids


def test_variant_preset_still_fetchable_by_id(client):
    """Hidden landing presets must still be loadable by id (the in-app
    builder posts to /api/build-preset-prompt with the variant id)."""
    r = client.get("/api/presets/masem-ncs18")
    assert r.status_code == 200
    body = r.json()
    assert body.get("landing_hidden") is True


def test_umbrella_masem_is_blank_records_starter(client):
    """The umbrella ``masem`` preset is the Direct-information variant:
    it extracts pairwise effect sizes (plus inline reliability and
    instrument metadata) into a flat ``records[]`` array per sample,
    alongside coded study metadata.  No factor-analysis fields are
    pre-baked — the matching Indirect-information starter
    (``masem-ncs18``) covers that path."""
    r = client.get("/api/presets/masem")
    body = r.json()
    p = body["template_params"]
    assert p["data_sources"] == ["records"]
    # Factor-analysis fields stay blank in the Direct variant — those
    # are the Indirect variant's territory.
    assert p.get("factor_naming") in ([], None)
    assert p.get("cfa_item_assignment") in ({}, None)
    assert p.get("item_texts") in ([], None)
    # Rendered prompt is the effect-sizes template (no factor-analytic
    # steps).  Headline structure: per-sample ``records[]`` of effect
    # sizes + inline reliability/instrument fields + sample metadata.
    prompt = body["prompt"]
    assert "records" in prompt
    assert '"var1"' in prompt
    assert '"rel1_type"' in prompt
    # No factor-analytic content in this template.
    assert "EXTRACT FACTOR LOADINGS" not in prompt
    # Sub-views: Effect sizes + Descriptives, each scoped to its own
    # confidence categories so the badge row above the panel matches
    # the active tab's data.
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["effectsizes", "descriptives"]


def test_ncs18_variant_renders_with_pre_baked_scaffold(client):
    """The NCS-18 sub-preset ships the NCS-18 scaffold inside the
    VARIABLE SCALE CONFIGURATION header: scale name, item count, the
    18 verbatim item texts (NCS-18 was chosen as the default example
    because its items can be distributed without the scale-copyright
    friction of TAS-20), and the auto-generated factor_key_mapping."""
    r = client.get("/api/presets/masem-ncs18")
    assert r.status_code == 200
    body = r.json()
    prompt = body["prompt"]
    # VARIABLE SCALE CONFIGURATION block + values
    assert "VARIABLE SCALE CONFIGURATION" in prompt
    assert "[scale_name]: Need for Cognition Scale (NCS-18)" in prompt
    assert "[n_items]: 18" in prompt
    # Factor-key mapping auto-generated for the default 2 factors.
    assert "[factor_key_mapping]" in prompt
    assert "F-I, FI, Factor I, Factor 1, Component 1 -> F1" in prompt
    assert "F-II, FII, Factor II, Factor 2, Component 2 -> F2" in prompt
    # Item-text list ships verbatim — the whole point of switching the
    # default example from TAS-20 to NCS-18.
    assert "1: I would prefer complex to simple problems." in prompt
    assert "18: I usually end up deliberating about issues even when they do not affect me personally." in prompt
    # New template uses uppercased ``# STEP 9: ...`` heading.
    assert "STEP 9: SELF-ASSESS EXTRACTION CONFIDENCE" in prompt
    assert '"extraction_confidence"' in prompt
    # Sub-views: factor loadings + factor correlations + descriptives.
    sub_ids = [sv["id"] for sv in body["sub_views"]]
    assert sub_ids == ["loadings", "correlations", "descriptives"]


def test_ncs18_variant_includes_user_supplied_item_texts(client):
    """Users paste their own item texts into the Item labels textarea.
    When that posts back through /api/build-preset-prompt with the
    NCS-18 (or any masem-* Indirect) starter, the rendered prompt must
    surface those item texts in the VARIABLE SCALE CONFIGURATION block."""
    r = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem-ncs18",
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


def test_direct_default_renders_canonical_effect_sizes(client):
    """The Direct (records-based) preset's masem.json now ships
    ``effect_sizes`` defaults (r / Correlation, or / Odds ratios) which
    must reach the rendered prompt via the ${effect_sizes_block}
    placeholder.  Empty user input keeps these defaults."""
    direct = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem", "template_params": {},
    }).json()
    p = direct["prompt"]
    assert '"r" = Correlation' in p
    assert '"or" = Odds ratios' in p


def test_direct_user_override_replaces_effect_sizes(client):
    """User-supplied ``effect_sizes`` overrides the default list — the
    prompt only carries the user's entries, not the canonical defaults
    leaking through."""
    direct = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem",
        "template_params": {
            "effect_sizes": [
                {"code": "smd", "label": "Standardised mean difference"},
                {"code": "g",   "label": "Hedges' g"},
            ],
        },
    }).json()
    p = direct["prompt"]
    assert '"smd" = Standardised mean difference' in p
    assert "Hedges' g" in p
    # Defaults should have been displaced.
    assert '"r" = Correlation' not in p
    assert '"or" = Odds ratios' not in p


def test_direct_effect_sizes_accepts_string_short_codes(client):
    """Compact form — a list of bare strings — also renders, falling
    back to a default label for known short codes."""
    direct = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem",
        "template_params": {"effect_sizes": ["r", "d"]},
    }).json()
    p = direct["prompt"]
    assert '"r" = Correlation'         in p
    assert '"d" = Cohen\'s d'          in p


def test_extraction_confidence_block_in_default_prompt(client):
    """Both MASEMiner presets now ask the model to self-assess its
    extraction confidence and emit an ``extraction_confidence`` object
    per sample.  The Indirect (factor-analytic) preset rates loadings /
    correlations / metadata.  The Direct (records-based) preset rates
    effect sizes / reliabilities / metadata — different conceptual
    extraction targets but the same high/medium/low scale.  Adding the
    block to Direct was an explicit decision to give every donation a
    self-rated reliability signal."""
    # ── Direct / records-based preset ──
    direct = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem", "template_params": {},
    }).json()
    p_direct = direct["prompt"]
    assert "SELF-ASSESS EXTRACTION CONFIDENCE" in p_direct
    assert "``effect_sizes``"  in p_direct
    assert "``reliabilities``" in p_direct
    assert "``metadata``"      in p_direct
    assert '"extraction_confidence"' in p_direct

    # ── Indirect / factor-analytic preset ──
    indirect = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem-ncs18", "template_params": {},
    }).json()
    p_indirect = indirect["prompt"]
    # New template uses uppercased ``# STEP 9: ...`` heading.
    assert "STEP 9: SELF-ASSESS EXTRACTION CONFIDENCE" in p_indirect
    assert "``factor_loadings``" in p_indirect
    assert "``factor_correlations``" in p_indirect
    assert "``metadata``" in p_indirect
    assert '"extraction_confidence"' in p_indirect


def test_masem_presets_pass_prompt_readiness_check(client):
    """The structural readiness check at /api/extract must NOT block any
    preset-driven prompt — both MASEMiner templates inline the canonical
    evidence + extraction_confidence blocks.  A regression here would
    surface as an unexpected "Proceed anyway" modal on the production
    workflows the user runs most often."""
    from prompt_check import prompt_has_extraction_signals
    for preset_id in ("masem", "masem-ncs18"):
        built = client.post("/api/build-preset-prompt", json={
            "preset_id": preset_id, "template_params": {},
        }).json()
        readiness = prompt_has_extraction_signals(built["prompt"])
        assert readiness["ok"] is True, (
            f"{preset_id} prompt failed structural readiness check: {readiness}"
        )


# ── econ-headline preset ────────────────────────────────────────────────

def test_econ_headline_preset_loads(client):
    """Smoke: GET /api/presets/econ-headline returns the preset descriptor
    with the canonical fields (id, title, prompt body, sub_views).  Without
    this the workflow-card on step 1 would silently omit the econ option."""
    r = client.get("/api/presets/econ-headline")
    assert r.status_code == 200
    data = r.json()
    assert data["id"]    == "econ-headline"
    assert data["mode"]  == "extraction"
    assert isinstance(data.get("title"), str) and data["title"]
    assert isinstance(data.get("prompt"), str) and "samples" in data["prompt"]
    assert isinstance(data.get("sub_views"), list) and len(data["sub_views"]) == 1


def test_econ_headline_sub_view_is_unified_metadata(client):
    """The preset uses a single ``regmeta`` sub-view that surfaces every
    field of every entry — the loader groups flat per-regression data
    into per-table entries (one HTML table per regression group), so
    separate Specification/Estimates/Classification tabs are redundant.
    The single tab must still carry every confidence-category key so
    the per-block confidence badges keep working."""
    data = client.get("/api/presets/econ-headline").json()
    ids = [sv["id"] for sv in data["sub_views"]]
    assert ids == ["regmeta"]
    sv = data["sub_views"][0]
    expected_categories = {
        "regressions_metadata", "regressions_specification",
        "regressions_estimates", "regressions_classification",
        "paper_metadata",
    }
    assert expected_categories.issubset(set(sv["confidence_keys"]))


def test_econ_headline_prompt_passes_readiness_check(client):
    """The rendered prompt must declare both an evidence array and an
    extraction_confidence object structurally (not just in prose) so the
    /api/extract readiness gate lets it through without forcing the
    Proceed-anyway modal."""
    from prompt_check import prompt_has_extraction_signals
    built = client.post("/api/build-preset-prompt", json={
        "preset_id": "econ-headline", "template_params": {},
    }).json()
    readiness = prompt_has_extraction_signals(built["prompt"])
    assert readiness["ok"] is True, (
        f"econ-headline prompt failed structural readiness check: {readiness}"
    )


def test_ai_findings_preset_loads(client):
    """The AI-findings preset is the first one shipped via the
    structured prompt-designer flow (see web/static/prompt-designer.js).
    Smoke: it auto-discovers and ships a single unified ``details``
    sub-view that surfaces every per-finding + paper-metadata field
    (the loader synthesises a "Paper metadata" entry at samples[0],
    so per-tab splitting would mostly produce empty tabs)."""
    r = client.get("/api/presets/ai-findings")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "ai-findings"
    assert data["mode"] == "extraction"
    assert isinstance(data.get("prompt"), str) and "samples" in data["prompt"]
    ids = [sv["id"] for sv in data["sub_views"]]
    assert ids == ["details"]


def test_ai_findings_prompt_passes_readiness_check(client):
    """Like every preset prompt, ai-findings must pass the structural
    readiness gate at /api/extract — both evidence and
    extraction_confidence blocks present with worked JSON examples."""
    from prompt_check import prompt_has_extraction_signals
    built = client.post("/api/build-preset-prompt", json={
        "preset_id": "ai-findings", "template_params": {},
    }).json()
    readiness = prompt_has_extraction_signals(built["prompt"])
    assert readiness["ok"] is True, (
        f"ai-findings prompt failed structural readiness check: {readiness}"
    )


def test_ai_findings_confidence_categories_match_sub_view_keys(client):
    """The three extraction_confidence categories the prompt rates
    (paper_metadata, findings, subtopics) must each be covered by at
    least one sub-view's confidence_keys so the badges render in the
    right tab.  This matches the FLAT-entry preset shape: the model
    emits one ``findings`` confidence rating covering every per-finding
    field, and the three top sub-views (Effect size / Comparison /
    Classification) all show that single rating because their
    confidence_keys include ``findings``.

    Drift here would surface as hidden / mis-tabbed confidence ratings
    on the live review panel."""
    data = client.get("/api/presets/ai-findings").json()
    rated = {"paper_metadata", "findings", "subtopics"}
    covered = set()
    for sv in data["sub_views"]:
        covered.update(sv.get("confidence_keys") or [])
    missing = rated - covered
    assert not missing, (
        f"sub_views don't cover these rated categories: {sorted(missing)}"
    )


def test_econ_headline_confidence_keys_align_with_prompt(client):
    """Every confidence category the prompt rates (paper_metadata +
    regressions_{metadata,specification,estimates,classification}) must
    have a matching sub-view ``confidence_keys`` entry — otherwise some
    extraction_confidence ratings emitted by the model would render with
    no associated tab (silently hidden badges)."""
    data = client.get("/api/presets/econ-headline").json()
    prompt = data["prompt"]
    rated_categories = {
        "paper_metadata", "regressions_metadata", "regressions_specification",
        "regressions_estimates", "regressions_classification",
    }
    # Each must appear as a JSON-quoted key inside the rendered prompt body
    # (matches the worked extraction_confidence example).
    for cat in rated_categories:
        assert f'"{cat}"' in prompt, (
            f"prompt does not declare confidence category {cat!r}"
        )
    # And each must be covered by at least one sub-view's confidence_keys.
    declared_keys = set()
    for sv in data["sub_views"]:
        declared_keys.update(sv.get("confidence_keys") or [])
    missing = rated_categories - declared_keys
    assert not missing, (
        f"sub_views don't cover these rated categories: {sorted(missing)}"
    )


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


def test_build_preset_prompt_echoes_default_data_sources_preserves_explicit_sub_views(client):
    """Regression: the masem-builder posts the FULL template_params on
    every form change, including the unchanged ``data_sources`` value
    inherited from the preset's own defaults.  The previous
    ``overrode_sources`` check fired purely on key presence, mis-
    classified that echo as a real override, fell into auto-generation,
    and — because ``records`` isn't in _SUB_VIEW_SPECS — returned an
    empty list.  The empty list then propagated into
    ``state.activePreset.sub_views`` via the builder's auto-commit and
    wiped the Effect-sizes + Descriptives sub-tabs from the result panel.

    Override semantics: only treat data_sources as overridden when it
    DIFFERS from the preset default.  Echoing the default keeps the
    preset's explicit sub_views intact."""
    direct = client.get("/api/presets/masem").json()
    preset_default_sources = direct["template_params"]["data_sources"]
    assert preset_default_sources == ["records"]

    # Builder-style payload: full template_params including the
    # unchanged data_sources value.
    built = client.post("/api/build-preset-prompt", json={
        "preset_id":       "masem",
        "template_params": {**direct["template_params"]},
    }).json()
    sub_ids = [sv["id"] for sv in built["sub_views"]]
    assert sub_ids == ["effectsizes", "descriptives"], (
        "Echoing the preset's own data_sources must NOT trigger sub_views "
        "auto-regeneration; the explicit JSON sub_views should pass through."
    )


def test_build_preset_prompt_overrides_data_sources(client):
    """User-supplied ``data_sources`` regenerates ``sub_views`` to match
    those sources (auto-generation path), overriding any explicit
    ``sub_views`` declared on the preset.  Direct vs Indirect mode is
    now selected by picking the preset (``masem`` vs ``masem-ncs18``),
    not by overriding ``data_sources``, but the override path remains
    valid for users who want to tune sources inside a preset."""
    r = client.post("/api/build-preset-prompt", json={
        "preset_id": "masem-ncs18",
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
    # The factor-analytic template structures loadings + correlations
    # as Step 5 + 6 (uppercased ``# STEP …`` headings in the v3 prompt).
    assert "STEP 5: EXTRACT FACTOR LOADINGS" in prompt
    assert "STEP 6: EXTRACT FACTOR CORRELATIONS" in prompt
    # User-overridden scale config flows through to the prompt body.
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
