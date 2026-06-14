"""Tests for prompt_builder.py and the evidence-schema heuristic."""
from __future__ import annotations

import prompt_builder


def test_evidence_appendix_contains_required_terms():
    text = prompt_builder.EVIDENCE_APPENDIX.lower()
    for term in ("evidence", "snippet", "page", "source"):
        assert term in text, f"appendix missing required term: {term!r}"


def test_evidence_appendix_specifies_table_marker():
    """Tabular data must be instructed to use the explicit _table marker so the
    viewer doesn't have to guess shape."""
    text = prompt_builder.EVIDENCE_APPENDIX
    assert "_table" in text
    # An example showing the expected wrap pattern
    assert '"_table"' in text


def test_meta_prompt_extraction_mentions_table_marker():
    """AI-generated prompts must learn about the _table convention so they
    instruct the downstream model to use it."""
    prompt = prompt_builder.build_meta_prompt("extraction", "extract loadings", "")
    assert "_table" in prompt


def test_evidence_appendix_has_good_bad_examples():
    """The appendix must explicitly contrast good vs. bad evidence so the model
    stops emitting methodology-only snippets."""
    text = prompt_builder.EVIDENCE_APPENDIX
    # Markers we use to delineate the lists
    assert "BAD" in text
    assert "GOOD" in text
    # Concrete bad/good phrasings — domain-neutral anchors after the
    # de-MASEM refactor (was: Cronbach / TAS-20 / factor_loadings).
    assert "fit indices" in text  # generic bad-evidence anchor
    assert "TABLE" in text        # good-evidence anchor


def test_evidence_appendix_has_no_masem_isms():
    """The appendix is shown to users on every domain.  It must NOT
    leak factor-analysis or MASEMiner-specific anchors that would
    confuse a reader extracting a different kind of data."""
    text = prompt_builder.EVIDENCE_APPENDIX
    for bad in ("TAS-20", "Cronbach", "factor_loadings", "factor_correlations",
                "factor matrix", "three-factor"):
        assert bad not in text, (
            f"appendix contains the MASEM-specific anchor {bad!r} — "
            "domain leakage; replace with a neutral placeholder"
        )


def test_evidence_appendix_requires_table_caption_evidence():
    """For every _table emitted, the evidence array must include the caption."""
    text = prompt_builder.EVIDENCE_APPENDIX
    # The non-negotiable wording should be present
    assert "verbatim caption" in text.lower() or "verbatim table caption" in text.lower() \
        or "verbatim text" in text.lower()
    # Tied explicitly to _table
    assert "_table" in text
    # And the requirement is mandatory
    assert "MUST" in text


def test_evidence_appendix_demands_extraction_confidence():
    """Every auto-generated prompt (built via ``/api/generate-prompt`` →
    ``generated + EVIDENCE_APPENDIX``) must instruct the downstream model
    to self-rate extraction confidence — symmetric with the MASEM presets
    which carry the same self-assessment block.  Without this, users who
    skip the preset builder and ask the AI to draft a prompt get evidence
    but no confidence signal."""
    text = prompt_builder.EVIDENCE_APPENDIX
    assert '"extraction_confidence"' in text
    # The three rating levels must all be named explicitly.
    for level in ('"high"', '"medium"', '"low"'):
        assert level in text
    # And notes must be required on the non-high levels.
    lower = text.lower()
    assert "notes" in lower
    assert "must" in lower


def test_extraction_confidence_block_in_generated_prompt():
    """Smoke: a real ``build_meta_prompt`` + appendix concat (what the
    API actually returns) must contain the confidence block."""
    bare = prompt_builder.build_meta_prompt("extraction", "extract loadings", "")
    full = bare + prompt_builder.EVIDENCE_APPENDIX
    assert "extraction_confidence" in full
    assert "SELF-ASSESS" in full


def test_evidence_appendix_field_is_json_path():
    """The 'field' property must be specified as a JSON path that mirrors the
    output structure (so we can map evidence -> cell)."""
    text = prompt_builder.EVIDENCE_APPENDIX
    assert "JSON path" in text or "json path" in text.lower()
    # Concrete path examples — keep ``samples[0]`` (still the canonical
    # extraction top-level key) but the field placeholder is now
    # domain-neutral (``<your_field>`` rather than ``factor_loadings``).
    assert "samples[0]" in text
    assert "_table[0]" in text


def test_meta_prompt_propagates_caption_and_path_conventions():
    """The meta-prompt for extraction must teach AI-generated prompts to demand
    table-caption evidence and JSON-path 'field' values."""
    prompt = prompt_builder.build_meta_prompt("extraction", "x", "")
    assert "table caption" in prompt.lower()
    assert "JSON path" in prompt or "json path" in prompt.lower()
    assert "samples[0]" in prompt


def test_build_meta_prompt_extraction_has_schema_guidance():
    prompt = prompt_builder.build_meta_prompt("extraction", "extract sample size", "")
    lower = prompt.lower()
    assert "json schema" in lower
    # We added a tabular-data hint — confirm it's present
    assert "tabular" in lower or "table" in lower


def test_build_meta_prompt_labeling_mentions_required_fields():
    prompt = prompt_builder.build_meta_prompt("labeling", "classify papers", "")
    lower = prompt.lower()
    assert "label" in lower
    assert "rationale" in lower
    assert "json" in lower


def test_build_meta_prompt_summarize_mentions_sections_and_evidence():
    prompt = prompt_builder.build_meta_prompt("summarize", "summarise the paper", "")
    lower = prompt.lower()
    assert "summaries" in lower
    assert "evidence" in lower
    assert "json" in lower
    # Sections and the field-path convention so the renderer can link clicks
    # back to evidence pages.
    assert "section" in lower
    assert "summaries[0]" in lower


def test_build_meta_prompt_includes_user_question():
    prompt = prompt_builder.build_meta_prompt(
        "extraction",
        "MY UNIQUE QUESTION TOKEN",
        "MY UNIQUE CONTEXT TOKEN",
    )
    assert "MY UNIQUE QUESTION TOKEN" in prompt
    assert "MY UNIQUE CONTEXT TOKEN" in prompt


def test_load_example_prompts_handles_missing_dir():
    """If prompts/ doesn't exist, the loader should return an empty string,
    not raise."""
    result = prompt_builder.load_example_prompts("extraction")
    assert isinstance(result, str)


def test_prompt_has_evidence_schema_positive():
    """A prompt that contains the EVIDENCE_APPENDIX should pass the heuristic."""
    from server import _prompt_has_evidence_schema
    full = "Extract the sample size.\n" + prompt_builder.EVIDENCE_APPENDIX
    assert _prompt_has_evidence_schema(full)


def test_prompt_has_evidence_schema_negative_bare_prompt():
    """A bare prompt without evidence/snippet/page/source should fail."""
    from server import _prompt_has_evidence_schema
    assert not _prompt_has_evidence_schema("Extract sample sizes from the paper.")


def test_prompt_has_evidence_schema_rejects_keyword_only():
    """After the structural-checker refactor, keyword-only prompts
    (mentioning evidence/snippet/page/source in prose without any JSON
    example block) must NOT pass.  This is the failure mode that
    motivated tightening the helper — the old ≥3-of-4 keyword count was
    gameable by any prompt that happened to use the words in passing."""
    from server import _prompt_has_evidence_schema
    # All four keywords present in prose, but no JSON block → must fail
    assert not _prompt_has_evidence_schema(
        "Quote the snippet from the page and cite source for evidence."
    )
    # Two keywords → also fails
    assert not _prompt_has_evidence_schema("Quote the snippet from the page.")
