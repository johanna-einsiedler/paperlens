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
    # Concrete bad/good phrasings should appear so the model has memorable anchors
    assert "Cronbach" in text or "fit indices" in text  # bad-evidence anchors
    assert "TABLE" in text                              # good-evidence anchor


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


def test_evidence_appendix_field_is_json_path():
    """The 'field' property must be specified as a JSON path that mirrors the
    output structure (so we can map evidence -> cell)."""
    text = prompt_builder.EVIDENCE_APPENDIX
    assert "JSON path" in text or "json path" in text.lower()
    # Concrete path examples
    assert "samples[0]" in text
    assert "factor_loadings._table[0]" in text


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


def test_prompt_has_evidence_schema_threshold_at_three():
    """Heuristic requires ≥3 of {evidence, snippet, page, source}."""
    from server import _prompt_has_evidence_schema
    # 2 hits → false
    assert not _prompt_has_evidence_schema("Quote the snippet from the page.")
    # 3 hits → true
    assert _prompt_has_evidence_schema("Quote the snippet from the page and cite source.")
