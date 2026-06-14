"""Tests for the deterministic prompt-readiness checker.

These are the structural rules that gate /api/extract.  A prompt is
``ok`` only when BOTH the evidence-array spec and the
extraction_confidence-object spec are present with worked JSON example
blocks following them — keyword mentions alone aren't enough.
"""
from __future__ import annotations

import prompt_check
from prompt_builder import EVIDENCE_APPENDIX, build_meta_prompt


_GOOD_EVIDENCE_BLOCK = """
"evidence": [
  {"snippet": "TABLE 1. Descriptive statistics...",
   "page": 3,
   "source": "Table 1",
   "field": "samples[0].demographics"}
]
"""

_GOOD_CONFIDENCE_BLOCK = """
"extraction_confidence": {
  "samples": {"level": "high"},
  "metadata": {"level": "medium", "notes": "country inferred"}
}
"""


def test_full_canonical_structure_is_ok():
    """A prompt containing both example blocks → ok=True."""
    prompt = _GOOD_EVIDENCE_BLOCK + _GOOD_CONFIDENCE_BLOCK
    r = prompt_check.prompt_has_extraction_signals(prompt)
    assert r["ok"] is True
    assert r["has_evidence_structure"]   is True
    assert r["has_confidence_structure"] is True
    assert r["missing"] == []


def test_evidence_only_is_not_ok():
    """Evidence block present, confidence block absent → ok=False, only
    ``extraction_confidence`` listed as missing."""
    r = prompt_check.prompt_has_extraction_signals(_GOOD_EVIDENCE_BLOCK)
    assert r["ok"] is False
    assert r["has_evidence_structure"]   is True
    assert r["has_confidence_structure"] is False
    assert r["missing"] == ["extraction_confidence"]


def test_confidence_only_is_not_ok():
    """Confidence block present, evidence block absent → ok=False, only
    ``evidence`` listed as missing."""
    r = prompt_check.prompt_has_extraction_signals(_GOOD_CONFIDENCE_BLOCK)
    assert r["ok"] is False
    assert r["has_evidence_structure"]   is False
    assert r["has_confidence_structure"] is True
    assert r["missing"] == ["evidence"]


def test_bare_prompt_is_not_ok():
    """A plain "extract sample sizes" prompt has neither structure."""
    r = prompt_check.prompt_has_extraction_signals(
        "Extract the sample size and mean age from the paper."
    )
    assert r["ok"] is False
    assert r["has_evidence_structure"]   is False
    assert r["has_confidence_structure"] is False
    assert set(r["missing"]) == {"evidence", "extraction_confidence"}


def test_keyword_mentions_without_json_block_do_not_pass():
    """A prompt that names the keys in prose but never shows a JSON
    example must NOT pass.  This is the key distinction between the
    new structural check and the old keyword-count heuristic."""
    prompt = (
        "Please return evidence for each value with a snippet and a page "
        "number from the source.  Also include the extraction_confidence "
        "as a rating per field."
    )
    r = prompt_check.prompt_has_extraction_signals(prompt)
    # Keywords are there in prose, but no `"evidence": [` opening and no
    # `"extraction_confidence": {` opening → both verdicts must be False.
    assert r["ok"] is False
    assert r["has_evidence_structure"]   is False
    assert r["has_confidence_structure"] is False


def test_evidence_with_too_few_subkeys_fails():
    """Evidence list opens, but fewer than 3 of (snippet/page/source/field)
    appear as JSON keys → not structural."""
    prompt = '''
    "evidence": [
      {"snippet": "...", "page": 1}
    ]
    '''
    r = prompt_check.prompt_has_extraction_signals(prompt)
    assert r["has_evidence_structure"] is False


def test_confidence_with_level_token_in_value_passes():
    """The level rating can come from the literal "high"/"medium"/"low"
    appearing in a JSON value position — that's the canonical worked
    example shape."""
    prompt = '''
    "extraction_confidence": {
      "samples": {"level": "high"}
    }
    '''
    r = prompt_check.prompt_has_extraction_signals(prompt)
    assert r["has_confidence_structure"] is True


def test_empty_string_is_handled():
    """Defensive: empty input should not raise."""
    r = prompt_check.prompt_has_extraction_signals("")
    assert r["ok"] is False
    assert r["missing"] == ["evidence", "extraction_confidence"]


def test_none_is_handled():
    """Defensive: None should be coerced to empty, not crash."""
    r = prompt_check.prompt_has_extraction_signals(None)  # type: ignore[arg-type]
    assert r["ok"] is False


# ── Integration with the rest of the prompt-builder surface ─────────

def test_evidence_appendix_alone_passes_readiness_check():
    """``EVIDENCE_APPENDIX`` is what /api/generate-prompt bolts onto every
    AI-generated prompt.  By itself, it must contain both worked example
    blocks (evidence + extraction_confidence) — otherwise the readiness
    gate would block every generated prompt that the AI happened to
    write without inline examples."""
    r = prompt_check.prompt_has_extraction_signals(EVIDENCE_APPENDIX)
    assert r["ok"] is True, f"EVIDENCE_APPENDIX failed readiness: missing={r['missing']}"


def test_meta_prompt_demands_both_structures():
    """The strengthened ``build_meta_prompt`` must instruct the prompt-writer
    LLM to include BOTH structures in the prompt it writes."""
    prompt = build_meta_prompt("extraction", "extract sample sizes", "")
    lower = prompt.lower()
    assert '"evidence"'              in prompt
    assert '"extraction_confidence"' in prompt
    # Mentions a worked JSON example requirement for each
    assert "worked json example" in lower or "worked example" in lower


# ── Legacy helper back-compat ────────────────────────────────────────

def test_legacy_helper_still_works():
    """``_prompt_has_evidence_schema`` is the old yes/no helper.  After
    the refactor it delegates to the new checker but its public
    signature and semantics must stay identical so older callers
    aren't broken."""
    from _helpers import _prompt_has_evidence_schema
    # Positive: a prompt with the canonical evidence block
    assert _prompt_has_evidence_schema(_GOOD_EVIDENCE_BLOCK) is True
    # Negative: a bare prompt
    assert _prompt_has_evidence_schema("extract sample sizes.") is False
