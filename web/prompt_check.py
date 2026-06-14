"""Deterministic readiness check for AI-generated and hand-written prompts.

The single function ``prompt_has_extraction_signals`` answers two
questions about an arbitrary prompt string:

  - does it instruct the model to emit an ``evidence`` array with the
    canonical four keys (snippet, page, source, field)?
  - does it instruct the model to emit an ``extraction_confidence``
    object with ``level`` ratings?

Both checks are deliberately strict on one point: the prompt must
contain not just the keywords but an actual JSON-example block (``[`` or
``{``) following the canonical key.  A prompt that mentions the words
``evidence`` and ``extraction_confidence`` in prose but never SHOWS the
model what the structure looks like is treated as missing — the
downstream model can't infer a structure from naming alone.

This module has zero dependencies beyond ``re``.  It is callable from
the request handlers, from the client (via /api/check-prompt-readiness),
and from the existing legacy helper ``_prompt_has_evidence_schema``
(which is now a thin wrapper around the evidence half).
"""

from __future__ import annotations

import re
from typing import TypedDict

# A JSON-quoted key — matches both "evidence": and 'evidence':
# (single quotes appear in code snippets some users paste).
_KEY = lambda name: re.compile(rf"""["']{re.escape(name)}["']\s*:""")

_EVIDENCE_KEY            = _KEY("evidence")
_EVIDENCE_KEY_OPENS_LIST = re.compile(r"""["']evidence["']\s*:\s*\[""")
_SNIPPET_KEY             = _KEY("snippet")
_PAGE_KEY                = _KEY("page")
_SOURCE_KEY              = _KEY("source")
_FIELD_KEY               = _KEY("field")

_CONFIDENCE_KEY            = _KEY("extraction_confidence")
_CONFIDENCE_KEY_OPENS_OBJ  = re.compile(r"""["']extraction_confidence["']\s*:\s*\{""")
# Either the literal key "level" or one of the three rating literals,
# either as a JSON value ("high"/"medium"/"low") or in prose
# (high / medium / low confidence).
_LEVEL_TOKEN = re.compile(
    r"""(?:["']level["']\s*:)|(?:["'](?:high|medium|low)["'])""",
    re.IGNORECASE,
)


class ReadinessResult(TypedDict):
    ok:                       bool
    has_evidence_structure:   bool
    has_confidence_structure: bool
    missing:                  list[str]


def _count_evidence_subkeys(prompt: str) -> int:
    """How many of the four evidence sub-keys (snippet, page, source,
    field) appear as JSON-quoted keys in the prompt.  We require ≥3 of
    4 to count the evidence block as structurally specified."""
    return (
        bool(_SNIPPET_KEY.search(prompt))
        + bool(_PAGE_KEY.search(prompt))
        + bool(_SOURCE_KEY.search(prompt))
        + bool(_FIELD_KEY.search(prompt))
    )


def _has_evidence_structure(prompt: str) -> bool:
    """All of:
      1) ``"evidence":`` appears as a JSON-quoted key
      2) a list opens after it (``[``) — i.e. the prompt SHOWS the array
      3) ≥3 of the four sub-keys (snippet/page/source/field) appear
    """
    if not _EVIDENCE_KEY.search(prompt):
        return False
    if not _EVIDENCE_KEY_OPENS_LIST.search(prompt):
        return False
    return _count_evidence_subkeys(prompt) >= 3


def _has_confidence_structure(prompt: str) -> bool:
    """All of:
      1) ``"extraction_confidence":`` appears as a JSON-quoted key
      2) an object opens after it (``{``)
      3) at least one ``"level":`` key OR one of the three rating
         literals ("high"/"medium"/"low") appears nearby
    """
    if not _CONFIDENCE_KEY.search(prompt):
        return False
    if not _CONFIDENCE_KEY_OPENS_OBJ.search(prompt):
        return False
    return bool(_LEVEL_TOKEN.search(prompt))


def prompt_has_extraction_signals(prompt: str) -> ReadinessResult:
    """Inspect a prompt for the two structural signals the downstream
    renderer depends on.  Returns a dict describing exactly what's
    present and what's missing.

    A prompt is ``ok`` only when BOTH structural blocks are present.
    The renderer (PDF evidence highlighter + per-block confidence
    badges) cannot function without both — though the user is allowed
    to acknowledge and proceed regardless via the front-end modal.
    """
    has_evidence   = _has_evidence_structure(prompt or "")
    has_confidence = _has_confidence_structure(prompt or "")
    missing: list[str] = []
    if not has_evidence:
        missing.append("evidence")
    if not has_confidence:
        missing.append("extraction_confidence")
    return {
        "ok":                       has_evidence and has_confidence,
        "has_evidence_structure":   has_evidence,
        "has_confidence_structure": has_confidence,
        "missing":                  missing,
    }
