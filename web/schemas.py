"""Pydantic request models for the FastAPI routes.

Kept separate from ``server.py`` so the route file stays focused on
HTTP wiring and the schemas are easy to find/extend.  Nothing here
depends on the app instance — these are pure dataclasses.
"""
from __future__ import annotations

from pydantic import BaseModel


class GeneratePromptIn(BaseModel):
    """Payload for the AI meta-prompt generator (``/api/generate-prompt``).
    The model receives ``question`` + ``context`` and emits a structured
    extraction prompt for the user to review/edit."""
    api_key:  str = ""
    model:    str = "gpt-4o-mini"
    mode:     str = "extraction"
    question: str = ""
    context:  str = ""
    base_url: str | None = None


class CheckSchemaIn(BaseModel):
    """Payload for ``/api/check-evidence-schema``.  The route returns
    whether the prompt already asks for evidence (so the UI knows
    whether to nudge the user to add an evidence schema)."""
    prompt: str = ""


class TestConnectionIn(BaseModel):
    """Payload for ``/api/test-connection`` — credential sanity check
    against a provider before kicking off a long extraction."""
    api_key:  str = ""
    model:    str = "gpt-4o-mini"
    base_url: str | None = None


class BatchEmailIn(BaseModel):
    """Payload for ``/api/batches/{batch_id}/email`` — attach (or
    update) an address that gets the batch-complete notification."""
    email: str = ""


class AdaptPromptIn(BaseModel):
    """Payload for ``/api/adapt-prompt`` — minimally rewrite an
    existing prompt to also request evidence snippets."""
    api_key:  str = ""
    model:    str = "gpt-4o-mini"
    prompt:   str = ""
    base_url: str | None = None


class BuildPresetPromptIn(BaseModel):
    """Payload for the guided MASEMiner prompt builder.  ``preset_id``
    picks the underlying template (typically one of the ``masem-*``
    variants); ``template_params`` is the form's serialised state that
    overrides the preset's default params."""
    preset_id:       str
    template_params: dict = {}
