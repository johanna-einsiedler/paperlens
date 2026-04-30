"""Domain-workflow presets — discovery, parsing, validation.

A preset is a JSON file under ``web/presets/<id>.json`` describing a
tailored workflow (branding + recommended provider/model + a pre-built
extraction prompt + step-skip hint).  See ``presets/masem.json`` for the
canonical example.

The prompt body is large enough to be awkward in JSON, so each preset may
reference a sibling ``prompt_file`` (relative to the JSON, typically a
``.prompt.md``).  The loader inlines that into the returned dict under
``prompt`` — frontend never needs to know the file split.

Loaded at request time (not boot time), so dropping in a new preset takes
effect on the next API call without a restart.  Bad files are logged and
skipped rather than crashing the server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PRESETS_DIR = Path(__file__).parent / "presets"

# Required fields a preset JSON must declare to be considered valid.
_REQUIRED_KEYS = ("id", "title", "tagline", "mode")
# Either "prompt" inline or "prompt_file" pointing to a sibling text file.
_PROMPT_KEYS = ("prompt", "prompt_file")


def _read_prompt_body(meta: dict, source_path: Path) -> str:
    """If the preset references a sibling prompt file, read it; otherwise
    use the inline ``prompt`` field. Returns the prompt text or empty
    string if neither is present."""
    if meta.get("prompt"):
        return str(meta["prompt"])
    pf = meta.get("prompt_file")
    if pf:
        candidate = (source_path.parent / pf).resolve()
        # Defensive: don't allow ../ escapes out of the presets dir
        try:
            candidate.relative_to(PRESETS_DIR.resolve())
        except ValueError:
            print(f"[presets] prompt_file outside presets dir, ignoring: {pf}",
                  file=sys.stderr, flush=True)
            return ""
        try:
            return candidate.read_text(encoding="utf-8")
        except OSError as e:
            print(f"[presets] could not read prompt_file {pf}: {e}",
                  file=sys.stderr, flush=True)
            return ""
    return ""


def _load_one(path: Path) -> dict[str, Any] | None:
    """Parse a single preset JSON. Returns None if invalid."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[presets] skipping {path.name}: {e}",
              file=sys.stderr, flush=True)
        return None
    if not isinstance(meta, dict):
        print(f"[presets] {path.name}: top-level value must be an object",
              file=sys.stderr, flush=True)
        return None
    missing = [k for k in _REQUIRED_KEYS if not meta.get(k)]
    if missing:
        print(f"[presets] {path.name}: missing required keys {missing}",
              file=sys.stderr, flush=True)
        return None
    if not any(meta.get(k) for k in _PROMPT_KEYS):
        print(f"[presets] {path.name}: must define 'prompt' or 'prompt_file'",
              file=sys.stderr, flush=True)
        return None

    # Inline the prompt body so callers never have to think about file split
    body = _read_prompt_body(meta, path)
    if not body:
        print(f"[presets] {path.name}: prompt body is empty",
              file=sys.stderr, flush=True)
        return None
    meta["prompt"] = body
    meta.pop("prompt_file", None)
    return meta


def load_all() -> dict[str, dict[str, Any]]:
    """Discover and return every valid preset, keyed by id.  Cheap enough
    to call per request — typical deployments have a handful of files."""
    out: dict[str, dict[str, Any]] = {}
    if not PRESETS_DIR.is_dir():
        return out
    for path in sorted(PRESETS_DIR.glob("*.json")):
        meta = _load_one(path)
        if meta is None:
            continue
        pid = meta["id"]
        if pid in out:
            print(f"[presets] duplicate id {pid!r} in {path.name}; ignoring",
                  file=sys.stderr, flush=True)
            continue
        out[pid] = meta
    return out


def get(preset_id: str) -> dict[str, Any] | None:
    """Return one preset by id, or None if not found / invalid."""
    return load_all().get(preset_id)


def list_summaries() -> list[dict[str, Any]]:
    """Lightweight list for the workflows menu — no prompt body included."""
    out = []
    for preset in load_all().values():
        out.append({
            "id":           preset["id"],
            "title":        preset["title"],
            "tagline":      preset.get("tagline", ""),
            "description":  preset.get("description", ""),
            "accent_color": preset.get("accent_color"),
        })
    return out
