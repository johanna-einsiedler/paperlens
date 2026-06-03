"""Tiny ``.env`` loader for local development.

Reads ``web/.env`` (gitignored) when present and populates ``os.environ``
with any keys not already set by the deploy environment.  On Fly,
secrets come in via ``fly secrets set`` so the file doesn't exist and
``load()`` is a no-op.

Extracted from ``server.py`` so standalone scripts and diagnostics
(``python -c "..."``) can pick up ``.env`` without importing the
FastAPI stack.

Format
------
- One ``KEY=value`` per line.
- ``#``-prefixed lines are comments.
- Surrounding single/double quotes on the value are stripped.
- Values already present in ``os.environ`` win — the file never
  clobbers ``export``-ed shell variables.

Multi-line values (e.g. PEM private keys) are NOT supported.  Keep
those in your shell ``export``s.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / ".env"


def load(path: Path | None = None) -> None:
    """Populate ``os.environ`` from ``.env`` (defaults to ``web/.env``).
    Silently no-op when the file is absent."""
    env_file = path or _DEFAULT_PATH
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        v = value.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        os.environ.setdefault(key, v)
