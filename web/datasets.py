"""Read-only listing of datasets in the curated public repo.

Powers ``GET /api/datasets`` (Phase 3a — discovery) and is the source of
truth for the Phase 3b "Extend an existing dataset" landing card.

Design
------

- We read the dataset folders from ``metalens-datasets`` via the GitHub
  Contents API.  Each ``datasets/<slug>/metadata.json`` carries the
  donor-supplied title / description / schema_version / password hash;
  this module fetches each one, strips the password hash, and returns a
  cleaned-up listing.
- The bot's GitHub App token (set up for the donation PRs in donor.py)
  is used when available — gives us a 5000/h rate limit against the
  GitHub API instead of the 60/h anonymous cap.  Falls back to
  unauthenticated requests when ``donor.is_enabled()`` is False (e.g.
  on a local maseminer install without bot credentials).
- The full listing is cached in-process for ``_CACHE_TTL_SEC`` (default
  1 hour) so each /api/datasets request doesn't hit GitHub fresh.
  Single-process cache; if we ever scale out we'll swap to a shared
  store, but currently fly.io runs one machine.

Security
--------

``password_hash`` is **stripped from every response**.  Verification of
the password happens entirely server-side in a separate endpoint
(Phase 3c) — the hash never leaves the server, even though it's
technically already public in the GitHub repo.  This avoids tempting
clients to compare hashes locally (which would force them to hold the
bcrypt cost factor and make the per-attempt timing visible).
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import httpx

import donor


_CACHE_TTL_SEC = 60 * 60   # 1 hour
_CACHE: dict[str, Any] = {"datasets": [], "fetched_at": 0.0}


def _gh_repo() -> tuple[str, str]:
    """Returns (owner, name) from PAPERLENS_GH_REPO; raises if unset."""
    repo = os.environ.get("PAPERLENS_GH_REPO", "")
    if "/" not in repo:
        raise RuntimeError("PAPERLENS_GH_REPO must be 'owner/repo'")
    owner, name = repo.split("/", 1)
    return owner, name


def _auth_headers(client: httpx.Client) -> dict[str, str]:
    """Authenticated headers when the bot is configured, otherwise
    plain Accept headers.  Falling back to anonymous keeps local
    maseminer installs working — the dataset listing is from a public
    repo so unauthed reads are allowed (just rate-limited harder)."""
    base = {
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if donor.is_enabled():
        try:
            token = donor._get_installation_token(client)
            base["Authorization"] = f"Bearer {token}"
        except Exception:  # noqa: BLE001
            # Auth misconfigured — fall through to anonymous reads.
            # The endpoint stays useful for browsing even when the
            # write-side donation flow can't open PRs.
            pass
    return base


def _strip_password_hash(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove the bcrypt password hash from a metadata.json payload
    before it leaves the server.  Gated-ness is signalled via a derived
    ``gated`` boolean so the frontend can render a 🔒 indicator."""
    sanitized = {k: v for k, v in metadata.items() if k != "password_hash"}
    sanitized["gated"] = bool(metadata.get("password_hash"))
    return sanitized


def _parse_one(content_b64: str) -> dict[str, Any] | None:
    """Decode + JSON-parse a metadata.json blob; return None on any
    error (rather than 500ing the whole listing because one folder has
    a malformed metadata file)."""
    try:
        raw = base64.b64decode(content_b64).decode("utf-8")
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return None


def _fetch_listing() -> list[dict[str, Any]]:
    """Walk ``datasets/*/metadata.json`` in the curated repo, return a
    list of sanitized records.  Caller is responsible for caching."""
    owner, name = _gh_repo()
    listings: list[dict[str, Any]] = []
    with httpx.Client(timeout=20.0) as client:
        headers = _auth_headers(client)
        # 1. List ``datasets/`` contents — each entry is a dataset folder.
        r = client.get(
            f"https://api.github.com/repos/{owner}/{name}/contents/datasets",
            headers=headers,
        )
        if r.status_code == 404:
            # Repo exists but no datasets/ folder yet — empty listing,
            # not an error.
            return []
        r.raise_for_status()
        folders = [item for item in r.json() if item.get("type") == "dir"]

        # 2. For each folder, fetch metadata.json.  Sequential to keep
        # the code simple — folder count is bounded (<200) for years.
        for folder in folders:
            folder_name = folder["name"]
            r = client.get(
                f"https://api.github.com/repos/{owner}/{name}/contents/datasets/{folder_name}/metadata.json",
                headers=headers,
            )
            if r.status_code != 200:
                continue
            blob = r.json()
            metadata = _parse_one(blob.get("content", ""))
            if metadata is None:
                continue
            sanitized = _strip_password_hash(metadata)
            # Surface useful derived fields the metadata file itself
            # doesn't carry — the GitHub folder URL and an explicit
            # paper-count from the extraction block (donor.py already
            # populates extraction.paper_count).
            sanitized["github_url"] = (
                f"https://github.com/{owner}/{name}/tree/main/datasets/{folder_name}"
            )
            # The metadata file has its own top-level ``schema_version``
            # (e.g. "metadata-v1") that describes the metadata file
            # format, not the extraction's data shape.  The frontend
            # needs the EXTRACTION schema (e.g. "masem-v3") to decide
            # which preset to pre-load when extending — surface that
            # one explicitly and demote the metadata format version
            # to ``metadata_schema_version`` for clarity.
            extraction = metadata.get("extraction") or {}
            sanitized["metadata_schema_version"] = sanitized.get("schema_version")
            sanitized["schema_version"]          = extraction.get("schema_version")
            sanitized["paper_count"]             = extraction.get("paper_count")
            sanitized["model_used"]              = extraction.get("model_used") or []
            listings.append(sanitized)

    # Newest first — most users want to see what's been added recently.
    listings.sort(key=lambda d: d.get("created_at", ""), reverse=True)
    return listings


def list_datasets(*, force_refresh: bool = False) -> dict[str, Any]:
    """Cached entry point.  Returns the payload that ``/api/datasets``
    serves verbatim (a wrapper dict so we can include cache metadata)."""
    now = time.time()
    if (not force_refresh
            and _CACHE["fetched_at"]
            and (now - _CACHE["fetched_at"]) < _CACHE_TTL_SEC):
        return {
            "datasets":       _CACHE["datasets"],
            "fetched_at":     _iso(_CACHE["fetched_at"]),
            "cache_age_sec":  int(now - _CACHE["fetched_at"]),
        }
    datasets = _fetch_listing()
    _CACHE["datasets"]   = datasets
    _CACHE["fetched_at"] = now
    return {
        "datasets":       datasets,
        "fetched_at":     _iso(now),
        "cache_age_sec":  0,
    }


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def invalidate_cache() -> None:
    """Drop the cached listing.  Called by donor.donate() so a fresh
    donation surfaces immediately in /api/datasets instead of waiting
    for the next TTL expiry."""
    _CACHE["datasets"]   = []
    _CACHE["fetched_at"] = 0.0


# ── Server-side single-dataset lookup ────────────────────────────────────────

def get_dataset_metadata(dataset_id: str) -> dict[str, Any] | None:
    """Fetch ONE dataset's metadata.json directly from GitHub, including
    the ``password_hash`` field (which is normally stripped before
    leaving the server).  For server-side use only — callers MUST NOT
    return the result verbatim through any HTTP endpoint.

    Used by the Phase 3c verify-password handler so it can compare the
    user-supplied password against the stored bcrypt hash without ever
    surfacing the hash to a client.  Returns None when the dataset
    folder doesn't exist or the metadata file is malformed.

    NOT cached — password verification should always see the latest
    hash on disk (and the call is on a slow user-facing path anyway,
    bounded by the rate-limiter)."""
    if not _is_safe_slug(dataset_id):
        return None
    owner, name = _gh_repo()
    with httpx.Client(timeout=20.0) as client:
        headers = _auth_headers(client)
        r = client.get(
            f"https://api.github.com/repos/{owner}/{name}"
            f"/contents/datasets/{dataset_id}/metadata.json",
            headers=headers,
        )
        if r.status_code != 200:
            return None
        blob = r.json()
        return _parse_one(blob.get("content", ""))


# Strict slug pattern — letters, digits, hyphens.  Rejects anything
# that could be a path-traversal vector (``../``, slashes, dots).
import re as _re
_SAFE_SLUG = _re.compile(r"^[a-z0-9][a-z0-9-]{0,99}$")


def _is_safe_slug(slug: str) -> bool:
    """Defence-in-depth: even though FastAPI's path param won't accept
    slashes, we reject anything that doesn't match the bot's generated
    slug shape to prevent any future surprise from a wider validator."""
    return bool(slug) and bool(_SAFE_SLUG.match(slug))
