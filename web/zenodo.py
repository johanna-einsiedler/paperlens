"""Zenodo deposit client — creates a DRAFT deposit per donation, uploads
the bundle files, and writes the metadata.  Does NOT publish (which would
mint the DOI and lock the deposit); publish is either manual on Zenodo or
done by a separate Action inside the curated dataset repo once a PR is
merged.

Environment
-----------

PAPERLENS_ZENODO_TOKEN     Personal access token with ``deposit:write`` and
                           ``deposit:actions`` scopes.  When missing, the
                           client is "not configured" and the donation
                           flow skips the Zenodo step entirely.
PAPERLENS_ZENODO_SANDBOX   Truthy → use ``sandbox.zenodo.org`` (throwaway
                           DOIs, good for dev).  Falsy → production
                           ``zenodo.org`` (real DOIs).

API flow (per deposit)
----------------------

1. ``POST /api/deposit/depositions`` with ``{}``
   → returns ``{id, links: {bucket, html, ...}}``
2. For each file in the bundle: ``PUT <bucket>/<filename>`` with the
   file body as raw bytes — Zenodo's bucket API streams uploads, no
   multipart needed.
3. ``PUT /api/deposit/depositions/{id}`` with the metadata block.
4. (NOT done here) ``POST /api/deposit/depositions/{id}/actions/publish``
   to mint the DOI and lock the deposit.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


_SANDBOX_BASE = "https://sandbox.zenodo.org/api"
_PROD_BASE    = "https://zenodo.org/api"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


# ── DISABLED: Zenodo integration paused (2026-06) ──────────────────
# The donation flow no longer mints Zenodo DOIs.  ``is_configured()``
# is hard-wired to False so every caller's existing ``if zenodo.is_configured():``
# guard skips the deposit step.  To re-enable: delete this banner and
# restore the original body (read PAPERLENS_ZENODO_TOKEN from env).
# The rest of this module (deposit creation, file upload, metadata) is
# kept intact so a re-enable is a one-line change here.
def is_configured() -> bool:
    """DISABLED — always returns False while the Zenodo step is paused."""
    return False


def _base_url() -> str:
    return _SANDBOX_BASE if _truthy(os.environ.get("PAPERLENS_ZENODO_SANDBOX")) else _PROD_BASE


def _token() -> str:
    tok = os.environ.get("PAPERLENS_ZENODO_TOKEN", "")
    if not tok:
        raise RuntimeError("PAPERLENS_ZENODO_TOKEN is not set")
    return tok


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept":        "application/json",
    }


# ── Metadata assembly ─────────────────────────────────────────────────────────

def _creators_block(attribution_mode: str, name: str, affiliation: str) -> list[dict[str, str]]:
    """Build the ``creators`` array Zenodo requires.  Always at least one
    entry — anonymous donations get a single ``Anonymous`` creator since
    Zenodo rejects empty arrays."""
    if attribution_mode == "attributed" and name:
        entry: dict[str, str] = {"name": name}
        if affiliation:
            entry["affiliation"] = affiliation
        return [entry]
    return [{"name": "Anonymous"}]


def _metadata_block(
    *,
    title: str,
    description: str,
    attribution_mode: str,
    attribution_name: str,
    attribution_affiliation: str,
    dataset_id: str,
    github_pr_url: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "title":         title,
        "description":   description or f"Dataset <code>{dataset_id}</code> donated via MetaPaperLens.",
        "upload_type":   "dataset",
        "creators":      _creators_block(attribution_mode, attribution_name, attribution_affiliation),
        "access_right":  "open",
        "license":       "cc-by-4.0",
        "keywords":      ["MetaPaperLens", "research-extraction", dataset_id],
    }
    # Cross-link to the GitHub PR when we have one — Zenodo renders these
    # as "Related identifiers" so anyone viewing the deposit can jump back
    # to the source/discussion.
    if github_pr_url:
        body["related_identifiers"] = [{
            "identifier":  github_pr_url,
            "relation":    "isSupplementTo",
            "resource_type": "publication-other",
            "scheme":      "url",
        }]
    return {"metadata": body}


# ── Public entry point ────────────────────────────────────────────────────────

def _raise_with_body(r: httpx.Response, step: str) -> None:
    """``r.raise_for_status()`` swallows the response body, so a 403
    becomes the useless ``Client error '403 FORBIDDEN' for URL ...``.
    Zenodo always returns a JSON body explaining what went wrong —
    bubble that through instead."""
    if r.is_success:
        return
    try:
        body = r.json()
        msg = body.get("message") or body.get("status") or str(body)[:300]
    except ValueError:
        msg = (r.text or "")[:300]
    raise RuntimeError(
        f"Zenodo {step} returned HTTP {r.status_code}: {msg}.  "
        f"Common causes: token missing 'deposit:write' scope; sandbox token "
        f"sent to production (or vice versa); account hasn't accepted the "
        f"Zenodo terms of use yet (try creating one deposit via the web UI "
        f"first to trigger the ToS prompt)."
    )


def create_draft_deposit(
    *,
    files: dict[str, str],
    title: str,
    description: str,
    attribution_mode: str,
    attribution_name: str,
    attribution_affiliation: str,
    dataset_id: str,
    github_pr_url: str | None = None,
) -> dict[str, Any]:
    """Create a Zenodo draft deposit, upload the bundle files, set
    metadata, and return the result dict.  Caller is responsible for
    catching exceptions and degrading gracefully (the donation flow does
    this — a failed Zenodo step still leaves a successful GitHub PR).

    Returns:
        {
          "deposit_id": int,
          "html_url":   str,   # browser URL the user visits to review/publish
          "doi":        str | None,  # pre-reserved DOI (becomes real on publish)
        }
    """
    if not is_configured():
        raise RuntimeError("Zenodo is not configured (PAPERLENS_ZENODO_TOKEN missing)")

    base = _base_url()
    with httpx.Client(timeout=60.0, headers=_auth_headers()) as client:
        # 1. Create empty draft deposit.
        r = client.post(f"{base}/deposit/depositions", json={})
        _raise_with_body(r, "deposit creation")
        deposit = r.json()
        deposit_id = deposit["id"]
        bucket_url = deposit["links"]["bucket"]
        html_url   = deposit["links"].get("html") or f"{base}/deposit/{deposit_id}"
        # Zenodo reserves a DOI at draft creation; it becomes "live" only
        # after publish but the string is stable.
        prereserved_doi = (deposit.get("metadata") or {}).get("prereserve_doi", {}).get("doi")

        # 2. Upload each bundle file via the bucket API (one PUT per file).
        for filename, content in files.items():
            r = client.put(
                f"{bucket_url}/{filename}",
                content=content.encode("utf-8"),
                headers={"Content-Type": "application/octet-stream"},
            )
            _raise_with_body(r, f"file upload ({filename})")

        # 3. Set deposit metadata.
        meta = _metadata_block(
            title                   = title,
            description             = description,
            attribution_mode        = attribution_mode,
            attribution_name        = attribution_name,
            attribution_affiliation = attribution_affiliation,
            dataset_id              = dataset_id,
            github_pr_url           = github_pr_url,
        )
        r = client.put(f"{base}/deposit/depositions/{deposit_id}", json=meta)
        _raise_with_body(r, "metadata update")

        return {
            "deposit_id": deposit_id,
            "html_url":   html_url,
            "doi":        prereserved_doi,
        }
