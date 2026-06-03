"""Donation pipeline: build a citable dataset bundle from a finished batch and
open a PR against the curated public repo.

Flow
----

POST /api/donate {batch_id, title, attribution, visibility, consents}
  → ``donate()``  (this module)
    1.  Pull all done jobs in the batch from SQLite.
    2.  Strip each result to a publishable whitelist (no PDFs, no raw bytes).
    3.  Hash the donor's IP (peppered SHA-256) for rate-limit dedup.
    4.  Render ``results.json``, ``prompt.md``, ``metadata.json``,
        ``README.md``, ``CITATION.cff``.
    5.  In dry-run (default): write the bundle to /tmp/paperlens-donations/<id>
        and return a stub result.
        In live mode (PAPERLENS_DONATE_LIVE=1): open a PR via GitHub App auth.

GitHub App auth (two-step)
--------------------------

1.  Sign an RS256 JWT with the App's private key.  ``iss`` is the App ID
    (or Client ID — both work for the JWT signature, but our env var name
    is PAPERLENS_GH_APP_ID for clarity).
2.  POST that JWT to ``/app/installations/{id}/access_tokens`` to get an
    installation access token, valid ~1h.  We cache it in-process until 60s
    before expiry to avoid signing a fresh JWT on every PR.

Environment
-----------

PAPERLENS_DONATE_ENABLED   "1" to expose /api/donate at all (off by default).
PAPERLENS_DONATE_LIVE      "1" to actually open PRs (off by default → dry-run).
PAPERLENS_DONATE_IP_PEPPER 32-byte hex; ``sha256(pepper + ip)`` powers the rate-limit.
PAPERLENS_GH_REPO          "owner/repo", e.g. "johanna-einsiedler/metalens-datasets".
PAPERLENS_GH_APP_ID        GitHub App ID (integer string).
PAPERLENS_GH_INSTALLATION_ID  GitHub App installation ID (integer string).
PAPERLENS_GH_APP_PRIVATE_KEY  PEM contents of the App's private key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bcrypt
import httpx
import jwt

import db
import zenodo
from pdf_utils import _parse_result_json


# ── Configuration helpers ─────────────────────────────────────────────────────

GH_API = "https://api.github.com"
DRY_RUN_BUNDLE_DIR = Path(tempfile.gettempdir()) / "paperlens-donations"

# Fields we propagate from a job's result JSON into the published dataset.
# Anything outside this whitelist is dropped — no raw PDF bytes, no
# page-image blobs, no cached scan markers.  Adding a field here means
# committing it to the public format.
_PUBLISH_TOP_LEVEL_KEYS = {
    "paper_metadata",
    "samples",
    "summaries",
    "records",
    "studies",
    "evidence",
    "metric",
    "notes",
    "schema_version",
}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """``/api/donate`` is only exposed when this is True."""
    return _truthy(os.environ.get("PAPERLENS_DONATE_ENABLED"))


def is_live() -> bool:
    """Live mode actually POSTs to GitHub.  Off by default → dry-run."""
    return _truthy(os.environ.get("PAPERLENS_DONATE_LIVE"))


def hash_ip(ip: str) -> str:
    """SHA-256 over ``pepper + ip``.  Used purely for rate-limit dedup —
    the pepper makes log-leak attacks useless (you can't reverse the hash
    to the IP without it)."""
    pepper = os.environ.get("PAPERLENS_DONATE_IP_PEPPER", "")
    if not pepper:
        # Defensive: a missing pepper would make hashes predictable.  Fail
        # closed rather than silently degrade.
        raise RuntimeError("PAPERLENS_DONATE_IP_PEPPER is not set")
    return hashlib.sha256((pepper + (ip or "")).encode("utf-8")).hexdigest()


# ── GitHub App auth ───────────────────────────────────────────────────────────

# Cached installation token: (token_string, expiry_unix).  We refresh ~60s
# before expiry.  Single-process cache is fine — Fly restarts cycle the
# process and a fresh token is cheap to obtain.
_INSTALLATION_TOKEN_CACHE: tuple[str, float] | None = None


def _sign_app_jwt() -> str:
    app_id = os.environ.get("PAPERLENS_GH_APP_ID")
    private_key = os.environ.get("PAPERLENS_GH_APP_PRIVATE_KEY")
    if not app_id or not private_key:
        raise RuntimeError("PAPERLENS_GH_APP_ID / PAPERLENS_GH_APP_PRIVATE_KEY missing")
    now = int(time.time())
    payload = {
        "iat": now - 60,        # 60s back-tolerance for clock skew
        "exp": now + 9 * 60,    # GitHub caps JWTs at 10 min — stay below
        "iss": app_id,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def _get_installation_token(client: httpx.Client) -> str:
    global _INSTALLATION_TOKEN_CACHE
    now = time.time()
    if _INSTALLATION_TOKEN_CACHE and _INSTALLATION_TOKEN_CACHE[1] - 60 > now:
        return _INSTALLATION_TOKEN_CACHE[0]
    install_id = os.environ.get("PAPERLENS_GH_INSTALLATION_ID")
    if not install_id:
        raise RuntimeError("PAPERLENS_GH_INSTALLATION_ID missing")
    app_jwt = _sign_app_jwt()
    r = client.post(
        f"{GH_API}/app/installations/{install_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {app_jwt}",
            "Accept":        "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30.0,
    )
    if r.status_code != 201:
        raise RuntimeError(f"GitHub installation-token exchange failed: {r.status_code} {r.text}")
    body = r.json()
    token = body["token"]
    # expires_at is ISO-8601; parse to unix.  Easier: GitHub returns ~1h, so
    # just stamp +50 min and trust the API.
    _INSTALLATION_TOKEN_CACHE = (token, now + 50 * 60)
    return token


# ── Schema strip ──────────────────────────────────────────────────────────────

def _strip_to_publishable(result_text: str | None) -> dict[str, Any] | None:
    """Parse a job's raw result string and return the whitelisted publishable
    subset.  Returns None if the result isn't parseable as JSON or is empty.

    Uses ``pdf_utils._parse_result_json`` so the same fence-stripping,
    preamble-stripping and truncation-repair logic that powers the result
    viewer also applies here — without it, Gemini's ```json ... ``` wrappers
    cause every paper to be dropped silently."""
    if not result_text:
        return None
    parsed = _parse_result_json(result_text)
    if not isinstance(parsed, dict):
        return None
    return {k: v for k, v in parsed.items() if k in _PUBLISH_TOP_LEVEL_KEYS}


# ── Slug / metadata helpers ───────────────────────────────────────────────────

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify_title(title: str) -> str:
    """ASCII kebab-case slug.  Empty → 'untitled'."""
    s = (title or "").strip().lower()
    s = _SLUG_NON_ALNUM.sub("-", s).strip("-")
    return s or "untitled"


def build_dataset_id(title: str, *, now: float | None = None) -> str:
    """``<slug>-<YYYY-MM>``.  Caller may pass ``now`` for deterministic tests."""
    t = now if now is not None else time.time()
    ts = time.strftime("%Y-%m", time.gmtime(t))
    return f"{slugify_title(title)}-{ts}"


def hash_password(plaintext: str) -> str:
    """bcrypt hash; cost 12.  Salt is embedded in the returned string."""
    return bcrypt.hashpw(plaintext.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plaintext: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plaintext.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── Bundle assembly ───────────────────────────────────────────────────────────

@dataclass
class DonationRequest:
    batch_id: str
    title: str
    description: str
    attribution_mode: str        # 'anonymous' | 'attributed'
    attribution_name: str        # '' when anonymous
    attribution_affiliation: str # '' when anonymous
    visibility: str              # 'public' | 'gated'
    password: str                # '' when public


def _load_batch_jobs(batch_id: str) -> list[dict[str, Any]]:
    """Done jobs from the batch, ordered by creation time."""
    jobs = db.list_jobs_in_batch(batch_id)
    return [j for j in jobs if j.get("status") == "done"]


def build_bundle(req: DonationRequest, *, now: float | None = None) -> dict[str, Any]:
    """Build the in-memory dataset bundle.

    Returns a dict with:
      - ``dataset_id``:  the slug
      - ``files``:       {"results.json": "<text>", "prompt.md": "<text>", ...}
      - ``schema_version``: detected from first parseable result
      - ``paper_count``: how many papers landed in results.json
      - ``prompt_sha256``: integrity hash of the prompt
    """
    jobs = _load_batch_jobs(req.batch_id)
    if not jobs:
        raise ValueError("No completed jobs in this batch — nothing to donate.")

    # The prompt is identical across the batch (all jobs share the same
    # prompt body).  Take the first non-null one and verify the others match.
    prompts = {j.get("prompt") for j in jobs if j.get("prompt")}
    if len(prompts) > 1:
        # Shouldn't happen in normal use, but be explicit if it does.
        raise ValueError("Jobs in this batch do not share a single prompt — refusing to donate.")
    prompt_body = (prompts.pop() if prompts else "") or "(prompt not recorded)"
    prompt_sha256 = hashlib.sha256(prompt_body.encode("utf-8")).hexdigest()

    # Per-paper publishable subset.
    papers: list[dict[str, Any]] = []
    schema_versions: set[str] = set()
    for j in jobs:
        stripped = _strip_to_publishable(j.get("result"))
        if stripped is None:
            continue
        sv = stripped.get("schema_version")
        if isinstance(sv, str):
            schema_versions.add(sv)
        papers.append({
            "filename":        j.get("filename"),
            "model":           j.get("model"),
            "resolved_model":  j.get("resolved_model"),
            "pages_processed": j.get("pages_processed"),
            "evidence_count":  j.get("evidence_count"),
            "result":          stripped,
        })

    if not papers:
        raise ValueError("No publishable results in this batch.")

    schema_version = (
        next(iter(schema_versions)) if len(schema_versions) == 1
        else "mixed"
    )

    dataset_id = build_dataset_id(req.title, now=now)
    iso_date = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(now if now is not None else time.time()))

    # metadata.json
    donor_block: dict[str, Any]
    if req.attribution_mode == "attributed":
        donor_block = {
            "mode":        "attributed",
            "name":        req.attribution_name,
            "affiliation": req.attribution_affiliation,
        }
    else:
        donor_block = {"mode": "anonymous"}

    metadata: dict[str, Any] = {
        "schema_version":  "metadata-v1",
        "dataset_id":      dataset_id,
        "title":           req.title,
        "description":     req.description,
        "donor":           donor_block,
        "visibility":      req.visibility,
        "extraction": {
            "schema_version": schema_version,
            "paper_count":    len(papers),
            "model_used":     sorted({p["model"] for p in papers if p.get("model")}),
            "prompt_sha256":  prompt_sha256,
        },
        "created_at":      iso_date,
    }
    if req.visibility == "gated":
        if not req.password:
            raise ValueError("Gated datasets require a password.")
        metadata["password_hash"] = hash_password(req.password)

    # results.json
    results_doc = {
        "schema_version": schema_version,
        "dataset_id":     dataset_id,
        "papers":         papers,
    }

    # README.md — terse, citation-block-friendly
    donor_line = (
        f"{req.attribution_name}"
        + (f" ({req.attribution_affiliation})" if req.attribution_affiliation else "")
        if req.attribution_mode == "attributed" and req.attribution_name
        else "Anonymous"
    )
    description_block = f"\n\n{req.description}\n" if req.description else "\n"
    readme = (
        f"# {req.title}\n\n"
        f"Dataset ID: `{dataset_id}`  \n"
        f"Schema version: `{schema_version}`  \n"
        f"Donor: {donor_line}  \n"
        f"Visibility: **{req.visibility}**  \n"
        f"Created: {iso_date}{description_block}\n"
        f"## Contents\n\n"
        f"- `results.json` — {len(papers)} extracted paper(s) in the schema declared above.\n"
        f"- `prompt.md` — the exact prompt used during extraction (sha256 `{prompt_sha256[:12]}…`).\n"
        f"- `metadata.json` — donor + provenance metadata.\n"
        f"- `CITATION.cff` — machine-readable citation block.\n\n"
        f"## Citation\n\n"
        f"```\n"
        f"{donor_line}. {req.title}. MetaPaperLens dataset {dataset_id}, {iso_date[:10]}.\n"
        f"```\n\n"
        f"## Reproducibility\n\n"
        f"This dataset was produced by an LLM-based extraction pipeline. "
        f"The exact prompt used is stored in `prompt.md`; its SHA-256 is "
        f"recorded in `metadata.json` so any rerun against the same model "
        f"and prompt can be verified for fidelity.\n"
    )

    # CITATION.cff — minimal but valid
    authors_block: list[dict[str, str]]
    if req.attribution_mode == "attributed" and req.attribution_name:
        authors_block = [{"name": req.attribution_name}]
        if req.attribution_affiliation:
            authors_block[0]["affiliation"] = req.attribution_affiliation
    else:
        authors_block = [{"name": "Anonymous"}]
    cff_lines = [
        "cff-version: 1.2.0",
        f'title: "{req.title}"',
        f"date-released: {iso_date[:10]}",
        "type: dataset",
        "authors:",
    ]
    for a in authors_block:
        cff_lines.append(f'  - name: "{a["name"]}"')
        if "affiliation" in a:
            cff_lines.append(f'    affiliation: "{a["affiliation"]}"')
    cff_lines.append(f'identifiers:')
    cff_lines.append(f'  - type: other')
    cff_lines.append(f'    value: "metalens:{dataset_id}"')
    cff_text = "\n".join(cff_lines) + "\n"

    return {
        "dataset_id":     dataset_id,
        "schema_version": schema_version,
        "paper_count":    len(papers),
        "prompt_sha256":  prompt_sha256,
        "files": {
            "results.json":  json.dumps(results_doc, indent=2, ensure_ascii=False) + "\n",
            "prompt.md":     prompt_body + ("\n" if not prompt_body.endswith("\n") else ""),
            "metadata.json": json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            "README.md":     readme,
            "CITATION.cff":  cff_text,
        },
    }


# ── GitHub PR creation ────────────────────────────────────────────────────────

def _gh_repo() -> tuple[str, str]:
    repo = os.environ.get("PAPERLENS_GH_REPO", "")
    if "/" not in repo:
        raise RuntimeError("PAPERLENS_GH_REPO must be 'owner/repo'")
    owner, name = repo.split("/", 1)
    return owner, name


def _open_pr(
    files: dict[str, str],
    *,
    dataset_id: str,
    branch_name: str,
    pr_title: str,
    pr_body: str,
    extends_dataset_id: str | None,
) -> dict[str, Any]:
    """Six-call sequence: ref → blobs → tree → commit → branch ref → PR.

    File paths are placed under ``datasets/<dataset_id>/`` for fresh donations.
    For extensions, files land under ``datasets/<extends_dataset_id>/papers/<dataset_id>.json``
    so they append rather than overwrite — but Phase-2 only opens the
    base case; extension submissions are Phase 3.
    """
    owner, name = _gh_repo()
    with httpx.Client(timeout=60.0) as client:
        token = _get_installation_token(client)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        base_url = f"{GH_API}/repos/{owner}/{name}"

        # 1. Get default-branch SHA so we know what to base on.
        r = client.get(f"{base_url}", headers=headers)
        r.raise_for_status()
        default_branch = r.json()["default_branch"]
        r = client.get(f"{base_url}/git/ref/heads/{default_branch}", headers=headers)
        # Empty repos return 409 ("Git Repository is empty") on the ref
        # lookup; 404 also surfaces when the branch hasn't been pushed.
        # Either way the fix is the same: the repo needs one initial
        # commit on the default branch.  Translate both to an actionable
        # error so the modal shows something useful instead of a raw 500.
        if r.status_code in (404, 409):
            raise RuntimeError(
                f"The dataset repo '{owner}/{name}' has no commits on its "
                f"default branch ({default_branch}).  Add one initial "
                f"commit (e.g. a README) on GitHub and retry — the donation "
                f"bot needs an existing branch to base its PR on."
            )
        r.raise_for_status()
        base_sha = r.json()["object"]["sha"]

        # 2. Create one blob per file.
        blobs: list[dict[str, str]] = []
        for rel_path, content in files.items():
            r = client.post(
                f"{base_url}/git/blobs",
                headers=headers,
                json={"content": content, "encoding": "utf-8"},
            )
            r.raise_for_status()
            blob_sha = r.json()["sha"]
            blobs.append({
                "path": f"datasets/{dataset_id}/{rel_path}",
                "mode": "100644",
                "type": "blob",
                "sha":  blob_sha,
            })

        # 3. Create a tree pointing at those blobs, based on the default branch.
        r = client.post(
            f"{base_url}/git/trees",
            headers=headers,
            json={"base_tree": base_sha, "tree": blobs},
        )
        r.raise_for_status()
        tree_sha = r.json()["sha"]

        # 4. Create the commit.
        r = client.post(
            f"{base_url}/git/commits",
            headers=headers,
            json={
                "message": f"Add dataset {dataset_id}\n\n{pr_title}",
                "tree":    tree_sha,
                "parents": [base_sha],
            },
        )
        r.raise_for_status()
        commit_sha = r.json()["sha"]

        # 5. Create the branch ref.
        r = client.post(
            f"{base_url}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": commit_sha},
        )
        # 422 happens if the branch already exists — push a fresh attempt name.
        if r.status_code == 422:
            branch_name = f"{branch_name}-{int(time.time())}"
            r = client.post(
                f"{base_url}/git/refs",
                headers=headers,
                json={"ref": f"refs/heads/{branch_name}", "sha": commit_sha},
            )
        r.raise_for_status()

        # 6. Open the PR.
        r = client.post(
            f"{base_url}/pulls",
            headers=headers,
            json={
                "title": pr_title,
                "body":  pr_body,
                "head":  branch_name,
                "base":  default_branch,
            },
        )
        r.raise_for_status()
        pr = r.json()
        return {
            "pr_url":    pr["html_url"],
            "pr_number": pr["number"],
            "branch":    branch_name,
        }


# ── Public entry point ────────────────────────────────────────────────────────

def donate(req: DonationRequest, *, ip_hash: str) -> dict[str, Any]:
    """Run the donation end-to-end.  Returns the donation result dict.

    Always writes a row to the ``donations`` table for audit.  In dry-run
    mode the bundle is written to /tmp/paperlens-donations/<dataset_id>/ and
    no GitHub call is made; the row's status becomes ``"dry-run"``.
    """
    bundle = build_bundle(req)
    dataset_id = bundle["dataset_id"]
    donation_id = uuid.uuid4().hex

    db.create_donation(
        donation_id,
        batch_id=req.batch_id,
        dataset_id=dataset_id,
        extends_dataset_id=None,
        title=req.title,
        visibility=req.visibility,
        attribution_mode=req.attribution_mode,
        ip_hash=ip_hash,
    )

    if not is_live():
        # Dry-run: write bundle to disk so we can inspect it.
        target = DRY_RUN_BUNDLE_DIR / dataset_id
        target.mkdir(parents=True, exist_ok=True)
        for rel_path, content in bundle["files"].items():
            (target / rel_path).write_text(content, encoding="utf-8")
        db.update_donation_result(donation_id, status="dry-run")
        return {
            "donation_id":     donation_id,
            "dataset_id":      dataset_id,
            "mode":            "dry-run",
            "bundle_path":     str(target),
            "paper_count":     bundle["paper_count"],
            "schema_version":  bundle["schema_version"],
        }

    # Live: open the PR, then (optionally) create a Zenodo draft deposit.
    try:
        donor_line = (
            req.attribution_name
            if req.attribution_mode == "attributed" and req.attribution_name
            else "Anonymous"
        )
        pr = _open_pr(
            bundle["files"],
            dataset_id=dataset_id,
            branch_name=f"dataset/{dataset_id}",
            pr_title=f"Add dataset: {req.title}",
            pr_body=(
                f"**Dataset ID:** `{dataset_id}`\n"
                f"**Donor:** {donor_line}\n"
                f"**Visibility:** {req.visibility}\n"
                f"**Paper count:** {bundle['paper_count']}\n"
                f"**Schema:** `{bundle['schema_version']}`\n\n"
                f"_Opened by the MetaPaperLens donation bot.  "
                f"Review the dataset folder, then merge — the merge "
                f"action will trigger Zenodo DOI minting (when wired)._\n"
            ),
            extends_dataset_id=None,
        )
        db.update_donation_result(
            donation_id,
            status="pr-opened",
            github_pr_url=pr["pr_url"],
            github_pr_number=pr["pr_number"],
        )

        # Zenodo step is best-effort.  A failure here doesn't undo the
        # PR — the GitHub side is the durable record.  We log the
        # error on the donation row so curators can retry the deposit
        # manually if needed, but the user still gets a successful
        # response (with just the PR URL).
        zenodo_info: dict[str, Any] | None = None
        zenodo_error: str | None = None
        if zenodo.is_configured():
            try:
                zenodo_info = zenodo.create_draft_deposit(
                    files                   = bundle["files"],
                    title                   = req.title,
                    description             = req.description,
                    attribution_mode        = req.attribution_mode,
                    attribution_name        = req.attribution_name,
                    attribution_affiliation = req.attribution_affiliation,
                    dataset_id              = dataset_id,
                    github_pr_url           = pr["pr_url"],
                )
                db.update_donation_result(
                    donation_id,
                    status="pr-opened",
                    zenodo_deposit_id=str(zenodo_info["deposit_id"]),
                )
            except Exception as exc:  # noqa: BLE001
                zenodo_error = f"Zenodo deposit failed: {exc}"
                db.update_donation_result(
                    donation_id,
                    status="pr-opened",
                    error=zenodo_error,
                )

        result: dict[str, Any] = {
            "donation_id":    donation_id,
            "dataset_id":     dataset_id,
            "mode":           "live",
            "pr_url":         pr["pr_url"],
            "pr_number":      pr["pr_number"],
            "paper_count":    bundle["paper_count"],
            "schema_version": bundle["schema_version"],
        }
        if zenodo_info:
            result["zenodo_html_url"]   = zenodo_info["html_url"]
            result["zenodo_deposit_id"] = zenodo_info["deposit_id"]
            result["zenodo_doi"]        = zenodo_info.get("doi")
        if zenodo_error:
            result["zenodo_error"] = zenodo_error
        return result
    except Exception as exc:  # noqa: BLE001
        db.update_donation_result(donation_id, status="failed", error=str(exc))
        raise
