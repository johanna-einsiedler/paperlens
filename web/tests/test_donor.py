"""Unit tests for donor.py.

These exercise the pure helpers (schema strip, slugify, password hash,
bundle assembly) and the dry-run end-to-end path.  The live GitHub PR
path is NOT covered here — it makes real network calls and needs the
GitHub App credentials; a smoke test against a sandbox repo lives
outside the test suite.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

import db
import donor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(monkeypatch, tmp_path):
    """Isolated SQLite DB per test."""
    monkeypatch.setenv("PAPERLENS_DB_PATH", str(tmp_path / "test.sqlite"))
    db.init()
    return tmp_path / "test.sqlite"


@pytest.fixture
def pepper(monkeypatch):
    """Predictable pepper so hash_ip is deterministic across tests."""
    monkeypatch.setenv("PAPERLENS_DONATE_IP_PEPPER", "test-pepper")


@pytest.fixture
def seeded_batch(db_path):
    """A batch with two completed jobs.  Returns the batch_id."""
    batch_id = uuid.uuid4().hex
    db.create_batch(batch_id, notify_email=None, session_id="test")
    for i, fname in enumerate(["paper1.pdf", "paper2.pdf"]):
        job_id = uuid.uuid4().hex
        db.create_job(
            job_id, fname,
            batch_id=batch_id,
            prompt="The shared extraction prompt across the batch.",
            model="gpt-4o-mini",
        )
        db.mark_done(
            job_id,
            result=json.dumps({
                "paper_metadata": {"title": f"Paper {i+1}", "doi": None, "year": 2024, "authors": ["Smith J"]},
                "samples":        [{"sample_id": "S1", "n": 100 + i, "evidence": []}],
                "evidence":       [{"snippet": "...", "page": 1, "source": None, "field": "samples[0]"}],
                "schema_version": "masem-v3",
                # An intentionally-non-publishable field that the strip
                # must drop — pretend it's a 5MB page-image blob.
                "pageImages":     "PRETEND-THIS-IS-5MB-OF-BASE64",
            }),
            pages_processed=8 + i,
            evidence_count=1,
            finish_reason="stop",
            token_usage={"prompt": 1000, "completion": 200, "total": 1200},
            resolved_model="gpt-4o-mini-2024-07-18",
        )
    return batch_id


# ── Pure helpers ──────────────────────────────────────────────────────────────

def test_slugify_title_basic():
    assert donor.slugify_title("Need for Cognition Scale (NCS-18)") == "need-for-cognition-scale-ncs-18"
    assert donor.slugify_title("  Leading & trailing!! ") == "leading-trailing"
    assert donor.slugify_title("") == "untitled"


def test_build_dataset_id_includes_year_month():
    # Fixed timestamp: 2024-03-15
    fixed = 1710460800  # 2024-03-15 00:00 UTC
    assert donor.build_dataset_id("Hello World", now=fixed).startswith("hello-world-2024-03")


def test_hash_password_roundtrip():
    h = donor.hash_password("correct horse battery staple")
    assert donor.verify_password("correct horse battery staple", h)
    assert not donor.verify_password("wrong password", h)


def test_hash_ip_requires_pepper(monkeypatch):
    monkeypatch.delenv("PAPERLENS_DONATE_IP_PEPPER", raising=False)
    with pytest.raises(RuntimeError, match="PAPERLENS_DONATE_IP_PEPPER"):
        donor.hash_ip("1.2.3.4")


def test_hash_ip_deterministic(pepper):
    assert donor.hash_ip("1.2.3.4") == donor.hash_ip("1.2.3.4")
    assert donor.hash_ip("1.2.3.4") != donor.hash_ip("5.6.7.8")


def test_strip_to_publishable_drops_pageimages():
    raw = json.dumps({
        "samples":        [{"sample_id": "S1"}],
        "paper_metadata": {"title": "T"},
        "pageImages":     "MUST-BE-STRIPPED",
        "raw_response":   "MUST-BE-STRIPPED",
    })
    stripped = donor._strip_to_publishable(raw)
    assert "pageImages" not in stripped
    assert "raw_response" not in stripped
    assert stripped["samples"] == [{"sample_id": "S1"}]
    assert stripped["paper_metadata"] == {"title": "T"}


def test_strip_to_publishable_handles_bad_json():
    assert donor._strip_to_publishable("not json") is None
    assert donor._strip_to_publishable("") is None
    assert donor._strip_to_publishable(None) is None


# ── build_bundle ──────────────────────────────────────────────────────────────

def _make_req(**overrides) -> donor.DonationRequest:
    base = dict(
        batch_id                = "",
        title                   = "NCS-18 factor loadings",
        description             = "Pilot dataset from the example PDFs.",
        attribution_mode        = "anonymous",
        attribution_name        = "",
        attribution_affiliation = "",
        visibility              = "public",
        password                = "",
    )
    base.update(overrides)
    return donor.DonationRequest(**base)


def test_build_bundle_strips_pageimages_in_results(seeded_batch):
    req = _make_req(batch_id=seeded_batch)
    bundle = donor.build_bundle(req)
    assert "pageImages" not in bundle["files"]["results.json"]
    assert bundle["paper_count"] == 2
    assert bundle["schema_version"] == "masem-v3"


def test_build_bundle_schema_version_unspecified_when_model_omits(db_path):
    """A paper whose result JSON doesn't include schema_version should
    produce a dataset labelled 'unspecified', NOT 'mixed' — the
    earlier behaviour was misleading (it suggested a conflict that
    didn't exist).  Reproduces the test2-2026-06 regression seen in
    production."""
    import uuid as _uuid
    batch_id = _uuid.uuid4().hex
    db.create_batch(batch_id, notify_email=None, session_id="test")
    job_id = _uuid.uuid4().hex
    db.create_job(job_id, "p.pdf", batch_id=batch_id, prompt="P", model="gpt-4o")
    db.mark_done(job_id,
        result=json.dumps({
            "paper_metadata": {"title": "T", "doi": None, "year": None, "authors": None},
            "samples": [{"sample_id": "S1"}],
            "evidence": [],
            # NB: no schema_version key — mirrors what gemini-2.5-pro
            # emitted for the user's test2 donation.
        }),
        pages_processed=1, evidence_count=0, finish_reason="stop",
        token_usage={})
    req = _make_req(batch_id=batch_id)
    bundle = donor.build_bundle(req)
    assert bundle["schema_version"] == "unspecified"
    assert bundle["paper_count"] == 1


def test_build_bundle_refuses_mixed_schema_donation(db_path):
    """Two papers with DIFFERENT schema_versions in one batch should
    be REFUSED at donation time — a published 'mixed' dataset is
    confusing for downstream consumers and a hard case for the
    Phase 3g extension-consistency check.  Fail loudly so the donor
    re-extracts under a single preset rather than shipping the
    ambiguity."""
    import uuid as _uuid
    batch_id = _uuid.uuid4().hex
    db.create_batch(batch_id, notify_email=None, session_id="test")
    for sv in ["masem-v3", "summarize-v1"]:
        job_id = _uuid.uuid4().hex
        db.create_job(job_id, f"p-{sv}.pdf", batch_id=batch_id, prompt="P", model="gpt-4o")
        db.mark_done(job_id,
            result=json.dumps({
                "samples": [{"sample_id": "S1"}],
                "evidence": [],
                "schema_version": sv,
            }),
            pages_processed=1, evidence_count=0, finish_reason="stop",
            token_usage={})
    req = _make_req(batch_id=batch_id)
    with pytest.raises(ValueError, match="different schema_version"):
        donor.build_bundle(req)


def test_build_bundle_records_prompt(seeded_batch):
    req = _make_req(batch_id=seeded_batch)
    bundle = donor.build_bundle(req)
    assert "shared extraction prompt" in bundle["files"]["prompt.md"]
    # Metadata must reference the prompt's sha256 so anyone can verify integrity.
    md = json.loads(bundle["files"]["metadata.json"])
    assert md["extraction"]["prompt_sha256"] == bundle["prompt_sha256"]


def test_build_bundle_gated_requires_password(seeded_batch):
    req = _make_req(batch_id=seeded_batch, visibility="gated", password="")
    with pytest.raises(ValueError, match="password"):
        donor.build_bundle(req)


def test_build_bundle_gated_stores_password_hash(seeded_batch):
    req = _make_req(batch_id=seeded_batch, visibility="gated", password="extend-me-please")
    bundle = donor.build_bundle(req)
    md = json.loads(bundle["files"]["metadata.json"])
    assert "password_hash" in md
    assert donor.verify_password("extend-me-please", md["password_hash"])
    assert not donor.verify_password("wrong", md["password_hash"])


def test_build_bundle_anonymous_omits_password_hash(seeded_batch):
    req = _make_req(batch_id=seeded_batch)
    bundle = donor.build_bundle(req)
    md = json.loads(bundle["files"]["metadata.json"])
    assert "password_hash" not in md
    assert md["donor"] == {"mode": "anonymous"}


def test_build_bundle_attributed_donor_block(seeded_batch):
    req = _make_req(
        batch_id=seeded_batch,
        attribution_mode="attributed",
        attribution_name="Jane Smith",
        attribution_affiliation="University of Basel",
    )
    bundle = donor.build_bundle(req)
    md = json.loads(bundle["files"]["metadata.json"])
    assert md["donor"] == {
        "mode": "attributed",
        "name": "Jane Smith",
        "affiliation": "University of Basel",
    }
    # CITATION.cff should carry the name too
    assert "Jane Smith" in bundle["files"]["CITATION.cff"]


def test_build_bundle_empty_batch_rejected(db_path):
    req = _make_req(batch_id="nonexistent-batch")
    with pytest.raises(ValueError, match="No completed jobs"):
        donor.build_bundle(req)


def test_build_bundle_multi_prompt_rejected(seeded_batch):
    # Add a third job with a *different* prompt — should refuse to donate.
    job_id = uuid.uuid4().hex
    db.create_job(
        job_id, "rogue.pdf",
        batch_id=seeded_batch,
        prompt="A DIFFERENT prompt — this batch is inconsistent.",
        model="gpt-4o-mini",
    )
    db.mark_done(
        job_id,
        result=json.dumps({"samples": [], "schema_version": "masem-v3"}),
        pages_processed=1,
        evidence_count=0,
        finish_reason="stop",
        token_usage={},
    )
    req = _make_req(batch_id=seeded_batch)
    with pytest.raises(ValueError, match="single prompt"):
        donor.build_bundle(req)


# ── dry-run end-to-end ────────────────────────────────────────────────────────

def test_donate_dry_run_writes_bundle(seeded_batch, pepper, monkeypatch, tmp_path):
    """Default: PAPERLENS_DONATE_LIVE is unset → dry-run → bundle written
    to a /tmp folder, no GitHub call, donation row marked 'dry-run'."""
    monkeypatch.delenv("PAPERLENS_DONATE_LIVE", raising=False)
    # Redirect DRY_RUN_BUNDLE_DIR to tmp_path so the test is hermetic.
    monkeypatch.setattr(donor, "DRY_RUN_BUNDLE_DIR", tmp_path)
    ip_hash = donor.hash_ip("1.2.3.4")
    req = _make_req(batch_id=seeded_batch)
    result = donor.donate(req, ip_hash=ip_hash)
    assert result["mode"] == "dry-run"
    bundle_dir = tmp_path / result["dataset_id"]
    assert (bundle_dir / "results.json").is_file()
    assert (bundle_dir / "prompt.md").is_file()
    assert (bundle_dir / "metadata.json").is_file()
    assert (bundle_dir / "README.md").is_file()
    assert (bundle_dir / "CITATION.cff").is_file()
    # Donation row exists with the right status.
    row = db.get_donation(result["donation_id"])
    assert row["status"] == "dry-run"
    assert row["ip_hash"] == ip_hash


def test_donate_rate_limit_counter(seeded_batch, pepper, monkeypatch, tmp_path):
    """Confirm count_donations_by_ip counts what we expect — the route
    layer enforces the threshold; this just verifies the underlying counter."""
    monkeypatch.delenv("PAPERLENS_DONATE_LIVE", raising=False)
    monkeypatch.setattr(donor, "DRY_RUN_BUNDLE_DIR", tmp_path)
    ip_hash = donor.hash_ip("1.2.3.4")
    req = _make_req(batch_id=seeded_batch)
    assert db.count_donations_by_ip(ip_hash, 3600) == 0
    donor.donate(req, ip_hash=ip_hash)
    assert db.count_donations_by_ip(ip_hash, 3600) == 1
    donor.donate(req, ip_hash=ip_hash)
    assert db.count_donations_by_ip(ip_hash, 3600) == 2


# ── Result parsing ────────────────────────────────────────────────────────────

def test_strip_to_publishable_handles_gemini_fences():
    """Gemini wraps JSON in ```json ... ``` fences — the publishable
    strip must use the tolerant parser so fenced output isn't silently
    dropped (which manifested as 'no publishable results in this batch'
    in production)."""
    fenced = '```json\n{"paper_metadata": {"title": "T"}, "samples": [{"sample_id": "S1"}]}\n```'
    stripped = donor._strip_to_publishable(fenced)
    assert stripped is not None
    assert stripped["paper_metadata"] == {"title": "T"}
    assert stripped["samples"] == [{"sample_id": "S1"}]


def test_strip_to_publishable_handles_gemini_fences_with_preamble():
    """And tolerates trailing prose after the JSON."""
    messy = '```json\n{"samples": [{"sample_id": "S1"}], "schema_version": "v1"}\n```\nAdditional notes here.'
    stripped = donor._strip_to_publishable(messy)
    assert stripped is not None
    assert stripped["samples"] == [{"sample_id": "S1"}]
    assert stripped["schema_version"] == "v1"


# ── Zenodo wiring ─────────────────────────────────────────────────────────────
# DISABLED (2026-06): the Zenodo integration is paused — see the banner
# at the top of web/zenodo.py.  ``is_configured()`` is hard-wired to
# False, so the env-reading test below would fail; the helper-level
# tests still pass (they only touch internal helpers) but we skip the
# whole section so the disabled-feature status is loud in test output.
# Re-enable by removing the @pytest.mark.skip decorators.

@pytest.mark.skip(reason="Zenodo integration disabled — see web/zenodo.py")
def test_donate_skips_zenodo_when_unconfigured(seeded_batch, pepper, monkeypatch, tmp_path):
    """When PAPERLENS_ZENODO_TOKEN isn't set, the donation flow should
    just skip the Zenodo step silently — no Zenodo URL in the result,
    no error logged, GitHub PR still succeeds (or in dry-run, no PR)."""
    monkeypatch.delenv("PAPERLENS_DONATE_LIVE", raising=False)
    monkeypatch.delenv("PAPERLENS_ZENODO_TOKEN", raising=False)
    monkeypatch.setattr(donor, "DRY_RUN_BUNDLE_DIR", tmp_path)
    req = _make_req(batch_id=seeded_batch)
    result = donor.donate(req, ip_hash=donor.hash_ip("9.9.9.9"))
    assert result["mode"] == "dry-run"
    assert "zenodo_html_url" not in result
    assert "zenodo_error" not in result


@pytest.mark.skip(reason="Zenodo integration disabled — see web/zenodo.py")
def test_zenodo_is_configured_reads_env(monkeypatch):
    import zenodo as zenodo_mod
    monkeypatch.delenv("PAPERLENS_ZENODO_TOKEN", raising=False)
    assert zenodo_mod.is_configured() is False
    monkeypatch.setenv("PAPERLENS_ZENODO_TOKEN", "fake-token")
    assert zenodo_mod.is_configured() is True


@pytest.mark.skip(reason="Zenodo integration disabled — see web/zenodo.py")
def test_zenodo_base_url_switches_on_sandbox_flag(monkeypatch):
    import zenodo as zenodo_mod
    monkeypatch.delenv("PAPERLENS_ZENODO_SANDBOX", raising=False)
    assert zenodo_mod._base_url() == "https://zenodo.org/api"
    monkeypatch.setenv("PAPERLENS_ZENODO_SANDBOX", "1")
    assert zenodo_mod._base_url() == "https://sandbox.zenodo.org/api"


@pytest.mark.skip(reason="Zenodo integration disabled — see web/zenodo.py")
def test_zenodo_creators_anonymous_and_attributed():
    import zenodo as zenodo_mod
    anon = zenodo_mod._creators_block("anonymous", "", "")
    assert anon == [{"name": "Anonymous"}]
    attr = zenodo_mod._creators_block("attributed", "Jane Smith", "University of Basel")
    assert attr == [{"name": "Jane Smith", "affiliation": "University of Basel"}]
    # Attributed-but-empty-name still falls back to Anonymous so Zenodo
    # never sees an invalid empty creators array.
    fallback = zenodo_mod._creators_block("attributed", "", "")
    assert fallback == [{"name": "Anonymous"}]


def test_build_bundle_records_verification_block_default_unverified(seeded_batch):
    """Default donation has no human-verification claim — metadata.json
    must carry ``verification: {human_verified: False, notes: ''}`` so
    downstream consumers can filter / weight verified vs raw outputs
    without having to interpret absent-as-true."""
    req = _make_req(batch_id=seeded_batch)
    bundle = donor.build_bundle(req)
    md = json.loads(bundle["files"]["metadata.json"])
    assert md["verification"] == {"human_verified": False, "notes": ""}


def test_build_bundle_records_human_verification_when_claimed(seeded_batch):
    """When the donor checks the human-verified radio + supplies notes,
    both fields land in the bundle's metadata.json verbatim."""
    req = _make_req(
        batch_id=seeded_batch,
        human_verified=True,
        verification_notes="all 53 effect-size records cross-checked against tables",
    )
    bundle = donor.build_bundle(req)
    md = json.loads(bundle["files"]["metadata.json"])
    assert md["verification"]["human_verified"] is True
    assert "53 effect-size records" in md["verification"]["notes"]


def test_build_extension_payload_records_verification(seeded_batch):
    """Extension submissions carry their own per-batch verification
    block in papers/<batch>.json — a dataset can have multiple
    extension batches with differing verification status."""
    req = _make_req(
        batch_id=seeded_batch,
        extends_dataset_id="ncs-18-2026-06",
        human_verified=True,
        verification_notes="spot-checked 5 of 8 papers",
    )
    payload = donor.build_extension_payload(req)
    parsed = json.loads(payload["file_content"])
    assert parsed["verification"]["human_verified"] is True
    assert "spot-checked 5 of 8" in parsed["verification"]["notes"]


def test_build_extension_payload_strips_pageimages(seeded_batch):
    """Phase 3e: extension payload still runs through the schema-strip
    whitelist — same posture as a fresh donation."""
    req = _make_req(batch_id=seeded_batch, extends_dataset_id="ncs-18-2026-06")
    payload = donor.build_extension_payload(req)
    assert payload["dataset_id"] == "ncs-18-2026-06"
    assert payload["paper_count"] == 2
    assert payload["file_path"] == f"papers/{seeded_batch}.json"
    assert "pageImages" not in payload["file_content"]


def test_build_extension_payload_records_prompt_sha(seeded_batch):
    req = _make_req(batch_id=seeded_batch, extends_dataset_id="ncs-18-2026-06")
    payload = donor.build_extension_payload(req)
    parsed = json.loads(payload["file_content"])
    assert parsed["extension_of"] == "ncs-18-2026-06"
    assert parsed["batch_id"] == seeded_batch
    assert parsed["schema_version"] == "masem-v3"
    assert parsed["extraction"]["prompt_sha256"] == payload["prompt_sha256"]


def test_build_extension_payload_attributed_donor(seeded_batch):
    req = _make_req(
        batch_id=seeded_batch,
        extends_dataset_id="ncs-18-2026-06",
        attribution_mode="attributed",
        attribution_name="Jane Smith",
        attribution_affiliation="Basel",
    )
    payload = donor.build_extension_payload(req)
    parsed = json.loads(payload["file_content"])
    assert parsed["donor"] == {
        "mode": "attributed",
        "name": "Jane Smith",
        "affiliation": "Basel",
    }


def test_build_extension_payload_requires_extends_id(seeded_batch):
    req = _make_req(batch_id=seeded_batch)  # extends_dataset_id is None
    with pytest.raises(ValueError, match="extends_dataset_id"):
        donor.build_extension_payload(req)


def test_donate_extension_dry_run_writes_to_papers_subfolder(
    seeded_batch, pepper, monkeypatch, tmp_path
):
    """Dry-run extension stages the new file under the EXISTING dataset's
    papers/ subfolder so reviewers can inspect what would be added."""
    monkeypatch.delenv("PAPERLENS_DONATE_LIVE", raising=False)
    monkeypatch.setattr(donor, "DRY_RUN_BUNDLE_DIR", tmp_path)
    # The extension flow checks that the target dataset actually
    # exists.  Stub get_dataset_metadata so the check passes without
    # hitting GitHub.
    monkeypatch.setattr(
        donor.datasets_mod, "get_dataset_metadata",
        lambda did: {"title": "Existing", "visibility": "public"},
    )
    ip_hash = donor.hash_ip("1.2.3.4")
    req = _make_req(batch_id=seeded_batch, extends_dataset_id="ncs-18-2026-06")
    result = donor.donate(req, ip_hash=ip_hash)
    assert result["mode"] == "dry-run"
    assert result["kind"] == "extension"
    assert result["extends"] == "ncs-18-2026-06"
    expected = tmp_path / "ncs-18-2026-06" / "papers" / f"{seeded_batch}.json"
    assert expected.is_file()
    contents = expected.read_text()
    assert "extension_of" in contents
    assert "ncs-18-2026-06" in contents


def test_donate_extension_refuses_missing_target_dataset(
    seeded_batch, pepper, monkeypatch, tmp_path
):
    """If the target dataset doesn't exist in the curated repo, the
    extension flow refuses rather than creating an orphan folder."""
    monkeypatch.delenv("PAPERLENS_DONATE_LIVE", raising=False)
    monkeypatch.setattr(donor, "DRY_RUN_BUNDLE_DIR", tmp_path)
    monkeypatch.setattr(
        donor.datasets_mod, "get_dataset_metadata",
        lambda did: None,
    )
    ip_hash = donor.hash_ip("9.9.9.9")
    req = _make_req(batch_id=seeded_batch, extends_dataset_id="bogus-dataset")
    with pytest.raises(ValueError, match="doesn't exist"):
        donor.donate(req, ip_hash=ip_hash)


@pytest.mark.skip(reason="Zenodo integration disabled — see web/zenodo.py")
def test_zenodo_metadata_includes_related_pr_when_provided():
    import zenodo as zenodo_mod
    body = zenodo_mod._metadata_block(
        title="Test dataset", description="A description",
        attribution_mode="anonymous", attribution_name="", attribution_affiliation="",
        dataset_id="test-2026-06",
        github_pr_url="https://github.com/x/y/pull/1",
    )["metadata"]
    assert body["title"] == "Test dataset"
    assert body["upload_type"] == "dataset"
    assert body["license"] == "cc-by-4.0"
    assert body["related_identifiers"][0]["identifier"] == "https://github.com/x/y/pull/1"
    assert body["related_identifiers"][0]["relation"] == "isSupplementTo"
