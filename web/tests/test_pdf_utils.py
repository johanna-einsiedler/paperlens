"""Tests for pdf_utils.py — PDF rendering, text extraction, evidence parsing."""
from __future__ import annotations

import base64

import pdf_utils


# ── pdf_to_images ────────────────────────────────────────────────────────────

def test_pdf_to_images_returns_one_per_page(native_pdf_bytes):
    images = pdf_utils.pdf_to_images(native_pdf_bytes)
    assert len(images) == 3
    # Each entry is a base64 string that decodes to non-empty bytes
    for b64 in images:
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0


def test_pdf_to_images_jpeg_branch(native_pdf_bytes):
    images = pdf_utils.pdf_to_images(native_pdf_bytes, fmt="jpeg")
    assert len(images) == 3
    decoded = base64.b64decode(images[0])
    # JPEG starts with FF D8 FF
    assert decoded[:3] == b"\xff\xd8\xff"


def test_pdf_to_images_respects_max_pages(native_pdf_bytes, monkeypatch):
    # Force a tighter cap and re-render
    monkeypatch.setattr(pdf_utils, "MAX_PAGES", 2)
    images = pdf_utils.pdf_to_images(native_pdf_bytes)
    assert len(images) == 2


# ── pdf_to_markdown ──────────────────────────────────────────────────────────

def test_pdf_to_markdown_returns_text_and_count(native_pdf_bytes):
    text, n = pdf_utils.pdf_to_markdown(native_pdf_bytes)
    assert n == 3
    assert "147 undergraduate" in text
    assert "Table 2" in text
    # Page section markers from our wrapping
    assert "--- PDF page 1 of 3 ---" in text


def test_pdf_to_markdown_empty_pdf(empty_pdf_bytes):
    text, n = pdf_utils.pdf_to_markdown(empty_pdf_bytes)
    assert n == 1
    # No real content but the section header should still appear
    assert "--- PDF page 1 of 1 ---" in text


# ── _normalize_snippet ───────────────────────────────────────────────────────

def test_normalize_snippet_handles_curly_quotes_and_dashes():
    raw = "“Factor loadings” — see table"
    assert pdf_utils._normalize_snippet(raw) == '"Factor loadings" - see table'


def test_normalize_snippet_handles_ligatures():
    raw = "efﬁciency and ﬂow"
    assert pdf_utils._normalize_snippet(raw) == "efficiency and flow"


def test_normalize_snippet_collapses_whitespace():
    raw = "two   spaces\nand\ta tab"
    assert pdf_utils._normalize_snippet(raw) == "two spaces and a tab"


def test_normalize_snippet_strips_soft_hyphens():
    raw = "compli­cated"
    assert pdf_utils._normalize_snippet(raw) == "complicated"


# ── extract_evidence_snippets ────────────────────────────────────────────────

def test_extract_evidence_handles_fenced_json(evidence_json_payload):
    by_page = pdf_utils.extract_evidence_snippets(evidence_json_payload)
    assert by_page[1] == ["N = 147 undergraduate students participated"]
    assert by_page[3] == ["Table 2. Rotated factor matrix"]


def test_extract_evidence_handles_bare_json():
    payload = '{"evidence": [{"snippet": "abc", "page": 4, "source": null}]}'
    by_page = pdf_utils.extract_evidence_snippets(payload)
    assert by_page == {4: ["abc"]}


def test_extract_evidence_string_page_numbers():
    payload = '{"evidence": [{"snippet": "x", "page": "7"}]}'
    by_page = pdf_utils.extract_evidence_snippets(payload)
    assert by_page == {7: ["x"]}


def test_extract_evidence_invalid_json_returns_empty():
    assert pdf_utils.extract_evidence_snippets("not json at all") == {}


def test_extract_evidence_skips_invalid_pages():
    # Page 0 and negatives are filtered out
    payload = '{"evidence": [{"snippet": "a", "page": 0}, {"snippet": "b", "page": -1}, {"snippet": "c", "page": 2}]}'
    by_page = pdf_utils.extract_evidence_snippets(payload)
    assert by_page == {2: ["c"]}


def test_extract_evidence_recurses_into_nested_objects():
    payload = '{"results": {"sub": {"nested": [{"snippet": "deep", "page": 5}]}}}'
    by_page = pdf_utils.extract_evidence_snippets(payload)
    assert by_page == {5: ["deep"]}


# ── pdf_to_highlighted_images ────────────────────────────────────────────────

def test_pdf_to_highlighted_images_no_snippets_returns_one_per_page(native_pdf_bytes):
    images = pdf_utils.pdf_to_highlighted_images(native_pdf_bytes, {})
    assert len(images) == 3


def test_pdf_to_highlighted_images_unmatched_snippet_does_not_raise(native_pdf_bytes):
    # Snippet doesn't appear in the PDF — should still render all pages
    images = pdf_utils.pdf_to_highlighted_images(
        native_pdf_bytes,
        {1: ["this text is definitely not in the document"]},
    )
    assert len(images) == 3


def test_pdf_to_highlighted_images_matched_snippet(native_pdf_bytes):
    # Snippet matches page 1 text — should render without error
    images = pdf_utils.pdf_to_highlighted_images(
        native_pdf_bytes,
        {1: ["N = 147 undergraduate students"]},
    )
    assert len(images) == 3


# ── _find_table_caption ──────────────────────────────────────────────────────

def test_find_table_caption_locates_real_caption(native_pdf_bytes):
    """The fixture PDF has 'Table 2. Rotated factor matrix.' on page 3."""
    import fitz
    doc = fitz.open(stream=native_pdf_bytes, filetype="pdf")
    page = doc[2]  # page 3 (0-indexed)
    rect = pdf_utils._find_table_caption(page, "2")
    assert rect is not None, "should have found 'Table 2.' caption"
    doc.close()


def test_find_table_caption_returns_none_when_absent(native_pdf_bytes):
    import fitz
    doc = fitz.open(stream=native_pdf_bytes, filetype="pdf")
    page = doc[0]  # page 1 has no Table N
    assert pdf_utils._find_table_caption(page, "9") is None
    doc.close()


def test_count_evidence_entries_includes_orphans():
    """count_evidence_entries should include entries with snippet but no page."""
    payload = """{
      "evidence": [
        {"snippet": "with page",    "page": 2, "source": null, "field": "x"},
        {"snippet": "without page", "field": "y"},
        {"snippet": "also no page", "field": "z"}
      ]
    }"""
    assert pdf_utils.count_evidence_entries(payload) == 3


def test_count_evidence_entries_skips_empty_snippets():
    payload = '{"evidence": [{"snippet": "", "field": "x"}, {"snippet": "real", "field": "y"}]}'
    assert pdf_utils.count_evidence_entries(payload) == 1


def test_count_evidence_entries_returns_zero_on_invalid_json():
    assert pdf_utils.count_evidence_entries("not json at all") == 0


def test_orphan_snippets_extraction():
    payload = """{"evidence": [
      {"snippet": "has page", "page": 1},
      {"snippet": "no page"},
      {"snippet": "bad page", "page": "abc"},
      {"snippet": "zero page", "page": 0}
    ]}"""
    orphans = pdf_utils._orphan_snippets(payload)
    assert "no page"    in orphans
    assert "bad page"   in orphans
    assert "zero page"  in orphans
    assert "has page" not in orphans


def test_recover_orphan_pages_locates_snippets_in_pdf(native_pdf_bytes):
    """The fixture PDF page 1 contains 'N = 147 undergraduate students';
    recovery should figure out the page even though the model omitted it."""
    payload = """{"samples": [{
      "evidence": [
        {"snippet": "N = 147 undergraduate students participated.", "field": "samples[0].n"}
      ]
    }]}"""
    recovered = pdf_utils.recover_orphan_pages(payload, native_pdf_bytes)
    # Should have located the snippet on page 1 of the fixture PDF
    assert 1 in recovered
    assert any("147 undergraduate" in s for s in recovered[1])


def test_merge_snippet_dicts_dedupes_within_pages():
    a = {1: ["snippet A", "snippet B"]}
    b = {1: ["snippet B", "snippet C"], 2: ["snippet D"]}
    merged = pdf_utils.merge_snippet_dicts(a, b)
    assert merged[1] == ["snippet A", "snippet B", "snippet C"]
    assert merged[2] == ["snippet D"]


def test_pdf_to_highlighted_images_table_ref_in_body_finds_caption(native_pdf_bytes):
    """When the snippet is a body-text mention of 'Table 2' on a page where the
    table caption also exists, the renderer should highlight without error and
    not crash even though the snippet text and caption are in different places."""
    images = pdf_utils.pdf_to_highlighted_images(
        native_pdf_bytes,
        {3: ["values are presented in Table 2."]},
    )
    assert len(images) == 3
