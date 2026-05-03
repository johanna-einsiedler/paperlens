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


def test_pdf_to_pages_with_rects_returns_pages_and_rects(native_pdf_bytes):
    """The new renderer returns plain page images + rect metadata per
    evidence entry that could be located in the PDF text."""
    items = [
        {"page": 1, "snippet": "N = 147 undergraduate students participated.",
         "field": "samples[0].sample_size", "source": None},
        {"page": 3, "snippet": "Table 2",
         "field": "samples[0].factor_loadings", "source": "Table 2"},
        {"page": 1, "snippet": "definitely not in the PDF",
         "field": "samples[0].notes", "source": None},
    ]
    pages, highlights, scanned_pages = pdf_utils.pdf_to_pages_with_rects(native_pdf_bytes, items)
    # 3 pages in the fixture
    assert len(pages) == 3
    # The two locatable snippets get rect entries; the third is silently dropped
    assert len(highlights) >= 2
    # Each highlight carries the metadata the client needs to filter by sub-view
    for h in highlights:
        assert "page"    in h and isinstance(h["page"], int)
        assert "snippet" in h
        assert "field"   in h
        assert "rects"   in h and isinstance(h["rects"], list) and h["rects"]
        # Each rect is [x, y, width, height] in image-pixel coords (positive numbers)
        for r in h["rects"]:
            assert len(r) == 4
            assert all(isinstance(v, (int, float)) and v >= 0 for v in r)
    # scanned_pages is a list of 1-indexed pages with empty text layers — the
    # fixture has variable text density across pages, so we just check shape.
    assert isinstance(scanned_pages, list)
    assert all(isinstance(p, int) for p in scanned_pages)


def test_evidence_items_from_result_preserves_field_and_source():
    payload = """{"samples": [{
      "evidence": [
        {"snippet": "abc", "page": 2, "source": "Table 1", "field": "samples[0].x"},
        {"snippet": "no page",                "field": "samples[0].y"},
        {"snippet": "bad page", "page": "?",  "field": "samples[0].z"}
      ]
    }]}"""
    items = pdf_utils.evidence_items_from_result(payload)
    # Only the entry with a usable page number survives
    assert len(items) == 1
    assert items[0]["page"]   == 2
    assert items[0]["field"]  == "samples[0].x"
    assert items[0]["source"] == "Table 1"


def test_merge_snippet_dicts_dedupes_within_pages():
    a = {1: ["snippet A", "snippet B"]}
    b = {1: ["snippet B", "snippet C"], 2: ["snippet D"]}
    merged = pdf_utils.merge_snippet_dicts(a, b)
    assert merged[1] == ["snippet A", "snippet B", "snippet C"]
    assert merged[2] == ["snippet D"]


def test_probe_text_layer_classifies_text_pdf(native_pdf_bytes):
    """A native text PDF should report text_layer_present=True with no
    scanned pages."""
    p = pdf_utils.probe_text_layer(native_pdf_bytes)
    assert p["total_pages"] == 3
    assert p["total_text_chars"] > 0
    assert p["text_layer_present"] is True
    assert isinstance(p["scanned_pages"], list)


def test_probe_text_layer_classifies_image_only_pdf():
    """A PDF whose pages have no text layer should be flagged as
    image-only (text_layer_present=False, every page in scanned_pages)."""
    import fitz
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=595, height=842)   # blank pages — no text layer
    pdf_bytes = doc.tobytes()
    doc.close()

    p = pdf_utils.probe_text_layer(pdf_bytes)
    assert p["total_pages"] == 3
    assert p["text_layer_present"] is False
    assert p["scanned_pages"] == [1, 2, 3]


def test_probe_text_layer_classifies_mixed_pdf():
    """Half-scanned, half-text PDF: majority decides text_layer_present."""
    import fitz
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)  # text page
    p1.insert_text(
        (50, 50),
        "This page has plenty of text in its text layer to clear the "
        "50-character threshold for scanned-page detection.",
        fontsize=10,
    )
    doc.new_page(width=595, height=842)        # blank — counts as scanned
    p3 = doc.new_page(width=595, height=842)
    p3.insert_text(
        (50, 50),
        "Third page has its own substantial text layer well above threshold.",
        fontsize=10,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    p = pdf_utils.probe_text_layer(pdf_bytes)
    assert p["total_pages"] == 3
    assert p["scanned_pages"] == [2]
    # 1 of 3 scanned → still text-readable overall
    assert p["text_layer_present"] is True


def test_probe_text_layer_classifies_ocr_scan_as_scanned():
    """Real-world feedback case: scanned papers that have had OCR run on
    them (either by the user or by the journal before download) acquire
    a machine-generated text layer.  The old char-count heuristic
    treated those as native text PDFs and routed them through the
    text-extraction path, which gave poor / hallucinated results.

    The new check looks for a near-full-page image — the visual signal
    that the page is fundamentally a raster scan, regardless of whether
    a text layer has been overlaid on top."""
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # Insert a full-page image (a plain white pixmap is enough to make
    # ``page.get_image_info`` report a page-spanning bbox).
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 595, 842))
    pix.set_rect(pix.irect, (255, 255, 255))
    page.insert_image(page.rect, pixmap=pix)
    # Add an OCR-style text layer on top with plenty of characters so
    # the old char-count heuristic would have classified this as text.
    page.insert_text(
        (50, 50),
        "This page has a full-page raster image with an OCR-generated "
        "text layer overlaid on top — the typical shape of a scanned "
        "journal article that was OCR'd before download.",
        fontsize=10,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    p = pdf_utils.probe_text_layer(pdf_bytes)
    assert p["total_pages"] == 1
    # Even though the text layer has well over 50 chars, the page is
    # dominated by a full-page image — must be flagged as scanned.
    assert p["scanned_pages"] == [1]
    assert p["text_layer_present"] is False


def test_pdf_to_pages_with_rects_flags_scanned_pages():
    """A PDF page with no text layer (image-only / scanned) should be reported
    in ``scanned_pages`` so the client can show a clear notice instead of a
    confusingly-empty overlay."""
    import fitz
    doc = fitz.open()
    # Page 1: image-only (no text inserted, no images either — empty page
    # qualifies as 'no text layer')
    doc.new_page(width=595, height=842)
    # Page 2: native text — should NOT be flagged
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 50),
        "This page has a real text layer with plenty of words to clear the "
        "50-character threshold used for scanned-page detection.",
        fontsize=10,
    )
    pdf_bytes = doc.tobytes()
    doc.close()

    pages, highlights, scanned_pages = pdf_utils.pdf_to_pages_with_rects(pdf_bytes, [])
    assert len(pages) == 2
    assert 1 in scanned_pages
    assert 2 not in scanned_pages


def test_pdf_to_highlighted_images_table_ref_in_body_finds_caption(native_pdf_bytes):
    """When the snippet is a body-text mention of 'Table 2' on a page where the
    table caption also exists, the renderer should highlight without error and
    not crash even though the snippet text and caption are in different places."""
    images = pdf_utils.pdf_to_highlighted_images(
        native_pdf_bytes,
        {3: ["values are presented in Table 2."]},
    )
    assert len(images) == 3
