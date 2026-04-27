"""PDF processing utilities: image conversion, text extraction, and evidence highlighting."""

from __future__ import annotations

import base64
import json
import re
from collections import defaultdict

# Pattern that recognises table identifiers such as "Table 2", "TABLE A1", "Appendix Table 3"
_TABLE_REF_RE = re.compile(
    r"\b(?:appendix\s+)?(?:table|tbl\.?)\s*([A-Z]?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

MAX_PAGES      = 40
EXTRACTION_DPI = 200   # high-res PNG sent to vision models
DISPLAY_DPI    = 144   # JPEG resolution for the browser viewer


# ── Image conversion ──────────────────────────────────────────────────────────

def pdf_to_images(pdf_bytes: bytes, dpi: int = EXTRACTION_DPI, fmt: str = "png") -> list[str]:
    """Convert PDF bytes to a list of base64-encoded images, one per page.

    Args:
        dpi: Rendering resolution.
        fmt: "png" for lossless (vision model input) or "jpeg" for smaller display images.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for page_num in range(min(len(doc), MAX_PAGES)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg", jpg_quality=72) if fmt == "jpeg" else pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode())

    doc.close()
    return images


# ── Text extraction (DeepSeek / text-only path) ───────────────────────────────

def pdf_to_markdown(pdf_bytes: bytes, max_pages: int = MAX_PAGES) -> tuple[str, int]:
    """Extract the text layer of a PDF and return it as labelled markdown sections.

    Uses PyMuPDF's built-in text extraction.  Works well for native text PDFs;
    scanned / image-only PDFs will produce little or no output.

    Returns:
        (markdown_text, page_count)
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = min(len(doc), max_pages)
    pages = []

    for i in range(n):
        page = doc[i]
        # get_text("markdown") is available in PyMuPDF ≥ 1.24; fall back to plain text.
        try:
            text = page.get_text("markdown")
        except (TypeError, Exception):
            text = page.get_text("text")
        pages.append(f"--- PDF page {i + 1} of {n} ---\n{text.strip()}")

    doc.close()
    return "\n\n".join(pages), n


# ── Evidence parsing ──────────────────────────────────────────────────────────

def _parse_result_json(result_text: str):
    """Strip markdown fences and parse the model's JSON output.  Returns the
    parsed object or None if it can't be parsed."""
    text = result_text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_evidence_snippets(result_text: str) -> dict[int, list[str]]:
    """Parse a model JSON response and return {page_num: [snippets]}.

    Only includes entries that have BOTH a snippet AND a usable page number —
    these are the ones we can actually highlight in the PDF viewer.
    """
    parsed = _parse_result_json(result_text)
    if parsed is None:
        return {}

    snippets_by_page: dict[int, list[str]] = defaultdict(list)

    def recurse(obj: object) -> None:
        if isinstance(obj, dict):
            if "snippet" in obj and "page" in obj:
                snippet = obj.get("snippet") or ""
                raw_page = obj.get("page")
                if snippet and raw_page is not None:
                    try:
                        page_num = int(float(raw_page))
                        if page_num > 0:
                            snippets_by_page[page_num].append(str(snippet))
                    except (ValueError, TypeError):
                        pass
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(parsed)
    return dict(snippets_by_page)


def count_evidence_entries(result_text: str) -> int:
    """Count every evidence-like entry that has a non-empty snippet, regardless
    of whether it carries a usable page number.

    This lets the frontend tell apart 'no evidence at all' (prompt issue) from
    'evidence returned but no pages were specified' (model issue).
    """
    parsed = _parse_result_json(result_text)
    if parsed is None:
        return 0
    n = 0

    def walk(obj):
        nonlocal n
        if isinstance(obj, dict):
            if obj.get("snippet"):
                n += 1
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(parsed)
    return n


def _orphan_snippets(result_text: str) -> list[str]:
    """Snippets emitted by the model that lack a usable page number.

    These can sometimes be recovered by searching the PDF text for them —
    see ``recover_orphan_pages`` below.
    """
    parsed = _parse_result_json(result_text)
    if parsed is None:
        return []
    out: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            snip = obj.get("snippet")
            if snip:
                raw_page = obj.get("page")
                page_ok = False
                if raw_page is not None:
                    try:
                        page_ok = int(float(raw_page)) > 0
                    except (ValueError, TypeError):
                        page_ok = False
                if not page_ok:
                    out.append(str(snip))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(parsed)
    return out


def _locate_snippet_page(doc, snippet: str) -> int | None:
    """Try to find which 1-indexed page a snippet appears on.

    Tries the full normalized snippet first, then progressively shorter
    prefixes — same fallback chain as ``pdf_to_highlighted_images`` so the
    odds of finding *something* are high even when the model paraphrased a bit.
    Returns the first matching page, or None.
    """
    norm = _normalize_snippet(snippet)
    for query in (norm, norm[:120], norm[:80], norm[:50]):
        if not query:
            continue
        for page_num in range(min(len(doc), MAX_PAGES)):
            page = doc[page_num]
            try:
                if page.search_for(query):
                    return page_num + 1
            except Exception:
                continue
    return None


def recover_orphan_pages(
    result_text: str, pdf_bytes: bytes,
) -> dict[int, list[str]]:
    """For every evidence snippet the model returned WITHOUT a page number,
    search the PDF for the snippet and assign it to the page where it was
    found.  Returns the same {page: [snippets]} shape as
    ``extract_evidence_snippets`` for easy merging.
    """
    orphans = _orphan_snippets(result_text)
    if not orphans:
        return {}

    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: dict[int, list[str]] = defaultdict(list)
    try:
        for snippet in orphans:
            page = _locate_snippet_page(doc, snippet)
            if page is not None:
                out[page].append(snippet)
    finally:
        doc.close()
    return dict(out)


def merge_snippet_dicts(*dicts: dict[int, list[str]]) -> dict[int, list[str]]:
    """Combine multiple {page: [snippets]} dicts; preserves insertion order
    and de-dupes within each page."""
    merged: dict[int, list[str]] = defaultdict(list)
    for d in dicts:
        for page, snippets in d.items():
            for s in snippets:
                if s not in merged[page]:
                    merged[page].append(s)
    return dict(merged)


# ── Highlighted display images ────────────────────────────────────────────────

def _normalize_snippet(text: str) -> str:
    """Normalize model-quoted text for better PDF text-layer matching.

    Replaces common Unicode characters that differ between model output and the
    PDF text layer: curly quotes, em/en dashes, ligatures, non-breaking spaces.
    Also collapses whitespace.
    """
    replacements = {
        "\u2013": "-", "\u2014": "-",       # en-dash, em-dash
        "\u2018": "'", "\u2019": "'",        # curly single quotes
        "\u201c": '"', "\u201d": '"',        # curly double quotes
        "\ufb01": "fi", "\ufb02": "fl",      # fi, fl ligatures
        "\ufb03": "ffi", "\ufb04": "ffl",    # ffi, ffl ligatures
        "\u00a0": " ",                       # non-breaking space
        "\u00ad": "",                        # soft hyphen
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.split())


_CAPTION_PATTERNS = (
    "TABLE {n}.", "Table {n}.",
    "TABLE {n}:", "Table {n}:",
    "TABLE {n} ", "Table {n} ",
)


def _find_table_caption(page, table_num: str):
    """Locate the caption rect for 'Table N' on the page, if present.

    The model often cites a table from body text ("...presented in Table 1.").
    Searching for the snippet alone anchors the highlight in the prose, not the
    table.  Looking up the caption directly lets us expand from the table's
    actual top edge.  Returns the first match's rect, or None.
    """
    for pat in _CAPTION_PATTERNS:
        rects = page.search_for(pat.format(n=table_num))
        if rects:
            return rects[0]
    return None


def _expand_to_table_region(page, anchor_rects: list) -> list | None:
    """Given highlight rects for a table caption/header, try to expand them to
    cover the full table body.

    Strategy: find the bounding box of the anchor rects, then collect all text
    blocks below it (within the same column band) until we hit a gap > 1.5 ×
    average block height or a new section heading (all-caps short line).
    Returns a list of fitz.Rect covering the expanded region, or None if
    expansion is not possible.
    """
    import fitz  # PyMuPDF

    if not anchor_rects:
        return None

    # Union of anchor rects
    anchor_union = anchor_rects[0]
    for r in anchor_rects[1:]:
        anchor_union = anchor_union | r

    # Gather all text blocks on the page
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    if not blocks:
        return None

    # Filter to blocks whose top edge is at or below the anchor's bottom
    below = [b for b in blocks if b[1] >= anchor_union.y1 - 2]
    if not below:
        return None

    # Sort top-to-bottom
    below.sort(key=lambda b: b[1])

    # Walk blocks downward, stopping at a large vertical gap or section heading
    avg_height = sum(b[3] - b[1] for b in below[:8]) / max(len(below[:8]), 1)
    table_rects = list(anchor_rects)
    prev_bottom = anchor_union.y1

    for b in below:
        gap = b[1] - prev_bottom
        text = b[4].strip()
        # Stop on large gap (new section)
        if gap > max(avg_height * 2.5, 20):
            break
        # Stop if this looks like a new section heading: short, all-caps or starts "Note"
        words = text.split()
        if len(words) <= 6 and (text == text.upper() or text.startswith("Note")):
            break
        table_rects.append(fitz.Rect(b[0], b[1], b[2], b[3]))
        prev_bottom = b[3]

    if len(table_rects) <= len(anchor_rects):
        return None  # no expansion happened

    return table_rects


def pdf_to_highlighted_images(
    pdf_bytes: bytes,
    snippets_by_page: dict[int, list[str]],
    dpi: int = DISPLAY_DPI,
) -> list[str]:
    """Render display-resolution JPEG images with evidence snippets highlighted.

    Uses PyMuPDF's text search to locate each snippet on its page and draws a
    yellow highlight annotation over it.  When the snippet references a table
    (e.g. "Table 2"), attempts to expand the highlight to cover the full table body.
    Falls back gracefully when text cannot be found.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for page_num in range(min(len(doc), MAX_PAGES)):
        page = doc[page_num]
        page_1indexed = page_num + 1

        for snippet in snippets_by_page.get(page_1indexed, []):
            if len(snippet) < 5:
                continue
            norm = _normalize_snippet(snippet)
            is_table_ref = bool(_TABLE_REF_RE.search(norm))

            def _search(text: str) -> list:
                rects = page.search_for(text)
                if not rects:
                    try:
                        rects = page.search_for(text, flags=1)
                    except Exception:
                        pass
                return rects or []

            rects = (
                _search(norm)
                or _search(norm[:120])
                or _search(norm[:80])
            )
            if not rects:
                sentences = [
                    s.strip() for s in re.split(r"(?<=[.!?])\s+", norm)
                    if len(s.strip()) >= 20
                ]
                for sent in sentences:
                    rects = _search(sent)
                    if rects:
                        break
            if not rects:
                words = norm.split()
                for i in range(max(0, len(words) - 3)):
                    chunk = " ".join(words[i : i + 4])
                    rects = _search(chunk)
                    if rects:
                        break

            # If the snippet cites a table, try to highlight the full table
            # by locating its caption on the page (anchor of last resort).
            table_rects: list = []
            if is_table_ref:
                m = _TABLE_REF_RE.search(norm)
                if m:
                    caption = _find_table_caption(page, m.group(1))
                    if caption:
                        expanded = _expand_to_table_region(page, [caption])
                        table_rects = expanded if expanded else [caption]

            if rects:
                # Snippet text was found.  If we also found the table caption,
                # highlight both so the user sees the source quote AND the table
                # it refers to.  Otherwise expand around the snippet itself.
                if table_rects:
                    rects = list(rects) + list(table_rects)
                elif is_table_ref:
                    expanded = _expand_to_table_region(page, list(rects))
                    if expanded:
                        rects = expanded
            elif table_rects:
                # Snippet not found in the text, but we located the table by name.
                rects = table_rects

            if rects:
                try:
                    annot = page.add_highlight_annot(rects)
                    annot.update()
                except Exception:
                    pass

        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg", jpg_quality=85)
        images.append(base64.b64encode(img_bytes).decode())

    doc.close()
    return images
