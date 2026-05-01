"""PDF processing utilities: image conversion, text extraction, and evidence highlighting."""

from __future__ import annotations

import base64
import json
import os
import re
from collections import defaultdict

# Set PAPERLENS_DEBUG_HL=1 to print rect-locator outcomes per snippet to
# stdout.  Useful when the user reports "no highlights" — read the server
# logs and see exactly which snippets failed to locate.
_DEBUG_HL = os.environ.get("PAPERLENS_DEBUG_HL") == "1"

# Pattern that recognises table identifiers such as "Table 2", "TABLE A1",
# "Appendix Table 3", "Tabelle 4" (German), "Tableau 4" (French),
# "Tabla 4" (Spanish), "Tabella 4" (Italian), "Tabela 4" (Portuguese).
# We also accept the model's normalisation back to "TABLE N" even when the
# underlying PDF is non-English.
_TABLE_WORDS = ("table", "tbl", "tabelle", "tableau", "tabla", "tabella", "tabela")
_TABLE_REF_RE = re.compile(
    r"\b(?:appendix\s+)?(?:" + "|".join(_TABLE_WORDS) + r")\.?\s*([A-Z]?\d+(?:\.\d+)?)\b",
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
    parsed object or None if it can't be parsed.

    Tolerant of: code fences, preamble before the JSON, trailing text after
    the JSON, and mid-output truncation (we close open strings/containers
    and retry).  This lets us still surface partial evidence/data even when
    the model gets cut off mid-output.
    """
    text = result_text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Locate the first plausible JSON container start.
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    candidate = text[min(starts):]

    # Try shrinking from the end — strips trailing prose.
    for end in range(len(candidate), 0, -1):
        if candidate[end - 1] not in "}]":
            continue
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            continue

    # Truncation repair: walk forward as a tiny JSON tokenizer.  Whenever we
    # observe a *complete value* (string, number/literal, or closed container),
    # remember the position + open-container stack.  At the end, truncate at
    # the last such position and append the missing closing brackets.
    stack: list[str] = []
    in_string = False
    escape = False
    expecting_value = False  # True at top level, after `:` in obj, or `[`/`,` in arr
    safe_cut = 0
    safe_stack: list[str] = []

    def mark_safe(pos: int) -> None:
        nonlocal safe_cut, safe_stack
        if stack:
            safe_cut = pos
            safe_stack = list(stack)

    i = 0
    n = len(candidate)
    while i < n:
        ch = candidate[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                if expecting_value:
                    mark_safe(i + 1)
                    expecting_value = False
            i += 1
            continue
        if ch in " \t\n\r":
            i += 1
            continue
        if ch == '"':
            in_string = True
            i += 1
            continue
        if ch == "{":
            stack.append("}")
            expecting_value = False  # next token is a key
            i += 1
            continue
        if ch == "[":
            stack.append("]")
            expecting_value = True
            i += 1
            continue
        if ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            mark_safe(i + 1)
            expecting_value = False
            i += 1
            continue
        if ch == ":":
            expecting_value = True
            i += 1
            continue
        if ch == ",":
            mark_safe(i)  # safe to truncate *before* a dangling comma
            expecting_value = stack[-1:] == ["]"]
            i += 1
            continue
        # Number / true / false / null literal — consume until a structural char
        j = i
        while j < n and candidate[j] not in '{}[],:" \t\n\r':
            j += 1
        if expecting_value:
            mark_safe(j)
            expecting_value = False
        i = j

    if not safe_stack:
        return None
    repaired = candidate[:safe_cut].rstrip().rstrip(",").rstrip()
    repaired += "".join(reversed(safe_stack))
    try:
        return json.loads(repaired)
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


_CAPTION_TEMPLATES = ("{w} {n}.", "{w} {n}:", "{w} {n} ", "{w}{n}.")
# Word forms covered (mixed case + uppercase variant for each).  Both English
# (Table) and the most common European-language equivalents are included so a
# German PDF with "Tabelle 4." still gets its caption located when the model
# emitted "TABLE 4." in evidence.
_CAPTION_WORDS = (
    "Table", "TABLE",
    "Tabelle", "TABELLE",
    "Tableau", "TABLEAU",
    "Tabla", "TABLA",
    "Tabella", "TABELLA",
    "Tabela", "TABELA",
)


def _snippet_candidates(norm: str):
    """Yield search candidates for ``norm`` in priority order, from most
    specific to most generic.

    Handles the common shapes that defeat exact ``page.search_for``:
      * leading numeric markers ("1. Mir ist..." → "Mir ist...")
      * trailing numeric loadings (".75" / "0.69" at end)
      * line-broken contiguous text (sentence and word-window slices)
    """
    seen: set[str] = set()

    def emit(s: str):
        s = s.strip(" \t\n.,;:")
        if len(s) >= 6 and s not in seen:
            seen.add(s)
            return s
        return None

    cand = emit(norm)
    if cand: yield cand

    # Strip leading "12. " / "12) " markers.
    stripped_lead = re.sub(r"^\s*\d+[.\)]\s+", "", norm)
    cand = emit(stripped_lead)
    if cand: yield cand

    # Strip trailing numeric loadings: " .75", " 0.69", " -.42"
    stripped_tail = re.sub(r"\s+-?\.?\d+(?:[.,]\d+)?\s*$", "", stripped_lead)
    cand = emit(stripped_tail)
    if cand: yield cand

    # Length-based truncations on the cleaned string
    base = stripped_tail or stripped_lead or norm
    for n in (180, 120, 80):
        if len(base) > n:
            cand = emit(base[:n])
            if cand: yield cand

    # Sentence-style splits, longest first
    sentences = sorted(
        (s.strip() for s in re.split(r"(?<=[.!?])\s+", base) if len(s.strip()) >= 20),
        key=len, reverse=True,
    )
    for s in sentences:
        cand = emit(s)
        if cand: yield cand

    # Sliding word window — try larger windows first so we get the most
    # context (and the most unique match) before falling back to short ones.
    words = base.split()
    for size in (10, 8, 6, 4):
        if size > len(words):
            continue
        for i in range(len(words) - size + 1):
            cand = emit(" ".join(words[i : i + size]))
            if cand: yield cand


def _find_table_caption(page, table_num: str):
    """Locate the caption rect for 'Table N' on the page, if present.

    The model often cites a table from body text ("...presented in Table 1.").
    Searching for the snippet alone anchors the highlight in the prose, not the
    table.  Looking up the caption directly lets us expand from the table's
    actual top edge.  Returns the first match's rect, or None.

    Multilingual: tries Table / Tabelle / Tableau / Tabla / Tabella / Tabela
    captions so a German or other-language PDF still resolves correctly when
    the model normalised the table name to "TABLE N" in evidence.
    """
    for word in _CAPTION_WORDS:
        for tpl in _CAPTION_TEMPLATES:
            rects = page.search_for(tpl.format(w=word, n=table_num))
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


def pdf_to_pages_with_rects(
    pdf_bytes: bytes,
    evidence_items: list[dict],
    dpi: int = DISPLAY_DPI,
) -> tuple[list[str], list[dict]]:
    """Render plain JPEG page images PLUS rect metadata for each evidence
    entry's location, so the client can overlay highlights selectively
    (filtered by the active sub-view).

    ``evidence_items`` is a list of dicts ``{page, snippet, field, source}``.
    ``page`` is 1-indexed.

    Returns:
        pages       — list of base64-encoded JPEG strings, one per PDF page
        highlights  — list of dicts with keys:
                        ``page``    1-indexed PDF page
                        ``snippet`` original snippet text (for tooltips)
                        ``field``   JSON path declared by the model (used to
                                    filter by sub-view), or None
                        ``source``  table/figure id or None
                        ``rects``   list of [x, y, width, height] in
                                    image-pixel coordinates at the
                                    requested DPI

    Memory-friendly alternative to ``pdf_to_highlighted_images``: we render
    each page only once (no per-sub-view duplication) and the client decides
    which rects to draw on top.
    """
    import fitz  # PyMuPDF
    _DEHY = getattr(fitz, "TEXT_DEHYPHENATE", 0)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[str]      = []
    highlights: list[dict] = []

    mat   = fitz.Matrix(dpi / 72, dpi / 72)
    scale = dpi / 72.0  # PDF points → image pixels

    # Group evidence items by page for one-pass rendering
    by_page: dict[int, list[dict]] = {}
    for ev in evidence_items:
        page = ev.get("page")
        if not isinstance(page, int) or page < 1:
            continue
        by_page.setdefault(page, []).append(ev)

    for page_idx in range(min(len(doc), MAX_PAGES)):
        page      = doc[page_idx]
        page_1idx = page_idx + 1

        for ev in by_page.get(page_1idx, []):
            snippet = ev.get("snippet") or ""
            if not snippet or len(snippet) < 5:
                continue
            norm         = _normalize_snippet(snippet)
            is_table_ref = bool(_TABLE_REF_RE.search(norm))

            def _search(text: str) -> list:
                # First pass: plain search.  Second pass: dehyphenated, which
                # also helps when the PDF text spans hyphenated line breaks
                # ("Hauptkomponenten-\nanalyse" → "Hauptkomponentenanalyse").
                rects = page.search_for(text) or []
                if not rects and _DEHY:
                    try:
                        rects = page.search_for(text, flags=_DEHY) or []
                    except Exception:
                        pass
                return rects

            rects: list = []
            matched_via: str | None = None
            for cand in _snippet_candidates(norm):
                rects = _search(cand)
                if rects:
                    matched_via = cand
                    break

            if _DEBUG_HL:
                tag = f"p{page_1idx} field={ev.get('field')!r}"
                if rects:
                    print(f"[hl] FOUND  {tag} via {matched_via[:60]!r} -> {len(rects)} rect(s)", flush=True)
                else:
                    print(f"[hl] MISS   {tag} snippet={norm[:80]!r}", flush=True)

            # Optional table-region expansion
            table_rects: list = []
            if is_table_ref:
                m = _TABLE_REF_RE.search(norm)
                if m:
                    caption = _find_table_caption(page, m.group(1))
                    if _DEBUG_HL:
                        print(f"[hl] table-ref t={m.group(1)!r} caption={'YES' if caption else 'no'} on p{page_1idx}", flush=True)
                    if caption:
                        expanded = _expand_to_table_region(page, [caption])
                        table_rects = expanded if expanded else [caption]

            if rects:
                if table_rects:
                    rects = list(rects) + list(table_rects)
                elif is_table_ref:
                    expanded = _expand_to_table_region(page, list(rects))
                    if expanded:
                        rects = expanded
            elif table_rects:
                rects = table_rects

            if rects:
                pixel_rects = [
                    [r.x0 * scale, r.y0 * scale,
                     (r.x1 - r.x0) * scale, (r.y1 - r.y0) * scale]
                    for r in rects
                ]
                highlights.append({
                    "page":    page_1idx,
                    "snippet": snippet,
                    "field":   ev.get("field"),
                    "source":  ev.get("source"),
                    "rects":   pixel_rects,
                })

        # Render this page WITHOUT highlight annotations
        pix       = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg", jpg_quality=85)
        pages.append(base64.b64encode(img_bytes).decode())

    if _DEBUG_HL:
        print(
            f"[hl] summary: {len(evidence_items)} evidence items -> "
            f"{len(highlights)} highlight rects, {len(pages)} pages rendered",
            flush=True,
        )

    doc.close()
    return pages, highlights


def evidence_items_from_result(result_text: str) -> list[dict]:
    """Walk the parsed JSON and return every evidence-like entry that has a
    snippet AND a usable page number, preserving ``field`` and ``source``.

    This is the structured input to ``pdf_to_pages_with_rects``.
    """
    parsed = _parse_result_json(result_text)
    if parsed is None:
        return []
    out: list[dict] = []

    def walk(obj):
        if isinstance(obj, dict):
            snip = obj.get("snippet")
            page = obj.get("page")
            if snip and page is not None:
                try:
                    page_num = int(float(page))
                    if page_num > 0:
                        out.append({
                            "page":    page_num,
                            "snippet": str(snip),
                            "field":   obj.get("field"),
                            "source":  obj.get("source"),
                        })
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(parsed)
    return out
