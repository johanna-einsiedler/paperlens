"""Shared pytest fixtures.

PDFs are generated on the fly with PyMuPDF so we don't need to commit binary
fixtures.  Tests that need a real PDF use the byte-string fixtures below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure `web/` is on sys.path so tests can import `server`, `providers`, etc.
WEB_DIR = Path(__file__).resolve().parent.parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))


@pytest.fixture
def native_pdf_bytes() -> bytes:
    """A 3-page PDF with a real text layer."""
    import fitz
    doc = fitz.open()
    pages_text = [
        "Title: A Study of Things\nN = 147 undergraduate students participated.",
        "Methods\nA two-factor solution was retained.",
        "Table 2. Rotated factor matrix.\nItem 1  0.83  0.12\nItem 2  0.45  0.71",
    ]
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 100), text, fontsize=11)
    out = doc.tobytes()
    doc.close()
    return out


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """A PDF with a single blank page (no text layer worth speaking of)."""
    import fitz
    doc = fitz.open()
    doc.new_page()
    out = doc.tobytes()
    doc.close()
    return out


@pytest.fixture
def evidence_json_payload() -> str:
    """A typical extraction response from the model — includes evidence array."""
    return """```json
{
  "samples": [
    {
      "sample_id": "S1",
      "n": 147,
      "factor_loadings": {
        "item1": {"F1": 0.83, "F2": 0.12},
        "item2": {"F1": 0.45, "F2": 0.71}
      },
      "evidence": [
        {"snippet": "N = 147 undergraduate students participated", "page": 1, "source": null, "field": "sample identification"},
        {"snippet": "Table 2. Rotated factor matrix", "page": 3, "source": "Table 2", "field": "factor loadings"}
      ]
    }
  ]
}
```"""


@pytest.fixture
def labeling_json_payload() -> str:
    """A typical labeling response — flat object with label/rationale/confidence."""
    return """{
  "label": "Psychology",
  "rationale": "The paper's primary focus is on cognitive psychology.",
  "confidence": 0.95,
  "evidence": [
    {"snippet": "cognitive psychology of memory", "page": 1, "source": null, "field": "domain"}
  ]
}"""
