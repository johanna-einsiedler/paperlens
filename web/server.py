"""
Flask backend for the Research Paper Processor.
Serves the static frontend and exposes three API endpoints:
  POST /api/generate-prompt  — AI-generated extraction/labeling prompt
  POST /api/extract          — PDF extraction via vision (OpenAI/Gemini) or text (DeepSeek)
  POST /api/pages            — PDF→display images for the review-only flow
"""

import openai
from flask import Flask, request, jsonify, send_from_directory

from pdf_utils import (
    EXTRACTION_DPI,
    extract_evidence_snippets,
    pdf_to_highlighted_images,
    pdf_to_images,
    pdf_to_markdown,
)
from prompt_builder import EVIDENCE_APPENDIX, build_meta_prompt
from providers import extract_with_images, extract_with_text, generate_text, get_provider

app = Flask(__name__, static_folder="static")


# ── Static serving ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ── API: generate prompt ──────────────────────────────────────────────────────

@app.route("/api/generate-prompt", methods=["POST"])
def generate_prompt():
    data     = request.get_json(silent=True) or {}
    api_key  = data.get("api_key", "").strip()
    model    = data.get("model", "gpt-4o-mini")
    mode     = data.get("mode", "extraction")
    question = data.get("question", "").strip()
    context  = data.get("context", "").strip()

    if not api_key:
        return jsonify({"error": "API key is required."}), 400
    if not question:
        return jsonify({"error": "Question is required."}), 400
    if mode not in ("extraction", "labeling"):
        return jsonify({"error": "Invalid mode."}), 400

    meta_prompt = build_meta_prompt(mode, question, context)

    try:
        generated = generate_text(model, api_key, meta_prompt, temperature=0.3)
        generated += EVIDENCE_APPENDIX
        return jsonify({"prompt": generated, "model_used": model})
    except openai.AuthenticationError:
        return jsonify({"error": "Invalid API key."}), 401
    except openai.RateLimitError:
        return jsonify({"error": "Rate limit exceeded. Please wait and try again."}), 429
    except openai.NotFoundError:
        return jsonify({"error": f"Model '{model}' not found or not available for your account."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: extract ──────────────────────────────────────────────────────────────

@app.route("/api/extract", methods=["POST"])
def extract():
    api_key  = request.form.get("api_key", "").strip()
    model    = request.form.get("model", "gpt-4o-mini")
    prompt   = request.form.get("prompt", "").strip()

    if not api_key:
        return jsonify({"error": "API key is required."}), 400
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    if "pdf" not in request.files:
        return jsonify({"error": "PDF file is required."}), 400

    pdf_file = request.files["pdf"]
    filename = pdf_file.filename or ""
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file (.pdf)."}), 400

    try:
        pdf_bytes = pdf_file.read()
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 400

    provider          = get_provider(model)
    use_text          = request.form.get("use_text_extraction", "0") == "1" or provider == "deepseek"
    pdf_kb            = len(pdf_bytes) // 1024

    try:
        if use_text:
            result, finish_reason, n, token_usage = _extract_text_path(
                model, api_key, prompt, pdf_bytes, filename, pdf_kb
            )
        else:
            result, finish_reason, n, token_usage = _extract_vision_path(
                model, api_key, prompt, pdf_bytes, filename, pdf_kb
            )
    except openai.AuthenticationError:
        return jsonify({"error": "Invalid API key."}), 401
    except openai.RateLimitError:
        return jsonify({"error": "Rate limit exceeded. Please wait and try again."}), 429
    except openai.BadRequestError as e:
        return jsonify({"error": f"Request rejected by OpenAI: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    print(
        f"[extract] {filename!r}: finish_reason={finish_reason!r}\n"
        f"[extract] {filename!r}: response={result[:200]!r}",
        flush=True,
    )

    snippets_by_page = extract_evidence_snippets(result)
    n_snippets = sum(len(v) for v in snippets_by_page.values())
    print(
        f"[extract] {filename!r}: evidence pages={sorted(snippets_by_page.keys())} "
        f"({n_snippets} snippet{'s' if n_snippets != 1 else ''} total)",
        flush=True,
    )

    display_images = pdf_to_highlighted_images(pdf_bytes, snippets_by_page)
    return jsonify({
        "result":          result,
        "finish_reason":   finish_reason,
        "pages_processed": n,
        "filename":        filename,
        "page_images":     [f"data:image/jpeg;base64,{b}" for b in display_images],
        "evidence_count":  n_snippets,
        "token_usage":     token_usage,
    })


def _extract_vision_path(
    model: str, api_key: str, prompt: str,
    pdf_bytes: bytes, filename: str, pdf_kb: int,
) -> tuple[str, str, int, dict]:
    """Convert PDF to images and run vision-based extraction."""
    extraction_images = pdf_to_images(pdf_bytes, dpi=EXTRACTION_DPI, fmt="png")
    if not extraction_images:
        raise ValueError("The PDF appears to be empty.")

    n = len(extraction_images)
    _log_extract(filename, model, n, pdf_kb, pdf_bytes, prompt)

    page_instruction = (
        f"\n\nThis document has been split into {n} page image{'s' if n != 1 else ''}. "
        "They are provided below in order, each labelled with its sequential PDF page number "
        "(1 = first page of the PDF, 2 = second page, etc.). "
        "IMPORTANT: when citing evidence, always use this sequential PDF page number — "
        "NOT any journal or book page number that may be printed in the header, footer, or margin of the page itself."
    )
    content = [{"type": "text", "text": prompt + page_instruction}]
    for i, b64 in enumerate(extraction_images):
        content.append({"type": "text", "text": f"PDF page {i + 1} of {n}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    result, finish_reason, token_usage = extract_with_images(
        model=model,
        api_key=api_key,
        content_blocks=content,
        extraction_images=extraction_images,
        prompt=prompt,
        page_instruction=page_instruction,
        n=n,
    )
    return result, finish_reason, n, token_usage


def _extract_text_path(
    model: str, api_key: str, prompt: str,
    pdf_bytes: bytes, filename: str, pdf_kb: int,
) -> tuple[str, str, int, dict]:
    """Extract PDF text layer and run text-based extraction."""
    markdown_text, n = pdf_to_markdown(pdf_bytes)
    if not markdown_text.strip():
        raise ValueError(
            "No text layer found in this PDF. Text extraction requires a native text PDF — "
            "scanned (image-only) PDFs cannot be processed via this method."
        )

    _log_extract(filename, model, n, pdf_kb, pdf_bytes, prompt)

    page_instruction = (
        f"\n\nThe document text has been extracted and split into {n} labelled page sections "
        "below. IMPORTANT: when citing evidence, use the PDF page number shown in the section "
        "headers (e.g. '--- PDF page 4 of 12 ---') — NOT any journal or book page number "
        "printed in the document itself."
    )

    result, finish_reason, token_usage = extract_with_text(
        model=model,
        api_key=api_key,
        markdown_text=markdown_text,
        prompt=prompt,
        page_instruction=page_instruction,
    )
    return result, finish_reason, n, token_usage


def _log_extract(
    filename: str, model: str, n: int, pdf_kb: int,
    pdf_bytes: bytes, prompt: str,
) -> None:
    import fitz  # noqa: PLC0415
    try:
        _doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        _sample = "".join(_doc[i].get_text() for i in range(min(3, len(_doc)))).strip()
        _doc.close()
        text_layer = "YES" if len(_sample) > 100 else "NO (likely scanned)"
    except Exception:
        text_layer = "unknown"
    print(
        f"[extract] {filename!r}: model={model} {n} pages, {pdf_kb} KB, text_layer={text_layer}\n"
        f"[extract] {filename!r}: prompt_hash={hash(prompt)} prompt_len={len(prompt)}",
        flush=True,
    )


# ── API: pages (review-only flow) ─────────────────────────────────────────────

@app.route("/api/pages", methods=["POST"])
def pages():
    """Convert a PDF to highlighted display images without running AI extraction.

    Used when loading a previously exported JSON for human review: the user can
    supply the original PDFs so page images appear with evidence highlights.

    Form fields:
        pdf    — the PDF file
        result — (optional) raw JSON result string used to locate evidence snippets
    """
    if "pdf" not in request.files:
        return jsonify({"error": "PDF file is required."}), 400

    pdf_file = request.files["pdf"]
    filename = pdf_file.filename or ""
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file (.pdf)."}), 400

    result_text = request.form.get("result", "")

    try:
        pdf_bytes = pdf_file.read()
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 400

    try:
        snippets_by_page = extract_evidence_snippets(result_text) if result_text else {}
        display_images   = pdf_to_highlighted_images(pdf_bytes, snippets_by_page)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    print(f"[pages] {filename!r}: {len(display_images)} page images rendered", flush=True)
    return jsonify({
        "filename":    filename,
        "page_images": [f"data:image/jpeg;base64,{b}" for b in display_images],
    })


if __name__ == "__main__":
    import os
    app.run(debug=True, port=int(os.environ.get("PORT", 5001)))
