# PaperLens — Setup & Architecture Documentation

PaperLens is a single-page web application for **AI-assisted structured-data extraction** from academic PDFs. It ships with a generic prompt-engineering flow and a brand-skinned **MASEMiner** workflow tailored to Meta-Analytic Structural Equation Modeling.

The whole stack runs as one FastAPI service plus a vanilla-JS frontend. There is no SPA framework, no backend rendering, no message queue — just HTTP routes, SQLite for job state, and daemon threads for in-process work.

---

## 1. What the tool does

A user uploads one or more PDFs, picks an LLM provider + model, describes (or pre-loads) an extraction prompt, and gets back **structured JSON per paper** with each value linked to a verbatim quote and rendered highlights on the source page.

Two activation paths:

- **Generic extraction**: user describes a task in plain language → an AI generates the prompt → user reviews → extraction runs.
- **MASEMiner preset** (`/maseminer`): branded entry point with a guided builder; pre-baked structural scaffolding for psychometric / structural-equation workflows; "TAS-20 example" sub-preset.

---

## 2. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (vanilla JS, no framework)                         │
│  - app.js : main state machine + accordion UI               │
│  - masem-builder.js : guided MASEMiner prompt builder       │
│  - index.html, style.css                                    │
└──────────────────┬──────────────────────────────────────────┘
                   │ JSON over HTTP
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI app (server.py) running under uvicorn              │
│  - 23 routes  (REST-ish)                                    │
│  - StaticFiles for /static                                  │
│  - per-request session header (X-Session-Id)                │
└────┬───────────────────────┬─────────────────┬──────────────┘
     │                       │                 │
     ▼                       ▼                 ▼
┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│ jobs.py     │  │ providers.py     │  │ pdf_utils.py        │
│ daemon-     │  │ OpenAI / Google  │  │ PyMuPDF rendering,  │
│ threaded    │  │ Mistral /        │  │ text-layer probe,   │
│ extraction  │  │ DeepSeek / vLLM  │  │ rect locator, OCR-  │
│ runner +    │  │ routing          │  │ scan detection      │
│ in-mem      │  └──────────────────┘  └─────────────────────┘
│ page cache  │
└──────┬──────┘
       │
       ▼
┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│ db.py       │  │ presets_loader   │  │ prompt_builder.py   │
│ SQLite WAL: │  │ JSON-driven      │  │ Meta-prompt for the │
│ jobs +      │  │ MASEMiner        │  │ "describe → AI      │
│ batches +   │  │ template render, │  │ writes prompt"      │
│ session_id  │  │ sub-views        │  │ generic flow        │
└─────────────┘  └──────────────────┘  └─────────────────────┘
```

State that survives a process restart lives in **SQLite** (`paperlens.sqlite3`, WAL mode). Page-image blobs and rect overlays live in **in-memory dicts keyed by job-id** (cleared on process restart — papers needing a re-render simply re-extract).

Background work uses **`threading.Thread(daemon=True)`** rather than `asyncio.create_task` because the LLM SDKs and PyMuPDF are sync; threads also survive past the HTTP request that submitted them, which `asyncio` tasks under FastAPI's `TestClient` do not.

---

## 3. Repository layout

```
pipeline/
├── README.md
└── web/
    ├── server.py              FastAPI app + all routes
    ├── jobs.py                In-process extraction runner (daemon threads)
    ├── db.py                  SQLite schema + queries (jobs, batches, sessions)
    ├── providers.py           LLM provider routing (5 providers)
    ├── pdf_utils.py           PyMuPDF: probe, render, rect locator, OCR-scan detect
    ├── presets_loader.py      JSON preset discovery + template renderer
    ├── prompt_builder.py      Meta-prompt for generic AI prompt generation
    ├── notifier.py            Email when batch finishes
    ├── presets/
    │   ├── masem.json         Umbrella MASEMiner preset (general MASEM)
    │   ├── masem-tas20.json   TAS-20 sub-preset (factor loadings + correlations)
    │   └── masem.template.md  Parameterised prompt template
    ├── static/
    │   ├── index.html         Single-page accordion UI
    │   ├── app.js             ~3 kLOC main state machine
    │   ├── masem-builder.js   Guided MASEMiner form
    │   └── style.css
    ├── tests/                 pytest suite (~130 tests)
    │   ├── conftest.py
    │   ├── test_routes.py
    │   ├── test_presets.py
    │   ├── test_pdf_utils.py
    │   ├── test_providers.py
    │   └── test_prompt_builder.py
    ├── requirements.txt
    ├── requirements-dev.txt
    ├── pyproject.toml
    ├── Dockerfile
    ├── fly.toml               Fly.io deployment config
    └── Procfile
```

---

## 4. Running locally

Prerequisites: Python 3.11+, `uv` (or plain `pip`).

```bash
cd web
uv venv && uv sync             # or: python -m venv .venv && pip install -r requirements.txt
./.venv/bin/python -m uvicorn server:app --reload --port 5001
```

Then open <http://localhost:5001>.

For the MASEMiner-branded entry: <http://localhost:5001/maseminer>.

Run the test suite:

```bash
cd web
./.venv/bin/python -m pytest tests/ -q
```

JS files have no build step — they are static and lint-checked with `node --check static/app.js static/masem-builder.js`.

---

## 5. Deployment (Fly.io)

`fly.toml` mounts a 1 GB volume at `/data` for the SQLite database. Single-machine deployment is intended (in-memory page-image caches and daemon threads only live on the host that received the extract request — multi-machine would require sticky sessions or moving the cache to Redis).

```bash
fly secrets set PAPERLENS_DB_PATH=/data/paperlens.sqlite3
fly volumes create paperlens_data --size 1 --region lhr
fly deploy
```

The Dockerfile ships uvicorn on port 8080 and serves both the API and the static frontend.

Cache headers on `/static/*` and `/` are `no-cache, no-store, must-revalidate` so JS/CSS updates always reach the browser on a normal reload (no hard-refresh needed after each deploy).

---

## 6. HTTP API surface

| Method | Path                                  | Purpose                                               |
|--------|---------------------------------------|-------------------------------------------------------|
| GET    | `/`                                   | Single-page app (index.html)                          |
| GET    | `/maseminer`                          | Same SPA, with MASEMiner hero                         |
| GET    | `/masemminer`                         | 301 → `/maseminer` (legacy URL)                       |
| GET    | `/api/config`                         | `max_batch_papers`, `max_pdf_bytes`                   |
| POST   | `/api/check-pdf`                      | Probe text layer (scanned-page detection)             |
| POST   | `/api/test-connection`                | One-token credential check                            |
| POST   | `/api/generate-prompt`                | Meta-prompt → generated extraction prompt             |
| POST   | `/api/adapt-prompt`                   | Add evidence-schema requirements to a prompt          |
| POST   | `/api/check-evidence-schema`          | Does this prompt request an `evidence` array?         |
| GET    | `/api/presets`                        | Landing list (hides `landing_hidden: true` variants)  |
| GET    | `/api/presets/{id}`                   | Full preset incl. rendered prompt                     |
| POST   | `/api/build-preset-prompt`            | Re-render MASEMiner template with form-supplied params|
| POST   | `/api/extract`                        | Submit an extraction job (returns `job_id`)           |
| GET    | `/api/jobs/{job_id}`                  | Poll job status / phase / result                      |
| GET    | `/api/jobs/{job_id}/pages`            | Page-image data-URIs + highlight rects                |
| POST   | `/api/jobs/{job_id}/cancel`           | Cooperative cancel between phases                     |
| GET    | `/api/batches`                        | History (filtered by `X-Session-Id`)                  |
| GET    | `/api/batches/{batch_id}`             | Single batch detail                                   |
| POST   | `/api/batches/{batch_id}/cancel`      | Cancel every pending paper in a batch                 |
| POST   | `/api/batches/{batch_id}/email`       | Set / update email for completion notification        |
| POST   | `/api/pages`                          | Render PDF pages on demand (used by review-from-file) |

All `/api/*` calls send an `X-Session-Id` header from the browser's `localStorage` UUID. `/api/batches` uses it to scope the history list per browser.

---

## 7. The extraction pipeline

1. **Upload-side probe** ([pdf_utils.py:`probe_text_layer`](web/pdf_utils.py)) runs as soon as a file is added in the browser. Returns `{total_pages, total_text_chars, scanned_pages, text_layer_present}`. The detector flags a page as scanned when (a) its text-layer character count is below 50 OR (b) a single image covers ≥ 70 % of the page area — this catches OCR'd scans where a text layer was generated post-hoc.

2. **Routing** ([app.js `processPaper`](web/static/app.js)). Default extraction mode is **text** (`useTextExtraction: true`). For PDFs the probe flagged as scanned (`text_layer_present === false`), the per-paper router automatically falls back to **vision** so the model reads the image directly. Provider-specific overrides:
   - DeepSeek is text-only (no fallback).
   - Mistral text models (`mistral-*`) are text-only; Pixtral models (`pixtral-*`) accept vision.

3. **Submit** ([server.py:`/api/extract`](web/server.py)). Accepts the PDF, prompt, model, API key, batch id (client-generated), notify-email. Spawns a daemon thread:
   ```python
   t = threading.Thread(target=_run_extraction, daemon=True, kwargs=…)
   t.start()
   ```
   Returns `job_id` immediately.

4. **Run** ([jobs.py:`_run_extraction`](web/jobs.py)). Phases tracked in SQLite (`update_phase`):
   - `Rendering pages` (vision path) or `Extracting text layer` (text path)
   - `Calling vision model` / `Calling text model`
   - `Highlighting evidence`
   - between every phase: `_check_cancelled` raises `_Cancelled` if the user clicked Stop.

5. **Highlight evidence** ([pdf_utils.py:`pdf_to_pages_with_rects`](web/pdf_utils.py)). For every snippet in the model's `evidence[]`, search the PDF text layer with progressively more lenient candidates (full snippet → leading-numeral stripped → trailing-loading-number stripped → truncated 180/120/80 chars → sentence splits → 4–10-word sliding windows → dehyphenated retry). Multilingual table-caption detection covers Table / Tabelle / Tableau / Tabla / Tabella / Tabela.

6. **Persist** to in-memory caches (`_PAGE_IMAGES`, `_PAGE_HIGHLIGHTS`, `_SCANNED_PAGES`) keyed by `job_id`. Result text + token usage + finish reason → SQLite.

7. **Poll** ([app.js `pollJob`](web/static/app.js)). Backoff 1 s → 1.3× per tick → 8 s cap, until `status` is `done` / `error` / `cancelled`.

8. **Render** ([app.js `displayPaper`](web/static/app.js)). Result column: parsed JSON via `parseEntries` → `renderValueHtml`, which auto-detects table shapes (dotted-numeric `F1.1` keys, `_table` markers, array-of-objects). Right column: page image + SVG overlay of highlight rects (filtered by active sub-view).

If the model returns un-parseable text, the result column shows the **fail-soft panel**: warning, raw response in a `<details>` collapsible, "Re-run extraction" + "Fill in manually" buttons. Clicking "Fill in manually" creates an empty schema scaffold the user can populate.

---

## 8. Provider routing

[providers.py](web/providers.py)'s `get_provider(model, base_url)` returns one of `openai` / `google` / `mistral` / `deepseek` / `vllm`:

| Prefix          | Provider | Wire format       | Vision? |
|-----------------|----------|-------------------|---------|
| `gpt-*`         | openai   | OpenAI native     | yes     |
| `gemini-*`      | google   | Google Gen AI SDK | yes     |
| `mistral-*`     | mistral  | OpenAI-compat     | no      |
| `pixtral-*`     | mistral  | OpenAI-compat     | yes     |
| `deepseek-*`    | deepseek | OpenAI-compat     | no      |
| any + base_url  | vllm     | OpenAI-compat     | varies  |

`extract_provider_message` digs the user-readable error out of `.body.error.message` (OpenAI-style) or `.message` (Gemini-style) so toast messages don't show the noisy `Error code: 401 - {"error": {…}}` blob.

DeepSeek is the only provider with auto-chunking for text mode (its 64 k context is smaller than OpenAI / Gemini / Mistral). The chunker re-runs the prompt per page-section and merges the resulting JSON arrays.

---

## 9. The MASEMiner preset system

A **preset** is a JSON file in `web/presets/`. The loader ([presets_loader.py](web/presets_loader.py)) is permissive — bad files are logged and skipped, so a malformed preset can't crash the server.

Three flavours of preset are supported, in priority order:

1. **Inline prompt**: `"prompt": "...full text..."` (simplest; no further processing).
2. **External prompt file**: `"prompt_file": "foo.prompt.md"` (raw inline-on-load).
3. **Parameterised template**: `"prompt_template_file": "masem.template.md"` + `"template_params": {…}` (rendered through `string.Template` with computed-block substitution).

The third path powers MASEMiner.

### Template rendering

`string.Template`-style `${variable}` placeholders (chosen over `str.format` to avoid escaping every `{` / `}` in the prompt's JSON examples).

Derived fields are computed from primitive params and substituted in:

| Param                      | Drives                                                                |
|----------------------------|-----------------------------------------------------------------------|
| `instrument_name`          | `${instrument_name}` everywhere ("TAS-20" / "the target constructs")  |
| `instrument_name_long`     | Preamble ("…extracting Toronto Alexithymia Scale (TAS-20, 20 items)…")|
| `n_items`, `n_factors`     | JSON-schema fragment dimensions, prose ranges, key counts             |
| `factor_naming` (list)     | `## Factor naming + item identification` body + synonyms block        |
| `cfa_item_assignment`      | CFA-only fallback bullet block                                        |
| `data_sources` (list)      | Which sections render: `factor_loadings`, `factor_correlations`, `correlation_matrix`, `single_correlations` |
| `content_scope`            | Instrument-filter sentence wording (concrete_items / content_groups / theoretical_constructs) |
| `variables` (list)         | "Variables / scales / constructs to look for" bullet list             |
| `item_texts` (list)        | Optional `## Reference item texts` block                              |
| `study_characteristics_text` | Optional `## About these studies` block (free-form context)         |

`_OPTIONAL_PLACEHOLDERS` collects every placeholder that's allowed to be empty, so missing data doesn't raise `KeyError`. The renderer's `_collapse_blank_lines` post-step squashes triple-newlines so skipped sections don't leave gaps.

### Sub-views are auto-generated

`presets_loader._build_sub_views_from_sources(data_sources)` produces the result-panel tabs from `template_params.data_sources`:

- `factor_loadings`     → "Factor loadings"     (`evidence_keys: ["factor_loadings"]`)
- `factor_correlations` → "Factor correlations" (`evidence_keys: ["factor_correlations"]`)
- `correlation_matrix`  → "Correlation matrix"  (`evidence_keys: ["correlation_matrix"]`)
- `single_correlations` → "Single correlations" (`evidence_keys: ["single_correlations"]`)

Plus a final "Descriptives" tab that excludes every data-source key (so it shows only metadata).

`evidence_keys` are stricter than `include_keys`: they scope the page-nav and rect-overlay filtering to the sub-view's primary domain, so e.g. clicking the "Factor loadings" tab won't pull pages that only carry sample-size evidence.

### Hidden variants

Presets that set `"landing_hidden": true` (currently `masem-tas20.json`) are filtered out of `GET /api/presets` so they don't clutter the landing-screen workflow picker. They remain fully loadable by id (the in-app builder posts to `/api/build-preset-prompt` with the variant id).

### Adding a new preset

1. Drop a JSON file into `web/presets/`. Required keys: `id`, `title`, `tagline`, `mode`. Plus one of `prompt` / `prompt_file` / `prompt_template_file`.
2. Optionally: `description`, `default_provider`, `default_model`, `task_description`, `context`, `accent_color`, `skip_to`, `sub_views` (auto-generated from `data_sources` if omitted), `landing_hidden`.
3. No restart needed — the loader runs per request.

---

## 10. The guided MASEMiner builder

[`masem-builder.js`](web/static/masem-builder.js) replaces the freeform "describe your task" textarea (step 3) with a structured form when a MASEMiner preset is active. Generic users keep the freeform path.

Form sections:

- **Starter** — two cards: `TAS-20 example` (default for the umbrella preset) and `Blank / General`. Picks the underlying variant id.
- **A: Direct correlation information** — toggle for `correlation_matrix` + `single_correlations` data sources.
- **B: Reconstructed information** — toggle for `factor_loadings` + `factor_correlations` data sources.
- **C: Identification information** — one entry per line. Parsed depending on the active data sources:
  - With B (factor loadings): treated as **item texts**, leading numbering stripped, fed into `template_params.item_texts` with `include_item_texts = true`.
  - Without B: treated as **variables** with optional `Name | Definition | synonym1, synonym2` syntax → `template_params.variables`.
- **D: Study & analysis characteristics** — free-form prose → `template_params.study_characteristics_text`.

Content scope is **inferred** from the active cards (B → `concrete_items`; A only → `theoretical_constructs`) so the prompt's filter sentence always matches what is actually being asked for.

### Auto-commit pipeline

Every preview render also writes the result into `state.generatedPrompt` and `state.activePreset.sub_views`. This means a user who proceeds through the flow without clicking "Use this prompt" still ends up with the right prompt — no orphaned default leaking through.

The initial render on starter switch is **immediate** (no debounce) so a fast user doesn't outrun the 350 ms timer used for textarea changes.

### TAS-20 copyright stance

The TAS-20 starter preset (`masem-tas20.json`) ships **only the structural scaffold** — factor naming (DIF / DDF / EOT), CFA item-to-factor mapping, instrument filter, factor-loading + factor-correlation sub-views — but **no verbatim item texts**. Item content is copyrighted (Bagby, Parker & Taylor, 1994); users paste their own copy into section C if they want the model to match items by semantic content rather than item number.

The renderer's fallback example (when no item texts are supplied) uses the placeholder `"a recognisable item phrase from the instrument"` instead of an actual TAS-20 item — so no copyrighted content can leak into the prompt regardless of how the preset is configured.

---

## 11. Frontend rendering & sub-views

[`renderValueHtml`](web/static/app.js) is the type-routed JSON-to-HTML core. It detects:

| Detector             | Renders as                                                     |
|----------------------|----------------------------------------------------------------|
| `isMarkedTable`      | Explicit `{"_table": [{...row...}]}` → real HTML table         |
| `isTableArray`       | Array of objects with ≥ 60 % shared keys → table               |
| `isTableMap`         | Object whose values are homogeneous objects → table            |
| `isDottedNumericTable` | Dict of `<group>.<index>` keys → dotted-key auto-table       |
| `isNumericObject`    | ≥ 4 numeric values → compact grid                              |
| `isLabelingResult`   | `{label, rationale}` → label badge + rationale prose           |

`isDottedNumericTable` accepts a small number of stray non-matching keys (≥ 75 % must match the dotted regex) so common model abbreviations like `"...": null` for empty factor columns don't flip the renderer to the flat fallback layout.

### Editable cells + click-to-jump

Every leaf value is a `<span class="rv-editable" contenteditable="true" data-path="…">`. Click → `handleCellEvidenceJump` reads the `data-path`, walks `paper.parsed.evidence` (longest-prefix match against the field), and navigates the right-hand PDF panel to that page. Edits are tracked in `paper.overrides[entryIndex]` keyed by path; downloads include both the original and the override.

### Sub-view filtering

When a sub-view is active (e.g. "Factor loadings" on a MASEMiner result):

- `_filterEntryBySubView` shows only the keys in `include_keys` / hides those in `exclude_keys`.
- `_evidencePagesForSubView` restricts the page navigator to pages whose evidence's `field` matches `evidence_keys` (segment-exact match, not substring).
- `_highlightMatchesSubView` does the same filtering for the SVG overlay rects.

---

## 12. Page navigation + scanned-page handling

The page navigator always spans **every page** in the PDF (1…N) — sub-views only choose which page is shown first. Users can flip backwards / forwards through the entire document; the counter tints to the primary colour when the current page carries any sub-view-matched highlights.

When the displayed page is in `paper.scannedPages`, an inline notice appears below the image: "This page looks like a scan (little or no text layer) — highlight rectangles may be incomplete or misaligned. The data extracted from this page is still shown on the left." This catches OCR'd scans where the post-hoc text layer doesn't faithfully reflect the visible content.

---

## 13. Re-run, manual mode, downloads

- **Re-run** ([app.js `retryPaper`](web/static/app.js)) wipes the result side (entries, parsed, highlights, evidencePages, overrides) but **keeps `pageImages`** so the right panel doesn't go blank during the new extraction. The "↻ Re-run" button on the result header reappears as soon as the new run finishes.
- **Manual mode**: when `parseEntries` returns null but the model produced text, the result column renders an "extraction-failed-panel" with a "Fill in manually" button that creates a typed empty scaffold (full MASEM schema for MASEMiner; `[{}]` for generic extractions). User edits feed straight into `paper.entries` and the existing override pipeline.
- **Downloads** ([app.js `_entriesFromPaper`](web/static/app.js)) annotate every CSV row / JSON entry with `_extraction_failed` and `_evidence_present` flags. Failed papers without manual fill-in still emit a stub row carrying `_llm_raw_response` so nothing is silently dropped. JSON downloads wrap the data in an envelope with `extraction_failed`, `evidence_present`, `entries`, `human_overrides`, and `original_model_response`.

---

## 14. Configuration & environment

| Env var                      | Default              | Purpose                                            |
|------------------------------|----------------------|----------------------------------------------------|
| `PAPERLENS_DB_PATH`          | `paperlens.sqlite3`  | SQLite path; on Fly mount this to `/data/...`      |
| `PAPERLENS_MAX_BATCH_PAPERS` | 20                   | Per-batch upload cap                               |
| `PAPERLENS_MAX_PDF_BYTES`    | 50 MB                | Per-file size cap                                  |
| `PAPERLENS_DEBUG_HL`         | unset                | When `1`: log per-snippet rect-locator outcomes    |
| `PAPERLENS_SMTP_*`           | unset                | Outbound SMTP for batch-completion email           |

---

## 15. Testing

Run the full pytest suite from `web/`:

```bash
./.venv/bin/python -m pytest tests/ -q
```

~130 tests, ~5 s. Coverage:

| File                      | What it covers                                                  |
|---------------------------|-----------------------------------------------------------------|
| `test_routes.py`          | All HTTP routes (extract, jobs, presets, batches, check-pdf)    |
| `test_presets.py`         | Loader (path-traversal guard, malformed-file recovery), all four MASEM presets, template rendering, sub-views, build-preset-prompt route |
| `test_pdf_utils.py`       | `probe_text_layer` (text vs scanned vs OCR'd-scan), rect locator, `_normalize_snippet`, evidence-snippet extraction |
| `test_providers.py`       | `get_provider` routing for all five providers, `_openai_compat_client` URL normalisation |
| `test_prompt_builder.py`  | Meta-prompt construction + evidence-appendix injection          |

JS files have no test framework wired up; lint-check with `node --check`.

---

## 16. Known constraints

- **Single-machine only** in production: in-memory page caches and daemon threads don't survive across hosts. Multi-machine deployment would require sticky sessions or moving caches to Redis.
- **PyMuPDF only**: scanned PDFs without a text layer go through vision extraction (which works), but the rect locator can't find snippets — the inline scanned-page notice tells the user their highlights may be incomplete.
- **No Tesseract OCR** integration: if a PDF is fully scanned with no text layer at all, vision extraction still works but the highlight overlays can't render.
- **MASEMiner template is `string.Template`-based**, not Jinja2 — sufficient for the current shapes but doesn't support loops or conditionals in the template itself; conditional sections are pre-built in Python and substituted as opaque blocks.
- **Browser autofill** for the API-key field can interfere with the per-provider credential cache; the input has `type="password" autocomplete="off"` to mitigate. The credential cache also runs a self-healing pass on session restore that wipes any cached key whose shape clearly belongs to a different provider (e.g. an `AIza…` Gemini key cached under the OpenAI slot).
