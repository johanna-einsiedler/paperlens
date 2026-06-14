# MetaPaperLens

AI-powered data extraction and review for academic papers — and a growing set of domain presets built on top of it.

Upload PDFs, describe what you want to extract through a structured form (or pick a pre-built workflow), and get JSON back with yellow highlights marking every passage the model cited as evidence. Review, correct, donate the result to a community dataset, or pick up an existing dataset and contribute additional papers to it.

## What it does

- **Extract** — pull structured data (factor loadings, effect sizes, regression results, study metadata, qualitative findings) from PDFs into a JSON schema you control.
- **Verify** — results sit side-by-side with the source PDF, with yellow highlights on the exact passages the model cited.
- **Edit** — click any value to correct it; edits are tracked as `human_overrides` in the downloaded JSON.
- **Review** — load previously exported results back into the viewer for continued human review (no new extraction).
- **Extend** — pick an existing community dataset and contribute more papers without restarting the schema.
- **Donate** — share your extractions as a community dataset (GitHub PR + optional Zenodo deposit).

Supports OpenAI (GPT-4o / GPT-5 family), Anthropic (Claude), Google (Gemini), DeepSeek, Mistral (Pixtral), and any vLLM/Ollama endpoint with an OpenAI-compatible API.

## Two ways to use it

| Mode | Where | Use it when |
|---|---|---|
| **Hosted MetaPaperLens** | <https://paperlens.fly.dev> (or your own deploy) | General PDF extraction; you want the full preset picker, the structured prompt designer, the donation flow. |
| **MASEMiner local** | clone `johanna-einsiedler/maseminer` and `python server.py` | Factor-loadings / inter-factor-correlations / study-metadata extraction for psychometric MASEM. Your PDFs and API key never leave your computer. |

The two are the same codebase. The hosted app serves both audiences; the local distribution is a curated MASEMiner-only build, auto-mirrored from this repo on every version tag.

## Presets shipped

A preset bundles a tuned prompt template + a domain-specific review UI (sub-tabs, confidence badges, evidence-key routing). Each lives in [`web/presets/`](web/presets) as a `.json` + `.template.md` pair.

| ID | What it extracts | Sub-tabs |
|---|---|---|
| `masem` | Direct-information MASEM — pairwise effect sizes per sample | Effect sizes · Descriptives |
| `masem-ncs18` | Indirect-information MASEM — factor loadings + factor correlations | Loadings · Correlations · Descriptives |
| `econ-headline` | Empirical-economics — every regression per paper, grouped by source table | Metadata · Specification · Estimates · Classification · Paper metadata |
| `ai-findings` | AI-and-labour research — every effect-size finding plus subtopic mapping | Effect size · Comparison · Classification · Paper metadata |
| `summarize` | Per-section paper summarisation (background / methods / findings / limitations) | one tab per section |
| `forestplot` | Forest-plot extraction (study × effect-size with CIs) | Effect sizes · Descriptives |

The renderer is preset-agnostic: anything declared in `sub_views` becomes a tab, anything in `confidence_keys` lights up a badge, anything in `evidence_keys` routes PDF highlights to the right tab.

## The structured prompt designer ("Create prompt")

For domains without a bundled preset, the **Create prompt** path on step 3 captures the extraction hierarchy through a structured form:

1. **Unit of analysis** — what one entry represents (sample, table, finding, regression…)
2. **Information chunks** — logical groupings of fields (one tab per chunk in the review UI)
3. **Paper-level metadata** — extracted once per paper
4. **Free-form context** — anything else

Submitting the form generates **both** a prompt (sent through `/api/generate-prompt`) **and** a custom preset descriptor (saved to your browser's `localStorage` under `paperlens.userPresets.v1`). The resulting extraction renders with sub-tabs and confidence badges — Tier 2 UX with zero JSON authoring.

User-built presets appear on step 1 under "Your workflows" alongside the bundled ones. Delete via the × on hover. Storage is browser-local only.

If you already have a prompt, **I have a prompt** path lets you paste it directly. A compliance info block at the top spells out the canonical evidence + extraction_confidence structures the prompt must include for the UI to render correctly.

## Local development

```bash
git clone https://github.com/johanna-einsiedler/paperlens.git
cd paperlens/web
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open <http://localhost:5001>. To run as **MASEMiner-only** (hides the generic chrome, redirects `/` to `/maseminer`):

```bash
PAPERLENS_MASEMINER_ONLY=1 python server.py
```

### Running tests

```bash
cd web
pip install -r requirements-dev.txt
pytest
```

The suite covers pure functions (PDF parsing, evidence detection, provider routing, preset rendering), the FastAPI route handlers (with LLM clients mocked), the donation pipeline, and the prompt-readiness gate. Currently **237 tests, ~13 s**.

## Architecture

For the long-form walk-through — modules, request flow, preset system, evidence highlighting, deployment specifics — see [`documentation.md`](documentation.md).

Quick orientation:

- **Backend**: FastAPI + uvicorn. Extraction runs in daemon threads launched per job; HTTP returns immediately with a `job_id` and the frontend polls.
- **Persistence**: SQLite (`paperlens.sqlite3`) for job status/result; page images live in process memory and are re-derivable from the original PDF.
- **Frontend**: vanilla JS, no build step. Everything in [`web/static/`](web/static).
- **Presets**: declared as `web/presets/<id>.json` + `web/presets/<id>.template.md` pairs. Auto-discovered on boot by [`web/presets_loader.py`](web/presets_loader.py); the `_SUB_VIEW_SPECS` map at the top of that file controls per-sub-tab filtering.
- **Prompt-readiness gate**: every prompt (bundled, generated, or pasted) is structurally checked by [`web/prompt_check.py`](web/prompt_check.py) for an `evidence` array + `extraction_confidence` object. Extraction blocks at `/api/extract` unless the gate passes or the user explicitly acknowledges in the warning modal.

## Deployment

The hosted app runs on [Fly.io](https://fly.io). Deploy config: [`web/fly.toml`](web/fly.toml). After first-time `fly launch`:

```bash
cd web
fly deploy
```

Persistent SQLite lives on a Fly volume mounted at `/data` (`PAPERLENS_DB_PATH`). [Render](https://render.com) also works — point root at `web/` and start with `uvicorn server:app --host 0.0.0.0 --port $PORT`.

## The MASEMiner public mirror

A curated MASEMiner-only build auto-publishes to a separate public repo on every version tag. Researchers who want to run MASEMiner locally fork the mirror, never the full MetaPaperLens one.

- **Public repo**: <https://github.com/johanna-einsiedler/maseminer>
- **Sync workflow**: [`.github/workflows/sync-maseminer.yml`](.github/workflows/sync-maseminer.yml) — fires on `v*` tag pushes; rsync-style allowlist; force-pushes a flat history.
- **Overlay files** (branding for the public build): [`.github/maseminer-overlay/`](.github/maseminer-overlay)

To cut a release: `git tag v0.X.Y && git push origin v0.X.Y`. The Action stages the public tree, runs a leak check (no `fly.toml`, no SQLite, no internal docs), force-pushes `main` + the matching tag on the public repo.

---

## Community datasets — `metalens-datasets`

MetaPaperLens has a built-in donation flow that publishes extractions as community datasets in a curated public repo plus an optional Zenodo deposit.

- **Repo**: <https://github.com/johanna-einsiedler/metalens-datasets>
- **Layout**: one folder per dataset under `datasets/<slug>-<yyyy-mm>/`, containing `results.json` + `prompt.md` + `metadata.json` + `README.md` + `CITATION.cff`.
- **License**: data is CC-BY 4.0; scaffolding is MIT.

### How donations work

After downloading a batch, click **Share this dataset →**. The donate modal captures:
- title, description
- attribution (anonymous or attributed name + affiliation; multi-author is supported)
- visibility (public or password-gated for **extension**, not for visibility — the data is always public)
- human-verification status (raw model output vs human-reviewed; recorded in `metadata.json`)
- CC-BY-4.0 + sharing-rights consents

The server (`web/donor.py`):
1. Strips non-publishable fields (no `pageImages`, no raw PDF bytes — JSON only).
2. Hashes the password with bcrypt if gated.
3. Builds the deposit folder in a temp dir.
4. Opens a GitHub PR via a scoped GitHub App installation (no PAT).
5. Optionally creates a Zenodo draft deposit; publishes after the maintainer merges the PR.
6. Writes a row to the SQLite `donations` table (IP-pepper-hashed for rate limiting; 3/day per IP).

Maintainer reviews the PR on GitHub. On merge, a GitHub Action inside `metalens-datasets` mints the Zenodo DOI and commits it back into the dataset's `README.md` and `CITATION.cff`.

### Extending an existing dataset

The **Extend an existing dataset** card on step 1 calls `GET /api/datasets` (cached 5 min server-side), lists every public dataset, and lets you pick one to add papers to. If gated, you enter the password — verified server-side by `POST /api/datasets/{id}/verify-password`; the bcrypt hash never leaves the server.

After extraction, **Add to dataset** opens a PR that adds `datasets/<slug>/papers/<batch_id>.json` (existing files untouched). For Zenodo-linked datasets, this also mints a new version DOI under the same concept DOI.

Schema compatibility is enforced: `schema_version` must match between the dataset and the new batch.

### Seed datasets

| Dataset | Source meta-analysis | Papers · samples · records |
|---|---|---|
| `bm-vg-pa-2026-06` | Marker, Gnambs, Appel (2022) "Exploring the myth of the chubby gamer" | 10 · 17 · 53 |
| (test) `test-2026-06` / `test2-2026-06` | smoke-test PRs | small |

---

## External research-project integrations

MetaPaperLens is increasingly used as the **review and curation layer** for external extraction pipelines. The contract is small: anything that emits the canonical `{papers: [{filename, entries, ...}]}` JSON shape loads in **Review existing results** and gets the full sub-tab UX (when a matching preset exists).

The pattern that has worked across three projects:

1. The external pipeline runs its own LLM extraction off-line (gemini, multi-stage, with project-specific prompts).
2. A consolidation step (often called `aggregate.py`) merges per-paper outputs into one MetaPaperLens-canonical `results.json` — grouping per the natural unit of analysis for that domain.
3. MetaPaperLens ships a domain preset that matches the canonical sub-objects emitted by the pipeline.

### `41_meta_econometrics/pipeline2` → `econ-headline` preset

- **What it extracts**: every regression result from empirical-economics papers, classified as headline / robustness / non-treatment.
- **Pipeline prompt**: [`pipeline2/headline_results/extraction_prompt.txt`](../41_meta_econometrics/pipeline2/headline_results/extraction_prompt.txt) (460 lines, with full definitions + per-regression confidence rating block).
- **Aggregator**: [`pipeline2/headline_results/aggregate.py`](../41_meta_econometrics/pipeline2/headline_results/aggregate.py) — groups regressions by source table; pivots each table's regressions into side-by-side columns under four `_table`-wrapped sub-blocks (Metadata / Specification / Estimates / Classification).
- **Output**: 5 papers → ~17 table-rows in the sidebar; each row renders 5 sub-tabs. Headline status is a Classification-tab cell next to peer columns.
- **Run**: `python3 pipeline2/headline_results/aggregate.py --out output/results.json --pretty`

### `43_ai_labor_dashboard/dashboard/data/results.json` → `ai-findings` preset

- **What it extracts**: every reported effect-size finding from AI-and-labour research papers, tagged with finding-type + subtopic, plus paper-level metadata and a topical summary.
- **Native shape**: per-paper `{paper_metadata, subtopics, findings, evidence, extraction_confidence}` under a `result` wrapper.
- **Transform**: one-shot Python script in this repo's chat history (carves each finding into `effect_size` / `comparison` / `classification` sub-objects, lifts evidence + confidence to paper level). Output renders 77 papers → 543 findings.
- **Smoke output**: `/tmp/ai-findings-results.json` (2.3 MB)

### What an integrator needs to know

- The MetaPaperLens loader contract (`papers[]` array, per-paper `filename` + `entries` + `original_model_response`) is documented in [`web/static/faq-prompt-structure.html`](web/static/faq-prompt-structure.html) and surfaced in the UI's "I have a prompt" path.
- Evidence field paths should be rooted at `samples[N]....` so the sub-view filter matches segments correctly.
- The five-category `extraction_confidence` block (one entry per data sub-object) drives the per-tab badges.
- A preset's `_SUB_VIEW_SPECS` entry maps a data-source name → tab id + `evidence_keys` + `confidence_keys`. Add one entry per new sub-tab in [`web/presets_loader.py`](web/presets_loader.py).

---

## How it works (one paragraph)

You either pick a pre-built workflow (preset), build one through the structured form, or paste your own prompt → the readiness gate verifies the prompt's structure → you upload one or more PDFs → each is processed against the chosen LLM (page images for vision models, extracted text for DeepSeek / text mode) → the response is parsed into per-paper records → evidence snippets are located in the PDF and rendered as yellow highlights → you review, edit, optionally donate, and download.

## API keys

Your API key is entered in the browser and sent directly to the provider — never stored on the server. The hosted app retains nothing about you beyond per-batch job state (auto-cleaned after seven days). For sensitive PDFs, the **local distribution** is the right answer: nothing leaves your machine except the LLM call you'd be making anyway.

## License

MIT (code) · CC-BY 4.0 (community datasets in `metalens-datasets`).
