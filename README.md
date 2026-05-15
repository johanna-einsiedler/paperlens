# PaperLens

AI-powered data extraction and labelling for academic papers — and the MASEMiner workflow built on top of it.

Upload PDFs, describe what you want extracted in plain language, and get structured JSON back, with highlighted source passages for every value the model claims to have found.

## What it does

- **Extract** — pull structured data (statistics, factor loadings, effect sizes, metadata) from academic PDFs into a defined JSON schema.
- **Label** — classify papers by content using custom categories you define.
- **Verify** — results sit side-by-side with the source PDF, with yellow highlights marking the exact passages the model cited as evidence.
- **Edit** — click any extracted value to correct it; edits are tracked as human overrides in the downloaded JSON.
- **Review** — load previously exported results back into the viewer for continued human review.

Supports OpenAI (GPT-4o family), Google Gemini, DeepSeek, and any vLLM/Ollama endpoint with an OpenAI-compatible API.

## Two ways to use it

| Mode | Where | Use it when |
|---|---|---|
| **Hosted PaperLens** | <https://paperlens.fly.dev> (or your own deploy) | General PDF extraction; you want the full preset picker and the "describe your task → AI builds the prompt" flow. |
| **MASEMiner local** | clone `johanna-einsiedler/maseminer` and `python server.py` | Factor-loadings / inter-factor-correlations / study-metadata extraction for psychometric MASEM. Your PDFs and API key never leave your computer. |

The two are the same codebase. The hosted app serves both audiences; the local distribution is a curated MASEMiner-only build, auto-mirrored from this repo on every version tag.

## Local development (PaperLens, full app)

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

The suite covers the pure functions (PDF parsing, evidence detection, provider routing, preset rendering) and the FastAPI route handlers with LLM clients mocked. Currently 136 tests, ~5 s.

## Architecture

For the long-form walk-through — modules, request flow, preset system, MASEMiner builder, evidence highlighting, deployment specifics — see [`documentation.md`](documentation.md).

Quick orientation:

- **Backend**: FastAPI + uvicorn. Extraction runs in daemon threads launched per job; HTTP requests return immediately with a `job_id` and the frontend polls.
- **Persistence**: SQLite (`paperlens.sqlite3`) for job status/result; page images live in process memory and are re-derivable from the original PDF.
- **Frontend**: vanilla JS, no build step. Everything in [`web/static/`](web/static).
- **MASEMiner**: a preset in [`web/presets/`](web/presets) (template + JSON config) plus a guided form in [`web/static/masem-builder.js`](web/static/masem-builder.js). The backend is preset-agnostic.

## Deployment

The hosted app runs on [Fly.io](https://fly.io). The deploy config is [`web/fly.toml`](web/fly.toml). After first-time setup (`fly launch` once):

```bash
cd web
fly deploy
```

Persistent SQLite lives on a Fly volume mounted at `/data`; the path is set via the `PAPERLENS_DB_PATH` env var.

[Render](https://render.com) also works — point the root directory at `web/` and start command at `uvicorn server:app --host 0.0.0.0 --port $PORT`.

## The MASEMiner public mirror

A curated MASEMiner-only build is automatically published to a separate public repo on every version tag. Researchers who want to run MASEMiner locally fork that repo, never the full PaperLens one.

- **Public repo:** <https://github.com/johanna-einsiedler/maseminer>
- **Sync workflow:** [`.github/workflows/sync-maseminer.yml`](.github/workflows/sync-maseminer.yml) — fires on `v*` tag pushes; rsync-style allowlist; force-pushes a flat history to the mirror.
- **Overlay files** (replace branding for the public build): [`.github/maseminer-overlay/`](.github/maseminer-overlay)

To cut a release: `git tag v0.X.Y && git push origin v0.X.Y`. The Action stages the public tree, runs a leak check (no `fly.toml`, no SQLite, no internal docs), force-pushes `main` + the matching tag on the public repo, and substitutes `{{VERSION}}` / `{{OWNER}}` into the README and LOCAL.md so the citation block is concrete.

## How it works (one paragraph)

You write a plain-language description of what to extract → the app builds (or you pre-load) a JSON-emitting prompt → you upload one or more PDFs → each is processed sequentially against the chosen LLM (page images for vision models, extracted text for DeepSeek / text mode) → the response is parsed into structured per-paper records → evidence snippets are located in the PDF text layer and rendered as yellow highlights → you review, edit, and download.

## API keys

Your API key is entered in the browser and sent directly to the provider — never stored on the server. The hosted app holds nothing about you beyond per-batch job state (which is auto-cleaned after seven days). For sensitive PDFs, the **local distribution** is the right answer: nothing ever leaves your machine except the LLM call you'd be making anyway.

## License

MIT.
