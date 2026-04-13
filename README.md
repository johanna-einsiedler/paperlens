# PaperLens

AI-powered data extraction and labeling for academic papers.

Upload PDFs, describe what you need in plain language, and get structured JSON results — with highlighted source passages for every extracted value.

## What it does

- **Extract** — pull structured data (statistics, factor loadings, effect sizes, metadata) from academic PDFs into a defined JSON schema
- **Label** — classify papers by content using custom categories you define
- **Verify** — results are shown side-by-side with the source PDF, with yellow highlights marking the exact passages the model cited as evidence
- **Edit** — click any extracted value to correct it; edits are tracked as human overrides in the downloaded JSON
- **Review** — load previously exported results back into the viewer for continued human review

Supports OpenAI (GPT-4o), Google Gemini, and DeepSeek models.

## Local development

### Prerequisites

- Python 3.11+
- An API key for OpenAI, Google Gemini, or DeepSeek

### Setup

```bash
cd web
pip install -r requirements.txt
python server.py
```

Open `http://localhost:5001` in your browser.

### Project structure

```
pipeline/
└── web/                    # Flask web application (Railway root directory)
    ├── server.py           # Flask routes
    ├── pdf_utils.py        # PDF → images, text extraction, evidence highlighting
    ├── providers.py        # LLM provider routing (OpenAI / Gemini / DeepSeek)
    ├── prompt_builder.py   # Meta-prompt construction
    ├── requirements.txt
    ├── Procfile
    ├── prompts/            # Example prompts shown during prompt generation
    │   ├── 01_detection_prompt.txt
    │   ├── 02_extraction_prompt_factor_loadings.txt
    │   ├── 03_extraction_prompt_correlations.txt
    │   └── 04_extraction_prompt_metadata.txt
    └── static/
        ├── index.html
        ├── app.js
        └── style.css
```

## Deployment (Railway)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select the repo, then set **Root Directory** to `web/`
4. Railway auto-detects Python and runs the `Procfile`
5. Your app is live — no environment variables needed (users enter their own API keys in the UI)

The same steps work on [Render](https://render.com) — set the Root Directory to `web/` and the Start Command to `gunicorn server:app`.

## How it works

1. **Describe your task** — you write a plain-language description; the app uses an AI meta-prompt to generate a detailed extraction or labeling prompt
2. **Upload PDFs** — one or more papers are processed sequentially
3. **AI extracts / labels** — the model reads each PDF (as page images for vision models, or as extracted text for DeepSeek / text-extraction mode) and returns structured JSON
4. **Evidence highlighting** — extracted snippets are located in the PDF text layer and highlighted in yellow
5. **Human review** — view, edit, and download results with a full audit trail

## API keys

Your API key is entered in the browser and sent directly to the provider (OpenAI / Google / DeepSeek). It is never stored on the server.
