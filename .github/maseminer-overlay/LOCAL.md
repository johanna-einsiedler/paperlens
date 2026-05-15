# MASEMiner — local quickstart

Self-contained tool for extracting factor loadings, inter-factor correlations, and study metadata from PDFs of primary psychometric studies. Runs entirely on your machine; your PDFs and API key never leave it.

## Prerequisites

- **Python 3.11 or newer** (or Docker — see below)
- **An LLM API key** for OpenAI / Google Gemini / DeepSeek, or a self-hosted vLLM endpoint
- **The PDFs** of the studies you want to mine

## Run it (Python)

```bash
git clone https://github.com/{{OWNER}}/maseminer.git
cd maseminer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
PAPERLENS_MASEMINER_ONLY=1 python server.py
```

Then open <http://localhost:5001> in your browser. The MASEMiner builder is the landing page.

On Windows PowerShell:
```powershell
$env:PAPERLENS_MASEMINER_ONLY = "1"
python server.py
```

## Run it (Docker)

```bash
docker build -t maseminer .
docker run --rm -p 8080:8080 maseminer
```

The `Dockerfile` already bakes in `PAPERLENS_MASEMINER_ONLY=1`. Open <http://localhost:8080>.

## What you get

Each batch produces, per PDF, a JSON object with:

- `samples[]` — one entry per analyzed sample/group containing
  - `sample_id`
  - `factor_loadings` (item × factor matrix as flat keys, e.g. `"F1.3": 0.62`)
  - `factor_correlations` (unique off-diagonal pairs, e.g. `"R1.2": 0.71`)
  - `pubyear`, `country`, `continent`, `lang`, `pubtype`, `n`, `female`, `age`, `clinical`, `res`, `nfac`, `cfa`, `met`, `rot`, `notes`
  - `extraction_confidence`: `{ factor_loadings, factor_correlations, metadata }` — each `"high"` / `"medium"` / `"low"`
- `evidence[]` — verbatim snippet + PDF page number + source identifier + JSON-path of the value the snippet supports

The same schema the hosted MetaPaperLens version emits — outputs are interchangeable.

## Citing the version you used

You are reading **MASEMiner version `{{VERSION}}`**.  The same string lives in the `VERSION` file at the repo root — cite that exact string for reproducibility.

```
Einsiedler, J. (2026). MASEMiner (version {{VERSION}}) [Computer software].
https://github.com/{{OWNER}}/maseminer
```

## Consuming the output in R

No R package yet — but the JSON is straightforward to flatten:

```r
library(jsonlite)
library(dplyr)

res <- fromJSON("path/to/batch_results.json", simplifyVector = FALSE)

# Pull one row per (paper, sample) with metadata flattened
samples <- lapply(res$papers, function(p) {
  lapply(p$result$samples, function(s) {
    data.frame(
      paper       = p$filename,
      sample_id   = s$sample_id %||% NA,
      n           = s$n %||% NA,
      nfac        = s$nfac %||% NA,
      lang        = s$lang %||% NA,
      country     = s$country %||% NA,
      cfa         = s$cfa %||% NA,
      met         = s$met %||% NA,
      rot         = s$rot %||% NA,
      conf_load   = s$extraction_confidence$factor_loadings   %||% NA,
      conf_corr   = s$extraction_confidence$factor_correlations %||% NA,
      conf_meta   = s$extraction_confidence$metadata          %||% NA,
      stringsAsFactors = FALSE
    )
  }) |> dplyr::bind_rows()
}) |> dplyr::bind_rows()

# Factor loadings + correlations live as named lists on each sample —
# build per-sample matrices from $factor_loadings and $factor_correlations
# and feed them to metaSEM / OpenMx / lavaan as usual.
```

A thin R wrapper around the local HTTP server is on the roadmap — open an issue if you want it.

## Troubleshooting

- **PyMuPDF install slow / failing** on Python 3.13+: pin to Python 3.11 or 3.12 where pre-built wheels exist.
- **Port already in use**: `python server.py --port 5002` (or set `$PORT`).
- **"Scanned PDF — vision only"** badge: that PDF has no text layer, the extraction will use the vision pipeline automatically (slower, slightly more expensive).
- **Email notifications** are off by default. Set `PAPERLENS_SMTP_HOST` and friends if you want completion emails — see the project README.

## Reporting issues

GitHub Issues on this repo. Please include:
- The `VERSION` string from your checkout
- A redacted snippet of the prompt + the LLM provider/model you used
- The error message or unexpected output
