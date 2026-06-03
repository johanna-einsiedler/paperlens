# Contributing to metalens-datasets

This repo is fed by the [MetaPaperLens](https://paper.metalens.tech)
donate flow. Almost all PRs are opened by the donation bot — you
generally don't author PRs here directly.

## How a donation lands here

```
researcher extracts in MetaPaperLens
  → Download All
    → modal: title / attribution / visibility / consent
      → POST /api/donate
        → bot opens this repo's PR with one new folder under datasets/
        → bot creates a draft Zenodo deposit in parallel
maintainer reviews + merges
  → donor visits the Zenodo draft, clicks Publish → DOI minted
```

A maintainer review typically takes minutes — there's no code to
audit; the review is content-level.

## What the maintainer checks before merging

1. **Folder structure** — the PR adds exactly one new directory
   `datasets/<slug>-<yyyy-mm>/` containing the five expected files
   (`results.json`, `prompt.md`, `metadata.json`, `README.md`,
   `CITATION.cff`). Nothing outside that directory should change.

2. **No PDFs / no raw bytes** — `results.json` must contain only the
   keys whitelisted by the donor's schema strip:
   `paper_metadata, samples, summaries, records, studies, evidence,
   metric, notes, schema_version`. No base64 blobs, no `pageImages`.

3. **`metadata.json` is well-formed** — `schema_version` is a known
   value (`masem-v3`, `summarize-v1`, etc.); `donor.mode` is
   `anonymous` or `attributed`; `visibility.mode` is `public` or
   `gated`; if gated, a `password_hash` (bcrypt) is present.

4. **Consent record** — the PR body mentions which consent checkboxes
   the donor ticked. Sharing rights + CC-BY 4.0 license both required.

5. **No PII** — sample sizes and effect sizes are facts; donor name
   and affiliation appear only if the donor opted in to attribution.
   Other personally-identifying info should not appear anywhere.

If anything looks off, comment on the PR and ask the donor to revise
(the donate flow can re-submit; the bot opens a new PR with a fresh
folder).

## After merge: minting the DOI

The donor's Zenodo deposit is created as a **draft** during donation.
It doesn't appear in DOI resolvers until someone clicks **Publish** on
Zenodo. Two ways:

- **Donor self-publishes** — the donate-modal success message includes
  the draft URL. They open it, review, publish.
- **Maintainer publishes on merge** — visit the draft via the URL
  recorded in `metadata.json`, click Publish.

A follow-up enhancement is a GitHub Action in this repo that calls
`POST /api/deposit/depositions/{id}/actions/publish` automatically on
PR merge. Not built yet.

## Deposit format reference

### `results.json`

```json
{
  "schema_version": "masem-v3",
  "dataset_id":     "ncs-18-2026-06",
  "papers": [
    {
      "filename":        "smith2018.pdf",
      "model":           "gpt-5",
      "resolved_model":  "gpt-5-2025-09-15",
      "pages_processed": 12,
      "evidence_count":  47,
      "result": {
        "paper_metadata": {"title": "...", "doi": "10.1037/...", "year": 2018, "authors": [...]},
        "samples":        [...],
        "evidence":       [...],
        "schema_version": "masem-v3"
      }
    }
  ]
}
```

### `metadata.json`

```json
{
  "schema_version":  "metadata-v1",
  "dataset_id":      "ncs-18-2026-06",
  "title":           "Display title of the dataset",
  "description":     "Optional short summary",
  "donor":           {"mode": "attributed", "name": "...", "affiliation": "..."},
  "visibility":      "public" | "gated",
  "password_hash":   "$2b$12$...",                  // present iff gated
  "extraction": {
    "schema_version": "masem-v3",
    "paper_count":    1,
    "model_used":     ["gpt-5"],
    "prompt_sha256":  "abcdef…"
  },
  "created_at":      "2026-06-03T17:43:20Z"
}
```

### `prompt.md`

Plain markdown / text. The exact prompt body the model received,
verbatim. The `prompt_sha256` in `metadata.json` lets anyone verify the
file wasn't tampered with after deposit.

### `CITATION.cff`

Standard [Citation File Format](https://citation-file-format.github.io/)
— machine-readable for GitHub's "Cite this repository" UI and reference
managers (Zotero/Mendeley import via cff-to-bibtex).

## Schema versions

Schema versions are deliberately frozen — extensions to an existing
dataset must match. Current versions in use:

| Version | Source mode | Notes |
|---|---|---|
| `masem-v3` | MASEMiner Indirect (factor loadings + correlations) | NCS-18 example ships v3 |
| `masem-effect-sizes-v2` | MASEMiner Direct (records[]) | |
| `summarize-v1` | MetaPaperLens Summarise mode | |
| `forestplot-v1` | Forest-plot preset | |

If a donor extracts under a newer schema, that schema is treated as a
separate dataset family. Mixing versions in one folder is rejected by
the schema-strip step.

## Manual edits

If you absolutely need to edit a merged dataset by hand (e.g., to fix
a typo in `README.md`), open a small follow-up PR. **Don't edit
`results.json` directly** — it's the canonical scientific record. To
correct extractions, donate a new dataset via the MetaPaperLens UI.
