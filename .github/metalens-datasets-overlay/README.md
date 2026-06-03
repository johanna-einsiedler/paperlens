# metalens-datasets

A curated, citable archive of research-paper extractions produced with
[MetaPaperLens](https://paper.metalens.tech) and its
[MASEMiner](https://paper.metalens.tech/maseminer) workflow.

Each folder under [`datasets/`](./datasets) is one community-donated dataset:
the structured JSON the model emitted, the exact prompt that produced it,
a machine-readable citation, and a Zenodo mirror for DOI minting.

---

## What this repo contains

Each dataset lives in its own folder under `datasets/<slug>-<yyyy-mm>/`
and ships **JSON only** — never the source PDFs:

```
datasets/
└── ncs-18-2026-06/
    ├── results.json    one entry per paper, in the schema declared below
    ├── prompt.md       the verbatim prompt that produced results.json
    ├── metadata.json   donor, dates, schema_version, optional password hash
    ├── README.md       human-readable summary + citation block
    └── CITATION.cff    machine-readable citation
```

`metadata.json` carries a `schema_version` field that fixes the exact
shape of `results.json` for that dataset. Extension submissions must
match it.

## How to use a dataset

```r
library(jsonlite)
data <- fromJSON("https://raw.githubusercontent.com/johanna-einsiedler/metalens-datasets/main/datasets/<slug>/results.json",
                 simplifyDataFrame = FALSE)
data$papers     # list of per-paper extractions
```

```python
import requests
data = requests.get(
    "https://raw.githubusercontent.com/johanna-einsiedler/metalens-datasets/main/datasets/<slug>/results.json"
).json()
for paper in data["papers"]:
    print(paper["filename"], paper["result"]["paper_metadata"])
```

The Zenodo DOI listed in each folder's `README.md` is the canonical
citation target — it pins a specific dataset version.

## Citing a dataset

Each merged dataset has a Zenodo deposit with a DOI of the form
`10.5281/zenodo.NNNNNN` (sandbox uses `10.5072/...`). Use the citation
block in the folder's `README.md` or `CITATION.cff`. Example:

> Smith J. (2026). *NCS-18 factor loadings across 50 samples*.
> MetaPaperLens dataset `ncs-18-2026-06`. https://doi.org/10.5281/zenodo.123456

## Donating a dataset

You don't open PRs by hand — the MetaPaperLens donate flow handles it.
Run an extraction at <https://paper.metalens.tech> (or locally), click
**Download All**, and the donate modal will offer to publish the
extraction. A bot opens a PR here, and a draft Zenodo deposit is
created in parallel.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for the deposit format and the
review process.

## What we accept / don't accept

| ✓ Accept | ✗ Reject |
|---|---|
| Structured extractions (sample sizes, statistics, effect sizes, factor loadings, etc.) | The source PDFs themselves |
| Per-paper metadata (title, DOI, year, authors) | Personally-identifying information about non-authors |
| Short verbatim quotes for verification (the `evidence` array) | Long verbatim passages that effectively reproduce the paper |
| Donations marked with a clear license (default CC-BY 4.0) | Content that the donor doesn't have lawful access to share |

The bot's schema strip enforces "no PDFs" automatically — the donor's
checkbox is for the lawful-access-to-source-papers aspect, where
software can't tell the difference.

## License

- **Data** in `datasets/` is published under **CC-BY 4.0** (each donor
  releases their dataset under this license — see consent record in the
  PR / `metadata.json`).
- **Scaffolding** (this README, CONTRIBUTING, schema templates) is
  released under the **MIT** license.

## Maintainer

Johanna Einsiedler — see <https://paper.metalens.tech>.
