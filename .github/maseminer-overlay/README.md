# MASEMiner

Local extraction of factor loadings, inter-factor correlations, and study metadata from psychometric PDFs — for Meta-Analytic Structural Equation Modeling (MASEM).

Upload PDFs (or point the tool at a folder), describe your target scale, and get one JSON record per analyzed sample with:

- item-level factor loadings,
- unique off-diagonal factor correlations,
- coded study/sample metadata (n, language, country, EFA/CFA, rotation, etc.),
- verbatim evidence snippets with PDF page numbers,
- a self-assessed extraction-confidence rating per sample.

Your PDFs and API key never leave your computer — the LLM call is made from the local server directly to OpenAI / Google / DeepSeek (or your own vLLM endpoint).

## Quick start

See [LOCAL.md](LOCAL.md) — one page, four commands.

## Hosted version

A hosted version is available if you'd rather not run anything locally. See the project page.

## Citation

If you use MASEMiner in published work, please cite the specific version you ran:

```
Einsiedler, J. (2026). MASEMiner (version <see VERSION file>) [Computer software].
https://github.com/<owner>/maseminer
```

`cat VERSION` inside a checkout prints the release tag.

## License

MIT — see [LICENSE](LICENSE).
