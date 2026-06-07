# TASK

Extract eligible effect sizes and metadata from a PDF reporting an empirical study.

Return EXACTLY ONE valid JSON object following the required schema.

# DOMAIN CONFIGURATION

## EFFECT SIZES

Canonical effect size labels:
${effect_sizes_block}

Extract ONLY these canonical effect sizes.

## VARIABLES

Canonical variable labels:
${variables_block}

Extract ONLY these canonical variables.

IGNORE all other variables.

Map all eligible measures to ONE canonical variable category.

Use ONLY canonical variable labels in:
- "var1"
- "var2"

Study-specific operationalizations belong in:
- "desc1"
- "desc2"

# EXTRACTION RULES

Extract ONLY:
- eligible variables,
- eligible effect sizes,
- zero-order associations,
- explicitly reported statistics.

Eligible statistics may appear in:
- tables,
- text,
- figures,
- supplements embedded in the PDF.

Correlation matrices should be interpreted using standard row-column table structure.

A reported matrix cell constitutes an explicitly reported effect size when:
- both variables are identifiable from the matrix labels,
- and the numeric value appears in the table.

Extract correlations from:
- upper triangles,
- lower triangles,
- multitrait-multimethod matrices,
when explicitly displayed.

Do NOT infer missing matrix cells from symmetry.

Extract only directly reported statistics.
Do NOT:
- infer,
- estimate,
- derive,
- impute,
- reconstruct,
- combine,
- transform statistics into alternative effect size metrics.

Ignore:
- adjusted effects,
- regression coefficients,
- SEM paths,
- mediation effects,
- multilevel estimates,
- partial correlations,
- model-based estimates.

Treat analytically distinct groups as separate samples when:
- independent cohorts,
- subgroup analyses,
- separate waves,
- separate samples,
are reported independently.

Do not create separate samples for:
- robustness checks,
- sensitivity analyses,
- repeated mentions of the same participants.

# FIELD EXTRACTION RULES

For each effect size extract:
- sample size,
- reliability,
- instrument name,
when explicitly reported.

Use null when unavailable.

## Reliability hierarchy

Use reliabilities reported for the analyzed sample only.

Preferred hierarchy:
1. alpha
2. omega
3. icc
4. kappa
5. other

If multiple reliabilities are reported:
- select the highest-priority type,
- report others in notes.

## Record alignment rules

- Reorder variables alphabetically only in the final JSON output.
- All fields ending in "1" correspond to "var1".
- All fields ending in "2" correspond to "var2".

## Numeric constraints

Numeric fields must contain JSON numbers only.

Do NOT use:
- strings,
- percentages,
- ranges,
- textual qualifiers.

Use null when unavailable.

# SAMPLE METADATA

Extract when explicitly available:
- publication year,
- country,
- continent,
- language of respondents,
- publication type,
- percentage female,
- mean age,
- clinical status.

Use null when unavailable.

## Publication type coding

- 1 = journal article
- 2 = book
- 3 = thesis/dissertation
- 4 = proceedings/presentation
- 5 = other

## Clinical coding

- 0 = nonclinical sample
- 1 = clinical sample
- 2 = mixed clinical/nonclinical sample

# EVIDENCE RULES

Each extracted sample/group must include evidence for:
- sample identification,
- effect size source,
- sample size.

Each evidence entry must contain EXACTLY:
- "snippet"
- "page"
- "field"

Evidence snippets must:
- quote verbatim PDF text,
- support the referenced field,
- correspond to the same sample/group,
- avoid paraphrasing or reconstruction.

# SELF-ASSESS EXTRACTION CONFIDENCE

For EACH extracted sample/group, return an ``extraction_confidence`` object with one rating per high-level extraction target.

Required keys (all MUST be present):

- ``effect_sizes``: confidence in the per-pair effect-size values (the ``records`` array) for this sample — including correct var1/var2 mapping and correct numeric values.
- ``reliabilities``: confidence in the reliability fields within records (rel1, rel2, rel1_type, rel2_type) for this sample.
- ``metadata``: confidence in the sample metadata block (pubyear, country, continent, lang, pubtype, female, age, clinical) for this sample.

Each rating MUST be one of EXACTLY these three strings (lower-case):

- ``"high"``: the relevant numeric values / metadata are clearly stated in the paper, the table layout was unambiguous, no major OCR or interpretation issues, and the values were extracted directly without inference.
- ``"medium"``: values were extractable but the source had at least one of: ambiguous variable labels requiring careful mapping to canonical short names, partial OCR artifacts, suppressed correlation values, sparse metadata, or a non-trivial reconciliation between matrix labels and prose names.
- ``"low"``: substantial ambiguity remained — e.g. heavily damaged OCR, conflicting matrices, missing or unclear variable identities, large fractions of unreported cells, or significant guesswork required.

Calibration:

- If a category was not extractable at all (no reliability reported, no metadata reported) — still emit a rating (``"low"``) AND explain in ``notes``.
- The confidence rating reflects how reliably the values match the paper, NOT how complete or theoretically pleasing the dataset is.
- Be conservative: prefer ``"medium"`` over ``"high"`` when in doubt; prefer ``"low"`` over ``"medium"`` when in doubt.

# PAPER METADATA

In addition to the per-sample blocks, extract paper-level identifying metadata from the PDF front matter / header / footer. These fields identify the source paper itself and are used to generate citations for downstream datasets.

- title:    the full paper title verbatim. Required — fall back to the best-effort title if the front matter is mangled, but never emit an empty string.
- doi:      the DOI string (e.g. "10.1037/abc.0000123") if present anywhere in the front matter, header/footer, references, or copyright block. Use null if no DOI is reported.
- year:     publication year as a JSON integer (e.g. 2021). Use null if you cannot determine the year.
- authors:  the author list as an array of strings, one per author, in the order printed (e.g. ["Smith J", "Jones K"]). Use null only if no authors are listed.

# OUTPUT SCHEMA

{
  "paper_metadata": {
    "title":   "string",
    "doi":     null,
    "year":    null,
    "authors": null
  },

  "samples": [
    {
      "sample_id": "string",

      "records": [
        {
          "var1": null,
          "var2": null,
          "desc1": null,
          "desc2": null,
          "es": null,
          "type": null,
          "n": null,
          "rel1": null,
          "rel2": null,
          "rel1_type": null,
          "rel2_type": null,
          "instr1": null,
          "instr2": null
        }
      ],

      "pubyear": null,
      "country": null,
      "continent": null,
      "lang": null,
      "pubtype": null,
      "female": null,
      "age": null,
      "clinical": null,

      "extraction_confidence": {
        "effect_sizes":  "medium",
        "reliabilities": "medium",
        "metadata":      "medium"
      },

      "notes": ""
    }
  ],

  "evidence": [
    {
      "snippet": "",
      "page": null,
      "field": ""
    }
  ]

}

# VALIDATION RULES

Before returning output, validate:
- all effect sizes are within valid numeric ranges,
- all evidence fields reference existing JSON paths,
- every sample contains at least one effect size,
- all required fields are present,
- every sample has an extraction_confidence object with all three required keys,
- paper_metadata.title is populated,
- output is valid JSON parseable by json.loads.

Return:
- EXACTLY ONE top-level JSON object,
- JSON only,
- no markdown,
- no explanations,
- no comments,
- no code fences.
