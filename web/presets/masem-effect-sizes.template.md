# TASK

Extract eligible effect sizes and metadata from a PDF reporting an empirical study.

Return EXACTLY ONE valid JSON object following the required schema.

# DOMAIN CONFIGURATION

## EFFECT SIZES

Canonical effect size labels:
- "r" = Correlation
- "or" = Odds ratios

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

# OUTPUT SCHEMA

{
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
- output is valid JSON parseable by json.loads.

Return:
- EXACTLY ONE top-level JSON object,
- JSON only,
- no markdown,
- no explanations,
- no comments,
- no code fences.
