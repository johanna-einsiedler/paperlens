You are extracting ${instrument_name_long} ${preamble_data_qualifier}data from an academic paper for Meta-Analytic Structural Equation Modeling (MASEM).
${study_characteristics_block}
## Goal

For every distinct sample/group reported in the paper, extract:

${goal_items}

Return EXACTLY ONE valid JSON object, parseable by `json.loads`. NO surrounding prose, NO markdown code fences, NO commentary.

## JSON schema

```json
${json_schema}
```

${schema_invariants_line}

## Multiple samples

If the paper reports separate solutions for separate groups (e.g. boys vs girls, clinical vs non-clinical, English vs French, Sample 1 vs Sample 2, age bands), output ONE object per visibly-separated group. Do NOT combine samples.

If the paper reports a "total" sample alongside subgroups, include each as its own entry.

If the grouping is unclear, output a single entry with `sample_id: "sample1"` and explain in `notes`.

`sample_id` should be a short identifier derived from visible labels: `"total"`, `"boys"`, `"girls"`, `"clinical"`, `"nonclinical"`, `"english"`, `"french"`, `"sample1"`, `"sample2"`, etc. If no label is visible, use `"sample1"`.
${factor_naming_section}
${factor_loadings_section}
${factor_correlations_section}
${correlation_matrix_section}
${single_correlations_section}
${multiple_models_section}

${instrument_filter_line}

## Sample metadata (coded scheme)

Each sample's record carries the following fields. **Fields documented as integer codes MUST be integers — do not output strings for coded fields.**

- **`pubyear`** — Publication year (integer). From the paper or citation header. Null if not visible.
- **`country`** — Country of participants (open text). Only fill if explicitly stated or clearly inferable from recruitment location ("German university students in Berlin" → `"Germany"`). If origin is unclear but author affiliation is mainly United States, use `"USA"`. Else null.
- **`continent`** — Continent of participants (open text). If `country` is set, derive (e.g. Germany → `"Europe"`). If only continent is stated explicitly, fill that. Else null.
- **`lang`** — Language of the questionnaire instrument (open text), e.g. `"German"`, `"English"`, `"French"`. Use explicit statements like "German version", "translated into …". If multiple language versions exist, match the language to THIS record's `sample_id`. If the sample is from an English-speaking country with no explicit translation note, use `"English"`. Else null.
- **`pubtype`** —
  - `1` = article in a journal
  - `2` = book
  - `3` = thesis (Master's / PhD)
  - `4` = presentation / proceedings
  - `5` = other
  Null if uncertain (and explain in `notes`).
- **`n`** — Sample size (integer) corresponding to THIS record's sample. Usually in the abstract or method section ("the sample comprised…", "administered to…"). If multiple groups, match by `sample_id`. Null if not clearly matched.
- **`sex`** — Percentage of women in the sample, in [0, 100]. If only counts are given (e.g. "143 girls of 290"), compute `women / (women + men) × 100` and note the computation in `notes`. Null if not reported.
- **`age`** — Mean age in years (number). If reported separately per gender, compute the n-weighted mean. If only a range or median is given, set null and note. Prefer the reported mean.
- **`clinical`** —
  - `0` = non-clinical
  - `1` = clinical patients
  - `2` = mixed
  Null if unclear.
${res_field_block}
- **`nfac`** — Number of factors in the chosen solution (integer 1–10). Must match the loadings/correlations you actually extracted (e.g. if all ${nonexistent_factor_glob_example} are null, `nfac` is 3).
- **`cfa`** —
  - `0` = EFA (exploratory factor analysis)
  - `1` = CFA (confirmatory factor analysis)
  Search for the literal phrases. If both EFA and CFA are reported and you extracted the EFA solution, code `0`; mention the alternative method in `notes`.
- **`met`** — Extraction / estimation method:
  - `1` = principal component (PCA)
  - `2` = principal axis (PAF)
  - `3` = maximum likelihood (ML)
  - `4` = other (briefly name in `notes` — e.g. `"ULS"`, `"WLSMV"`, `"GLS"`, `"Bayesian"`)
  Null if not stated.
- **`rot`** — Rotation method:
  - `1` = orthogonal (varimax, quartimax, equamax, orthomax, parsimax)
  - `2` = oblique (promax, oblimin, quartimin, biquartimin, geomin)
  If all factor correlations are zero (orthogonal solution), use `1`. Null only if rotation is not meaningful (e.g. single-factor CFA).

## Notes field

Use `notes` for:
- Ambiguities ("table cut off at item 17"),
- Computed values ("% women computed from n=143/290"),
- The choice of solution when multiple exist,
- Factor-naming caveats ("factors labelled D1/D2/D3 in this paper, mapped to F1/F2/F3 by order of appearance"),
- The `pubtype` reasoning if uncertain,
- Any extra factors beyond F5 that exist but were truncated.

Anything worth flagging for human review during the meta-analytic coding stage.

## Strict output constraints

- Return JSON only — no markdown fences, no preamble, no commentary.
- Use `null` for missing/unreported values; use `0.0` only for suppressed cells in an existing factor.
- Coded fields (`pubtype`, `clinical`, `cfa`, `met`, `rot`) MUST be integers, not strings or null with descriptions.

---

## SUPPORTING EVIDENCE REQUIREMENT (mandatory)

Add an `evidence` array at the top level of the output. Each entry has exactly four keys:

- **`snippet`**: the EXACT verbatim text from the PDF that supports an extracted value. Quote character-for-character — do not paraphrase, summarise, or add ellipses. If you cannot quote a string verbatim from the PDF, find a different snippet.
- **`page`**: integer (1-indexed PDF page; **NOT** the journal/book page number printed in the document header/footer).
- **`source`**: the table or figure identifier (e.g. `"Table 2"`, `"Figure 1A"`), or `null` if from body text.
- **`field`**: a JSON path identifying which extracted value(s) this evidence supports — formatted exactly like the JSON structure you are emitting. Examples:
  - `"samples[0].factor_loadings"` — the whole loadings dict
  - `"samples[0].factor_loadings.F1.5"` — a specific loading cell
  - `"samples[0].factor_correlations"` — the full correlation matrix
  - `"samples[0].factor_correlations.R1.2"` — a specific correlation
  - `"samples[0].n"` — the sample-size field
  - `"samples[0].country"` — the country field
  - `"samples[1]"` — sample 2 as a whole

⚠️ ALL FOUR KEYS ARE MANDATORY ON EVERY EVIDENCE ENTRY. The `page` field is the most commonly forgotten — DO NOT omit it. Entries missing `page` cannot be linked to the source PDF and will be discarded by the post-processor. If you are uncertain which page a snippet appears on, count from page 1 and give your best estimate; never omit the field.

### Required coverage

For every numeric **table** you extract from (factor loadings + factor correlations), the evidence array MUST contain at least one entry whose `snippet` is the verbatim TABLE CAPTION (e.g. `"TABLE 1. Parameter estimates from confirmatory factor analyses of the TAS-20K..."`). The viewer relies on this to highlight the table region.

Also include:
- One snippet with the verbatim sentence identifying the sample (e.g. `"the sample comprised 147 non-clinical adolescents"`).
- One snippet stating the chosen model / factor count (e.g. `"A three-factor solution was retained"`).
- One snippet for each metadata field that's not implicit (especially `n`, `country`, `lang`, `age`, `sex`, `cfa`, `met`, `rot`).

### Evidence quality (good vs bad)

❌ BAD evidence (do NOT use):
- "The fit indices reached acceptable standards." — methodology, contains no extracted value
- "The parameter estimates are presented in Table 2." — reference to the source, not the source itself
- "Cronbach's α was 0.87." — about reliability, irrelevant to factor loadings

✅ GOOD evidence:
- `"TABLE 1. Parameter estimates from the results of confirmatory factor analyses..."` — table caption
- `"1   .539   .576   .488"` — literal row from the loadings table
- `"the sample comprised 147 non-clinical adolescents"` — sentence containing literal sample-size value
- `"A three-factor solution was retained"` — sentence stating the model
- `"the Korean version (TAS-20K)"` — language / instrument-version evidence
- `"varimax rotation"` — rotation-method evidence

---

Now perform the extraction on the supplied PDF. Return ONLY the JSON object — no preamble, no fences, no explanation.
