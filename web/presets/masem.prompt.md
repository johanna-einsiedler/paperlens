You are extracting Toronto Alexithymia Scale (TAS-20, 20 items) factor-analytic data from an academic paper for Meta-Analytic Structural Equation Modeling (MASEM).

## Goal

For every distinct sample/group reported in the paper, extract:

1. **Factor loadings** — item-level standardised factor loadings for items 1–20, allowing up to FIVE factors (F1–F5).
2. **Factor correlations** — inter-factor (latent) correlations among up to 5 factors (the upper-triangle: 10 unique pairs).
3. **Study + sample metadata** — coded according to a fixed scheme so the records are usable in a downstream meta-analytic database.

Return EXACTLY ONE valid JSON object, parseable by `json.loads`. NO surrounding prose, NO markdown code fences, NO commentary.

## JSON schema

```json
{
  "samples": [
    {
      "sample_id": "string",

      "factor_loadings": {
        "F1.1":  number|null, "F1.2":  number|null, "F1.3":  number|null,
        "F1.4":  number|null, "F1.5":  number|null, "F1.6":  number|null,
        "F1.7":  number|null, "F1.8":  number|null, "F1.9":  number|null,
        "F1.10": number|null, "F1.11": number|null, "F1.12": number|null,
        "F1.13": number|null, "F1.14": number|null, "F1.15": number|null,
        "F1.16": number|null, "F1.17": number|null, "F1.18": number|null,
        "F1.19": number|null, "F1.20": number|null,

        "F2.1":  number|null, "...":   "(F2.2 ... F2.20)",
        "F3.1":  number|null, "...":   "(F3.2 ... F3.20)",
        "F4.1":  number|null, "...":   "(F4.2 ... F4.20)",
        "F5.1":  number|null, "...":   "(F5.2 ... F5.20)"
      },

      "factor_correlations": {
        "R1.2": number|null, "R1.3": number|null, "R1.4": number|null, "R1.5": number|null,
        "R2.3": number|null, "R2.4": number|null, "R2.5": number|null,
        "R3.4": number|null, "R3.5": number|null,
        "R4.5": number|null
      },

      "pubyear":   number|null,
      "country":   "string|null",
      "continent": "string|null",
      "lang":      "string|null",
      "pubtype":   "1|2|3|4|5|null",
      "n":         "integer|null",
      "sex":       "number 0..100 |null",
      "age":       "number|null",
      "clinical":  "0|1|2|null",
      "res":       "integer >= 5 |null",
      "nfac":      "integer 1..10 |null",
      "cfa":       "0|1|null",
      "met":       "1|2|3|4|null",
      "rot":       "1|2|null",

      "notes": "string"
    }
  ],
  "evidence": [
    {"snippet": "...", "page": 3, "source": "Table 1", "field": "samples[0].factor_loadings"}
  ]
}
```

You MUST include all 100 loading keys (`F1.1`…`F5.20`) and all 10 correlation keys (`R1.2`…`R4.5`) on every sample, even if the values are null.

## Multiple samples

If the paper reports separate solutions for separate groups (e.g. boys vs girls, clinical vs non-clinical, English vs French, Sample 1 vs Sample 2, age bands), output ONE object per visibly-separated group. Do NOT combine samples.

If the paper reports a "total" sample alongside subgroups, include each as its own entry.

If the grouping is unclear, output a single entry with `sample_id: "sample1"` and explain in `notes`.

`sample_id` should be a short identifier derived from visible labels: `"total"`, `"boys"`, `"girls"`, `"clinical"`, `"nonclinical"`, `"english"`, `"french"`, `"sample1"`, `"sample2"`, etc. If no label is visible, use `"sample1"`.

## Factor naming + item identification

If the paper clearly uses the standard TAS-20 three-factor structure, map:
- F1 = **DIF** (Difficulty Identifying Feelings)
- F2 = **DDF** (Difficulty Describing Feelings)
- F3 = **EOT** (Externally Oriented Thinking)

Synonyms map to the same factor index:
- "Difficulty identifying feelings" → DIF → F1
- "Difficulty describing feelings"  → DDF → F2
- "Externally oriented thinking"    → EOT → F3

If the paper uses different naming or ordering, follow the paper's definition. For 4- or 5-factor solutions, do NOT invent meanings for F4/F5 unless explicitly defined; keep them as Factor 4 / Factor 5 and explain ambiguity in `notes`.

Recognise TAS-20 items by item numbers 1–20 OR by recognisable item text fragments (e.g., "I am often confused about what emotion I am feeling…"). Use the item number as the canonical identifier (the `<item>` part of `F<factor>.<item>`).

## Factor loadings (`factor_loadings`)

Output keys for ALL items 1–20 across factors F1–F5 (`F1.1` through `F5.20`).

Fill values:
- Loading explicitly shown for an item-factor cell → record as a number (negative allowed).
- Cell suppressed/blank/below-threshold within the reported solution → `0.0` (not null — the cell exists, the value is just below threshold).
- Entire factor does NOT exist in the chosen solution (e.g. F4/F5 in a 3-factor model) → set ALL `F4.*` / `F5.*` to `null`.
- Table genuinely cut off / not visible → `null` AND explain in `notes`.
- Table reports only primary loadings (one factor per item) → enter the reported loading in that factor's key and set the other factors for that item to `0.0` (for factors that exist in the solution).

**Do not impute or estimate values.**

### Multi-level / spanning headers

Tables may use multi-level headers (super-headings spanning multiple columns, group labels above factor columns, factor names as subheaders). Read the LOWEST-LEVEL header row that directly labels the numeric loading columns. Treat super-headings as structural grouping only; do not misinterpret them as separate variables.

If the header structure is too ambiguous to recover column identities, extract row-wise as best as possible and explain the ambiguity in `notes`.

### CFA-only fallback for the standard 3-factor structure

If the results stem EXCLUSIVELY from a CFA with three factors AND the table doesn't explicitly label item→factor, you MAY assume the standard TAS-20 three-factor item assignment:
- F1 = items 1, 3, 6, 7, 9, 13, 14
- F2 = items 2, 4, 11, 12, 17
- F3 = items 5, 8, 10, 15, 16, 18, 19, 20

Use this ONLY to decide where to place a single reported loading per item when factor identity isn't otherwise recoverable. For CFA models with MORE or FEWER than three factors, do NOT assume any a priori item-to-factor assignment.

## Factor correlations (`factor_correlations`)

Output ALL 10 unique factor-pair correlations: `R1.2`, `R1.3`, `R1.4`, `R1.5`, `R2.3`, `R2.4`, `R2.5`, `R3.4`, `R3.5`, `R4.5`. Use the `R<i>.<j>` form with **i < j only** — do NOT duplicate symmetric pairs.

Look in the results section for keywords near each other: `correl*` (correlated/correlation), `factors`, numeric values in [-1, 1], possibly an `r` indicator. Correlations may appear as either (a) the lower/upper triangle with `nfac × (nfac-1) / 2` unique correlations, or (b) the full symmetric correlation matrix. Either form is fine — only output the i<j pairs.

Fill values:
- Correlation explicitly reported → number (negative allowed; typically in [-1, 1]).
- Correlation involves a factor that doesn't exist in this solution (e.g. R1.4 in a 3-factor model) → `null`.
- Correlation simply not reported / suppressed / not visible → `null`.
- ⚠️ **Orthogonal-rotation special case**: if the rotation method is one of `varimax`, `quartimax`, `equamax`, `orthomax`, `parsimax` (orthogonal — factors are uncorrelated by construction), set ALL existing-factor correlations to `0` (not `null`). Use `null` only if the value is missing/unreported, not when it's structurally zero.

## Multiple models / rotations / methods

If the paper reports MULTIPLE factor solutions for the same sample (EFA + CFA, rotated + unrotated, 2-factor + 3-factor, Model A + Model B):
- Extract the solution with the HIGHEST number of factors.
- Tied factor counts → prefer the one labelled main / final / preferred.
- Still tied → prefer EFA over CFA, and oblique over orthogonal rotation.
- Note the chosen solution and any alternatives in `notes`.

Ignore any solution that includes items from measures other than the TAS-20.

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
- **`res`** — Number of Likert response options for the TAS-20 (integer ≥ 5). Standard TAS-20 uses 5; if no adaptation/translation note exists, default to `5`. Else null.
- **`nfac`** — Number of factors in the chosen solution (integer 1–10). Must match the loadings/correlations you actually extracted (e.g. if all `F4.*` / `F5.*` are null, `nfac` is 3).
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
