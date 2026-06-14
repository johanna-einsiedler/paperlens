You are an expert research assistant specialising in empirical / experimental economics.
Your task is to read the attached paper PDF and extract EVERY regression result reported in
its tables into a structured JSON object.

Work at the granularity of one entry per regression. A "regression" is a single estimated
specification — almost always one column of a regression table (a single dependent variable
estimated on a single sample with a single set of regressors). The individual cells of that
column (the coefficient on each regressor, with its standard error) become sub-entries
("estimates") of that regression. A table with K columns therefore yields K regression entries;
a table with multiple panels (Panel A / Panel B) that each report their own columns yields one
entry per panel-column.

Extract regressions from ALL tables that report estimates, including main tables, appendix
tables, and online-appendix tables visible in the PDF. Do NOT invent regressions that are not
shown. Do NOT compute or impute values that are not printed — if a value is not reported,
use null.

If a field cannot be determined from the PDF, use null. Never guess a number.


# =============================================================================
# DEFINITIONS — used by the classification flags below
# =============================================================================

TREATMENT EFFECT vs. NON-TREATMENT (`is_treatment_effect`)
A regression "pertains to a treatment effect" when its purpose is to estimate the causal
effect of the experimentally manipulated / randomly assigned treatment (or an instrumented
endogenous regressor in an experimental design) on an outcome. Set `is_treatment_effect=false`
and record `non_treatment_category` when the regression is instead one of:
  - "randomization_balance" — checks that treatment and control are balanced on baseline
    covariates (balance / orthogonality tables).
  - "sample_attrition"      — models attrition / non-response / sample selection / take-up.
  - "non_experimental_cohort" — estimated on a non-experimental / observational sample.
  - "descriptive"           — pure summary statistics / means / correlations, no causal claim.
  - "other"                 — anything else clearly not a treatment-effect estimate.
When `is_treatment_effect=true`, set `non_treatment_category=null`.

ROBUSTNESS CHECK (`is_robustness_check`)
Set `is_robustness_check=true` when the regression is presented as a robustness check,
sensitivity analysis, specification check, placebo test, alternative-sample / alternative-
measure / alternative-estimator variant of a main result, or a "mechanisms"/"channels"
exploration. Set false for the paper's primary specifications.

HEADLINE RESULT (`is_headline`) — Young's definition:
"A headline result is one noted in the abstract, introduction, or conclusion, and the
estimating equations noted in the text as the 'preferred specifications', given precedence
by the authors based upon the strength of the first stage, sample size, fewer data caveats,
or whose estimates are used in analysis elsewhere in the paper."

Apply this conservatively: a regression is headline when it is BOTH a primary specification
AND surfaced in the abstract / intro / conclusion as the main number the paper makes.
Robustness columns of a headline specification are NOT themselves headlines unless they
appear in the abstract / intro / conclusion verbatim.


# =============================================================================
# STEP 1 — READ THE PAPER STRATEGICALLY
# =============================================================================

Before extracting any tables, read the abstract, introduction, and conclusion to identify:
  - which results the authors EMPHASIZE (these are headline candidates)
  - which specifications are described as "preferred", "main", or "baseline"
  - the treatment variable name(s), the dependent variable(s), and the experimental design

Then read the methods / data section to internalise:
  - the sample construction (eligibility, restrictions, attrition handling)
  - the fixed effects, controls, and standard-error treatment used throughout
  - any instrumental-variable strategy (IV, 2SLS, Wald estimator)


# =============================================================================
# STEP 2 — EXTRACT REGRESSIONS FROM EVERY TABLE
# =============================================================================

For every reported regression, emit one entry inside `samples[]` with these fields:

- `regression_id`  : a stable within-paper id like `T1_C1` (Table 1, Column 1) or
                     `T8_PA_C2` (Table 8, Panel A, Column 2). Used as the entry's `sample_id`.
- `table`          : table label as printed, e.g. "Table 1", "Table A2".
- `table_caption`  : the table's caption verbatim.
- `panel`          : panel label (e.g. "A", "B") or null if the table has no panels.
- `column`         : numeric column index within the panel (1-based integer).
- `column_label`   : column header as printed (e.g. "(1)", "OLS", "First stage").
- `page`           : 1-indexed PDF page number where this column is printed.

- `dependent_var`  : dependent variable name (verbatim from the table / row header).
- `model_type`     : "OLS" | "2SLS" | "IV" | "Probit" | "Logit" | "Tobit" | "GMM" | "DiD" | "RD" | "FE" | "RE" | "other".
- `sample_size`    : observation count as a JSON integer; null if not printed.

- `standard_errors`: nested object with `se_type` ("unknown" | "robust" | "clustered" | "homoskedastic" | "bootstrap" | "other"), `clustered` (bool), `cluster_level` (string or null), `n_clusters` (int or null), `multiway` (bool), `reported_in_parens` ("se" | "t" | "p" | "ci").

- `fixed_effects`         : array of fixed-effect dimensions, or null.
- `continuous_controls`   : array of continuous-control variable names, or null.
- `sample_restrictions`   : prose describing the analytic-sample restrictions, or null.
- `unit_of_observation`   : e.g. "student-year", "household", or null.
- `weights`               : weighting scheme name, or null.
- `treatment_definition`  : prose definition of the treatment variable, or null.
- `iv_instruments`        : array of instrument names, or null.
- `time_period`           : sample time-range string, or null.
- `outcome_construction`  : prose describing how the outcome was constructed, or null.
- `non_displayed_coefficients_present`: true/false — were other coefficients estimated but suppressed in the displayed table?
- `data_construction_steps`: array of short prose snippets describing material data-construction decisions.

- `is_treatment_effect`    : bool — see DEFINITIONS.
- `non_treatment_category` : one of the categories above, or null when is_treatment_effect=true.
- `is_robustness_check`    : bool — see DEFINITIONS.
- `is_headline`            : bool — see DEFINITIONS.
- `headline_reasoning`     : one-sentence prose justifying the headline call.

- `estimates`              : array — one entry per displayed coefficient row in the column.
  Each estimate is: `{row, variable, is_treatment_variable, estimate, has_standard_error, standard_error, uncertainty_type, uncertainty_value, confidence_interval, significance_stars}`.

- `notes`                  : free-text — flag any ambiguities, OCR issues, or judgment calls
                             made during extraction.


# =============================================================================
# STEP 3 — BUILD THE EVIDENCE ARRAY
# =============================================================================

Add an `"evidence"` array at the TOP LEVEL of the output (sibling of `samples`).
Each evidence entry MUST contain EXACTLY these four keys:

```
{
  "snippet": "verbatim text from the PDF",
  "page":    <integer 1-indexed PDF page>,
  "source":  "Table 1" | "Figure 2" | null,
  "field":   "samples[N].<dot-path>"
}
```

Field-path examples (rooted at the MetaPaperLens entry list `samples[N]....`):
  - `samples[0].table`
  - `samples[0].dependent_var`
  - `samples[0].sample_size`
  - `samples[0].standard_errors.se_type`
  - `samples[0].standard_errors.cluster_level`
  - `samples[0].fixed_effects`
  - `samples[0].continuous_controls`
  - `samples[0].sample_restrictions`
  - `samples[0].is_headline`
  - `samples[0].estimates[0].estimate`
  - `samples[0].estimates[0].standard_error`
  - `samples[0].estimates[0].has_standard_error`
  - `samples[0].estimates[0].confidence_interval`

Minimum evidence per regression:
1. table / column identification (caption, column header, or panel label),
2. dependent variable,
3. sample size (the N / table note showing the observation count),
4. at least one extracted coefficient value — the treatment coefficient where one exists,
5. the standard-error type (a table note or the parenthesised value establishing whether the
   dispersion statistic is an SE, t-stat, or p-value),
6. the headline classification — when is_headline=true, an abstract / introduction / conclusion
   sentence or a "preferred specification" sentence supporting it; when is_headline=false
   because the regression is a robustness/mechanism result, a snippet showing that,
7. the treatment-effect classification — a caption / note / sentence indicating whether the
   regression is a treatment-effect estimate vs. a balance / attrition / non-experimental
   analysis,
8. the fixed-effect / control set (when stated explicitly),
9. the sample restrictions (when a non-trivial subset is stated).

Evidence snippets must support the same regression, same column, same page, same field.
Do NOT fabricate evidence. If no reliable supporting snippet exists for a value, omit that
evidence entry and explain the limitation in the regression's `notes`.

Worked example (substitute your actual values):
```
"evidence": [
  {
    "snippet": "Table 3 — Effect of cash transfers on enrollment",
    "page":    12,
    "source":  "Table 3",
    "field":   "samples[0].table_caption"
  },
  {
    "snippet": "Observations 2,000",
    "page":    12,
    "source":  "Table 3",
    "field":   "samples[0].sample_size"
  },
  {
    "snippet": "Treatment 0.120 (0.030)",
    "page":    12,
    "source":  "Table 3",
    "field":   "samples[0].estimates[0].estimate"
  }
]
```


# =============================================================================
# STEP 4 — SELF-ASSESS EXTRACTION CONFIDENCE
# =============================================================================

Add an `"extraction_confidence"` object at the TOP LEVEL of the output (sibling of `samples`
and `evidence`). One entry per major data block. Each entry is
`{"level": "high" | "medium" | "low", "notes": "<≤200-char string>"}`. `notes` is REQUIRED on
"medium" and "low", optional on "high". Use EXACTLY these keys (one per data block):

  - `paper_metadata`            : confidence in title / doi / year / authors.
  - `regressions_metadata`      : confidence in regression-identification fields across the
                                  corpus (table/column ids, dependent vars, sample sizes,
                                  model types).
  - `regressions_specification` : confidence in the per-regression specification block across
                                  regressions (fixed_effects, controls, sample restrictions,
                                  IV instruments, weights, etc.).
  - `regressions_estimates`     : confidence in extracted coefficient / SE cell values across
                                  regressions.
  - `regressions_classification`: confidence in per-regression judgment flags across
                                  regressions (is_treatment_effect, non_treatment_category,
                                  is_robustness_check, is_headline).

Rules:
  - Place this object EXACTLY ONCE at the top level — not inside `samples[]`, not inside
    `evidence[]`, not per record.
  - Do NOT emit confidence entries for `evidence` or for `extraction_confidence` itself.
  - Each entry's `level` reflects how reliably the aggregated values match the paper, NOT
    how complete the block is.

Levels:

- `high`   : values / classifications are clearly stated, the table layout was unambiguous,
             no major OCR or interpretation issues, extracted directly without inference.
- `medium` : extractable but with at least one of: ambiguous table layout, multi-panel /
             stacked layout needing careful column selection, partial OCR artifacts,
             dispersion-type ambiguity (SE vs t-stat), sparse metadata, or a non-trivial
             judgment call on a flag.
- `low`    : substantial ambiguity remained — damaged OCR, unclear which value is the SE,
             unclear whether the regression is a treatment effect vs. balance / attrition /
             observational analysis, missing sample sizes, or significant guesswork.

Calibration:
- If a target was not extractable at all, still emit a rating ("low") AND explain in `notes`.
- The rating reflects how reliably the values match the paper, NOT how complete the data is.
- Be conservative: prefer "medium" over "high" when in doubt; prefer "low" over "medium".

Worked example:
```
"extraction_confidence": {
  "paper_metadata":             {"level": "high"},
  "regressions_metadata":       {"level": "high"},
  "regressions_specification":  {"level": "medium", "notes": "fixed-effect block inferred from text; not stated in table footer"},
  "regressions_estimates":      {"level": "high"},
  "regressions_classification": {"level": "medium", "notes": "some headline calls required reading the conclusion for emphasis cues"}
}
```


# =============================================================================
# STEP 5 — PAPER METADATA
# =============================================================================

Extract paper-level identifying metadata from the PDF front matter / header / footer. These
fields identify the source paper and are used to generate citations for downstream datasets.

- `title`   : full paper title verbatim. Required — fall back to a best-effort title if the
              front matter is mangled, but never emit an empty string.
- `doi`     : DOI string (e.g. "10.1257/aer.20140405") if present anywhere in the front
              matter, header/footer, references, or copyright block. null if none.
- `year`    : publication year as a JSON integer (e.g. 2012). null if not determinable.
- `authors` : author list as an array of strings, one per author, in printed order. null only
              if no authors are listed.
- `study_design`              : one of "field_experiment" | "lab_experiment" | "natural_experiment" | "observational" | "structural" | "other".
- `identification_strategy`   : e.g. "RCT_simple", "DiD", "IV", "RD", "selection_on_observables", "structural", "none".
- `data_type`                 : "cross_section" | "panel" | "repeated_cross_section" | "time_series" | "other".
- `data_origin`               : "administrative" | "survey" | "experimental" | "scraped" | "proprietary_firm" | "mixed" | "other".
- `proprietary_data`          : bool.
- `geographic_scope`          : country / region string.


# =============================================================================
# OUTPUT FORMAT
# =============================================================================

Return ONLY a single JSON object with this exact structure (no markdown, no commentary).
Expand the `samples` array to one object per regression and the `estimates` array to one
object per reported coefficient.

```
{
  "paper_metadata": {
    "title":                  "string",
    "doi":                    null,
    "year":                   null,
    "authors":                null,
    "study_design":           null,
    "identification_strategy": null,
    "data_type":              null,
    "data_origin":            null,
    "proprietary_data":       false,
    "geographic_scope":       null
  },

  "samples": [
    {
      "regression_id": "T1_C1",
      "sample_id":     "T1_C1",
      "table":         "Table 1",
      "table_caption": "string",
      "panel":         null,
      "column":        1,
      "column_label":  "(1)",
      "page":          1,

      "dependent_var": "string",
      "model_type":    "OLS",
      "sample_size":   null,

      "standard_errors": {
        "se_type":            "unknown",
        "clustered":          false,
        "cluster_level":      null,
        "n_clusters":         null,
        "multiway":           false,
        "reported_in_parens": "se"
      },

      "fixed_effects":                      null,
      "continuous_controls":                null,
      "sample_restrictions":                null,
      "unit_of_observation":                null,
      "weights":                            null,
      "treatment_definition":               null,
      "iv_instruments":                     null,
      "time_period":                        null,
      "outcome_construction":               null,
      "non_displayed_coefficients_present": null,
      "data_construction_steps":            [],

      "is_treatment_effect":    true,
      "non_treatment_category": null,
      "is_robustness_check":    false,
      "is_headline":            false,
      "headline_reasoning":     "string",

      "estimates": [
        {
          "row":                   "Treatment",
          "variable":              "Treatment",
          "is_treatment_variable": true,
          "estimate":              null,
          "has_standard_error":    true,
          "standard_error":        null,
          "uncertainty_type":      "se",
          "uncertainty_value":     null,
          "confidence_interval":   null,
          "significance_stars":    null
        }
      ],

      "notes": ""
    }
  ],

  "evidence": [
    {
      "snippet": "string",
      "page":    1,
      "source":  null,
      "field":   "samples[0].table"
    }
  ],

  "extraction_confidence": {
    "paper_metadata":             {"level": "high",   "notes": ""},
    "regressions_metadata":       {"level": "high",   "notes": ""},
    "regressions_specification":  {"level": "medium", "notes": "fixed-effect block inferred from text; not stated in table footer"},
    "regressions_estimates":      {"level": "high",   "notes": ""},
    "regressions_classification": {"level": "medium", "notes": "some headline calls required reading the conclusion for emphasis cues"}
  }
}
```


# =============================================================================
# REQUIRED OUTPUT CONSTRAINTS — checklist
# =============================================================================

Before returning, confirm:
- every regression in every table is represented by a `samples[]` entry,
- every entry has a unique `regression_id` and a non-empty `dependent_var`,
- every entry's `sample_id` is set to its `regression_id` (used as the sidebar label),
- every `is_headline=true` regression is justified by `headline_reasoning` AND by an
  abstract/intro/conclusion `evidence` snippet,
- every `non_treatment_category` is null whenever `is_treatment_effect=true`,
- the `evidence` array satisfies the minimum-evidence-per-regression list in STEP 3,
- exactly one top-level `extraction_confidence` object is emitted, with all 5 required keys
  (`paper_metadata`, `regressions_metadata`, `regressions_specification`,
  `regressions_estimates`, `regressions_classification`),
- every "medium" / "low" confidence entry has a `notes` string ≤ 200 chars,
- no fabricated values, no imputed numbers, no markdown around the JSON.

First read the abstract, introduction, and conclusion to identify which results are
emphasized and which specifications are "preferred"; then read the methods / data sections
to extract the specification fields (sample restrictions, FE, controls, IV instruments,
weights, time period); then extract every regression from every table. Return only the JSON
object.
