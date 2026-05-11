You are extracting **${scale_name}** factor-analytic data from an academic paper for Meta-Analytic Structural Equation Modeling (MASEM).
${study_characteristics_block}
## Goal

For every distinct sample/group reported in the paper, extract:

1. **Factor loadings** — item-level standardized factor loadings for items 1–**${n_items}**, allowing up to **${n_factors_max}** factors.
2. **Factor correlations** — inter-factor correlations among up to **${n_factors_max}** factors (upper/lower triangle: **${n_factors_max} × (${n_factors_max} - 1) / 2 = ${n_factors_pairs}** unique pairs).
3. **Study + sample metadata** — coded according to a fixed scheme so the records are usable in a downstream meta-analytic database.

Return EXACTLY ONE valid JSON object, parseable by `json.loads`. NO surrounding prose, NO markdown code fences, NO commentary.

## Core extraction rule: choose the solution with the largest number of factors

If the paper reports more than one factor solution for the same sample, extract the solution with the **largest number of factors**.

Examples:
- If a 2-factor, 3-factor, and 4-factor solution are presented, extract the 4-factor solution.
- If a 3-factor, 4-factor, and 5-factor solution are presented, extract the 5-factor solution.

This rule takes priority over whether a lower-dimensional solution is described as conventional, standard, preferred, theoretically expected, or better fitting.

Tied factor counts:
1. Prefer the solution labeled main / final / retained / preferred by the authors.
2. If still tied, prefer EFA over CFA.
3. If still tied, prefer oblique over orthogonal rotation.
4. If still tied, choose the clearest table with the most complete item-level loadings.

Always document the chosen solution and any relevant alternatives in `notes`.

Ignore any solution that includes items from measures other than the **${scale_name}**.

## JSON schema

```json
{
  "samples": [
    {
      "sample_id": "string",

${factor_loadings_schema}

${factor_correlations_schema}

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

You MUST include all ${n_loading_keys} loading keys (`${loading_first_key}` through `${loading_last_key}`) and all ${n_factors_pairs} factor-correlation keys (`${correlation_first_key}` through `${correlation_last_key}`) on every sample, even if the values are null.

## Multiple samples / groups

If the paper reports separate factor solutions for the **${scale_name}** across separate samples or groups, output ONE object per visibly separated group. Do NOT combine samples.

Separate samples/groups may be defined by:
- Study labels: Sample 1 vs. Sample 2, calibration vs. validation, derivation vs. replication.
- Language or country: English vs. French, Arabic vs. Canadian, German vs. Italian.
- Clinical status: clinical vs. nonclinical, patients vs. controls, patients vs. normal participants, psychiatric vs. community.
- Sex/gender: male vs. female, men vs. women, boys vs. girls.
- Age group: adolescents vs. adults, young vs. old, age bands.
- Any other explicitly separated group used for factor analysis.

If the paper reports a total sample alongside subgroups, include each as its own entry.

If multiple samples/groups are described in the paper but factor loadings are reported only for one of them, extract only the group(s) for which factor loadings are available and explain this in `notes`.

If the grouping is unclear, output a single entry with `sample_id: "sample1"` and explain the ambiguity in `notes`.

`sample_id` should be a short identifier derived from visible labels:
- `"total"`
- `"sample1"`, `"sample2"`
- `"male"`, `"female"`, `"men"`, `"women"`, `"boys"`, `"girls"`
- `"clinical"`, `"patient"`, `"patients"`, `"nonclinical"`, `"normal"`, `"controls"`
- `"adolescents"`, `"adults"`, `"young"`, `"old"`, `"students"`,
- `"English"`, `"French"`, `"Chinese"`, `"Arabic"`, `"German"`, etc.

Use the paper's own terminology where possible, but keep the identifier short and machine-readable. If no label is visible, use `"sample1"`.

## Factor naming and item identification

If the paper clearly uses the standard **${scale_name}** factor structure, map:
${factor_labels_block}

Make sure that the same factor labels are used for the same factors.

If the paper uses different naming or ordering, follow the paper's definition and order. For 4- or 5-factor solutions, do NOT invent meanings for F4/F5 unless explicitly defined; keep them as Factor 4 / Factor 5 and explain ambiguity in `notes`.

Recognize **${scale_name}** items by:

1. Explicit item numbers: 1–**${n_items}**.
2. Item labels or abbreviations that clearly correspond to **${scale_name}** items.
3. Recognizable text fragments${item_labels_inline}.

Use the item number as the canonical identifier, i.e., the `<item>` part of `F<factor>.<item>`.
${item_labels_block}
### Item-to-factor assignment

The item-to-factor assignment is usually recoverable from the same table that reports the factor loadings. In many tables, each item row appears together with loadings under factor columns, which directly identifies both the item number and the factor.

However, the item-to-factor assignment may also be reported in a separate table, appendix, model-description table, item-composition table, or text passage. If the loading table does not itself make the item-to-factor assignment clear, search other parts of the paper for:

- item-composition tables,
- scale/subscale assignment tables,
- CFA model specification tables,
- descriptions of which **${scale_name}** items define each factor,
- appendices listing **${scale_name}** items or subscales.

Use the separate item-assignment source only when it clearly corresponds to the same chosen factor solution and the same sample/group. Document this in `notes` and cite the relevant source in `evidence`.

Do NOT use a separate item-assignment table if it refers to a different model, different factor solution, different scale version, or different sample than the chosen loading solution.

## Factor loadings (`factor_loadings`)

Output keys for ALL items across ALL extracted factors.

Fill values:
- Loading explicitly shown for an item-factor cell → record as a number. Negative values are allowed.
- Cell suppressed, blank, or below threshold within the reported solution → `0.0` because the cell exists but the loading is not shown.
- Entire factor does NOT exist in the chosen solution, e.g., F4/F5 in a 3-factor model → set ALL corresponding factor keys to `null`.
- Table genuinely cut off or not visible → `null` and explain in `notes`.
- Table reports only primary loadings, one factor per item → enter the reported loading in that factor's key and set the other factors for that item to `0.0` for all factors that exist in the chosen solution.
- Table reports a CFA solution with one loading per item and the item-to-factor assignment is given elsewhere → use the external item-assignment information to place each loading in the correct factor and set other existing-factor loadings for that item to `0.0`.

Do not impute or estimate values.

Post-processing rule for empty factors:

- After extracting the factor loadings, check each existing factor.
- If all item loadings of a factor are `0.0`, treat this factor as not actually present in the extracted solution.
- Set all loading keys for this factor to `null`.
- Set all factor correlations involving this factor to `null`.
- Exclude this factor from the factor count and update `nfac` accordingly.
- Example: If a nominally 4-factor table contains an F4 column but all F4 item loadings are blank, suppressed, or coded as `0.0`, then F4 is not retained as an extracted factor. Set all `F4.*` values to `null`, set all correlations involving F4 to `null`, and set `nfac = 3`.
- Do not apply this rule to a factor with at least one nonzero loading. If a factor has at least one explicitly reported nonzero loading, keep the factor and retain `0.0` for suppressed/blank cells within that existing factor.

### Multi-level / spanning headers

Tables may use multi-level headers, such as superheadings spanning multiple columns, group labels above factor columns, or factor names as subheaders. Read the LOWEST-LEVEL header row that directly labels the numeric loading columns. Treat superheadings as structural grouping only; do not misinterpret them as separate variables.

If the table separates multiple samples/groups by columns, make sure the loadings are assigned to the correct `sample_id`.

If the header structure is too ambiguous to recover column identities, extract row-wise as best as possible and explain the ambiguity in `notes`.

## Factor correlations (`factor_correlations`)

Output ALL ${n_factors_pairs} unique factor-pair correlations: ${correlations_key_list}. Use the `R<i>.<j>` form with **i < j only**. Do NOT duplicate symmetric pairs.

Extract **latent factor correlations for the chosen factor-analytic solution only**. Do NOT extract correlations between **${scale_name}** subscale scores, scale scores, sum scores, factor scores, external variables, reliability coefficients, validity coefficients, or other observed variables as factor correlations.

Factor correlations may be reported anywhere in the manuscript, not only in the same table as the factor loadings. Search all tables, appendices, figures, and relevant text passages for factor correlations, factor intercorrelations, latent correlations, factor correlation matrices, Phi matrices, or interfactor correlation matrices. Use the correlations only if they clearly refer to the chosen **${scale_name}** factor solution and the same sample/group.

Do NOT treat correlations among **${scale_name}** subscales, scale scores, or derived factor scores as latent factor correlations unless the paper explicitly identifies them as correlations among the latent factors of the same chosen factor model.

Look for keywords such as `factor correlation`, `factor intercorrelation`, `interfactor correlation`, `latent correlation`, `Phi`, `phi matrix`, `factor correlation matrix`, or `interfactor correlation matrix`. Correlations may appear as either:

- the lower/upper triangle with `nfac × (nfac - 1) / 2` unique correlations, or
- the full symmetric factor-correlation matrix.

Either form is fine. Only output the i<j pairs.

Fill values:
- Correlation explicitly reported for the latent factors of the chosen factor-analytic solution → number. Negative values are allowed and values are typically in [-1, 1].
- Correlation involves a factor that does not exist in this solution, e.g., R1.4 in a 3-factor model → `null`.
- Correlation involves an empty factor removed by the post-processing rule for empty factors → `null`.
- Correlation simply not reported, suppressed, or not visible → `null`.

Priority rule for orthogonal rotations:

- If the chosen factor-analytic solution is explicitly described as using an orthogonal rotation, this overrides any correlations reported elsewhere between **${scale_name}** scales, subscales, factor scores, or "factors".
- For the chosen factor-analytic solution, set all correlations among existing factors to `0`.
- Orthogonal rotations include `varimax`, `quartimax`, `equamax`, `orthomax`, `parsimax`, or any rotation explicitly described as `orthogonal`.
- In an orthogonally rotated solution, latent factor correlations are structurally zero by definition.
- Do NOT replace these structural zeros with correlations reported elsewhere unless the paper explicitly states that those correlations are the latent factor correlations of the same chosen obliquely rotated factor model.
- Correlations between derived scales, subscale scores, factor scores, sum scores, or post-hoc "factors" are not latent factor correlations for an orthogonally rotated EFA solution.
- If a factor is removed by the empty-factor rule, do not set correlations involving that removed factor to `0`; set them to `null`.

Procrustes-rotation rule:

- Procrustes rotation is not automatically orthogonal.
- Code `rot = 1` and set existing-factor correlations to `0` only if the paper explicitly states "orthogonal Procrustes" or otherwise makes clear that the target rotation was orthogonal.
- Code `rot = 2` if the paper states "oblique Procrustes", allows correlated factors, or reports factor correlations for the Procrustes-rotated solution.
- If the paper only says "Procrustes rotation" without clarification, set `rot = null`; extract explicitly reported latent factor correlations if available, otherwise set them to `null`, and explain the ambiguity in `notes`.

## Multiple models / rotations / methods

If the paper reports MULTIPLE factor solutions for the same sample, such as:

- EFA + CFA,
- rotated + unrotated,
- 2-factor + 3-factor + 4-factor,
- 3-factor + 4-factor + 5-factor,
- Model A + Model B,
- competing clinical/nonclinical, male/female, or age-group solutions,

apply the following hierarchy:

1. Extract the **${scale_name}** solution with the HIGHEST number of factors.
2. If factor counts are tied, prefer the solution labeled main / final / retained / preferred.
3. If still tied, prefer EFA over CFA.
4. If still tied, prefer oblique over orthogonal rotation.
5. If still tied, prefer the table with the most complete item-level loading information.

If different samples/groups have different numbers of factors, apply this rule separately within each sample/group. For example, if men have a 3-factor and 4-factor solution and women have only a 3-factor solution, extract the 4-factor solution for men and the 3-factor solution for women.

After applying this hierarchy, apply the post-processing rule for empty factors. If the highest-dimensional table includes a factor whose extracted loadings are all `0.0`, remove that empty factor by setting all its loadings and correlations to `null` and reduce `nfac` accordingly.

Set `nfac` to the number of factors in the final extracted solution after removing empty factors.

Ignore any solution that includes items from measures other than the **${scale_name}**.

## Sample metadata (coded scheme)

Each sample's record carries the following fields. Fields documented as integer codes MUST be integers. Do not output strings for coded fields.

- **`pubyear`** — Publication year as an integer. Use the paper or citation header. Null if not visible.
- **`country`** — Country of participants as open text. Only fill if explicitly stated or clearly inferable from recruitment location, e.g., "German university students in Berlin" → `"Germany"`. If origin is unclear but author affiliation is mainly United States, use `"USA"`. Else null.
- **`continent`** — Continent of participants as open text. If `country` is set, derive it, e.g., Germany → `"Europe"`. If only continent is stated explicitly, fill that. Else null.
- **`lang`** — Language of the questionnaire instrument as open text, e.g., `"German"`, `"English"`, `"French"`. Use explicit statements such as "German version" or "translated into ...". If multiple language versions exist, match the language to THIS record's `sample_id`. If the sample is from an English-speaking country with no explicit translation note, use `"English"`. Else null.
- **`pubtype`** —
  - `1` = article in a journal
  - `2` = book
  - `3` = thesis, Master's thesis, or PhD dissertation
  - `4` = presentation or proceedings
  - `5` = other
  Null if uncertain and explain in `notes`.
- **`n`** — Sample size as an integer corresponding to THIS record's sample. Usually in the abstract or method section. If multiple groups are reported by sex/gender, clinical status, language, age group, or sample number, match the sample size to the corresponding `sample_id`. Null if not clearly matched.
- **`sex`** — Percentage of women in the sample, in [0, 100]. If only counts are given, compute `women / (women + men) × 100` and note the computation in `notes`. If the sample itself is all female/women/girls, set `100`. If the sample itself is all male/men/boys, set `0`. Null if not reported and not inferable from the group label.
- **`age`** — Mean age in years as a number. If reported separately by gender within the same sample, compute the n-weighted mean if the necessary information is available. If the sample is split by age groups, match the age value to the corresponding age-group record if possible. If only a range or median is given, set null and note. Prefer the reported mean.
- **`clinical`** —
  - `0` = nonclinical
  - `1` = clinical patients
  - `2` = mixed
  Null if unclear.
  If the sample itself is explicitly labeled patients, clinical, psychiatric, medical, or a named clinical diagnosis group, code `1`. If it is labeled normal, control, community, student, or nonclinical, code `0`. If both clinical and nonclinical participants are combined in the same factor analysis, code `2`.
- **`res`** — Number of Likert response options for the **${scale_name}** as an integer. Standard **${scale_name}** uses 5. If no adaptation/translation note exists, default to `5`. Else null.
- **`nfac`** — Number of factors in the final extracted solution as an integer from 1 to 10. Count only factors with at least one nonzero item loading. If all loadings for a factor are `0.0`, treat that factor as not present: set all loadings for that factor to `null`, set all correlations involving that factor to `null`, and do not count it in `nfac`. For example, if F1–F3 have at least one nonzero loading but all F4.* values are `0.0` and all F5.* values are `null`, set `nfac = 3`.
- **`cfa`** —
  - `0` = EFA, exploratory factor analysis
  - `1` = CFA, confirmatory factor analysis
  Search for literal phrases. If both EFA and CFA are reported and you extracted the EFA solution, code `0`; mention the alternative method in `notes`.
- **`met`** — Extraction / estimation method:
  - `1` = principal component analysis, PCA
  - `2` = principal axis factoring, PAF
  - `3` = maximum likelihood, ML
  - `4` = other; briefly name in `notes`, e.g., `"ULS"`, `"WLSMV"`, `"GLS"`, `"Bayesian"`
  Null if not stated.
- **`rot`** — Rotation method:
  - `1` = orthogonal, e.g., varimax, quartimax, equamax, orthomax, parsimax, or any rotation explicitly described as orthogonal
  - `2` = oblique, e.g., promax, oblimin, quartimin, biquartimin, geomin
  If the chosen factor-analytic solution is explicitly described as using an orthogonal rotation, code `rot = 1`, and set all correlations among existing factors to `0` regardless of any scale, subscale, or factor-score correlations reported elsewhere. Null only if rotation is not meaningful, e.g., single-factor CFA, or if the rotation type is ambiguous, e.g., unspecified Procrustes rotation.

## Notes field

Use `notes` for:

- Ambiguities, e.g., "table cut off at item 17".
- Computed values, e.g., "% women computed from n=143/290".
- The choice of solution when multiple factor solutions exist.
- The reason for extracting the highest-dimensional solution.
- Whether an initially extracted factor was removed because all its loadings were `0.0`.
- The source of item-to-factor assignment if it came from a separate table, appendix, or text passage.
- Factor-naming caveats, e.g., "factors labeled D1/D2/D3 in this paper, mapped to F1/F2/F3 by order of appearance".
- Any mismatch between loading table and item-assignment table.
- Whether reported correlations elsewhere were ignored because the chosen solution used an orthogonal rotation.
- The `pubtype` reasoning if uncertain.
- For any extra factor beyond **${n_factors_max}** truncate.
- Any sample/group split that required interpretation, e.g., sex/gender, clinical status, or age-group labels.

Include anything worth flagging for human review during the meta-analytic coding stage.

## Strict output constraints

- Return JSON only. No markdown fences, no preamble, no commentary.
- Use `null` for missing or unreported values.
- Use `0.0` only for suppressed loading cells in an existing factor.
- If all loadings of a factor are `0.0`, replace all loadings of that factor with `null`, set all correlations involving that factor to `null`, and reduce `nfac` accordingly.
- Use `0` for factor correlations that are structurally zero because the chosen solution used an orthogonal rotation.
- Coded fields (`pubtype`, `clinical`, `cfa`, `met`, `rot`) MUST be integers, not strings or null with descriptions.
- Numeric fields must be JSON numbers, not strings.
- The `samples` array must contain one entry per extracted sample/group.
- Every sample entry must contain all ${n_loading_keys} loading keys and all ${n_factors_pairs} factor-correlation keys.

## SUPPORTING EVIDENCE REQUIREMENT

Add an `evidence` array at the top level of the output. Each entry has exactly four keys:

- **`snippet`**: the EXACT verbatim text from the PDF that supports an extracted value. Quote character-for-character. Do not paraphrase, summarize, or add ellipses. If you cannot quote a string verbatim from the PDF, find a different snippet.
- **`page`**: integer, 1-indexed PDF page; NOT the journal/book page number printed in the document header/footer.
- **`source`**: the table or figure identifier, e.g., `"Table 2"`, `"Figure 1A"`, or `null` if from body text.
- **`field`**: a JSON path identifying which extracted value(s) this evidence supports, formatted exactly like the JSON structure you are emitting. Examples:
  - `"samples[0].factor_loadings"` — the whole loadings dictionary.
  - `"samples[0].factor_loadings.F1.5"` — a specific loading cell.
  - `"samples[0].factor_correlations"` — the full factor-correlation matrix.
  - `"samples[0].factor_correlations.R1.2"` — a specific factor correlation.
  - `"samples[0].n"` — the sample-size field.
  - `"samples[0].country"` — the country field.
  - `"samples[1]"` — sample 2 as a whole.

ALL FOUR KEYS ARE MANDATORY ON EVERY EVIDENCE ENTRY. The `page` field is the most commonly forgotten field. DO NOT omit it. Entries missing `page` cannot be linked to the source PDF and will be discarded by the post-processor. If you are uncertain which page a snippet appears on, count from page 1 and give your best estimate; never omit the field.

### Required evidence coverage

For every numeric table you extract from, including factor loadings and factor correlations, the evidence array MUST contain at least one entry whose `snippet` is the verbatim TABLE CAPTION, e.g., `"TABLE 1. Parameter estimates from confirmatory factor analyses of the ${scale_name}"`. The viewer relies on this to highlight the table region.

If item-to-factor assignment is taken from a separate table, appendix, or text passage, include evidence for that source as well.

If factor correlations are taken from a different table, appendix, figure, or text passage than the factor loadings, include separate evidence for that source.

If factor correlations are set to `0` because the chosen solution used an orthogonal rotation, include evidence for the orthogonal rotation statement, e.g., `"varimax rotation"` or `"orthogonal rotation"`.

If a factor is removed because all its extracted loadings are `0.0`, include evidence for the table or source showing those empty/suppressed loadings, and explain the removal in `notes`.

Also include:

- One snippet with the verbatim sentence identifying the sample or group.
- One snippet stating the chosen model or factor count.
- One snippet supporting the highest-factor-solution choice if multiple models are reported.
- One snippet for each metadata field that is not implicit, especially `n`, `country`, `lang`, `age`, `sex`, `clinical`, `cfa`, `met`, and `rot`.

### Evidence quality

BAD evidence — do NOT use:

- `"The fit indices reached acceptable standards."` — methodology, contains no extracted value.
- `"The parameter estimates are presented in Table 2."` — reference to the source, not the source itself.
- `"Cronbach's α was 0.87."` — about reliability, irrelevant to factor loadings.

GOOD evidence:

- `"TABLE 1. Parameter estimates from the results of confirmatory factor analyses..."` — table caption.
- `"1   .539   .576   .488"` — literal row from the loadings table.
- `"the sample comprised 147 non-clinical adolescents"` — sentence containing a literal sample-size value.
- `"A three-factor solution was retained"` — sentence stating the model.
- `"the Korean version ${scale_name}"` — language / instrument-version evidence.
- `"varimax rotation"` — rotation-method evidence.
- `"the two factors were rotated using the varimax rotation method"` — evidence that factor correlations are structurally zero.
- `"men and women were analyzed separately"` — group-splitting evidence.
- `"patients and normal controls"` — clinical-status group evidence.

Now perform the extraction on the supplied PDF. Return ONLY the JSON object — no preamble, no fences, no explanation.
