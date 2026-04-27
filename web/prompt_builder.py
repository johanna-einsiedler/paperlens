"""Prompt-building utilities: meta-prompt construction and evidence appendix."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Appended verbatim to every generated prompt before it is shown to the user.
EVIDENCE_APPENDIX = """

---
SUPPORTING EVIDENCE REQUIREMENT

Add an "evidence" key to every sample/result object in your output. Its value is a JSON array. Each array element must have exactly these four keys:

- "snippet": the EXACT verbatim text from the paper that is the source of the extracted value.
  Quote it character-for-character — do not paraphrase, do not summarise, do not add ellipses. If you cannot quote a string that appears verbatim in the PDF, use a different snippet (or set the value to null and explain in notes).
- "page": the sequential PDF page number as an INTEGER (1 = first page). Do NOT use the journal or book page number printed in the document header/footer.
- "source": the table or figure identifier (e.g. "Table 2", "Figure 1A"), or null if not from a named table/figure.
- "field": a JSON path identifying which extracted value(s) this evidence supports — formatted exactly like the JSON structure you are emitting.  Examples:
    "samples[0].factor_loadings"            -- evidence for the whole loadings table
    "samples[0].factor_loadings._table[0]"  -- evidence for row 0 of that table
    "samples[0].n"                          -- evidence for the sample size field
    "samples[0]"                            -- evidence for the sample as a whole (e.g. its identification)

⚠️ ALL FOUR KEYS ARE MANDATORY ON EVERY EVIDENCE ENTRY.
The "page" field is the most commonly forgotten — DO NOT omit it.  Evidence entries that are missing "page" cannot be linked to the source PDF and will not render highlights for the user.  If you are uncertain which page a snippet appears on, count from page 1 of the supplied PDF and give your best estimate; never omit the field.  An entry like {"snippet": "...", "field": "..."} (no page, no source) is INVALID and will be discarded.

EVIDENCE QUALITY (required)

Each snippet must be the ACTUAL SOURCE of an extracted value — never methodology, never a forward reference, never an adjacent claim about reliability or model fit.

❌ BAD evidence (do NOT use):
  - "The fit indices reached acceptable standards."        — methodology, contains no value
  - "The parameter estimates are presented in Table 2."    — reference to the source, not the source itself
  - "Cronbach's alpha was 0.87."                            — about reliability, irrelevant to factor loadings
  - "A factor analysis was conducted on the 20 items."     — generic procedural sentence

✅ GOOD evidence:
  - The literal table caption: "TABLE 2. Rotated factor matrix for the TAS-20..."
  - A literal row from the table: "1  .539  .576  .488"
  - A sentence containing the literal value: "the sample comprised 147 non-clinical adolescents"
  - The literal sentence stating the model: "A three-factor solution was retained"

REQUIRED COVERAGE

For every "_table" object you output, the evidence array MUST contain at least one entry whose snippet is the verbatim caption of that table (e.g. "TABLE 1. Parameter estimates...").  This is non-negotiable: the viewer relies on this to highlight the table region.

Beyond the per-table caption requirement, also include:
  - one entry whose snippet contains the verbatim sample-identification text (e.g. "147 non-clinical adolescents")
  - one entry whose snippet states the chosen model / factor count (e.g. "A three-factor solution was retained")

Do NOT try to embed snippet/page/source inline with numeric values — keep all evidence in the "evidence" array only.

Example (note how every "_table" is matched by a caption snippet, and every field path mirrors the JSON structure):
"factor_loadings": {"_table": [...]},
"evidence": [
  {"snippet": "TABLE 1. Parameter estimates from the results of confirmatory factor analyses...",
   "page": 3, "source": "Table 1", "field": "samples[0].factor_loadings"},
  {"snippet": "1  .539  .576  .488",
   "page": 3, "source": "Table 1", "field": "samples[0].factor_loadings._table[0]"},
  {"snippet": "the sample of 147 non-clinical adolescents aged 12 to 16 years",
   "page": 2, "source": null, "field": "samples[0].sample_id"},
  {"snippet": "A three-factor solution was retained",
   "page": 3, "source": null, "field": "samples[0].factor_loadings"}
]

---
TABULAR DATA — REQUIRED FORMAT

Whenever you would naturally return tabular data (factor loadings across items, correlation matrices, descriptive stats per group, parameter estimates per condition, etc.), wrap it with a "_table" marker:

  "<field_name>": {
    "_table": [
      {"<col_key>": value, "<col_key>": value, ...},   ← row 1
      {"<col_key>": value, "<col_key>": value, ...},   ← row 2
      ...
    ]
  }

Each row is one object in the array; object keys are column names; values are cell contents (numbers, strings, or null).  The "_table" key is a marker the viewer uses to render the array as a real HTML table — no guessing.

Examples:
- Item-level factor loadings:
    "factor_loadings": {"_table": [
        {"item": 1, "F1": 0.83, "F2": 0.12, "F3": 0.05},
        {"item": 2, "F1": 0.45, "F2": 0.71, "F3": 0.08},
        ...
    ]}
- Sample stats per group:
    "demographics": {"_table": [
        {"group": "control",   "n": 100, "mean_age": 22, "pct_female": 0.51},
        {"group": "treatment", "n": 100, "mean_age": 23, "pct_female": 0.48}
    ]}
- Inter-factor correlations:
    "factor_correlations": {"_table": [
        {"factor": "F1", "F1": 1.00, "F2": 0.42, "F3": 0.31},
        {"factor": "F2", "F1": 0.42, "F2": 1.00, "F3": 0.27},
        {"factor": "F3", "F1": 0.31, "F2": 0.27, "F3": 1.00}
    ]}

DO NOT use flat composite-key dicts like {"F1.1": 0.83, "F1.2": 0.45} — these are ambiguous and harder to render.  Always use the "_table" wrapper for tabular output."""


EXAMPLE_FILES = {
    "extraction": [
        "02_extraction_prompt_factor_loadings.txt",
        "03_extraction_prompt_correlations.txt",
        "04_extraction_prompt_metadata.txt",
    ],
    "labeling": [
        "01_detection_prompt.txt",
    ],
}


def load_example_prompts(mode: str) -> str:
    examples = []
    for fname in EXAMPLE_FILES.get(mode, []):
        path = PROMPTS_DIR / fname
        if path.exists():
            examples.append(f"=== Example prompt: {fname} ===\n{path.read_text().strip()}")
    return "\n\n".join(examples)


def build_meta_prompt(mode: str, question: str, context: str) -> str:
    examples = load_example_prompts(mode)

    if mode == "extraction":
        task_description = (
            "structured data extraction from academic papers — "
            "pulling specific values, statistics, or information into a structured JSON format"
        )
        output_guidance = (
            "Define the exact JSON schema for the output, including field names, types, "
            "and rules for null/missing values. Include rules for ambiguous cases such as "
            "multiple samples, merged table headers, or missing data. "
            "For any tabular data in the output (factor loadings, correlation matrices, "
            "descriptives per group, parameter estimates, etc.), wrap it with the explicit "
            "'_table' marker: "
            "<field>: {\"_table\": [{<col>: value, ...}, ...]}.  Each row is one object; "
            "keys are columns.  Do NOT use flat composite-key dicts like {\"F1.1\": 0.83}. "
            "Also require, for every '_table' the model emits, at least one evidence entry "
            "whose snippet is the verbatim table caption (e.g. \"TABLE 1. ...\"), and that "
            "the evidence 'field' property be a JSON path mirroring the output structure "
            "(e.g. \"samples[0].factor_loadings\", \"samples[0].factor_loadings._table[0]\")."
        )
    else:
        task_description = (
            "classification and labeling of academic papers — "
            "assigning structured categorical labels based on content"
        )
        output_guidance = (
            "Define the exact label categories with precise, mutually exclusive criteria. "
            "Specify what counts and what does not count for each label, and include rules "
            "for borderline or ambiguous cases. "
            "Output a JSON object with at minimum a 'label' field (the assigned category) "
            "and a 'rationale' field (one-sentence justification). "
            "Add any additional structured fields the task requires (e.g. sub-labels, confidence). "
            "The output must be parseable by json.loads with no extra text."
        )

    context_block = (
        f"\nAdditional context provided by the user:\n{context.strip()}"
        if context.strip()
        else ""
    )

    examples_block = f"\nHere are example prompts that define the expected style, structure, and level of detail:\n\n{examples}\n" if examples else ""

    return f"""You are an expert at writing high-quality prompts for AI-assisted research data {mode}.

Your task: write a professional, detailed prompt that an AI will use to perform {task_description}.

The user's research question / task description:
{question.strip()}
{context_block}
{examples_block}
Now write a new prompt tailored to the user's specific task. The prompt must:
1. Clearly state the task and what the AI is expected to do
2. {output_guidance}
3. Specify what to include and what to explicitly exclude
4. Handle edge cases and ambiguous situations
5. Match the professional quality and detail level of the examples above

Return only the prompt text itself, ready to be used directly with an AI model. Do not include any preamble or explanation."""
