"""Prompt-building utilities: meta-prompt construction and evidence appendix.

The appendix here is appended to every AI-generated prompt (see
``server.py`` ``/api/generate-prompt``).  Preset-driven workflows
(MASEMiner etc.) render their own ``.template.md`` files and never
touch this module — they carry their own evidence/confidence rules
inline because the wording is hand-tuned to the domain.

Anchors and examples in this file are DELIBERATELY domain-neutral.
The old version contained MASEM-specific anchors (TAS-20, Cronbach,
factor_loadings) that leaked factor-analytic wording into prompts
for unrelated tasks.  Examples now use generic placeholders so a
reader unfamiliar with factor analysis can still understand what
the appendix asks for.
"""

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
- "field": a JSON path identifying which extracted value(s) this evidence supports — formatted exactly like the JSON structure you are emitting.  Substitute your own top-level key and field names; the path must mirror the shape your prompt asks the model to produce.  Examples:
    "samples[0].<your_field>"               -- evidence for that single field on sample 0
    "samples[0].<your_field>._table[0]"     -- evidence for row 0 of a tabular field
    "samples[0].n"                          -- evidence for the sample size field
    "samples[0]"                            -- evidence for the sample as a whole (e.g. its identification)

⚠️ ALL FOUR KEYS ARE MANDATORY ON EVERY EVIDENCE ENTRY.
The "page" field is the most commonly forgotten — DO NOT omit it.  Evidence entries that are missing "page" cannot be linked to the source PDF and will not render highlights for the user.  If you are uncertain which page a snippet appears on, count from page 1 of the supplied PDF and give your best estimate; never omit the field.  An entry like {"snippet": "...", "field": "..."} (no page, no source) is INVALID and will be discarded.

EVIDENCE QUALITY (required)

Each snippet must be the ACTUAL SOURCE of an extracted value — never methodology, never a forward reference, never an adjacent claim about a related but different quantity.

❌ BAD evidence (do NOT use):
  - "The fit indices reached acceptable standards."        — methodology, contains no value
  - "The parameter estimates are presented in Table 2."    — reference to the source, not the source itself
  - "A standard analysis was conducted on the items."      — generic procedural sentence

✅ GOOD evidence:
  - The literal table caption: "TABLE 1. Descriptive statistics by group..."
  - A literal row from the table: "Control   100   22.4   0.51"
  - A sentence containing the literal value: "the sample comprised 147 non-clinical adolescents"
  - The literal sentence stating the chosen analytic decision

REQUIRED COVERAGE

For every "_table" object you output, the evidence array MUST contain at least one entry whose snippet is the verbatim caption of that table (e.g. "TABLE 1. ...").  This is non-negotiable: the viewer relies on this to highlight the table region.

Beyond the per-table caption requirement, also include:
  - one entry whose snippet contains the verbatim sample-identification text (e.g. "147 non-clinical adolescents")
  - one entry per non-trivial extracted quantity whose snippet is the literal value or its surrounding sentence

Do NOT try to embed snippet/page/source inline with numeric values — keep all evidence in the "evidence" array only.

Example (substitute your own top-level key and field names — the structure is what matters):
"<your_field>": {"_table": [...]},
"evidence": [
  {"snippet": "TABLE 1. Descriptive statistics by group...",
   "page": 3, "source": "Table 1", "field": "samples[0].<your_field>"},
  {"snippet": "Control   100   22.4   0.51",
   "page": 3, "source": "Table 1", "field": "samples[0].<your_field>._table[0]"},
  {"snippet": "the sample of 147 non-clinical adolescents aged 12 to 16 years",
   "page": 2, "source": null, "field": "samples[0].sample_id"}
]

---
TABULAR DATA — REQUIRED FORMAT

Whenever you would naturally return tabular data (descriptives per group, parameter estimates, correlation matrices, per-condition statistics, etc.), wrap it with a "_table" marker:

  "<field_name>": {
    "_table": [
      {"<col_key>": value, "<col_key>": value, ...},   ← row 1
      {"<col_key>": value, "<col_key>": value, ...},   ← row 2
      ...
    ]
  }

Each row is one object in the array; object keys are column names; values are cell contents (numbers, strings, or null).  The "_table" key is a marker the viewer uses to render the array as a real HTML table — no guessing.

Examples (generic shapes — substitute your own column names):
- Sample stats per group:
    "demographics": {"_table": [
        {"group": "control",   "n": 100, "mean_age": 22, "pct_female": 0.51},
        {"group": "treatment", "n": 100, "mean_age": 23, "pct_female": 0.48}
    ]}
- Per-condition outcomes:
    "outcomes": {"_table": [
        {"condition": "A", "mean": 4.20, "sd": 0.81, "n": 50},
        {"condition": "B", "mean": 4.65, "sd": 0.77, "n": 50}
    ]}
- Pairwise correlations:
    "correlations": {"_table": [
        {"row": "x1", "x1": 1.00, "x2": 0.42, "x3": 0.31},
        {"row": "x2", "x1": 0.42, "x2": 1.00, "x3": 0.27},
        {"row": "x3", "x1": 0.31, "x2": 0.27, "x3": 1.00}
    ]}

DO NOT use flat composite-key dicts like {"a.1": 0.83, "a.2": 0.45} — these are ambiguous and harder to render.  Always use the "_table" wrapper for tabular output.

---
EXTRACTION CONFIDENCE — SELF-ASSESS (REQUIRED)

Add an "extraction_confidence" object at the TOP LEVEL of your output (a sibling of "samples", "summaries", "labels", or whatever top-level data keys your schema uses).  For every major extracted-data key in your output — but NOT for "evidence" itself, and NOT for "extraction_confidence" itself — include one entry rated:

  - "high"   — value(s) were clearly stated in the paper; multiple cues agreed; standard reporting format
  - "medium" — value(s) inferred from partial information; one wording or table required interpretation; minor ambiguity
  - "low"    — value(s) reconstructed from incomplete reporting; major ambiguity, contradictions, or guesswork

For any field rated "medium" or "low", you MUST add a short "notes" string (≤ 200 characters) explaining what was uncertain or how you resolved it.  A "high" entry does NOT need notes.

Example (substitute your own top-level data block names — one entry per major block you extract):
"extraction_confidence": {
  "<data_block_1>": {"level": "high"},
  "<data_block_2>": {"level": "medium", "notes": "country was inferred from author affiliations; the paper does not state it explicitly"},
  "<data_block_3>": {"level": "low",    "notes": "values reported only for the overall scale, not subgroups; subgroup entries were estimated from partial reporting"}
}

Use this exactly once at the top level — do NOT put it inside "samples", inside "evidence", or per-record."""


def load_example_prompts(mode: str) -> str:
    """Kept as a no-op for back-compat with any external import path.

    The previous version referenced hardcoded ``.txt`` example files
    (factor-loadings / correlations / metadata extraction prompts).
    Those files were never present in the repo, so the function always
    returned an empty string.  Signature preserved; the meta-prompt no
    longer asks the LLM to mimic example files — the appendix carries
    the full spec instead.
    """
    return ""


def build_meta_prompt(mode: str, question: str, context: str) -> str:
    if mode == "extraction":
        task_description = (
            "structured data extraction from academic papers — "
            "pulling specific values, statistics, or information into a structured JSON format"
        )
        output_guidance = (
            "Define the exact JSON schema for the output, including field names, types, "
            "and rules for null/missing values. Include rules for ambiguous cases such as "
            "multiple samples, merged table headers, or missing data. "
            "For any tabular data in the output (descriptives per group, parameter estimates, "
            "correlation matrices, per-condition statistics, etc.), wrap it with the explicit "
            "'_table' marker: <field>: {\"_table\": [{<col>: value, ...}, ...]}. Each row is "
            "one object; keys are columns. Do NOT use flat composite-key dicts like "
            "{\"a.1\": 0.83}. Also require, for every '_table' the model emits, at least one "
            "evidence entry whose snippet is the verbatim table caption (e.g. \"TABLE 1. ...\"), "
            "and that the evidence 'field' property be a JSON path mirroring the output structure "
            "(e.g. \"samples[0].<your_field>\", \"samples[0].<your_field>._table[0]\")."
        )
    elif mode == "summarize":
        task_description = (
            "structured per-section summarisation of academic papers — "
            "producing a concise prose summary broken into named sections "
            "(e.g. background, methods, findings, limitations) with verbatim "
            "page-anchored evidence supporting each section"
        )
        output_guidance = (
            "Output a JSON object with a 'summaries' array — one element per distinct "
            "empirical study reported in the paper (typically just one). Each element "
            "carries named section keys whose values are markdown text written in concise "
            "academic English, plus an 'evidence' array of verbatim page-cited snippets. "
            "Each evidence entry MUST have a JSON-path 'field' value pointing to the "
            "summary section it supports — e.g. \"summaries[0].findings\". Require AT "
            "LEAST one evidence entry per non-null section so every claim is verifiable "
            "against the source. The output must be parseable by json.loads with no extra text."
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
            "Add any additional structured fields the task requires (e.g. sub-labels). "
            "The output must be parseable by json.loads with no extra text."
        )

    context_block = (
        f"\nAdditional context provided by the user:\n{context.strip()}"
        if context.strip()
        else ""
    )

    # The meta-prompt explicitly REQUIRES the generated prompt to declare
    # both an "evidence" array spec and an "extraction_confidence" object
    # spec, each with a worked JSON example INSIDE the generated prompt.
    # This shifts most of the load off the trailing EVIDENCE_APPENDIX —
    # the appendix becomes a backstop rather than the only place the
    # model learns about these structures.  The downstream readiness
    # check (web/prompt_check.py) verifies the generated prompt actually
    # contains both blocks before extraction starts.
    structure_requirements = """
You MUST instruct the downstream model — inside the prompt you write — to emit BOTH of the following JSON structures, and you must show worked examples of each inside the generated prompt body:

(A) An "evidence" array.  Spec required:
    - Each entry has exactly four keys: "snippet" (verbatim text from the paper), "page" (integer PDF page), "source" (table/figure identifier or null), "field" (JSON path into the output mirroring your output schema).
    - Include a worked JSON example showing 2-3 evidence entries with realistic snippets and field paths that match the output schema you defined above.
    - State that this requirement is non-negotiable.

(B) An "extraction_confidence" object at the top level of the output (a sibling of the main data array).  Spec required:
    - One entry per major top-level data block in your output schema.
    - Each entry has {"level": "high" | "medium" | "low"} plus, on medium/low, a "notes" string (≤ 200 chars) explaining the uncertainty.
    - Include a worked JSON example of the extraction_confidence object showing all three levels.
    - State that this requirement is mandatory.

These two blocks MUST appear in the prompt body you write — not just mentioned in passing, but specified with field-by-field rules and worked JSON examples.  The downstream tooling depends on them: the evidence array drives PDF highlighting; the extraction_confidence object drives per-block confidence badges.  A prompt without both will produce extractions that cannot be verified or trusted."""

    return f"""You are an expert at writing high-quality prompts for AI-assisted research data {mode}.

Your task: write a professional, detailed prompt that an AI will use to perform {task_description}.

The user's research question / task description:
{question.strip()}
{context_block}

{structure_requirements}

Now write a new prompt tailored to the user's specific task. The prompt must:
1. Clearly state the task and what the AI is expected to do
2. {output_guidance}
3. Specify what to include and what to explicitly exclude
4. Handle edge cases and ambiguous situations
5. Include both the evidence-array spec and the extraction_confidence-object spec (see above), each with a worked JSON example

Return only the prompt text itself, ready to be used directly with an AI model. Do not include any preamble or explanation."""
