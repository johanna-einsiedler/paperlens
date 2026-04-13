"""Prompt-building utilities: meta-prompt construction and evidence appendix."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Appended verbatim to every generated prompt before it is shown to the user.
EVIDENCE_APPENDIX = """

---
SUPPORTING EVIDENCE REQUIREMENT

Add an "evidence" key to every sample/result object in your output. Its value is a JSON array. Each array element must have exactly these four keys:

- "snippet": the exact verbatim text from the paper that supports an extraction (do not paraphrase — quote it directly)
- "page": the sequential PDF page number (1 = first page). Do NOT use the journal or book page number printed in the document header/footer.
- "source": the table or figure identifier (e.g. "Table 2", "Figure 1A"), or null if not from a named table/figure
- "field": a short label saying what this evidence supports (e.g. "F1 loadings", "sample identification", "factor count")

Coverage: include at least one evidence entry for the sample identification, one entry per factor column (citing the table that contains the loadings), and one entry for the total number of factors extracted.

Do NOT try to embed snippet/page/source inline with numeric values — keep all evidence in the "evidence" array only.

Example:
"evidence": [
  {"snippet": "Table 2. Rotated factor matrix...", "page": 4, "source": "Table 2", "field": "factor loadings"},
  {"snippet": "N = 147 undergraduate students", "page": 2, "source": null, "field": "sample identification"},
  {"snippet": "A two-factor solution was retained", "page": 3, "source": null, "field": "factor count"}
]"""


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
            "multiple samples, merged table headers, or missing data."
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
