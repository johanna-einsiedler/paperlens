"""Domain-workflow presets — discovery, parsing, validation.

A preset is a JSON file under ``web/presets/<id>.json`` describing a
tailored workflow (branding + recommended provider/model + a pre-built
extraction prompt + step-skip hint).  See ``presets/masem.json`` for the
canonical example.

The prompt body is large enough to be awkward in JSON, so each preset may
reference a sibling ``prompt_file`` (relative to the JSON, typically a
``.prompt.md``).  The loader inlines that into the returned dict under
``prompt`` — frontend never needs to know the file split.

Loaded at request time (not boot time), so dropping in a new preset takes
effect on the next API call without a restart.  Bad files are logged and
skipped rather than crashing the server.
"""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path
from typing import Any

PRESETS_DIR = Path(__file__).parent / "presets"

# Required fields a preset JSON must declare to be considered valid.
_REQUIRED_KEYS = ("id", "title", "tagline", "mode")
# Either "prompt" inline, "prompt_file" pointing to a sibling text file,
# or "prompt_template_file" pointing to a sibling template that gets
# rendered with the preset's "template_params" block.
_PROMPT_KEYS = ("prompt", "prompt_file", "prompt_template_file")


def _read_sibling(name: str, source_path: Path) -> str | None:
    """Read a file relative to the preset JSON, with a path-traversal guard
    so a malicious preset can't read arbitrary files via ``../``.  Returns
    None on read failure (caller logs and falls back)."""
    candidate = (source_path.parent / name).resolve()
    try:
        candidate.relative_to(PRESETS_DIR.resolve())
    except ValueError:
        print(f"[presets] file outside presets dir, ignoring: {name}",
              file=sys.stderr, flush=True)
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[presets] could not read {name}: {e}",
              file=sys.stderr, flush=True)
        return None


def _read_prompt_body(meta: dict, source_path: Path) -> str:
    """Resolve the preset's prompt body in priority order:
       1. ``prompt`` inline (used by simple presets that don't need a template)
       2. ``prompt_template_file`` + ``template_params`` (rendered)
       3. ``prompt_file`` (legacy, raw file inlined verbatim)
    Returns "" if none of the above are usable."""
    if meta.get("prompt"):
        return str(meta["prompt"])

    tpl_file = meta.get("prompt_template_file")
    if tpl_file:
        body = _read_sibling(tpl_file, source_path)
        if body is None:
            return ""
        params = meta.get("template_params") or {}
        return _render_template(body, params)

    pf = meta.get("prompt_file")
    if pf:
        body = _read_sibling(pf, source_path)
        return body or ""
    return ""


# ── Template rendering for parameterised presets ───────────────────────────
#
# A preset can ship its prompt as a ``$placeholder``-bearing template and a
# ``template_params`` block in the JSON.  The loader generates derived fields
# (schema fragments, lists of factor-correlation keys, conditional sections)
# from the primitive params, then runs ``string.Template`` substitution.
#
# Why ``string.Template`` and not ``str.format`` / Jinja:
#   * Prompts contain a LOT of literal ``{`` / ``}`` (JSON examples).  Using
#     ``str.format`` would force escaping every brace.  ``$placeholder``
#     syntax sidesteps the conflict entirely.
#   * Jinja is overkill — we don't need loops or conditionals in the
#     template; the renderer pre-builds blocks (a string per conditional
#     section) and substitutes them in.

# Placeholders the template may reference but the params may not provide
# (e.g., a starter preset that doesn't use the CFA fallback).  Defaults to
# empty string so the rendered output skips that block cleanly.
_OPTIONAL_PLACEHOLDERS = (
    # Inner blocks (used inside section bodies)
    "factor_naming_block",
    "factor_synonyms_block",
    "factor_labels_block",
    "factor_labels_list",
    "factor_key_mapping",
    "cfa_assignment_block",
    "item_texts_block",
    "item_labels_block",
    "item_labels_list",
    "item_labels_inline",
    # Section-level placeholders (used in the legacy template)
    "factor_naming_section",
    "factor_loadings_section",
    "factor_correlations_section",
    "correlation_matrix_section",
    "single_correlations_section",
    "effect_sizes_section",
    "variables_block",
    "multiple_models_section",
    "res_field_block",
    "schema_invariants_line",
    "study_characteristics_block",
)


def _render_template(template_str: str, params: dict) -> str:
    """Render a preset prompt template with the provided params.  Builds
    derived fields (generated schema fragments + section blocks gated on
    ``data_sources``), merges them with the raw params, then performs
    ``$var`` substitution and collapses any extra blank lines created by
    skipped optional sections."""
    derived = _build_derived_template_vars(params)
    full: dict[str, Any] = {**params, **derived}
    # string.Template only accepts string values; coerce numerics.
    sub: dict[str, str] = {}
    for k, v in full.items():
        if isinstance(v, (str, int, float)):
            sub[k] = str(v)
        else:
            # Lists / dicts / None get skipped — they're meant to feed the
            # derived-block builders, not to be substituted directly.
            continue
    for k in _OPTIONAL_PLACEHOLDERS:
        sub.setdefault(k, "")
    try:
        rendered = string.Template(template_str).substitute(sub)
    except KeyError as e:
        # Surface the missing key but don't crash the whole loader.
        print(f"[presets] template references unfilled placeholder: {e}",
              file=sys.stderr, flush=True)
        rendered = string.Template(template_str).safe_substitute(sub)
    return _collapse_blank_lines(rendered)


def _collapse_blank_lines(text: str) -> str:
    """Collapse runs of three or more newlines down to exactly two — i.e.
    collapse multiple blank lines down to a single blank line.  This
    keeps the output tidy when an optional section placeholder is empty
    (otherwise the section's surrounding blank lines stack up)."""
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _build_derived_template_vars(p: dict) -> dict[str, Any]:
    """Translate the primitive ``template_params`` into the larger set of
    placeholders the template references — generated schema fragments,
    pretty-formatted lists, conditional sections, etc."""
    n_items   = int(p.get("n_items")   or 0)
    n_factors = int(p.get("n_factors") or 0)

    # Counts referenced in prose (e.g. "all 100 loading keys").
    n_loading_keys = n_items * n_factors
    n_corr_pairs   = n_factors * (n_factors - 1) // 2

    # Pretty word for n_factors (FIVE / THREE / etc.) — used in the goal
    # statement.  Keep capitalised to match the existing prompt's style.
    n_factors_word = _english_number_word(n_factors).upper()

    # Generated schema fragments.
    factor_loadings_schema     = _render_factor_loadings_schema(n_factors, n_items)
    factor_correlations_schema = _render_factor_correlations_schema(n_factors)
    correlations_key_list      = _render_correlations_key_list(n_factors)

    # Conditional blocks.
    factor_naming    = p.get("factor_naming") or []
    factor_naming_block    = _render_factor_naming_block(factor_naming)
    factor_synonyms_block  = _render_factor_synonyms_block(factor_naming)
    factor_labels_block    = _render_factor_labels_block(factor_naming)
    cfa_assignment_block   = _render_cfa_assignment_block(p.get("cfa_item_assignment") or {})
    # Item-labels: the new (post-v2) template uses ``item_labels`` in two
    # places — an inline phrase ("Recognizable text fragments of the
    # items below") and a list block at the end of the section.  Always
    # build both regardless of include_item_texts (which was the old
    # template's gating flag) — the new template renders them
    # unconditionally when the list is non-empty.
    item_texts_list        = p.get("item_texts") or []
    include_items          = bool(item_texts_list)
    item_texts_block       = _render_item_texts_block(item_texts_list, include_items)
    item_labels_block      = _render_item_labels_block(item_texts_list)
    item_labels_list       = _render_item_labels_list(item_texts_list)
    factor_labels_list     = _render_factor_labels_list(factor_naming)
    factor_key_mapping     = _render_factor_key_mapping(n_factors)
    item_labels_inline     = " of the items listed below" if item_texts_list else ""

    # First item-text used as the verbatim example in section "Factor
    # naming + item identification".  When the user supplied item_texts
    # we use their first one; otherwise fall back to a generic
    # placeholder so we never leak copyrighted instrument content into
    # the prompt.
    item_texts = p.get("item_texts") or []
    item_text_example = (
        _shorten_item_text(item_texts[0]) if (item_texts and p.get("include_item_texts", False))
        else "a recognisable item phrase from the instrument"
    )

    # Examples that reference factor indices that don't exist in this
    # solution (e.g. "F4/F5 in a 3-factor model").  Cap at n_factors so
    # the example stays meaningful for any factor count.
    if n_factors >= 4:
        nonexistent_factors_example = "F4/F5 in a 3-factor model"
        nonexistent_factor_glob_example = "`F4.*` / `F5.*`"
        nonexistent_correlation_example = "R1.4 in a 3-factor model"
    elif n_factors == 3:
        nonexistent_factors_example = "F3 in a 2-factor model"
        nonexistent_factor_glob_example = "`F3.*`"
        nonexistent_correlation_example = "R1.3 in a 2-factor model"
    else:
        nonexistent_factors_example = "extra factors"
        nonexistent_factor_glob_example = "extra factor keys"
        nonexistent_correlation_example = "extra-factor correlations"

    factor_range_short = f"F1–F{n_factors}" if n_factors > 1 else "F1"

    # First/last keys used in prose ranges like ``\`F1.1\`…\`F5.20\``.
    loading_first_key     = "F1.1" if (n_items and n_factors) else ""
    loading_last_key      = f"F{n_factors}.{n_items}" if (n_items and n_factors) else ""
    correlation_first_key = "R1.2" if n_factors >= 2 else ""
    correlation_last_key  = f"R{n_factors - 1}.{n_factors}" if n_factors >= 2 else ""

    # Instrument-filter sentence — generalises the original "Ignore any
    # solution that includes items from measures other than the TAS-20"
    # using the configured ``content_scope``.
    content_scope = p.get("content_scope") or "concrete_items"
    instrument_name = p.get("instrument_name") or "the target instrument"
    instrument_filter_line = _render_instrument_filter(content_scope, instrument_name)

    # Data-source-gated sections.  Each helper returns either the entire
    # section (with leading "\n\n" + heading) or "" — the renderer's
    # ``_collapse_blank_lines`` cleans up double blanks afterwards.
    sources = p.get("data_sources") or []
    fl_active   = "factor_loadings"      in sources
    fc_active   = "factor_correlations"  in sources
    cmat_active = "correlation_matrix"   in sources
    scor_active = "single_correlations"  in sources
    es_active   = "effect_sizes"         in sources

    inner_subs = {
        "instrument_name":                  instrument_name,
        "instrument_name_long":             p.get("instrument_name_long") or instrument_name,
        "n_items":                          n_items,
        "n_factors":                        n_factors,
        "n_factors_word":                   n_factors_word,
        "n_loading_keys":                   n_loading_keys,
        "n_corr_pairs":                     n_corr_pairs,
        "factor_range_short":               factor_range_short,
        "loading_first_key":                loading_first_key,
        "loading_last_key":                 loading_last_key,
        "correlation_first_key":            correlation_first_key,
        "correlation_last_key":             correlation_last_key,
        "factor_naming_block":              factor_naming_block,
        "factor_synonyms_block":            factor_synonyms_block,
        "cfa_assignment_block":             cfa_assignment_block,
        "item_texts_block":                 item_texts_block,
        "item_text_example":                item_text_example,
        "nonexistent_factors_example":      nonexistent_factors_example,
        "nonexistent_factor_glob_example":  nonexistent_factor_glob_example,
        "nonexistent_correlation_example":  nonexistent_correlation_example,
        "correlations_key_list":            correlations_key_list,
    }

    factor_naming_section = _section_factor_naming(fl_active, inner_subs) if fl_active else ""
    factor_loadings_section = _section_factor_loadings(fl_active, inner_subs) if fl_active else ""
    factor_correlations_section = (
        _section_factor_correlations(inner_subs) if fc_active else ""
    )
    correlation_matrix_section   = _section_correlation_matrix(p)   if cmat_active else ""
    single_correlations_section  = _section_single_correlations(p)  if scor_active else ""
    effect_sizes_section         = _section_effect_sizes(p)         if es_active   else ""
    multiple_models_section      = _section_multiple_models()       if fl_active else ""

    # The ``res`` (Likert response options) field belongs to the
    # psychometric / factor-loadings world.  Drop it from the metadata
    # bullets when factor_loadings isn't being extracted.
    res_field_block = _section_res_field(instrument_name) if fl_active else ""

    # Schema-invariants line near the top of the prompt.  Adapts to
    # whichever data sources are active.
    schema_invariants_line = _render_schema_invariants_line(
        fl_active, fc_active, cmat_active, scor_active, es_active, inner_subs,
    )

    # Goal items — the numbered list at the top.  Always ends with the
    # metadata bullet; the data-source-specific items lead.
    goal_items = _render_goal_items(
        fl_active, fc_active, cmat_active, scor_active, es_active, inner_subs,
    )

    # JSON schema — assembled from data-source-specific fragments.
    json_schema = _render_full_json_schema(
        fl_active, fc_active, cmat_active, scor_active, es_active, inner_subs,
    )

    # Preamble qualifier — "factor-analytic " (with trailing space) when
    # factor loadings are being extracted, otherwise empty so the
    # preamble reads naturally for correlation-only presets.
    preamble_data_qualifier = "factor-analytic " if fl_active else ""

    # Free-form study-context paragraph the user typed into section D
    # of the guided builder.  Rendered as a small "## About these
    # studies" block right after the preamble; absent when the user
    # didn't fill section D in.
    study_characteristics_block = _render_study_characteristics_block(
        p.get("study_characteristics_text") or "",
    )

    return {
        # Aliases for the new (post-v2) template's placeholder names.
        # The internal canonical names (instrument_name, n_factors,
        # n_corr_pairs) keep their existing meaning everywhere else.
        "scale_name":                  p.get("scale_name") or p.get("instrument_name") or "the target instrument",
        "n_factors_max":               n_factors,
        "n_factors_pairs":             n_corr_pairs,
        # Canonical derived values
        "n_loading_keys":              n_loading_keys,
        "n_corr_pairs":                n_corr_pairs,
        "n_factors_word":              n_factors_word,
        "factor_loadings_schema":      factor_loadings_schema,
        "factor_correlations_schema":  factor_correlations_schema,
        "correlations_key_list":       correlations_key_list,
        "factor_naming_block":         factor_naming_block,
        "factor_synonyms_block":       factor_synonyms_block,
        "factor_labels_block":         factor_labels_block,
        "factor_labels_list":          factor_labels_list,
        "factor_key_mapping":          factor_key_mapping,
        "cfa_assignment_block":        cfa_assignment_block,
        "item_texts_block":            item_texts_block,
        "item_labels_block":           item_labels_block,
        "item_labels_list":            item_labels_list,
        "item_labels_inline":          item_labels_inline,
        "item_text_example":           item_text_example,
        "nonexistent_factors_example":      nonexistent_factors_example,
        "nonexistent_factor_glob_example":  nonexistent_factor_glob_example,
        "nonexistent_correlation_example":  nonexistent_correlation_example,
        "factor_range_short":          factor_range_short,
        "loading_first_key":           loading_first_key,
        "loading_last_key":            loading_last_key,
        "correlation_first_key":       correlation_first_key,
        "correlation_last_key":        correlation_last_key,
        "instrument_filter_line":      instrument_filter_line,
        # Section blocks
        "factor_naming_section":       factor_naming_section,
        "factor_loadings_section":     factor_loadings_section,
        "factor_correlations_section": factor_correlations_section,
        "correlation_matrix_section":  correlation_matrix_section,
        "single_correlations_section": single_correlations_section,
        "effect_sizes_section":        effect_sizes_section,
        "multiple_models_section":     multiple_models_section,
        # Variables block — list of canonical variables / synonyms /
        # definitions, used by both the data-source-section helpers and
        # by the standalone effect-sizes template.
        "variables_block":             _render_variables_block(p.get("variables") or []) or "(no variables defined — extract every reported pairwise effect size)",
        "res_field_block":             res_field_block,
        "schema_invariants_line":      schema_invariants_line,
        "goal_items":                  goal_items,
        "json_schema":                 json_schema,
        "preamble_data_qualifier":     preamble_data_qualifier,
        "study_characteristics_block": study_characteristics_block,
    }


def _shorten_item_text(text: str) -> str:
    """Produce a short item-fragment example by clipping a long item text
    to a leading clause.  Avoids dumping the full sentence into the
    "(e.g., …)" example placeholder."""
    cleaned = text.strip().rstrip(".")
    # Cut at the first comma if there is one (so we get just the head clause)
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    return cleaned


# ── Block builders ────────────────────────────────────────────────────────

def _render_factor_loadings_schema(n_factors: int, n_items: int) -> str:
    """Build the ``"factor_loadings": { ... }`` JSON-fragment lines used
    inside the template's ```json``` schema block.  Layout matches the
    original hand-written TAS-20 fragment:
      * F1 enumerates all items (one per ~3 keys per line).
      * F2..Fk get a one-line summary each (``F2.1 ... F2.20``).
    """
    if n_factors < 1 or n_items < 1:
        return '      "factor_loadings": {},'

    lines: list[str] = ['      "factor_loadings": {']

    # F1: full enumeration.  Match the original's grouping of 3 keys per
    # line, with widths derived from how the original lined up the colon.
    f1_keys = [f"F1.{i}" for i in range(1, n_items + 1)]
    chunks = []
    for i in range(0, n_items, 3):
        chunk = f1_keys[i : i + 3]
        formatted = ", ".join(f'"{k}":{_pad(k)}number|null' for k in chunk)
        chunks.append("        " + formatted + ",")
    # When there are no further factors (n_factors == 1), the very last
    # F1 line drops its trailing comma since it's the final entry in
    # ``factor_loadings``.  When F2..Fk follow, every F1 line keeps its
    # comma (matches the original prompt's layout).
    if chunks and n_factors == 1:
        chunks[-1] = chunks[-1].rstrip(",")
    lines.extend(chunks)
    if n_factors > 1:
        lines.append("")  # blank separator before F2

    # F2..Fk: one-line summary in the original style.
    for f in range(2, n_factors + 1):
        last_idx = n_items
        line = (
            f'        "F{f}.1":  number|null, "...":   '
            f'"(F{f}.2 ... F{f}.{last_idx})"'
        )
        if f < n_factors:
            line += ","
        lines.append(line)

    lines.append("      },")
    return "\n".join(lines)


def _pad(key: str) -> str:
    """Match the original's two-space alignment for one-digit item indices
    (`"F1.1":  number` has two spaces) and one-space alignment for two-
    digit indices (`"F1.10": number` has one space)."""
    # The keys are like "F1.1" or "F1.10" — split on dot to get item idx
    try:
        item = int(key.rsplit(".", 1)[1])
    except (ValueError, IndexError):
        return " "
    return "  " if item < 10 else " "


def _render_factor_correlations_schema(n_factors: int) -> str:
    """Build the ``"factor_correlations": { ... }`` JSON fragment.  Lines
    are grouped by lower factor index, matching the original layout
    (R1.* on one line, R2.* on the next, etc.)."""
    if n_factors < 2:
        return '      "factor_correlations": {},'

    lines: list[str] = ['      "factor_correlations": {']
    pair_groups: list[list[str]] = []
    for i in range(1, n_factors):
        group = [f"R{i}.{j}" for j in range(i + 1, n_factors + 1)]
        pair_groups.append(group)

    for idx, group in enumerate(pair_groups):
        formatted = ", ".join(f'"{k}": number|null' for k in group)
        is_last = idx == len(pair_groups) - 1
        lines.append("        " + formatted + ("" if is_last else ","))

    lines.append("      },")
    return "\n".join(lines)


def _render_correlations_key_list(n_factors: int) -> str:
    """Build the comma-separated, backticked list of correlation keys
    that appears in prose: \"`R1.2`, `R1.3`, …, `R4.5`\"."""
    keys = [f"R{i}.{j}" for i in range(1, n_factors)
                          for j in range(i + 1, n_factors + 1)]
    return ", ".join(f"`{k}`" for k in keys)


def _render_factor_naming_block(naming: list) -> str:
    """If the preset declares per-factor names (e.g. for TAS-20:
    DIF/DDF/EOT), emit the bullet list mapping ``F1`` → factor.  Each
    entry is either a string (rendered verbatim with bold) or a dict
    ``{"abbrev": "DIF", "name": "Difficulty Identifying Feelings"}`` —
    the dict form gets rendered as ``- F1 = **DIF** (Difficulty…)``.
    Empty list → empty string so the surrounding section is skipped."""
    if not naming:
        return ""
    lines = []
    for i, entry in enumerate(naming, 1):
        if isinstance(entry, dict):
            abbrev = entry.get("abbrev") or ""
            name   = entry.get("name") or ""
            if abbrev and name:
                lines.append(f"- F{i} = **{abbrev}** ({name})")
            elif abbrev:
                lines.append(f"- F{i} = **{abbrev}**")
            elif name:
                lines.append(f"- F{i} = **{name}**")
        else:
            lines.append(f"- F{i} = **{entry}**")
    return "\n".join(lines)


def _render_factor_labels_block(naming: list) -> str:
    """Render the ``${factor_labels_block}`` placeholder used by the new
    MASEMiner template — a numbered list of ``F<i> = **<abbrev>** (<name>)``
    lines.  Format matches the format the user pastes into the builder's
    factor-labels textarea:

        1. F1 = **DIF** (Difficulty Identifying Feelings)
        2. F2 = **DDF** (Difficulty Describing Feelings)
        3. F3 = **EOT** (Externally Oriented Thinking)

    When ``naming`` is empty, falls back to a generic "F1, F2, F3, …"
    description so the surrounding sentence still reads naturally."""
    if not naming:
        return "- Use `F1`, `F2`, `F3`, … as factor identifiers in the order the paper presents them, without any a-priori semantic labels."
    lines = []
    for i, entry in enumerate(naming, 1):
        if isinstance(entry, dict):
            abbrev = (entry.get("abbrev") or "").strip()
            name   = (entry.get("name") or "").strip()
            if abbrev and name:
                lines.append(f"{i}. F{i} = **{abbrev}** ({name})")
            elif abbrev:
                lines.append(f"{i}. F{i} = **{abbrev}**")
            elif name:
                lines.append(f"{i}. F{i} = **{name}**")
        elif isinstance(entry, str) and entry.strip():
            lines.append(f"{i}. F{i} = {entry.strip()}")
    return "\n".join(lines)


def _render_item_labels_list(items: list) -> str:
    """Render the plain ``${item_labels_list}`` placeholder used by the
    v3 MASEMiner template — a flat numbered list of ``<i>: <text>``
    lines with NO heading and NO surrounding blank lines (the template
    adds its own).  Returns "" when ``items`` is empty so the SCALE
    SPECIFICATION block just shows the descriptive line."""
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return ""
    return "\n".join(f"{i}: {txt}" for i, txt in enumerate(cleaned, 1))


def _render_factor_labels_list(naming: list) -> str:
    """Render the plain ``${factor_labels_list}`` placeholder used by
    the v3 MASEMiner template — ``F<i> = <abbrev> (<name>)`` lines
    with NO leading numbering and NO ``**bold**`` markers.  Empty
    when no naming is supplied."""
    if not naming:
        return ""
    lines: list[str] = []
    for i, entry in enumerate(naming, 1):
        if isinstance(entry, dict):
            abbrev = (entry.get("abbrev") or "").strip()
            name   = (entry.get("name")   or "").strip()
            if abbrev and name:
                lines.append(f"F{i} = {abbrev} ({name})")
            elif abbrev:
                lines.append(f"F{i} = {abbrev}")
            elif name:
                lines.append(f"F{i} = {name}")
        elif isinstance(entry, str) and entry.strip():
            lines.append(f"F{i} = {entry.strip()}")
    return "\n".join(lines)


# Roman numerals used in the auto-generated factor-key mapping block.
# Capped at 10 (any larger n_factors_max is treated as exceptional and
# falls back to plain "Factor <n>" for the trailing entries).
_ROMAN_NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")


def _render_factor_key_mapping(n_factors_max: int) -> str:
    """Render the ``${factor_key_mapping}`` block — one mapping line
    per factor up to ``n_factors_max`` translating common
    factor-label variants in the source paper into the JSON factor
    keys F1..Fn. """
    if n_factors_max <= 0:
        return ""
    lines: list[str] = [
        "Map factor labels in the paper to JSON factor keys by factor position/number:"
    ]
    for i in range(1, n_factors_max + 1):
        roman = _ROMAN_NUMERALS[i - 1] if i <= len(_ROMAN_NUMERALS) else str(i)
        lines.append(
            f"- F-{roman}, F{roman}, Factor {roman}, Factor {i}, Component {i} -> F{i}"
        )
    return "\n".join(lines)


def _render_item_labels_block(items: list) -> str:
    """Render the ``${item_labels_block}`` placeholder used by the new
    MASEMiner template — a numbered list of item texts in the format
    ``<i>: <text>``.  When ``items`` is empty, returns an empty string
    so the surrounding paragraph reads naturally without a block."""
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return ""
    lines = ["", "### Reference item texts", ""]
    for i, txt in enumerate(cleaned, 1):
        lines.append(f"{i}: {txt}")
    lines.append("")
    return "\n".join(lines)


def _render_factor_synonyms_block(naming: list) -> str:
    """Emit the "Synonyms" bullet list (``"Difficulty identifying feelings" → DIF → F1``)
    used in the prompt's "Factor naming + item identification" section.
    Skipped when factor_naming is empty or doesn't carry both abbrev+name.
    Aligns the arrows by right-padding the longest quoted name (matching
    the original prompt's manual-aligned layout)."""
    items: list[tuple[str, str]] = []
    for entry in naming:
        if not isinstance(entry, dict):
            continue
        abbrev = (entry.get("abbrev") or "").strip()
        name   = (entry.get("name")   or "").strip()
        if not (abbrev and name):
            continue
        items.append((_sentence_case(name), abbrev))
    if not items:
        return ""
    max_quoted_len = max(len(f'"{n}"') for n, _ in items)
    lines = []
    for i, (name, abbrev) in enumerate(items, 1):
        padded = f'"{name}"'.ljust(max_quoted_len)
        lines.append(f"- {padded} → {abbrev} → F{i}")
    return "\n".join(lines)


def _sentence_case(name: str) -> str:
    """Lowercase all words after the first.  Used to convert factor
    names from Title Case ("Difficulty Identifying Feelings") to the
    sentence-case form expected by the synonym list ("Difficulty
    identifying feelings")."""
    parts = name.split()
    if not parts:
        return name
    return parts[0] + (" " + " ".join(p.lower() for p in parts[1:]) if len(parts) > 1 else "")


def _render_cfa_assignment_block(cfa: dict) -> str:
    """Emit the optional CFA item-to-factor assignment paragraph used as
    a fallback when the table doesn't explicitly label item→factor.  The
    block matches the original TAS-20 wording when the standard
    7/5/8-item assignment is supplied; absent → empty string."""
    if not cfa:
        return ""
    lines = []
    for factor in sorted(cfa.keys()):
        items = cfa[factor]
        items_str = ", ".join(str(i) for i in items)
        lines.append(f"- {factor} = items {items_str}")
    return "\n".join(lines)


def _render_item_texts_block(items: list, include: bool) -> str:
    """Optional list of item texts the model is told to match against,
    so reordered or paraphrased items in the source paper still get
    located correctly.  Suppressed when ``include_item_texts`` is False
    or the list is empty."""
    if not include or not items:
        return ""
    lines = ["", "### Reference item texts",
             "",
             "The official item content (in original order) is:",
             ""]
    for i, txt in enumerate(items, 1):
        lines.append(f"{i}. {txt}")
    lines.append("")
    lines.append(
        "If the paper presents items in a different order or with paraphrased "
        "wording, match each table row to the closest item by SEMANTIC content "
        "and use the original item number above as the canonical "
        "`<item>` index.  Note any reorderings or paraphrases in `notes`."
    )
    return "\n".join(lines)


def _render_study_characteristics_block(text: str) -> str:
    """Optional "About these studies" paragraph the user filled into
    section D of the guided builder.  Empty input → empty block (no
    extra heading dropped into the prompt)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return f"\n## About these studies\n\n{cleaned}\n"


def _render_instrument_filter(scope: str, instrument_name: str) -> str:
    """Build the one-liner that tells the model what NOT to extract.
    Wording adapts to the configured content scope."""
    if scope == "concrete_items":
        # Definite article reads more naturally for instrument names
        # (e.g. "other than the TAS-20" vs "other than TAS-20"), and
        # mirrors the original prompt's phrasing.
        return (
            f"Ignore any solution that includes items from measures other than the {instrument_name}."
        )
    if scope == "content_groups":
        return (
            "Ignore any reported correlation that does NOT involve at least one "
            "of the listed scales / instruments."
        )
    if scope == "theoretical_constructs":
        return (
            "Ignore any reported correlation whose endpoints don't map to one "
            "of the listed theoretical constructs (use the supplied synonyms "
            "to recognise alternative wordings)."
        )
    return f"Restrict extraction to data relevant to the {instrument_name}."


_NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _english_number_word(n: int) -> str:
    """Spell small integers as English words (used in the goal statement
    where the original prompt says "up to FIVE factors")."""
    return _NUMBER_WORDS.get(n, str(n))


# ── Section builders ──────────────────────────────────────────────────────
#
# Each ``_section_*`` returns the entire ``## Heading\n\n…body…`` chunk
# (with one leading blank line, no trailing blank line) when active, or
# ``""`` when the section is gated off by ``data_sources``.  The renderer
# stitches them together and ``_collapse_blank_lines`` cleans up any
# double-blanks left by skipped sections.
#
# Section bodies are ``string.Template`` strings rendered against the
# same param map as the top-level template, so the same placeholders
# work inside.


def _render_subtemplate(body: str, subs: dict[str, Any]) -> str:
    """Render a section body — a small ``string.Template`` — with the
    inner-substitution map (already string-coerced upstream).  Missing
    keys fall through via ``safe_substitute`` so a partially-configured
    preset still produces something useful."""
    sub = {k: str(v) for k, v in subs.items() if isinstance(v, (str, int, float))}
    return string.Template(body).safe_substitute(sub)


def _section_factor_naming(active: bool, subs: dict) -> str:
    if not active:
        return ""
    body = """

## Factor naming + item identification

If the paper clearly uses the standard ${instrument_name} three-factor structure, map:
${factor_naming_block}

Synonyms map to the same factor index:
${factor_synonyms_block}

If the paper uses different naming or ordering, follow the paper's definition. For 4- or 5-factor solutions, do NOT invent meanings for F4/F5 unless explicitly defined; keep them as Factor 4 / Factor 5 and explain ambiguity in `notes`.

Recognise ${instrument_name} items by item numbers 1–${n_items} OR by recognisable item text fragments (e.g., "${item_text_example}…"). Use the item number as the canonical identifier (the `<item>` part of `F<factor>.<item>`).${item_texts_block}"""
    return _render_subtemplate(body, subs)


def _section_factor_loadings(active: bool, subs: dict) -> str:
    if not active:
        return ""
    body = """

## Factor loadings (`factor_loadings`)

Output keys for ALL items 1–${n_items} across factors ${factor_range_short} (`${loading_first_key}` through `${loading_last_key}`).

Fill values:
- Loading explicitly shown for an item-factor cell → record as a number (negative allowed).
- Cell suppressed/blank/below-threshold within the reported solution → `0.0` (not null — the cell exists, the value is just below threshold).
- Entire factor does NOT exist in the chosen solution (e.g. ${nonexistent_factors_example}) → set ALL ${nonexistent_factor_glob_example} to `null`.
- Table genuinely cut off / not visible → `null` AND explain in `notes`.
- Table reports only primary loadings (one factor per item) → enter the reported loading in that factor's key and set the other factors for that item to `0.0` (for factors that exist in the solution).

**Do not impute or estimate values.**

### Multi-level / spanning headers

Tables may use multi-level headers (super-headings spanning multiple columns, group labels above factor columns, factor names as subheaders). Read the LOWEST-LEVEL header row that directly labels the numeric loading columns. Treat super-headings as structural grouping only; do not misinterpret them as separate variables.

If the header structure is too ambiguous to recover column identities, extract row-wise as best as possible and explain the ambiguity in `notes`.

### CFA-only fallback for the standard 3-factor structure

If the results stem EXCLUSIVELY from a CFA with three factors AND the table doesn't explicitly label item→factor, you MAY assume the standard ${instrument_name} three-factor item assignment:
${cfa_assignment_block}

Use this ONLY to decide where to place a single reported loading per item when factor identity isn't otherwise recoverable. For CFA models with MORE or FEWER than three factors, do NOT assume any a priori item-to-factor assignment."""
    return _render_subtemplate(body, subs)


def _section_factor_correlations(subs: dict) -> str:
    body = """

## Factor correlations (`factor_correlations`)

Output ALL ${n_corr_pairs} unique factor-pair correlations: ${correlations_key_list}. Use the `R<i>.<j>` form with **i < j only** — do NOT duplicate symmetric pairs.

Look in the results section for keywords near each other: `correl*` (correlated/correlation), `factors`, numeric values in [-1, 1], possibly an `r` indicator. Correlations may appear as either (a) the lower/upper triangle with `nfac × (nfac-1) / 2` unique correlations, or (b) the full symmetric correlation matrix. Either form is fine — only output the i<j pairs.

Fill values:
- Correlation explicitly reported → number (negative allowed; typically in [-1, 1]).
- Correlation involves a factor that doesn't exist in this solution (e.g. ${nonexistent_correlation_example}) → `null`.
- Correlation simply not reported / suppressed / not visible → `null`.
- ⚠️ **Orthogonal-rotation special case**: if the rotation method is one of `varimax`, `quartimax`, `equamax`, `orthomax`, `parsimax` (orthogonal — factors are uncorrelated by construction), set ALL existing-factor correlations to `0` (not `null`). Use `null` only if the value is missing/unreported, not when it's structurally zero."""
    return _render_subtemplate(body, subs)


def _section_correlation_matrix(p: dict) -> str:
    """The variable × variable correlation matrix typical of general
    MASEM (Big Five etc.) — what the model should extract, in what
    shape, and how to handle missing cells."""
    variables = p.get("variables") or []
    var_lines = _render_variables_block(variables) or "(no variables defined — extract any reported correlation matrix)"
    body = """

## Correlation matrix (`correlation_matrix`)

Extract the variable × variable correlation matrix(es) reported in the paper.  Output the matrix using the explicit `_table` marker so the viewer renders it as a real HTML table:

```json
"correlation_matrix": {
  "_table": [
    {"variable": "<row_var>", "<col_var_1>": <r>, "<col_var_2>": <r>, ...},
    {"variable": "<row_var>", "<col_var_1>": <r>, "<col_var_2>": <r>, ...},
    ...
  ]
}
```

Variables / scales / constructs to look for in this paper:
${variables_block}

Rules:
- One row per variable; the row's `"variable"` key is the variable's name.  Other keys are the variable names of the columns; their values are the reported correlation coefficients (number in [-1, 1]).
- Diagonal cells (variable correlated with itself) → `1.0`.
- Cells reported only on the lower or upper triangle → fill the symmetric side with the same value (it's a symmetric matrix).
- Suppressed / blank cells → `null`.
- Variables in the paper that don't match any of the listed names (and aren't synonyms) → drop them from the table; do not invent a column for them.
- If the paper reports multiple matrices (e.g. for sub-samples), output each as its own object with a separate `correlation_matrix` row in `samples[]` (i.e., split into multiple sample entries)."""
    return _render_subtemplate(body, {"variables_block": var_lines})


def _section_single_correlations(p: dict) -> str:
    """Single correlations reported in prose ("the correlation between
    Extraversion and Conscientiousness was r = -.12")."""
    variables = p.get("variables") or []
    var_lines = _render_variables_block(variables) or "(no variables defined — extract any reported pairwise correlation)"
    body = """

## Single correlations from prose (`single_correlations`)

Extract pairwise correlations reported as inline text (not as a matrix).  Output one row per reported correlation using the `_table` marker:

```json
"single_correlations": {
  "_table": [
    {"var1": "<name>", "var2": "<name>", "r": <number>, "n": <int|null>, "p": <number|null>, "page": <int|null>},
    ...
  ]
}
```

Variables / scales / constructs to look for in this paper:
${variables_block}

Rules:
- Each row records ONE pairwise correlation: `var1`, `var2` (string names), `r` (the correlation coefficient as a number), and — when reported — `n` (sample size for that correlation), `p` (p-value), `page` (1-indexed PDF page where the correlation was stated).
- If both `var1` and `var2` are listed variables (or recognisable synonyms), include the correlation.  If either endpoint is unlisted, drop it.
- Don't include correlations that already appear in `correlation_matrix` for this sample — the matrix takes precedence.
- For each correlation extracted, the supporting evidence's `field` should be `"samples[i].single_correlations._table[j]"`."""
    return _render_subtemplate(body, {"variables_block": var_lines})


def _section_effect_sizes(p: dict) -> str:
    """Unified ``effect_sizes`` table — one row per reported effect
    size, whether the source was a matrix cell or prose.  Schema:
    ``es_id``, ``var1``, ``var2``, ``desc1``, ``desc2``, ``es``,
    ``type``.  This is the Direct-information schema downstream
    meta-analytic tools consume."""
    variables = p.get("variables") or []
    var_lines = _render_variables_block(variables) or "(no variables defined — extract every reported pairwise effect size)"
    body = """

## Effect sizes (`effect_sizes`)

Extract every reported pairwise effect size — correlations from prose, correlations from matrices, and any other bivariate effect-size statistic — into ONE unified table.  Output uses the `_table` marker so the viewer renders it as a real HTML table:

```json
"effect_sizes": {
  "_table": [
    {"es_id": 1, "var1": "<canonical>", "var2": "<canonical>", "desc1": "<verbatim wording>", "desc2": "<verbatim wording>", "es": <number>, "type": "<r|d|OR|...>"},
    ...
  ]
}
```

Variables / scales / constructs to look for in this paper:
${variables_block}

Rules:
- One row per pairwise effect size.  `es_id` is a 1-indexed sequential integer within this sample.
- `var1`, `var2` are the canonical SHORT names from the list above (e.g. `"vg"`, `"bm"`).  If the paper uses a synonym, map it to the canonical short name.  If either endpoint isn't listed (and isn't a recognisable synonym), drop the row.
- `desc1`, `desc2` are the VERBATIM wording the paper uses for each variable (e.g. `"Length of video game play during one sitting"`, `"Body mass index (BMI)"`).  Keep them as the paper printed them — this is the audit trail.
- `es` is the numeric effect-size value (e.g. `0.27`).
- `type` is the effect-size kind: `"r"` for a Pearson correlation, `"d"` for Cohen's d, `"OR"` for an odds ratio, etc.  Use `"r"` as the default for correlation matrices and prose-reported r-values.
- Reporting source — extract whichever form the paper provides:
  - **Matrix cells**: one row per unique off-diagonal pairing (upper or lower triangle, not both).  Drop the diagonal (variable correlated with itself).  Suppressed / blank cells produce no row.
  - **Prose** ("the correlation between X and Y was r = .42"): one row per reported correlation.
  - **Sub-sample matrices** (e.g. multiple groups): emit each as its own `samples[]` entry rather than collapsing them.
- For each row extracted, the supporting evidence's `field` should be `"samples[i].effect_sizes._table[j]"`."""
    return _render_subtemplate(body, {"variables_block": var_lines})


def _render_variables_block(variables: list) -> str:
    """Format the user's ``variables`` parameter as a bullet list with
    optional definitions and synonyms."""
    if not variables:
        return ""
    lines: list[str] = []
    for v in variables:
        if not isinstance(v, dict):
            continue
        name = (v.get("name") or "").strip()
        if not name:
            continue
        definition = (v.get("definition") or "").strip()
        synonyms = v.get("synonyms") or []
        line = f"- **{name}**"
        if definition:
            line += f" — {definition}"
        if synonyms:
            syn_str = ", ".join(f'"{s}"' for s in synonyms)
            line += f"  (synonyms: {syn_str})"
        lines.append(line)
    return "\n".join(lines)


def _section_multiple_models() -> str:
    """The "## Multiple models" section is factor-loadings-specific
    (rotations / EFA-vs-CFA terminology).  Skipped when the preset
    doesn't extract factor loadings."""
    return """

## Multiple models / rotations / methods

If the paper reports MULTIPLE factor solutions for the same sample (EFA + CFA, rotated + unrotated, 2-factor + 3-factor, Model A + Model B):
- Extract the solution with the HIGHEST number of factors.
- Tied factor counts → prefer the one labelled main / final / preferred.
- Still tied → prefer EFA over CFA, and oblique over orthogonal rotation.
- Note the chosen solution and any alternatives in `notes`."""


def _section_res_field(instrument_name: str) -> str:
    """The ``res`` (Likert response options) metadata bullet — only
    meaningful when extracting factor loadings, since it's a property
    of the instrument's items."""
    return (
        f"- **`res`** — Number of Likert response options for the {instrument_name} "
        f"(integer ≥ 5). Standard {instrument_name} uses 5; if no adaptation/translation "
        f"note exists, default to `5`. Else null."
    )


def _render_schema_invariants_line(fl: bool, fc: bool, cmat: bool, scor: bool, es: bool, subs: dict) -> str:
    """One-liner that follows the JSON schema, telling the model which
    keys are mandatory.  Adapts to whichever data sources are active."""
    parts = []
    if fl:
        parts.append(
            f"all {subs['n_loading_keys']} loading keys "
            f"(`{subs['loading_first_key']}`…`{subs['loading_last_key']}`)"
        )
    if fc:
        parts.append(
            f"all {subs['n_corr_pairs']} correlation keys "
            f"(`{subs['correlation_first_key']}`…`{subs['correlation_last_key']}`)"
        )
    if not parts:
        if cmat or scor or es:
            return "Include the metadata block on every sample, even when individual fields are null."
        return ""
    if len(parts) == 1:
        return f"You MUST include {parts[0]} on every sample, even if the values are null."
    return f"You MUST include {' and '.join(parts)} on every sample, even if the values are null."


def _render_goal_items(fl: bool, fc: bool, cmat: bool, scor: bool, es: bool, subs: dict) -> str:
    """The numbered list under "## Goal".  Always ends with the metadata
    bullet; data-source items appear in source-priority order."""
    items: list[str] = []
    if fl:
        items.append(
            f"**Factor loadings** — item-level standardised factor loadings for items "
            f"1–{subs['n_items']}, allowing up to {subs['n_factors_word']} factors "
            f"({subs['factor_range_short']})."
        )
    if fc:
        items.append(
            f"**Factor correlations** — inter-factor (latent) correlations among up to "
            f"{subs['n_factors']} factors (the upper-triangle: {subs['n_corr_pairs']} "
            f"unique pairs)."
        )
    if cmat:
        items.append(
            "**Variable correlation matrix** — the reported correlation matrix(es) "
            "between named variables / scales / constructs in this paper."
        )
    if scor:
        items.append(
            "**Single correlations from prose** — pairwise correlations reported as "
            "inline text rather than in a matrix (e.g. \"the correlation between X "
            "and Y was r = .42\")."
        )
    if es:
        items.append(
            "**Effect sizes** — every reported pairwise effect size (correlations from "
            "matrices and prose, plus other bivariate effect-size statistics) collected "
            "into one unified table with `es_id, var1, var2, desc1, desc2, es, type`."
        )
    items.append(
        "**Study + sample metadata** — coded according to a fixed scheme so the "
        "records are usable in a downstream meta-analytic database."
    )
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _render_full_json_schema(fl: bool, fc: bool, cmat: bool, scor: bool, es: bool, subs: dict) -> str:
    """Assemble the JSON-schema block from data-source-specific fragments.
    Indentation matches the original TAS-20 fragment so byte-identity
    holds when the standard TAS-20 sources are active."""
    lines: list[str] = ['{', '  "samples": [', '    {', '      "sample_id": "string",', '']
    if fl:
        # Inserted as-is — already starts with proper indentation.
        lines.extend(_render_factor_loadings_schema(
            int(subs.get("n_factors") or 0),
            int(subs.get("n_items") or 0),
        ).splitlines())
        lines.append("")
    if fc:
        lines.extend(_render_factor_correlations_schema(
            int(subs.get("n_factors") or 0),
        ).splitlines())
        lines.append("")
    if cmat:
        lines.extend([
            '      "correlation_matrix": {',
            '        "_table": [',
            '          {"variable": "<row_var>", "<col_var_1>": number, "<col_var_2>": number}',
            '        ]',
            '      },',
            '',
        ])
    if scor:
        lines.extend([
            '      "single_correlations": {',
            '        "_table": [',
            '          {"var1": "<name>", "var2": "<name>", "r": number, "n": integer|null, "p": number|null, "page": integer|null}',
            '        ]',
            '      },',
            '',
        ])
    if es:
        lines.extend([
            '      "effect_sizes": {',
            '        "_table": [',
            '          {"es_id": integer, "var1": "<canonical>", "var2": "<canonical>", "desc1": "<verbatim>", "desc2": "<verbatim>", "es": number, "type": "<r|d|OR|...>"}',
            '        ]',
            '      },',
            '',
        ])
    # Metadata fields (always present).  ``res`` is gated on fl.
    lines.extend([
        '      "pubyear":   number|null,',
        '      "country":   "string|null",',
        '      "continent": "string|null",',
        '      "lang":      "string|null",',
        '      "pubtype":   "1|2|3|4|5|null",',
        '      "n":         "integer|null",',
        '      "sex":       "number 0..100 |null",',
        '      "age":       "number|null",',
        '      "clinical":  "0|1|2|null",',
    ])
    if fl:
        lines.append('      "res":       "integer >= 5 |null",')
    lines.extend([
        '      "nfac":      "integer 1..10 |null",',
        '      "cfa":       "0|1|null",',
        '      "met":       "1|2|3|4|null",',
        '      "rot":       "1|2|null",',
        '',
        '      "notes": "string"',
        '    }',
        '  ],',
    ])
    # Pick a representative ``field`` value for the evidence example —
    # whichever data source is the headline of this preset.
    if fl:
        evidence_field = "samples[0].factor_loadings"
    elif es:
        evidence_field = "samples[0].effect_sizes._table[0]"
    elif cmat:
        evidence_field = "samples[0].correlation_matrix"
    elif scor:
        evidence_field = "samples[0].single_correlations"
    else:
        evidence_field = "samples[0]"
    lines.extend([
        '  "evidence": [',
        f'    {{"snippet": "...", "page": 3, "source": "Table 1", "field": "{evidence_field}"}}',
        '  ]',
        '}',
    ])
    return "\n".join(lines)


def _load_one(path: Path) -> dict[str, Any] | None:
    """Parse a single preset JSON. Returns None if invalid."""
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"[presets] skipping {path.name}: {e}",
              file=sys.stderr, flush=True)
        return None
    if not isinstance(meta, dict):
        print(f"[presets] {path.name}: top-level value must be an object",
              file=sys.stderr, flush=True)
        return None
    missing = [k for k in _REQUIRED_KEYS if not meta.get(k)]
    if missing:
        print(f"[presets] {path.name}: missing required keys {missing}",
              file=sys.stderr, flush=True)
        return None
    if not any(meta.get(k) for k in _PROMPT_KEYS):
        print(f"[presets] {path.name}: must define 'prompt' or 'prompt_file'",
              file=sys.stderr, flush=True)
        return None

    # Inline the prompt body so callers never have to think about file split
    body = _read_prompt_body(meta, path)
    if not body:
        print(f"[presets] {path.name}: prompt body is empty",
              file=sys.stderr, flush=True)
        return None
    meta["prompt"] = body
    meta.pop("prompt_file", None)
    meta.pop("prompt_template_file", None)

    # Auto-generate ``sub_views`` from ``template_params.data_sources``
    # when the preset doesn't declare its own.  This keeps the variant
    # starter presets concise — they only need to set data_sources and
    # the corresponding result-panel tabs follow automatically.  Presets
    # that want a custom layout still win by declaring ``sub_views``
    # explicitly in the JSON.
    if "sub_views" not in meta:
        params = meta.get("template_params") or {}
        sources = params.get("data_sources") or []
        if sources:
            meta["sub_views"] = _build_sub_views_from_sources(sources)
    return meta


# Mapping data_source key → sub-view config.  Adding a new data source
# requires extending this map AND adding the corresponding section
# helper above (and the schema fragment in _render_full_json_schema).
_SUB_VIEW_SPECS = {
    "factor_loadings": {
        "id":            "loadings",
        "label":         "Factor loadings",
        "include_keys":  ["sample_id", "n", "factor_loadings"],
        "evidence_keys": ["factor_loadings"],
    },
    "factor_correlations": {
        "id":            "correlations",
        "label":         "Factor correlations",
        "include_keys":  ["sample_id", "n", "factor_correlations"],
        "evidence_keys": ["factor_correlations"],
    },
    "correlation_matrix": {
        "id":            "corrmatrix",
        "label":         "Correlation matrix",
        "include_keys":  ["sample_id", "n", "correlation_matrix"],
        "evidence_keys": ["correlation_matrix"],
    },
    "single_correlations": {
        "id":            "singlecorrs",
        "label":         "Single correlations",
        "include_keys":  ["sample_id", "n", "single_correlations"],
        "evidence_keys": ["single_correlations"],
    },
    "effect_sizes": {
        "id":            "effectsizes",
        "label":         "Effect sizes",
        "include_keys":  ["sample_id", "n", "effect_sizes"],
        "evidence_keys": ["effect_sizes"],
    },
}


def _build_sub_views_from_sources(sources: list[str]) -> list[dict]:
    """Generate sub_views from a list of data sources.  The result-panel
    gets one tab per data source plus a final ``Descriptives`` tab that
    excludes every data-source key (so it shows only the metadata)."""
    sub_views: list[dict] = []
    excludes: list[str] = []
    for src in sources:
        spec = _SUB_VIEW_SPECS.get(src)
        if not spec:
            continue
        sub_views.append(dict(spec))   # copy so callers can mutate freely
        # Track the data-source-bearing key for the descriptives exclude list.
        excludes.append(src)
    if sub_views:
        sub_views.append({
            "id":           "descriptives",
            "label":        "Descriptives",
            "exclude_keys": excludes,
        })
    return sub_views


def load_all() -> dict[str, dict[str, Any]]:
    """Discover and return every valid preset, keyed by id.  Cheap enough
    to call per request — typical deployments have a handful of files."""
    out: dict[str, dict[str, Any]] = {}
    if not PRESETS_DIR.is_dir():
        return out
    for path in sorted(PRESETS_DIR.glob("*.json")):
        meta = _load_one(path)
        if meta is None:
            continue
        pid = meta["id"]
        if pid in out:
            print(f"[presets] duplicate id {pid!r} in {path.name}; ignoring",
                  file=sys.stderr, flush=True)
            continue
        out[pid] = meta
    return out


def get(preset_id: str) -> dict[str, Any] | None:
    """Return one preset by id, or None if not found / invalid."""
    return load_all().get(preset_id)


def read_template_for(preset_id: str) -> str | None:
    """Return the raw (unrendered) template body for ``preset_id``, or
    ``None`` when the preset doesn't ship a ``prompt_template_file``.
    Used by the build-preset-prompt route to re-render with user params."""
    if not PRESETS_DIR.is_dir():
        return None
    for path in sorted(PRESETS_DIR.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or meta.get("id") != preset_id:
            continue
        tpl_file = meta.get("prompt_template_file")
        if not tpl_file:
            return None
        return _read_sibling(tpl_file, path)
    return None


# Public alias so the server route can re-use the renderer without
# reaching into the underscore-prefixed implementation detail.
render_template = _render_template
build_sub_views = _build_sub_views_from_sources


def list_summaries(include_hidden: bool = False) -> list[dict[str, Any]]:
    """Lightweight list for the workflows menu — no prompt body included.
    Presets that set ``landing_hidden: true`` are skipped by default;
    the builder still loads them by id (they're sub-preset / example
    starters reachable from inside the parent preset's UI)."""
    out = []
    for preset in load_all().values():
        if not include_hidden and preset.get("landing_hidden"):
            continue
        out.append({
            "id":           preset["id"],
            "title":        preset["title"],
            "tagline":      preset.get("tagline", ""),
            "description":  preset.get("description", ""),
            "accent_color": preset.get("accent_color"),
        })
    return out
