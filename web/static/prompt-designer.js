/* Structured prompt designer (step 3, "Generate with AI" path).
 *
 * Captures a hierarchical declaration (unit + chunks + paper-metadata +
 * context) and produces:
 *   1. A meta-prompt input — written into questionInput / contextInput
 *      so the existing /api/generate-prompt pipeline picks it up
 *      unchanged.
 *   2. A preset descriptor (Phase B) — saved to localStorage under
 *      paperlens.userPresets so the resulting extraction renders with
 *      sub-tabs.
 *
 * Form state autosaves to localStorage on every input change.  All
 * data-model helpers (readForm, writeForm, addChunk, etc.) are pure
 * functions over the DOM so they can be unit-tested without mocking
 * a full session.
 *
 * Public surface (exposed on window for inline onclicks):
 *   - designerAddChunk()       — append a new empty chunk
 *   - designerRemoveChunk(id)  — remove a chunk by render id
 *   - designerSubmit()         — validate, derive prompt + preset,
 *                                proceed through existing pipeline
 */

const _DESIGNER_AUTO_SAVE_KEY = 'paperlens.designerDraft.v1';

/* In-memory representation that mirrors the form's data model.
 * One chunk per sub-tab in the eventual review UI.  ``_renderId`` is a
 * client-only counter used to key DOM nodes when re-rendering.
 */
let _designerState = {
  unit: {
    name:            '',
    cardinality:     'many',
    sidebar_label:   '',
    identifier_field: '',
  },
  chunks: [],                       // [{_renderId, name, description, display, fields}]
  paper_metadata_text: '',          // raw textarea contents; parsed at submit
  context: '',
};

let _designerChunkCounter = 0;

/* ── DOM read / write ────────────────────────────────────────────── */

function _designerReadFromDOM() {
  const get = id => (document.getElementById(id)?.value ?? '').trim();
  const card = state => state.chunks.map(c => ({
    _renderId:    c._renderId,
    name:         get(`designerChunk_${c._renderId}_name`),
    description:  get(`designerChunk_${c._renderId}_desc`),
    display:      _designerReadChunkDisplay(c._renderId),
    fields:       get(`designerChunk_${c._renderId}_fields`),
  }));

  _designerState = {
    unit: {
      name:             get('designerUnitName'),
      cardinality:      (document.querySelector('input[name="designerUnitCardinality"]:checked')?.value || 'many'),
      sidebar_label:    get('designerSidebarLabel'),
      identifier_field: get('designerIdField'),
    },
    chunks:              card(_designerState),
    paper_metadata_text: get('designerPaperFields'),
    context:             get('designerContext'),
  };
  return _designerState;
}

function _designerReadChunkDisplay(renderId) {
  const r = document.querySelector(
    `input[name="designerChunk_${renderId}_display"]:checked`);
  return r?.value || 'table';
}

function _designerWriteToDOM() {
  // One-shot restore from autosave.  Idempotent — wraps null checks
  // because step 3 may not be in the DOM yet on first visit.
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = v ?? '';
  };
  set('designerUnitName',     _designerState.unit.name);
  set('designerSidebarLabel', _designerState.unit.sidebar_label);
  set('designerIdField',      _designerState.unit.identifier_field);
  set('designerPaperFields',  _designerState.paper_metadata_text);
  set('designerContext',      _designerState.context);
  const radio = document.querySelector(
    `input[name="designerUnitCardinality"][value="${_designerState.unit.cardinality}"]`);
  if (radio) radio.checked = true;

  // Re-render chunks
  _designerRenderChunks();
}

/* ── Chunk rendering ─────────────────────────────────────────────── */

const _DESIGNER_DISPLAY_TYPES = [
  {value: 'table',     label: 'Table',     hint: 'Sortable rows; repeating fields'},
  {value: 'list',      label: 'List',      hint: 'Bulleted list of items'},
  {value: 'key-value', label: 'Key-value', hint: 'Label / value pairs'},
  {value: 'prose',     label: 'Prose',     hint: 'Free-form text'},
];

function _designerRenderChunks() {
  const host = document.getElementById('designerChunks');
  if (!host) return;
  if (!_designerState.chunks.length) {
    host.innerHTML = `<p class="designer-hint" style="font-style:italic">No chunks yet &mdash; click "Add chunk" to start.</p>`;
    return;
  }
  host.innerHTML = _designerState.chunks.map((c, i) =>
    _designerRenderOneChunk(c, i)).join('');
  // After re-rendering, bind autosave listeners on every input/textarea
  // inside the host (re-binding is idempotent because we use a flag).
  host.querySelectorAll('input, textarea').forEach(el => {
    if (el._designerBound) return;
    el._designerBound = true;
    el.addEventListener('input', _designerAutoSave);
    el.addEventListener('change', _designerAutoSave);
  });
}

function _designerRenderOneChunk(c, idx) {
  const rid    = c._renderId;
  const radios = _DESIGNER_DISPLAY_TYPES.map(t => `
    <label title="${t.hint}">
      <input type="radio" name="designerChunk_${rid}_display"
             value="${t.value}" ${c.display === t.value ? 'checked' : ''} />
      ${t.label}
    </label>
  `).join('');

  return `
    <div class="designer-chunk-card" data-chunk-rid="${rid}">
      <div class="designer-chunk-card-head">
        <span class="designer-chunk-num">Chunk ${idx + 1}</span>
        <button type="button" class="designer-chunk-remove"
                onclick="designerRemoveChunk(${rid})">Remove</button>
      </div>

      <label class="designer-field">
        <span class="designer-field-label">Name</span>
        <input id="designerChunk_${rid}_name" type="text"
               value="${_escAttr(c.name)}"
               placeholder="e.g. Effect size" autocomplete="off" />
        <span class="designer-field-hint">Becomes the sub-tab title in the review panel.</span>
      </label>

      <label class="designer-field">
        <span class="designer-field-label">Description <span class="optional">(optional)</span></span>
        <input id="designerChunk_${rid}_desc" type="text"
               value="${_escAttr(c.description)}"
               placeholder="One line summary" autocomplete="off" />
      </label>

      <fieldset class="designer-field designer-radioset designer-display-radioset">
        <legend class="designer-field-label">Display as</legend>
        ${radios}
      </fieldset>

      <label class="designer-field">
        <span class="designer-field-label">Fields</span>
        <textarea id="designerChunk_${rid}_fields" rows="5"
                  placeholder="One per line, <name>: <description>
e.g.
metric: the named outcome
value: the reported coefficient (number)
ci_low: lower 95% CI bound (number, null)">${_escAttr(c.fields)}</textarea>
        <span class="designer-field-hint">Each line: <code>field_name: description</code></span>
      </label>
    </div>
  `;
}

function _escAttr(s) {
  return String(s ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;')
                        .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ── Add / remove chunks ─────────────────────────────────────────── */

function designerAddChunk() {
  _designerReadFromDOM();
  _designerChunkCounter += 1;
  _designerState.chunks.push({
    _renderId:    _designerChunkCounter,
    name:         '',
    description:  '',
    display:      'table',
    fields:       '',
  });
  _designerRenderChunks();
  _designerAutoSave();
}

function designerRemoveChunk(renderId) {
  _designerReadFromDOM();
  _designerState.chunks = _designerState.chunks.filter(c => c._renderId !== renderId);
  _designerRenderChunks();
  _designerAutoSave();
}

/* ── Autosave / restore ──────────────────────────────────────────── */

function _designerAutoSave() {
  try {
    _designerReadFromDOM();
    localStorage.setItem(_DESIGNER_AUTO_SAVE_KEY,
                         JSON.stringify(_designerState));
  } catch (_) { /* localStorage disabled — silently no-op */ }
}

function _designerAutoRestore() {
  try {
    const raw = localStorage.getItem(_DESIGNER_AUTO_SAVE_KEY);
    if (!raw) return;
    const snapshot = JSON.parse(raw);
    if (!snapshot || typeof snapshot !== 'object') return;
    // Re-key renderIds so we don't collide with the in-memory counter.
    const restored = snapshot.chunks || [];
    _designerState = {
      unit: {
        name:             snapshot.unit?.name             ?? '',
        cardinality:      snapshot.unit?.cardinality      || 'many',
        sidebar_label:    snapshot.unit?.sidebar_label    ?? '',
        identifier_field: snapshot.unit?.identifier_field ?? '',
      },
      chunks: restored.map(c => {
        _designerChunkCounter += 1;
        return {
          _renderId:   _designerChunkCounter,
          name:        c.name        ?? '',
          description: c.description ?? '',
          display:     c.display     || 'table',
          fields:      c.fields      ?? '',
        };
      }),
      paper_metadata_text: snapshot.paper_metadata_text ?? '',
      context:             snapshot.context             ?? '',
    };
  } catch (_) { /* malformed snapshot — start fresh */ }
}

/* ── Field-block parsing ─────────────────────────────────────────── */

/* Parse a textarea of <name>: <description> lines into a list of
 * field objects.  Tolerates blank lines, leading bullets, and lines
 * without colons (treated as field name with empty description).
 * Returns: [{name, description}]. */
function _designerParseFields(text) {
  if (!text) return [];
  return String(text).split(/\r?\n/)
    .map(l => l.replace(/^[\s\-•\*]+/, '').trim())
    .filter(Boolean)
    .map(l => {
      const i = l.indexOf(':');
      if (i === -1) return {name: l, description: ''};
      return {
        name:        l.slice(0, i).trim(),
        description: l.slice(i + 1).trim(),
      };
    })
    .filter(f => f.name);
}

/* ── Submit / validation ────────────────────────────────────────── */

function _designerValidate(decl) {
  const errs = [];
  if (!decl.unit.name) {
    errs.push('Give the unit a name (e.g. "finding", "sample", "table").');
  }
  if (!decl.chunks.length) {
    errs.push('Add at least one information chunk.');
  }
  decl.chunks.forEach((c, i) => {
    if (!c.name) {
      errs.push(`Chunk ${i + 1} is missing a name.`);
    }
    if (!c.fields_parsed.length) {
      errs.push(`Chunk "${c.name || i + 1}" has no fields.`);
    }
  });
  return errs;
}

/* Resolve the form state into a clean declaration object — what the
 * meta-prompt and the preset-builder both consume.  No DOM access. */
function designerBuildDeclaration() {
  _designerReadFromDOM();
  const decl = {
    unit: {..._designerState.unit},
    chunks: _designerState.chunks.map(c => ({
      name:           c.name,
      description:    c.description,
      display:        c.display,
      fields_text:    c.fields,
      fields_parsed:  _designerParseFields(c.fields),
    })),
    paper_metadata_fields: _designerParseFields(_designerState.paper_metadata_text),
    context:               _designerState.context,
  };
  return decl;
}

async function designerSubmit() {
  const decl = designerBuildDeclaration();
  const errs = _designerValidate(decl);
  if (errs.length) {
    showToast(errs[0]);
    return;
  }
  // Hand off to the Phase B pipeline — assembles the meta-prompt
  // question/context, saves the preset to localStorage, then triggers
  // the existing /api/generate-prompt flow via submitStep3.
  if (typeof designerProcessDeclaration === 'function') {
    await designerProcessDeclaration(decl);
  } else {
    // Phase A standalone — temporarily route through the legacy text path.
    document.getElementById('questionInput').value = _designerLegacyQuestion(decl);
    document.getElementById('contextInput').value  = decl.context || '';
    if (typeof submitStep3 === 'function') submitStep3();
  }
}

/* Temporary stub used until Phase B's designerProcessDeclaration lands.
 * Produces a plain-English summary of the declaration as the "question"
 * input, so the existing /api/generate-prompt path still works during
 * Phase A development.  Replaced wholesale in Phase B. */
function _designerLegacyQuestion(decl) {
  const lines = [];
  lines.push(`Extract one entry per ${decl.unit.name} from each paper.`);
  lines.push(`Each entry should be grouped into the following information chunks:`);
  decl.chunks.forEach(c => {
    lines.push(`  - ${c.name} (${c.display}): ${c.description || ''}`);
    c.fields_parsed.forEach(f => {
      lines.push(`      ${f.name}: ${f.description || ''}`);
    });
  });
  if (decl.paper_metadata_fields.length) {
    lines.push(`Additionally extract paper-level metadata:`);
    decl.paper_metadata_fields.forEach(f => {
      lines.push(`  - ${f.name}: ${f.description || ''}`);
    });
  }
  return lines.join('\n');
}

/* ── Init ────────────────────────────────────────────────────────── */

function _designerInit() {
  _designerAutoRestore();
  // If the restore added chunks, render them.  Otherwise start with
  // one empty chunk so the user has somewhere to type immediately.
  if (!_designerState.chunks.length) {
    _designerChunkCounter += 1;
    _designerState.chunks.push({
      _renderId:   _designerChunkCounter,
      name:        '',
      description: '',
      display:     'table',
      fields:      '',
    });
  }
  _designerWriteToDOM();
  // Bind autosave on the static (non-chunk) inputs.
  ['designerUnitName', 'designerSidebarLabel', 'designerIdField',
   'designerPaperFields', 'designerContext'].forEach(id => {
    const el = document.getElementById(id);
    if (el && !el._designerBound) {
      el._designerBound = true;
      el.addEventListener('input',  _designerAutoSave);
      el.addEventListener('change', _designerAutoSave);
    }
  });
  document.querySelectorAll('input[name="designerUnitCardinality"]')
    .forEach(el => {
      if (el._designerBound) return;
      el._designerBound = true;
      el.addEventListener('change', _designerAutoSave);
    });
}

/* ── Phase B: declaration → meta-prompt input + preset descriptor ─── */

const _USER_PRESETS_KEY = 'paperlens.userPresets.v1';

function _designerSlugify(s) {
  return String(s || '').toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    || 'unnamed';
}

function _designerTitleize(s) {
  const t = String(s || '').trim();
  if (!t) return 'Custom extraction';
  return t.charAt(0).toUpperCase() + t.slice(1) + ' extraction';
}

/* Assemble the question + context inputs the existing /api/generate-prompt
 * pipeline expects.  We write everything into ``question`` because that
 * field is what the meta-prompt's "research question / task description"
 * block consumes verbatim; ``context`` is the user's free-form notes. */
function _designerBuildMetaPromptInput(decl) {
  const lines = [];
  const unitName = decl.unit.name || 'entry';

  lines.push(`Extract one entry per ${unitName} from each paper.`);
  if (decl.unit.cardinality === 'many') {
    lines.push(`A paper can contain multiple ${unitName}s; emit one entry per ${unitName}.`);
  } else {
    lines.push(`Each paper has exactly one ${unitName}; emit a single entry.`);
  }
  if (decl.unit.identifier_field) {
    lines.push(`Use the ${decl.unit.identifier_field} field as the sample_id for each entry.`);
  }
  if (decl.unit.sidebar_label) {
    lines.push(`The sample_id should follow this format: ${decl.unit.sidebar_label}`);
  }
  lines.push('');

  lines.push(`Each entry must group its fields into the following named sub-objects (one per review-panel tab):`);
  decl.chunks.forEach(c => {
    const id = _designerSlugify(c.name);
    const tableHint = (c.display === 'table')
      ? ' — render as a table using the {"_table": [...]} marker'
      : (c.display === 'list')
        ? ' — render as a JSON array'
        : (c.display === 'key-value')
          ? ' — render as a JSON object with named fields'
          : ' — render as a JSON string (markdown prose)';
    lines.push('');
    lines.push(`  Sub-object "${id}" ("${c.name}")${tableHint}`);
    if (c.description) lines.push(`    ${c.description}`);
    lines.push(`    Required fields:`);
    c.fields_parsed.forEach(f => {
      lines.push(`      - ${f.name}: ${f.description || '(see notes)'}`);
    });
  });
  lines.push('');

  // Paper metadata block — always asks for the four canonical fields,
  // plus anything the user added.  Lives at the paper-level top-level
  // ``paper_metadata`` key, not inside per-entry sub-objects.
  lines.push(`Paper-level metadata (top-level "paper_metadata" object, extracted once per paper):`);
  lines.push(`  - title: full paper title verbatim (required, never empty)`);
  lines.push(`  - doi: DOI string or null`);
  lines.push(`  - year: publication year as integer or null`);
  lines.push(`  - authors: array of author names in printed order, or null`);
  decl.paper_metadata_fields.forEach(f => {
    lines.push(`  - ${f.name}: ${f.description || '(see notes)'}`);
  });
  lines.push('');

  // Canonical output shape — the meta-prompt already demands evidence
  // + extraction_confidence, but spelling out the per-entry shape here
  // makes the LLM's job mechanical instead of inferential.
  lines.push(`Canonical output JSON shape:`);
  lines.push(`{`);
  lines.push(`  "paper_metadata": { title, doi, year, authors, ... },`);
  lines.push(`  "samples": [`);
  lines.push(`    {`);
  lines.push(`      "sample_id": "<derived from ${decl.unit.identifier_field || 'unit identifier'}>",`);
  decl.chunks.forEach(c => {
    const id = _designerSlugify(c.name);
    if (c.display === 'table') {
      lines.push(`      "${id}": {"_table": [{...field values per row...}, ...]},`);
    } else if (c.display === 'list') {
      lines.push(`      "${id}": [...],`);
    } else {
      lines.push(`      "${id}": {...},`);
    }
  });
  lines.push(`    }`);
  lines.push(`  ],`);
  lines.push(`  "evidence": [ {"snippet", "page", "source", "field"}, ... ],`);
  lines.push(`  "extraction_confidence": {`);
  lines.push(`    "paper_metadata":  {"level": "high|medium|low", "notes": "..."},`);
  decl.chunks.forEach(c => {
    const id = _designerSlugify(c.name);
    lines.push(`    "${id}":  {"level": "high|medium|low", "notes": "..."},`);
  });
  lines.push(`  }`);
  lines.push(`}`);
  lines.push('');
  lines.push(`Evidence field paths must mirror the JSON structure — e.g. "samples[0].<sub-object>.<field>".`);

  return {
    question: lines.join('\n'),
    context:  decl.context || '',
  };
}

/* Build a MetaPaperLens preset descriptor from the declaration.  The
 * shape mirrors ``web/presets/<id>.json`` so the same renderer / loader
 * code can consume it without branching.  Stored client-side only —
 * the descriptor is set on ``state.activePreset`` for the live session
 * and saved to localStorage so it persists across reloads.
 *
 * Each chunk becomes one sub-view; evidence_keys include the chunk's
 * slug AND every declared field name (so the model's emitted
 * "samples[0].chunk_id.field" path or bare "samples[0].field" path
 * both route to the right tab).  A final Descriptives tab covers the
 * paper-level metadata.
 */
function _designerBuildPresetDescriptor(decl) {
  const ts = (new Date()).toISOString().replace(/[^0-9]/g, '').slice(0, 14);
  const slug = _designerSlugify(decl.unit.name);
  const presetId = `user-${slug}-${ts}`;

  const sub_views = decl.chunks.map(c => {
    const id = _designerSlugify(c.name);
    const field_names = c.fields_parsed.map(f => f.name);
    return {
      id:              id,
      label:           c.name || id,
      include_keys:    ['sample_id', id],
      evidence_keys:   [id, ...field_names],
      confidence_keys: [id],
    };
  });
  // Paper-metadata tab — always present, named after the conventional
  // MASEM "Descriptives" so users have a stable mental model across
  // presets.
  sub_views.push({
    id:              'descriptives',
    label:           'Paper metadata',
    include_keys:    ['sample_id', 'paper_metadata'],
    evidence_keys:   ['paper_metadata',
                      ...decl.paper_metadata_fields.map(f => f.name),
                      'title', 'doi', 'year', 'authors'],
    confidence_keys: ['paper_metadata'],
  });

  return {
    id:                          presetId,
    title:                       _designerTitleize(decl.unit.name),
    tagline:                     `Custom extraction — one entry per ${decl.unit.name || 'unit'}, ${decl.chunks.length} sub-tab${decl.chunks.length === 1 ? '' : 's'}`,
    mode:                        'extraction',
    accent_color:                '#367380',
    sub_views:                   sub_views,
    template_params:             {data_sources: decl.chunks.map(c => _designerSlugify(c.name))},
    // Marker the loader / picker uses to differentiate user-built
    // presets from bundled server presets.  Also lets us re-load the
    // declaration into the form when the user clicks "Edit" later.
    _generated_from_designer:    true,
    _declaration:                decl,
    _created_at:                 new Date().toISOString(),
  };
}

/* ── localStorage user-preset store ─────────────────────────────── */

function _designerLoadUserPresets() {
  try {
    const raw = localStorage.getItem(_USER_PRESETS_KEY);
    return raw ? (JSON.parse(raw) || {}) : {};
  } catch (_) { return {}; }
}

function _designerSaveUserPreset(preset) {
  try {
    const all = _designerLoadUserPresets();
    all[preset.id] = preset;
    localStorage.setItem(_USER_PRESETS_KEY, JSON.stringify(all));
  } catch (_) { /* localStorage disabled — silently no-op */ }
}

/* ── Orchestrator: form → prompt + preset → /api/generate-prompt ──── */

async function designerProcessDeclaration(decl) {
  const metaInput = _designerBuildMetaPromptInput(decl);
  const preset    = _designerBuildPresetDescriptor(decl);

  // Persist the preset so its sub-tabs survive a page reload and so a
  // future "My workflows" picker on step 1 can re-surface it.
  _designerSaveUserPreset(preset);

  // Activate the preset for the in-flight session.  The renderer's
  // sub-tab + confidence-badge filter reads state.activePreset
  // directly — no server registration needed.
  if (typeof state !== 'undefined') {
    state.activePreset = preset;
    if (typeof document !== 'undefined' && document.body) {
      document.body.dataset.preset = preset.id;
    }
  }

  // Feed the assembled prompt-input into the existing generate-prompt
  // pipeline.  questionInput / contextInput are now hidden, but the
  // pipeline reads their .value directly so this works unchanged.
  const q = document.getElementById('questionInput');
  const c = document.getElementById('contextInput');
  if (q) q.value = metaInput.question;
  if (c) c.value = metaInput.context;
  if (typeof state !== 'undefined') {
    state.question = metaInput.question;
    state.context  = metaInput.context;
  }
  if (typeof autoSaveSession === 'function') autoSaveSession();

  if (typeof callGenerateAPI === 'function') {
    await callGenerateAPI();
  }
}

/* Run init when the section is first revealed.  We can't bind on
 * DOMContentLoaded because the section is hidden until the user picks
 * "Generate with AI" — its inputs aren't focusable until then, and
 * binding to disconnected nodes is wasted work. */
if (typeof window !== 'undefined') {
  window.designerAddChunk           = designerAddChunk;
  window.designerRemoveChunk        = designerRemoveChunk;
  window.designerSubmit             = designerSubmit;
  window._designerInit              = _designerInit;
  window.designerBuildDeclaration   = designerBuildDeclaration;
  window.designerProcessDeclaration = designerProcessDeclaration;
  // Exposed for Phase C "My workflows" picker on step 1.
  window._designerLoadUserPresets   = _designerLoadUserPresets;
  window._designerSaveUserPreset    = _designerSaveUserPreset;
}
