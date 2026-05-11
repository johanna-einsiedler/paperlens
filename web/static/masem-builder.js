/* MASEMiner guided prompt builder.
 *
 * Replaces the freeform "describe your task" textarea (step 3) when a
 * MASEMiner preset is active.  The user picks a starter (TAS-20 / Big
 * Five / General), toggles the data sources their meta-analysis needs,
 * names the variables of interest, and previews the rendered prompt
 * live.  On commit, the rendered prompt is dropped into
 * ``state.generatedPrompt`` and the auto-derived ``sub_views`` replace
 * the ones on ``state.activePreset`` so the result-panel tabs match.
 *
 * The form is data-source-driven on top of the same backend renderer
 * (POST /api/build-preset-prompt), so anything the JSON template can
 * express, the form can reach.
 */

const _MASEM_STARTERS = [
  { id: 'masem',          label: 'Blank / General',
    tagline: 'Start empty — define your own variables and pick the data sources you need.' },
  { id: 'masem-tas20',    label: 'TAS-20 example',
    tagline: 'Pre-filled for Toronto Alexithymia Scale meta-analyses (factor loadings + inter-factor correlations).' },
];

/* Available prompt templates the form can render against.  The
   ``file`` field is sent to /api/build-preset-prompt as
   ``prompt_template_file`` — when null, the preset's own default
   template is used. */
const _MASEM_TEMPLATES = [
  { id: 'default', label: 'Default',
    tagline: 'The original MASEMiner prompt with factor-naming + reference-item blocks.',
    file: null },
  { id: 'timo',    label: "Timo's Template",
    tagline: 'Stepwise extraction procedure with strict evidence rules and inference principles.',
    file: 'masem-timo.template.md' },
];

const _MASEM_BUILDER_STATE = {
  starter: null,            // active starter preset id
  templateId: 'default',    // active prompt-template id (from _MASEM_TEMPLATES)
  params:  {},              // current template_params (live form state)
  presetCache: {},          // id → fetched preset detail (avoids re-GET on starter switch)
  previewTimer: null,       // debounce timer for re-render after typing
};

/* Returns true when the active preset is one of the parameterised
   MASEMiner presets (the umbrella ``masem`` preset OR any ``masem-*``
   variant).  Used by app.js's applyPreset to swap step 3 into the
   builder view instead of the freeform-text fallback. */
function isMasemPreset(preset) {
  if (!preset) return false;
  return Boolean(preset.template_params)
      || (typeof preset.id === 'string' && preset.id.startsWith('masem'));
}

/* Open the guided builder.  Hides the other step-3 panels (choice card,
   AI-generate, manual prompt), shows the builder, fetches the active
   preset's defaults, and renders the form + initial preview. */
async function openMasemBuilder(presetId) {
  document.getElementById('step3Choice').style.display    = 'none';
  document.getElementById('aiSection').style.display      = 'none';
  document.getElementById('manualSection').style.display  = 'none';
  document.getElementById('masemBuilder').style.display   = '';
  _renderMasemTemplateCards();
  _renderMasemStarterCards();

  // Default starter: TAS-20.  The umbrella ``masem`` preset is general
  // MASEM (correlation matrices + prose correlations), but most users
  // entering via /maseminer are running TAS-20-style extractions and
  // would hit a parse-failure if the model got the general prompt for
  // a TAS-20 paper.  Defaulting the builder to the TAS-20 starter
  // pre-fills the form with factor loadings + correlations + item
  // texts, so the auto-committed prompt is TAS-20-ready out of the box;
  // a single click on "Blank / General" switches them.
  const startId = (!presetId || presetId === 'masem')
    ? 'masem-tas20'
    : presetId;
  await _selectMasemStarter(startId, /*isUserClick=*/ false);
}

/* Render the prompt-template selector cards.  Switching templates does
   NOT change form values — it just swaps which template file the
   backend renders against on the next preview refresh. */
function _renderMasemTemplateCards() {
  const row = document.getElementById('masemTemplateRow');
  if (!row) return;
  const active = _MASEM_BUILDER_STATE.templateId;
  row.innerHTML = _MASEM_TEMPLATES.map(t => `
    <button type="button"
            class="masem-template-card ${t.id === active ? 'active' : ''}"
            onclick="_selectMasemTemplate('${t.id}')">
      <div class="masem-starter-label">${escHtml(t.label)}</div>
      <div class="masem-starter-tagline">${escHtml(t.tagline)}</div>
    </button>`).join('');
}

/* Switch to a different prompt template.  Doesn't touch starter or
   form values — just triggers a preview re-render against the new
   template body. */
function _selectMasemTemplate(templateId) {
  if (!_MASEM_TEMPLATES.some(t => t.id === templateId)) return;
  _MASEM_BUILDER_STATE.templateId = templateId;
  _renderMasemTemplateCards();
  if (_MASEM_BUILDER_STATE.previewTimer) {
    clearTimeout(_MASEM_BUILDER_STATE.previewTimer);
    _MASEM_BUILDER_STATE.previewTimer = null;
  }
  _doRefreshMasemPreview();
}

/* Render the three "starter" cards at the top of the form.  Highlights
   the active one. */
function _renderMasemStarterCards() {
  const row = document.getElementById('masemStarterRow');
  if (!row) return;
  const active = _MASEM_BUILDER_STATE.starter;
  row.innerHTML = _MASEM_STARTERS.map(s => `
    <button type="button"
            class="masem-starter-card ${s.id === active ? 'active' : ''}"
            onclick="_selectMasemStarter('${s.id}', true)">
      <div class="masem-starter-label">${escHtml(s.label)}</div>
      <div class="masem-starter-tagline">${escHtml(s.tagline)}</div>
    </button>`).join('');
}

/* Switch to a starter preset.  Loads the preset's defaults (cached
   per-id), copies them into the form's working params, populates the
   form widgets, and re-renders the live preview. */
async function _selectMasemStarter(presetId, isUserClick) {
  let preset = _MASEM_BUILDER_STATE.presetCache[presetId];
  if (!preset) {
    try {
      const res = await fetchScoped(`/api/presets/${encodeURIComponent(presetId)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      preset = await res.json();
      _MASEM_BUILDER_STATE.presetCache[presetId] = preset;
    } catch (err) {
      console.warn('[masem-builder] preset fetch failed:', err);
      return;
    }
  }
  _MASEM_BUILDER_STATE.starter = presetId;
  _MASEM_BUILDER_STATE.params  = JSON.parse(JSON.stringify(preset.template_params || {}));
  _renderMasemStarterCards();
  _populateBuilderForm(_MASEM_BUILDER_STATE.params);
  // Immediate (non-debounced) render on starter switch so the auto-
  // commit happens before the user can race ahead — debouncing only
  // makes sense for the chatty character-by-character textarea events.
  if (_MASEM_BUILDER_STATE.previewTimer) {
    clearTimeout(_MASEM_BUILDER_STATE.previewTimer);
    _MASEM_BUILDER_STATE.previewTimer = null;
  }
  await _doRefreshMasemPreview();
}

/* This preset is factor-loadings-focused — data_sources is hard-wired
   so users don't have to think about it. */
const _FIXED_DATA_SOURCES = ["factor_loadings", "factor_correlations"];

/* Mirror the working params into the form widgets. */
function _populateBuilderForm(params) {
  // Compact scalar row — scale name / item count / max-factor count.
  const scaleName = params.scale_name
                 || params.instrument_name
                 || params.instrument_name_long
                 || '';
  const scaleEl = document.getElementById('masemScaleName');
  if (scaleEl) scaleEl.value = scaleName === 'the target scale' ? '' : scaleName;
  const nItemsEl = document.getElementById('masemNItems');
  if (nItemsEl) nItemsEl.value = (params.n_items != null) ? params.n_items : '';
  const nFactorsEl = document.getElementById('masemNFactorsMax');
  if (nFactorsEl) nFactorsEl.value = (params.n_factors_max != null)
                                       ? params.n_factors_max
                                       : (params.n_factors != null ? params.n_factors : '');

  // Item labels textarea — serialise item_texts back as "1: text".
  document.getElementById('masemCInput').value = _serialiseItemTexts(params.item_texts || []);

  // Factor labels textarea — serialise factor_naming back to one line
  // per factor in the user-friendly "1. F1 = ABBR (Long name)" format.
  document.getElementById('masemFactorLabelsInput').value
    = _serialiseFactorLabels(params.factor_naming || []);
}

/* Serialise a list of item texts back into the textarea body as
   ``1: <text>`` per line, matching the input format. */
function _serialiseItemTexts(items) {
  if (!Array.isArray(items) || !items.length) return '';
  return items.map((t, i) => `${i + 1}: ${t}`).join('\n');
}

/* Parse the item-labels textarea into a list of item texts in order.
   Strips a leading ``1:`` / ``1.`` / ``1)`` numbering if present. */
function _parseItemTexts(text) {
  const lines = (text || '').split('\n').map(l => l.trim()).filter(Boolean);
  return lines.map(l => l.replace(/^\s*\d+\s*[:.)]\s*/, '').trim()).filter(Boolean);
}

/* Serialise the structured factor_naming list back into a textarea body.
   Accepts either string entries or {abbrev, name} objects. */
function _serialiseFactorLabels(naming) {
  if (!Array.isArray(naming) || !naming.length) return '';
  return naming.map((f, i) => {
    const n = i + 1;
    if (typeof f === 'string') return `${n}. F${n} = ${f}`;
    if (f && typeof f === 'object') {
      const abbr = f.abbrev || f.abbreviation || '';
      const name = f.name   || f.long_name    || '';
      if (abbr && name) return `${n}. F${n} = ${abbr} (${name})`;
      if (abbr)         return `${n}. F${n} = ${abbr}`;
      if (name)         return `${n}. F${n} = ${name}`;
    }
    return '';
  }).filter(Boolean).join('\n');
}

/* Parse the factor-labels textarea into a structured list.  Each line is
   expected to look like ``1. F1 = DIF (Difficulty Identifying Feelings)``
   but we accept anything — we strip the leading number/Fn= prefix and
   try to split an ``ABBR (Long name)`` tail. */
function _parseFactorLabels(text) {
  const out = [];
  const lines = (text || '').split('\n').map(l => l.trim()).filter(Boolean);
  for (const raw of lines) {
    // Strip a leading numbering like "1." / "1)" / "1 -" / "F1 =".
    let body = raw.replace(/^\s*\d+\s*[.)\-]\s*/, '')
                  .replace(/^\s*F?\d+\s*[=:]\s*/i, '')
                  .trim();
    if (!body) continue;
    // ``ABBR (Long name)`` → split into abbrev + name.
    const m = body.match(/^(.*?)\s*\(([^)]*)\)\s*$/);
    if (m) out.push({ abbrev: m[1].trim(), name: m[2].trim() });
    else   out.push({ abbrev: body,        name: '' });
  }
  return out;
}


/* Read the current form values back into ``_MASEM_BUILDER_STATE.params``,
   preserving fields the form doesn't expose (cfa_item_assignment, etc.)
   — those come from the starter's defaults and can be tuned later via
   JSON edits if needed. */
function _readFormIntoParams() {
  const p = _MASEM_BUILDER_STATE.params;
  // Compact scalar row.
  const scaleEl = document.getElementById('masemScaleName');
  if (scaleEl) {
    const v = (scaleEl.value || '').trim();
    if (v) {
      p.scale_name = v;
      p.instrument_name = v;
      p.instrument_name_long = v;
    } else {
      p.scale_name = 'the target scale';
      p.instrument_name = 'the target scale';
      p.instrument_name_long = 'the target scale';
    }
  }
  const nItemsEl = document.getElementById('masemNItems');
  if (nItemsEl) {
    const n = parseInt(nItemsEl.value, 10);
    if (Number.isFinite(n) && n > 0) p.n_items = n;
  }
  const nFactorsEl = document.getElementById('masemNFactorsMax');
  if (nFactorsEl) {
    const n = parseInt(nFactorsEl.value, 10);
    if (Number.isFinite(n) && n > 0) {
      p.n_factors_max = n;
      p.n_factors = n;
    }
  }
  // Data sources are fixed for this preset — every run extracts factor
  // loadings + factor correlations.  Scope follows: factor-loadings
  // workflows = concrete items.
  p.data_sources  = _FIXED_DATA_SOURCES.slice();
  p.content_scope = 'concrete_items';
  // Item labels textarea → item_texts list.
  const items = _parseItemTexts(document.getElementById('masemCInput').value);
  p.item_texts         = items;
  p.include_item_texts = items.length > 0;
  // Drop legacy fields the simplified builder no longer surfaces.
  p.variables                  = [];
  p.study_characteristics_text = '';
  // Factor labels textarea → factor_naming list.
  p.factor_naming = _parseFactorLabels(document.getElementById('masemFactorLabelsInput').value);
}

/* Debounced re-render of the prompt preview after any form change.
   Calls /api/build-preset-prompt and writes the response into the
   preview <pre> + the character counter. */
function _refreshMasemPreview() {
  if (_MASEM_BUILDER_STATE.previewTimer) {
    clearTimeout(_MASEM_BUILDER_STATE.previewTimer);
  }
  _MASEM_BUILDER_STATE.previewTimer = setTimeout(_doRefreshMasemPreview, 350);
}

async function _doRefreshMasemPreview() {
  _readFormIntoParams();
  const presetId = _MASEM_BUILDER_STATE.starter;
  if (!presetId) return;
  const tplSpec = _MASEM_TEMPLATES.find(t => t.id === _MASEM_BUILDER_STATE.templateId);
  const tplFile = tplSpec && tplSpec.file ? tplSpec.file : null;
  try {
    const res = await fetchScoped('/api/build-preset-prompt', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        preset_id:            presetId,
        template_params:      _MASEM_BUILDER_STATE.params,
        prompt_template_file: tplFile,
      }),
    });
    if (!res.ok) {
      console.warn('[masem-builder] render failed:', res.status, await res.text().catch(() => ''));
      return;
    }
    const data = await res.json();
    document.getElementById('masemPreviewBox').textContent = data.prompt || '';
    document.getElementById('masemPreviewLen').textContent = (data.prompt || '').length;
    _MASEM_BUILDER_STATE.lastSubViews = data.sub_views || [];
    // Auto-commit the rendered preview into state.generatedPrompt and
    // sub_views.  Without this, a user who proceeds to step 5 / step 6
    // without explicitly clicking "Use this prompt" extracts with
    // whatever applyPreset originally wrote (the umbrella's general
    // prompt) — which is the wrong prompt for TAS-20 papers.  Mirroring
    // every preview into state keeps the live form and the actual
    // prompt-to-be-sent in sync at all times.
    _commitMasemBuilderToState(data.prompt || '');
  } catch (err) {
    console.warn('[masem-builder] render network error:', err);
  }
}

/* Mirror the current builder preview into state so the rest of the app
   uses the right prompt + sub_views.  Idempotent — safe to call on
   every form change. */
function _commitMasemBuilderToState(prompt) {
  if (!prompt || !prompt.trim()) return;
  state.generatedPrompt = prompt;
  state.inputMode       = 'manual';
  if (state.activePreset && Array.isArray(_MASEM_BUILDER_STATE.lastSubViews)) {
    state.activePreset.sub_views = _MASEM_BUILDER_STATE.lastSubViews;
  }
  // Mirror into the manual textarea + the read-only review display so
  // step 5 picks up the latest version automatically.
  const manualInput = document.getElementById('manualPromptInput');
  if (manualInput) manualInput.value = prompt;
  const promptDisplay = document.getElementById('promptDisplay');
  if (promptDisplay) promptDisplay.textContent = prompt;
}

/* Wire up form-change listeners ONCE.  Idempotent so applyPreset can
   call openMasemBuilder repeatedly without duplicating handlers.
   The A/B cards have their own onclick handlers in the HTML; only the
   textareas need debounced live re-rendering wired here. */
function _attachMasemBuilderListeners() {
  if (_attachMasemBuilderListeners._attached) return;
  _attachMasemBuilderListeners._attached = true;
  const ids = [
    'masemScaleName', 'masemNItems', 'masemNFactorsMax',
    'masemCInput', 'masemFactorLabelsInput',
  ];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input',  _refreshMasemPreview);
    el.addEventListener('change', _refreshMasemPreview);
  });
}

/* "Use this prompt" — flushes any pending debounced render, makes sure
   the preview is in state, and moves the user on to step 5.  Auto-
   commit on every form change handles the bulk of the work; this just
   advances the flow. */
async function masemBuilderCommit() {
  await _doRefreshMasemPreview();
  const prompt = document.getElementById('masemPreviewBox').textContent || '';
  if (!prompt.trim()) {
    showToast('Could not build a prompt — please fill in the scale name.');
    return;
  }
  // _doRefreshMasemPreview already mirrored the prompt into state.
  goTo(5);
}

/* "Edit raw prompt" — drops into the existing manual-prompt textarea so
   the user can hand-tweak whatever the form produced. */
function masemBuilderEditRaw() {
  _doRefreshMasemPreview().finally(() => {
    const prompt = document.getElementById('masemPreviewBox').textContent || '';
    const ta = document.getElementById('manualPromptInput');
    if (ta) ta.value = prompt;
    document.getElementById('masemBuilder').style.display = 'none';
    document.getElementById('manualSection').style.display = '';
  });
}

/* "Customise" button on the active-preset banner.  Reopens the builder
   form, scrolls to step 3, and re-renders against the current
   working params (so user edits survive a round-trip). */
function reopenMasemBuilder() {
  if (!state.activePreset) return;
  goTo(3);
  openMasemBuilder(_MASEM_BUILDER_STATE.starter || state.activePreset.id);
  const card = document.getElementById('masemBuilder');
  if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Wire listeners on first load (the form HTML is in the DOM at startup).
document.addEventListener('DOMContentLoaded', _attachMasemBuilderListeners);
