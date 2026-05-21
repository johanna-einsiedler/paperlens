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

// Inline SVGs are kept small (24×24) and use ``currentColor`` so the
// card's hover / active state can recolour them via CSS.
const _MASEM_STARTER_ICON_DIRECT = `
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none"
       stroke="currentColor" stroke-width="1.7"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M6 3h9l3 3v15H6z"/>
    <path d="M15 3v3h3"/>
    <path d="M9 12h6M9 16h6M9 8h3"/>
  </svg>`;
const _MASEM_STARTER_ICON_INDIRECT = `
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none"
       stroke="currentColor" stroke-width="1.7"
       stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <path d="M4 20h16"/>
    <rect x="6"  y="11" width="3" height="8" rx="0.5"/>
    <rect x="11" y="7"  width="3" height="12" rx="0.5"/>
    <rect x="16" y="13" width="3" height="6" rx="0.5"/>
  </svg>`;

const _MASEM_STARTERS = [
  { id: 'masem',          label: 'Direct information',
    tagline: 'Extract correlations from text and table(s).',
    icon: _MASEM_STARTER_ICON_DIRECT },
  { id: 'masem-tas20',    label: 'Indirect information',
    tagline: 'Extract factor loadings and factor correlations from text and table(s).',
    icon: _MASEM_STARTER_ICON_INDIRECT },
];

const _MASEM_BUILDER_STATE = {
  starter: null,            // active starter preset id
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
  _renderMasemStarterCards();

  // Default starter: "Direct information" (general ``masem`` preset).
  // Indirect-information / factor-loading extractions are the less common
  // path; users who need them switch with one click.
  const startId = presetId || 'masem';
  await _selectMasemStarter(startId, /*isUserClick=*/ false);
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
      <div class="masem-starter-icon">${s.icon || ''}</div>
      <div class="masem-starter-body">
        <div class="masem-starter-label">${escHtml(s.label)}</div>
        <div class="masem-starter-tagline">${escHtml(s.tagline)}</div>
      </div>
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

/* (Was a hard-wired factor-loadings data_sources list.  Removed —
   data_sources now flows from each starter preset's own JSON so the
   Direct (effect_sizes) and Indirect (factor_*) shapes are honoured
   without the form overwriting them.) */

/* Mirror the working params into the form widgets. */
function _populateBuilderForm(params) {
  // Compact scalar row — scale name + item count.
  const scaleName = params.scale_name
                 || params.instrument_name
                 || params.instrument_name_long
                 || '';
  const scaleEl = document.getElementById('masemScaleName');
  if (scaleEl) scaleEl.value = scaleName === 'the target scale' ? '' : scaleName;
  const nItemsEl = document.getElementById('masemNItems');
  if (nItemsEl) nItemsEl.value = (params.n_items != null) ? params.n_items : '';

  // Item labels textarea — serialise item_texts back as "1: text".
  document.getElementById('masemCInput').value = _serialiseItemTexts(params.item_texts || []);
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
  // n_factors / n_factors_max are no longer surfaced in the form — they
  // come from the preset's template_params defaults.
  // ``data_sources`` is NOT overwritten here: each starter preset
  // declares its own (``masem`` → ``["effect_sizes"]``, ``masem-tas20``
  // → ``["factor_correlations", "factor_loadings"]``).  Overwriting it
  // would force every starter through one fixed shape and silently
  // rebuild the sub_views to match — which is what produced Factor-
  // loadings tabs in Direct mode.
  p.content_scope = 'concrete_items';
  // Item labels textarea → item_texts list.
  const items = _parseItemTexts(document.getElementById('masemCInput').value);
  p.item_texts         = items;
  p.include_item_texts = items.length > 0;
  // Drop legacy fields the simplified builder no longer surfaces.
  p.variables                  = [];
  p.study_characteristics_text = '';
  // factor_naming is no longer collected from the form (the field was
  // removed because it tended to confuse the model on papers with
  // non-standard factor naming).  The factor_key_mapping section of
  // the prompt handles label → JSON-key mapping without it.
  p.factor_naming              = [];
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
  try {
    const res = await fetchScoped('/api/build-preset-prompt', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        preset_id:       presetId,
        template_params: _MASEM_BUILDER_STATE.params,
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
    'masemScaleName', 'masemNItems',
    'masemCInput',
  ];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input',  _refreshMasemPreview);
    el.addEventListener('change', _refreshMasemPreview);
  });
}

/* "Use this prompt" — flushes any pending debounced render, makes sure
   the preview is in state, and advances the flow.  Auto-commit on every
   form change handles the bulk of the work; this just decides where to
   land:
     * MASEMiner-mode users skip the "Review prompt" step entirely and
       go straight to Upload (confirmPrompt takes care of step-6 setup).
     * MetaPaperLens-mode users keep the Review-prompt step. */
async function masemBuilderCommit() {
  await _doRefreshMasemPreview();
  const prompt = document.getElementById('masemPreviewBox').textContent || '';
  if (!prompt.trim()) {
    showToast('Could not build a prompt — please fill in the scale name.');
    return;
  }
  // _doRefreshMasemPreview already mirrored the prompt into state.
  if (document.body.classList.contains('is-maseminer')) {
    confirmPrompt();
  } else {
    goTo(5);
  }
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
