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

const _MASEM_BUILDER_STATE = {
  starter: null,        // active starter preset id
  params:  {},          // current template_params (live form state)
  presetCache: {},      // id → fetched preset detail (avoids re-GET on starter switch)
  previewTimer: null,   // debounce timer for re-render after typing
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

/* The two data-source cards (A: direct correlations / B: reconstructed)
   each turn ON or OFF a pair of underlying ``data_sources`` keys.  This
   is just a UX simplification — the prompt-renderer still consumes the
   four canonical keys (factor_loadings / factor_correlations /
   correlation_matrix / single_correlations). */
const _SOURCE_CARD_GROUPS = {
  A: ["correlation_matrix", "single_correlations"],
  B: ["factor_loadings",    "factor_correlations"],
};

/* Mirror the working params into the form widgets. */
function _populateBuilderForm(params) {
  // A/B card pressed-state derives from data_sources.
  const sources = new Set(params.data_sources || []);
  for (const card of ["A", "B"]) {
    const active = _SOURCE_CARD_GROUPS[card].some(k => sources.has(k));
    const el = document.getElementById("masemSource" + card);
    if (el) {
      el.classList.toggle("active", active);
      el.setAttribute("aria-pressed", active ? "true" : "false");
    }
  }
  // Section C — combined identification list.  We serialise variables
  // (with | syntax) AND item texts into the same textarea, since they
  // express the same intent ("here's what to look for") and switching
  // between them is just a matter of A vs B card selection.
  document.getElementById('masemCInput').value = _serialiseSectionC(params);
  // Section D — free-form study context.
  document.getElementById('masemDInput').value = params.study_characteristics_text || '';
}

/* Pick whichever of (variables, item_texts) is non-empty and emit it as
   the textarea body.  When both are present (rare), prefer item_texts
   since they represent more concrete content. */
function _serialiseSectionC(params) {
  const items = params.item_texts || [];
  if (items.length) return items.join('\n');
  const vars = params.variables || [];
  return vars.map(v => {
    if (typeof v === 'string') return v;
    if (!v || typeof v !== 'object') return '';
    const parts = [v.name || ''];
    if (v.definition) parts.push(v.definition);
    if (Array.isArray(v.synonyms) && v.synonyms.length) parts.push(v.synonyms.join(', '));
    return parts.join(' | ');
  }).filter(Boolean).join('\n');
}

/* Parse section C into either variables (with optional | syntax) OR
   item_texts, depending on what the active data sources call for.
   * If B is active (reconstructed information / factor analysis), the
     content scope is "concrete items" and each line is treated as an
     item text in original order.
   * Otherwise (A only, direct correlations), each line is a
     variable / scale / construct name with optional definition +
     synonyms via the |-separated syntax. */
function _parseSectionC(text, dataSources) {
  const lines = (text || '').split('\n').map(l => l.trim()).filter(Boolean);
  const fl = dataSources.includes('factor_loadings')
          || dataSources.includes('factor_correlations');
  if (fl) {
    // Lines feed item_texts.  We strip a leading "1. " / "12) "
    // numbering if the user pasted a numbered list.
    const cleaned = lines.map(l => l.replace(/^\s*\d+[.)]\s+/, ''));
    return { item_texts: cleaned, include_item_texts: cleaned.length > 0,
             variables: [] };
  }
  const variables = [];
  for (const line of lines) {
    const parts = line.split('|').map(p => p.trim());
    const entry = { name: parts[0] };
    if (parts[1]) entry.definition = parts[1];
    if (parts[2]) entry.synonyms = parts[2].split(',').map(s => s.trim()).filter(Boolean);
    variables.push(entry);
  }
  return { variables, item_texts: [], include_item_texts: false };
}

/* Toggle one of the A/B data-source cards.  Each card maps to a pair of
   underlying data_sources keys; toggling the card flips both keys
   on/off together.  Refreshes the live preview after every click. */
function _toggleMasemSource(card) {
  const el = document.getElementById('masemSource' + card);
  if (!el) return;
  const wasActive = el.classList.contains('active');
  const willBeActive = !wasActive;
  el.classList.toggle('active', willBeActive);
  el.setAttribute('aria-pressed', willBeActive ? 'true' : 'false');

  const p = _MASEM_BUILDER_STATE.params;
  const keys = new Set(p.data_sources || []);
  for (const k of _SOURCE_CARD_GROUPS[card]) {
    if (willBeActive) keys.add(k);
    else              keys.delete(k);
  }
  p.data_sources = Array.from(keys);
  // Scope follows: B (factor analysis) → concrete items; A only →
  // theoretical constructs.  Keeps the prompt's wording aligned with
  // what's actually being asked of the model.
  const fl = p.data_sources.includes('factor_loadings')
          || p.data_sources.includes('factor_correlations');
  p.content_scope = fl ? 'concrete_items' : 'theoretical_constructs';
  // Re-parse section C in case the meaning of the textarea changed.
  Object.assign(p, _parseSectionC(document.getElementById('masemCInput').value, p.data_sources));
  _refreshMasemPreview();
}

/* Read the current form values back into ``_MASEM_BUILDER_STATE.params``,
   preserving fields the form doesn't expose (n_items, factor_naming,
   cfa_item_assignment, etc.) — those come from the starter's defaults
   and can be tuned later via JSON edits if needed. */
function _readFormIntoParams() {
  const p = _MASEM_BUILDER_STATE.params;
  // data_sources is updated synchronously by _toggleMasemSource — we
  // don't try to derive it from DOM here (the cards' .active class is
  // the cosmetic layer; p.data_sources is the source of truth).
  // Sections C and D are textareas, so we re-read them on every render.
  Object.assign(p, _parseSectionC(document.getElementById('masemCInput').value, p.data_sources || []));
  p.study_characteristics_text = document.getElementById('masemDInput').value || '';
  // Scope follows from data_sources (same logic as _toggleMasemSource).
  const fl = (p.data_sources || []).includes('factor_loadings')
          || (p.data_sources || []).includes('factor_correlations');
  p.content_scope = fl ? 'concrete_items' : 'theoretical_constructs';
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
  ['masemCInput', 'masemDInput'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', _refreshMasemPreview);
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
    showToast('Could not build a prompt — pick at least one data source.');
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
