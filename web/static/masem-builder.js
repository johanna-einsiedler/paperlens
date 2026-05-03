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

  const startId = presetId || state.activePreset?.id || 'masem';
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
  _refreshMasemPreview();
}

/* Mirror the working params into the form widgets. */
function _populateBuilderForm(params) {
  const sources = params.data_sources || [];
  document.querySelectorAll('#masemDataSourcesRow input[data-source]').forEach(box => {
    box.checked = sources.includes(box.dataset.source);
  });
  const scope = params.content_scope || 'concrete_items';
  document.querySelectorAll('#masemScopeRow input[name="scope"]').forEach(r => {
    r.checked = (r.value === scope);
  });
  document.getElementById('masemVariablesInput').value = _serialiseVariables(params.variables || []);
  document.getElementById('masemItemTextsInput').value = (params.item_texts || []).join('\n');
  document.getElementById('masemIncludeItemTexts').checked = !!params.include_item_texts;
  _updateMasemConditionalUI();
}

/* Hide / show conditional fields based on the current scope. */
function _updateMasemConditionalUI() {
  const scope = _MASEM_BUILDER_STATE.params.content_scope || 'concrete_items';
  const itemTextsGroup = document.getElementById('masemItemTextsGroup');
  if (itemTextsGroup) {
    itemTextsGroup.style.display = (scope === 'concrete_items') ? '' : 'none';
  }
  const hint = document.getElementById('masemVariablesHint');
  if (hint) {
    hint.textContent = ({
      concrete_items:        '(usually empty for concrete-items scope — variables are the instrument items)',
      content_groups:        '(scale / instrument names — one per line)',
      theoretical_constructs:'(construct names — one per line; add definitions and synonyms via "Name | Definition | synonym1, synonym2")',
    }[scope]) || '';
  }
}

/* Variables are stored as ``[{name, definition?, synonyms?}, ...]``;
   on screen they're a one-line-per-variable textarea with an optional
   ``Name | Definition | synonym1, synonym2`` triple-pipe shape. */
function _serialiseVariables(variables) {
  return variables.map(v => {
    if (typeof v === 'string') return v;
    if (!v || typeof v !== 'object') return '';
    const parts = [v.name || ''];
    if (v.definition) parts.push(v.definition);
    if (Array.isArray(v.synonyms) && v.synonyms.length) parts.push(v.synonyms.join(', '));
    return parts.join(' | ');
  }).filter(Boolean).join('\n');
}

function _parseVariables(text) {
  const out = [];
  for (const raw of (text || '').split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split('|').map(p => p.trim());
    const entry = { name: parts[0] };
    if (parts[1]) entry.definition = parts[1];
    if (parts[2]) entry.synonyms = parts[2].split(',').map(s => s.trim()).filter(Boolean);
    out.push(entry);
  }
  return out;
}

/* Read the current form values back into ``_MASEM_BUILDER_STATE.params``,
   preserving fields the form doesn't expose (n_items, factor_naming,
   cfa_item_assignment, etc.) — those come from the starter's defaults
   and can be tuned later via JSON edits if needed. */
function _readFormIntoParams() {
  const p = _MASEM_BUILDER_STATE.params;
  const sources = [];
  document.querySelectorAll('#masemDataSourcesRow input[data-source]').forEach(box => {
    if (box.checked) sources.push(box.dataset.source);
  });
  p.data_sources = sources;
  const scopeEl = document.querySelector('#masemScopeRow input[name="scope"]:checked');
  if (scopeEl) p.content_scope = scopeEl.value;
  p.variables = _parseVariables(document.getElementById('masemVariablesInput').value);
  p.item_texts = (document.getElementById('masemItemTextsInput').value || '').split('\n')
    .map(s => s.trim()).filter(Boolean);
  p.include_item_texts = document.getElementById('masemIncludeItemTexts').checked;
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
  } catch (err) {
    console.warn('[masem-builder] render network error:', err);
  }
}

/* Wire up form-change listeners ONCE.  Idempotent so applyPreset can
   call openMasemBuilder repeatedly without duplicating handlers. */
function _attachMasemBuilderListeners() {
  if (_attachMasemBuilderListeners._attached) return;
  _attachMasemBuilderListeners._attached = true;
  document.querySelectorAll(
    '#masemDataSourcesRow input, #masemScopeRow input, #masemVariablesInput, ' +
    '#masemItemTextsInput, #masemIncludeItemTexts'
  ).forEach(el => {
    el.addEventListener('change', () => {
      _readFormIntoParams();
      _updateMasemConditionalUI();
      _refreshMasemPreview();
    });
    el.addEventListener('input', _refreshMasemPreview);
  });
}

/* "Use this prompt" — commits the current preview into state and moves
   the user on to step 5 (review prompt). */
async function masemBuilderCommit() {
  await _doRefreshMasemPreview();   // make sure preview is up-to-date
  const prompt = document.getElementById('masemPreviewBox').textContent || '';
  if (!prompt.trim()) {
    showToast('Could not build a prompt — pick at least one data source.');
    return;
  }
  state.generatedPrompt = prompt;
  state.inputMode       = 'manual';
  // Replace the active preset's sub_views with the data-source-driven
  // set so the result panel's tabs match what was actually extracted.
  if (state.activePreset && Array.isArray(_MASEM_BUILDER_STATE.lastSubViews)) {
    state.activePreset.sub_views = _MASEM_BUILDER_STATE.lastSubViews;
  }
  // Mirror the prompt into the manual textarea so the review step can
  // show it and the user can still hand-edit before extracting.
  const manualInput = document.getElementById('manualPromptInput');
  if (manualInput) manualInput.value = prompt;
  const promptDisplay = document.getElementById('promptDisplay');
  if (promptDisplay) promptDisplay.textContent = prompt;
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
