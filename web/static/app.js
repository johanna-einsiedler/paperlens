/* ──────────────────────────────────────────────────────────
   State
────────────────────────────────────────────────────────── */

// Server-enforced limits — fetched at page load so the UI can surface them.
const config = {
  maxBatchPapers: 20,           // overridden by GET /api/config on load
  maxPdfBytes:    50 * 1024 * 1024,
};

/* Browser-scoped session id.  Generated once per browser, persisted in
   localStorage, sent on every request as X-Session-Id.  The server uses it
   to scope /api/batches to "your" batches so the History view doesn't
   show other people's work.  Clearing localStorage forfeits history. */
function _getOrCreateSessionId() {
  try {
    let sid = localStorage.getItem('paperlens.sessionId');
    if (!sid) {
      sid = (crypto && crypto.randomUUID
        ? crypto.randomUUID()
        : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = (Math.random() * 16) | 0;
            return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
          }));
      localStorage.setItem('paperlens.sessionId', sid);
    }
    return sid;
  } catch (_) {
    // localStorage disabled — fall back to a per-tab id
    return 'transient-' + Math.random().toString(36).slice(2);
  }
}
const SESSION_ID = _getOrCreateSessionId();

/* Wrapper that always attaches the session header.  Use in place of fetch
   for any call that should be scoped to the current browser. */
function fetchScoped(url, opts = {}) {
  const headers = new Headers(opts.headers || {});
  headers.set('X-Session-Id', SESSION_ID);
  return fetch(url, { ...opts, headers });
}

const state = {
  step: 1,
  mode: null,
  activePreset: null,           // domain-workflow preset (e.g. MASEMiner), or null
  provider: 'openai',
  model: 'gpt-4o',
  apiKey: '',
  baseUrl: '',            // vLLM only: base URL of the OpenAI-compatible server
  // Per-provider credential cache.  Avoids the bug where switching from
  // OpenAI to Gemini left the OpenAI key visible (and saved) under the
  // Gemini provider — the (provider, apiKey, model, baseUrl) tuple is
  // now consistently scoped to one provider at a time.
  // Shape: { openai: {apiKey, model, baseUrl}, google: {...}, ... }
  providerCredentials: {},
  question: '',
  context: '',
  generatedPrompt: '',
  inputMode: 'generate',
  // Default extraction mode is text-layer parsing (faster, cheaper, supports
  // highlighting).  Vision is only used as a fallback when the upfront PDF
  // probe reports no usable text layer (or when the user explicitly picks
  // it via the parsing-method radio).
  useTextExtraction: true,
  notifyEmail: '',        // optional — server emails when the batch finishes
  batchId: null,          // shared id for all papers in one upload
  selectedFiles: [],
  papers: [],
  activePaperId: null,
  loadedFromFile: false,
  setupReturnStep: null,
};

/*  Paper object shape:
    {
      id:             string   (uuid)
      file:           File
      filename:       string
      status:         'pending' | 'processing' | 'done' | 'error'
      result:         string   (raw JSON from API)
      pageImages:     string[] (data-URIs, 0-indexed = page 1)
      entries:        Array | null
      entryIndex:     number
      pagesProcessed: number
      error:          string | null
    }
*/

// Models grouped per provider.  These are the BAKED-IN FALLBACK list —
// loadModelsConfig() overwrites them at startup from /static/models.json
// (regenerated weekly by the model-sync GitHub Action).  If that fetch
// fails or the file is malformed, these defaults keep the app working.
// Declared with ``let`` so the loader can reassign them.
let PROVIDER_MODELS = {
  openai:   [
    { value: 'gpt-5.5',      label: 'GPT-5.5' },
    { value: 'gpt-5-mini',   label: 'GPT-5 Mini' },
    { value: 'gpt-5',        label: 'GPT-5' },
    { value: 'gpt-5-nano',   label: 'GPT-5 Nano' },
    { value: 'gpt-4o-mini',  label: 'GPT-4o Mini' },
    { value: 'gpt-4o',       label: 'GPT-4o' },
    { value: 'gpt-4-turbo',  label: 'GPT-4 Turbo' },
  ],
  google:   [
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
    { value: 'gemini-2.5-pro',   label: 'Gemini 2.5 Pro' },
    { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  ],
  mistral:  [
    { value: 'mistral-large-latest', label: 'Mistral Large' },
    { value: 'mistral-small-latest', label: 'Mistral Small' },
    { value: 'pixtral-large-latest', label: 'Pixtral Large (vision)' },
    { value: 'pixtral-12b-2409',     label: 'Pixtral 12B (vision)' },
  ],
  deepseek: [
    { value: 'deepseek-chat',     label: 'DeepSeek Chat' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner (R1)' },
  ],
  anthropic: [
    { value: 'claude-opus-4-1',   label: 'Claude Opus 4.1' },
    { value: 'claude-sonnet-4-5', label: 'Claude Sonnet 4.5' },
    { value: 'claude-haiku-4-5',  label: 'Claude Haiku 4.5' },
  ],
  vllm: [], // model name is entered as free text
};

const PROVIDER_KEY_PLACEHOLDER = {
  openai:    'sk-...',
  google:    'AIza...',
  mistral:   'your Mistral API key',
  deepseek:  'sk-...',
  anthropic: 'sk-ant-...',
  vllm:      'any string (or leave blank if auth is disabled)',
};

const PROVIDER_KEY_LABEL = {
  openai:    'OpenAI API key',
  google:    'Google Gemini API key',
  mistral:   'Mistral API key',
  deepseek:  'DeepSeek API key',
  anthropic: 'Anthropic API key',
  vllm:      'API key',
};

function getProvider(model) {
  if (state.provider === 'vllm')    return 'vllm';
  if (model.startsWith('gemini'))   return 'google';
  if (model.startsWith('deepseek')) return 'deepseek';
  if (model.startsWith('claude'))   return 'anthropic';
  if (model.startsWith('mistral') || model.startsWith('pixtral')) return 'mistral';
  return 'openai';
}

// Returns true for vision-based models.  Text-only: DeepSeek and the plain
// ``mistral-*`` family.  ``pixtral-*`` models on Mistral support vision;
// Claude models are all vision-capable.  vLLM models default to vision —
// user can toggle to text extraction if their hosted model doesn't
// support image input.
function isVisionModel(model) {
  if (model.startsWith('deepseek')) return false;
  if (model.startsWith('mistral'))  return false;   // pixtral-* still passes
  return true;
}

/* ──────────────────────────────────────────────────────────
   Navigation (one-pager accordion + separate results view)
────────────────────────────────────────────────────────── */

// Each numeric "step" used by the rest of the code maps to one of the five
// accordion sections (or to the standalone results panel).  Loading states
// (steps 4 and 7) share their parent section with the post-loading content.
const STEP_TO_SECTION = {
  1: 'step1',
  2: 'step2',
  3: 'step3',
  4: 'step5',   // generating-prompt loading lives in the Review section
  5: 'step5',
  6: 'step6',
  7: 'step6',   // extracting loading lives in the Upload section
};

// Number shown in the section header (1..5).  Multiple steps map to one slot.
const STEP_TO_SECTION_NUM = { 1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5 };

function goTo(step) {
  state.step = step;

  // Any navigation past the initial state collapses the preset hero
  // landing (the /maseminer intro card).  Without this, the auto-
  // applied preset path (loadServerConfig → applyPreset → goTo(2))
  // leaves the hero visible above every step including the results
  // panel, which reads as "the intro reappeared after extraction".
  document.querySelectorAll('.preset-landing').forEach(el => { el.style.display = 'none'; });
  document.body.classList.add('masem-in-flow');

  const onResults = step === 8;
  const onepager  = document.getElementById('onepager');
  const results   = document.getElementById('step8');

  // Step 8 takes over: hide the accordion, show the full-width results card
  if (onResults) {
    if (onepager) onepager.style.display = 'none';
    if (results)  { results.style.display = ''; results.classList.add('active'); }
    document.body.classList.add('wide-mode');
    window.scrollTo({ top: 0, behavior: 'smooth' });
    updateSectionStatuses(step);
    return;
  }

  // Otherwise: show the accordion, hide results
  if (onepager) onepager.style.display = '';
  if (results)  { results.style.display = 'none'; results.classList.remove('active'); }
  document.body.classList.remove('wide-mode');

  // Toggle inline loading vs. content for sections 4 and 5
  const set = (id, on) => {
    const el = document.getElementById(id);
    if (el) el.style.display = on ? '' : 'none';
  };
  set('step4Loading', step === 4);
  set('step5Content', step === 5);
  set('step7Loading', step === 7);
  set('step6Content', step === 6);

  // Open the target section, collapse the others.  We close prior sections
  // explicitly so the accordion doesn't drift wide-open over time — the user
  // can always re-open any one by clicking its header.
  const targetId = STEP_TO_SECTION[step];
  document.querySelectorAll('.acc-section').forEach(s => {
    s.open = (s.id === targetId);
  });

  updateSectionStatuses(step);

  // Smoothly bring the active section to the top of the viewport
  const target = document.getElementById(targetId);
  if (target) {
    // Defer to allow any open/close transition to settle first
    requestAnimationFrame(() => target.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }
}

/* Mark the section header for the current step as "active", earlier sections as
   "done", later ones as "pending".  Also updates the inline summary text shown
   beside the section title (e.g. "Extract data" once mode is selected). */
function updateSectionStatuses(step) {
  const currentSection = STEP_TO_SECTION_NUM[step] || (step === 8 ? 6 : 1);
  // In MASEMiner mode the user sees a 3-step flow — step1 ("Choose
  // your task") and step5 ("Review prompt") are hidden, so the visible
  // chips need to read 1/2/3 instead of the raw 2/3/5 they carry in
  // MetaPaperLens mode.  This map is the single source of truth for
  // what number to render in each header in MASEMiner mode; everything
  // else (active / done class, sidebar mirror) stays unchanged.
  const isMasem    = document.body.classList.contains('is-maseminer');
  const masemLabel = { step2: '1', step3: '2', step6: '3' };
  for (let n = 1; n <= 5; n++) {
    const sectionId = ['step1', 'step2', 'step3', 'step5', 'step6'][n - 1];
    const el        = document.getElementById(sectionId);
    if (!el) continue;
    el.classList.remove('acc-pending', 'acc-active', 'acc-done');
    let kind = 'acc-pending';
    if      (n  < currentSection) kind = 'acc-done';
    else if (n === currentSection) kind = 'acc-active';
    el.classList.add(kind);
    // Swap the number for a ✓ on completed sections.  In MASEMiner
    // mode use the remapped label so the chips read 1·2·3 not 2·3·5.
    const num = el.querySelector('.acc-num');
    if (num) {
      const label = isMasem ? (masemLabel[sectionId] || String(n)) : String(n);
      num.textContent = (kind === 'acc-done') ? '✓' : label;
    }
  }
  // Mirror the same active/done state into the left sidebar tracker.
  _updateSidebarSteps(currentSection, step === 8);

  // Per-section summary text — short labels of what's been picked
  const summaries = {
    1: state.mode === 'extraction' ? 'Extract data'
      : state.mode === 'labeling'   ? 'Label paper'
      : state.loadedFromFile        ? 'Review existing results'
      : '',
    2: state.model
      ? `${({openai:'OpenAI', google:'Gemini', mistral:'Mistral', deepseek:'DeepSeek', anthropic:'Anthropic', vllm:'Custom'}[state.provider] || state.provider)} · ${state.model}`
      : '',
    3: state.generatedPrompt
      ? (state.inputMode === 'manual' ? 'Custom prompt' : 'Prompt generated')
      : '',
    4: state.generatedPrompt && state.step >= 6 ? 'Prompt confirmed' : '',
    5: state.papers.length
      ? `${state.papers.length} paper${state.papers.length !== 1 ? 's' : ''}`
      : '',
  };
  for (const [n, text] of Object.entries(summaries)) {
    const el = document.getElementById('summary' + n);
    if (el) el.textContent = text ? `· ${text}` : '';
  }
}

/* Reflect the same active/done/pending state into the left-sidebar step
   tracker.  Buttons numbered 1..5 mirror the accordion sections; the
   sixth button (data-step="8") is the "Results" item, active only when
   we're on the wide-mode results panel.

   ``currentSection`` is 1..5 — the active accordion section.
   ``onResults`` is true when state.step === 8. */
function _updateSidebarSteps(currentSection, onResults) {
  const buttons = document.querySelectorAll('#mplSidebarSteps .mpl-sidebar-step');
  buttons.forEach(btn => {
    btn.classList.remove('is-active', 'is-done');
    const ds = btn.getAttribute('data-step');
    if (ds === '8') {
      if (onResults) btn.classList.add('is-active');
      return;
    }
    // Map data-step (1,2,3,5,6) → sidebar sequence index 1..5
    const seq = { '1': 1, '2': 2, '3': 3, '5': 4, '6': 5 }[ds];
    if (!seq) return;
    if (!onResults && seq === currentSection) btn.classList.add('is-active');
    else if (seq < currentSection || onResults) btn.classList.add('is-done');
  });
}

/* ──────────────────────────────────────────────────────────
   Step 1 — Mode
────────────────────────────────────────────────────────── */

function selectMode(mode) {
  state.mode = mode;
  autoSaveSession();
  goTo(2);
}

/* ──────────────────────────────────────────────────────────
   Step 2 — Provider / Model + API key
────────────────────────────────────────────────────────── */

/* Persist whatever credentials are currently in `state` under the active
   provider's slot.  Called whenever the user types in the API key, base URL,
   or model fields, AND just before switching to a different provider. */
function _stashProviderCredentials() {
  const p = state.provider;
  if (!p) return;
  state.providerCredentials = state.providerCredentials || {};
  state.providerCredentials[p] = {
    apiKey:  state.apiKey  || '',
    model:   state.model   || '',
    baseUrl: state.baseUrl || '',
  };
}

function onProviderChange() {
  // 1. Save the OUTGOING provider's credentials before switching, so we
  //    don't lose them when the user toggles between providers.
  _stashProviderCredentials();

  const oldProvider = state.provider;
  const provider    = document.getElementById('providerSelect').value;
  const isVllm      = provider === 'vllm';
  const models      = PROVIDER_MODELS[provider] || [];
  const sel         = document.getElementById('modelSelect');
  const modelText   = document.getElementById('modelTextInput');

  // 2. Switch UI: model dropdown vs free-text input, etc.
  sel.style.display       = isVllm ? 'none' : '';
  modelText.style.display = isVllm ? ''     : 'none';
  if (!isVllm) {
    sel.innerHTML = models.map(m => `<option value="${m.value}">${escHtml(m.label)}</option>`).join('');
  }

  document.getElementById('apiKeyInput').placeholder = PROVIDER_KEY_PLACEHOLDER[provider] || '';
  document.getElementById('apiKeyLabel').textContent  = PROVIDER_KEY_LABEL[provider] || 'API key';
  document.getElementById('deepseekWarningGroup').style.display = provider === 'deepseek' ? '' : 'none';
  document.getElementById('openaiInfoGroup').style.display      = provider === 'openai'   ? '' : 'none';
  document.getElementById('vllmGroup').style.display            = isVllm               ? '' : 'none';

  // 3. Update state.provider AFTER stashing the outgoing credentials,
  //    then load the INCOMING provider's cached credentials so the form
  //    fields reflect the right key/model/baseUrl for this provider.
  state.provider = provider;
  const cached   = (state.providerCredentials || {})[provider] || {};
  state.apiKey   = cached.apiKey  || '';
  state.baseUrl  = cached.baseUrl || '';
  // Reflect into form fields so the user sees the right values
  document.getElementById('apiKeyInput').value = state.apiKey;
  if (isVllm) {
    document.getElementById('vllmBaseUrl').value = state.baseUrl;
    if (cached.model) modelText.value = cached.model;
    state.model = modelText.value || cached.model || '';
  } else {
    if (cached.model && sel.querySelector(`option[value="${CSS.escape(cached.model)}"]`)) {
      sel.value = cached.model;
    }
    state.model = sel.value || (models[0]?.value || '');
  }

  // 4. Clear stale connection-status indicator (test result no longer applies)
  const conn = document.getElementById('connStatus');
  if (conn) conn.style.display = 'none';

  // 5. Re-evaluate the scanned-batch warning — the new provider may be
  //    text-only (DeepSeek), which turns a soft "scans won't get highlights"
  //    notice into a hard "extraction itself will fail" warning.
  if (typeof renderScannedBatchWarning === 'function') renderScannedBatchWarning();

  // 6. Persist if we actually swapped providers
  if (oldProvider && oldProvider !== provider) autoSaveSession();
}

/* ── localStorage auto-save (configuration only — files & results are not persisted) */
const _AUTO_SAVE_KEY = 'paperlens.session.v1';
const _AUTO_SAVE_FIELDS = [
  'mode', 'provider', 'model', 'apiKey', 'baseUrl',
  'question', 'context', 'inputMode',
  'generatedPrompt', 'useTextExtraction',
  'providerCredentials',   // per-provider {apiKey, model, baseUrl} cache
];

function autoSaveSession() {
  try {
    const snapshot = {};
    _AUTO_SAVE_FIELDS.forEach(k => snapshot[k] = state[k]);
    localStorage.setItem(_AUTO_SAVE_KEY, JSON.stringify(snapshot));
  } catch (_) { /* localStorage may be disabled */ }
}

/* Self-healing for sessions corrupted by an older build that stashed
   credentials under the wrong provider slot (e.g. a Gemini "AIza..." key
   ended up under the openai slot when the user opened MASEMiner with a
   prior Gemini session).  Detects keys whose shape clearly belongs to a
   different provider and wipes the bad entry. */
const _PROVIDER_KEY_SHAPE = {
  google:    /^AIza[0-9A-Za-z_\-]{20,}$/,
  openai:    /^sk-[0-9A-Za-z_\-]{20,}$/,
  deepseek:  /^sk-[0-9A-Za-z_\-]{20,}$/,
  anthropic: /^sk-ant-[0-9A-Za-z_\-]{20,}$/,
  // Mistral keys are short opaque strings (no fixed prefix) — they don't
  // look like any other provider's pattern, so we leave them out of the
  // shape table.  _looksWrongForProvider returns false when the active
  // provider has no shape entry, which is exactly what we want for Mistral.
};
function _looksWrongForProvider(provider, apiKey) {
  if (!apiKey || provider === 'vllm') return false;
  // Reverse check: does the key shape match a *different* provider's pattern
  // while clearly NOT matching this provider's pattern?
  const ownPat = _PROVIDER_KEY_SHAPE[provider];
  if (!ownPat || ownPat.test(apiKey)) return false;
  for (const [other, pat] of Object.entries(_PROVIDER_KEY_SHAPE)) {
    if (other === provider) continue;
    if (pat.test(apiKey)) return true;
  }
  return false;
}
function _scrubProviderCredentials(creds) {
  if (!creds || typeof creds !== 'object') return;
  for (const [prov, slot] of Object.entries(creds)) {
    if (slot && _looksWrongForProvider(prov, slot.apiKey)) {
      slot.apiKey = '';
    }
  }
}

function autoRestoreSession() {
  try {
    const raw = localStorage.getItem(_AUTO_SAVE_KEY);
    if (!raw) return false;
    const snapshot = JSON.parse(raw);
    _AUTO_SAVE_FIELDS.forEach(k => {
      if (snapshot[k] !== undefined && snapshot[k] !== null) state[k] = snapshot[k];
    });

    // Backwards-compat: pre-2026 snapshots stored only the flat (apiKey, model,
    // baseUrl) tuple under the active provider.  Hydrate providerCredentials
    // from those flat fields so the rest of the code can rely on the map.
    state.providerCredentials = state.providerCredentials || {};
    if (state.provider && !state.providerCredentials[state.provider]) {
      state.providerCredentials[state.provider] = {
        apiKey:  state.apiKey  || '',
        model:   state.model   || '',
        baseUrl: state.baseUrl || '',
      };
    }
    // Heal wrong-provider keys left behind by an older build.
    _scrubProviderCredentials(state.providerCredentials);
    if (_looksWrongForProvider(state.provider, state.apiKey)) {
      state.apiKey = '';
    }
    // Pre-fill the active provider's flat fields from the credentials map
    // (the source of truth going forward).
    const cached = state.providerCredentials[state.provider] || {};
    if (cached.apiKey)  state.apiKey  = cached.apiKey;
    if (cached.model)   state.model   = cached.model;
    if (cached.baseUrl) state.baseUrl = cached.baseUrl;

    // Reflect into the visible form fields so what the user typed last time
    // actually appears in the inputs (otherwise it's just hidden in state).
    if (state.provider) document.getElementById('providerSelect').value = state.provider;
    onProviderChange();   // populates model dropdown + swaps in cached values
    if (state.question) document.getElementById('questionInput').value = state.question;
    if (state.context)  document.getElementById('contextInput').value  = state.context;
    if (state.generatedPrompt) {
      document.getElementById('manualPromptInput').value = state.generatedPrompt;
      document.getElementById('promptDisplay').textContent = state.generatedPrompt;
      document.getElementById('modelBadge').textContent    = state.model || 'restored';
    }
    // notifyEmail is no longer captured at upload time — it's collected during
    // processing via the in-loading prompt — so nothing to restore here.
    return true;
  } catch (_) {
    return false;
  }
}

function clearAutoSave() {
  try { localStorage.removeItem(_AUTO_SAVE_KEY); } catch (_) { /* ignore */ }
}

/* Load the weekly-synced model list + pricing from /static/models.json.
   Overwrites the baked-in PROVIDER_MODELS / _MODEL_RATES fallbacks.  Any
   failure (missing file, malformed JSON, empty providers) is swallowed —
   the fallbacks stay in place so the app never breaks on a bad sync. */
async function loadModelsConfig() {
  try {
    const res = await fetch('/static/models.json', { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.providers && typeof data.providers === 'object') {
      const next = {};
      for (const [prov, list] of Object.entries(data.providers)) {
        if (!Array.isArray(list)) continue;
        next[prov] = list
          .filter(m => m && m.value)
          .map(m => ({ value: m.value, label: m.label || m.value }));
      }
      // vLLM is free-text entry (no model list) — preserve whatever the
      // fallback had so the provider stays selectable.
      if (!next.vllm) next.vllm = PROVIDER_MODELS.vllm || [];
      // Only adopt if at least one provider actually has models.
      if (Object.values(next).some(l => l.length)) PROVIDER_MODELS = next;
    }
    if (data && data.rates && typeof data.rates === 'object') {
      const rates = {};
      for (const [model, r] of Object.entries(data.rates)) {
        if (r && typeof r.in === 'number' && typeof r.out === 'number') {
          rates[model] = { in: r.in, out: r.out };
        }
      }
      if (Object.keys(rates).length) _MODEL_RATES = rates;
    }
    // Re-render the dropdown now that the live list is in (the initial
    // onProviderChange ran against the fallback).  Preserves the current
    // selection if it still exists in the refreshed list.
    onProviderChange();
  } catch (_) { /* keep baked-in fallback */ }
}

// Initialise the model list on page load
document.addEventListener('DOMContentLoaded', () => {
  onProviderChange(); // populate model list for default provider (fallback)
  loadModelsConfig(); // then overwrite with the weekly-synced list + rates
  initUploadZone();
  initResultDisplay();
  initZoomPan();
  initReuploadPdfZone();
  // Cell-click → evidence-page jump + focused highlight.
  _attachEvidenceClickHandler();

  // Restore the user's last session (provider, model, prompt, etc.) so an
  // accidental refresh doesn't wipe their configuration.
  autoRestoreSession();
  refreshPastBatches();
  loadServerConfig();
  // /maseminer path → hero landing; ?preset=<id> → auto-apply; else inline list
  applyPathOrQueryPreset();

  // The "How does this work?" panel stays closed by default — users open
  // it on demand.  (Previously it auto-opened for first-time visitors;
  // removed because the closed default reads as more polished and the
  // summary text alone is enough to invite a click.)

  // Save on form changes — capture user input as they type.
  ['providerSelect','modelSelect','modelTextInput','vllmBaseUrl',
   'apiKeyInput','questionInput','contextInput','manualPromptInput']
    .forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', () => {
        // Mirror form values back into state before saving
        const provider = document.getElementById('providerSelect').value;
        state.provider = provider;
        state.apiKey   = document.getElementById('apiKeyInput').value.trim();
        state.model    = provider === 'vllm'
          ? document.getElementById('modelTextInput').value.trim()
          : document.getElementById('modelSelect').value;
        state.baseUrl  = provider === 'vllm' ? document.getElementById('vllmBaseUrl').value.trim() : '';
        state.question = document.getElementById('questionInput').value;
        state.context  = document.getElementById('contextInput').value;
        // Always keep the per-provider credential slot in sync so the (provider,
        // apiKey, model, baseUrl) tuple is consistent even after a swap.
        _stashProviderCredentials();
        autoSaveSession();
      });
    });
});

function submitStep2() {
  const provider  = document.getElementById('providerSelect').value;
  const apiKey    = document.getElementById('apiKeyInput').value.trim();
  const isVllm    = provider === 'vllm';
  const model     = isVllm
    ? document.getElementById('modelTextInput').value.trim()
    : document.getElementById('modelSelect').value;
  const baseUrl   = isVllm ? document.getElementById('vllmBaseUrl').value.trim() : '';

  if (!isVllm && !apiKey) { showToast('Please enter your API key.'); return; }
  if (isVllm && !model)   { showToast('Please enter the model name (e.g. meta-llama/Llama-3-8B-Instruct).'); return; }
  if (isVllm && !baseUrl) { showToast('Please enter the vLLM server URL (e.g. http://localhost:8000).'); return; }

  state.apiKey    = apiKey;
  state.model     = model;
  state.provider  = provider;
  state.baseUrl   = baseUrl;

  // Smart return: if the user opened section 2 mid-flow to edit credentials
  // (either via the legacy "Edit setup" button or by clicking the section
  // header directly), don't push them forward — restore them to where they
  // were so the edit doesn't disrupt their place.
  if (state.setupReturnStep) {
    const dest = state.setupReturnStep;
    state.setupReturnStep = null;
    goTo(dest);
    return;
  }
  if (state.step > 2 && state.generatedPrompt) {
    // Header-click edit path: keep them on whatever step they were on.
    goTo(state.step);
    return;
  }
  // Preset path:
  //   * Parameterised MASEMiner presets land on step 3 so the user can
  //     visit the guided builder (refine data sources, variables, etc.)
  //     before the prompt is committed.  Without this, submitStep2 used
  //     to jump straight to step 5 because state.generatedPrompt was
  //     already populated by applyPreset — bypassing the builder UX.
  //   * Other (fixed-prompt) presets keep their old shortcut: prompt
  //     was preset-loaded, no further input needed → skip to review.
  if (state.activePreset
      && typeof isMasemPreset === 'function' && isMasemPreset(state.activePreset)) {
    goTo(3);
    return;
  }
  if (state.activePreset && state.generatedPrompt) {
    document.getElementById('promptSummaryText').textContent  = state.generatedPrompt;
    document.getElementById('promptSummaryModel').textContent = state.model;
    goTo(5);
    updateEvidenceWarning();
    return;
  }

  if (state.mode === 'extraction') {
    document.getElementById('step3Heading').textContent = 'Describe what you want to extract';
    document.getElementById('step3Sub').textContent     = 'Be specific about what information you need from the papers';
    document.getElementById('questionInput').placeholder =
      'E.g., Extract the sample size, mean age, percentage of female participants, and number of factors from each study…';
    document.getElementById('contextInput').placeholder =
      'E.g., Papers use different notations for factors. Age is always reported as mean ± SD…';
  } else {
    document.getElementById('step3Heading').textContent = 'Describe how to label the papers';
    document.getElementById('step3Sub').textContent     = 'Define the categories and criteria for labeling';
    document.getElementById('questionInput').placeholder =
      'E.g., Classify each page as: (A) contains a factor loadings table, (B) contains a correlation matrix, (C) both, or (D) neither…';
    document.getElementById('contextInput').placeholder =
      'E.g., Only count tables reporting items 1–20, not general reliability tables…';
  }
  showStep3Choice();
  goTo(3);
}

async function testConnection() {
  const provider = document.getElementById('providerSelect').value;
  const isVllm   = provider === 'vllm';
  const apiKey   = document.getElementById('apiKeyInput').value.trim();
  const model    = isVllm
    ? document.getElementById('modelTextInput').value.trim()
    : document.getElementById('modelSelect').value;
  const baseUrl  = isVllm ? document.getElementById('vllmBaseUrl').value.trim() : '';
  const btn      = document.getElementById('testConnBtn');
  const status   = document.getElementById('connStatus');

  if (!apiKey && !baseUrl) {
    showToast('Please enter your API key first.');
    return;
  }
  if (isVllm && !model) {
    showToast('Please enter the model name.');
    return;
  }

  btn.disabled    = true;
  btn.textContent = 'Testing…';
  status.className   = 'conn-status conn-status-pending';
  status.style.display = 'flex';
  status.textContent = 'Pinging the provider…';

  try {
    const res  = await fetchScoped('/api/test-connection', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({
        api_key:  apiKey,
        model:    model,
        base_url: baseUrl || undefined,
      }),
    });
    const data = await res.json();
    if (res.ok && data.ok) {
      status.className   = 'conn-status conn-status-ok';
      status.textContent = `✓ Connection works · ${data.model}`;
    } else {
      const msg = data.detail || data.error || 'Connection failed.';
      status.className   = 'conn-status conn-status-err';
      status.textContent = '✕ ' + msg;
    }
  } catch (err) {
    status.className   = 'conn-status conn-status-err';
    status.textContent = '✕ ' + (err.message || 'Network error');
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Test connection';
  }
}

function editSetup() {
  document.getElementById('providerSelect').value = state.provider;
  onProviderChange();  // repopulate model list / toggle vLLM fields
  if (state.provider === 'vllm') {
    document.getElementById('modelTextInput').value = state.model;
    document.getElementById('vllmBaseUrl').value    = state.baseUrl;
  } else {
    document.getElementById('modelSelect').value = state.model;
  }
  document.getElementById('apiKeyInput').value = state.apiKey;
  state.setupReturnStep = state.step;
  goTo(2);
}

/* ──────────────────────────────────────────────────────────
   Step 3 — Prompt method choice → AI or manual
────────────────────────────────────────────────────────── */

function showStep3Choice() {
  document.getElementById('step3Choice').style.display   = '';
  document.getElementById('aiSection').style.display     = 'none';
  document.getElementById('manualSection').style.display = 'none';
  const b = document.getElementById('masemBuilder');
  if (b) b.style.display = 'none';
}

/* "Back" handler for the manual-prompt textarea.  Normally returns the
   user to the generic AI/manual picker (step3Choice), but when a MASEM
   preset is active the user got here by clicking "Edit raw prompt" in
   the MASEMiner builder — they expect Back to take them to the
   Direct/Indirect template cards, not to a picker they never saw.

   Critically, we just toggle visibility — we do NOT call
   ``openMasemBuilder`` here, because that re-fetches the preset and
   resets the builder form + overwrites the user's custom-edited
   prompt.  Switching task (clicking a different Direct/Indirect card)
   is what regenerates the prompt; navigating back must not. */
function goBackFromManualPrompt() {
  if (typeof isMasemPreset === 'function' && isMasemPreset(state.activePreset)) {
    document.getElementById('manualSection').style.display = 'none';
    const builder = document.getElementById('masemBuilder');
    if (builder) builder.style.display = '';
    return;
  }
  showStep3Choice();
}

function setInputMode(mode) {
  state.inputMode = mode;
  const isManual = mode === 'manual';
  document.getElementById('step3Choice').style.display   = 'none';
  document.getElementById('aiSection').style.display     = isManual ? 'none' : '';
  document.getElementById('manualSection').style.display = isManual ? ''     : 'none';
  const b = document.getElementById('masemBuilder');
  if (b) b.style.display = 'none';
}

function submitStep3() {
  const question = document.getElementById('questionInput').value.trim();
  const context  = document.getElementById('contextInput').value.trim();
  if (!question) { showToast('Please describe your task before continuing.'); return; }
  state.question = question;
  state.context  = context;
  callGenerateAPI();
}

function useManualPrompt() {
  const prompt = document.getElementById('manualPromptInput').value.trim();
  if (!prompt) { showToast('Please enter a prompt.'); return; }
  state.generatedPrompt = prompt;
  document.getElementById('promptDisplay').textContent = prompt;
  document.getElementById('modelBadge').textContent    = 'manual';
  resetCopyBtn('copyBtn');
  if (document.body.classList.contains('is-maseminer')) {
    confirmPrompt();
  } else {
    goTo(5);
    updateEvidenceWarning();
  }
}

/* ── Evidence-schema warning + adapt ──────────────────────────────────── */

function _hasEvidenceSchema(prompt) {
  // Mirror server-side check: ≥3 of (evidence, snippet, page, source)
  const p = (prompt || '').toLowerCase();
  let hits = 0;
  for (const tok of ['evidence', 'snippet', 'page', 'source']) {
    if (p.includes(tok)) hits++;
  }
  return hits >= 3;
}

function updateEvidenceWarning() {
  const el = document.getElementById('promptEvidenceWarning');
  if (!el) return;
  el.style.display = _hasEvidenceSchema(state.generatedPrompt) ? 'none' : 'flex';
}

async function adaptPromptForEvidence() {
  const btn = document.getElementById('adaptPromptBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Adapting…'; }
  try {
    const res = await fetchScoped('/api/adapt-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key:  state.apiKey,
        model:    state.model,
        prompt:   state.generatedPrompt,
        base_url: state.baseUrl || undefined,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Failed to adapt prompt.');

    state.generatedPrompt = data.prompt;
    document.getElementById('promptDisplay').textContent = data.prompt;
    updateEvidenceWarning();
    autoSaveSession();
    showToast('Prompt adapted — evidence requirement added.', 'success');
  } catch (err) {
    showToast(err.message);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = '&#10024;&ensp;Adapt prompt for evidence'; }
  }
}

/* ──────────────────────────────────────────────────────────
   Prompt generation API (steps 4 → 5)
────────────────────────────────────────────────────────── */

async function callGenerateAPI() {
  goTo(4);
  try {
    const res = await fetchScoped('/api/generate-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: state.apiKey, model: state.model, mode: state.mode,
        question: state.question, context: state.context,
        base_url: state.baseUrl || undefined,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Failed to generate prompt.');
    state.generatedPrompt = data.prompt;
    document.getElementById('promptDisplay').textContent = data.prompt;
    document.getElementById('modelBadge').textContent    = data.model_used;
    resetCopyBtn('copyBtn');
    if (document.body.classList.contains('is-maseminer')) {
      confirmPrompt();
    } else {
      goTo(5);
      updateEvidenceWarning();
    }
    autoSaveSession();
  } catch (err) {
    showToast(err.message);
    goTo(3);
  }
}

/* ──────────────────────────────────────────────────────────
   Step 5 — Review prompt
────────────────────────────────────────────────────────── */

function copyPrompt() { copyToClipboard(state.generatedPrompt, 'copyBtn'); }

function regenerate() {
  if (state.inputMode === 'manual') goTo(3);
  else callGenerateAPI();
}

/* First vision-capable model for a provider, or null if it has none.
   Used to suggest a same-provider alternative when the user has a
   text-only model selected but wants image (VLM) extraction. */
function _suggestVisionModel(provider) {
  const vis = _visionModelsForProvider(provider);
  return vis.length ? vis[0] : null;   // {value, label}
}

function confirmPrompt() {
  document.getElementById('promptSummaryText').textContent  = state.generatedPrompt;
  document.getElementById('promptSummaryModel').textContent = state.model;
  const note = document.getElementById('visionNote');
  if (!isVisionModel(state.model)) {
    // Text-only model — explain it won't do image analysis, and point
    // the user at a vision-capable model (preferring the same provider).
    const sug = _suggestVisionModel(state.provider);
    const suggestion = sug
      ? ` Not compatible with image (VLM) extraction — for scanned PDFs or image-only tables, switch to a vision model such as ${sug.label} (same provider) in step 1.`
      : ` Not compatible with image (VLM) extraction, and this provider has no vision model — switch to OpenAI, Anthropic, Gemini, or Mistral (Pixtral) in step 1 for image-based extraction.`;
    note.textContent =
      `ℹ️ ${state.model} uses text extraction from the PDF's text layer instead of image analysis.${suggestion}`;
    note.style.display = 'block';
  } else {
    note.style.display = 'none';
  }
  // Show parsing method toggle only for vision-capable models (DeepSeek
  // always uses the text path).  Default is now text — vision kicks in
  // automatically per-paper when the upfront probe says the PDF is scanned.
  const parsingGroup = document.getElementById('parsingMethodGroup');
  if (parsingGroup) {
    parsingGroup.style.display = isVisionModel(state.model) ? '' : 'none';
    const textRadio = parsingGroup.querySelector('input[value="text"]');
    if (textRadio) { textRadio.checked = true; state.useTextExtraction = true; }
  }
  goTo(6);
}

function onParseMethodChange(radio) {
  state.useTextExtraction = radio.value === 'text';
  renderCostEstimate();
  renderSizeWarning();
}

/* ──────────────────────────────────────────────────────────
   Step 6 — Multi-file upload
────────────────────────────────────────────────────────── */

function initUploadZone() {
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('click', () => document.getElementById('pdfInput').click());
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    addFiles(Array.from(e.dataTransfer.files));
  });
}

function handleFileSelect(event) {
  addFiles(Array.from(event.target.files));
  // Do NOT clear event.target.value here — doing so can invalidate File object
  // references in Safari and some Firefox builds before the upload completes.
}

async function addFiles(files) {
  const existing = new Set(state.selectedFiles.map(f => f.name + f.size));
  const cap      = config.maxBatchPapers;
  const maxMb    = Math.round(config.maxPdfBytes / (1024 * 1024));
  for (const file of files) {
    if (state.selectedFiles.length >= cap) {
      showToast(`Batch limit reached: only ${cap} papers per batch. Remove some or start another extraction.`);
      break;
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast(`"${file.name}" is not a PDF — skipped.`);
      continue;
    }
    if (file.size > config.maxPdfBytes) {
      showToast(`"${file.name}" exceeds ${maxMb} MB — skipped.`);
      continue;
    }
    const key = file.name + file.size;
    if (existing.has(key)) continue; // skip duplicate

    // Read into an ArrayBuffer immediately so the data is captured regardless
    // of what happens to the input element or file reference later.
    const buffer = await file.arrayBuffer();
    const entry  = {
      name: file.name,
      size: file.size,
      buffer,                                // stable copy of the bytes
      blob: new Blob([buffer], { type: 'application/pdf' }),
      probe: null,                           // filled in by _probePdf below
      probing: true,
    };
    state.selectedFiles.push(entry);
    existing.add(key);
    // Fire and forget: kick off the upfront text-layer probe.  Doesn't block
    // adding more files; the per-file badge updates as results come in.
    _probePdf(entry);
  }
  renderFileList();
}

/* Send the file's bytes to /api/check-pdf so we can tell the user upfront
   whether the PDF is text-readable or scanned/image-only.  Result is stored
   on the file entry so the badge in the upload list and the warning before
   extraction both see it. */
async function _probePdf(entry) {
  try {
    const fd = new FormData();
    fd.append('pdf', entry.blob, entry.name);
    const res = await fetchScoped('/api/check-pdf', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    entry.probe = await res.json();
  } catch (err) {
    // Probe is a hint, not a hard requirement — fall back to "unknown" and
    // let extraction proceed.  Vision-mode extraction works regardless.
    entry.probe = { error: String(err && err.message || err) };
  } finally {
    entry.probing = false;
    renderFileList();
    renderScannedBatchWarning();
  }
}

function removeFile(index) {
  state.selectedFiles.splice(index, 1);
  renderFileList();
}

function renderFileList() {
  const list = document.getElementById('fileList');
  const zone = document.getElementById('uploadZone');
  const btn  = document.getElementById('extractBtn');

  if (state.selectedFiles.length === 0) {
    list.innerHTML = '';
    list.style.display = 'none';
    zone.style.display = '';
    btn.disabled = true;
    renderCostEstimate();
    renderSizeWarning();
    return;
  }

  zone.style.display = 'none';
  list.style.display = 'block';
  btn.disabled = false;

  list.innerHTML = `
    <div class="file-list-header">
      <span>${state.selectedFiles.length} file${state.selectedFiles.length > 1 ? 's' : ''} selected</span>
      <button class="file-list-add" onclick="document.getElementById('pdfInput').click()">+ Add more</button>
    </div>
    ${state.selectedFiles.map((f, i) => `
      <div class="file-list-item">
        <span class="file-icon">📄</span>
        <span class="file-name">${escHtml(f.name)}</span>
        ${_renderProbeBadge(f)}
        <span class="file-size">${formatBytes(f.size)}</span>
        <button class="file-remove" onclick="removeFile(${i})" title="Remove">✕</button>
      </div>
    `).join('')}`;

  renderCostEstimate();
  renderSizeWarning();
  renderScannedBatchWarning();
}

/* Compact per-file badge:
     · "Probing…"   while the upfront /api/check-pdf call is in flight
     · "Text PDF"   green, when the PDF has a usable text layer
     · "Scanned — vision only" amber, when image-only (no rect highlights)
     · "Mixed"      when some pages are scanned and some aren't
*/
function _renderProbeBadge(file) {
  if (file.probing)                  return '<span class="probe-badge probe-loading">Probing…</span>';
  const p = file.probe;
  if (!p || p.error)                 return '<span class="probe-badge probe-unknown" title="Could not inspect this PDF — vision extraction will still work">Unknown</span>';
  const total   = p.total_pages   || 0;
  const scanned = (p.scanned_pages || []).length;
  if (scanned === 0)                 return '<span class="probe-badge probe-text" title="Has a machine-readable text layer">Text PDF</span>';
  if (scanned === total)             return '<span class="probe-badge probe-scanned" title="Image-only PDF — vision extraction will still work, but page highlights cannot be drawn">Scanned · vision only</span>';
  return `<span class="probe-badge probe-mixed" title="${scanned} of ${total} pages have no text layer — those pages won't get highlights">Mixed (${scanned}/${total} scanned)</span>`;
}

/* Top-of-list summary banner.  Default extraction mode is text — for any
   selected PDF that's scanned (no text layer), processPaper auto-falls
   back to vision for THAT file.  This banner explains the routing so users
   know what to expect (especially: no rect highlights for scanned PDFs).
   When the user has picked DeepSeek (text-only with no vision fallback)
   it becomes a hard warning instead. */
function renderScannedBatchWarning() {
  const el = document.getElementById('scannedBatchWarning');
  if (!el) return;
  const files = state.selectedFiles || [];
  const scanned = files.filter(f => f.probe && !f.probe.error
                                 && (f.probe.scanned_pages || []).length > 0);
  if (scanned.length === 0) { el.style.display = 'none'; return; }

  const fullScans       = scanned.filter(f => (f.probe.scanned_pages || []).length === f.probe.total_pages);
  const providerHasVis  = state.provider !== 'deepseek';

  let msg;
  if (!providerHasVis && fullScans.length > 0) {
    // DeepSeek + scanned = extraction will return nothing.  Hard block.
    msg = `<strong>DeepSeek can't read ${fullScans.length} of ${scanned.length} ` +
          `selected PDF${scanned.length !== 1 ? 's' : ''}</strong> — they're scans ` +
          `with no text layer, and DeepSeek is text-only. Switch to a vision-capable ` +
          `model (e.g. GPT-4o or Gemini) to process them.`;
    el.classList.remove('warning-soft');
    el.classList.add('warning-hard');
  } else {
    msg = `<strong>${scanned.length} of ${files.length} selected PDF${files.length !== 1 ? 's are' : ' is'} ` +
          `scanned</strong> (no text layer). ${scanned.length === 1 ? 'It' : 'They'} will ` +
          `automatically use vision-mode extraction instead of text parsing — slower and ` +
          `more expensive, and page highlights won't be drawn for ${scanned.length === 1 ? 'it' : 'these files'}.`;
    el.classList.remove('warning-hard');
    el.classList.add('warning-soft');
  }
  el.innerHTML = `<span class="probe-warning-icon">&#9432;</span><div>${msg}</div>`;
  el.style.display = 'flex';
}

/* ── Cost estimator ─────────────────────────────────────────────────────────
   USD per 1M tokens.  BAKED-IN FALLBACK — loadModelsConfig() overwrites this
   from /static/models.json (regenerated weekly).  Declared ``let`` so the
   loader can reassign it; a model with no entry here shows "no estimate"
   rather than breaking. */
let _MODEL_RATES = {
  'gpt-5.5':             {in: 5.00,  out: 30.00},
  'gpt-5':               {in: 1.25,  out: 10.00},
  'gpt-5-mini':          {in: 0.25,  out: 2.00},
  'gpt-5-nano':          {in: 0.05,  out: 0.40},
  'gpt-4o':              {in: 2.50,  out: 10.00},
  'gpt-4o-mini':         {in: 0.15,  out: 0.60},
  'gpt-4-turbo':         {in: 10.00, out: 30.00},
  'gemini-2.5-pro':      {in: 1.25,  out: 5.00},
  'gemini-2.5-flash':    {in: 0.075, out: 0.30},
  'gemini-2.0-flash':    {in: 0.075, out: 0.30},
  'deepseek-chat':       {in: 0.27,  out: 1.10},
  'deepseek-reasoner':   {in: 0.55,  out: 2.19},
  'pixtral-large-latest':{in: 2.00,  out: 6.00},
  'pixtral-12b-2409':    {in: 0.15,  out: 0.15},
  'mistral-large-latest':{in: 2.00,  out: 6.00},
  'mistral-small-latest':{in: 0.20,  out: 0.60},
};

/* Fallback tokens-per-page when we don't have the page's real pixel
   dimensions (probe missing / errored).  When dimensions ARE known we
   compute a model-specific count via _visionTokensForPage below. */
const _VISION_TOKENS_PER_PAGE = 1450;   // ~US-letter at 200 DPI, OpenAI high-detail

/* Estimate the input-token cost of ONE page image for a given model,
   from its rendered pixel dimensions (width, height at EXTRACTION_DPI).

   Image tokenisation is model-family-specific and the providers change
   it over time, so these are best-effort formulas — accurate for the
   common cases, with a flat fallback for anything unrecognised:

   - OpenAI "tile" models (gpt-4o, gpt-4-turbo): downscale to fit
     2048², then shortest side to 768, count 512² tiles, 85 + 170·tiles.
   - OpenAI "patch" models (gpt-5 family): 32×32-pixel patches, capped.
   - Gemini: ~258 tokens per 768² tile (flat 258 for small images).
   - Anything else / unknown: the flat per-page fallback. */
function _visionTokensForPage(model, w, h) {
  w = w || 1700; h = h || 2200;
  const m = (model || '').toLowerCase();

  // OpenAI patch-based (GPT-5 family): 1 token per 32×32 patch, capped.
  if (m.startsWith('gpt-5')) {
    const patches = Math.ceil(w / 32) * Math.ceil(h / 32);
    return Math.min(patches, 1536);
  }
  // OpenAI tile-based (gpt-4o, gpt-4-turbo, gpt-4o-mini).
  if (m.startsWith('gpt-4')) {
    let pw = w, ph = h;
    const fit = 2048 / Math.max(pw, ph);
    if (fit < 1) { pw = Math.round(pw * fit); ph = Math.round(ph * fit); }
    const shortest = Math.min(pw, ph);
    if (shortest > 768) { const s = 768 / shortest; pw = Math.round(pw * s); ph = Math.round(ph * s); }
    const tiles = Math.ceil(pw / 512) * Math.ceil(ph / 512);
    const base  = 85 + 170 * tiles;
    // gpt-4o-mini bills image tokens at a large multiple of the base.
    return m.includes('mini') ? base * 33 : base;
  }
  // Gemini: 258 tokens per 768×768 tile (flat 258 when it fits in one).
  if (m.startsWith('gemini')) {
    const tiles = Math.max(1, Math.ceil(w / 768) * Math.ceil(h / 768));
    return tiles * 258;
  }
  // Unknown model — flat fallback.
  return _VISION_TOKENS_PER_PAGE;
}
/* Rough chars-per-byte ratio for native-text PDFs (extracted text is
   usually 25-40% of the file size).  Then ~4 chars per token. */
const _TEXT_CHARS_PER_BYTE = 0.30;
const _TEXT_CHARS_PER_TOKEN = 4;
/* Output is dominated by extracted JSON.  4k tokens covers most prompts;
   labeling is much smaller but vision factor-loadings is on the high end. */
const _OUTPUT_TOKENS_PER_PAPER = 4000;

function _estimatePagesFromBytes(bytes) {
  // Rough map from file size to page count.  100 KB ≈ 5 pages, 1 MB ≈ 20,
  // 5 MB ≈ 50 (capped at the server's MAX_PAGES limit).
  const kb = bytes / 1024;
  const pages = Math.max(1, Math.round(kb / 80) + 2);
  return Math.min(pages, 40);
}

function estimateBatchCostUsd() {
  // Self-hosted (vLLM / Ollama) — cost is the user's own compute, not USD.
  if (state.provider === 'vllm') return { selfHosted: true };
  const rate = _MODEL_RATES[state.model];
  // No rate on file for this model (e.g. the weekly model sync added a
  // model whose price LiteLLM doesn't list yet).  Signal "no estimate"
  // rather than breaking — renderCostEstimate hides the dollar figure.
  if (!rate) return { noRate: true };

  const useText = state.useTextExtraction || state.provider === 'deepseek';
  let inputTokens = 0;
  let exact = true;   // true while every paper used real probe data
  for (const f of state.selectedFiles) {
    // Prefer the upfront /api/check-pdf probe (exact page count + text
    // chars + per-page pixel dims) over the file-size heuristic.  The
    // probe runs on upload, so its data is usually ready by estimate time.
    const probe = (f.probe && !f.probe.error) ? f.probe : null;
    if (useText) {
      if (probe && typeof probe.total_text_chars === 'number') {
        inputTokens += probe.total_text_chars / _TEXT_CHARS_PER_TOKEN;
      } else {
        inputTokens += (f.size * _TEXT_CHARS_PER_BYTE) / _TEXT_CHARS_PER_TOKEN;
        exact = false;
      }
    } else {
      if (probe && Array.isArray(probe.page_dims_px) && probe.page_dims_px.length) {
        for (const [w, h] of probe.page_dims_px) {
          inputTokens += _visionTokensForPage(state.model, w, h);
        }
      } else if (probe && typeof probe.total_pages === 'number') {
        inputTokens += probe.total_pages * _VISION_TOKENS_PER_PAGE;
        exact = false;
      } else {
        inputTokens += _estimatePagesFromBytes(f.size) * _VISION_TOKENS_PER_PAGE;
        exact = false;
      }
    }
    // Add prompt overhead (~ length of the generated prompt) per paper
    inputTokens += (state.generatedPrompt?.length || 4000) / 4;
  }
  const outputTokens = _OUTPUT_TOKENS_PER_PAPER * state.selectedFiles.length;
  const usd = (inputTokens / 1e6) * rate.in + (outputTokens / 1e6) * rate.out;
  return { usd, inputTokens, outputTokens, useText, exact };
}

function renderCostEstimate() {
  const el = document.getElementById('costEstimate');
  if (!el) return;
  if (state.selectedFiles.length === 0) {
    el.style.display = 'none';
    return;
  }
  const est = estimateBatchCostUsd();
  if (!est || est.selfHosted) {
    // Self-hosted (vLLM / Ollama) — cost is the user's own compute.
    el.style.display = 'flex';
    el.innerHTML = `<span class="cost-est-icon">≈</span>
      <span><strong>Self-hosted model</strong> — runs on your own server, no per-token cost.</span>`;
    return;
  }
  if (est.noRate) {
    // We have no published price for this model (common right after the
    // weekly model sync adds a brand-new model).  Hide the dollar figure
    // entirely rather than show a misleading number or break.
    el.style.display = 'flex';
    el.innerHTML = `<span class="cost-est-icon">≈</span>
      <span>No price on file for <strong>${escHtml(state.model)}</strong> yet — cost estimate unavailable. Actual token usage is still reported after the run.</span>`;
    return;
  }
  // Show a ±50 % range to communicate genuine uncertainty
  const spread = est.exact ? 0.25 : 0.5;
  const low  = est.usd * (1 - spread);
  const high = est.usd * (1 + spread);
  const fmt  = n => n < 0.01 ? '< 0.01' : n.toFixed(2);
  const n    = state.selectedFiles.length;
  el.style.display = 'flex';
  el.innerHTML =
    `<span class="cost-est-icon">≈</span>` +
    `<span><strong>Estimated cost: \$${fmt(low)} – \$${fmt(high)}</strong> ` +
    `for ${n} paper${n !== 1 ? 's' : ''} on ${escHtml(state.model)} ` +
    `(${est.useText ? 'text' : 'vision'} mode). ` +
    `<span class="cost-est-note">${est.exact
      ? 'Input measured from the PDF; only output size is estimated.'
      : 'Approximate — actual usage may vary.'}</span></span>`;
}

/* ── Vision request-size warning ───────────────────────────────────────────
   For vision mode, every page becomes a base64-encoded PNG embedded in the
   request body.  OpenAI enforces ~50 MB per request, and the base64 payload
   for image-heavy PDFs can blow past that.  We compute a conservative
   per-paper estimate from file size and warn before submission. */

const _PAGE_SIZE_AT_200_DPI = 600 * 1024;       // ~600 KB/page (typical research PDF)
const _BASE64_OVERHEAD      = 4 / 3;             // base64 inflates bytes by 33 %
const _JSON_ENVELOPE_OVERHEAD = 1.05;            // request scaffolding + headers

function _estimateVisionRequestMb(fileBytes, dpi = 200) {
  const kb    = fileBytes / 1024;
  const pages = Math.max(1, Math.min(40, Math.round(kb / 80) + 2));
  // Bytes scale with rendered area, which scales with DPI²
  const dpiArea = (dpi / 200) ** 2;
  const bytes   = pages * _PAGE_SIZE_AT_200_DPI * dpiArea
                * _BASE64_OVERHEAD * _JSON_ENVELOPE_OVERHEAD;
  return bytes / (1024 * 1024);
}

// Per-paper threshold above which we warn.  Set well under OpenAI's 50 MB
// per-request hard limit because image-heavy PDFs render 2-3× our estimate.
const _SIZE_WARN_PER_PAPER_MB = 25;
const _SIZE_WARN_BATCH_MB     = 90;

function renderSizeWarning() {
  const el = document.getElementById('sizeWarning');
  if (!el) return;
  // Only matters for vision mode; skip the warning if the user already
  // chose text extraction (or the provider forces it, e.g. DeepSeek).
  if (state.useTextExtraction || state.provider === 'deepseek' || state.selectedFiles.length === 0) {
    el.style.display = 'none';
    return;
  }
  const perPaper = state.selectedFiles.map(f => ({
    name: f.name,
    mb:   _estimateVisionRequestMb(f.size),
  }));
  const oversized = perPaper.filter(p => p.mb > _SIZE_WARN_PER_PAPER_MB);
  const total     = perPaper.reduce((s, p) => s + p.mb, 0);
  const tooBigBatch = total > _SIZE_WARN_BATCH_MB;

  if (oversized.length === 0 && !tooBigBatch) {
    el.style.display = 'none';
    return;
  }

  // Warn-by-warn: list the offenders, suggest a fix
  let body = '';
  if (oversized.length) {
    const list = oversized
      .slice(0, 4)
      .map(p => `&middot; ${escHtml(p.name)} (~${Math.round(p.mb)} MB request)`)
      .join('<br>');
    const more = oversized.length > 4 ? `<br>&middot; …and ${oversized.length - 4} more` : '';
    body +=
      `<strong>Some papers may be too large for vision mode.</strong> The request body ` +
      `includes the full page render at 200 DPI, and very long or image-heavy PDFs can ` +
      `exceed the provider's per-request size limit (~50 MB on OpenAI).<br>` +
      `<div class="size-warning-list">${list}${more}</div>`;
  } else {
    body +=
      `<strong>Total batch may exceed safe request sizes.</strong> ` +
      `Combined estimate: ~${Math.round(total)} MB across ${perPaper.length} papers in vision mode. ` +
      `Some providers (notably OpenAI) cap requests around 50 MB per paper.<br>`;
  }
  body +=
    `<div class="size-warning-actions">` +
    `<button class="btn btn-primary btn-sm" onclick="switchToTextExtraction()">Switch to text extraction</button> ` +
    `<span class="size-warning-note">or remove the largest papers and try again.</span>` +
    `</div>`;

  el.innerHTML =
    `<span class="size-warning-icon">&#9888;</span><div class="size-warning-body">${body}</div>`;
  el.style.display = 'flex';
}

function switchToTextExtraction() {
  state.useTextExtraction = true;
  // Reflect into the radio buttons
  const textRadio = document.querySelector('input[name="parseMethod"][value="text"]');
  if (textRadio) textRadio.checked = true;
  renderCostEstimate();
  renderSizeWarning();
  showToast('Switched to text extraction — vision-mode size warnings cleared.', 'success');
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function submitUpload() {
  if (state.selectedFiles.length === 0) { showToast('Please select at least one PDF.'); return; }
  if (state.selectedFiles.length > config.maxBatchPapers) {
    showToast(`Batch limit is ${config.maxBatchPapers} papers. Please remove ${state.selectedFiles.length - config.maxBatchPapers} file(s) or split into multiple batches.`);
    return;
  }

  // crypto.randomUUID is only available in secure contexts (HTTPS or localhost).
  // Fall back to a manual UUID v4 if the browser doesn't expose it.
  const uuid = () => (crypto && crypto.randomUUID
    ? crypto.randomUUID()
    : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
        const r = (Math.random() * 16) | 0;
        return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
      }));

  try {
    // Incremental upload: when the user has already processed some papers
    // in this session and adds more files, keep the existing papers' state
    // (results, edits, page images) and only run the freshly-added ones.
    // We index existing papers by filename+size so re-adding the *same*
    // file doesn't double-process it.
    const existingByKey = new Map();
    for (const p of state.papers || []) {
      // Match what the file probe used so the keys agree.
      const sf = (state.selectedFiles || []).find(f => f.name === p.filename);
      if (sf) existingByKey.set(p.filename + sf.size, p);
    }

    const newPapers = [];
    const toProcess = [];
    for (const f of state.selectedFiles) {
      const key = f.name + f.size;
      const existing = existingByKey.get(key);
      if (existing && (existing.status === 'done' || existing.status === 'processing')) {
        // Keep its result/edits as-is.
        newPapers.push(existing);
        continue;
      }
      // First-time file (or previous run errored / was pending) — make a
      // fresh paper object and queue it for processing.
      const paper = {
        id: uuid(),
        blob: f.blob,
        filename: f.name,
        // Carry the upfront PDF probe so processPaper can auto-fall back to
        // vision when the text layer is unusable.
        probe: f.probe || null,
        status: 'pending',
        phase: null,
        result: '',
        rawResponse: null,
        jobId: null,
        pageImages: [],
        pageImagesFetched: false,
        entries: null,
        entryIndex: 0,
        evidencePages: [],
        evidencePageIdx: 0,
        evidenceCount: null,
        tokenUsage: null,
        pagesProcessed: 0,
        error: null,
        overrides: {},
        viewMode: 'parsed',     // 'parsed' = formatted editable view; 'raw' = model JSON
        browseAllPagesIdx: 0,   // when there's no evidence, used to flip through every captured page
      };
      newPapers.push(paper);
      toProcess.push(paper);
    }

    if (toProcess.length === 0) {
      // Nothing new to do.  Jump straight to the results view so the user
      // can review their existing papers without a noop "extract" click.
      state.papers = newPapers;
      state.activePaperId = state.papers[0]?.id || null;
      renderPaperSidebar();
      if (state.activePaperId) displayPaper(state.papers[0]);
      goTo(8);
      return;
    }

    // Only allocate a fresh batch id when there's actual new work — that
    // way the new papers form their own batch (separate completion email,
    // separate History row) and don't piggy-back onto the previous run.
    state.batchId    = uuid();
    state.notifyEmail = '';   // captured later via the in-loading-screen prompt

    state.papers = newPapers;
    // Incremental run: keep the user on step 8 viewing whatever they had
    // open; new papers stream into the sidebar as they finish.  First-time
    // run: clear active id so the first completion takes focus.
    const incremental = existingByKey.size > 0
                     && newPapers.some(p => p.status === 'done');
    if (!incremental) state.activePaperId = null;
    state._incrementalRun = incremental;
  } catch (err) {
    console.error('[submitUpload] failed to build papers queue:', err);
    showToast('Could not start extraction: ' + err.message);
    return;
  }

  // Surface any uncaught error from the async queue
  processQueue().catch(err => {
    console.error('[processQueue] uncaught error:', err);
    showToast('Extraction failed: ' + err.message);
  });
}

/* ──────────────────────────────────────────────────────────
   Processing queue
────────────────────────────────────────────────────────── */

// Countdown sleep — resolves after `ms` ms, calling onTick(remainingSeconds) every 500 ms.
function sleepWithCountdown(ms, onTick) {
  return new Promise(resolve => {
    const end = Date.now() + ms;
    function tick() {
      const remaining = Math.max(0, Math.ceil((end - Date.now()) / 1000));
      onTick(remaining);
      if (remaining <= 0) { resolve(); return; }
      setTimeout(tick, 500);
    }
    tick();
  });
}

async function processQueue() {
  const incremental = state._incrementalRun === true;
  state._incrementalRun = false;

  if (!incremental) {
    const n = state.papers.length;
    document.getElementById('loadingTitle').textContent   = 'Extracting data\u2026';
    document.getElementById('extractingNote').textContent =
      `Submitting ${n} paper${n > 1 ? 's' : ''} for processing\u2026`;
  }

  // Reset the inline email prompt to its default form state for this batch
  const emailWrap   = document.getElementById('emailPrompt');
  const emailStatus = document.getElementById('emailPromptStatus');
  const emailForm   = emailWrap?.querySelector('.email-prompt-form');
  const emailInput  = document.getElementById('batchEmailInput');
  if (emailWrap)   emailWrap.style.display   = 'none';
  if (emailStatus) emailStatus.style.display = 'none';
  if (emailForm)   emailForm.style.display   = '';
  if (emailInput)  emailInput.value          = '';
  // Surface 'Taking a long time?' after ~8s so it isn't in the user's face
  // when the batch is short.  Skipped for incremental runs (we stay on the
  // results view, not the loading screen, so the prompt has nowhere to go).
  const emailPromptTimer = incremental ? null : setTimeout(() => {
    if (emailWrap && state.step === 7) emailWrap.style.display = 'block';
  }, 8000);

  if (incremental) {
    // Stay on the results view; sidebar already shows the new papers as
    // 'pending' and will animate them through 'processing' \u2192 'done'.
    goTo(8);
    renderPaperSidebar();
  } else {
    goTo(7);
  }

  // Submit all jobs in parallel (server runs them concurrently via asyncio).
  await Promise.all(state.papers.map(p =>
    p.status === 'pending' ? processPaper(p) : Promise.resolve()
  ));

  clearTimeout(emailPromptTimer);
  if (emailWrap) emailWrap.style.display = 'none';

  // If every paper errored before any result was shown, fall back to upload.
  if (state.activePaperId === null) {
    const allErrored = state.papers.every(p => p.status === 'error');
    if (allErrored) goTo(6);
  }
}

/* ── In-loading email submit ───────────────────────────────────────────────── */
async function submitBatchEmail(event) {
  event.preventDefault();
  const input  = document.getElementById('batchEmailInput');
  const btn    = document.getElementById('batchEmailBtn');
  const form   = input.closest('.email-prompt-form');
  const status = document.getElementById('emailPromptStatus');
  const email  = input.value.trim();
  if (!state.batchId) {
    status.textContent  = 'No active batch \u2014 refresh and try again.';
    status.className    = 'email-prompt-status email-prompt-err';
    status.style.display = '';
    return;
  }
  btn.disabled    = true;
  btn.textContent = 'Saving\u2026';
  try {
    const res  = await fetchScoped(`/api/batches/${state.batchId}/email`, {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ email }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Could not save email.');
    state.notifyEmail = email;
    form.style.display     = 'none';
    status.style.display   = '';
    status.className       = 'email-prompt-status email-prompt-ok';
    status.textContent     = data.sent_now
      ? '\u2713 Email sent! Check your inbox.'
      : "\u2713 You're set. We'll email you when the batch is done.";
  } catch (err) {
    status.style.display = '';
    status.className     = 'email-prompt-status email-prompt-err';
    status.textContent   = '\u2717 ' + err.message;
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Notify me';
  }
}

async function processPaper(paper) {
  paper.status = 'processing';
  renderPaperSidebar();

  // Per-paper routing.  Priority:
  //   1. ``paper.forceMode`` — explicit per-paper override set by the
  //      split "Re-run · text" / "Re-run · vision" buttons.  Wins
  //      unconditionally so the user can always force either mode.
  //   2. Otherwise the global ``state.useTextExtraction`` flag, with an
  //      auto-fallback to vision when the upfront probe says the PDF
  //      has no usable text layer (text extraction would return empty).
  let useText;
  if (paper.forceMode === 'text')   useText = true;
  else if (paper.forceMode === 'vision') useText = false;
  else {
    useText = state.useTextExtraction;
    if (useText && paper.probe && !paper.probe.error
        && paper.probe.text_layer_present === false) {
      useText = false;
      paper.autoVisionFallback = true;   // surfaced in the UI as a small note
    }
  }
  // Record the mode actually used so the result UI can show the
  // appropriate re-run buttons (split vs. single) afterwards.
  paper.lastModeUsed = useText ? 'text' : 'vision';

  // Per-paper model override.  Set by the VLM-rerun picker so a
  // single text-only paper can be re-run against a vision-capable
  // model without touching the global state.model.  Falls back to
  // state.model when not set.
  const effectiveModel = paper.forceModel || state.model;
  paper.lastModelUsed  = effectiveModel;

  const form = new FormData();
  form.append('api_key',             state.apiKey);
  form.append('model',               effectiveModel);
  form.append('prompt',              state.generatedPrompt);
  form.append('use_text_extraction', useText ? '1' : '0');
  if (state.baseUrl) form.append('base_url', state.baseUrl);
  if (state.batchId) form.append('batch_id', state.batchId);
  // Email is collected after submission via /api/batches/<id>/email — see submitBatchEmail()
  form.append('pdf',                 paper.blob, paper.filename);

  // Step 1: submit job, get job_id
  let jobId;
  try {
    const res  = await fetchScoped('/api/extract', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Failed to submit job.');
    jobId = data.job_id;
    paper.jobId = jobId;
  } catch (err) {
    paper.status = 'error';
    paper.error  = err.message;
    showToast(`"${paper.filename}": ${err.message}`);
    renderPaperSidebar();
    return;
  }

  // Step 2: poll until done or error
  try {
    const result = await pollJob(jobId, (status, data) => {
      paper.phase = (data && data.phase) || null;
      if (state.step === 7) {
        const done = state.papers.filter(p =>
          p.status === 'done' || p.status === 'error' || p.status === 'cancelled'
        ).length;
        const phase = paper.phase ? ` \u00b7 ${paper.phase}` : '';
        document.getElementById('extractingNote').textContent =
          `${done} of ${state.papers.length} complete \u00b7 "${paper.filename}" ${status}${phase}\u2026`;
      }
      renderPaperSidebar();
    });

    if (result.status === 'cancelled') throw new Error('Cancelled.');
    if (result.status === 'error')     throw new Error(result.error || 'Extraction failed.');

    if (result.result) paper.rawResponse = result.result;

    if (result.finish_reason === 'content_filter') {
      throw new Error('Blocked by content filter \u2014 try a different model or fewer pages.');
    }

    paper.status            = 'done';
    paper.result            = result.result || '';
    // IMPORTANT: don't blank pageImages here.  On first run it's already
    // empty (will be filled by ensurePageImagesLoaded); on a re-run we
    // want the previous PDF renderings to stay visible until the new
    // ones arrive, so the right-hand panel doesn't flash empty.
    paper.pageImagesFetched = false;
    paper.pagesProcessed    = result.pages_processed || 0;
    paper.entries           = parseEntries(result.result);
    paper.parsed            = parseFull(result.result);
    paper.entryIndex        = 0;
    paper.evidenceCount     = result.evidence_count ?? null;
    paper.evidenceTotal     = result.evidence_total ?? null;
    paper.tokenUsage        = result.token_usage    ?? null;
    // Dated model snapshot the provider actually served (e.g.
    // ``gpt-5-2025-09-15``).  Captured per-paper so the eventual
    // JSON export carries proof of which model produced the output.
    paper.resolvedModel     = result.resolved_model ?? null;

    if (state.activePaperId === null) {
      state.activePaperId = paper.id;
      displayPaper(paper);
      goTo(8);
      ensurePageImagesLoaded(paper);
    } else if (state.activePaperId === paper.id) {
      // Active paper just completed (typically after a re-run) — re-render
      // so the result column shows the new entries AND the "Re-run" header
      // button reappears (it's hidden during pending/processing).
      displayPaper(paper);
      ensurePageImagesLoaded(paper);
    } else {
      renderPaperSidebar();
    }
  } catch (err) {
    // Distinguish a deliberate cancel from a real error
    if (err.message === 'Cancelled.') {
      paper.status = 'cancelled';
      paper.error  = null;
    } else {
      paper.status = 'error';
      paper.error  = err.message;
      showToast(state.activePaperId === null ? err.message : `"${paper.filename}": ${err.message}`);
    }
    renderPaperSidebar();
  }
}

/* Poll a job's status until done or error.  Backoff: 1s -> 1.3 per tick -> 8s cap. */
async function pollJob(jobId, onUpdate) {
  let interval = 1000;
  while (true) {
    const res = await fetchScoped(`/api/jobs/${jobId}`);
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`Job lookup failed (${res.status}) ${text}`);
    }
    const data = await res.json();
    if (onUpdate) onUpdate(data.status, data);
    if (data.status === 'done' || data.status === 'error' || data.status === 'cancelled') return data;
    await new Promise(r => setTimeout(r, interval));
    interval = Math.min(Math.round(interval * 1.3), 8000);
  }
}

/* Lazily fetch page images for a finished paper. Called when a paper is shown. */
async function ensurePageImagesLoaded(paper) {
  if (paper.pageImagesFetched || !paper.jobId) return;
  paper.pageImagesFetched = true;
  paper.pageImagesLoading = true;
  // Re-render the page panel so the skeleton appears immediately
  if (state.activePaperId === paper.id && (!paper.entries || paper.entries.length === 0)) {
    showPageImage(paper, null);
  }
  try {
    const res  = await fetchScoped(`/api/jobs/${paper.jobId}/pages`);
    if (!res.ok) return;
    const data = await res.json();
    paper.pageImages   = data.page_images   || [];
    paper.highlights   = data.highlights    || [];   // [{page, snippet, field, source, rects}]
    paper.scannedPages = data.scanned_pages || [];   // 1-indexed pages with no usable text layer
    if (state.activePaperId === paper.id) {
      if (paper.entries && paper.entries.length > 0) renderEntry(paper);
      else showPageImage(paper, 1);
    }
  } catch (err) {
    console.warn(`Could not fetch page images for ${paper.filename}:`, err);
  } finally {
    paper.pageImagesLoading = false;
  }
}

/* ──────────────────────────────────────────────────────────
   Step 8 — Papers sidebar
────────────────────────────────────────────────────────── */

function renderPaperSidebar() {
  const sidebar  = document.getElementById('papersSidebar');
  const subViews = state.activePreset?.sub_views;
  const html = [];
  for (const p of state.papers) {
    const baseName  = escHtml(p.filename.replace(/\.pdf$/i, ''));
    const icon      = { pending: '○', processing: '⟳', done: '✓', error: '✕', cancelled: '⊘' }[p.status] || '·';
    const isActiveP = p.id === state.activePaperId;
    const clickable = p.status === 'done' || p.status === 'error' || p.status === 'cancelled';
    const phaseLabel = (p.status === 'processing' && p.phase)
      ? `<span class="paper-phase">${escHtml(p.phase)}</span>` : '';
    const stopBtn = (p.status === 'pending' || p.status === 'processing')
      ? `<button class="paper-stop" onclick="event.stopPropagation(); cancelPaper('${p.id}')" title="Stop this paper">✕</button>`
      : '';

    // Multi-sample papers expand into one row per sample so the user can
    // see and jump to each sample directly from the overview, instead of
    // clicking through numeric tabs hidden inside the result panel.
    const entries = (p.status === 'done' && Array.isArray(p.entries)) ? p.entries : null;
    const splitSamples = entries && entries.length > 1;

    if (!splitSamples) {
      const cls = ['paper-item', isActiveP ? 'active' : '', `status-${p.status}`]
        .filter(Boolean).join(' ');
      const onclick = clickable ? `onclick="setActivePaperEntry('${p.id}', 0)"` : '';
      html.push(`
        <div class="${cls}" ${onclick}>
          <span class="paper-status-icon">${icon}</span>
          <span class="paper-name-wrap">
            <span class="paper-name">${baseName}</span>
            ${phaseLabel}
          </span>
          ${stopBtn}
        </div>`);
      if (isActiveP && subViews?.length && p.status === 'done') {
        html.push(_renderSidebarSubTabs(p, subViews));
      }
    } else {
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i];
        const rawSample  = entry && typeof entry.sample_id === 'string' ? entry.sample_id.trim() : '';
        const sampleName = rawSample || `Sample ${i + 1}`;
        const isActiveSample = isActiveP && p.entryIndex === i;
        const cls = ['paper-item', 'paper-item-sample', isActiveSample ? 'active' : '', `status-${p.status}`]
          .filter(Boolean).join(' ');
        const onclick = `onclick="setActivePaperEntry('${p.id}', ${i})"`;
        html.push(`
          <div class="${cls}" ${onclick}>
            <span class="paper-status-icon">${icon}</span>
            <span class="paper-name-wrap">
              <span class="paper-name">${baseName} <span class="paper-sample-suffix">&mdash; ${escHtml(sampleName)}</span></span>
            </span>
          </div>`);
        if (isActiveSample && subViews?.length) {
          html.push(_renderSidebarSubTabs(p, subViews));
        }
      }
    }
  }
  sidebar.innerHTML = html.join('');
  updateRetryAllButton();
}

/* Renders the preset-driven sub-tabs (Loadings / Correlations / etc.) that
   live underneath the active sidebar row.  Extracted so the multi-sample
   rendering branch can reuse it without duplicating the markup. */
function _renderSidebarSubTabs(paper, subViews) {
  const activeId = paper.subView || subViews[0].id;
  return `<div class="paper-subtabs">${subViews.map(sv => `
    <button class="paper-subtab ${sv.id === activeId ? 'active' : ''}"
            onclick="event.stopPropagation(); setSubView('${escHtml(sv.id)}')"
            title="${escHtml(sv.label)}">
      ${escHtml(sv.label)}
    </button>`).join('')}</div>`;
}

/* ── Preset sub-views (e.g. MASEM Loadings/Correlations/Descriptives) ──── */

function _activeSubViewFor(paper) {
  const subViews = state.activePreset?.sub_views;
  if (!subViews?.length) return null;
  const id = paper.subView || subViews[0].id;
  return subViews.find(s => s.id === id) || subViews[0];
}

/* Filter an entry object so only the keys for the current sub-view are
   rendered.  Original data is untouched — we just hide what's irrelevant. */
function _filterEntryBySubView(entry, subView) {
  if (!subView || !entry || typeof entry !== 'object' || Array.isArray(entry)) return entry;
  const out = {};
  if (Array.isArray(subView.include_keys)) {
    for (const k of subView.include_keys) {
      if (k in entry) out[k] = entry[k];
    }
    return out;
  }
  if (Array.isArray(subView.exclude_keys)) {
    for (const [k, v] of Object.entries(entry)) {
      if (!subView.exclude_keys.includes(k)) out[k] = v;
    }
    return out;
  }
  return entry;
}

/* Split a JSON path like "samples[0].factor_loadings.F1.1" into its segment
   identifiers ("samples", "factor_loadings", "F1", "1"), so we can match
   include/exclude_keys exactly instead of doing a naive substring check
   (which would let "n" match every field containing the letter n). */
function _fieldSegments(field) {
  if (!field) return [];
  return field.split('.').map(seg => seg.replace(/\[\d+\]$/, ''));
}

/* Match an evidence entry's `field` path against the sub-view's keys.
   Sub-views can declare an optional ``evidence_keys`` array — if present,
   it overrides ``include_keys`` for evidence/highlight matching only.
   This lets the data column include context fields like ``sample_id`` /
   ``n`` while the page-nav and overlay rects stay scoped strictly to the
   sub-view's primary domain (e.g. only ``factor_loadings``). */
function _evidenceMatchesSubView(field, subView) {
  if (!subView || !field) return true;
  const segs = _fieldSegments(field);
  if (Array.isArray(subView.evidence_keys)) {
    return subView.evidence_keys.some(k => segs.includes(k));
  }
  if (Array.isArray(subView.include_keys)) {
    return subView.include_keys.some(k => segs.includes(k));
  }
  if (Array.isArray(subView.exclude_keys)) {
    return !subView.exclude_keys.some(k => segs.includes(k));
  }
  return true;
}

/* Pull the entry index out of a `samples[N]....` field path. */
function _evidenceEntryIndex(field) {
  const m = (field || '').match(/^samples\[(\d+)\]/);
  return m ? parseInt(m[1], 10) : null;
}

/* Pages cited by the evidence entries that match the current sub-view AND
   the current entry index.  Returns a sorted, deduped list. */
function _evidencePagesForSubView(parsed, subView, entryIndex) {
  if (!Array.isArray(parsed?.evidence)) return [];
  const pages = new Set();
  for (const e of parsed.evidence) {
    if (!e || typeof e !== 'object') continue;
    const field = e.field || '';
    const idx   = _evidenceEntryIndex(field);
    if (idx !== null && idx !== entryIndex) continue;          // wrong sample
    if (!_evidenceMatchesSubView(field, subView)) continue;     // wrong sub-view
    const p = toPageNum(e.page);
    if (p !== null) pages.add(p);
  }
  return [...pages].sort((a, b) => a - b);
}

function setSubView(subViewId) {
  const p = getActivePaper();
  if (!p) return;
  p.subView = subViewId;
  renderPaperSidebar();
  if (p.entries && p.entries.length > 0) renderEntry(p);
  else displayPaper(p);
}

/* ── Cancel ────────────────────────────────────────────────────────────────── */
async function cancelPaper(paperId) {
  const p = state.papers.find(x => x.id === paperId);
  if (!p || !p.jobId) return;
  try {
    await fetchScoped(`/api/jobs/${p.jobId}/cancel`, { method: 'POST' });
  } catch (err) {
    console.warn('cancelPaper failed:', err);
  }
}

/* ── Domain workflow presets (MASEMiner etc.) ───────────────────────────────
   A preset re-skins the page (title, tagline, accent color), pre-fills the
   relevant state (mode, provider, model, prompt), and jumps directly to
   the step indicated by `skip_to`.  Activated either by URL (?preset=masem)
   or via the "Pre-built workflows" modal on the landing screen. */

async function applyPreset(presetId) {
  // MASEMiner has its own dedicated entry point with branded chrome
  // (logo + navy/teal palette).  When a user picks any masem* preset
  // from the generic landing, forward to /maseminer so they land on
  // the branded hero instead of just re-skinning the current page.
  // Skipped when we're already on /maseminer to avoid a reload loop.
  if (typeof presetId === 'string'
      && presetId.startsWith('masem')
      && window.location.pathname !== '/maseminer') {
    window.location.href = '/maseminer';
    return true;
  }

  let preset;
  try {
    const res = await fetchScoped(`/api/presets/${encodeURIComponent(presetId)}`);
    if (!res.ok) {
      showToast(`Preset "${presetId}" not found.`);
      return false;
    }
    preset = await res.json();
  } catch (err) {
    showToast('Could not load preset: ' + err.message);
    return false;
  }

  state.activePreset = preset;

  // Re-skin: page title + tagline + accent color
  document.title = preset.title;
  const titleEl   = document.getElementById('appTitle');
  const taglineEl = document.getElementById('appTagline');
  if (titleEl)   titleEl.textContent   = preset.title;
  if (taglineEl) taglineEl.textContent = preset.tagline || taglineEl.textContent;
  if (preset.accent_color) {
    document.documentElement.style.setProperty('--primary',      preset.accent_color);
    document.documentElement.style.setProperty('--primary-dark', preset.accent_color);
  }
  document.body.dataset.preset = preset.id;

  // Banner — visible while preset is active.  In MASEMiner mode the
  // brand identity is carried by the header logo + the in-step mode
  // toggle, so the banner is reduced to just the "Switch to
  // MetaPaperLens" button (no redundant wordmark before it).  Other
  // (non-MASEM) presets still show their plain-text title.
  const banner = document.getElementById('presetBanner');
  if (banner) {
    const labelEl = banner.querySelector('.preset-banner-label');
    const titleEl = document.getElementById('presetBannerTitle');
    const isMasem = typeof preset.id === 'string' && preset.id.startsWith('masem');
    if (isMasem) {
      titleEl.textContent = '';
      titleEl.classList.remove('masem-wordmark');
      if (labelEl) labelEl.style.display = 'none';
    } else {
      titleEl.textContent = preset.title;
      titleEl.classList.remove('masem-wordmark');
      if (labelEl) labelEl.style.display = '';
    }
    banner.style.display = 'flex';
  }

  // Apply preset values to state (provider/model are handled below — must
  // route through onProviderChange so the credential cache stays consistent
  // when the preset switches us to a different provider than the user's
  // last session).
  if (preset.mode)              state.mode             = preset.mode;
  if (preset.task_description)  state.question         = preset.task_description;
  if (preset.context)           state.context          = preset.context;
  if (preset.prompt) {
    state.generatedPrompt = preset.prompt;
    state.inputMode       = 'manual';   // user can still hit Regenerate later
  }

  // Provider/model swap.  Set the dropdown FIRST, then call onProviderChange:
  // it stashes the OUTGOING provider's credentials under that provider's
  // own slot, then loads the INCOMING provider's cached credentials (or
  // empty if the user has never used that provider).  This avoids the bug
  // where the user's Gemini key would get re-saved under the openai slot.
  const provSel = document.getElementById('providerSelect');
  if (provSel && preset.default_provider) {
    provSel.value = preset.default_provider;
    onProviderChange();
  }
  const modelSel = document.getElementById('modelSelect');
  if (modelSel && preset.default_model && preset.default_provider !== 'vllm') {
    if (modelSel.querySelector(`option[value="${CSS.escape(preset.default_model)}"]`)) {
      modelSel.value = preset.default_model;
      state.model    = preset.default_model;
      _stashProviderCredentials();
    }
  }
  const qInput = document.getElementById('questionInput');
  const cInput = document.getElementById('contextInput');
  const mInput = document.getElementById('manualPromptInput');
  if (qInput && preset.task_description) qInput.value = preset.task_description;
  if (cInput && preset.context)          cInput.value = preset.context;
  if (mInput && preset.prompt)           mInput.value = preset.prompt;
  const promptDisplay = document.getElementById('promptDisplay');
  const modelBadge    = document.getElementById('modelBadge');
  if (promptDisplay && preset.prompt) promptDisplay.textContent = preset.prompt;
  if (modelBadge)                     modelBadge.textContent    = preset.default_model || preset.id;
  // Step 3 layout differs by preset:
  //  * MASEMiner (parameterised template) → guided builder form
  //  * Other presets that ship a fully-formed prompt → manual-prompt panel
  const choiceEl = document.getElementById('step3Choice');
  const aiEl     = document.getElementById('aiSection');
  const manEl    = document.getElementById('manualSection');
  const builderEl = document.getElementById('masemBuilder');
  if (typeof isMasemPreset === 'function' && isMasemPreset(preset)) {
    if (choiceEl)  choiceEl.style.display  = 'none';
    if (aiEl)      aiEl.style.display      = 'none';
    if (manEl)     manEl.style.display     = 'none';
    if (builderEl) builderEl.style.display = '';
    if (typeof openMasemBuilder === 'function') openMasemBuilder(preset.id);
  } else if (preset.prompt) {
    if (choiceEl)  choiceEl.style.display  = 'none';
    if (aiEl)      aiEl.style.display      = 'none';
    if (manEl)     manEl.style.display     = '';
    if (builderEl) builderEl.style.display = 'none';
  }

  // Land on the configured step.  For MASEMiner we go to step 2 (setup);
  // submitStep2 then skips section 3 (describe) and jumps to section 4
  // (review prompt) since the prompt is already loaded.
  const SKIP = { upload: 6, prompt: 5, question: 3, setup: 2, task: 1 };
  const targetStep = SKIP[preset.skip_to] || 2;
  goTo(targetStep);
  return true;
}

/* Used by the /maseminer hero "Get started" button — applies the preset,
   hides the landing, shows the configuration accordion, lands on step 2.
   Also flips body.masem-in-flow so the global <header> reappears with
   the MASEMiner mark + title above every flow step (CSS hides the
   header during the hero, shows it during the flow). */
async function startPresetFromLanding(presetId) {
  const landing  = document.getElementById('masemLanding');
  const onepager = document.getElementById('onepager');
  if (landing)  landing.style.display  = 'none';
  if (onepager) onepager.style.display = '';
  document.body.classList.add('masem-in-flow');
  await applyPreset(presetId);
}

function clearPreset() {
  if (!state.activePreset) return;
  // In MASEMiner-only mode (local distribution) there is no generic
  // PaperLens to clear back to — keep the active preset and branding.
  if (window.__MASEMINER_ONLY__) return;
  // On the dedicated /maseminer route, the path itself re-applies the
  // MASEMiner hero on every load — clearing the in-memory preset
  // state isn't enough.  Navigate to the generic root instead so the
  // user actually lands on MetaPaperLens.
  if (window.location.pathname === '/maseminer') {
    window.location.href = '/';
    return;
  }
  // Reset URL so a refresh doesn't re-apply
  if (window.history && window.history.replaceState) {
    const url = new URL(window.location.href);
    url.searchParams.delete('preset');
    window.history.replaceState({}, '', url.toString());
  }
  // Restore the default branding
  document.title = 'MetaPaperLens';
  document.getElementById('appTitle').textContent   = 'MetaPaperLens';
  document.getElementById('appTagline').textContent = 'AI-powered data extraction and labeling for academic papers';
  document.documentElement.style.removeProperty('--primary');
  document.documentElement.style.removeProperty('--primary-dark');
  delete document.body.dataset.preset;
  document.getElementById('presetBanner').style.display = 'none';
  state.activePreset = null;
  // Soft reset — clear the prompt + question/context so the user starts fresh
  startOver();
}

/* Render the inline "Pre-built workflows" section on step 1.  Hidden
   entirely when the server has no presets configured. */
async function renderInlineWorkflows() {
  const wrap = document.getElementById('prebuiltWorkflows');
  const list = document.getElementById('prebuiltWorkflowsList');
  if (!wrap || !list) return;
  try {
    const res  = await fetchScoped('/api/presets');
    if (!res.ok) { wrap.style.display = 'none'; return; }
    const data = await res.json();
    const items = data.presets || [];
    if (!items.length) { wrap.style.display = 'none'; return; }
    wrap.style.display = '';
    list.innerHTML = items.map(p => `
      <button class="workflow-card option-card-load" onclick="applyPreset('${escHtml(p.id)}')">
        <div class="option-icon">🔬</div>
        <div class="option-card-load-text">
          <h3>${escHtml(p.title)}</h3>
          <p>${escHtml(p.tagline || p.description || '')}</p>
        </div>
      </button>
    `).join('');
  } catch (_) {
    wrap.style.display = 'none';
  }
}

/* On page load:
   1. If pathname is /maseminer (or similar), show the dedicated hero landing.
   2. Otherwise, if ?preset=<id> is in the query string, auto-apply that preset
      (legacy / direct-link entry).
   3. Otherwise, populate the inline workflows section so users can pick one. */
const _PRESET_PATHS = {
  // Canonical URL (single "M") plus the legacy double-"m" alias for any
  // bookmarks that pre-date the brand spelling change.
  '/maseminer':  'masem',
  '/maseminer': 'masem',
  // Add new dedicated path → preset mappings here as more workflows are added.
};

async function applyPathOrQueryPreset() {
  const path      = window.location.pathname;
  const params    = new URLSearchParams(window.location.search);

  // ``?batch=<id>`` — the link in batch-complete emails points here.
  // Skip the preset/hero handling entirely and jump straight to the
  // results view for the named batch.
  const batchId = params.get('batch');
  if (batchId) {
    await loadPastBatch(batchId);
    return;
  }

  const presetForPath = _PRESET_PATHS[path];
  if (presetForPath) {
    // Show the hero landing for this preset; user clicks "Get started" to proceed
    document.getElementById('onepager').style.display = 'none';
    const landing = document.getElementById(presetForPath + 'Landing');
    if (landing) landing.style.display = '';
    return;
  }
  const id = params.get('preset');
  if (id) {
    await applyPreset(id);
    return;
  }
  // No preset path / query — populate the inline workflows list on step 1
  renderInlineWorkflows();
}

/* ── Server config (batch limits) ───────────────────────────────────────── */
async function loadServerConfig() {
  try {
    const res = await fetchScoped('/api/config');
    if (!res.ok) return;
    const data = await res.json();
    if (typeof data.max_batch_papers === 'number') config.maxBatchPapers = data.max_batch_papers;
    if (typeof data.max_pdf_bytes    === 'number') config.maxPdfBytes    = data.max_pdf_bytes;
    // MASEMiner-only deployments (local distribution) swap the page
    // chrome so users never see PaperLens branding.  Hosted PaperLens
    // leaves ``maseminer_only`` false and this whole block is a no-op.
    if (data.maseminer_only) {
      window.__MASEMINER_ONLY__ = true;
      const altBtn = document.getElementById('genericPaperlensBtn');
      if (altBtn) altBtn.style.display = 'none';
    }
    // Apply MASEMiner branding on either /maseminer (hosted) OR the
    // locked-down local distribution.  Sets the header h1 + document
    // title to "MASEMiner", flips the body classes that swap the
    // palette (--primary navy/teal-blue) and the header logo
    // (MetaPaperLens M → MASEMiner mark via CSS), and hides the
    // left sidebar (the masemLanding hero / flow header carry the
    // brand identity instead).
    const onMaseminerPath = window.location.pathname === '/maseminer';
    if (data.maseminer_only || onMaseminerPath) {
      document.body.classList.add('mpl-no-sidebar');
      document.body.classList.add('is-maseminer');
      document.title = 'MASEMiner';
      const t = document.getElementById('appTitle');
      if (t) t.textContent = 'MASEMiner';
      // Swap the single header logo from the MetaPaperLens M
      // (/static/logo.svg) to the MASEMiner mark.  Only one logo is
      // present in the markup so there's nothing to hide.
      const logo = document.getElementById('headerLogoImg');
      if (logo) logo.src = '/static/maseminer-mark.svg';
      // Point the wrapping anchor at the MASEMiner landing so clicking
      // the mark resets the user back to the hero (MetaPaperLens mode
      // already links to /).
      const logoLink = document.getElementById('headerLogoLink');
      if (logoLink) logoLink.href = '/maseminer';
      // Step 1 in MASEMiner mode is the MetaPaperLens/MASEMiner mode
      // toggle (rendered statically in index.html) — no per-render
      // population needed.  The ``masem`` preset is applied when the
      // user clicks "Get started" on the welcome hero
      // (``startPresetFromLanding`` → ``applyPreset`` → ``goTo(2)``).
      // We do NOT auto-apply it here: that would fire ``goTo(2)``
      // immediately, and ``goTo`` hides every ``.preset-landing``
      // element — which would kill the welcome hero before the user
      // ever saw it.  The Direct/Indirect task choice surfaces in
      // step 3 once the user has entered the flow.
      state.mode = 'extraction';
      // MASEMiner skips step 1 ("Choose your task") and step 5
      // ("Review prompt") — the user sees 1·2·3 mapped to
      //   1 = Configure AI model   (real id #step2)
      //   2 = Describe your task   (real id #step3)
      //   3 = Upload your papers   (real id #step6)
      // CSS hides #step1 + #step5; we just renumber the visible
      // section headers here so the chips read 1·2·3.
      const renumber = (sectionId, n) => {
        const el = document.querySelector(`#${sectionId} .acc-num`);
        if (el) el.textContent = n;
      };
      renumber('step2', '1');
      renumber('step3', '2');
      renumber('step6', '3');
    }
    // Update the upload-zone hint text now that we know the real limits
    const hint = document.getElementById('uploadLimitHint');
    if (hint) {
      const mb = Math.round(config.maxPdfBytes / (1024 * 1024));
      hint.textContent = `Up to ${config.maxBatchPapers} papers per batch · max ${mb} MB per file`;
    }
  } catch (_) { /* keep defaults */ }
}

/* ── Past extractions (history) ──────────────────────────────────────────── */
async function refreshPastBatches() {
  const wrap = document.getElementById('pastBatches');
  const list = document.getElementById('pastBatchesList');
  if (!wrap || !list) return;
  try {
    const res  = await fetchScoped('/api/batches');
    const data = await res.json();
    const batches = (data.batches || []).filter(b => b.n_total > 0);
    if (!batches.length) { wrap.style.display = 'none'; return; }
    wrap.style.display = '';
    list.innerHTML = batches.map(b => {
      const date  = new Date(b.created_at * 1000);
      const when  = date.toLocaleString();
      const file  = b.sample_filename ? b.sample_filename.replace(/\.pdf$/i, '') : '(unnamed)';
      const model = b.model || '?';
      const counts =
        `<span class="batch-count batch-count-done">${b.n_done || 0} done</span>` +
        ((b.n_error     || 0) ? `<span class="batch-count batch-count-err">${b.n_error} failed</span>`        : '') +
        ((b.n_cancelled || 0) ? `<span class="batch-count batch-count-can">${b.n_cancelled} cancelled</span>` : '') +
        ((b.n_pending   || 0) ? `<span class="batch-count batch-count-pen">${b.n_pending} in flight</span>`   : '');
      return `<button class="past-batch-row" onclick="loadPastBatch('${b.id}')">
        <span class="past-batch-name">${escHtml(file)}${b.n_total > 1 ? ` (+${b.n_total - 1})` : ''}</span>
        <span class="past-batch-meta">${escHtml(model)} &middot; ${when}</span>
        <span class="past-batch-counts">${counts}</span>
      </button>`;
    }).join('');
  } catch (err) {
    console.warn('refreshPastBatches:', err);
    wrap.style.display = 'none';
  }
}

async function loadPastBatch(batchId) {
  // Fetch the batch + jobs, build paper objects from the persisted result strings
  let data;
  try {
    const res = await fetchScoped(`/api/batches/${batchId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    showToast('Could not load batch: ' + err.message);
    return;
  }
  const jobs = data.jobs || [];
  if (!jobs.length) { showToast('This batch has no jobs.'); return; }

  state.batchId = batchId;
  state.papers = jobs.map(j => {
    const result = j.result || '';
    return {
      id:                 j.id,
      jobId:              j.id,
      blob:               null,
      filename:           j.filename || 'paper.pdf',
      status:             j.status === 'done' || j.status === 'error' || j.status === 'cancelled' ? j.status : 'done',
      phase:              null,
      result:             result,
      rawResponse:        result,
      pageImages:         [],
      pageImagesFetched:  false,
      pagesProcessed:     j.pages_processed || 0,
      entries:            parseEntries(result),
      parsed:             parseFull(result),
      entryIndex:         0,
      evidencePages:      [],
      evidencePageIdx:    0,
      evidenceCount:      j.evidence_count ?? null,
      tokenUsage:         j.token_usage    ?? null,
      resolvedModel:      j.resolved_model ?? null,
      error:              j.error || null,
      overrides:          {},
    };
  });
  // Use the prompt + model from the first job for the badge / re-runs
  const j0 = jobs[0];
  state.generatedPrompt = j0.prompt || '';
  state.model           = j0.model  || state.model;
  state.loadedFromFile  = true;
  state.activePaperId   = state.papers.find(p => p.status === 'done')?.id || state.papers[0].id;
  displayPaper(state.papers.find(p => p.id === state.activePaperId));
  goTo(8);
  // Lazily load page images for each finished paper as the user navigates to them.
  // Each ensurePageImagesLoaded call refreshes the re-upload notice — if the
  // server has nothing cached (e.g. restarted since extraction), the notice
  // surfaces and asks the user to re-upload the matching PDF(s).
  const checks = state.papers
    .filter(p => p.status === 'done')
    .map(p => Promise.resolve(ensurePageImagesLoaded(p)).catch(() => {}));
  Promise.all(checks).finally(updateReuploadNotice);
}

async function cancelBatch() {
  if (!state.batchId) return;
  if (!confirm('Stop all in-flight papers in this batch?')) return;
  try {
    const res  = await fetchScoped(`/api/batches/${state.batchId}/cancel`, { method: 'POST' });
    const data = await res.json();
    showToast(`Cancellation requested for ${data.cancelled || 0} paper(s).`, 'success');
  } catch (err) {
    showToast('Could not request cancellation: ' + err.message);
  }
}

function setActivePaper(id) {
  // Backwards-compatible entry point — selects the first sample.
  setActivePaperEntry(id, 0);
}

/* Click handler for sidebar rows.  Multi-sample papers split into one
   row per sample, so the row carries both the paper id and the entry
   index.  This sets both pieces of state and re-renders. */
function setActivePaperEntry(id, idx) {
  const paper = state.papers.find(p => p.id === id);
  if (!paper) return;
  // Only navigate when the paper has reached a viewable state.
  if (paper.status !== 'done' && paper.status !== 'error') return;
  // Clamp entry index to whatever the paper currently has.
  const total = Array.isArray(paper.entries) ? paper.entries.length : 0;
  const safeIdx = total > 0 ? Math.min(Math.max(0, idx | 0), total - 1) : 0;
  paper.entryIndex    = safeIdx;
  state.activePaperId = id;
  renderEvidenceWarning(paper);
  displayPaper(paper);
  renderPaperSidebar();
  // Fetch page images lazily — saves bandwidth/memory when the user has many papers
  if (paper.status === 'done') ensurePageImagesLoaded(paper);
}

/* ──────────────────────────────────────────────────────────
   Step 8 — Display a paper's results
────────────────────────────────────────────────────────── */

function renderEvidenceWarning(paper) {
  const el   = document.getElementById('evidenceWarning');
  const body = document.getElementById('evidenceWarningBody');
  if (!el || !body) return;

  if (paper.status !== 'done') { el.style.display = 'none'; return; }
  if (paper.evidenceWarningDismissed) { el.style.display = 'none'; return; }

  const total      = paper.evidenceTotal ?? 0;   // entries with a snippet
  const usable     = paper.evidenceCount ?? 0;   // entries we could highlight
  // Highlights work — no notice needed
  if (usable > 0) { el.style.display = 'none'; return; }
  // No evidence emitted at all — and the server-side recovery couldn't
  // help.  Either the prompt didn't ask, or the model stayed silent.
  const promptHasSchema = _hasEvidenceSchema(state.generatedPrompt);
  if (total === 0) {
    if (!promptHasSchema) {
      body.innerHTML =
        `Page highlights aren't available — your prompt doesn't request an ` +
        `<code>evidence</code> array. Results are shown on the left; the PDF on ` +
        `the right is browsable but unhighlighted. ` +
        `<button class="btn btn-outline btn-sm" onclick="goToAdaptPrompt()">` +
        `Add evidence to prompt</button>`;
    } else {
      body.innerHTML =
        `Page highlights aren't available for this paper — the model didn't ` +
        `return any evidence references this run. The extracted data is shown ` +
        `on the left; you can browse the PDF unhighlighted on the right. ` +
        `<button class="btn btn-outline btn-sm" onclick="retryPaper('${paper.id}')">` +
        `Re-run this paper</button>`;
    }
  } else {
    body.innerHTML =
      `Page highlights aren't available — ${total} snippet${total !== 1 ? 's were' : ' was'} ` +
      `returned without page numbers, and we couldn't locate them in the PDF text. ` +
      `The extracted data is shown on the left; the snippets are in the raw response. ` +
      `<button class="btn btn-outline btn-sm" onclick="setViewMode('raw')">View raw response</button> ` +
      `<button class="btn btn-outline btn-sm" onclick="retryPaper('${paper.id}')">Re-run</button>`;
  }
  el.style.display = 'flex';
}

function dismissEvidenceWarning() {
  const paper = getActivePaper();
  if (paper) paper.evidenceWarningDismissed = true;
  const el = document.getElementById('evidenceWarning');
  if (el) el.style.display = 'none';
}

/* Jump back to the prompt-review section so the user can hit Adapt. */
function goToAdaptPrompt() {
  goTo(5);
  // Scroll into view + flash the warning so it's obvious where to click
  setTimeout(() => {
    const w = document.getElementById('promptEvidenceWarning');
    if (w) {
      w.style.display = 'flex';
      w.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, 200);
}

function renderTokenFooter(paper) {
  const el = document.getElementById('tokenSummary');
  if (!el) return;
  const u = paper.tokenUsage;
  const hasTokens = u && (u.prompt || u.completion || u.total);
  const resolved  = paper.resolvedModel || null;
  if (!hasTokens && !resolved) {
    el.style.display = 'none';
    return;
  }
  const parts = [];
  if (hasTokens) {
    const fmt = n => (n || 0).toLocaleString();
    parts.push(
      `<span class="token-label">Tokens</span>` +
      `<span class="token-stat">${fmt(u.prompt)} in</span>` +
      `<span class="token-sep">·</span>` +
      `<span class="token-stat">${fmt(u.completion)} out</span>` +
      `<span class="token-sep">·</span>` +
      `<span class="token-stat token-total">${fmt(u.total)} total</span>`
    );
  }
  if (resolved) {
    // ``model`` carries the alias the user picked; ``resolvedModel``
    // the dated snapshot the API served.  Show the snapshot so the
    // user can tell which exact build produced the output.  Tooltip
    // surfaces the alias for context.
    parts.push(
      `<span class="token-sep">·</span>` +
      `<span class="token-stat" title="Alias requested: ${escHtml(state.model || '')}">${escHtml(resolved)}</span>`
    );
  }
  el.innerHTML = parts.join('');
  el.style.display = 'inline-flex';
}

/* ── Download dropdown ─────────────────────────────────────────────────────── */
function toggleDownloadMenu(event) {
  if (event) event.stopPropagation();
  document.getElementById('downloadMenu').classList.toggle('open');
}
function closeDownloadMenu() {
  const m = document.getElementById('downloadMenu');
  if (m) m.classList.remove('open');
}
document.addEventListener('click', e => {
  // Click outside the dropdown closes it
  if (!e.target.closest('#downloadDropdown')) closeDownloadMenu();
});

/* ── Batch retry ───────────────────────────────────────────────────────────── */
function retryAllFailed() {
  const failed = state.papers.filter(p => p.status === 'error');
  if (!failed.length) return;
  failed.forEach(p => {
    p.status            = 'pending';
    p.error             = null;
    p.jobId             = null;
    p.pageImagesFetched = false;
  });
  renderPaperSidebar();
  updateRetryAllButton();
  // Run all retries in parallel via the same path used for first-time submission
  Promise.all(failed.map(p => processPaper(p))).catch(err =>
    console.error('[retryAllFailed] uncaught:', err)
  );
}

/* Show / hide the "Retry failed" button based on error counts. */
function updateRetryAllButton() {
  const btn   = document.getElementById('retryAllBtn');
  const count = document.getElementById('retryAllCount');
  if (!btn) return;
  const n = state.papers.filter(p => p.status === 'error').length;
  if (n > 0) {
    btn.style.display    = '';
    count.textContent    = `(${n})`;
  } else {
    btn.style.display = 'none';
  }
}

function displayPaper(paper) {
  document.getElementById('resultsSubtitle').textContent =
    paper.status === 'error'
      ? paper.filename
      : `${paper.filename} · ${paper.pagesProcessed} page${paper.pagesProcessed !== 1 ? 's' : ''}`;

  renderEvidenceWarning(paper);
  renderTokenFooter(paper);

  // Show the "Re-run" header button whenever this paper isn't currently
  // being processed.  Hidden during 'pending' / 'processing' to avoid
  // letting the user fire a second job on top of the first.
  // When the previous run used text mode, surface BOTH text + vision
  // re-run buttons so the user can compare against a vision re-run
  // (typical fix path when the text-layer parse missed values that a
  // VLM would catch from the table image).
  const rerunGroup  = document.getElementById('rerunGroup');
  const rerunBtn    = document.getElementById('rerunActiveBtn');
  const rerunVisBtn = document.getElementById('rerunVisionBtn');
  const visible = (paper.status === 'done' || paper.status === 'error');
  if (rerunGroup) rerunGroup.style.display = visible ? '' : 'none';
  if (visible && paper.lastModeUsed === 'text') {
    if (rerunBtn)    rerunBtn.innerHTML       = '&#8635; Re-run · text';
    if (rerunVisBtn) rerunVisBtn.style.display = '';
  } else {
    if (rerunBtn)    rerunBtn.innerHTML       = '&#8635; Re-run';
    if (rerunVisBtn) rerunVisBtn.style.display = 'none';
  }

  const nav     = document.getElementById('entryNav');
  const display = document.getElementById('resultDisplay');

  if (paper.status === 'pending' || paper.status === 'processing') {
    // Re-run in progress — show a small "running" notice in the result
    // column and leave the right-hand PDF panel untouched (the previous
    // page image stays visible while the new job runs).
    nav.style.display       = 'none';
    display.dataset.paperId  = paper.id;
    display.dataset.entryIdx = 0;
    const phase = paper.phase ? ` &middot; ${escHtml(paper.phase)}` : '';
    display.innerHTML = `
      <div class="rerun-progress" role="status" aria-live="polite">
        <span class="rerun-spinner" aria-hidden="true">&#10227;</span>
        <span>Re-running this paper${phase}&hellip;</span>
      </div>`;
    renderPaperSidebar();
    return;
  }

  if (paper.status === 'error') {
    nav.style.display  = 'none';
    display.dataset.paperId  = paper.id;
    display.dataset.entryIdx = 0;
    display.innerHTML = `
      <div class="paper-error-panel">
        <div class="paper-error-icon">✕</div>
        <h3 class="paper-error-title">Extraction failed</h3>
        <p class="paper-error-msg">${escHtml(paper.error || 'Unknown error')}</p>
        ${paper.rawResponse ? `
        <details class="error-response-details">
          <summary>Model response</summary>
          <pre class="error-response-pre">${escHtml(paper.rawResponse)}</pre>
        </details>` : ''}
        <button class="btn btn-primary" onclick="retryPaper('${paper.id}')">Retry extraction</button>
      </div>`;
    // Hide page image for error state
    document.getElementById('pageDisplayImg').style.display  = 'none';
    document.getElementById('pageDisplayNone').style.display = 'block';
    document.getElementById('pageDisplayNone').innerHTML     = '';
    document.getElementById('pageDisplayLabel').textContent  = 'Page —';
    renderPaperSidebar();
    return;
  }

  if (paper.entries && paper.entries.length > 0) {
    nav.style.display = 'flex';
    renderEntry(paper);
  } else {
    // Extraction couldn't be parsed into structured entries.  Surface the
    // raw model response, warn the user, and let them either ignore the
    // paper or fill the values in manually using a schema-aware scaffold.
    nav.style.display = 'none';
    const hasContent = (paper.result || '').trim().length > 0;
    if (hasContent) {
      // Use the Direct/Indirect task framing the user actually picked,
      // rather than the internal preset title (which surfaces the
      // implementation id like "MASEMiner — TAS-20" that's meaningless
      // to a user who picked "Indirect information" in step 3).
      const pid = state.activePreset?.id;
      const presetLabel =
        pid === 'masem'        ? 'MASEMiner — direct information' :
        pid === 'masem-ncs18'  ? 'MASEMiner — indirect information' :
        (state.activePreset?.title || 'paper');
      display.innerHTML = `
        <div class="extraction-failed-panel">
          <div class="extraction-failed-warn">
            <strong>We couldn't parse this response into structured entries.</strong>
            The model returned text but it didn't match the expected ${escHtml(presetLabel)} schema.
            You can fill in the values manually below — the raw response is shown for reference,
            and the right-hand panel lets you flip through every page of the PDF.
          </div>
          <div class="extraction-failed-actions">
            <button class="btn btn-primary" onclick="enterManualMode('${paper.id}')">
              Fill in manually
            </button>
            <button class="btn btn-outline" onclick="retryPaper('${paper.id}')">
              Re-run extraction
            </button>
          </div>
          <details class="extraction-raw-details">
            <summary>View raw model response</summary>
            <pre class="raw-text">${escHtml(stripFences(paper.result))}</pre>
          </details>
        </div>`;
    } else {
      display.innerHTML = '<p class="rv-null">No data returned.</p>';
    }
    // Browse-all-pages mode kicks in automatically because evidencePages is empty.
    paper.evidencePages = [];
    paper.browseAllPagesIdx = 0;
    updatePageNav(paper);
    showPageImage(paper, 1);
  }

  renderPaperSidebar();
}

/* Mark a paper as user-filled.  Builds an empty-but-typed scaffold (so the
   existing renderEntry + override pipeline can edit it) and switches the
   results panel into edit mode.  The original LLM response is preserved on
   ``paper.result`` and exported alongside the human-entered data. */
function enterManualMode(paperId) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper) return;
  paper.manualMode    = true;
  paper.manualEntries = _buildManualScaffold(state.activePreset);
  paper.entries       = paper.manualEntries;
  paper.entryIndex    = 0;
  paper.overrides     = {};   // start clean — every cell is the user's input
  paper.evidencePages = [];   // forces browse-all-pages on the right
  if (state.activePaperId === paper.id) renderEntry(paper);
  renderPaperSidebar();
}

/* Empty-but-typed entry scaffold so the user can fill in values manually.
   For the MASEM preset the shape matches the extraction schema exactly so
   downstream tooling sees a familiar structure even when the LLM failed. */
function _buildManualScaffold(activePreset) {
  if (!activePreset) return [{}];
  // Pick the scaffold shape from the preset's id.  The Direct
  // (``masem``) variant returns one empty effect-size table plus the
  // sample metadata; the Indirect (``masem-ncs18``) variant returns a
  // factor-loading/correlation dotted-key matrix.  Hard-coded shapes
  // were left over from when ``masem`` was the only preset and meant
  // factor analysis — easy to misread now that ``masem`` means Direct.
  if (activePreset.id === 'masem') {
    return [{
      sample_id: '',
      effect_sizes:  { _table: [] },
      reliabilities: { _table: [] },
      pubyear: null, country: '', continent: '', lang: '', pubtype: null,
      n: null, female: null, age: null, clinical: null, notes: '',
    }];
  }
  if (activePreset.id === 'masem-ncs18' || /^masem-/.test(activePreset.id || '')) {
    const factor_loadings = {};
    for (let f = 1; f <= 5; f++) {
      for (let i = 1; i <= 20; i++) factor_loadings[`F${f}.${i}`] = null;
    }
    const factor_correlations = {};
    for (let i = 1; i <= 4; i++) {
      for (let j = i + 1; j <= 5; j++) factor_correlations[`R${i}.${j}`] = null;
    }
    return [{
      sample_id: '', factor_loadings, factor_correlations,
      pubyear: null, country: '', continent: '', lang: '', pubtype: null,
      n: null, female: null, age: null, clinical: null, res: null,
      nfac: null, cfa: null, met: null, rot: null, notes: '',
    }];
  }
  // Generic: a single empty entry — user can edit existing keys freely.
  return [{}];
}

function retryPaper(id) {
  const paper = state.papers.find(p => p.id === id);
  if (!paper) return;
  // Clear the result-side state so the new run replaces the old data.
  // IMPORTANT: keep ``pageImages`` and ``scannedPages`` — the same PDF
  // re-extracts to the same page renderings, so the right-hand panel
  // should stay visible during the re-run instead of going blank.  We
  // also keep ``browseAllPagesIdx`` so the user's current page survives
  // the round-trip; highlights will be replaced when the new job's
  // /pages endpoint returns.
  paper.status              = 'pending';
  paper.phase               = null;
  paper.error               = null;
  paper.jobId               = null;
  paper.result              = '';
  paper.entries             = null;
  paper.entryIndex          = 0;
  paper.parsed              = null;
  paper.highlights          = [];          // will be repopulated by the new run
  paper.evidencePages       = [];
  paper.evidencePageIdx     = 0;
  paper.focusedField        = null;        // any cell-click focus from the previous run
  paper.evidenceCount       = null;
  paper.evidenceTotal       = null;
  paper.tokenUsage          = null;
  paper.pagesProcessed      = 0;
  paper.overrides           = {};
  paper.manualMode          = false;
  paper.manualEntries       = null;
  paper.evidenceWarningDismissed = false;
  paper.autoVisionFallback  = false;
  // Mark page images stale so ensurePageImagesLoaded re-fetches the new
  // job's overlay rects when the run completes.  ``pageImages`` itself
  // stays in place visually until the new ones overwrite.
  paper.pageImagesFetched   = false;
  renderPaperSidebar();
  if (state.activePaperId === paper.id) displayPaper(paper);
  processPaper(paper);
}

/* "Re-run" button on the results header — re-extracts the currently-active
   paper, replacing its previous result.  Confirms before discarding edits
   the user has already made (overrides), since those will be lost.

   When called by the primary "Re-run" / "Re-run · text" button, the
   per-paper ``forceMode`` is cleared, so processPaper picks the mode
   from the global default + probe (same as the original first run).
   The split "Re-run · VLM" button calls ``rerunActivePaperWithMode``
   instead. */
function rerunActivePaper() {
  rerunActivePaperWithMode(null);
}

/* Re-run the active paper, optionally forcing a specific extraction
   mode for this run.  ``mode`` is one of:
     - ``'text'``    → force text-layer extraction
     - ``'vision'``  → force the VLM (image) pipeline; if multiple
                        vision models exist for the current provider,
                        the user picks one via the VLM picker modal
                        (state.model is left untouched — only this
                        paper's forceModel is set)
     - ``null``      → use the global default (with probe fallback) */
function rerunActivePaperWithMode(mode) {
  const p = getActivePaper();
  if (!p) return;
  const hasEdits = p.overrides && Object.keys(p.overrides).length > 0;
  if (hasEdits && !confirm(
    'Re-run this paper? Your manual edits to the previous result will be discarded.'
  )) return;

  // Reset any previous per-paper overrides; they're re-set below if
  // the requested mode needs them.
  p.forceMode  = null;
  p.forceModel = null;

  if (mode === 'text') {
    p.forceMode = 'text';
    retryPaper(p.id);
    return;
  }
  if (mode === 'vision') {
    p.forceMode = 'vision';
    _kickOffVisionRerun(p);
    return;
  }
  // mode === null / other → use global default
  retryPaper(p.id);
}

/* Decide which vision model to use for this re-run, then kick off
   retryPaper.  Behaviour depends on how many vision-capable models
   the current provider exposes:
     - 0 models (e.g. DeepSeek)  → toast a clear "switch provider" hint
     - 1 model                    → set forceModel and rerun
     - 2+ models                  → open the picker modal, user chooses */
function _kickOffVisionRerun(paper) {
  const provider     = state.provider;
  const visionModels = _visionModelsForProvider(provider);
  const currentIsVision = isVisionModel(state.model);

  if (visionModels.length === 0 && !currentIsVision) {
    showToast(
      `${state.model} is not compatible with VLM (image) extraction, and ${provider} has no ` +
      `vision-capable model. Switch to OpenAI, Anthropic, Gemini, or Mistral (Pixtral) in ` +
      `step 1, then re-run.`,
    );
    paper.forceMode = null;
    return;
  }

  // Build the option list.  Always include the current model if it's
  // vision-capable, so the user can pick "stay on what I'm using" too.
  const seen = new Set();
  const options = [];
  if (currentIsVision) {
    options.push({ value: state.model, label: state.model });
    seen.add(state.model);
  }
  for (const m of visionModels) {
    if (!seen.has(m.value)) { options.push(m); seen.add(m.value); }
  }

  if (options.length === 1) {
    paper.forceModel = options[0].value;
    retryPaper(paper.id);
    return;
  }
  // Multiple options → open the picker.
  _openVlmPicker({
    currentModel: state.model,
    options,
    onConfirm: (picked) => {
      paper.forceModel = picked;
      retryPaper(paper.id);
    },
  });
}

/* Return the subset of PROVIDER_MODELS[provider] that are vision-
   capable, per ``isVisionModel``.  Used by the VLM picker to populate
   its option list. */
function _visionModelsForProvider(provider) {
  const models = PROVIDER_MODELS[provider] || [];
  return models.filter(m => isVisionModel(m.value));
}

/* Open the VLM picker modal.  ``opts``:
     - currentModel: string  — model the paper was last extracted with
     - options:      [{value, label}, ...]
     - onConfirm:    (pickedValue) => void
   The modal handles its own selection state via .is-selected on the
   chosen .vlm-picker-option button; ``Re-run`` reads the current
   selection at click time. */
function _openVlmPicker(opts) {
  const overlay  = document.getElementById('vlmPickerOverlay');
  const intro    = document.getElementById('vlmPickerIntro');
  const listEl   = document.getElementById('vlmPickerOptions');
  const confirm_ = document.getElementById('vlmPickerConfirm');
  if (!overlay || !intro || !listEl || !confirm_) return;

  // Default selection: the current model if it's in the list,
  // otherwise the first option.
  let selected = opts.options.find(o => o.value === opts.currentModel)
              ? opts.currentModel
              : opts.options[0].value;

  intro.textContent =
    `Pick the vision-capable model to use for this re-run. The choice ` +
    `applies to this paper only — your default model in step 2 stays the same.`;

  listEl.innerHTML = opts.options.map(o => {
    const isCurrent  = o.value === opts.currentModel;
    const isSelected = o.value === selected;
    return `
      <button type="button" class="vlm-picker-option ${isSelected ? 'is-selected' : ''}"
              data-value="${escHtml(o.value)}"
              onclick="_selectVlmPickerOption('${escHtml(o.value)}')">
        <span class="vlm-picker-option-radio" aria-hidden="true"></span>
        <span class="vlm-picker-option-label">${escHtml(o.label)}</span>
        ${isCurrent ? '<span class="vlm-picker-option-current">current</span>' : ''}
      </button>`;
  }).join('');

  // Stash callback + selected on the overlay so Re-run can read them.
  overlay._mplPickerState = { onConfirm: opts.onConfirm, selected };
  confirm_.onclick = () => {
    const state_ = overlay._mplPickerState;
    if (state_ && state_.onConfirm) state_.onConfirm(state_.selected);
    _closeVlmPicker();
  };

  overlay.style.display = 'flex';
}

function _selectVlmPickerOption(value) {
  const overlay = document.getElementById('vlmPickerOverlay');
  if (!overlay) return;
  if (overlay._mplPickerState) overlay._mplPickerState.selected = value;
  overlay.querySelectorAll('.vlm-picker-option').forEach(btn => {
    btn.classList.toggle('is-selected', btn.getAttribute('data-value') === value);
  });
}

function _closeVlmPicker() {
  const overlay = document.getElementById('vlmPickerOverlay');
  if (!overlay) return;
  overlay.style.display = 'none';
  delete overlay._mplPickerState;
}

/* Mapping from the human-readable label shown next to each badge to the
   model-emitted key in ``extraction_confidence``.  Keeping these
   parallel arrays (rather than a hash) preserves the display order:
   loadings → correlations → metadata. */
const _CONFIDENCE_CATEGORIES = [
  { key: 'factor_loadings',     label: 'Loadings' },
  { key: 'factor_correlations', label: 'Correlations' },
  { key: 'metadata',            label: 'Metadata' },
];

/* Render the confidence-badge row above the parsed entry.  Reads
   ``entry.extraction_confidence`` (object with one rating per category)
   and emits one coloured pill per known category.  Hides the row
   entirely when the entry has no confidence block — old runs and
   non-MASEM presets stay clean. */
function _renderConfidenceBadges(entry) {
  const row = document.getElementById('confidenceRow');
  if (!row) return;
  const conf = entry && entry.extraction_confidence;
  if (!conf || typeof conf !== 'object') {
    row.style.display = 'none';
    row.innerHTML     = '';
    return;
  }
  const parts = [`<span class="confidence-row-label">Confidence</span>`];
  let hadAny = false;
  for (const cat of _CONFIDENCE_CATEGORIES) {
    const raw   = conf[cat.key];
    const level = _normaliseConfidence(raw);
    if (raw == null && level === 'unknown') continue;  // skip categories the model omitted
    hadAny = true;
    const displayValue = (typeof raw === 'string' && raw.trim()) ? raw.trim() : '—';
    parts.push(`
      <span class="confidence-badge confidence-${level}" title="${escHtml(cat.label)}: ${escHtml(displayValue)}">
        <span class="confidence-badge-label">${escHtml(cat.label)}</span>
        <span class="confidence-badge-value">${escHtml(displayValue)}</span>
      </span>
    `);
  }
  if (!hadAny) {
    row.style.display = 'none';
    row.innerHTML     = '';
    return;
  }
  row.style.display = '';
  row.innerHTML     = parts.join('');
}

/* Scan a MASEMiner sample for item-rows / factor-rows that have NO
   reported value — usually a sign the LLM dropped that item.  Reports
   the offending row labels grouped by table (factor_loadings,
   factor_correlations) and renders them as a small warning banner
   above the parsed result.

   Detection: keys matching ``F<i>.<n>`` (loadings) and ``R<i>.<j>``
   (correlations) are pulled out of the entry, grouped by the trailing
   ``.n`` (item index) for loadings, and by the leading factor index
   for correlations.  An "empty row" is one where every cell in the
   group is null/undefined.  Renders nothing when no offenders found
   or when the entry isn't a MASEMiner shape. */
/* Populate #masemStep1Cards with the MASEM starter cards (Blank /
   TAS-20).  Called once after loadServerConfig confirms we're in
   MASEMiner mode.  ``_MASEM_STARTERS`` is defined in
   masem-builder.js and is in global scope (both scripts are
   loaded as classic scripts, not modules). */
function _renderMasemStep1Cards() {
  const row = document.getElementById('masemStep1Cards');
  if (!row) return;
  if (typeof _MASEM_STARTERS === 'undefined') return;
  row.innerHTML = _MASEM_STARTERS.map(s => `
    <button class="option-card" type="button"
            onclick="_pickMasemStep1Starter('${escHtml(s.id)}')">
      <div class="option-icon" aria-hidden="true">
        <svg viewBox="0 0 64 64" width="38" height="38"><use href="#masemMark"/></svg>
      </div>
      <h3>${escHtml(s.label)}</h3>
      <p>${escHtml(s.tagline)}</p>
    </button>
  `).join('');
}

/* Click handler for the step-1 MASEM starter cards.  Applies the
   chosen preset (Blank / General or TAS-20) — applyPreset then
   handles state.mode + the goTo() to step 2 for setup. */
function _pickMasemStep1Starter(starterId) {
  state.mode = 'extraction';
  applyPreset(starterId);
}

function _renderMasemRowWarnings(entry) {
  const row = document.getElementById('masemWarningRow');
  if (!row) return;
  if (!entry || typeof entry !== 'object') {
    row.style.display = 'none';
    row.innerHTML     = '';
    return;
  }

  // ── factor_loadings: keys like "F1.5", group by ".n" (item index) ──
  const loadings = entry.factor_loadings;
  const emptyItems = [];
  if (loadings && typeof loadings === 'object' && !Array.isArray(loadings)) {
    const byItem = new Map();   // itemIdx → array of values
    for (const [k, v] of Object.entries(loadings)) {
      const m = /^F(\d+)\.(\d+)$/.exec(k);
      if (!m) continue;
      const idx = m[2];
      if (!byItem.has(idx)) byItem.set(idx, []);
      byItem.get(idx).push(v);
    }
    // Sort numerically so "Item 9" precedes "Item 10".
    const sortedIdx = Array.from(byItem.keys()).sort((a, b) => Number(a) - Number(b));
    for (const idx of sortedIdx) {
      const vals = byItem.get(idx);
      if (vals.length && vals.every(v => v === null || v === undefined)) {
        emptyItems.push(idx);
      }
    }
  }

  // ── factor_correlations: keys like "R1.2", group by leading factor ──
  const corrs = entry.factor_correlations;
  const emptyFactors = [];
  if (corrs && typeof corrs === 'object' && !Array.isArray(corrs)) {
    const byFactor = new Map();
    for (const [k, v] of Object.entries(corrs)) {
      const m = /^R(\d+)\.(\d+)$/.exec(k);
      if (!m) continue;
      // A factor participates in many R keys; group every key it
      // appears in (either side) so the all-null check covers all
      // correlations involving that factor.
      for (const fac of [m[1], m[2]]) {
        if (!byFactor.has(fac)) byFactor.set(fac, []);
        byFactor.get(fac).push(v);
      }
    }
    const sortedFac = Array.from(byFactor.keys()).sort((a, b) => Number(a) - Number(b));
    for (const fac of sortedFac) {
      const vals = byFactor.get(fac);
      if (vals.length && vals.every(v => v === null || v === undefined)) {
        emptyFactors.push(fac);
      }
    }
  }

  if (emptyItems.length === 0 && emptyFactors.length === 0) {
    row.style.display = 'none';
    row.innerHTML     = '';
    return;
  }

  const parts = [
    '<span class="masem-warning-icon" aria-hidden="true">&#9888;</span>',
    '<div class="masem-warning-body">',
  ];
  if (emptyItems.length) {
    parts.push(
      '<div><strong>Factor loadings:</strong> ' +
      `item${emptyItems.length > 1 ? 's' : ''} ` +
      emptyItems.map(escHtml).join(', ') +
      ` ${emptyItems.length > 1 ? 'have' : 'has'} no reported loading on any factor. ` +
      'Either the item was genuinely not loaded in this paper, or the model missed it — verify against the source table.</div>',
    );
  }
  if (emptyFactors.length) {
    parts.push(
      '<div><strong>Factor correlations:</strong> ' +
      `factor${emptyFactors.length > 1 ? 's' : ''} F` +
      emptyFactors.map(escHtml).join(', F') +
      ` ${emptyFactors.length > 1 ? 'have' : 'has'} no reported correlations. ` +
      'Check the inter-factor correlation matrix on the source page.</div>',
    );
  }
  parts.push('</div>');
  row.innerHTML     = parts.join('');
  row.style.display = '';
}

/* Map a free-form confidence value (string, possibly with case/spacing
   variation) to one of the four CSS classes the badge palette knows
   about.  Falls back to ``unknown`` for anything unrecognised so the
   grey pill flags it without crashing the render. */
function _normaliseConfidence(v) {
  if (typeof v !== 'string') return 'unknown';
  const s = v.trim().toLowerCase();
  if (s === 'high'   || s === 'h') return 'high';
  if (s === 'medium' || s === 'med' || s === 'm') return 'medium';
  if (s === 'low'    || s === 'l') return 'low';
  return 'unknown';
}

function renderEntry(paper) {
  const entry = paper.entries[paper.entryIndex];
  const total = paper.entries.length;

  // Sample label — prefer the model-extracted ``sample_id`` when present,
  // otherwise fall back to "Sample N".  Same logic the sidebar uses, so
  // both surfaces show the same name for each row.
  const rawSample  = entry && typeof entry.sample_id === 'string' ? entry.sample_id.trim() : '';
  const sampleName = rawSample || `Sample ${paper.entryIndex + 1}`;
  document.getElementById('entryCounter').textContent = total > 1
    ? `${sampleName} · ${paper.entryIndex + 1} of ${total}`
    : sampleName;
  // Inter-sample nav buttons are redundant when there's only one sample
  // (the sidebar handles selection); hide them in that case.
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  prevBtn.style.display = total > 1 ? '' : 'none';
  nextBtn.style.display = total > 1 ? '' : 'none';
  prevBtn.disabled = paper.entryIndex === 0;
  nextBtn.disabled = paper.entryIndex === total - 1;

  // The hint that used to tell users to "use numbered tabs to flip
  // through entries" is now redundant — each sample has its own row in
  // the left-hand overview.  Hide it.
  const hint  = document.getElementById('multiEntryHint');
  const tabs  = document.getElementById('entryTabs');
  if (hint) hint.style.display = 'none';

  if (tabs) {
    // The strip used to also host numeric "1 2 3" tabs for sample
    // navigation — those moved to the sidebar.  We keep only the
    // "+ Add sample" / "− Remove sample" controls so the user can still
    // grow / shrink the sample list, with text labels so they're
    // discoverable on their own.
    tabs.style.display = 'flex';
    const addBtn = `<button class="entry-tab entry-tab-add" title="Append a new sample"
                            onclick="addEntryToPaper('${paper.id}')">+ Add sample</button>`;
    const delBtn = total > 1
      ? `<button class="entry-tab entry-tab-del" title="Delete this sample"
                onclick="removeActiveEntry('${paper.id}')">&minus; Remove sample</button>`
      : '';
    tabs.innerHTML = addBtn + delBtn;
  }

  // Reflect the current view-mode in the toggle buttons
  const parsedBtn = document.getElementById('viewParsedBtn');
  const rawBtn    = document.getElementById('viewRawBtn');
  if (parsedBtn && rawBtn) {
    const isRaw = paper.viewMode === 'raw';
    parsedBtn.classList.toggle('active', !isRaw);
    rawBtn.classList.toggle('active',     isRaw);
  }

  const display = document.getElementById('resultDisplay');
  display.dataset.paperId  = paper.id;
  display.dataset.entryIdx = paper.entryIndex;

  // Confidence badges above the rendered entry — driven by the
  // `extraction_confidence` block the model emits per sample.  Hidden
  // when absent so old/unconfigured runs are unaffected.
  _renderConfidenceBadges(entry);
  _renderMasemRowWarnings(entry);

  // Sub-view filter (preset-driven, e.g. MASEM Loadings/Correlations/Descriptives)
  const subView       = _activeSubViewFor(paper);
  let   filteredEntry = subView ? _filterEntryBySubView(entry, subView) : entry;
  // The confidence block is surfaced as coloured badges above — strip
  // it from the parsed-data render so it doesn't appear twice.
  if (filteredEntry && typeof filteredEntry === 'object' && !Array.isArray(filteredEntry)
      && 'extraction_confidence' in filteredEntry) {
    const { extraction_confidence: _drop, ...rest } = filteredEntry;
    filteredEntry = rest;
  }

  if (paper.viewMode === 'raw') {
    // Raw mode — show the verbatim model output, untouched
    display.innerHTML =
      `<pre class="raw-response">${escHtml(stripFences(paper.result || paper.rawResponse || ''))}</pre>`;
  } else {
    display.innerHTML = renderValueHtml(filteredEntry);
    applyOverrides(paper);
  }

  // Evidence pages: when a sub-view is active, restrict to evidence whose
  // `field` path matches the sub-view's keys; otherwise use the original
  // walk-the-entry-and-fall-back-to-global logic.
  let evidencePages;
  if (subView) {
    evidencePages = _evidencePagesForSubView(paper.parsed, subView, paper.entryIndex);
  } else {
    const entryPages  = [...findAllEntryPages(entry)].sort((a, b) => a - b);
    evidencePages     = entryPages.length ? entryPages
      : [...findAllEntryPages(paper.parsed)].sort((a, b) => a - b);
  }
  paper.evidencePages   = evidencePages;
  paper.evidencePageIdx = 0;
  // Open on the first cited page so the relevant content is what the
  // user sees first, but the nav can flip through every page in the PDF.
  const initialPage = paper.evidencePages[0] ?? (paper.pageImages.length ? 1 : null);
  paper.browseAllPagesIdx = initialPage ? (initialPage - 1) : 0;
  updatePageNav(paper);
  showPageImage(paper, initialPage);
}

/* Append a new empty sample to the paper.  For preset workflows the
   scaffold is shape-aware (e.g. MASEM gets the full schema); for generic
   extractions we just push a copy of the first entry's keys with cleared
   values so the user has familiar fields to fill in. */
function addEntryToPaper(paperId) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper) return;
  paper.entries = paper.entries || [];
  let scaffold;
  if (state.activePreset) {
    scaffold = _buildManualScaffold(state.activePreset)[0] || {};
  } else if (paper.entries.length > 0) {
    scaffold = _emptyLikeEntry(paper.entries[0]);
  } else {
    scaffold = {};
  }
  paper.entries.push(scaffold);
  paper.entryIndex = paper.entries.length - 1;
  if (state.activePaperId === paper.id) renderEntry(paper);
  renderPaperSidebar();
}

/* Build an empty-keyed copy of an existing entry so a freshly-added sample
   has the same field shape (cleared to nulls / empty strings).  Recurses
   into nested dicts; arrays become empty arrays. */
function _emptyLikeEntry(entry) {
  if (entry === null || entry === undefined) return null;
  if (Array.isArray(entry))                  return [];
  if (typeof entry !== 'object')             return null;
  const out = {};
  for (const [k, v] of Object.entries(entry)) {
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      // Nested dict — preserve key shape but clear values
      const inner = {};
      for (const k2 of Object.keys(v)) inner[k2] = null;
      out[k] = inner;
    } else if (Array.isArray(v)) {
      out[k] = [];
    } else if (typeof v === 'string') {
      out[k] = '';
    } else {
      out[k] = null;
    }
  }
  return out;
}

/* Remove the currently-displayed sample and shift focus to a neighbour. */
function removeActiveEntry(paperId) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper || !paper.entries || paper.entries.length <= 1) return;
  const idx = paper.entryIndex;
  if (!confirm(`Delete sample ${idx + 1}? Its values will be removed.`)) return;
  paper.entries.splice(idx, 1);
  // Drop overrides for the deleted entry; renumber the rest.
  const ovIn  = paper.overrides || {};
  const ovOut = {};
  for (const [k, v] of Object.entries(ovIn)) {
    const i = parseInt(k, 10);
    if (i === idx)         continue;
    if (i > idx)            ovOut[i - 1] = v;
    else                    ovOut[i]     = v;
  }
  paper.overrides   = ovOut;
  paper.entryIndex  = Math.min(idx, paper.entries.length - 1);
  if (state.activePaperId === paper.id) renderEntry(paper);
  renderPaperSidebar();
}

function jumpToEntry(i) {
  const p = getActivePaper();
  if (!p || !p.entries || i < 0 || i >= p.entries.length) return;
  p.entryIndex = i;
  renderEntry(p);
}

function setViewMode(mode) {
  const p = getActivePaper();
  if (!p) return;
  p.viewMode = mode === 'raw' ? 'raw' : 'parsed';
  renderEntry(p);
}

function showPageImage(paper, pageNum) {
  const img   = document.getElementById('pageDisplayImg');
  const none  = document.getElementById('pageDisplayNone');
  const label = document.getElementById('pageDisplayLabel');
  const n     = paper.pageImages.length;

  // pageNum is 1-indexed; pageImages is 0-indexed
  zoomReset();

  let displayedPage = null;
  if (pageNum && pageNum >= 1 && pageNum <= n) {
    img.src            = paper.pageImages[pageNum - 1];
    img.style.display  = 'block';
    none.style.display = 'none';
    label.textContent  = `Page ${pageNum}`;
    displayedPage      = pageNum;
  } else if (pageNum && pageNum > n && n > 0) {
    img.src            = paper.pageImages[n - 1];
    img.style.display  = 'block';
    none.style.display = 'none';
    label.textContent  = `Page ${pageNum} (beyond p.\u202f${n} \u2014 showing last captured)`;
    displayedPage      = n;
  } else if (!pageNum && n > 0) {
    img.src            = paper.pageImages[0];
    img.style.display  = 'block';
    none.style.display = 'none';
    label.textContent  = 'Page 1';
    displayedPage      = 1;
  } else {
    img.style.display  = 'none';
    none.style.display = 'block';
    if (paper.pageImagesLoading) {
      none.innerHTML = `<div class="page-skeleton">
        <div class="page-skeleton-shimmer"></div>
        <p class="page-skeleton-text">Loading page preview\u2026</p>
      </div>`;
    } else if (paper.pageImages.length === 0) {
      none.innerHTML = 'No page preview available.<br>Upload the PDF to see page images.';
    } else {
      none.innerHTML = 'No page reference<br>found in this entry.';
    }
    label.textContent  = 'Page \u2014';
  }

  // Keep the always-all-pages navigator counter in sync with whatever
  // page actually got displayed (covers cell-click jumps, browse-all
  // flips, and the initial-page seeding from renderEntry).
  if (displayedPage != null) {
    paper.browseAllPagesIdx = displayedPage - 1;
    updatePageNav(paper);
  }

  // Render SVG highlight overlay for the displayed page (filtered by sub-view).
  renderHighlightOverlay(paper, displayedPage);

  // If this page has no text layer, show an inline notice — the overlay
  // can't draw rects against an image-only PDF.
  const notice = document.getElementById('scannedPageNotice');
  if (notice) {
    const isScanned = displayedPage != null
      && Array.isArray(paper.scannedPages)
      && paper.scannedPages.includes(displayedPage);
    notice.style.display = isScanned ? 'flex' : 'none';
  }
}

/* Returns true if this evidence entry should be drawn given the active
   sub-view filter.  Evidence with no field info (recovered orphans) is
   shown only in the default (no-sub-view) overlay.  Honours
   ``evidence_keys`` first when set (see _evidenceMatchesSubView). */
function _highlightMatchesSubView(highlight, subView) {
  if (!subView) return true;
  const field = highlight.field || '';
  if (!field) return false;
  const segs = _fieldSegments(field);
  if (Array.isArray(subView.evidence_keys)) {
    return subView.evidence_keys.some(k => segs.includes(k));
  }
  if (Array.isArray(subView.include_keys)) {
    return subView.include_keys.some(k => segs.includes(k));
  }
  if (Array.isArray(subView.exclude_keys)) {
    return !subView.exclude_keys.some(k => segs.includes(k));
  }
  return true;
}

function _withImageDims(paper, pageIndex0, callback) {
  paper._imgDims = paper._imgDims || {};
  if (paper._imgDims[pageIndex0]) { callback(paper._imgDims[pageIndex0]); return; }
  const probe = new Image();
  probe.onload  = () => {
    paper._imgDims[pageIndex0] = { w: probe.naturalWidth, h: probe.naturalHeight };
    callback(paper._imgDims[pageIndex0]);
  };
  probe.onerror = () => callback(null);
  probe.src = paper.pageImages[pageIndex0];
}

function renderHighlightOverlay(paper, pageNum) {
  const svg = document.getElementById('highlightOverlay');
  if (!svg) return;
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!pageNum || !Array.isArray(paper.highlights) || paper.highlights.length === 0) {
    svg.style.display = 'none';
    return;
  }
  const subView = _activeSubViewFor(paper);
  const matching = paper.highlights.filter(h =>
    h.page === pageNum && _highlightMatchesSubView(h, subView)
  );
  if (!matching.length) { svg.style.display = 'none'; return; }

  const focusedField = paper.focusedField || null;

  _withImageDims(paper, pageNum - 1, dims => {
    if (!dims) { svg.style.display = 'none'; return; }
    svg.setAttribute('viewBox', `0 0 ${dims.w} ${dims.h}`);
    svg.style.display = 'block';
    for (const h of matching) {
      const isFocused = focusedField && h.field === focusedField;
      for (const r of (h.rects || [])) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x',      r[0]);
        rect.setAttribute('y',      r[1]);
        rect.setAttribute('width',  r[2]);
        rect.setAttribute('height', r[3]);
        rect.setAttribute('class',  isFocused ? 'highlight-rect highlight-rect-focused'
                                              : 'highlight-rect');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = (h.field ? `[${h.field}] ` : '') + (h.snippet || '');
        rect.appendChild(title);
        svg.appendChild(rect);
      }
    }
  });
}

/* Map a clicked cell's ``data-path`` to the best-matching evidence
   entry on the active paper.

   ``data-path`` may be either:
     * Result-rooted, e.g. ``samples[0].factor_loadings.F1.5`` — when
       renderValueHtml walked the full parsed object, or
     * Sample-rooted, e.g. ``factor_loadings.F1.5`` — when renderEntry
       handed it the sample-level entry (the common case in MASEMiner
       sub-views, which strip the ``samples[i]`` wrapper).

   The evidence list always uses result-rooted field paths
   (``samples[i]...``).  When the clicked path lacks the wrapper, we
   try the same set of matches with ``samples[paper.entryIndex].``
   prepended.

   Match strategy, in order, on each candidate path:
     1. Exact field match.
     2. Item-row siblings — for factor_loadings cells of the form
        ``samples[i].factor_loadings.F<j>.<n>``, look for any
        ``samples[i].factor_loadings.F<k>.<n>`` (same item index n,
        any factor k).  The prompt asks the model to emit one
        per-row evidence entry anchored at one cell of the row;
        this fallback lets a click on any cell in that row find
        the row's source-table line.
     3. Progressive parent-path trim — table → sample → paper.

   Returns the {page, field, snippet, source, rects} object or null. */
function _findEvidenceForField(paper, path) {
  if (!paper || !paper.parsed || !path) return null;
  const evidence = paper.parsed.evidence;
  if (!Array.isArray(evidence) || !evidence.length) return null;

  const candidates = [path];
  if (!path.startsWith('samples[') && typeof paper.entryIndex === 'number') {
    candidates.push(`samples[${paper.entryIndex}].${path}`);
  }

  for (const candidate of candidates) {
    // (1) Exact match
    const exact = evidence.find(e => e && e.field === candidate);
    if (exact) return exact;

    // (2) Item-row sibling fallback for factor_loadings cells.
    // ``samples[0].factor_loadings.F3.5`` → look for any
    // ``samples[0].factor_loadings.F<*>.5`` entry.
    const rowRe = /^(samples\[\d+\]\.factor_loadings)\.F(\d+)\.(\d+)$/;
    const rowMatch = rowRe.exec(candidate);
    if (rowMatch) {
      const tableBase = rowMatch[1];
      const itemIdx   = rowMatch[3];
      const wantRe = new RegExp(
        '^' + tableBase.replace(/[.[\]]/g, c => '\\' + c) +
        '\\.F\\d+\\.' + itemIdx + '$',
      );
      const rowHit = evidence.find(e => e && typeof e.field === 'string' && wantRe.test(e.field));
      if (rowHit) return rowHit;
    }

    // (3) Progressive parent-path trim
    let cur = candidate;
    while (cur) {
      const dotIdx = cur.lastIndexOf('.');
      const brackIdx = cur.lastIndexOf('[');
      const cutAt = Math.max(dotIdx, brackIdx);
      if (cutAt <= 0) break;
      cur = cur.slice(0, cutAt);
      const hit = evidence.find(e => e && e.field === cur);
      if (hit) return hit;
    }
  }
  return null;
}

/* Delegated click handler.  When the user clicks a value cell with a
   ``data-path`` attribute, jump the PDF viewer to that field's
   evidence page (if any) and mark the matching highlight rect with
   the ``highlight-rect-focused`` class so it pops out from the rest.

   Also handles the ``×`` delete buttons attached to ``_table`` rows
   (see renderTableHtml).  Both behaviours share one listener so
   renderEntry doesn't have to re-attach anything when it rewrites the
   inner HTML — event delegation off ``#resultDisplay`` is enough.

   Attached once at startup; idempotent guard so renderEntry can't
   double-attach. */
function _attachEvidenceClickHandler() {
  if (_attachEvidenceClickHandler._attached) return;
  _attachEvidenceClickHandler._attached = true;
  const display = document.getElementById('resultDisplay');
  if (!display) return;
  display.addEventListener('click', e => {
    // (a) Row-delete button on _table rows.  Handled first so the click
    //     doesn't fall through into the evidence handler.
    const delBtn = e.target.closest('[data-delete-path]');
    if (delBtn && display.contains(delBtn)) {
      e.stopPropagation();
      e.preventDefault();
      _deleteTableRow(delBtn.getAttribute('data-delete-path'));
      return;
    }
    // (b) Cell click → jump to evidence page + focus the highlight.
    const cell = e.target.closest('[data-path]');
    if (!cell || !display.contains(cell)) return;
    const path = cell.getAttribute('data-path');
    if (!path) return;
    const paper = getActivePaper();
    if (!paper) return;

    // Single-selection: move the ``rv-selected`` marker to the newly
    // clicked cell so only one cell ever looks "active" at a time.
    display.querySelectorAll('.rv-selected').forEach(el => el.classList.remove('rv-selected'));
    cell.classList.add('rv-selected');

    const hit = _findEvidenceForField(paper, path);
    if (!hit || !hit.page) {
      // No evidence for this cell — drop any lingering focused-rect
      // highlight from the previous click so the PDF view doesn't
      // keep pointing at the old field.
      if (paper.focusedField) {
        paper.focusedField = null;
        // browseAllPagesIdx is the single source of truth for which
        // page is currently displayed (kept in sync by showPageImage).
        const displayedPage = (paper.browseAllPagesIdx || 0) + 1;
        renderHighlightOverlay(paper, displayedPage);
      }
      return;
    }
    paper.focusedField = hit.field;
    // Sync the page-nav bookkeeping so prev/next arrows still work.
    if (Array.isArray(paper.evidencePages)) {
      const idx = paper.evidencePages.indexOf(hit.page);
      if (idx >= 0) paper.evidencePageIdx = idx;
    }
    paper.browseAllPagesIdx = hit.page - 1;
    showPageImage(paper, hit.page);
  });
}

/* Path-walker for delete-row.  ``path`` is a sample-relative dotted
   path that ends in ``[N]`` — e.g.  ``correlation_matrix._table[2]``.
   Walks into the active paper's current entry, splices the row out of
   the parent array, and re-renders.  No-ops if anything along the
   way is missing (defensive — the model can omit keys the UI doesn't
   know about). */
function _deleteTableRow(path) {
  if (!path) return;
  const paper = getActivePaper();
  if (!paper || !paper.entries) return;
  const entry = paper.entries[paper.entryIndex];
  if (!entry) return;

  // Split a path like "a.b._table[2]" into segments + final index.
  const m = /^(.+)\[(\d+)\]$/.exec(path);
  if (!m) return;
  const arrPath = m[1];
  const idx     = parseInt(m[2], 10);

  // Walk into entry along arrPath (dot-separated keys, no further
  // brackets expected for our _table use case).
  const parts = arrPath.split('.');
  let node = entry;
  for (const part of parts) {
    if (node == null || typeof node !== 'object') return;
    node = node[part];
  }
  if (!Array.isArray(node) || idx < 0 || idx >= node.length) return;
  node.splice(idx, 1);

  // Drop any per-cell overrides that targeted the removed row.  The
  // override keys look like ``correlation_matrix._table[2].r`` — once
  // the row is gone, those entries point at stale data.  Cheaper to
  // wipe overrides for the whole sub-tree than to re-index the rest.
  const ovs = paper.overrides[paper.entryIndex];
  if (ovs && typeof ovs === 'object') {
    const prefix = `${path}.`;
    for (const k of Object.keys(ovs)) {
      if (k === path || k.startsWith(prefix)) delete ovs[k];
    }
  }

  renderEntry(paper);
  autoSaveSession?.();
}

/* ──────────────────────────────────────────────────────────
   Entry navigation
────────────────────────────────────────────────────────── */

function getActivePaper() {
  return state.papers.find(p => p.id === state.activePaperId) || null;
}

function prevEntry() {
  const p = getActivePaper();
  if (p && p.entryIndex > 0) { p.entryIndex--; renderEntry(p); }
}

function nextEntry() {
  const p = getActivePaper();
  if (p && p.entries && p.entryIndex < p.entries.length - 1) { p.entryIndex++; renderEntry(p); }
}

/* ── Evidence page nav (flip through pages cited by one entry) ── */

/* Two modes for the page navigator:
   - if there are evidence pages cited by the entry, flip through those
     The navigator now ALWAYS spans every captured PDF page so the user
     can flip through the whole document.  ``evidencePages`` is still
     used as a hint — the navigator opens on the first cited page so the
     relevant content is what the user sees first — but it doesn't
     restrict navigation.  Highlights remain filtered by the active
     sub-view (so flipping to a non-evidence page just shows that page
     unhighlighted, while flipping to an evidence page draws its rects).

   ``paper.browseAllPagesIdx`` is the single 0-indexed cursor into
   ``paper.pageImages``; ``evidencePageIdx`` is no longer used for
   navigation but kept on the paper object for backwards compatibility
   with serialised history payloads. */
function _isBrowseAllMode(paper) {
  return (paper.pageImages?.length || 0) > 1;
}

function updatePageNav(paper) {
  const navEl  = document.getElementById('pageEvidenceNav');
  const prevEl = document.getElementById('pageNavPrev');
  const nextEl = document.getElementById('pageNavNext');
  const cntEl  = document.getElementById('pageNavCounter');

  const total = paper.pageImages?.length || 0;
  const idx   = paper.browseAllPagesIdx || 0;

  if (total > 1) {
    navEl.style.display = 'flex';
    cntEl.textContent   = `Page ${idx + 1} / ${total}`;
    prevEl.disabled     = idx === 0;
    nextEl.disabled     = idx === total - 1;
    // Mark the current page as "highlighted" if it carries any sub-view-
    // matched rects, so the user can see at a glance which pages have
    // evidence to look at.
    const hasHl = Array.isArray(paper.evidencePages) && paper.evidencePages.includes(idx + 1);
    navEl.classList.toggle('page-nav-on-evidence', hasHl);
  } else {
    navEl.style.display = 'none';
  }
}

function prevEvidencePage() {
  const p = getActivePaper();
  if (!p) return;
  if ((p.browseAllPagesIdx || 0) === 0) return;
  p.browseAllPagesIdx = (p.browseAllPagesIdx || 0) - 1;
  updatePageNav(p);
  showPageImage(p, p.browseAllPagesIdx + 1);
}

function nextEvidencePage() {
  const p = getActivePaper();
  if (!p) return;
  const max = (p.pageImages?.length || 1) - 1;
  if ((p.browseAllPagesIdx || 0) >= max) return;
  p.browseAllPagesIdx = (p.browseAllPagesIdx || 0) + 1;
  updatePageNav(p);
  showPageImage(p, p.browseAllPagesIdx + 1);
}


/* ──────────────────────────────────────────────────────────
   JSON parsing helpers
────────────────────────────────────────────────────────── */

function stripFences(text) {
  // Remove markdown code fences: ```json ... ``` or ``` ... ```
  return text.replace(/^```(?:json)?\s*\n?/i, '').replace(/\n?```\s*$/i, '').trim();
}

function parseFull(text) {
  const cleaned = stripFences(text || '');
  try { return JSON.parse(cleaned); } catch {}
  return _repairTruncatedJson(cleaned);
}

/* Mirror of pdf_utils._parse_result_json's repair path: tolerates preamble,
   trailing prose, and mid-output truncation by closing open strings/containers. */
function _repairTruncatedJson(text) {
  if (!text) return null;
  const startA = text.indexOf('{');
  const startB = text.indexOf('[');
  const candidates = [startA, startB].filter(i => i !== -1);
  if (!candidates.length) return null;
  const candidate = text.slice(Math.min(...candidates));

  // Try shrinking from the end — strips trailing prose.
  for (let end = candidate.length; end > 0; end--) {
    const c = candidate[end - 1];
    if (c !== '}' && c !== ']') continue;
    try { return JSON.parse(candidate.slice(0, end)); } catch {}
  }

  // Truncation repair: tokenizer-style walk.
  const stack = [];
  let inString = false, escape = false;
  let expectingValue = false;
  let safeCut = 0;
  let safeStack = [];
  const markSafe = (pos) => {
    if (stack.length) { safeCut = pos; safeStack = stack.slice(); }
  };

  for (let i = 0; i < candidate.length; ) {
    const ch = candidate[i];
    if (inString) {
      if (escape) escape = false;
      else if (ch === '\\') escape = true;
      else if (ch === '"') {
        inString = false;
        if (expectingValue) { markSafe(i + 1); expectingValue = false; }
      }
      i++; continue;
    }
    if (ch === ' ' || ch === '\t' || ch === '\n' || ch === '\r') { i++; continue; }
    if (ch === '"') { inString = true; i++; continue; }
    if (ch === '{') { stack.push('}'); expectingValue = false; i++; continue; }
    if (ch === '[') { stack.push(']'); expectingValue = true;  i++; continue; }
    if (ch === '}' || ch === ']') {
      if (stack.length && stack[stack.length - 1] === ch) stack.pop();
      markSafe(i + 1);
      expectingValue = false;
      i++; continue;
    }
    if (ch === ':') { expectingValue = true;  i++; continue; }
    if (ch === ',') {
      markSafe(i);
      expectingValue = stack[stack.length - 1] === ']';
      i++; continue;
    }
    // Number / true / false / null — consume until a structural char
    let j = i;
    while (j < candidate.length && !'{}[],:" \t\n\r'.includes(candidate[j])) j++;
    if (expectingValue) { markSafe(j); expectingValue = false; }
    i = j;
  }

  if (!safeStack.length) return null;
  let repaired = candidate.slice(0, safeCut).replace(/[\s,]+$/, '');
  for (let k = safeStack.length - 1; k >= 0; k--) repaired += safeStack[k];
  try { return JSON.parse(repaired); } catch { return null; }
}

function parseEntries(text) {
  const parsed = parseFull(text);
  if (!parsed) return null;
  if (Array.isArray(parsed)) return parsed.length ? parsed : null;
  if (typeof parsed === 'object') {
    for (const val of Object.values(parsed)) {
      if (Array.isArray(val) && val.length > 0) return val;
    }
    return [parsed]; // single object — one entry
  }
  return null;
}

/* ── Page-number helpers ─────────────────────────────────────────────────── */

const PAGE_SKIP = new Set(['factor_loadings', 'factor_correlations']);

function toPageNum(v) {
  if (typeof v === 'number' && Number.isInteger(v) && v > 0) return v;
  if (typeof v === 'string') {
    const n = parseInt(v, 10);
    if (!isNaN(n) && n > 0) return n;
  }
  return null;
}

/* Collect ALL page numbers referenced in evidence blocks within obj.
   Returns a sorted array of unique page numbers. */
function findAllEntryPages(obj, collected = new Set()) {
  if (!obj || typeof obj !== 'object') return collected;
  if (Array.isArray(obj)) {
    for (const item of obj) findAllEntryPages(item, collected);
    return collected;
  }
  if (obj.evidence) {
    const ev = obj.evidence;
    if (!Array.isArray(ev) && typeof ev === 'object') {
      const p = toPageNum(ev.page);
      if (p !== null) collected.add(p);
    }
    if (Array.isArray(ev)) {
      for (const e of ev) {
        if (e) { const p = toPageNum(e.page); if (p !== null) collected.add(p); }
      }
    }
  }
  const direct = toPageNum(obj.page);
  if (direct !== null) collected.add(direct);
  for (const [key, val] of Object.entries(obj)) {
    if (PAGE_SKIP.has(key) || !val || typeof val !== 'object') continue;
    findAllEntryPages(val, collected);
  }
  return collected;
}

/* Convenience: return the first page number found, or null. */
function findEntryPage(obj) {
  const pages = [...findAllEntryPages(obj)].sort((a, b) => a - b);
  return pages.length ? pages[0] : null;
}

/* ──────────────────────────────────────────────────────────
   Formatted result renderer
────────────────────────────────────────────────────────── */

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatKey(k) {
  return k.replace(/_/g, ' ')
    .replace(/([A-Z])/g, ' $1')
    .trim()
    .replace(/\b\w/g, c => c.toUpperCase());
}

/* Detect objects whose values are all numbers/null — render as compact grid */
function isNumericObject(obj) {
  const vals = Object.values(obj);
  return vals.length >= 4 && vals.every(v => v === null || typeof v === 'number');
}

/* Detect evidence blocks: objects with snippet or (page + source) */
function isEvidenceBlock(obj) {
  return typeof obj === 'object' && obj !== null && !Array.isArray(obj)
    && ('snippet' in obj || ('page' in obj && 'source' in obj));
}

/* ── Table detection ──────────────────────────────────────────────────────────
   Two shapes we treat as tables:
   A) Array of objects with shared keys: [{a:1,b:2}, {a:3,b:4}]
   B) Object whose values are objects with shared keys: {row1:{a:1,b:2}, row2:{a:3,b:4}}
   Evidence arrays are excluded — they have a dedicated renderer. */

function _shareKeys(objects) {
  if (objects.length < 2) return false;
  const allKeys = new Set();
  objects.forEach(o => Object.keys(o).forEach(k => allKeys.add(k)));
  if (allKeys.size === 0) return false;
  // Average overlap with the union must be ≥ 60%
  const totalOverlap = objects.reduce((sum, o) => sum + Object.keys(o).length, 0);
  return totalOverlap / (objects.length * allKeys.size) >= 0.6;
}

function isTableArray(arr) {
  if (!Array.isArray(arr) || arr.length < 2) return false;
  if (!arr.every(x => x && typeof x === 'object' && !Array.isArray(x))) return false;
  if (arr.some(isEvidenceBlock)) return false; // evidence has its own renderer
  return _shareKeys(arr);
}

function isTableMap(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false;
  const vals = Object.values(obj);
  if (vals.length < 2) return false;
  if (!vals.every(x => x && typeof x === 'object' && !Array.isArray(x))) return false;
  if (vals.some(isEvidenceBlock)) return false;
  return _shareKeys(vals);
}

function _collectColumns(rows) {
  // Preserve insertion order from the first row, then append any new keys from later rows
  const cols = [];
  const seen = new Set();
  for (const row of rows) {
    for (const k of Object.keys(row)) {
      if (!seen.has(k)) { seen.add(k); cols.push(k); }
    }
  }
  return cols;
}

function _renderCellHtml(value, path) {
  // Render a single cell — leaf values are editable; nested values fall back to renderValueHtml
  if (value === null || value === undefined) {
    return `<span class="rv-null rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path)}" data-orig="">—</span>`;
  }
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string') {
    return renderValueHtml(value, 1, path);
  }
  // Nested object/array inside a cell — render compactly
  return renderValueHtml(value, 2, path);
}

/* Explicit table marker: the prompt instructs the model to wrap tabular data
   as {"_table": [{...row...}, ...]}.  No shape inference, no guessing — if the
   key is present and contains an array of objects, it's a table.  Rendered
   columns come from the union of row keys (in order of first appearance). */
function isMarkedTable(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false;
  const t = obj._table;
  if (!Array.isArray(t) || t.length === 0) return false;
  return t.every(r => r && typeof r === 'object' && !Array.isArray(r));
}

function renderMarkedTable(data, path) {
  const rows      = data._table;
  const columns   = _collectColumns(rows);
  const tablePath = path ? `${path}._table` : '_table';
  return renderTableHtml(rows, columns, null, tablePath, 'marker');
}

/* Detect a flat dict whose keys follow "<group>.<index>" and whose values are
   all numbers/null (e.g. {F1.1: 0.5, F1.2: 0.7, F2.1: 0, ...}).  This is the
   shape produced by extraction prompts that ask for individual loadings as
   leaf keys; pivoting them into a 2-D table is far more readable. */
const _DOTTED_KEY_RE = /^([A-Za-z][A-Za-z0-9]*)\.(\d+)$/;

function isDottedNumericTable(obj) {
  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return false;
  const keys = Object.keys(obj);
  if (keys.length < 4) return false;
  // Tolerate a small number of stray non-matching keys (e.g. the model
  // sometimes emits an abbreviated ``"...": null`` placeholder for empty
  // factor columns instead of enumerating each F4.* / F5.* explicitly).
  // A single such stray used to flip the entire dict to the flat
  // "numgrid" fallback layout.  We now require ≥75 % of the keys to
  // match the dotted regex, which keeps generic dicts from being
  // misclassified while making the table renderer robust to common
  // model abbreviations.
  //
  // A single group (e.g. a one-dimensional factor solution with only
  // F1.1…F1.n) is accepted too: renderDottedTable renders it as a
  // clean two-column "Item | F1" table rather than the misleading
  // multi-column card grid the numgrid fallback produced.
  const groups = new Set();
  const items  = new Set();
  let matches = 0;
  for (const k of keys) {
    const m = k.match(_DOTTED_KEY_RE);
    if (!m) continue;
    const v = obj[k];
    if (v !== null && typeof v !== 'number') continue;
    groups.add(m[1]);
    items.add(parseInt(m[2], 10));
    matches++;
  }
  return matches >= 4
      && groups.size >= 1
      && items.size  >= 2
      && matches >= keys.length * 0.75;
}

/* Resolve an entry-relative path like ``factor_loadings`` or
   ``samples[0].factor_loadings`` to the live nested object inside
   ``entry``.  Returns null when any segment is missing. */
function _resolveEntryPath(entry, path) {
  if (!path) return entry;
  const tokens = path.match(/[^.\[\]]+|\[\d+\]/g) || [];
  let cur = entry;
  for (const t of tokens) {
    if (cur == null || typeof cur !== 'object') return null;
    if (t.startsWith('[')) cur = cur[parseInt(t.slice(1, -1), 10)];
    else                   cur = cur[t];
  }
  return cur ?? null;
}

/* Compute the next group label for an "add column" action.  Looks at the
   numeric suffix on the existing group prefixes and increments.  E.g.
   ``["F1","F2","F3"]`` → ``"F4"``; ``["R1","R2"]`` → ``"R3"``.  Falls back
   to ``"F1"`` when there are no existing groups. */
function _nextDottedGroup(groups) {
  if (!groups.length) return 'F1';
  const last = groups[groups.length - 1];
  const m = last.match(/^([A-Za-z]+)(\d+)$/);
  if (!m) return last + '_new';
  const prefix = m[1];
  let maxN = 0;
  for (const g of groups) {
    const mm = g.match(/^([A-Za-z]+)(\d+)$/);
    if (mm && mm[1] === prefix) maxN = Math.max(maxN, parseInt(mm[2], 10));
  }
  return prefix + (maxN + 1);
}

/* Mutators for the dotted-table renderer.  Each operates on the live
   entry's dict (no copy), then re-renders.  Confirms before destructive
   actions to avoid accidental clicks. */
function addDottedColumn(paperId, tablePath) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper) return;
  const obj = _resolveEntryPath(paper.entries[paper.entryIndex], tablePath);
  if (!obj || typeof obj !== 'object') return;
  const groups = [], items = new Set(), seen = new Set();
  for (const k of Object.keys(obj)) {
    const m = k.match(_DOTTED_KEY_RE);
    if (!m) continue;
    if (!seen.has(m[1])) { seen.add(m[1]); groups.push(m[1]); }
    items.add(parseInt(m[2], 10));
  }
  const newGroup = _nextDottedGroup(groups);
  const itemList = items.size ? [...items].sort((a, b) => a - b) : [1];
  for (const it of itemList) obj[`${newGroup}.${it}`] = null;
  if (state.activePaperId === paper.id) renderEntry(paper);
}

function removeDottedColumn(paperId, tablePath, group) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper) return;
  if (!confirm(`Delete column ${group}? All values in that column will be removed.`)) return;
  const obj = _resolveEntryPath(paper.entries[paper.entryIndex], tablePath);
  if (!obj || typeof obj !== 'object') return;
  for (const k of Object.keys(obj)) {
    const m = k.match(_DOTTED_KEY_RE);
    if (m && m[1] === group) delete obj[k];
  }
  // Sweep stale overrides for the deleted cells so they don't leak into export.
  const ov = paper.overrides[paper.entryIndex] || {};
  for (const p of Object.keys(ov)) {
    const tail = tablePath ? p.slice(tablePath.length + 1) : p;
    const m = tail.match(_DOTTED_KEY_RE);
    if (m && m[1] === group) delete ov[p];
  }
  if (state.activePaperId === paper.id) renderEntry(paper);
}

function addDottedRow(paperId, tablePath) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper) return;
  const obj = _resolveEntryPath(paper.entries[paper.entryIndex], tablePath);
  if (!obj || typeof obj !== 'object') return;
  const groups = [], items = new Set(), seen = new Set();
  for (const k of Object.keys(obj)) {
    const m = k.match(_DOTTED_KEY_RE);
    if (!m) continue;
    if (!seen.has(m[1])) { seen.add(m[1]); groups.push(m[1]); }
    items.add(parseInt(m[2], 10));
  }
  const sortedItems = [...items].sort((a, b) => a - b);
  const newItem = (sortedItems.length ? sortedItems[sortedItems.length - 1] : 0) + 1;
  const groupList = groups.length ? groups : ['F1'];
  for (const g of groupList) obj[`${g}.${newItem}`] = null;
  if (state.activePaperId === paper.id) renderEntry(paper);
}

function removeDottedRow(paperId, tablePath, item) {
  const paper = state.papers.find(p => p.id === paperId);
  if (!paper) return;
  if (!confirm(`Delete row ${item}? All values in that row will be removed.`)) return;
  const obj = _resolveEntryPath(paper.entries[paper.entryIndex], tablePath);
  if (!obj || typeof obj !== 'object') return;
  for (const k of Object.keys(obj)) {
    const m = k.match(_DOTTED_KEY_RE);
    if (m && parseInt(m[2], 10) === item) delete obj[k];
  }
  const ov = paper.overrides[paper.entryIndex] || {};
  for (const p of Object.keys(ov)) {
    const tail = tablePath ? p.slice(tablePath.length + 1) : p;
    const m = tail.match(_DOTTED_KEY_RE);
    if (m && parseInt(m[2], 10) === item) delete ov[p];
  }
  if (state.activePaperId === paper.id) renderEntry(paper);
}

function renderDottedTable(obj, path) {
  // Preserve insertion order for groups, numeric sort for items
  const groupOrder = [];
  const groupSeen  = new Set();
  const itemSet    = new Set();
  for (const k of Object.keys(obj)) {
    const m = k.match(_DOTTED_KEY_RE);
    if (!m) continue;
    if (!groupSeen.has(m[1])) { groupSeen.add(m[1]); groupOrder.push(m[1]); }
    itemSet.add(parseInt(m[2], 10));
  }
  const items = [...itemSet].sort((a, b) => a - b);

  const escPath = escHtml(path || '');
  const paperId = escHtml(state.activePaperId || '');

  const head = `<thead><tr>
    <th class="rv-tbl-rowlabel">Item</th>
    ${groupOrder.map(g => `<th>
      <div class="rv-tbl-coltitle">
        <span>${escHtml(g)}</span>
        <button class="rv-tbl-del" title="Delete column ${escHtml(g)}"
                onclick="removeDottedColumn('${paperId}', '${escPath}', '${escHtml(g)}')">&times;</button>
      </div>
    </th>`).join('')}
    <th class="rv-tbl-addcol">
      <button class="rv-tbl-add-btn" title="Add column"
              onclick="addDottedColumn('${paperId}', '${escPath}')">+</button>
    </th>
  </tr></thead>`;

  const body = items.map(item => {
    const labelCell = `<td class="rv-tbl-rowlabel">
      <div class="rv-tbl-rowlabel-inner">
        <span>${item}</span>
        <button class="rv-tbl-del" title="Delete row ${item}"
                onclick="removeDottedRow('${paperId}', '${escPath}', ${item})">&times;</button>
      </div>
    </td>`;
    const cells = groupOrder.map(g => {
      const key      = `${g}.${item}`;
      const val      = obj[key];
      const cellPath = path ? `${path}.${key}` : key;
      return `<td>${_renderCellHtml(val === undefined ? null : val, cellPath)}</td>`;
    }).join('');
    return `<tr>${labelCell}${cells}<td class="rv-tbl-addcol-cell"></td></tr>`;
  }).join('');

  const addRowFooter = `<tr class="rv-tbl-addrow">
    <td class="rv-tbl-rowlabel">
      <button class="rv-tbl-add-btn" title="Add row"
              onclick="addDottedRow('${paperId}', '${escPath}')">+</button>
    </td>
    ${groupOrder.map(() => `<td></td>`).join('')}
    <td></td>
  </tr>`;

  const caption = `<span class="rv-table-caption-icon">▦</span> Table · ${items.length} rows × ${groupOrder.length + 1} cols `
                + `<span class="rv-table-source rv-table-source-auto" data-tip="Auto-detected from dotted F1.1-style keys. Update your prompt to use the _table marker for explicit tables.">auto-detected</span>`;
  return `<div class="rv-table-wrap">
    <div class="rv-table-caption">${caption}</div>
    <table class="rv-table">${head}<tbody>${body}${addRowFooter}</tbody></table>
  </div>`;
}

function renderTableHtml(rows, columns, rowLabels, path, kind) {
  // rows:       array of row-data objects
  // columns:    list of column keys
  // rowLabels:  null (for plain array) or list of strings (parent keys for object-map)
  // path:       path prefix for editable cells
  // kind:       'marker' (explicit _table) | 'auto' | undefined — drives the caption
  // Only explicit ``_table`` arrays (kind='marker') AND tables driven by
  // a real array path (not object-map labels) get the trailing delete
  // column.  Auto-detected tables and dotted-key tables don't, since
  // their underlying shape isn't a plain array we can splice from.
  const showDelete = kind === 'marker' && !rowLabels && path;
  const labelHeader  = rowLabels ? '<th class="rv-tbl-rowlabel"></th>' : '';
  const deleteHeader = showDelete ? '<th class="rv-tbl-del-head" aria-label=""></th>' : '';
  const head = `<thead><tr>${labelHeader}${columns.map(c => `<th>${escHtml(formatKey(c))}</th>`).join('')}${deleteHeader}</tr></thead>`;
  const body = rows.map((row, i) => {
    const labelCell = rowLabels
      ? `<td class="rv-tbl-rowlabel">${escHtml(rowLabels[i])}</td>`
      : '';
    const cells = columns.map(col => {
      const cellPath = rowLabels
        ? (path ? `${path}.${rowLabels[i]}.${col}` : `${rowLabels[i]}.${col}`)
        : (path ? `${path}[${i}].${col}` : `[${i}].${col}`);
      const val = row[col];
      return `<td>${_renderCellHtml(val === undefined ? null : val, cellPath)}</td>`;
    }).join('');
    const deleteCell = showDelete
      ? `<td class="rv-tbl-del"><button type="button" class="rv-row-del"
                  data-delete-path="${escHtml(path)}[${i}]"
                  title="Delete this row" aria-label="Delete row">&times;</button></td>`
      : '';
    return `<tr>${labelCell}${cells}${deleteCell}</tr>`;
  }).join('');

  // Caption explains why the data is shown as a table
  const nRows = rows.length;
  const nCols = columns.length + (rowLabels ? 1 : 0) + (showDelete ? 1 : 0);
  const caption = kind === 'marker'
    ? `<span class="rv-table-caption-icon">▦</span> Table · ${nRows} rows × ${nCols} cols <span class="rv-table-source" data-tip="The model wrapped this data with the _table marker — rendered exactly as the model declared it.">explicit</span>`
    : kind === 'auto'
    ? `<span class="rv-table-caption-icon">▦</span> Table · ${nRows} rows × ${nCols} cols <span class="rv-table-source rv-table-source-auto" data-tip="Auto-detected from the data shape (array of objects, object of objects, or dotted F1.1 keys). Update your prompt to use the _table marker for explicit tables.">auto-detected</span>`
    : `<span class="rv-table-caption-icon">▦</span> Table · ${nRows} rows × ${nCols} cols`;

  return `<div class="rv-table-wrap">
    <div class="rv-table-caption">${caption}</div>
    <table class="rv-table">${head}<tbody>${body}</tbody></table>
  </div>`;
}

/* Detect labeling result: object with a string 'label' and string 'rationale'. */
function isLabelingResult(obj) {
  return typeof obj === 'object' && obj !== null && !Array.isArray(obj)
    && typeof obj.label === 'string' && obj.label.length > 0
    && typeof obj.rationale === 'string';
}

function renderLabelingResult(data, path) {
  const { label, rationale, confidence, evidence, ...rest } = data;
  let html = `<div class="label-result">`;

  // Label badge
  html += `<div class="label-badge-row">`;
  html += `<span class="label-badge rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path ? path + '.label' : 'label')}" data-orig="${escHtml(label)}">${escHtml(label)}</span>`;
  if (confidence != null) {
    const pct = typeof confidence === 'number'
      ? (confidence <= 1 ? Math.round(confidence * 100) + '%' : Math.round(confidence) + '%')
      : escHtml(String(confidence));
    html += `<span class="label-confidence">Confidence: <span class="rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path ? path + '.confidence' : 'confidence')}" data-orig="${escHtml(String(confidence))}">${pct}</span></span>`;
  }
  html += `</div>`;

  // Rationale prose
  html += `<div class="label-rationale rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path ? path + '.rationale' : 'rationale')}" data-orig="${escHtml(rationale)}">${escHtml(rationale)}</div>`;

  // Any extra keys (besides evidence)
  for (const [k, v] of Object.entries(rest)) {
    const keyPath = path ? `${path}.${k}` : k;
    html += `<div class="rv-row label-extra-row">
      <dt class="rv-key">${escHtml(formatKey(k))}</dt>
      <dd class="rv-val">${renderValueHtml(v, 1, keyPath)}</dd>
    </div>`;
  }

  // Evidence section
  if (Array.isArray(evidence) && evidence.length > 0) {
    html += `<div class="label-evidence-section">`;
    html += `<div class="label-evidence-heading">Evidence</div>`;
    html += `<div class="rv-list">${evidence.map((item, i) => `
      <div class="rv-list-item">
        <span class="rv-idx">${i + 1}</span>
        <div class="rv-list-body">${renderValueHtml(item, 1, path ? `${path}.evidence[${i}]` : `evidence[${i}]`)}</div>
      </div>`).join('')}</div>`;
    html += `</div>`;
  }

  html += `</div>`;
  return html;
}

function renderValueHtml(data, depth = 0, path = '') {
  if (data === null || data === undefined) {
    return `<span class="rv-null rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path)}" data-orig="">\u2014</span>`;
  }
  if (typeof data === 'boolean') {
    return `<span class="rv-bool rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path)}" data-orig="${data}">${data ? 'Yes' : 'No'}</span>`;
  }
  if (typeof data === 'number') {
    return `<span class="rv-num rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path)}" data-orig="${data}">${data}</span>`;
  }
  if (typeof data === 'string') {
    return data === ''
      ? `<span class="rv-null rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path)}" data-orig="">\u2014</span>`
      : `<span class="rv-str rv-editable" contenteditable="true" spellcheck="false" data-path="${escHtml(path)}" data-orig="${escHtml(data)}">${escHtml(data)}</span>`;
  }
  if (Array.isArray(data)) {
    if (data.length === 0) return '<span class="rv-null">empty list</span>';
    // Table shape: array of homogeneous objects
    if (isTableArray(data)) {
      return renderTableHtml(data, _collectColumns(data), null, path, 'auto');
    }
    return `<div class="rv-list">${data.map((item, i) => `
      <div class="rv-list-item">
        <span class="rv-idx">${i + 1}</span>
        <div class="rv-list-body">${renderValueHtml(item, depth + 1, path ? `${path}[${i}]` : `[${i}]`)}</div>
      </div>`).join('')}</div>`;
  }
  // Explicit table marker takes priority — no shape inference needed
  if (isMarkedTable(data)) {
    return renderMarkedTable(data, path);
  }
  // Object — labeling results get a dedicated layout (label badge + rationale prose)
  if (isLabelingResult(data)) {
    return renderLabelingResult(data, path);
  }
  // Object — evidence blocks are citation metadata, not editable data values
  if (isEvidenceBlock(data)) {
    const parts = [];
    if (data.snippet) {
      parts.push(`<blockquote class="ev-snippet">${escHtml(data.snippet)}</blockquote>`);
    }
    const tags = [];
    if (data.page != null) tags.push(`<span class="ev-tag ev-page">p.&nbsp;${data.page}</span>`);
    if (data.source)        tags.push(`<span class="ev-tag ev-source">${escHtml(data.source)}</span>`);
    if (tags.length) parts.push(`<div class="ev-tags">${tags.join('')}</div>`);
    return `<div class="rv-evidence">${parts.join('')}</div>`;
  }
  // Table shape: object whose values are homogeneous objects (e.g. {row1:{...}, row2:{...}})
  if (isTableMap(data)) {
    const rowLabels = Object.keys(data);
    const rows      = Object.values(data);
    return renderTableHtml(rows, _collectColumns(rows), rowLabels, path, 'auto');
  }
  // Table shape: flat dict with "<group>.<index>" keys (e.g. F1.1, F1.2, F2.1, ...)
  if (isDottedNumericTable(data)) {
    return renderDottedTable(data, path);
  }
  if (isNumericObject(data)) {
    // Compact grid for factor loadings etc.
    return `<div class="rv-numgrid">${Object.entries(data).map(([k, v]) => {
      const cellPath = path ? `${path}.${k}` : k;
      return `
        <div class="rv-numgrid-cell">
          <span class="rv-numgrid-k">${escHtml(k)}</span>
          <span class="${v === null ? 'rv-null rv-editable' : 'rv-num rv-editable'}" contenteditable="true" spellcheck="false" data-path="${escHtml(cellPath)}" data-orig="${v === null ? '' : v}">${v === null ? '\u2014' : v}</span>
        </div>`;
    }).join('')}</div>`;
  }
  const objEntries = Object.entries(data);
  if (objEntries.length === 0) return '<span class="rv-null">empty</span>';
  return `<dl class="rv-obj ${depth === 0 ? 'rv-root' : ''}">${objEntries.map(([k, v]) => `
    <div class="rv-row">
      <dt class="rv-key">${escHtml(formatKey(k))}</dt>
      <dd class="rv-val">${renderValueHtml(v, depth + 1, path ? `${path}.${k}` : k)}</dd>
    </div>`).join('')}</dl>`;
}

/* ──────────────────────────────────────────────────────────
   Editable fields — override tracking
────────────────────────────────────────────────────────── */

/* Re-apply stored overrides after a re-render, and mark edited cells. */
function applyOverrides(paper) {
  const display = document.getElementById('resultDisplay');
  const overrides = paper.overrides[paper.entryIndex] || {};
  display.querySelectorAll('[data-path]').forEach(el => {
    const path = el.dataset.path;
    if (overrides[path]) {
      el.textContent = overrides[path].final_value;
      el.classList.add('rv-edited');
    } else {
      el.classList.remove('rv-edited');
    }
  });
}

/* Blur handler for contenteditable leaf spans (via event delegation). */
function handleFieldEdit(event) {
  const el = event.target;
  if (!el.classList.contains('rv-editable')) return;
  const display  = document.getElementById('resultDisplay');
  const paperId  = display.dataset.paperId;
  const entryIdx = parseInt(display.dataset.entryIdx, 10);
  const path     = el.dataset.path;
  const orig     = el.dataset.orig;            // original value as string
  const final    = el.textContent.trim();

  const paper = state.papers.find(p => p.id === paperId);
  if (!paper || path === undefined) return;

  if (!paper.overrides[entryIdx]) paper.overrides[entryIdx] = {};

  // Treat "—" typed back in as a revert to null/empty
  const isReverted = final === orig || (final === '\u2014' && orig === '');
  if (isReverted) {
    delete paper.overrides[entryIdx][path];
    el.classList.remove('rv-edited');
  } else {
    paper.overrides[entryIdx][path] = {
      original_value: orig,
      final_value:    final,
      human_override: true,
    };
    el.classList.add('rv-edited');
  }
}

/* ──────────────────────────────────────────────────────────
   Zoom / pan for page image
────────────────────────────────────────────────────────── */

const zoom = { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0 };

function applyZoom() {
  const img = document.getElementById('pageDisplayImg');
  const svg = document.getElementById('highlightOverlay');
  const t   = `translate(${zoom.x}px,${zoom.y}px) scale(${zoom.scale})`;
  img.style.transform = t;
  // Keep the SVG overlay locked to the image at every zoom/pan tick
  if (svg) svg.style.transform = t;
}

function clampZoom() {
  if (zoom.scale <= 1) { zoom.x = 0; zoom.y = 0; return; }
  const container = document.getElementById('pageZoomContainer');
  const img       = document.getElementById('pageDisplayImg');
  const cw = container.offsetWidth;
  const ch = container.offsetHeight;
  const iw = img.offsetWidth  * zoom.scale;
  const ih = img.offsetHeight * zoom.scale;
  // Don't let more than 80% of the image go off-screen on either side
  const mx = cw * 0.8;
  const my = ch * 0.8;
  zoom.x = Math.min(zoom.x,  mx);
  zoom.x = Math.max(zoom.x, -(iw - cw + mx));
  zoom.y = Math.min(zoom.y,  my);
  zoom.y = Math.max(zoom.y, -(ih - ch + my));
}

function zoomIn()    { applyZoomDelta(1.4); }
function zoomOut()   { applyZoomDelta(1 / 1.4); }
function zoomReset() { zoom.scale = 1; zoom.x = 0; zoom.y = 0; applyZoom(); }

function applyZoomDelta(factor) {
  const container = document.getElementById('pageZoomContainer');
  // Zoom toward the center of the container
  const cx = container.offsetWidth  / 2;
  const cy = container.offsetHeight / 2;
  const newScale = Math.min(Math.max(zoom.scale * factor, 1), 10);
  zoom.x = cx - (cx - zoom.x) * (newScale / zoom.scale);
  zoom.y = cy - (cy - zoom.y) * (newScale / zoom.scale);
  zoom.scale = newScale;
  clampZoom();
  applyZoom();
}

function initZoomPan() {
  const container = document.getElementById('pageZoomContainer');

  // Scroll wheel → zoom centered on cursor
  container.addEventListener('wheel', e => {
    e.preventDefault();
    const rect     = container.getBoundingClientRect();
    const mx       = e.clientX - rect.left;
    const my       = e.clientY - rect.top;
    const factor   = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.min(Math.max(zoom.scale * factor, 1), 10);
    zoom.x = mx - (mx - zoom.x) * (newScale / zoom.scale);
    zoom.y = my - (my - zoom.y) * (newScale / zoom.scale);
    zoom.scale = newScale;
    clampZoom();
    applyZoom();
  }, { passive: false });

  // Mouse drag → pan
  container.addEventListener('mousedown', e => {
    if (zoom.scale <= 1) return;
    zoom.dragging = true;
    zoom.startX = e.clientX - zoom.x;
    zoom.startY = e.clientY - zoom.y;
    container.classList.add('dragging');
  });

  window.addEventListener('mousemove', e => {
    if (!zoom.dragging) return;
    zoom.x = e.clientX - zoom.startX;
    zoom.y = e.clientY - zoom.startY;
    clampZoom();
    applyZoom();
  });

  window.addEventListener('mouseup', () => {
    if (!zoom.dragging) return;
    zoom.dragging = false;
    document.getElementById('pageZoomContainer').classList.remove('dragging');
  });

  // Touch drag → pan; pinch → zoom
  let lastTouchDist = null;
  container.addEventListener('touchstart', e => {
    if (e.touches.length === 1 && zoom.scale > 1) {
      zoom.dragging = true;
      zoom.startX = e.touches[0].clientX - zoom.x;
      zoom.startY = e.touches[0].clientY - zoom.y;
    } else if (e.touches.length === 2) {
      zoom.dragging = false;
      lastTouchDist = Math.hypot(
        e.touches[1].clientX - e.touches[0].clientX,
        e.touches[1].clientY - e.touches[0].clientY,
      );
    }
  }, { passive: true });

  container.addEventListener('touchmove', e => {
    if (e.touches.length === 1 && zoom.dragging) {
      e.preventDefault();
      zoom.x = e.touches[0].clientX - zoom.startX;
      zoom.y = e.touches[0].clientY - zoom.startY;
      clampZoom();
      applyZoom();
    } else if (e.touches.length === 2 && lastTouchDist !== null) {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
      const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
      const dist = Math.hypot(
        e.touches[1].clientX - e.touches[0].clientX,
        e.touches[1].clientY - e.touches[0].clientY,
      );
      const factor   = dist / lastTouchDist;
      const newScale = Math.min(Math.max(zoom.scale * factor, 1), 10);
      zoom.x = cx - (cx - zoom.x) * (newScale / zoom.scale);
      zoom.y = cy - (cy - zoom.y) * (newScale / zoom.scale);
      zoom.scale = newScale;
      lastTouchDist = dist;
      clampZoom();
      applyZoom();
    }
  }, { passive: false });

  container.addEventListener('touchend', () => {
    zoom.dragging = false;
    lastTouchDist = null;
  });

  // Double-click → toggle 2.5× zoom at click point
  container.addEventListener('dblclick', e => {
    if (zoom.scale > 1.2) {
      zoomReset();
    } else {
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const newScale = 2.5;
      zoom.x = mx - (mx - zoom.x) * (newScale / zoom.scale);
      zoom.y = my - (my - zoom.y) * (newScale / zoom.scale);
      zoom.scale = newScale;
      clampZoom();
      applyZoom();
    }
  });
}

/* Wire up event listeners on the result display panel. */
function initResultDisplay() {
  const display = document.getElementById('resultDisplay');

  // Capture edits on blur
  display.addEventListener('blur', handleFieldEdit, true);

  // Click / focus on any value cell jumps the PDF panel to the page where
  // the model cited evidence for that field.  Falls back to the closest
  // parent path when the leaf doesn't have its own evidence entry.
  display.addEventListener('focusin', handleCellEvidenceJump);
  display.addEventListener('click',   handleCellEvidenceJump);

  // Prevent Enter from inserting <br>/<div>; treat it as commit
  display.addEventListener('keydown', e => {
    if (!e.target.classList.contains('rv-editable')) return;
    if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); }
  });

  // Paste as plain text only
  display.addEventListener('paste', e => {
    if (!e.target.classList.contains('rv-editable')) return;
    e.preventDefault();
    const text = (e.clipboardData || window.clipboardData).getData('text/plain');
    document.execCommand('insertText', false, text);
  });
}

/* Look up the page where the model cited evidence for a given JSON-path
   and navigate the right-hand PDF panel there.  Match priority:
     1. Exact ``field`` match (e.g. clicking ``samples[0].factor_loadings.F1.1``
        and there's an evidence entry with that exact field)
     2. Longest-prefix parent match (clicking the leaf falls back to the
        table caption evidence at ``samples[0].factor_loadings``)
     3. Sample-level fallback (any evidence under ``samples[i]``) so a
        click on ``samples[0].sex`` lands on Table 1's page even when the
        model didn't tag the leaf.
   When nothing maps to a usable page, the click is a no-op and the
   currently-displayed page stays put. */
function handleCellEvidenceJump(event) {
  const el = event.target;
  if (!el || !el.classList || !el.classList.contains('rv-editable')) return;
  const path = el.dataset.path;
  if (!path) return;
  const paper = getActivePaper();
  if (!paper || !paper.pageImages || !paper.pageImages.length) return;

  const page = _evidencePageForPath(paper, path);
  if (!page) return;
  // Skip the work (and the zoom reset inside showPageImage) when we're
  // already on the right page — clicking different cells on the same page
  // shouldn't keep collapsing the user's zoom.
  const img = document.getElementById('pageDisplayImg');
  const currentSrc = img && img.src;
  const targetSrc  = paper.pageImages[page - 1];
  if (currentSrc === targetSrc) return;

  showPageImage(paper, page);
  // Keep the page-evidence counter in sync if the resolved page is one of
  // the navigator's stops (purely cosmetic — overlay update was already
  // handled by showPageImage).
  if (Array.isArray(paper.evidencePages)) {
    const idx = paper.evidencePages.indexOf(page);
    if (idx >= 0) {
      paper.evidencePageIdx = idx;
      updatePageNav(paper);
    }
  }
}

/* Pick the most specific evidence entry whose ``field`` either equals or
   is an ancestor of ``path``, and return its 1-indexed page number.

   The rendered cell's ``data-path`` is relative to the active entry
   (e.g. ``factor_loadings.F1.1``).  Evidence fields, however, are emitted
   by the model relative to the full parsed object
   (``samples[0].factor_loadings.F1.1``).  We construct both candidate
   forms and match against either, so the lookup works regardless of
   whether the entries array was unwrapped from ``samples[]`` or sits at
   the top level. */
function _evidencePageForPath(paper, path) {
  const evidence = (paper.parsed && Array.isArray(paper.parsed.evidence))
    ? paper.parsed.evidence : [];
  if (!evidence.length || !path) return null;

  const idx = (paper.entryIndex == null) ? 0 : paper.entryIndex;
  const fullPath = `samples[${idx}].${path}`;
  const candidates = [path, fullPath];

  // Pass 1: exact match against either candidate form.
  for (const e of evidence) {
    if (!e || !e.field) continue;
    if (candidates.includes(e.field)) {
      const p = toPageNum(e.page);
      if (p) return p;
    }
  }
  // Pass 2: longest-prefix match — leaf cells fall back to their table /
  // section caption (e.g. F1.1 → factor_loadings caption).
  let bestPage = null, bestLen = -1;
  for (const e of evidence) {
    if (!e || !e.field) continue;
    for (const cand of candidates) {
      if (cand === e.field || cand.startsWith(e.field + '.')) {
        const p = toPageNum(e.page);
        if (p && e.field.length > bestLen) { bestPage = p; bestLen = e.field.length; }
      }
    }
  }
  if (bestPage) return bestPage;
  // Pass 3: any evidence under the same ``samples[i]`` parent so a click
  // on a leaf the model never explicitly cited still lands somewhere
  // sensible (typically Table 1 / the Methods page).
  const samplePrefix = `samples[${idx}]`;
  for (const e of evidence) {
    if (!e || !e.field) continue;
    if (e.field.startsWith(samplePrefix)) {
      const p = toPageNum(e.page);
      if (p) return p;
    }
  }
  return null;
}

/* ──────────────────────────────────────────────────────────
   Copy / download for active paper
────────────────────────────────────────────────────────── */

function copyResult() {
  const p = getActivePaper();
  if (p) copyToClipboard(p.result, 'resultCopyBtn');
}

function downloadResult() {
  const p = getActivePaper();
  if (!p) return;
  // When the user filled in values manually OR when we want to capture the
  // evidence/extraction-failed flags, the bare LLM response isn't enough —
  // wrap the export in a small envelope that preserves both the raw model
  // output AND the post-edit/manual data.
  const evidence = (p.evidenceTotal || 0) > 0;
  const failed   = p.manualMode === true;
  const payload = {
    filename:           p.filename,
    extraction_failed:  failed,
    evidence_present:   evidence,
    pages_processed:    p.pagesProcessed,
    token_usage:        p.tokenUsage || null,
    model:              state.model,
    resolved_model:     p.resolvedModel || null,
    entries:            p.entries || null,
    human_overrides:    _overrideList(p),
    llm_raw_response:   p.result || '',
  };
  _downloadBlob(JSON.stringify(payload, null, 2),
                p.filename.replace(/\.pdf$/i, '') + '.json',
                'application/json');
}

/* Flatten paper.overrides ({entryIdx: {path: {original_value, final_value}}})
   into a flat list suitable for JSON export. */
function _overrideList(paper) {
  const out = [];
  for (const [entryIdx, fields] of Object.entries(paper.overrides || {})) {
    for (const [fieldPath, ov] of Object.entries(fields)) {
      out.push({
        entry_index:    parseInt(entryIdx, 10),
        field_path:     fieldPath,
        original_value: ov.original_value,
        final_value:    ov.final_value,
        human_override: ov.human_override,
      });
    }
  }
  return out;
}

/* ── CSV export ─────────────────────────────────────────────────────────── */

/* Flatten an entry to {col: scalar} using dot notation.
   Evidence arrays are kept whole (serialized as JSON in one column) since
   they don't fit a flat tabular shape. */
function _flattenEntry(obj, prefix = '', out = {}) {
  if (obj === null || obj === undefined) {
    if (prefix) out[prefix] = '';
    return out;
  }
  if (typeof obj !== 'object') {
    out[prefix] = obj;
    return out;
  }
  if (Array.isArray(obj)) {
    // Keep evidence arrays as JSON strings; otherwise enumerate by index
    if (prefix.endsWith('evidence') || obj.every(isEvidenceBlock)) {
      out[prefix] = JSON.stringify(obj);
      return out;
    }
    obj.forEach((item, i) => _flattenEntry(item, `${prefix}[${i}]`, out));
    return out;
  }
  // Plain object
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    _flattenEntry(v, key, out);
  }
  return out;
}

function _csvEscape(value) {
  if (value === null || value === undefined) return '';
  const s = String(value);
  // RFC 4180: wrap in quotes if it contains comma, quote, newline; double internal quotes
  if (/[,"\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function _toCsv(rows, columns) {
  const header = columns.map(_csvEscape).join(',');
  const body   = rows.map(r => columns.map(c => _csvEscape(r[c])).join(',')).join('\n');
  return header + '\n' + body + '\n';
}

function _entriesFromPaper(paper) {
  // Apply human overrides on top of original entries before exporting.
  // Also annotate every row with two provenance flags so downstream consumers
  // can filter / audit:
  //   _extraction_failed  — true when the row is user-filled because the
  //                         model's output couldn't be parsed (manual mode)
  //   _evidence_present   — true when the model returned at least one
  //                         evidence snippet (regardless of whether we
  //                         could highlight it)
  const failed   = paper.manualMode === true ? 'true' : 'false';
  const evidence = (paper.evidenceTotal || 0) > 0 ? 'true' : 'false';
  if (!paper.entries || paper.entries.length === 0) {
    // Extraction failed AND user didn't fill in manually — emit a stub row
    // so the export still records the run (with the raw response for audit).
    return [{
      _extraction_failed: 'true',
      _evidence_present:  evidence,
      _llm_raw_response:  paper.result || '',
    }];
  }
  return paper.entries.map((e, i) => {
    const f = _flattenEntry(e);
    const ov = paper.overrides[i] || {};
    for (const [path, info] of Object.entries(ov)) {
      f[path] = info.final_value;
    }
    f._extraction_failed = failed;
    f._evidence_present  = evidence;
    return f;
  });
}

function _downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadResultCsv() {
  const p = getActivePaper();
  if (!p) return;
  const rows = _entriesFromPaper(p);
  if (!rows.length) { showToast('No tabular entries to export.'); return; }
  const columns = _collectColumns(rows);
  const csv     = _toCsv(rows, columns);
  _downloadBlob(csv, p.filename.replace(/\.pdf$/i, '') + '.csv', 'text/csv;charset=utf-8');
}

function downloadAllPapersCsv() {
  const done = state.papers.filter(p => p.status === 'done');
  if (!done.length) { showToast('No completed papers to download.'); return; }

  // Stack all entries from all papers; prepend a `_filename` column for traceability
  const allRows = [];
  for (const p of done) {
    for (const row of _entriesFromPaper(p)) {
      allRows.push({ _filename: p.filename, ...row });
    }
  }
  if (!allRows.length) { showToast('No tabular entries to export.'); return; }
  const columns = _collectColumns(allRows);
  // Ensure _filename comes first
  const ordered = ['_filename', ...columns.filter(c => c !== '_filename')];
  const csv     = _toCsv(allRows, ordered);
  _downloadBlob(csv, 'extraction_results_all.csv', 'text/csv;charset=utf-8');
}

/* Download all processed papers as a single consolidated JSON.
   Each paper includes its original extracted entries plus a flat list of
   human overrides so the caller can reconstruct final values. */
function downloadAllPapers() {
  const done = state.papers.filter(p => p.status === 'done');
  if (!done.length) { showToast('No completed papers to download.'); return; }

  const output = {
    exported_at:  new Date().toISOString(),
    prompt:       state.generatedPrompt,
    model:        state.model,
    papers: done.map(p => ({
      filename:                p.filename,
      extraction_failed:       p.manualMode === true,
      evidence_present:        (p.evidenceTotal || 0) > 0,
      pages_processed:         p.pagesProcessed,
      token_usage:             p.tokenUsage || null,
      // ``model`` is the alias the user picked.  ``resolved_model`` is
      // the dated snapshot the provider actually served (e.g.
      // ``gpt-5-2025-09-15``) — the reproducibility-grade record.
      model:                   state.model,
      resolved_model:          p.resolvedModel || null,
      entries:                 p.entries,
      human_overrides:         _overrideList(p),
      original_model_response: p.result,
    })),
  };

  const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'extraction_results_all.json';
  a.click();
  URL.revokeObjectURL(url);
}

/* ──────────────────────────────────────────────────────────
   Load results from file (step 1 → step 8 shortcut)
────────────────────────────────────────────────────────── */

// Staged data for the review flow
let reviewPdfFiles  = [];   // File[]
let reviewJsonData  = null; // parsed JSON, held until user clicks "Load"

function selectLoadOption() {
  document.getElementById('modeGrid').style.display        = 'none';
  document.getElementById('jsonUploadPanel').style.display = '';
  showJsonStage1();
}

function cancelLoadOption() {
  document.getElementById('modeGrid').style.display        = '';
  document.getElementById('jsonUploadPanel').style.display = 'none';
  const errEl = document.getElementById('jsonError');
  if (errEl) errEl.style.display = 'none';
  reviewPdfFiles = [];
  reviewJsonData = null;
  _resetJsonStage2();
}

function showJsonStage1() {
  document.getElementById('jsonStage1').style.display = '';
  document.getElementById('jsonStage2').style.display = 'none';
}

function showJsonStage2() {
  document.getElementById('jsonStage1').style.display = 'none';
  document.getElementById('jsonStage2').style.display = '';
  _resetJsonStage2();
  initJsonUploadZone();
  initReviewPdfZone();
}

function _resetJsonStage2() {
  reviewPdfFiles = [];
  reviewJsonData = null;
  const hint = document.getElementById('reviewPdfHint');
  if (hint) hint.textContent = 'or drop here';
  const jsonHint = document.getElementById('jsonDropHint');
  if (jsonHint) jsonHint.textContent = 'or drop here';
  const ready = document.getElementById('jsonReadyRow');
  if (ready) ready.style.display = 'none';
  const loadBtn = document.getElementById('jsonLoadBtn');
  if (loadBtn) loadBtn.disabled = true;
  const errEl = document.getElementById('jsonError');
  if (errEl) errEl.style.display = 'none';
}

function initDropZone(zoneId, onFiles) {
  const zone = document.getElementById(zoneId);
  if (!zone || zone.dataset.init) return;
  zone.dataset.init = '1';
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    onFiles([...e.dataTransfer.files]);
  });
}

function initJsonUploadZone() {
  initDropZone('jsonUploadZone', files => {
    const json = files.find(f => f.name.toLowerCase().endsWith('.json'));
    if (json) loadJsonFile(json);
    // Also capture any PDFs dropped at the same time
    const pdfs = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length) addReviewPdfs(pdfs);
  });
}

function initReviewPdfZone() {
  initDropZone('reviewPdfZone', files => {
    const pdfs = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length) addReviewPdfs(pdfs);
  });
}

/* Drop zone on the results page (step 8) for rehydrating page previews
   on a past batch whose page images aged out of the server's in-memory
   cache.  Reuses the same /api/pages endpoint as the review flow. */
function initReuploadPdfZone() {
  initDropZone('reuploadPdfZone', files => {
    const pdfs = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length) handleReuploadPdfSelect(pdfs);
  });
}

function addReviewPdfs(files) {
  for (const f of files) {
    if (!reviewPdfFiles.find(x => x.name === f.name)) reviewPdfFiles.push(f);
  }
  const hint = document.getElementById('reviewPdfHint');
  if (hint) hint.textContent = `${reviewPdfFiles.length} PDF${reviewPdfFiles.length !== 1 ? 's' : ''} ready`;
}

function handleJsonFileSelect(event) {
  const file = event.target.files[0];
  if (file) loadJsonFile(file);
}

function handleReviewPdfSelect(event) {
  addReviewPdfs([...event.target.files]);
}

function showJsonError(msg) {
  const el = document.getElementById('jsonError');
  el.textContent    = msg;
  el.style.display  = 'block';
}

function reconstructOverrides(overrideList) {
  const overrides = {};
  for (const ov of (overrideList || [])) {
    const idx = ov.entry_index;
    if (!overrides[idx]) overrides[idx] = {};
    overrides[idx][ov.field_path] = {
      original_value: ov.original_value,
      final_value:    ov.final_value,
      human_override: ov.human_override,
    };
  }
  return overrides;
}

// Step 1: parse and validate — does NOT navigate. Shows ready state + Load button.
async function loadJsonFile(file) {
  document.getElementById('jsonError').style.display = 'none';

  if (!file.name.toLowerCase().endsWith('.json')) {
    showJsonError('Please upload a .json file.');
    return;
  }

  let data;
  try {
    data = JSON.parse(await file.text());
  } catch (e) {
    showJsonError('Could not parse file: ' + e.message);
    return;
  }

  if (!data.papers || !Array.isArray(data.papers) || data.papers.length === 0) {
    showJsonError('This doesn\'t look like a valid results file — expected a "papers" array.');
    return;
  }

  reviewJsonData = data;

  // Show ready confirmation and enable Load button
  const n = data.papers.length;
  const jsonHint = document.getElementById('jsonDropHint');
  if (jsonHint) jsonHint.textContent = `\u2713 ${file.name} (${n} paper${n !== 1 ? 's' : ''})`;

  const ready = document.getElementById('jsonReadyRow');
  if (ready) ready.style.display = 'flex';

  const loadBtn = document.getElementById('jsonLoadBtn');
  if (loadBtn) loadBtn.disabled = false;
}

// Step 2: commit — build paper objects and navigate to results.
function commitLoadJson() {
  if (!reviewJsonData) return;
  const data = reviewJsonData;

  state.generatedPrompt = data.prompt || '';
  state.model           = data.model  || 'gpt-4o';
  state.loadedFromFile  = true;

  state.papers = data.papers.map(p => {
    const rawResult = p.original_model_response || '';
    return {
      id:              crypto.randomUUID(),
      blob:            null,
      filename:        p.filename || 'unknown.pdf',
      status:          'done',
      result:          rawResult,
      rawResponse:     rawResult,
      pageImages:      [],
      entries:         p.entries || null,
      parsed:          parseFull(rawResult) || p.entries || null,
      entryIndex:      0,
      evidencePages:   [],
      evidencePageIdx: 0,
      evidenceCount:   null,
      tokenUsage:      p.token_usage || null,
      resolvedModel:   p.resolved_model || null,
      pagesProcessed:  p.pages_processed || 0,
      error:           null,
      overrides:       reconstructOverrides(p.human_overrides),
    };
  });

  const pdfsToFetch = [...reviewPdfFiles];
  state.activePaperId = state.papers[0].id;
  cancelLoadOption();
  displayPaper(state.papers[0]);
  goTo(8);

  if (pdfsToFetch.length) fetchReviewPageImages(pdfsToFetch);
}

/* ──────────────────────────────────────────────────────────
   PDF re-upload on the results page
   ────────────────────────────────────────────────────────── */

/* Toggle the "Page preview unavailable" notice on step 8 based on
   whether any paper is missing rasterised page images.  Server-side
   page images live only in process memory (see jobs.py:_PAGE_IMAGES),
   so a Fly restart between extraction and review wipes them — the user
   then needs to re-upload the original PDFs to get the side-by-side
   viewer back. */
function updateReuploadNotice() {
  const el = document.getElementById('reuploadPdfNotice');
  if (!el) return;
  const done    = state.papers.filter(p => p.status === 'done');
  const missing = done.filter(p => !p.pageImages || p.pageImages.length === 0);
  if (!missing.length) {
    el.style.display = 'none';
    return;
  }
  const detail = document.getElementById('reuploadPdfDetail');
  if (detail) {
    const names = missing.slice(0, 5).map(p => p.filename).join(', ');
    const extra = missing.length > 5 ? `, +${missing.length - 5} more` : '';
    detail.textContent = `Upload the original PDF${missing.length !== 1 ? 's' : ''} to enable the side-by-side viewer (matching by filename): ${names}${extra}.`;
  }
  el.style.display = 'flex';
}

/* Drop or pick handler for the results-page PDF re-upload zone.
   Matches each PDF to a paper by filename and calls the shared
   ``fetchReviewPageImages`` helper to rasterise + attach the images. */
async function handleReuploadPdfSelect(eventOrFiles) {
  let files;
  if (eventOrFiles && eventOrFiles.target && eventOrFiles.target.files) {
    files = Array.from(eventOrFiles.target.files);
  } else {
    files = Array.from(eventOrFiles || []);
  }
  if (!files.length) return;
  const hint = document.getElementById('reuploadPdfHint');
  if (hint) hint.textContent = `Loading ${files.length} PDF${files.length !== 1 ? 's' : ''}…`;
  await fetchReviewPageImages(files);
  updateReuploadNotice();
  if (hint) {
    const stillMissing = state.papers.filter(p => p.status === 'done' && (!p.pageImages || !p.pageImages.length)).length;
    hint.textContent = stillMissing
      ? `${stillMissing} paper${stillMissing !== 1 ? 's' : ''} still need their PDF`
      : 'All PDFs loaded';
  }
  // Reset the input so the same file can be picked again if needed
  const input = document.getElementById('reuploadPdfInput');
  if (input) input.value = '';
}

async function fetchReviewPageImages(pdfFiles) {
  for (const pdfFile of pdfFiles) {
    // Match by filename (case-insensitive)
    const paper = state.papers.find(
      p => p.filename.toLowerCase() === pdfFile.name.toLowerCase()
    );
    if (!paper) continue; // PDF has no matching paper in the JSON

    const form = new FormData();
    form.append('pdf',    pdfFile, pdfFile.name);
    form.append('result', paper.result || '');

    try {
      const res  = await fetchScoped('/api/pages', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok || data.error) {
        console.warn(`[fetchReviewPageImages] ${pdfFile.name}: ${data.error}`);
        continue;
      }
      paper.pageImages = data.page_images || [];
      // If this paper is currently displayed, refresh the page view
      if (state.activePaperId === paper.id) {
        renderEntry(paper);
      }
    } catch (err) {
      console.warn(`[fetchReviewPageImages] ${pdfFile.name}:`, err);
    }
  }
}

/* ──────────────────────────────────────────────────────────
   Reset
────────────────────────────────────────────────────────── */

function goBackFromResults() {
  // If results came from a loaded file there's no upload step to go back to
  if (state.loadedFromFile) startOver();
  else goTo(6);
}

function startOver() {
  Object.assign(state, {
    mode: null, provider: 'openai', model: 'gpt-4o', apiKey: '', baseUrl: '',
    providerCredentials: {},
    question: '', context: '', inputMode: 'generate',
    generatedPrompt: '', useTextExtraction: true,
    notifyEmail: '', batchId: null,
    selectedFiles: [], papers: [],
    activePaperId: null, loadedFromFile: false, setupReturnStep: null,
  });
  document.getElementById('questionInput').value     = '';
  document.getElementById('contextInput').value      = '';
  document.getElementById('manualPromptInput').value = '';
  showStep3Choice();
  renderFileList();
  cancelLoadOption();
  clearAutoSave();
  goTo(1);
}

/* ──────────────────────────────────────────────────────────
   Shared utilities
────────────────────────────────────────────────────────── */

function copyToClipboard(text, btnId) {
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById(btnId);
    const orig = btn.innerHTML;
    btn.textContent = '✓ Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('copied'); }, 2200);
  }).catch(() => showToast('Could not copy — please select and copy the text manually.'));
}

function resetCopyBtn(btnId) {
  const btn = document.getElementById(btnId);
  btn.classList.remove('copied');
  btn.innerHTML = `
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
    Copy`;
}

function toggleHelpDrawer() {
  const drawer  = document.getElementById('helpDrawer');
  const overlay = document.getElementById('helpOverlay');
  const isOpen  = drawer.classList.toggle('open');
  overlay.classList.toggle('visible', isOpen);
  drawer.setAttribute('aria-hidden', isOpen ? 'false' : 'true');
  document.body.classList.toggle('help-open', isOpen);
}

// Global keyboard shortcuts.  Avoid intercepting keys while the user is
// typing in an input/textarea/contenteditable so editing values still works.
document.addEventListener('keydown', e => {
  const t = e.target;
  const typing = t && (
    t.tagName === 'INPUT' ||
    t.tagName === 'TEXTAREA' ||
    t.tagName === 'SELECT' ||
    t.isContentEditable
  );

  // Esc — close help drawer / download menu, regardless of focus
  if (e.key === 'Escape') {
    if (document.getElementById('helpDrawer').classList.contains('open')) {
      toggleHelpDrawer();
      e.preventDefault();
      return;
    }
    closeDownloadMenu();
    return;
  }

  if (typing) return;

  // ? — open help (Shift-/ on most layouts)
  if (e.key === '?') { toggleHelpDrawer(); e.preventDefault(); return; }

  // Results-page shortcuts (step 8 only)
  if (state.step !== 8) return;

  const papers = state.papers.filter(p => p.status === 'done' || p.status === 'error');
  const idx    = papers.findIndex(p => p.id === state.activePaperId);
  const active = papers[idx];

  // n / → next paper · p / ← prev paper
  if ((e.key === 'n' || e.key === 'ArrowRight') && idx >= 0 && idx < papers.length - 1) {
    setActivePaper(papers[idx + 1].id); e.preventDefault(); return;
  }
  if ((e.key === 'p' || e.key === 'ArrowLeft')  && idx > 0) {
    setActivePaper(papers[idx - 1].id); e.preventDefault(); return;
  }

  // j / ↓ next entry · k / ↑ prev entry (within the active paper)
  if ((e.key === 'j' || e.key === 'ArrowDown')) { nextEntry(); e.preventDefault(); return; }
  if ((e.key === 'k' || e.key === 'ArrowUp'))   { prevEntry(); e.preventDefault(); return; }

  // [ / ] flip evidence pages
  if (e.key === ']') { nextEvidencePage(); e.preventDefault(); return; }
  if (e.key === '[') { prevEvidencePage(); e.preventDefault(); return; }

  // e — start editing the first editable cell in the current entry
  if (e.key === 'e' && active) {
    const first = document.querySelector('#resultDisplay .rv-editable');
    if (first) { first.focus(); e.preventDefault(); }
  }
});

let toastTimer = null;
function showToast(message, kind = 'error') {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.classList.remove('error-toast', 'success-toast');
  el.classList.add(kind === 'success' ? 'success-toast' : 'error-toast');
  el.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('visible'), 4500);
}

// Initialisation is handled by the DOMContentLoaded listener near onProviderChange.

/* ──────────────────────────────────────────────────────────
   Debug helper — invoke from the browser console as `dbgHL()`
   to dump the highlight pipeline state for the active paper.

   Reports: total highlights received from server, how many match the
   currently-displayed page, how the active sub-view filters them, and the
   raw highlight entries (with rects, fields, snippets) so we can see
   exactly which evidence items the rect-locator could and couldn't place.
────────────────────────────────────────────────────────── */
window.dbgHL = function() {
  const p = getActivePaper();
  if (!p) { console.log('[dbgHL] no active paper'); return null; }

  const subView = _activeSubViewFor(p);
  const img     = document.getElementById('pageDisplayImg');
  const pageNum = (() => {
    // Reverse-lookup the displayed page from the img src's index in pageImages
    if (!img || !p.pageImages) return null;
    const i = p.pageImages.indexOf(img.src);
    return i >= 0 ? i + 1 : null;
  })();

  const highlights = p.highlights || [];
  const byPage = {};
  for (const h of highlights) {
    byPage[h.page] = byPage[h.page] || 0;
    byPage[h.page]++;
  }
  const matchingPage = highlights.filter(h => h.page === pageNum);
  const matchingPageAndSV = matchingPage.filter(h => _highlightMatchesSubView(h, subView));

  console.group('[dbgHL] paper', p.filename);
  console.log('subView          :', subView ? subView.id : '(none)');
  console.log('current pageNum  :', pageNum);
  console.log('paper.evidencePages:', p.evidencePages);
  console.log('paper.pageImages : count =', p.pageImages?.length || 0);
  console.log('paper.highlights : total =', highlights.length, 'by page =', byPage);
  console.log('on this page     :', matchingPage.length);
  console.log('+ sub-view filter:', matchingPageAndSV.length);
  console.table(highlights.map(h => ({
    page:    h.page,
    field:   h.field,
    rects:   (h.rects || []).length,
    snippet: (h.snippet || '').slice(0, 60),
  })));
  console.groupEnd();
  return { paper: p, highlights, byPage, pageNum, subView };
};
