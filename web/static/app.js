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
// crypto.randomUUID is only exposed in secure contexts (HTTPS or
// localhost).  Running the dev server over a LAN IP / plain http
// returns ``undefined`` for the function — calling it throws
// "crypto.randomUUID is not a function" and any UI path that mints
// an id (paper objects, session ids, batch ids) fails silently from
// the user's perspective.  Single helper, used everywhere, with a
// getRandomValues-backed v4 fallback when available and a
// Math.random fallback as a last resort.
function _uuidV4() {
  if (typeof crypto !== 'undefined') {
    if (typeof crypto.randomUUID === 'function') {
      try { return crypto.randomUUID(); } catch (_) { /* fall through */ }
    }
    if (typeof crypto.getRandomValues === 'function') {
      const b = new Uint8Array(16);
      crypto.getRandomValues(b);
      b[6] = (b[6] & 0x0f) | 0x40;   // version 4
      b[8] = (b[8] & 0x3f) | 0x80;   // variant 10
      const hex = [...b].map(x => x.toString(16).padStart(2, '0'));
      return `${hex.slice(0,4).join('')}-${hex.slice(4,6).join('')}-${hex.slice(6,8).join('')}-${hex.slice(8,10).join('')}-${hex.slice(10,16).join('')}`;
    }
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function _getOrCreateSessionId() {
  try {
    let sid = localStorage.getItem('paperlens.sessionId');
    if (!sid) {
      sid = _uuidV4();
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
      : state.mode === 'summarize'  ? 'Summarise paper'
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
  // activePreset carries sub_views + the per-sub-view confidence_keys
  // that drive the results-panel tabs and the confidence-badge filter.
  // Without it the results page renders the entry unfiltered (records
  // AND metadata in one view, no sub-tabs in the sidebar) after a
  // page refresh — the symptom that surfaced as "I don't see a
  // Descriptives tab".  Stored as-is so the sub_views array round-
  // trips intact; rehydrated on autoRestoreSession.
  'activePreset',
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

    // Cross-mode preset bleed-through guard: the masem preset only
    // belongs on /maseminer (or a maseminer-only deploy).  If the
    // snapshot has it but the user is currently on PaperLens, drop it
    // — otherwise downstream code (renderConfidenceBadges, manual
    // scaffold, Start-over) keeps acting like we're in MASEMiner.
    const onMaseminerNow = window.location.pathname === '/maseminer'
      || !!window.__MASEMINER_ONLY__;
    if (!onMaseminerNow
        && state.activePreset
        && typeof state.activePreset.id === 'string'
        && state.activePreset.id.startsWith('masem')) {
      state.activePreset = null;
    }

    // Re-fetch the active preset from the server in the background so
    // any local edits since the snapshot was written (renamed sub-views,
    // new confidence_keys, prompt changes) take effect.  We keep the
    // stale snapshot value synchronous so the rest of restoreSession
    // can proceed; the fresh fetch arrives moments later and overwrites
    // ``state.activePreset`` in place, then re-renders the sidebar so
    // the updated sub-tabs appear without a manual refresh.
    if (state.activePreset?.id) {
      fetchScoped(`/api/presets/${encodeURIComponent(state.activePreset.id)}`)
        .then(r => r.ok ? r.json() : null)
        .then(fresh => {
          if (fresh && fresh.id) {
            state.activePreset = fresh;
            if (typeof renderPaperSidebar === 'function') renderPaperSidebar();
            const active = (state.papers || []).find(p => p.id === state.activePaperId);
            if (active && active.status === 'done' && typeof renderEntry === 'function') {
              renderEntry(active);
            }
          }
        })
        .catch(() => { /* network failure → keep stale snapshot, harmless */ });
    }

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
  } else if (state.mode === 'summarize') {
    document.getElementById('step3Heading').textContent = 'Describe what each summary should cover';
    document.getElementById('step3Sub').textContent     = 'Name the sections and the level of detail you want, plus any focus areas';
    document.getElementById('questionInput').placeholder =
      'E.g., Summarise each paper in four sections — background, methods, findings, limitations — written for a researcher familiar with the field…';
    document.getElementById('contextInput').placeholder =
      'E.g., Findings section must include effect sizes verbatim where reported. Skip implementation details unless they bear on validity…';
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
  document.getElementById('step3Choice').style.display     = '';
  document.getElementById('aiSection').style.display       = 'none';
  document.getElementById('manualSection').style.display   = 'none';
  const designer = document.getElementById('designerSection');
  if (designer) designer.style.display = 'none';
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
  document.getElementById('step3Choice').style.display     = 'none';
  document.getElementById('manualSection').style.display   = isManual ? '' : 'none';
  // The "Generate with AI" path now goes through the structured prompt
  // designer (web/static/prompt-designer.js) instead of the legacy free-
  // text question + context textareas.  ``aiSection`` is hidden — its
  // hidden #questionInput / #contextInput still receive the assembled
  // prompt-input on submit so the downstream pipeline is unchanged.
  document.getElementById('aiSection').style.display = 'none';
  const designer = document.getElementById('designerSection');
  if (designer) {
    designer.style.display = isManual ? 'none' : '';
    if (!isManual && typeof window._designerInit === 'function') {
      window._designerInit();
    }
  }
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

/* ── Prompt readiness check + warning + adapt ─────────────────────────── */

// Client-side mirror of web/prompt_check.py — same detection rules so
// the warning can update without a round-trip on every keystroke.  The
// server check is the authoritative gate (at /api/extract submit time);
// this just keeps the banner honest while the user is editing.
function _checkPromptReadiness(prompt) {
  const p = String(prompt || '');
  const keyRe = name => new RegExp(`["']${name}["']\\s*:`);

  const evidenceKey   = keyRe('evidence').test(p);
  const evidenceOpens = /["']evidence["']\s*:\s*\[/.test(p);
  const subkeyHits =
    Number(keyRe('snippet').test(p))
    + Number(keyRe('page').test(p))
    + Number(keyRe('source').test(p))
    + Number(keyRe('field').test(p));
  const hasEvidence = evidenceKey && evidenceOpens && subkeyHits >= 3;

  const confKey   = keyRe('extraction_confidence').test(p);
  const confOpens = /["']extraction_confidence["']\s*:\s*\{/.test(p);
  const levelTok  = /(?:["']level["']\s*:)|(?:["'](?:high|medium|low)["'])/i.test(p);
  const hasConfidence = confKey && confOpens && levelTok;

  const missing = [];
  if (!hasEvidence)   missing.push('evidence');
  if (!hasConfidence) missing.push('extraction_confidence');
  return {
    ok:                       hasEvidence && hasConfidence,
    has_evidence_structure:   hasEvidence,
    has_confidence_structure: hasConfidence,
    missing,
  };
}

// Back-compat alias used in older call sites (notably the dataset
// extension code path).  Same yes/no semantics as the previous helper.
function _hasEvidenceSchema(prompt) {
  return _checkPromptReadiness(prompt).has_evidence_structure;
}

function _renderReadinessWarning(readiness) {
  const el = document.getElementById('promptReadinessWarning');
  if (!el) return;
  if (readiness.ok) { el.style.display = 'none'; return; }
  const body = document.getElementById('promptReadinessBody');
  if (body) {
    const parts = [];
    if (!readiness.has_evidence_structure) {
      parts.push(
        '<strong>This prompt does not request an evidence array.</strong> '
        + 'Without it, page highlighting and snippet matching won\'t work.'
      );
    }
    if (!readiness.has_confidence_structure) {
      parts.push(
        '<strong>This prompt does not request an extraction_confidence object.</strong> '
        + 'Without it, per-block confidence ratings (high / medium / low) won\'t display.'
      );
    }
    body.innerHTML = parts.join('<br><br>');
  }
  el.style.display = 'flex';
}

function updateEvidenceWarning() {
  // Recompute readiness from the current prompt and (re)render the
  // warning banner.  Called on prompt-generation success, manual prompt
  // submission, debounced textarea edits, and explicit re-fetches.
  const readiness = _checkPromptReadiness(state.generatedPrompt);
  state.lastReadiness = readiness;
  // Once a prompt becomes ok, drop any stale "Proceed anyway" ack.  This
  // means an edit that fixes the prompt re-arms the gate cleanly.
  if (readiness.ok) state.acknowledgedReadiness = false;
  _renderReadinessWarning(readiness);
}

function _openReadinessModal(readiness) {
  const overlay = document.getElementById('readinessModalOverlay');
  const body    = document.getElementById('readinessModalBody');
  if (!overlay) return;
  if (body) {
    const missing = readiness.missing || [];
    const labels = missing.map(m =>
      m === 'evidence'
        ? 'an <code>evidence</code> array (drives PDF highlighting)'
        : m === 'extraction_confidence'
          ? 'an <code>extraction_confidence</code> object (drives confidence badges)'
          : m
    );
    body.innerHTML =
      'Your prompt is missing ' + labels.join(' and ') + '. '
      + 'Without these, the corresponding UI panels will be empty after extraction.';
  }
  overlay.classList.add('is-open');
}

function _closeReadinessModal() {
  const overlay = document.getElementById('readinessModalOverlay');
  if (overlay) overlay.classList.remove('is-open');
}

function _acknowledgeReadinessAndProceed() {
  // User has read the warning and chosen to extract anyway.  Persist
  // the flag for this turn so submitUpload sends acknowledge_no_evidence=1
  // and the server-side gate lets the request through.  Cleared on any
  // prompt edit that fixes the readiness check.
  state.acknowledgedReadiness = true;
  _closeReadinessModal();
  // Continue the flow that the user had started — re-enter confirmPrompt
  // which will now bypass the gate because the ack flag is set.
  confirmPrompt();
}

// Expose to inline onclick handlers on the modal markup.
window._closeReadinessModal              = _closeReadinessModal;
window._acknowledgeReadinessAndProceed   = _acknowledgeReadinessAndProceed;

// Bind a debounced re-check to the manual-prompt textarea so the banner
// updates live as the user types/pastes.  Idempotent — safe to call
// multiple times (e.g. on script reload).
(function _bindManualPromptReadinessHook() {
  if (typeof document === 'undefined') return;
  const attach = () => {
    const ta = document.getElementById('manualPromptInput');
    if (!ta || ta._readinessBound) return;
    ta._readinessBound = true;
    let timer = null;
    ta.addEventListener('input', () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        // Treat the current textarea value as the prompt for the
        // banner check — the user hasn't necessarily clicked "Use this
        // prompt" yet, but we still want to surface the warning early.
        const readiness = _checkPromptReadiness(ta.value);
        state.lastReadiness = readiness;
        if (readiness.ok) state.acknowledgedReadiness = false;
        _renderReadinessWarning(readiness);
      }, 500);
    });
  };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
})();

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
  // Readiness gate — if the prompt is missing evidence and/or
  // extraction_confidence structure AND the user hasn't acknowledged
  // already, intercept and show the modal.  state.acknowledgedReadiness
  // is set by _acknowledgeReadinessAndProceed; cleared by any edit that
  // makes the prompt ok.  Preset-driven prompts always pass.
  if (!state.acknowledgedReadiness) {
    const readiness = _checkPromptReadiness(state.generatedPrompt);
    state.lastReadiness = readiness;
    if (!readiness.ok) {
      _openReadinessModal(readiness);
      return;
    }
  }

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

/* Compute the extraction mode (text / vision) processPaper would use for
   this paper *right now*, given the current state and the paper's
   probe.  Used by submitUpload's reuse check so toggling the text/vision
   switch in step 6 correctly triggers a fresh extraction instead of
   surfacing the previous mode's cached result.

   Mirrors the dispatch in processPaper:
   1. ``paper.forceMode``                 — explicit per-paper override
   2. ``state.useTextExtraction``         — global default
   3. probe-driven auto-fallback to vision when text layer is missing */
function _expectedModeForPaper(paper) {
  if (paper.forceMode === 'text')   return 'text';
  if (paper.forceMode === 'vision') return 'vision';
  let useText = state.useTextExtraction;
  if (useText && paper.probe && !paper.probe.error
      && paper.probe.text_layer_present === false) {
    useText = false;
  }
  return useText ? 'text' : 'vision';
}

async function submitUpload() {
  if (state.selectedFiles.length === 0) { showToast('Please select at least one PDF.'); return; }
  if (state.selectedFiles.length > config.maxBatchPapers) {
    showToast(`Batch limit is ${config.maxBatchPapers} papers. Please remove ${state.selectedFiles.length - config.maxBatchPapers} file(s) or split into multiple batches.`);
    return;
  }

  // Local alias for the top-level _uuidV4 helper — kept so the rest of
  // this function's body (which calls ``uuid()`` repeatedly) stays
  // untouched.
  const uuid = _uuidV4;

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
      if (existing) {
        // In-flight runs are left alone — interrupting them mid-job
        // creates orphan jobs on the server.  Done runs are reused only
        // when the model AND prompt AND extraction mode (text vs vision)
        // match what produced them; if the user went back to step 2
        // (changed model) or step 5 (edited prompt) or toggled the
        // text/vision switch in step 6, the cached result is stale and
        // must be re-processed.
        if (existing.status === 'processing') {
          newPapers.push(existing);
          continue;
        }
        const expectedMode = _expectedModeForPaper(existing);
        if (existing.status === 'done'
            && existing.lastModelUsed  === state.model
            && existing.lastPromptUsed === state.generatedPrompt
            && existing.lastModeUsed   === expectedMode) {
          newPapers.push(existing);
          continue;
        }
        // Stale (model/prompt/mode changed) — fall through to the
        // fresh-paper branch so the new settings actually run.
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
  // Record the prompt used too so the staleness check in
  // submitUpload can detect a back-to-step-5 prompt edit and refuse
  // to reuse the old result.
  paper.lastPromptUsed = state.generatedPrompt;

  const form = new FormData();
  form.append('api_key',             state.apiKey);
  form.append('model',               effectiveModel);
  form.append('prompt',              state.generatedPrompt);
  form.append('use_text_extraction', useText ? '1' : '0');
  if (state.baseUrl) form.append('base_url', state.baseUrl);
  if (state.batchId) form.append('batch_id', state.batchId);
  // When the user has clicked "Proceed anyway" in the readiness modal,
  // tell the server to bypass the structural readiness gate.  Without
  // this flag the server returns 400 with code=prompt_missing_structure.
  if (state.acknowledgedReadiness) form.append('acknowledge_no_evidence', '1');
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
      if (isActiveP && subViews?.length > 1 && p.status === 'done') {
        html.push(_renderSidebarSubTabs(p, subViews));
      }
    } else {
      // Multi-entry papers: one header row per paper showing just the
      // filename, then one indented child row per entry showing just
      // the sample_id (no paper-name prefix on every child).  Clicking
      // the header opens entry 0 so the header itself acts as a
      // shortcut to the first entry (typically "Paper metadata").
      const headerCls = ['paper-item', 'paper-item-header',
                         isActiveP ? 'active-parent' : '',
                         `status-${p.status}`].filter(Boolean).join(' ');
      const headerOnclick = clickable ? `onclick="setActivePaperEntry('${p.id}', 0)"` : '';
      html.push(`
        <div class="${headerCls}" ${headerOnclick}>
          <span class="paper-status-icon">${icon}</span>
          <span class="paper-name-wrap">
            <span class="paper-name">${baseName}</span>
            ${phaseLabel}
          </span>
          ${stopBtn}
        </div>`);
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
            <span class="paper-sample-bullet">•</span>
            <span class="paper-name-wrap">
              <span class="paper-sample-label">${escHtml(sampleName)}</span>
            </span>
          </div>`);
        if (isActiveSample && subViews?.length > 1) {
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

/* Pick the initial PDF page to show when opening an entry.  Prefers
 * the page of the entry's ``table_caption`` evidence so opening a
 * TABLE-5 entry lands on the table itself, not (say) the abstract
 * page where a headline-justifying snippet happens to live.  Falls
 * back to the lowest evidence page when no table_caption is present
 * for this entry. */
function _preferredInitialPage(parsed, entryIndex, evidencePages) {
  if (Array.isArray(parsed?.evidence)) {
    for (const e of parsed.evidence) {
      if (!e || typeof e !== 'object' || typeof e.field !== 'string') continue;
      const idx = _evidenceEntryIndex(e.field);
      if (idx !== null && idx !== entryIndex) continue;
      if (!e.field.endsWith('.table_caption')) continue;
      const p = toPageNum(e.page);
      if (p !== null) return p;
    }
  }
  return evidencePages[0] ?? null;
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
  // User-built presets (from the structured prompt designer) live in
  // localStorage, not on the server.  Check there first — if the id
  // matches a user preset, activate it without hitting the API.  This
  // also means user presets keep working offline / on a stale cache.
  if (typeof presetId === 'string'
      && presetId.startsWith('user-')
      && typeof window._designerLoadUserPresets === 'function') {
    const userPresets = window._designerLoadUserPresets();
    const candidate = userPresets[presetId];
    if (candidate) {
      preset = candidate;
    }
  }
  if (!preset) {
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

/* Populate the inline "Pre-built workflows" panel and decide whether
   the disclosure card that toggles it should appear on step 1.  Panel
   itself stays collapsed by default — only the disclosure card on the
   primary task picker reveals it.  When the server has zero presets
   the disclosure card is hidden entirely to avoid a dead-end click. */
async function renderInlineWorkflows() {
  const wrap     = document.getElementById('prebuiltWorkflows');
  const list     = document.getElementById('prebuiltWorkflowsList');
  const discCard = document.getElementById('workflowsDisclosureBtn');
  if (!wrap || !list) return;
  try {
    const res  = await fetchScoped('/api/presets');
    if (!res.ok) {
      wrap.style.display = 'none';
      if (discCard) discCard.style.display = 'none';
      return;
    }
    const data = await res.json();
    const items = data.presets || [];
    if (!items.length) {
      // No presets → both the disclosure card AND the (would-be)
      // expanded panel stay hidden.  This matches the pre-refactor
      // behaviour: workflows simply don't exist on this server.
      wrap.style.display = 'none';
      if (discCard) discCard.style.display = 'none';
      return;
    }
    // Presets exist — populate the panel and reveal the disclosure
    // card.  Panel itself remains hidden until the user clicks the
    // card (handled in toggleWorkflowsDisclosure).
    if (discCard) discCard.style.display = '';

    // User-built presets from the structured prompt designer (stored
    // in localStorage under paperlens.userPresets.v1).  Surfaced as
    // their own group at the top of the workflows panel so they're
    // easy to find and visually distinct from bundled presets.
    let userBlock = '';
    if (typeof window._designerLoadUserPresets === 'function') {
      const userPresets = window._designerLoadUserPresets();
      const userItems = Object.values(userPresets || {});
      if (userItems.length) {
        userBlock = `
          <div class="workflow-group-label">
            <span>Your workflows</span>
            <span class="workflow-group-hint">From the structured prompt designer (stored in this browser)</span>
          </div>
          ${userItems.map(p => `
            <div class="workflow-card-wrap">
              <button class="workflow-card option-card-load workflow-card-user"
                      onclick="applyPreset('${escHtml(p.id)}')">
                <div class="option-icon">✨</div>
                <div class="option-card-load-text">
                  <h3>${escHtml(p.title || p.id)}</h3>
                  <p>${escHtml(p.tagline || '')}</p>
                </div>
              </button>
              <button class="workflow-card-del" title="Delete this workflow"
                      onclick="event.stopPropagation(); deleteUserPreset('${escHtml(p.id)}')">
                &times;
              </button>
            </div>
          `).join('')}
          ${items.length ? `
            <div class="workflow-group-label" style="margin-top:14px">
              <span>Built-in workflows</span>
            </div>` : ''}
        `;
      }
    }

    list.innerHTML = userBlock + items.map(p => `
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
    if (discCard) discCard.style.display = 'none';
  }
}

/* Remove a user-built preset from localStorage and re-render the
 * workflows panel.  Confirms before deleting because there's no undo. */
function deleteUserPreset(presetId) {
  if (!confirm('Delete this workflow? This cannot be undone.')) return;
  try {
    const key = 'paperlens.userPresets.v1';
    const all = JSON.parse(localStorage.getItem(key) || '{}');
    delete all[presetId];
    localStorage.setItem(key, JSON.stringify(all));
  } catch (_) { /* localStorage disabled */ }
  renderInlineWorkflows();
}

/* Toggle the pre-built workflows panel from the disclosure card on the
   primary task picker.  Updates aria-expanded on the card so the
   chevron-rotation CSS picks the right state, and scrolls the
   newly-revealed panel into view smoothly on the open transition. */
function toggleWorkflowsDisclosure() {
  const card  = document.getElementById('workflowsDisclosureBtn');
  const panel = document.getElementById('prebuiltWorkflows');
  if (!card || !panel) return;
  const isOpen = card.getAttribute('aria-expanded') === 'true';
  if (isOpen) {
    panel.style.display = 'none';
    card.setAttribute('aria-expanded', 'false');
  } else {
    panel.style.display = '';
    card.setAttribute('aria-expanded', 'true');
    // Scroll the revealed panel into view so the user sees it without
    // having to scroll manually after the click.
    requestAnimationFrame(() => {
      panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
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
    // Stash the whole config payload so other scripts (e.g. donor.js) can
    // read feature flags without re-fetching.  Kept on window so the dev
    // console can poke at it too.
    window.__PAPERLENS_CONFIG__ = data;
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

  // Distinguish three failure modes:
  //  (a) Loaded from JSON, evidence present, PDF not uploaded — the
  //      bottleneck is the missing PDF, not the prompt or the model.
  //  (b) Loaded from JSON OR fresh extraction with no evidence array
  //      at all — either prompt didn't ask, or the model didn't comply.
  //  (c) Evidence array present but every entry is missing a page —
  //      the model emitted snippets but no page anchors.
  // Treating (a) as (b) ("prompt doesn't request evidence") was wrong
  // for the Review-existing-results flow: the JSON often does carry an
  // evidence array, the user just hasn't uploaded the matching PDF.
  const loadedFromFile = !!state.loadedFromFile;
  const hasPdfHere     = !!paper.blob || (paper.pageImages && paper.pageImages.length);
  if (total > 0 && loadedFromFile && !hasPdfHere) {
    body.innerHTML =
      `Page highlights aren't available — the JSON carries ${total} evidence ` +
      `snippet${total !== 1 ? 's' : ''}, but no PDF for this paper was uploaded ` +
      `alongside it.  Drag the original PDF onto the upload zone above (or click ` +
      `the "Original PDFs" tile on the previous step) to see highlights.`;
    el.style.display = 'flex';
    return;
  }

  // No evidence emitted at all — and the server-side recovery couldn't
  // help.  Either the prompt didn't ask, or the model stayed silent.
  const promptHasSchema = _hasEvidenceSchema(state.generatedPrompt);
  if (total === 0) {
    if (loadedFromFile) {
      // The JSON itself omits an evidence array — distinct from "the
      // prompt forgot to ask" because we have no prompt to blame.
      body.innerHTML =
        `Page highlights aren't available — the loaded JSON does not include ` +
        `an <code>evidence</code> array for this paper.  Results are shown on ` +
        `the left; the PDF on the right is browsable but unhighlighted.`;
    } else if (!promptHasSchema) {
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
      (loadedFromFile ? '' : `<button class="btn btn-outline btn-sm" onclick="retryPaper('${paper.id}')">Re-run</button>`);
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
    const w = document.getElementById('promptReadinessWarning');
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

/* Known confidence-category keys → display labels.  Ordered list so the
   display order is stable (loadings → correlations → effect sizes →
   reliabilities → metadata → summary sections).  Any key emitted by
   the model but absent from this list still renders — it just falls
   through to an auto-formatted label (``snake_case`` → ``Title Case``).
   Adding a known label here lets the picker show a nicer name and pins
   the display order; not strictly required. */
const _CONFIDENCE_CATEGORIES = [
  { key: 'factor_loadings',     label: 'Loadings' },
  { key: 'factor_correlations', label: 'Correlations' },
  { key: 'effect_sizes',        label: 'Effect sizes' },
  { key: 'reliabilities',       label: 'Reliabilities' },
  { key: 'studies',             label: 'Studies' },
  { key: 'metric',              label: 'Metric' },
  { key: 'background',          label: 'Background' },
  { key: 'methods',             label: 'Methods' },
  { key: 'findings',            label: 'Findings' },
  { key: 'limitations',         label: 'Limitations' },
  { key: 'metadata',            label: 'Metadata' },
];

/* Titleise a snake_case key for use as a badge label when the key
   isn't in the curated _CONFIDENCE_CATEGORIES list.  Keeps the
   renderer working for future presets that introduce new keys
   without a code change. */
function _autoLabelFromKey(key) {
  return String(key || '')
    .split(/[_\-]+/)
    .filter(Boolean)
    .map(w => w[0].toUpperCase() + w.slice(1))
    .join(' ') || 'Confidence';
}

/* Render the confidence-badge row above the parsed entry.  Reads
   ``entry.extraction_confidence`` (object with one rating per category)
   and emits one coloured pill per category.  Hides the row entirely
   when the entry has no confidence block — old runs and presets that
   don't ask for confidence stay clean.

   When ``subView`` is passed and declares ``confidence_keys``, the
   renderer ALSO filters the badges to that list — so the Effect sizes
   tab shows only effect_sizes / reliabilities ratings, the
   Descriptives tab shows only metadata, etc.  Without a sub-view (or
   without confidence_keys on it), all categories render.

   Order: known categories first (in their _CONFIDENCE_CATEGORIES
   order), then any extra keys the model emitted that we don't yet
   have a label for. */
function _renderConfidenceBadges(entry, subView) {
  const row = document.getElementById('confidenceRow');
  if (!row) return;
  const conf = entry && entry.extraction_confidence;
  if (!conf || typeof conf !== 'object') {
    row.style.display = 'none';
    row.innerHTML     = '';
    return;
  }

  const knownKeys = new Set(_CONFIDENCE_CATEGORIES.map(c => c.key));
  const order = [
    ..._CONFIDENCE_CATEGORIES.filter(c => c.key in conf),
    ...Object.keys(conf)
      .filter(k => !knownKeys.has(k))
      .map(k => ({ key: k, label: _autoLabelFromKey(k) })),
  ];

  // Per-sub-view filter — when the preset declares which confidence
  // categories belong to this tab, only those render.  Anything not in
  // the list is hidden from the badge row but stays in the underlying
  // entry data (raw view still shows everything).
  const allowedKeys = subView && Array.isArray(subView.confidence_keys)
    ? new Set(subView.confidence_keys)
    : null;

  const parts = [`<span class="confidence-row-label">Confidence</span>`];
  let hadAny = false;
  for (const cat of order) {
    if (allowedKeys && !allowedKeys.has(cat.key)) continue;
    const raw   = conf[cat.key];
    const level = _normaliseConfidence(raw);
    if (raw == null && level === 'unknown') continue;  // skip categories the model omitted
    hadAny = true;
    // Display the rating word ("high" / "medium" / "low") regardless of
    // whether the model emitted a plain string or the canonical
    // {level, notes} object shape.  Fall back to em-dash only when the
    // category exists but the level is unparseable.
    const displayValue = (typeof raw === 'string' && raw.trim())
      ? raw.trim()
      : (raw && typeof raw === 'object' && typeof raw.level === 'string' && raw.level.trim())
        ? raw.level.trim()
        : '—';
    const notes = (raw && typeof raw === 'object' && typeof raw.notes === 'string' && raw.notes.trim())
      ? raw.notes.trim() : '';
    const tip = notes
      ? `${cat.label}: ${displayValue} — ${notes}`
      : `${cat.label}: ${displayValue}`;
    parts.push(`
      <span class="confidence-badge confidence-${level}" title="${escHtml(tip)}">
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
  // String form: "high" / "medium" / "low" (or short aliases).
  // Object form: {level: "high", notes: "..."} — the spec's canonical
  // shape per the extraction prompt's EXTRACTION CONFIDENCE block.
  let s = null;
  if (typeof v === 'string') {
    s = v;
  } else if (v && typeof v === 'object' && typeof v.level === 'string') {
    s = v.level;
  }
  if (!s) return 'unknown';
  s = s.trim().toLowerCase();
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

  // Sub-view filter (preset-driven, e.g. MASEM Loadings/Correlations/Descriptives).
  // Computed FIRST so the confidence-badge renderer can scope its
  // badges to the active sub-view's declared confidence_keys.
  const subView       = _activeSubViewFor(paper);
  let   filteredEntry = subView ? _filterEntryBySubView(entry, subView) : entry;

  // Confidence badges above the rendered entry — driven by the
  // `extraction_confidence` block the model emits per sample.  When
  // the sub-view declares confidence_keys, the badge row filters to
  // those categories (so Effect sizes / Descriptives each show only
  // their relevant ratings).  Hidden entirely when the entry has no
  // confidence block.
  _renderConfidenceBadges(entry, subView);
  _renderMasemRowWarnings(entry);
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
  paper.evidencePages = evidencePages;
  // Prefer the table_caption page (so opening a TABLE entry lands on
  // the table, not on the abstract where a headline-justifying snippet
  // happens to live).  Falls back to evidencePages[0] when no
  // table_caption evidence exists for this entry.
  const initialPage = _preferredInitialPage(paper.parsed, paper.entryIndex, evidencePages)
                      ?? (paper.pageImages.length ? 1 : null);
  paper.evidencePageIdx = Math.max(0, paper.evidencePages.indexOf(initialPage));
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
  // ``focusedFields`` is the green-focus set — populated when the user
  // clicks an evidence_idx chip or a finding cell.  Focused rects
  // bypass the sub-view filter so the green outline is always visible
  // even when the matching evidence's ``field`` doesn't belong to the
  // active sub-view's ``evidence_keys``.
  const focusedFields = (paper.focusedFields instanceof Set)
    ? paper.focusedFields
    : new Set(Array.isArray(paper.focusedFields) ? paper.focusedFields
              : (paper.focusedField ? [paper.focusedField] : []));
  const matching = paper.highlights.filter(h =>
    h.page === pageNum
    && (focusedFields.has(h.field) || _highlightMatchesSubView(h, subView))
  );
  if (!matching.length) { svg.style.display = 'none'; return; }

  // The legacy single-field "focused" (teal) still fires when set
  // explicitly via paper.focusedField; the new ``focusedFields`` set
  // drives the green-focus class for chip/cell clicks.
  const teal = paper.focusedField || null;

  _withImageDims(paper, pageNum - 1, dims => {
    if (!dims) { svg.style.display = 'none'; return; }
    svg.setAttribute('viewBox', `0 0 ${dims.w} ${dims.h}`);
    svg.style.display = 'block';
    for (const h of matching) {
      const isGreen = focusedFields.has(h.field);
      const isTeal  = teal && h.field === teal && !isGreen;
      const cls = isGreen ? 'highlight-rect highlight-rect-green-focus'
                : isTeal  ? 'highlight-rect highlight-rect-focused'
                          : 'highlight-rect';
      for (const r of (h.rects || [])) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x',      r[0]);
        rect.setAttribute('y',      r[1]);
        rect.setAttribute('width',  r[2]);
        rect.setAttribute('height', r[3]);
        rect.setAttribute('class',  cls);
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = (h.field ? `[${h.field}] ` : '') + (h.snippet || '');
        rect.appendChild(title);
        svg.appendChild(rect);
      }
    }
    // Second layer: literal-value occurrences the cell-click triggered.
    // Drawn last so the green outline sits on top of the snippet rects.
    const vf = paper.valueFocusRects;
    if (vf && vf.page === pageNum && Array.isArray(vf.rects)) {
      for (const r of vf.rects) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x',      r[0]);
        rect.setAttribute('y',      r[1]);
        rect.setAttribute('width',  r[2]);
        rect.setAttribute('height', r[3]);
        rect.setAttribute('class',  'highlight-rect highlight-rect-value-focus');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `Value match: ${vf.text}`;
        rect.appendChild(title);
        svg.appendChild(rect);
      }
    }
  });
}

/* Set the green-focus field set on the active paper and re-render the
 * overlay.  ``fields`` is an array of evidence ``field`` strings.
 * Pass an empty array to clear the focus. */
function _setFocusedEvidenceFields(paper, fields) {
  paper.focusedFields = new Set(Array.isArray(fields) ? fields : []);
  const img = document.getElementById('pageDisplayImg');
  if (!img || !paper.pageImages) return;
  const currentIdx = paper.pageImages.indexOf(img.src);
  if (currentIdx >= 0) renderHighlightOverlay(paper, currentIdx + 1);
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
    // ``evidence_idx`` arrays index into the paper's top-level
    // ``evidence`` array (the AI-findings pattern: each finding lists
    // which evidence entries support it).  Render as clickable chips
    // instead of the generic numbered-list-of-editables so users can
    // jump straight to the supporting evidence.  Match on the leaf
    // segment of the path so it works whether the field sits inside
    // an entry sub-object or at the entry's top level.
    if (/(^|\.)evidence_idx$/.test(path || '')
        && data.every(v => Number.isInteger(v))) {
      return `<span class="ev-idx-chips">${data.map(n => `
        <button type="button" class="ev-idx-chip" data-evidence-idx="${n}"
                title="Jump to evidence entry #${n + 1}">${n + 1}</button>
      `).join('')}</span>`;
    }
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

  // ``evidence_idx`` chips have their own explicit lookup — each chip's
  // ``data-evidence-idx`` is a direct index into the paper's evidence
  // array, so we don't need the field-path matching that
  // ``handleCellEvidenceJump`` does.  Catches the click before the
  // generic handler so the chip's wrapping rv-editable (if any)
  // doesn't fire a parent-path jump that would land elsewhere.
  display.addEventListener('click', handleEvidenceIdxChipClick, true);

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

/* Handle a click on an ``.ev-idx-chip`` chip rendered for an
 * ``evidence_idx`` array.  The chip's ``data-evidence-idx`` is a
 * direct index into the active paper's evidence array — read the
 * referenced entry, navigate the PDF panel to its page, and flash
 * the highlight so the user sees the supporting snippet.
 */
function handleEvidenceIdxChipClick(event) {
  const chip = event.target.closest && event.target.closest('.ev-idx-chip');
  if (!chip) return;
  event.stopPropagation();
  event.preventDefault();
  const idx = parseInt(chip.dataset.evidenceIdx, 10);
  if (!Number.isFinite(idx)) return;
  const paper = getActivePaper();
  if (!paper) return;
  const evidence = (paper.parsed && Array.isArray(paper.parsed.evidence))
    ? paper.parsed.evidence : [];
  const target = evidence[idx];
  if (!target) return;

  // Inline expansion: render the snippet/source/page/field detail
  // right under the clicked chip's row.  Toggling the same chip clears
  // the expansion; clicking a different chip replaces it.
  _toggleEvidenceIdxDetail(chip, target, idx);

  // Green focus: outline this evidence's rect on its page.
  _setFocusedEvidenceFields(paper, [target.field].filter(Boolean));
  // Clear any prior value-focus rects (chip clicks are evidence-only).
  paper.valueFocusRects = null;

  const page = toPageNum(target.page);
  if (page && paper.pageImages && paper.pageImages.length) {
    showPageImage(paper, page);
  }
}

/* Render an inline expansion immediately under the row containing the
 * clicked evidence-idx chip, showing the snippet / source / page /
 * field of the referenced evidence entry.  Clicking the same chip
 * twice clears the expansion; clicking another chip replaces it. */
function _toggleEvidenceIdxDetail(chip, evidence, idx) {
  // Remove any existing detail panel anywhere in the active result view.
  const display = document.getElementById('resultDisplay');
  if (display) {
    display.querySelectorAll('.ev-idx-detail').forEach(el => el.remove());
    display.querySelectorAll('.ev-idx-chip-active').forEach(el => {
      el.classList.remove('ev-idx-chip-active');
    });
  }
  // If the user re-clicked the active chip, just clear (toggle off).
  if (chip.dataset.evidenceIdxOpen === '1') {
    delete chip.dataset.evidenceIdxOpen;
    return;
  }
  chip.classList.add('ev-idx-chip-active');
  chip.dataset.evidenceIdxOpen = '1';

  const snippet = evidence.snippet || '(no snippet)';
  const page    = evidence.page    != null ? `p. ${evidence.page}` : 'page unknown';
  const source  = evidence.source  || '';
  const field   = evidence.field   || '';

  const detail = document.createElement('div');
  detail.className = 'ev-idx-detail';
  detail.innerHTML =
    `<div class="ev-idx-detail-snippet">&ldquo;${escHtml(snippet)}&rdquo;</div>` +
    `<div class="ev-idx-detail-meta">` +
      `<span class="ev-idx-detail-page">${escHtml(page)}</span>` +
      (source ? `<span class="ev-idx-detail-source">${escHtml(source)}</span>` : '') +
      (field  ? `<span class="ev-idx-detail-field">supports <code>${escHtml(field)}</code></span>` : '') +
      `<span class="ev-idx-detail-idx">evidence[${idx}]</span>` +
    `</div>`;

  // Insert the detail panel after the chip's containing table row when
  // the chip lives inside a <td>; otherwise after the chip's nearest
  // block-level ancestor.  This keeps the panel visually grouped with
  // the finding-row it came from.
  const tr = chip.closest('tr');
  if (tr && tr.parentNode) {
    const wrapper = document.createElement('tr');
    wrapper.className = 'ev-idx-detail-row';
    const td = document.createElement('td');
    const colCount = tr.children.length || 1;
    td.colSpan = colCount;
    td.appendChild(detail);
    wrapper.appendChild(td);
    tr.parentNode.insertBefore(wrapper, tr.nextSibling);
  } else {
    const block = chip.closest('.rv-row, .rv-key-value, .rv-leaf, div');
    if (block && block.parentNode) {
      block.parentNode.insertBefore(detail, block.nextSibling);
    } else {
      chip.parentNode && chip.parentNode.appendChild(detail);
    }
  }
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

  // Collect EVERY evidence field whose path matches the clicked cell —
  // a single cell may be supported by multiple snippets (e.g. value +
  // table-caption + classification_reasoning), and we want all of them
  // outlined in green when the user clicks the cell.
  const matchingFields = _findAllEvidenceFieldsForPath(paper, path);
  _setFocusedEvidenceFields(paper, matchingFields);

  // Also outline literal occurrences of the cell's displayed value on
  // the resolved page.  Server roundtrip is fast (a single text search
  // per PDF page), and it gives precise green highlighting on the
  // actual number / token the user clicked rather than just the
  // surrounding snippet.
  const cellText = (el.textContent || '').trim();
  // Don't search for empty-cell placeholders, prose-length text, or
  // text that's obviously a stringified array/dict (rendered cells of
  // structured values).
  const isSearchable = cellText
    && cellText !== '—' && cellText !== 'null'
    && cellText.length >= 2 && cellText.length <= 80
    && !cellText.startsWith('[') && !cellText.startsWith('{');

  const page = _evidencePageForPath(paper, path);
  if (!page) {
    // No evidence-derived page — but maybe we can still surface a
    // value-match on whatever page is currently shown.
    if (isSearchable) {
      _findAndOutlineValueOnPage(paper, cellText, _currentDisplayedPage(paper));
    }
    return;
  }
  // Skip the work (and the zoom reset inside showPageImage) when we're
  // already on the right page — clicking different cells on the same page
  // shouldn't keep collapsing the user's zoom.
  const img = document.getElementById('pageDisplayImg');
  const currentSrc = img && img.src;
  const targetSrc  = paper.pageImages[page - 1];
  if (currentSrc === targetSrc) {
    // Same page but maybe new focus — re-paint the overlay so the green
    // updates without the showPageImage zoom-reset.
    renderHighlightOverlay(paper, page);
  } else {
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

  // Fire the value-text search asynchronously — overlay re-paints when
  // the rects come back so the value gets outlined without blocking
  // the page-jump above.
  if (isSearchable) _findAndOutlineValueOnPage(paper, cellText, page);
}

/* Return the 1-indexed page currently shown in the PDF viewer, or
 * null if no page is shown.  Used when the click handler wants to
 * search the visible page for the cell value even though no
 * evidence-derived page exists. */
function _currentDisplayedPage(paper) {
  const img = document.getElementById('pageDisplayImg');
  if (!img || !paper || !Array.isArray(paper.pageImages)) return null;
  const idx = paper.pageImages.indexOf(img.src);
  return idx >= 0 ? (idx + 1) : null;
}

/* Post (PDF, page, [text]) to /api/find-text, stash the returned rects
 * on the paper, and re-paint the overlay.  The rects are attached under
 * paper.valueFocusRects so renderHighlightOverlay can draw them as a
 * second green layer on top of the evidence highlights. */
async function _findAndOutlineValueOnPage(paper, text, page) {
  if (!paper || !paper.pdfFile || !text || !page) return;
  const form = new FormData();
  form.append('pdf',   paper.pdfFile, paper.pdfFile.name);
  form.append('page',  String(page));
  form.append('texts', text);
  try {
    const res  = await fetchScoped('/api/find-text', {method: 'POST', body: form});
    if (!res.ok) return;
    const data = await res.json();
    const rectsByText = (data && data.rects) || {};
    const rects = rectsByText[text] || [];
    paper.valueFocusRects = rects.length
      ? {page: data.page || page, text, rects}
      : null;
    // Re-paint only when we're still on the same page (the user may have
    // navigated away during the round-trip).
    const shownPage = _currentDisplayedPage(paper);
    if (shownPage === (data.page || page)) {
      renderHighlightOverlay(paper, shownPage);
    }
  } catch (_) { /* network failure — silently skip the value-outline layer */ }
}

/* Return every evidence ``field`` whose path matches the clicked cell.
 * Matches via exact equality OR longest-prefix on the path candidates
 * (samples-rooted vs sample-relative).  Used to drive the green-focus
 * overlay set for cell clicks. */
function _findAllEvidenceFieldsForPath(paper, path) {
  if (!paper || !paper.parsed || !path) return [];
  const evidence = paper.parsed.evidence;
  if (!Array.isArray(evidence) || !evidence.length) return [];
  const idx = (paper.entryIndex == null) ? 0 : paper.entryIndex;
  const candidates = [path, `samples[${idx}].${path}`];
  const fields = new Set();
  // Exact matches first — the precise cell-level evidence we want.
  for (const e of evidence) {
    if (!e || !e.field) continue;
    if (candidates.includes(e.field)) fields.add(e.field);
  }
  if (fields.size) return [...fields];
  // No exact match — fall back to longest-prefix so a click on a leaf
  // still surfaces the closest enclosing evidence (e.g. clicking
  // ``classification_reasoning`` falls back to ``samples[N]`` parent).
  let best = null, bestLen = -1;
  for (const e of evidence) {
    if (!e || !e.field) continue;
    for (const cand of candidates) {
      if (cand === e.field || cand.startsWith(e.field + '.')) {
        if (e.field.length > bestLen) { best = e.field; bestLen = e.field.length; }
      }
    }
  }
  return best ? [best] : [];
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

  // After a successful download, offer the donation flow.  The dataset
  // stored to GitHub is always the structured JSON (donor.py reads from
  // the DB, not the downloaded file) regardless of which format the user
  // grabbed for their own use here.
  if (typeof window.donorMaybeOffer === 'function') {
    window.donorMaybeOffer(state.batchId);
  }
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

  // After a successful download, offer the donation flow (no-op when the
  // feature flag is off or this batch was already donated).  donor.js
  // handles the offer/toast/modal lifecycle from here.
  if (typeof window.donorMaybeOffer === 'function') {
    window.donorMaybeOffer(state.batchId);
  }
}

/* ──────────────────────────────────────────────────────────
   Load results from file (step 1 → step 8 shortcut)
────────────────────────────────────────────────────────── */

// Staged data for the review flow
let reviewPdfFiles  = [];   // File[]
// One entry per accepted JSON file: {name, prompt, model, papers[]}.  Several
// files stage at once so a batch that was exported paper-by-paper (or split
// across runs) can be reviewed in a single session.
let reviewJsonFiles = [];
let reviewJsonData  = null; // merged envelope, held until user clicks "Load"

// Array-shaped keys that mark an object as a single paper's extraction.
// Shared with _canonicaliseLoadedPaper so the staging check and the loader
// agree on what counts as a paper.
const _LOADED_ENTRY_ARRAY_KEYS = ['entries', 'findings', 'samples',
                                  'regressions', 'records'];

function selectLoadOption() {
  document.getElementById('modeGrid').style.display        = 'none';
  document.getElementById('jsonUploadPanel').style.display = '';
  showJsonStage1();
}

/* ──────────────────────────────────────────────────────────
   Step 1 — "Extend an existing dataset" picker (Phase 3b)
────────────────────────────────────────────────────────── */

/* All loaded datasets, kept on window so the dev console can poke at
   them.  The search filter operates over this in-memory list rather
   than re-fetching on every keystroke. */
window.__EXTEND_DATASETS__ = [];

function selectExtendOption() {
  const overlay = document.getElementById('extendDatasetOverlay');
  if (!overlay) return;
  overlay.style.display = 'flex';
  document.getElementById('extendDatasetSearch').value = '';
  _showExtendState('loading');
  _loadExtendDatasets();
  // Focus the search box after a tick so the modal-overlay animation
  // doesn't steal focus back.
  setTimeout(() => document.getElementById('extendDatasetSearch').focus(), 60);
}

function closeExtendDatasetModal() {
  const overlay = document.getElementById('extendDatasetOverlay');
  if (overlay) overlay.style.display = 'none';
}

/* Mutually-exclusive view-state switching for the modal body —
   'loading' / 'empty' / 'error' / 'list'.  Keeps the markup simple
   and prevents the empty-state message flashing while the fetch
   is in flight. */
function _showExtendState(state, errMsg) {
  const list  = document.getElementById('extendDatasetList');
  const load  = document.getElementById('extendDatasetLoading');
  const empty = document.getElementById('extendDatasetEmpty');
  const err   = document.getElementById('extendDatasetError');
  list.style.display  = state === 'list'    ? '' : 'none';
  load.style.display  = state === 'loading' ? '' : 'none';
  empty.style.display = state === 'empty'   ? '' : 'none';
  err.style.display   = state === 'error'   ? '' : 'none';
  if (state === 'error') err.textContent = errMsg || 'Could not load datasets.';
}

async function _loadExtendDatasets(forceRefresh) {
  try {
    const url = forceRefresh ? '/api/datasets?refresh=1' : '/api/datasets';
    const res = await fetch(url);
    if (!res.ok) {
      // 503 (repo unconfigured) → useful hint; other 5xx → generic message.
      const body = await res.json().catch(() => ({}));
      _showExtendState('error',
        res.status === 503
          ? 'Dataset hosting is not configured on this server.'
          : (body.detail || `Server returned ${res.status}.`));
      return;
    }
    const data = await res.json();
    window.__EXTEND_DATASETS__ = Array.isArray(data.datasets) ? data.datasets : [];
    _updateExtendCacheNote(data);
    if (!window.__EXTEND_DATASETS__.length) {
      _showExtendState('empty');
      return;
    }
    _renderExtendDatasetList(window.__EXTEND_DATASETS__);
    _showExtendState('list');
  } catch (err) {
    _showExtendState('error', err.message || 'Network error.');
  }
}

/* Click handler for the Refresh button in the picker.  Shows the
   loading state while the bypass-cache fetch runs so users get
   feedback that something's happening (the network call is fast on
   most networks but the App-token JWT signing + GitHub round-trip
   can take ~1s on first hit). */
async function _refreshExtendDatasets() {
  const btn = document.getElementById('extendDatasetRefreshBtn');
  if (btn) { btn.disabled = true; btn.style.opacity = '0.7'; }
  _showExtendState('loading');
  try {
    await _loadExtendDatasets(/* forceRefresh */ true);
  } finally {
    if (btn) { btn.disabled = false; btn.style.opacity = ''; }
  }
}

/* "Last refreshed N min ago" indicator beneath the search box.
   Hidden when the payload is fresh (<10s) so the most common case
   stays uncluttered.  Mostly there to make stale data discoverable
   when a recently-merged donation hasn't shown up yet. */
function _updateExtendCacheNote(payload) {
  const note = document.getElementById('extendDatasetCacheNote');
  if (!note) return;
  const ageSec = payload && typeof payload.cache_age_sec === 'number' ? payload.cache_age_sec : 0;
  if (ageSec < 10) {
    note.style.display = 'none';
    return;
  }
  let label;
  if (ageSec < 60)         label = `${ageSec}s ago`;
  else if (ageSec < 3600)  label = `${Math.floor(ageSec / 60)} min ago`;
  else                     label = `${Math.floor(ageSec / 3600)}h ago`;
  note.textContent = `Listing last refreshed ${label}.  Don't see a recently-merged dataset? Click Refresh.`;
  note.style.display = '';
}

/* Render one row per dataset.  Click → Phase 3c (password prompt for
   gated datasets) → Phase 3d (preset pre-load).  Until 3c lands the
   click handler is a stub that toasts an "implementation coming"
   message — keeps the modal navigable while we're mid-build. */
function _renderExtendDatasetList(datasets) {
  const list = document.getElementById('extendDatasetList');
  list.innerHTML = datasets.map((d, i) => {
    const lock = d.gated ? '🔒' : '📊';
    const donor = (d.donor && d.donor.mode === 'attributed' && d.donor.name)
      ? escHtml(d.donor.name) + (d.donor.affiliation ? ` <span style="color:var(--text-muted)">(${escHtml(d.donor.affiliation)})</span>` : '')
      : '<span style="color:var(--text-muted)">Anonymous</span>';
    const created = d.created_at ? _relativeTime(d.created_at) : '';
    // Hide the schema badge when we don't have a confident value to
    // show — "mixed" (conflict) and "unspecified" (model didn't emit
    // the field) both signal "we don't know", which adds noise rather
    // than information when surfaced as a badge.  The internal value
    // still lives in the dataset object for Phase 3g's schema-match
    // enforcement.
    const showBadge = d.schema_version
      && d.schema_version !== 'mixed'
      && d.schema_version !== 'unspecified';
    const schema = showBadge ? `<span class="extend-dataset-badge">${escHtml(d.schema_version)}</span>` : '';
    const desc    = d.description ? `<div class="extend-dataset-desc">${escHtml(d.description)}</div>` : '';
    return `
      <button type="button" class="extend-dataset-row" data-idx="${i}"
              onclick="_extendDatasetPicked(${i})">
        <div class="extend-dataset-row-head">
          <span class="extend-dataset-icon" aria-hidden="true">${lock}</span>
          <span class="extend-dataset-title">${escHtml(d.title || d.dataset_id || '(untitled)')}</span>
          ${schema}
        </div>
        <div class="extend-dataset-meta">
          ${donor}
          <span class="extend-dataset-dot">·</span>
          ${d.paper_count != null ? `${d.paper_count} paper${d.paper_count === 1 ? '' : 's'}` : ''}
          ${created ? `<span class="extend-dataset-dot">·</span>${escHtml(created)}` : ''}
        </div>
        ${desc}
      </button>`;
  }).join('');
}

function _filterExtendDatasets() {
  const q = document.getElementById('extendDatasetSearch').value.trim().toLowerCase();
  const all = window.__EXTEND_DATASETS__ || [];
  if (!q) {
    if (!all.length) { _showExtendState('empty'); return; }
    _renderExtendDatasetList(all);
    _showExtendState('list');
    return;
  }
  const filtered = all.filter(d => {
    const hay = [
      d.title, d.description, d.schema_version, d.dataset_id,
      d.donor && d.donor.name, d.donor && d.donor.affiliation,
    ].filter(Boolean).join(' ').toLowerCase();
    return hay.includes(q);
  });
  if (!filtered.length) {
    _showExtendState('error', `No datasets match "${q}".`);
    return;
  }
  _renderExtendDatasetList(filtered);
  _showExtendState('list');
}

/* Handle a click on a dataset row.  For public datasets, jump
   straight to the (still stubbed) "selected" flow.  For gated
   datasets, swap the row inline for a password input + verify
   button — on a correct password, proceed to the same selected flow.

   The actual "what happens after a dataset is selected" (preset +
   schema pre-load, then user runs an extraction that submits as an
   extension) lands in Phase 3d/3e.  Until then, we toast the
   dataset's title so testers can confirm the gate works. */
function _extendDatasetPicked(idx) {
  const d = (window.__EXTEND_DATASETS__ || [])[idx];
  if (!d) return;
  if (d.gated) {
    _showPasswordPromptInRow(idx, d);
    return;
  }
  _afterDatasetVerified(d);
}

/* Replace the body of a dataset row with an inline password prompt.
   Keeps context (the user still sees which dataset they're unlocking)
   without yanking them into a separate modal.  Cancel restores the
   normal row content; Verify hits /api/datasets/{id}/verify-password
   and either advances or shows the failure inline. */
function _showPasswordPromptInRow(idx, dataset) {
  const row = document.querySelector(`.extend-dataset-row[data-idx="${idx}"]`);
  if (!row) return;
  // Swap the button-element for a non-clickable shell so clicks on
  // the password field don't toggle the row.  Replace with a div of
  // the same class for layout continuity.
  const shell = document.createElement('div');
  shell.className = row.className + ' extend-dataset-row-prompt';
  shell.setAttribute('data-idx', String(idx));
  shell.innerHTML = `
    <div class="extend-dataset-row-head">
      <span class="extend-dataset-icon" aria-hidden="true">🔒</span>
      <span class="extend-dataset-title">${escHtml(dataset.title || dataset.dataset_id)}</span>
    </div>
    <p class="extend-dataset-prompt-help">
      This dataset is gated.  Enter the donor's extension password to add new papers.
      <br><span style="font-size:11.5px;color:var(--text-muted)">Forgot the password? Contact the dataset's donor or repo maintainer to recover.</span>
    </p>
    <div class="extend-dataset-prompt-row">
      <input type="password" id="extendDatasetPwd-${idx}" placeholder="Extension password"
             autocomplete="off" autofocus
             style="flex:1;padding:8px 10px;border:1.5px solid var(--border);border-radius:6px;font:inherit" />
      <button class="btn btn-primary btn-sm" type="button"
              onclick="_verifyExtendPassword(${idx})">Verify</button>
      <button class="btn btn-ghost btn-sm" type="button"
              onclick="_cancelExtendPasswordPrompt(${idx})">Cancel</button>
    </div>
    <p class="extend-dataset-prompt-error" id="extendDatasetPwdErr-${idx}" style="display:none;color:#b91c1c;font-size:12.5px;margin:0"></p>
  `;
  row.replaceWith(shell);
  // Submit on Enter for keyboard users
  const input = document.getElementById(`extendDatasetPwd-${idx}`);
  if (input) {
    input.focus();
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        _verifyExtendPassword(idx);
      }
    });
  }
}

function _cancelExtendPasswordPrompt(idx) {
  // Re-render the list so the row reverts to its normal state.  Uses
  // the current filtered set if a search is active; otherwise the
  // full list.
  const q = document.getElementById('extendDatasetSearch').value.trim().toLowerCase();
  if (q) {
    _filterExtendDatasets();
  } else {
    _renderExtendDatasetList(window.__EXTEND_DATASETS__ || []);
    _showExtendState('list');
  }
}

async function _verifyExtendPassword(idx) {
  const d   = (window.__EXTEND_DATASETS__ || [])[idx];
  const pwd = (document.getElementById(`extendDatasetPwd-${idx}`) || {}).value || '';
  const err = document.getElementById(`extendDatasetPwdErr-${idx}`);
  if (!d || !pwd) {
    if (err) { err.textContent = 'Enter a password to continue.'; err.style.display = ''; }
    return;
  }
  err.style.display = 'none';
  try {
    const res  = await fetch(
      `/api/datasets/${encodeURIComponent(d.dataset_id || d.id)}/verify-password`,
      {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body:    JSON.stringify({password: pwd}),
      },
    );
    const body = await res.json().catch(() => ({}));
    if (res.status === 429) {
      err.textContent = body.detail || 'Too many attempts — try again later.';
      err.style.display = '';
      return;
    }
    if (!res.ok) {
      err.textContent = body.detail || `Server returned ${res.status}.`;
      err.style.display = '';
      return;
    }
    if (body.ok) {
      _afterDatasetVerified(d);
    } else {
      err.textContent = 'Incorrect password.';
      err.style.display = '';
    }
  } catch (e) {
    err.textContent = e.message || 'Network error.';
    err.style.display = '';
  }
}

/* Common entry point after a dataset is "unlocked" — public datasets
   skip straight here; gated ones land here once the verify-password
   step returns ok=true.  Phase 3d responsibility:
     1. Fetch /api/datasets/{id}/full to get the original prompt + the
        model_used list.
     2. Pre-load state.generatedPrompt so step-5 review surfaces the
        original prompt verbatim.
     3. Pick the closest available model for the user's current
        provider (or switch provider if the model is uniquely on
        another).
     4. Stash state.extendingFrom = {dataset_id, schema_version} so
        Phase 3e's POST /api/donate knows to submit as an extension.
     5. Navigate to step 2 (model picker) with the suggested model
        pre-selected — the user confirms API key + model, then jumps
        straight to upload (Phase 3e wires the rest). */
async function _afterDatasetVerified(dataset) {
  closeExtendDatasetModal();
  // Fetch full payload — needs the prompt body, which the listing
  // endpoint doesn't carry.
  let full;
  try {
    const res = await fetch(`/api/datasets/${encodeURIComponent(dataset.dataset_id || dataset.id)}/full`);
    if (!res.ok) {
      showToast(`Couldn't load that dataset (${res.status}).`); return;
    }
    full = await res.json();
  } catch (err) {
    showToast(err.message || 'Network error loading dataset.'); return;
  }

  // Pre-load the prompt verbatim.  ``inputMode = 'manual'`` keeps the
  // step-3 flow on the manual-prompt branch so the AI-generate step
  // is skipped — we already have the canonical prompt, which IS the
  // schema (transitively).  state.mode is set to 'extraction' for
  // every extension regardless of the original mode: the variable is
  // only used by the AI-generate flow + cosmetic labels, both
  // irrelevant here because the prompt body fully determines what
  // the model is asked to do.
  if (full.prompt) {
    state.generatedPrompt = full.prompt;
    state.inputMode       = 'manual';
  }
  state.mode = 'extraction';

  // Provider + model preselect.  Look up the first model in the
  // dataset's model_used list against PROVIDER_MODELS to find which
  // provider serves it.  If found, set both; otherwise leave the
  // current provider intact so the user can pick anything that works.
  const wantedModel = (full.model_used && full.model_used[0]) || '';
  const pick = _findProviderForModel(wantedModel);
  if (pick) {
    state.provider = pick.provider;
    state.model    = pick.model;
  }

  // Stash extension intent so the donation step (Phase 3e) can route
  // as an extension instead of a fresh dataset submission.
  // ``prompt_sha256`` is the canonical consistency key for Phase 3g:
  // an extension whose prompt hashes to the same value is by
  // construction schema-compatible, no string-label comparison
  // needed.  schema_version is kept for human-readable display in
  // the donate-success block but not used for enforcement.
  const extraction = (full.extraction && typeof full.extraction === 'object') ? full.extraction : {};
  state.extendingFrom = {
    dataset_id:     full.dataset_id,
    title:          full.title,
    schema_version: full.schema_version,
    prompt_sha256:  extraction.prompt_sha256 || null,
    github_url:     full.github_url,
  };

  // Send the user to step 2 with the pre-loaded values reflected.
  // Pre-fill the provider dropdown + model field so onProviderChange
  // sees the new selection on render.
  const providerSelect = document.getElementById('providerSelect');
  if (providerSelect && state.provider) {
    providerSelect.value = state.provider;
    onProviderChange();
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect && state.model) modelSelect.value = state.model;
  }
  showToast(
    `Loaded "${full.title || full.dataset_id}" — the dataset's prompt + suggested model are pre-set. Confirm your API key to continue.`,
    'success',
  );
  goTo(2);
}

/* Walk PROVIDER_MODELS to find which provider serves a model alias.
   Exact-match first (the common case), then a prefix match for dated
   snapshots (e.g. ``gpt-5-2025-09-15`` would map to ``gpt-5`` on
   openai).  Returns {provider, model} or null when nothing matches. */
function _findProviderForModel(modelAlias) {
  if (!modelAlias) return null;
  const alias = String(modelAlias).toLowerCase();
  for (const [provider, models] of Object.entries(PROVIDER_MODELS || {})) {
    for (const m of (models || [])) {
      if ((m.value || '').toLowerCase() === alias) {
        return { provider, model: m.value };
      }
    }
  }
  // Prefix match — handles dated snapshots that the user's listing
  // doesn't carry verbatim.
  for (const [provider, models] of Object.entries(PROVIDER_MODELS || {})) {
    for (const m of (models || [])) {
      const v = (m.value || '').toLowerCase();
      if (v && alias.startsWith(v)) {
        return { provider, model: m.value };
      }
    }
  }
  return null;
}

/* Tiny relative-time formatter — "yesterday" / "3 days ago" / "Jan 15".
   No external dep needed for one display location. */
function _relativeTime(iso) {
  const then = new Date(iso).getTime();
  if (!isFinite(then)) return '';
  const diffSec = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (diffSec < 60)                return 'just now';
  if (diffSec < 3600)              return `${Math.floor(diffSec / 60)} min ago`;
  if (diffSec < 86400)             return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 86400 * 2)         return 'yesterday';
  if (diffSec < 86400 * 30)        return `${Math.floor(diffSec / 86400)} days ago`;
  // Older — fall back to a month/day label for compactness.
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function cancelLoadOption() {
  document.getElementById('modeGrid').style.display        = '';
  document.getElementById('jsonUploadPanel').style.display = 'none';
  const errEl = document.getElementById('jsonError');
  if (errEl) errEl.style.display = 'none';
  reviewPdfFiles = [];
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
  reviewPdfFiles  = [];
  reviewJsonFiles = [];
  const hint = document.getElementById('reviewPdfHint');
  if (hint) hint.textContent = 'or drop here';
  const errEl = document.getElementById('jsonError');
  if (errEl) errEl.style.display = 'none';
  const input = document.getElementById('jsonInput');
  if (input) input.value = '';
  _refreshJsonReadyState();
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
    const jsons = files.filter(f => f.name.toLowerCase().endsWith('.json'));
    if (jsons.length) addJsonFiles(jsons);
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
  const files = [...event.target.files];
  // Clear the input so re-picking the same file still fires `change`.
  event.target.value = '';
  if (files.length) addJsonFiles(files);
}

/* Drop every staged JSON — the files accumulate, so this is the only way
   back to an empty slate without leaving the step. */
function clearReviewJson() {
  reviewJsonFiles = [];
  const input = document.getElementById('jsonInput');
  if (input) input.value = '';
  const errEl = document.getElementById('jsonError');
  if (errEl) errEl.style.display = 'none';
  _refreshJsonReadyState();
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

/* Wrap one file's contents in the {prompt, model, papers} envelope the
   loader works in.  Three inputs are accepted:

     - the consolidated "Download all → JSON"  → {prompt, model, papers: [...]}
     - a per-paper "Download JSON"             → a bare paper object, no wrapper
     - a bare array of paper objects

   Per-paper detection is deliberately loose — anything carrying an
   entries-shaped array (at the top level or under a ``result`` wrapper) or a
   raw model response is a paper, which is the same range of shapes
   _canonicaliseLoadedPaper already unpacks downstream.  Returns null when
   nothing paper-like is recognisable.  Pure function. */
function _normalizeResultsJson(data) {
  if (Array.isArray(data)) data = { papers: data };
  if (!data || typeof data !== 'object') return null;

  if (Array.isArray(data.papers)) {
    if (!data.papers.length) return null;
    return { prompt: data.prompt || '', model: data.model || '', papers: data.papers };
  }

  const result = (data.result && typeof data.result === 'object') ? data.result : null;
  const hasEntryArray = _LOADED_ENTRY_ARRAY_KEYS.some(
    k => Array.isArray(data[k]) || (result && Array.isArray(result[k])));
  const hasRawResponse = typeof data.llm_raw_response === 'string' ||
                         typeof data.original_model_response === 'string';
  if (hasEntryArray || hasRawResponse) {
    return { prompt: data.prompt || '', model: data.model || '', papers: [data] };
  }
  return null;
}

/* Fold every staged file into one envelope.  Papers are deduplicated by
   filename (first file staged wins) so dropping a per-paper export next to
   the consolidated one doesn't list the same paper twice; papers with no
   filename are always kept.  ``prompt`` and ``model`` come from the first
   file carrying them — per-paper exports have no prompt, so a mixed drop
   still recovers it. */
function _mergedReviewJson() {
  const papers = [];
  const seen   = new Set();
  let prompt = '', model = '';
  for (const f of reviewJsonFiles) {
    if (!prompt && f.prompt) prompt = f.prompt;
    if (!model  && f.model)  model  = f.model;
    for (const paper of f.papers) {
      const key = (paper && paper.filename || '').toLowerCase();
      if (key) {
        if (seen.has(key)) continue;
        seen.add(key);
      }
      papers.push(paper);
    }
  }
  return { prompt, model, papers };
}

/* Re-derive reviewJsonData + the stage-2 chrome from the staged files. */
function _refreshJsonReadyState() {
  const merged = _mergedReviewJson();
  reviewJsonData = merged.papers.length ? merged : null;

  const nFiles  = reviewJsonFiles.length;
  const nPapers = merged.papers.length;
  const plural  = (n, word) => `${n} ${word}${n !== 1 ? 's' : ''}`;

  const jsonHint = document.getElementById('jsonDropHint');
  if (jsonHint) {
    jsonHint.textContent = nFiles
      ? `\u2713 ${plural(nFiles, 'file')} \u00b7 ${plural(nPapers, 'paper')}`
      : 'or drop here';
  }

  const ready = document.getElementById('jsonReadyRow');
  if (ready) ready.style.display = nPapers ? 'flex' : 'none';

  const readyText = document.getElementById('jsonReadyText');
  if (readyText && nPapers) {
    readyText.textContent =
      `${plural(nPapers, 'paper')} from ${plural(nFiles, 'file')} ready. ` +
      'Add PDFs above if you want page previews, then click Load.';
  }

  const loadBtn = document.getElementById('jsonLoadBtn');
  if (loadBtn) loadBtn.disabled = !nPapers;
}

/* Step 1: parse and validate — does NOT navigate.  Files accumulate, so a
   folder of per-paper exports can be dropped in several goes.  A bad file is
   reported by name and skipped; the good ones still stage. */
async function addJsonFiles(files) {
  const errEl = document.getElementById('jsonError');
  if (errEl) errEl.style.display = 'none';

  const errors = [];
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith('.json')) {
      errors.push(`${file.name}: not a .json file`);
      continue;
    }
    let data;
    try {
      data = JSON.parse(await file.text());
    } catch (e) {
      errors.push(`${file.name}: could not parse (${e.message})`);
      continue;
    }
    const norm = _normalizeResultsJson(data);
    if (!norm) {
      errors.push(`${file.name}: not a results file — expected a "papers" array or a single-paper export`);
      continue;
    }
    // Re-dropping the same file replaces the earlier copy rather than
    // stacking a second one.
    const i = reviewJsonFiles.findIndex(f => f.name === file.name);
    const entry = { name: file.name, ...norm };
    if (i >= 0) reviewJsonFiles[i] = entry; else reviewJsonFiles.push(entry);
  }

  if (errors.length) showJsonError(errors.join(' · '));
  _refreshJsonReadyState();
}

/* Normalise an evidence entry's ``field`` path to the canonical
 * ``samples[N]....`` form that the renderer + click-to-jump both
 * expect.  Upstream pipelines emit ``findings[N]....`` (AI dataset),
 * ``regressions[N]....`` (econ), or ``records[N]....`` (effect-size
 * shape).  Only the leading top-level segment is rewritten; sub-
 * paths after the first ``.`` stay untouched.  Returns the same
 * entry unchanged when no rewrite is needed.  Pure function.
 */
const _EVIDENCE_PATH_REWRITES = [
  [/^findings\[/,    'samples['],
  [/^regressions\[/, 'samples['],
  [/^records\[/,     'samples['],
];
function _canonicaliseEvidenceField(e) {
  if (!e || typeof e !== 'object' || typeof e.field !== 'string') return e;
  let f = e.field;
  for (const [re, rep] of _EVIDENCE_PATH_REWRITES) {
    if (re.test(f)) { f = f.replace(re, rep); break; }
  }
  return f === e.field ? e : Object.assign({}, e, {field: f});
}

/* Detect which bundled preset best matches a loaded JSON's entry
 * shape and activate it.  Schema signatures (first match wins):
 *
 *   - ai-findings:    entries have ``finding_type`` AND ``subtopic``
 *   - econ-headline:  entries have ``regression_id`` OR (``table`` AND
 *                     ``estimates`` array)
 *   - masem-ncs18:    entries have ``factor_loadings``
 *   - masem:          entries have ``records`` array (effect-size shape)
 *
 * Skipped when state.activePreset is already set (the user picked one
 * deliberately) or when no entries look detectable.  Fetches the
 * preset from /api/presets/<id> so sub_views are populated; falls
 * back silently if the server can't be reached. */
async function _autoActivatePresetForLoadedData(papers) {
  if (state.activePreset && state.activePreset.id) return;
  const sample = (papers || []).find(p =>
    Array.isArray(p.entries) && p.entries.length > 0
  );
  if (!sample) return;
  const e0 = sample.entries[0] || {};
  let presetId = null;
  if (e0.finding_type !== undefined && e0.subtopic !== undefined) {
    presetId = 'ai-findings';
  } else if (e0.regression_id !== undefined
             || (e0.table !== undefined && Array.isArray(e0.estimates))) {
    presetId = 'econ-headline';
  } else if (e0.factor_loadings !== undefined) {
    presetId = 'masem-ncs18';
  } else if (Array.isArray(e0.records)) {
    presetId = 'masem';
  }
  if (!presetId) return;
  try {
    const r = await fetchScoped(`/api/presets/${encodeURIComponent(presetId)}`);
    if (!r.ok) return;
    const preset = await r.json();
    if (preset && preset.id) {
      state.activePreset = preset;
      if (document && document.body) document.body.dataset.preset = preset.id;
      showToast(`Activated preset: ${preset.title || preset.id}`, 'success');
      if (typeof renderPaperSidebar === 'function') renderPaperSidebar();
      const active = state.papers.find(p => p.id === state.activePaperId);
      if (active && typeof displayPaper === 'function') displayPaper(active);
    }
  } catch (_) { /* network failure — leave preset unset */ }
}

/* Group flat per-regression entries into per-table entries + a paper-
 * metadata entry, so the sidebar reads as
 *   [Paper metadata, TABLE 9, TABLE 10, ...]
 * instead of one row per regression cell.  Each per-table entry carries
 * the regressions as a ``regressions: {_table: [...]}`` block — the
 * renderer's existing ``_table`` marker turns that into an HTML table
 * (rows = property, columns = regression-column).
 *
 * Triggers only when every entry has ``regression_id`` + ``table`` —
 * the signature of the user's flat per-regression export.  Other shapes
 * pass through untouched.
 */
function _groupPerRegressionByTable(p, entries) {
  const isPerRegression = Array.isArray(entries) && entries.length > 0
    && entries.every(e => e && typeof e === 'object'
                          && typeof e.regression_id === 'string'
                          && typeof e.table === 'string');
  if (!isPerRegression) return {entries, indexMap: null};

  const grouped = [];
  let nextIdx = 0;

  // ── Paper-metadata entry (first) ─────────────────────────────────
  const META_FIELDS = ['title', 'doi', 'year', 'authors', 'id'];
  const meta = {};
  for (const f of META_FIELDS) {
    if (p[f] !== undefined && p[f] !== null) meta[f] = p[f];
  }
  if (Object.keys(meta).length) {
    meta.sample_id = 'Paper metadata';
    grouped.push(meta);
    nextIdx++;
  }

  // ── Group regressions by table (preserves insertion order) ──────
  // Also build a {tableName → newEntryIndex} map so we can rewrite
  // evidence's ``samples[N]`` indices to point at the grouped entry.
  const byTable = new Map();
  const tableToNewIdx = new Map();
  for (const e of entries) {
    const key = e.table;
    if (!byTable.has(key)) {
      byTable.set(key, []);
      tableToNewIdx.set(key, nextIdx++);
    }
    byTable.get(key).push(e);
  }
  // Per-old-index → new-index lookup for evidence rewriting.
  const indexMap = {};
  entries.forEach((e, oldIdx) => {
    indexMap[oldIdx] = tableToNewIdx.get(e.table);
  });

  // ── Build one entry per table with a _table-marked render block ─
  for (const [tableName, regs] of byTable) {
    const colLabel = (r) => r.regression_id;   // guaranteed unique
    const fmt      = (n) => (n == null ? null
                             : Number.isInteger(n) ? n
                             : Math.abs(n) >= 0.001 ? +n.toFixed(3) : n);

    // Treatment variable names — union across regressions in the group
    const treatNames = [];
    const seenT = new Set();
    for (const r of regs) for (const t of (r.treatment || [])) {
      const name = t.display || t.stata || '?';
      if (!seenT.has(name)) { seenT.add(name); treatNames.push(name); }
    }

    const mkRow = (label, valFn) => {
      const row = {' ': label};
      for (const r of regs) row[colLabel(r)] = valFn(r);
      return row;
    };

    const rows = [];
    rows.push(mkRow('Column', r => r.column || null));
    rows.push(mkRow('Dependent var', r =>
      r.dependent_var ? (r.dependent_var.display || r.dependent_var.stata) : null
    ));
    for (const tName of treatNames) {
      rows.push(mkRow(tName, r => {
        const m = (r.treatment || []).find(t => (t.display || t.stata) === tName);
        return m ? fmt(m.estimate) : null;
      }));
      rows.push(mkRow('  (SE)', r => {
        const m = (r.treatment || []).find(t => (t.display || t.stata) === tName);
        if (!m || m.standard_error == null) return null;
        return `(${(+m.standard_error).toFixed(3)})`;
      }));
    }
    rows.push(mkRow('N obs', r => r.n_obs_captured ?? null));
    rows.push(mkRow('Cluster', r => r.cluster ?? null));
    rows.push(mkRow('Fixed effects', r =>
      Array.isArray(r.fixed_effects) ? r.fixed_effects.join(', ')
                                     : (r.fixed_effects ?? null)
    ));
    rows.push(mkRow('Panel', r => r.panel ?? null));
    rows.push(mkRow('Spec ID', r => r.captured_spec_id ?? null));
    rows.push(mkRow('Decision', r => r.decision_final ?? null));

    grouped.push({
      sample_id:      tableName,
      table:          tableName,
      n_regressions:  regs.length,
      regressions:    {_table: rows},
    });
  }

  return {entries: grouped, indexMap};
}


/* Prepend a synthesised "Paper metadata" entry to AI-findings entries,
 * combining paper-level fields (paper_metadata + subtopics + one_line +
 * qual_notes) into one sidebar row so the reviewer can audit the
 * paper-level context separately from each individual finding.
 *
 * Triggers when every entry looks like an AI-findings record (carries
 * ``finding_type`` OR ``subtopic`` OR ``metric``+``value``) AND the
 * paper has at least one paper-level metadata field.  Other shapes
 * pass through untouched.
 *
 * Also returns an ``indexMap`` of {old-entry-index → new-entry-index}
 * so the caller can shift evidence ``samples[N]`` indices by +1 to
 * account for the synthesised entry at index 0.  Paper-level evidence
 * (``paper_metadata.*`` / ``subtopics.*`` / ``one_line`` / ``qual_notes``)
 * is rewritten in a separate helper since it lacks the ``samples[N]``
 * prefix entirely.
 */
function _synthesizeAiFindingsMetadataEntry(p, entries) {
  const isAiFindings = Array.isArray(entries) && entries.length > 0
    && entries.every(e => e && typeof e === 'object'
                          && ('finding_type' in e || 'subtopic' in e
                              || ('metric' in e && 'value' in e)));
  if (!isAiFindings) return {entries, indexMap: null};

  // Paper-level fields may sit at the top of the paper object OR under
  // a ``result`` wrapper (the user's prompt emits {id, result: {...}}).
  // Read each field with the wrapper as fallback so the synthesis works
  // for both shapes.
  const r = (p && typeof p === 'object' && p.result && typeof p.result === 'object')
    ? p.result : null;
  const pick = (k) => (p[k] !== undefined ? p[k] : (r ? r[k] : undefined));

  const meta = {sample_id: 'Paper metadata'};
  const pm = pick('paper_metadata');
  if (pm && typeof pm === 'object') meta.paper_metadata = pm;
  const st = pick('subtopics');
  if (st && typeof st === 'object') meta.subtopics = st;
  const ol = pick('one_line');
  if (typeof ol === 'string' && ol) meta.one_line = ol;
  const qn = pick('qual_notes');
  if (typeof qn === 'string' && qn) meta.qual_notes = qn;

  // Nothing to synthesise — leave entries alone so the loader doesn't
  // produce a phantom metadata row for non-AI papers.
  if (Object.keys(meta).length <= 1) return {entries, indexMap: null};

  const newEntries = [meta, ...entries];
  // Original findings[N] are now at entry index N+1.
  const indexMap = {};
  entries.forEach((_, i) => { indexMap[i] = i + 1; });
  return {entries: newEntries, indexMap};
}


/* Wrap a paper-level evidence field path under ``samples[0]`` so it
 * resolves against the synthesised metadata entry that
 * ``_synthesizeAiFindingsMetadataEntry`` prepends.  Recognises field
 * paths starting with ``paper_metadata`` / ``subtopics`` / ``one_line``
 * / ``qual_notes`` — everything else is returned unchanged. */
const _PAPER_LEVEL_EVIDENCE_PREFIXES = ['paper_metadata', 'subtopics', 'one_line', 'qual_notes'];
function _wrapPaperLevelEvidence(e) {
  if (!e || typeof e !== 'object' || typeof e.field !== 'string') return e;
  const f = e.field;
  for (const prefix of _PAPER_LEVEL_EVIDENCE_PREFIXES) {
    if (f === prefix || f.startsWith(prefix + '.') || f.startsWith(prefix + '[')) {
      return Object.assign({}, e, {field: `samples[0].${f}`});
    }
  }
  return e;
}


/* Rewrite the leading ``samples[N]`` index in an evidence field path to
 * match a new entry numbering.  Used after ``_groupPerRegressionByTable``
 * collapses multiple per-regression entries into a single per-table
 * entry: the original evidence still references the per-regression
 * indices it was emitted with, so without this rewrite the click-to-jump
 * + per-entry evidence filter silently drop every snippet. */
function _remapEvidenceIndex(e, indexMap) {
  if (!e || typeof e !== 'object' || typeof e.field !== 'string') return e;
  const m = e.field.match(/^samples\[(\d+)\](.*)$/);
  if (!m) return e;
  const oldIdx = parseInt(m[1], 10);
  if (!(oldIdx in indexMap)) return e;
  const newIdx = indexMap[oldIdx];
  if (newIdx === oldIdx) return e;
  return Object.assign({}, e, {field: `samples[${newIdx}]${m[2]}`});
}


/* Pull entries / evidence / extraction_confidence / paper_metadata /
 * original_model_response out of a loaded paper object, handling the
 * three common upstream shapes we've seen in the wild:
 *
 *   (1) canonical MetaPaperLens shape — fields already at paper level:
 *       {filename, entries: [...], evidence: [...], extraction_confidence,
 *        original_model_response, ...}
 *
 *   (2) one-level-of-nesting (43_ai_labor_dashboard / many external
 *       pipelines):
 *       {filename, model, result: {paper_metadata, findings|samples|
 *        regressions, evidence, extraction_confidence}}
 *
 *   (3) bare top-level extraction (the model's raw response written
 *       directly to a per-paper file with no wrapper):
 *       {filename, findings|samples|regressions, evidence, ...}
 *
 * For (2) and (3), the array-shaped key (``findings`` / ``samples`` /
 * ``regressions``) is hoisted to ``entries``.  The original ``result``
 * sub-object (or, for shape 3, the whole record minus ``filename`` /
 * ``model``) is serialised as ``original_model_response`` so the raw-
 * view + parseFull paths still work.
 *
 * Pure function — no DOM, no state.  Returns:
 *   {entries, evidence, extraction_confidence, paper_metadata,
 *    original_model_response}
 */
function _canonicaliseLoadedPaper(p) {
  const ENTRY_ARRAY_KEYS = _LOADED_ENTRY_ARRAY_KEYS;
  // Look for an entries-shaped array first at the top level, then
  // inside a ``result`` wrapper.  First non-empty hit wins.
  const result = (p && typeof p === 'object' && p.result && typeof p.result === 'object')
    ? p.result : null;
  let entries = null;
  let evidence = null;
  let confidence = null;
  let paperMeta = null;
  for (const key of ENTRY_ARRAY_KEYS) {
    if (Array.isArray(p[key])) { entries = p[key]; break; }
  }
  if (!entries && result) {
    for (const key of ENTRY_ARRAY_KEYS) {
      if (Array.isArray(result[key])) { entries = result[key]; break; }
    }
  }
  if (Array.isArray(p.evidence)) evidence = p.evidence;
  else if (result && Array.isArray(result.evidence)) evidence = result.evidence;
  evidence = evidence || [];

  if (p.extraction_confidence && typeof p.extraction_confidence === 'object') {
    confidence = p.extraction_confidence;
  } else if (result && result.extraction_confidence
             && typeof result.extraction_confidence === 'object') {
    confidence = result.extraction_confidence;
  }

  if (p.paper_metadata && typeof p.paper_metadata === 'object') {
    paperMeta = p.paper_metadata;
  } else if (result && result.paper_metadata
             && typeof result.paper_metadata === 'object') {
    paperMeta = result.paper_metadata;
  }

  // Two shape-specific transformations, mutually exclusive — at most
  // one fires per paper because the detection signatures don't overlap.
  // Both return an ``indexMap`` of {old-entry-index → new-entry-index}
  // so we can rewrite evidence ``samples[N]`` indices to track the
  // regrouping — without this the per-entry evidence filter would drop
  // every snippet pointing at an entry that's been moved or merged.
  //
  // (a) Econ per-regression → per-table grouping + paper-meta entry.
  // (b) AI-findings: prepend a "Paper metadata" entry so the reviewer
  //     sees paper-level context separately from individual findings.
  let _entryIndexMap = null;
  let _wrapPaperLevel = false;
  if (Array.isArray(entries)) {
    const grp = _groupPerRegressionByTable(p, entries);
    entries        = grp.entries;
    _entryIndexMap = grp.indexMap;
  }
  if (Array.isArray(entries) && !_entryIndexMap) {
    const syn = _synthesizeAiFindingsMetadataEntry(p, entries);
    entries        = syn.entries;
    _entryIndexMap = syn.indexMap;
    // The AI-findings synthesis inserts a synthesised metadata entry
    // at samples[0] — paper-level evidence (paper_metadata.* etc.) needs
    // wrapping under that prefix so it resolves to the right entry.
    _wrapPaperLevel = (syn.indexMap !== null);
  }

  // Synthesize a sample_id for each entry when it's missing.  Helps
  // the sidebar render readable labels even when the entries came
  // from a flat-shape upstream that didn't bother naming them.
  if (Array.isArray(entries)) {
    entries = entries.map((e, i) => {
      if (!e || typeof e !== 'object') return e;
      if (typeof e.sample_id === 'string' && e.sample_id.trim()) return e;
      const synthesized =
        (typeof e.regression_id === 'string' && e.regression_id) ||
        (typeof e.finding_type === 'string' && e.metric
            ? `${e.finding_type}: ${e.metric}` : null) ||
        (typeof e.metric === 'string' && e.metric) ||
        (typeof e.id === 'string' && e.id) ||
        `Entry ${i + 1}`;
      return Object.assign({}, e, {sample_id: synthesized});
    });
  }

  // original_model_response — for shape 1 use the existing field;
  // for shapes 2/3 serialise the source object so parseFull can
  // round-trip the structured view.
  let originalRaw = '';
  if (typeof p.original_model_response === 'string' && p.original_model_response) {
    originalRaw = p.original_model_response;
  } else if (typeof p.llm_raw_response === 'string' && p.llm_raw_response) {
    // Per-paper "Download JSON" export — same content, different key.
    originalRaw = p.llm_raw_response;
  } else if (result) {
    try { originalRaw = JSON.stringify(result); } catch (_) { originalRaw = ''; }
  } else {
    // Shape 3 — synthesise from the array-shaped key + sibling fields
    try {
      const synthRoot = {};
      for (const key of ENTRY_ARRAY_KEYS) {
        if (Array.isArray(p[key])) { synthRoot[key] = p[key]; }
      }
      if (Array.isArray(p.evidence))           synthRoot.evidence = p.evidence;
      if (p.extraction_confidence)             synthRoot.extraction_confidence = p.extraction_confidence;
      if (p.paper_metadata)                    synthRoot.paper_metadata = p.paper_metadata;
      originalRaw = JSON.stringify(synthRoot);
    } catch (_) { originalRaw = ''; }
  }

  // Recover evidence / confidence / paper metadata from the raw model
  // response when the paper object carries none of its own.  This app's
  // OWN exports are exactly that case: "Download all → JSON" keeps the
  // model's evidence array inside ``original_model_response`` and the
  // per-paper "Download JSON" inside ``llm_raw_response``, with nothing
  // at paper level.  Without this, re-loading a file the app itself
  // wrote produced a review view with no highlights, no page navigator
  // and no click-to-jump — commitLoadJson assigns ``canonical.evidence``
  // over the parsed response, so an empty array here erased evidence
  // that parseFull had already recovered.  parseFull (not JSON.parse)
  // so fenced and truncated responses are tolerated the same way the
  // live extraction path tolerates them.
  if ((!evidence.length || !confidence || !paperMeta) && originalRaw) {
    const fromRaw = parseFull(originalRaw);
    if (fromRaw && typeof fromRaw === 'object') {
      if (!evidence.length && Array.isArray(fromRaw.evidence)) {
        evidence = fromRaw.evidence;
      }
      if (!confidence && fromRaw.extraction_confidence
          && typeof fromRaw.extraction_confidence === 'object') {
        confidence = fromRaw.extraction_confidence;
      }
      if (!paperMeta && fromRaw.paper_metadata
          && typeof fromRaw.paper_metadata === 'object') {
        paperMeta = fromRaw.paper_metadata;
      }
    }
  }

  // Rewrite evidence ``field`` paths so the renderer's click-to-jump
  // logic + the sub-tab routing both work.  The canonical form the
  // app expects is ``samples[N]....``; upstream pipelines emit any of
  // ``findings[N]....`` (AI dataset), ``regressions[N]....`` (econ),
  // or ``records[N]....`` (effect-size shape).  Rewrite leading
  // segments only — sub-paths like ``.estimates[0].estimate`` stay
  // untouched.  Done in-place on a copy so we don't mutate the
  // user's loaded JSON object.
  evidence = evidence.map(_canonicaliseEvidenceField);
  if (_entryIndexMap) {
    evidence = evidence.map(e => _remapEvidenceIndex(e, _entryIndexMap));
  }
  if (_wrapPaperLevel) {
    evidence = evidence.map(_wrapPaperLevelEvidence);
  }
  // Propagate extraction_confidence onto every entry so the
  // per-entry confidence-badge renderer (_renderConfidenceBadges,
  // which reads entry.extraction_confidence) lights up.  The
  // canonical shape carries the ratings at paper level; copying the
  // reference onto each entry is cheap and keeps the badge logic
  // unchanged.  Synthesised entries (Paper metadata for AI-findings,
  // per-table groups for econ) get the same set so badges are
  // consistent across the sidebar.
  if (Array.isArray(entries) && confidence && typeof confidence === 'object') {
    entries = entries.map(e => (
      e && typeof e === 'object' && !('extraction_confidence' in e)
        ? Object.assign({}, e, {extraction_confidence: confidence})
        : e
    ));
  }
  // Surface the remap markers so commitLoadJson can apply the same
  // transformations to ``paper.parsed.evidence`` — without this,
  // click-to-jump silently fails after a synthesis fires because the
  // click handler reads parsed.evidence (which would still carry the
  // pre-synthesis sample indices) while the renderer uses
  // canonical.entries (post-synthesis).
  const _syncMeta = {
    indexMap:        _entryIndexMap,
    wrapPaperLevel:  _wrapPaperLevel,
  };

  return {
    entries:                  entries,
    evidence:                 evidence,
    extraction_confidence:    confidence,
    paper_metadata:           paperMeta,
    original_model_response:  originalRaw,
    _sync_meta:               _syncMeta,
  };
}

// Step 2: commit — build paper objects and navigate to results.
//
// Wrapped in try/catch so any synchronous throw in the rendering chain
// (parseFull, displayPaper, renderEntry, sidebar) surfaces a clear
// error message instead of failing silently and leaving the user
// staring at the unchanged Load button.  Without this guard a single
// malformed paper anywhere in a 30-paper bundle is invisible to the
// user — the click registers, the handler throws, nothing navigates.
function commitLoadJson() {
  if (!reviewJsonData) {
    showJsonError('Internal: no parsed JSON in memory. Re-drop the file.');
    return;
  }
  const data = reviewJsonData;
  try {
    state.generatedPrompt = data.prompt || '';
    state.model           = data.model  || 'gpt-4o';
    state.loadedFromFile  = true;

    state.papers = data.papers.map((p, idx) => {
      // Auto-adapter: external research pipelines commonly emit a
      // ``{filename, model, result: {paper_metadata, findings|samples|
      // regressions, evidence, extraction_confidence, ...}}`` shape
      // instead of MetaPaperLens' canonical (entries at the paper
      // level).  Treat ``paper.result`` as a transparent wrapper and
      // lift its array-shaped contents into the canonical fields.
      const canonical = _canonicaliseLoadedPaper(p);
      const rawResult = canonical.original_model_response;
      const parsed    = parseFull(rawResult) || canonical.entries || null;
      // Keep parsed in sync with the canonicalised view: the click-to-
      // jump handler reads ``paper.parsed.evidence`` while the renderer
      // reads ``paper.entries``.  If a synthesis fired (per-regression
      // grouping or AI-findings metadata-entry) the canonical evidence
      // has been remapped to the new entry indices but parsed.evidence
      // would still carry the pre-synthesis indices.  Sharing the same
      // arrays guarantees the two stay aligned.
      if (parsed && Array.isArray(canonical.evidence)) {
        parsed.evidence = canonical.evidence;
      }
      if (parsed && Array.isArray(canonical.entries)) {
        parsed.entries = canonical.entries;
      }
      // The loaded JSON may carry evidence at the paper level (the
      // canonical shape) or nested under ``result.evidence``.  The
      // canonical-iser handles both; surface counts so the
      // evidence-warning banner doesn't falsely claim the prompt
      // skipped evidence.  ``Count`` is an OPTIMISTIC estimate
      // (entries with an integer page) — when the user also uploads
      // PDFs, ``fetchReviewPageImages`` re-verifies and replaces this.
      const evArr  = canonical.evidence;
      const evTotal = evArr.length;
      const evCount = evArr.filter(e => Number.isInteger(e?.page)).length;
      return {
        id:              _uuidV4(),
        blob:            null,
        filename:        p.filename || `unknown-${idx}.pdf`,
        status:          'done',
        result:          rawResult,
        rawResponse:     rawResult,
        pageImages:      [],
        highlights:      [],
        entries:         canonical.entries,
        parsed:          parsed,
        entryIndex:      0,
        evidencePages:   [],
        evidencePageIdx: 0,
        evidenceTotal:   evTotal,
        evidenceCount:   evCount,
        tokenUsage:      p.token_usage || null,
        resolvedModel:   p.resolved_model || null,
        pagesProcessed:  p.pages_processed || 0,
        error:           null,
        overrides:       reconstructOverrides(p.human_overrides),
        _syncMeta:       canonical._sync_meta || null,
      };
    });

    const pdfsToFetch = [...reviewPdfFiles];
    state.activePaperId = state.papers[0].id;
    cancelLoadOption();

    // Auto-activate a matching preset based on the entries' field
    // signatures — gives users sub-tabs without forcing them to pick
    // the preset before loading.  Best-effort: fetches the preset
    // descriptor from the server in the background; the renderer
    // picks up state.activePreset.sub_views on the next displayPaper.
    _autoActivatePresetForLoadedData(state.papers);

    displayPaper(state.papers[0]);
    goTo(8);

    if (pdfsToFetch.length) fetchReviewPageImages(pdfsToFetch);
  } catch (err) {
    console.error('[commitLoadJson] failed:', err);
    showJsonError(
      'Load failed while preparing the review view: '
      + (err && err.message ? err.message : String(err))
      + '. Check the browser console for the full stack trace.'
    );
  }
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
  // Strip any leading directory components — the JSON often carries
  // ``pdfs/foo.pdf`` while the browser File API only gives us ``foo.pdf``.
  const _baseName = (s) =>
    (s || '').split(/[/\\]/).pop().toLowerCase();

  for (const pdfFile of pdfFiles) {
    // Match by basename (case-insensitive, path-prefix-tolerant)
    const uploadedBase = _baseName(pdfFile.name);
    const paper = state.papers.find(
      p => _baseName(p.filename) === uploadedBase
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
      paper.scannedPages = data.scanned_pages || [];
      // Keep the File object around so the cell-click value-search can
      // re-post the PDF to /api/find-text without asking the user to
      // re-upload.  The File object stays in browser memory cheaply
      // (it's a handle, not a copy of the bytes).
      paper.pdfFile = pdfFile;
      // Populate paper.highlights so the SVG overlay can draw yellow
      // baseline rects AND the green-focus class for clicked cells /
      // evidence_idx chips.  The server returns field paths from the
      // raw JSON (e.g. ``findings[0].value``) so we apply the same
      // canonicalisation + index-remap + paper-level wrap chain that
      // ran on paper.parsed.evidence — otherwise the green-focus
      // set (which keys off the canonicalised paths) won't match.
      paper.highlights = (data.highlights || []).map(h => {
        if (!h || typeof h !== 'object' || typeof h.field !== 'string') return h;
        let mapped = _canonicaliseEvidenceField(h);
        const sync = paper._syncMeta;
        if (sync) {
          if (sync.indexMap)       mapped = _remapEvidenceIndex(mapped, sync.indexMap);
          if (sync.wrapPaperLevel) mapped = _wrapPaperLevelEvidence(mapped);
        }
        return mapped;
      });

      // Journal-page-vs-PDF-page offset: when the JSON's evidence pages
      // are journal page numbers (e.g. 153) and the PDF is internally
      // 1-indexed at a smaller range, the server detects the constant
      // offset and we apply it to evidence here so click-to-jump and
      // the initial-page picker target the right PDF page.
      const offset = data.page_offset || 0;
      if (offset && paper.parsed && Array.isArray(paper.parsed.evidence)) {
        for (const e of paper.parsed.evidence) {
          if (typeof e.page === 'number') e.page = e.page + offset;
        }
        paper.evidenceCount = paper.parsed.evidence.filter(
          e => Number.isInteger(e?.page)
        ).length;
      }
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
  // Capture MASEMiner-ness BEFORE wiping state — but ONLY trust the
  // body class and the URL path.  state.activePreset is contaminated
  // by autoRestoreSession: a user who used MASEMiner once and then
  // navigated to "/" for plain MetaPaperLens will silently carry the
  // restored masem preset in state even though the visible UI is
  // PaperLens.  Trusting state.activePreset.id.startsWith("masem")
  // here would teleport them to /maseminer on Start-over — which is
  // exactly the "backwards-button lands on MASEMiner" bug.
  const inMasemMode = document.body.classList.contains('is-maseminer')
    || window.location.pathname === '/maseminer';

  Object.assign(state, {
    mode: null, provider: 'openai', model: 'gpt-4o', apiKey: '', baseUrl: '',
    providerCredentials: {},
    question: '', context: '', inputMode: 'generate',
    generatedPrompt: '', useTextExtraction: true,
    notifyEmail: '', batchId: null,
    selectedFiles: [], papers: [],
    activePaperId: null, loadedFromFile: false, setupReturnStep: null,
    activePreset: null,
  });
  document.getElementById('questionInput').value     = '';
  document.getElementById('contextInput').value      = '';
  document.getElementById('manualPromptInput').value = '';
  showStep3Choice();
  renderFileList();
  cancelLoadOption();
  clearAutoSave();

  // MASEMiner mode: the "fresh start" surface is the welcome hero, which
  // only renders at page-load.  goTo(1) wouldn't bring it back — and on
  // PaperLens-mode URLs with an active masem preset, goTo(1) would even
  // show the regular Extract/Label/Summarise cards because
  // body.is-maseminer isn't set there.  Reload the current path (or
  // /maseminer when on the dedicated route) for a clean welcome hero.
  // Autosave is already cleared above so nothing is lost.
  if (inMasemMode) {
    // Any masem-flavoured reset lands on /maseminer — whether we got
    // here via the dedicated route, the local maseminer-only deploy, or
    // a masem preset applied on the regular MetaPaperLens path.
    window.location.href = '/maseminer';
    return;
  }

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
