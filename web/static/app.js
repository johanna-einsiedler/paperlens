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
  useTextExtraction: false,
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

// Models that support image input (vision). DeepSeek is text-only.
const PROVIDER_MODELS = {
  openai:   [
    { value: 'gpt-4o',       label: 'GPT-4o — recommended' },
    { value: 'gpt-4o-mini',  label: 'GPT-4o Mini — faster & cheaper' },
    { value: 'gpt-4-turbo',  label: 'GPT-4 Turbo' },
  ],
  google:   [
    { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash — recommended' },
    { value: 'gemini-2.5-pro',   label: 'Gemini 2.5 Pro — most capable' },
    { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash' },
  ],
  deepseek: [
    { value: 'deepseek-chat',     label: 'DeepSeek Chat — text extraction' },
    { value: 'deepseek-reasoner', label: 'DeepSeek Reasoner (R1) — text extraction' },
  ],
  vllm: [], // model name is entered as free text
};

const PROVIDER_KEY_PLACEHOLDER = {
  openai:   'sk-...',
  google:   'AIza...',
  deepseek: 'sk-...',
  vllm:     'any string (or leave blank if auth is disabled)',
};

const PROVIDER_KEY_LABEL = {
  openai:   'OpenAI API key',
  google:   'Google Gemini API key',
  deepseek: 'DeepSeek API key',
  vllm:     'API key',
};

function getProvider(model) {
  if (state.provider === 'vllm')    return 'vllm';
  if (model.startsWith('gemini'))   return 'google';
  if (model.startsWith('deepseek')) return 'deepseek';
  return 'openai';
}

// Returns true for vision-based models, false for text-only (DeepSeek).
// vLLM models default to vision — user can toggle to text extraction if their model doesn't support it.
function isVisionModel(model) { return getProvider(model) !== 'deepseek'; }

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
  for (let n = 1; n <= 5; n++) {
    const sectionId = ['step1', 'step2', 'step3', 'step5', 'step6'][n - 1];
    const el        = document.getElementById(sectionId);
    if (!el) continue;
    el.classList.remove('acc-pending', 'acc-active', 'acc-done');
    let kind = 'acc-pending';
    if      (n  < currentSection) kind = 'acc-done';
    else if (n === currentSection) kind = 'acc-active';
    el.classList.add(kind);
    // Swap the number for a ✓ on completed sections
    const num = el.querySelector('.acc-num');
    if (num) num.textContent = (kind === 'acc-done') ? '✓' : String(n);
  }

  // Per-section summary text — short labels of what's been picked
  const summaries = {
    1: state.mode === 'extraction' ? 'Extract data'
      : state.mode === 'labeling'   ? 'Label paper'
      : state.loadedFromFile        ? 'Review existing results'
      : '',
    2: state.model
      ? `${({openai:'OpenAI', google:'Gemini', deepseek:'DeepSeek', vllm:'Custom'}[state.provider] || state.provider)} · ${state.model}`
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

  // 5. Persist if we actually swapped providers
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

// Initialise the model list on page load
document.addEventListener('DOMContentLoaded', () => {
  onProviderChange(); // populate model list for default provider
  initUploadZone();
  initResultDisplay();
  initZoomPan();

  // Restore the user's last session (provider, model, prompt, etc.) so an
  // accidental refresh doesn't wipe their configuration.
  autoRestoreSession();
  refreshPastBatches();
  loadServerConfig();
  // /masemminer path → hero landing; ?preset=<id> → auto-apply; else inline list
  applyPathOrQueryPreset();

  // First-time visitor: auto-open the "How does this work?" panel so the
  // landing screen isn't just three buttons with no orientation.
  try {
    if (!localStorage.getItem('paperlens.visited')) {
      const box = document.getElementById('howItWorksBox');
      if (box) box.open = true;
      localStorage.setItem('paperlens.visited', '1');
    }
  } catch (_) { /* localStorage may be disabled — ignore */ }

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
  // Preset path: prompt is already loaded, so skip section 3 (describe)
  // and go straight to section 4 (review prompt).
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
}

function setInputMode(mode) {
  state.inputMode = mode;
  const isManual = mode === 'manual';
  document.getElementById('step3Choice').style.display   = 'none';
  document.getElementById('aiSection').style.display     = isManual ? 'none' : '';
  document.getElementById('manualSection').style.display = isManual ? ''     : 'none';
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
  goTo(5);
  updateEvidenceWarning();
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
    goTo(5);
    updateEvidenceWarning();
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

function confirmPrompt() {
  document.getElementById('promptSummaryText').textContent  = state.generatedPrompt;
  document.getElementById('promptSummaryModel').textContent = state.model;
  const note = document.getElementById('visionNote');
  if (!isVisionModel(state.model)) {
    note.textContent =
      `ℹ️ ${state.model} uses text extraction from the PDF's text layer instead of image analysis. Works well for native text PDFs; scanned papers may not extract correctly.`;
    note.style.display = 'block';
  } else {
    note.style.display = 'none';
  }
  // Show parsing method toggle only for vision models (DeepSeek always uses text extraction)
  const parsingGroup = document.getElementById('parsingMethodGroup');
  if (parsingGroup) {
    parsingGroup.style.display = isVisionModel(state.model) ? '' : 'none';
    // Reset to vision (default) when switching models
    const visionRadio = parsingGroup.querySelector('input[value="vision"]');
    if (visionRadio) { visionRadio.checked = true; state.useTextExtraction = false; }
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
    state.selectedFiles.push({
      name: file.name,
      size: file.size,
      buffer,                                // stable copy of the bytes
      blob: new Blob([buffer], { type: 'application/pdf' }),
    });
    existing.add(key);
  }
  renderFileList();
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
        <span class="file-size">${formatBytes(f.size)}</span>
        <button class="file-remove" onclick="removeFile(${i})" title="Remove">✕</button>
      </div>
    `).join('')}`;

  renderCostEstimate();
  renderSizeWarning();
}

/* ── Cost estimator ─────────────────────────────────────────────────────────
   USD per 1M tokens.  Numbers reflect public rate cards as of mid-2025; they
   drift, so the estimate is framed as a range and labelled "approximate". */
const _MODEL_RATES = {
  'gpt-4o':              {in: 2.50,  out: 10.00},
  'gpt-4o-mini':         {in: 0.15,  out: 0.60},
  'gpt-4-turbo':         {in: 10.00, out: 30.00},
  'gemini-2.5-pro':      {in: 1.25,  out: 5.00},
  'gemini-2.5-flash':    {in: 0.075, out: 0.30},
  'gemini-2.0-flash':    {in: 0.075, out: 0.30},
  'deepseek-chat':       {in: 0.27,  out: 1.10},
  'deepseek-reasoner':   {in: 0.55,  out: 2.19},
};

/* Rough tokens-per-page when the PDF is sent as page images.  OpenAI's
   high-detail vision tokenises a 1700×2200 page as ~1100 input tokens.
   Gemini is similar order of magnitude.  vLLM/Ollama vary widely so we
   show "self-hosted" instead of dollars. */
const _VISION_TOKENS_PER_PAGE = 1100;
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
  if (state.provider === 'vllm') return null;
  const rate = _MODEL_RATES[state.model];
  if (!rate) return null;

  const useText = state.useTextExtraction || state.provider === 'deepseek';
  let inputTokens = 0;
  for (const f of state.selectedFiles) {
    if (useText) {
      const chars = f.size * _TEXT_CHARS_PER_BYTE;
      inputTokens += chars / _TEXT_CHARS_PER_TOKEN;
    } else {
      const pages = _estimatePagesFromBytes(f.size);
      inputTokens += pages * _VISION_TOKENS_PER_PAGE;
    }
    // Add prompt overhead (~ length of the generated prompt) per paper
    inputTokens += (state.generatedPrompt?.length || 4000) / 4;
  }
  const outputTokens = _OUTPUT_TOKENS_PER_PAPER * state.selectedFiles.length;
  const usd = (inputTokens / 1e6) * rate.in + (outputTokens / 1e6) * rate.out;
  return { usd, inputTokens, outputTokens, useText };
}

function renderCostEstimate() {
  const el = document.getElementById('costEstimate');
  if (!el) return;
  if (state.selectedFiles.length === 0) {
    el.style.display = 'none';
    return;
  }
  const est = estimateBatchCostUsd();
  if (est === null) {
    // Self-hosted or unknown model — skip the dollar amount but show a note
    el.style.display     = 'flex';
    el.innerHTML = `<span class="cost-est-icon">≈</span>
      <span><strong>Self-hosted model</strong> — runs on your own server, no per-token cost.</span>`;
    return;
  }
  // Show a ±50 % range to communicate genuine uncertainty
  const low  = est.usd * 0.5;
  const high = est.usd * 1.5;
  const fmt  = n => n < 0.01 ? '< 0.01' : n.toFixed(n < 1 ? 2 : 2);
  const n    = state.selectedFiles.length;
  el.style.display = 'flex';
  el.innerHTML =
    `<span class="cost-est-icon">≈</span>` +
    `<span><strong>Estimated cost: \$${fmt(low)} – \$${fmt(high)}</strong> ` +
    `for ${n} paper${n !== 1 ? 's' : ''} on ${escHtml(state.model)} ` +
    `(${est.useText ? 'text' : 'vision'} mode). ` +
    `<span class="cost-est-note">Approximate — actual usage may vary.</span></span>`;
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

  try {
    // crypto.randomUUID is only available in secure contexts (HTTPS or localhost).
    // Fall back to a manual UUID v4 if the browser doesn't expose it.
    const uuid = () => (crypto && crypto.randomUUID
      ? crypto.randomUUID()
      : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
          const r = (Math.random() * 16) | 0;
          return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
        }));

    // One batch id shared by every paper in this upload — server uses it to
    // group jobs, fire the completion email once, and surface the History view.
    state.batchId    = uuid();
    state.notifyEmail = '';   // captured later via the in-loading-screen prompt

    state.papers = state.selectedFiles.map(f => ({
      id: uuid(),
      blob: f.blob,
      filename: f.name,
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
    }));
    state.activePaperId = null;
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
  // Show loading screen for the batch
  const n = state.papers.length;
  document.getElementById('loadingTitle').textContent   = 'Extracting data\u2026';
  document.getElementById('extractingNote').textContent =
    `Submitting ${n} paper${n > 1 ? 's' : ''} for processing\u2026`;

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
  // when the batch is short.
  const emailPromptTimer = setTimeout(() => {
    if (emailWrap && state.step === 7) emailWrap.style.display = 'block';
  }, 8000);

  goTo(7);

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

  const form = new FormData();
  form.append('api_key',             state.apiKey);
  form.append('model',               state.model);
  form.append('prompt',              state.generatedPrompt);
  form.append('use_text_extraction', state.useTextExtraction ? '1' : '0');
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
    paper.pageImages        = [];
    paper.pageImagesFetched = false;
    paper.pagesProcessed    = result.pages_processed || 0;
    paper.entries           = parseEntries(result.result);
    paper.parsed            = parseFull(result.result);
    paper.entryIndex        = 0;
    paper.evidenceCount     = result.evidence_count ?? null;
    paper.evidenceTotal     = result.evidence_total ?? null;
    paper.tokenUsage        = result.token_usage    ?? null;

    if (state.activePaperId === null) {
      state.activePaperId = paper.id;
      displayPaper(paper);
      goTo(8);
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
    paper.pageImages = data.page_images || [];
    paper.highlights = data.highlights || [];   // [{page, snippet, field, source, rects}]
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
  sidebar.innerHTML = state.papers.map(p => {
    const isActive  = p.id === state.activePaperId;
    const icon      = { pending: '○', processing: '⟳', done: '✓', error: '✕', cancelled: '⊘' }[p.status] || '·';
    const clickable = p.status === 'done' || p.status === 'error' || p.status === 'cancelled';
    const cls       = ['paper-item', isActive ? 'active' : '', `status-${p.status}`].filter(Boolean).join(' ');
    const onclick   = clickable ? `onclick="setActivePaper('${p.id}')"` : '';
    const phaseLabel = (p.status === 'processing' && p.phase)
      ? `<span class="paper-phase">${escHtml(p.phase)}</span>` : '';
    // Per-paper Stop button while in flight
    const stopBtn = (p.status === 'pending' || p.status === 'processing')
      ? `<button class="paper-stop" onclick="event.stopPropagation(); cancelPaper('${p.id}')" title="Stop this paper">✕</button>`
      : '';
    // Sub-tabs (preset-driven) — only on the active, finished paper
    let subTabsHtml = '';
    if (isActive && subViews?.length && p.status === 'done') {
      const activeId = p.subView || subViews[0].id;
      subTabsHtml = `<div class="paper-subtabs">${subViews.map(sv => `
        <button class="paper-subtab ${sv.id === activeId ? 'active' : ''}"
                onclick="event.stopPropagation(); setSubView('${escHtml(sv.id)}')"
                title="${escHtml(sv.label)}">
          ${escHtml(sv.label)}
        </button>`).join('')}</div>`;
    }
    return `
      <div class="${cls}" ${onclick}>
        <span class="paper-status-icon">${icon}</span>
        <span class="paper-name-wrap">
          <span class="paper-name">${escHtml(p.filename.replace(/\.pdf$/i, ''))}</span>
          ${phaseLabel}
        </span>
        ${stopBtn}
      </div>
      ${subTabsHtml}`;
  }).join('');
  updateRetryAllButton();
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

/* Match an evidence entry's `field` path against the sub-view's keys. */
function _evidenceMatchesSubView(field, subView) {
  if (!subView || !field) return true;
  if (Array.isArray(subView.include_keys)) {
    return subView.include_keys.some(k => field.includes(k));
  }
  if (Array.isArray(subView.exclude_keys)) {
    return !subView.exclude_keys.some(k => field.includes(k));
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

  // Banner — visible while preset is active
  const banner = document.getElementById('presetBanner');
  if (banner) {
    document.getElementById('presetBannerTitle').textContent   = preset.title;
    document.getElementById('presetBannerTagline').textContent = preset.tagline || '';
    banner.style.display = 'flex';
  }

  // Apply preset values to state
  if (preset.mode)              state.mode             = preset.mode;
  if (preset.default_provider)  state.provider         = preset.default_provider;
  if (preset.default_model)     state.model            = preset.default_model;
  if (preset.task_description)  state.question         = preset.task_description;
  if (preset.context)           state.context          = preset.context;
  if (preset.prompt) {
    state.generatedPrompt = preset.prompt;
    state.inputMode       = 'manual';   // user can still hit Regenerate later
  }

  // Reflect into the visible form fields (in case the user expands earlier sections)
  const provSel = document.getElementById('providerSelect');
  if (provSel && preset.default_provider) provSel.value = preset.default_provider;
  onProviderChange();
  const modelSel = document.getElementById('modelSelect');
  if (modelSel && preset.default_model && preset.default_provider !== 'vllm') {
    if (modelSel.querySelector(`option[value="${CSS.escape(preset.default_model)}"]`)) {
      modelSel.value = preset.default_model;
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
  // Skip the AI-generated vs Manual sub-choice on step 3 — the prompt is
  // already loaded, so the manual section should be open by default if
  // the user expands step 3.
  if (preset.prompt) {
    const choiceEl = document.getElementById('step3Choice');
    const aiEl     = document.getElementById('aiSection');
    const manEl    = document.getElementById('manualSection');
    if (choiceEl) choiceEl.style.display = 'none';
    if (aiEl)     aiEl.style.display     = 'none';
    if (manEl)    manEl.style.display    = '';
  }

  // Land on the configured step.  For MASEMminer we go to step 2 (setup);
  // submitStep2 then skips section 3 (describe) and jumps to section 4
  // (review prompt) since the prompt is already loaded.
  const SKIP = { upload: 6, prompt: 5, question: 3, setup: 2, task: 1 };
  const targetStep = SKIP[preset.skip_to] || 2;
  goTo(targetStep);
  return true;
}

/* Used by the /masemminer hero "Get started" button — applies the preset,
   hides the landing, shows the configuration accordion, lands on step 2. */
async function startPresetFromLanding(presetId) {
  const landing  = document.getElementById('masemLanding');
  const onepager = document.getElementById('onepager');
  if (landing)  landing.style.display  = 'none';
  if (onepager) onepager.style.display = '';
  await applyPreset(presetId);
}

function clearPreset() {
  if (!state.activePreset) return;
  // Reset URL so a refresh doesn't re-apply
  if (window.history && window.history.replaceState) {
    const url = new URL(window.location.href);
    url.searchParams.delete('preset');
    window.history.replaceState({}, '', url.toString());
  }
  // Restore the default branding
  document.title = 'PaperLens';
  document.getElementById('appTitle').textContent   = 'PaperLens';
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
   1. If pathname is /masemminer (or similar), show the dedicated hero landing.
   2. Otherwise, if ?preset=<id> is in the query string, auto-apply that preset
      (legacy / direct-link entry).
   3. Otherwise, populate the inline workflows section so users can pick one. */
const _PRESET_PATHS = {
  '/masemminer': 'masem',
  // Add new dedicated path → preset mappings here as more workflows are added.
};

async function applyPathOrQueryPreset() {
  const path      = window.location.pathname;
  const presetForPath = _PRESET_PATHS[path];
  if (presetForPath) {
    // Show the hero landing for this preset; user clicks "Get started" to proceed
    document.getElementById('onepager').style.display = 'none';
    const landing = document.getElementById(presetForPath + 'Landing');
    if (landing) landing.style.display = '';
    return;
  }
  const params = new URLSearchParams(window.location.search);
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
  // Lazily load page images for each finished paper as the user navigates to them
  state.papers.forEach(p => p.status === 'done' && ensurePageImagesLoaded(p));
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
  const paper = state.papers.find(p => p.id === id);
  if (!paper || (paper.status !== 'done' && paper.status !== 'error')) return;
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
  if (!u || (!u.prompt && !u.completion && !u.total)) {
    el.style.display = 'none';
    return;
  }
  const fmt = n => n.toLocaleString();
  el.innerHTML =
    `<span class="token-label">Tokens</span>` +
    `<span class="token-stat">${fmt(u.prompt)} in</span>` +
    `<span class="token-sep">·</span>` +
    `<span class="token-stat">${fmt(u.completion)} out</span>` +
    `<span class="token-sep">·</span>` +
    `<span class="token-stat token-total">${fmt(u.total)} total</span>`;
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

  const nav     = document.getElementById('entryNav');
  const display = document.getElementById('resultDisplay');

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
    // Fallback: render raw text as a single formatted block
    nav.style.display = 'none';
    let parsed = null;
    try { parsed = JSON.parse(paper.result); } catch {}
    display.innerHTML =
      parsed ? renderValueHtml(parsed) : `<pre class="raw-text">${escHtml(stripFences(paper.result))}</pre>`;
    showPageImage(paper, 1);
  }

  renderPaperSidebar();
}

function retryPaper(id) {
  const paper = state.papers.find(p => p.id === id);
  if (!paper) return;
  paper.status            = 'pending';
  paper.error             = null;
  paper.jobId             = null;
  paper.pageImagesFetched = false;
  renderPaperSidebar();
  processPaper(paper);
}

function renderEntry(paper) {
  const entry = paper.entries[paper.entryIndex];
  const total = paper.entries.length;

  document.getElementById('entryCounter').textContent = `Entry ${paper.entryIndex + 1} of ${total}`;
  document.getElementById('prevBtn').disabled = paper.entryIndex === 0;
  document.getElementById('nextBtn').disabled = paper.entryIndex === total - 1;

  // Multi-entry hint + tab strip — only useful when there's more than one
  const hint  = document.getElementById('multiEntryHint');
  const text  = document.getElementById('multiEntryText');
  const tabs  = document.getElementById('entryTabs');
  if (hint && text) {
    if (total > 1) {
      hint.style.display = 'flex';
      text.innerHTML = `This paper has <strong>${total} entries</strong> &mdash; use the buttons or numbered tabs to flip through all of them.`;
    } else {
      hint.style.display = 'none';
    }
  }
  if (tabs) {
    if (total > 1) {
      tabs.style.display = 'flex';
      tabs.innerHTML = paper.entries.map((_, i) =>
        `<button class="entry-tab ${i === paper.entryIndex ? 'active' : ''}" onclick="jumpToEntry(${i})" title="Entry ${i + 1}">${i + 1}</button>`
      ).join('');
    } else {
      tabs.style.display = 'none';
      tabs.innerHTML = '';
    }
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

  // Sub-view filter (preset-driven, e.g. MASEM Loadings/Correlations/Descriptives)
  const subView       = _activeSubViewFor(paper);
  const filteredEntry = subView ? _filterEntryBySubView(entry, subView) : entry;

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
  paper.evidencePages     = evidencePages;
  paper.evidencePageIdx   = 0;
  paper.browseAllPagesIdx = 0;
  updatePageNav(paper);
  // If the entry has cited pages → show the first one (with highlights).
  // Otherwise fall back to page 1 of the captured PDF so the user can flip
  // through the whole document via the always-visible nav.
  const initialPage = paper.evidencePages[0] ?? (paper.pageImages.length ? 1 : null);
  showPageImage(paper, initialPage);
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

  // Render SVG highlight overlay for the displayed page (filtered by sub-view).
  renderHighlightOverlay(paper, displayedPage);
}

/* Returns true if this evidence entry should be drawn given the active
   sub-view filter.  Evidence with no field info (recovered orphans) is
   shown only in the default (no-sub-view) overlay. */
function _highlightMatchesSubView(highlight, subView) {
  if (!subView) return true;
  const field = highlight.field || '';
  if (!field) return false;
  if (Array.isArray(subView.include_keys)) {
    return subView.include_keys.some(k => field.includes(k));
  }
  if (Array.isArray(subView.exclude_keys)) {
    return !subView.exclude_keys.some(k => field.includes(k));
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

  _withImageDims(paper, pageNum - 1, dims => {
    if (!dims) { svg.style.display = 'none'; return; }
    svg.setAttribute('viewBox', `0 0 ${dims.w} ${dims.h}`);
    svg.style.display = 'block';
    for (const h of matching) {
      for (const r of (h.rects || [])) {
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x',      r[0]);
        rect.setAttribute('y',      r[1]);
        rect.setAttribute('width',  r[2]);
        rect.setAttribute('height', r[3]);
        rect.setAttribute('class',  'highlight-rect');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = (h.field ? `[${h.field}] ` : '') + (h.snippet || '');
        rect.appendChild(title);
        svg.appendChild(rect);
      }
    }
  });
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
     (current behaviour — the cited pages are highlighted)
   - if there are none, fall back to "browse every captured page" so the user
     can still scroll through the source PDF page by page. */
function _isBrowseAllMode(paper) {
  return (paper.evidencePages?.length || 0) === 0
      && (paper.pageImages?.length    || 0)  > 1;
}

function updatePageNav(paper) {
  const navEl  = document.getElementById('pageEvidenceNav');
  const prevEl = document.getElementById('pageNavPrev');
  const nextEl = document.getElementById('pageNavNext');
  const cntEl  = document.getElementById('pageNavCounter');

  const browseAll = _isBrowseAllMode(paper);
  const total     = browseAll ? paper.pageImages.length : paper.evidencePages.length;
  const idx       = browseAll ? (paper.browseAllPagesIdx || 0) : paper.evidencePageIdx;

  if (total > 1) {
    navEl.style.display = 'flex';
    cntEl.textContent   = browseAll
      ? `Page ${idx + 1} / ${total}`
      : `${idx + 1}/${total}`;
    prevEl.disabled     = idx === 0;
    nextEl.disabled     = idx === total - 1;
    navEl.classList.toggle('page-nav-browse', browseAll);
  } else {
    navEl.style.display = 'none';
  }
}

function prevEvidencePage() {
  const p = getActivePaper();
  if (!p) return;
  if (_isBrowseAllMode(p)) {
    if ((p.browseAllPagesIdx || 0) === 0) return;
    p.browseAllPagesIdx -= 1;
    updatePageNav(p);
    showPageImage(p, p.browseAllPagesIdx + 1);
  } else {
    if (p.evidencePageIdx === 0) return;
    p.evidencePageIdx--;
    updatePageNav(p);
    showPageImage(p, p.evidencePages[p.evidencePageIdx]);
  }
}

function nextEvidencePage() {
  const p = getActivePaper();
  if (!p) return;
  if (_isBrowseAllMode(p)) {
    const max = p.pageImages.length - 1;
    if ((p.browseAllPagesIdx || 0) >= max) return;
    p.browseAllPagesIdx = (p.browseAllPagesIdx || 0) + 1;
    updatePageNav(p);
    showPageImage(p, p.browseAllPagesIdx + 1);
  } else {
    if (p.evidencePageIdx >= p.evidencePages.length - 1) return;
    p.evidencePageIdx++;
    updatePageNav(p);
    showPageImage(p, p.evidencePages[p.evidencePageIdx]);
  }
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
  const groups = new Set();
  const items  = new Set();
  for (const k of keys) {
    const m = k.match(_DOTTED_KEY_RE);
    if (!m) return false;
    const v = obj[k];
    if (v !== null && typeof v !== 'number') return false;
    groups.add(m[1]);
    items.add(parseInt(m[2], 10));
  }
  return groups.size >= 2 && items.size >= 2;
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

  const head = `<thead><tr>
    <th class="rv-tbl-rowlabel">Item</th>
    ${groupOrder.map(g => `<th>${escHtml(g)}</th>`).join('')}
  </tr></thead>`;

  const body = items.map(item => {
    const labelCell = `<td class="rv-tbl-rowlabel">${item}</td>`;
    const cells = groupOrder.map(g => {
      const key      = `${g}.${item}`;
      const val      = obj[key];
      const cellPath = path ? `${path}.${key}` : key;
      return `<td>${_renderCellHtml(val === undefined ? null : val, cellPath)}</td>`;
    }).join('');
    return `<tr>${labelCell}${cells}</tr>`;
  }).join('');

  const caption = `<span class="rv-table-caption-icon">▦</span> Table · ${items.length} rows × ${groupOrder.length + 1} cols `
                + `<span class="rv-table-source rv-table-source-auto" data-tip="Auto-detected from dotted F1.1-style keys. Update your prompt to use the _table marker for explicit tables.">auto-detected</span>`;
  return `<div class="rv-table-wrap">
    <div class="rv-table-caption">${caption}</div>
    <table class="rv-table">${head}<tbody>${body}</tbody></table>
  </div>`;
}

function renderTableHtml(rows, columns, rowLabels, path, kind) {
  // rows:       array of row-data objects
  // columns:    list of column keys
  // rowLabels:  null (for plain array) or list of strings (parent keys for object-map)
  // path:       path prefix for editable cells
  // kind:       'marker' (explicit _table) | 'auto' | undefined — drives the caption
  const labelHeader = rowLabels ? '<th class="rv-tbl-rowlabel"></th>' : '';
  const head = `<thead><tr>${labelHeader}${columns.map(c => `<th>${escHtml(formatKey(c))}</th>`).join('')}</tr></thead>`;
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
    return `<tr>${labelCell}${cells}</tr>`;
  }).join('');

  // Caption explains why the data is shown as a table
  const nRows = rows.length;
  const nCols = columns.length + (rowLabels ? 1 : 0);
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
  const blob = new Blob([p.result], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = p.filename.replace(/\.pdf$/i, '') + '.json';
  a.click();
  URL.revokeObjectURL(url);
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
  // Apply human overrides on top of original entries before exporting
  if (!paper.entries || paper.entries.length === 0) return [];
  const flat = paper.entries.map((e, i) => {
    const f = _flattenEntry(e);
    const ov = paper.overrides[i] || {};
    for (const [path, info] of Object.entries(ov)) {
      f[path] = info.final_value;
    }
    return f;
  });
  return flat;
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
    papers: done.map(p => {
      // Flatten overrides across all entries into a readable list
      const overrideList = [];
      for (const [entryIdx, fields] of Object.entries(p.overrides)) {
        for (const [fieldPath, ov] of Object.entries(fields)) {
          overrideList.push({
            entry_index:    parseInt(entryIdx, 10),
            field_path:     fieldPath,
            original_value: ov.original_value,
            final_value:    ov.final_value,
            human_override: ov.human_override,
          });
        }
      }
      return {
        filename:                p.filename,
        pages_processed:         p.pagesProcessed,
        token_usage:             p.tokenUsage || null,
        entries:                 p.entries,
        human_overrides:         overrideList,
        original_model_response: p.result,
      };
    }),
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
    generatedPrompt: '', useTextExtraction: false,
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
