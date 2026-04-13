/* ──────────────────────────────────────────────────────────
   State
────────────────────────────────────────────────────────── */

const state = {
  step: 1,
  mode: null,
  provider: 'openai',
  model: 'gpt-4o',
  apiKey: '',
  question: '',
  context: '',
  generatedPrompt: '',
  inputMode: 'generate',
  useTextExtraction: false, // true = send PDF text layer instead of page images
  selectedFiles: [],      // File[] — files chosen in step 6
  papers: [],             // paper objects (see below)
  activePaperId: null,    // id of the paper currently shown in step 8
  loadedFromFile: false,  // true when results were loaded from a JSON file
  setupReturnStep: null,  // step to return to after editing API key/model mid-flow
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
};

const PROVIDER_KEY_PLACEHOLDER = {
  openai:   'sk-...',
  google:   'AIza...',
  deepseek: 'sk-...',
};

const PROVIDER_KEY_LABEL = {
  openai:   'OpenAI API key',
  google:   'Google Gemini API key',
  deepseek: 'DeepSeek API key',
};

function getProvider(model) {
  if (model.startsWith('gemini'))    return 'google';
  if (model.startsWith('deepseek'))  return 'deepseek';
  return 'openai';
}

// Returns true for vision-based models, false for text-extraction models (DeepSeek).
// Used only to show informational notes — text-extraction models ARE allowed for upload.
function isVisionModel(model) { return getProvider(model) !== 'deepseek'; }

/* ──────────────────────────────────────────────────────────
   Navigation
────────────────────────────────────────────────────────── */

const PROG_MAP = { 1: 1, 2: 2, 3: 3, 4: 3, 5: 4, 6: 5, 7: 5, 8: 6 };

function goTo(step) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('step' + step).classList.add('active');
  state.step = step;
  updateProgress(step);
  document.body.classList.toggle('wide-mode', step === 8);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateProgress(step) {
  const current = PROG_MAP[step] || step;
  for (let i = 1; i <= 6; i++) {
    const progEl = document.getElementById('prog' + i);
    const circEl = document.getElementById('circ' + i);
    const connEl = document.getElementById('conn' + i);
    progEl.classList.remove('active', 'done');
    if (i < current) {
      progEl.classList.add('done');
      circEl.textContent = '✓';
      if (connEl) connEl.classList.add('done');
    } else if (i === current) {
      progEl.classList.add('active');
      circEl.textContent = i;
      if (connEl) connEl.classList.remove('done');
    } else {
      circEl.textContent = i;
      if (connEl) connEl.classList.remove('done');
    }
  }
}

/* ──────────────────────────────────────────────────────────
   Step 1 — Mode
────────────────────────────────────────────────────────── */

function selectMode(mode) {
  state.mode = mode;
  goTo(2);
}

/* ──────────────────────────────────────────────────────────
   Step 2 — Provider / Model + API key
────────────────────────────────────────────────────────── */

function onProviderChange() {
  const provider = document.getElementById('providerSelect').value;
  const models   = PROVIDER_MODELS[provider] || [];
  const sel      = document.getElementById('modelSelect');
  sel.innerHTML  = models.map(m => `<option value="${m.value}">${escHtml(m.label)}</option>`).join('');
  document.getElementById('apiKeyInput').placeholder = PROVIDER_KEY_PLACEHOLDER[provider] || '';
  document.getElementById('apiKeyLabel').textContent  = PROVIDER_KEY_LABEL[provider] || 'API key';
  document.getElementById('deepseekWarningGroup').style.display = provider === 'deepseek' ? '' : 'none';
}

// Initialise the model list on page load
document.addEventListener('DOMContentLoaded', () => {
  onProviderChange(); // populate model list for default provider
  initUploadZone();
  initResultDisplay();
  initZoomPan();
});

function submitStep2() {
  const apiKey   = document.getElementById('apiKeyInput').value.trim();
  const model    = document.getElementById('modelSelect').value;
  const provider = document.getElementById('providerSelect').value;
  if (!apiKey) { showToast('Please enter your API key.'); return; }
  state.apiKey    = apiKey;
  state.model     = model;
  state.provider  = provider;

  // If we came here via "Edit API key / model" from a later step, go back there
  if (state.setupReturnStep) {
    const dest = state.setupReturnStep;
    state.setupReturnStep = null;
    goTo(dest);
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

function editSetup() {
  // Pre-fill current values so the user can see what they entered before
  document.getElementById('providerSelect').value = state.provider;
  onProviderChange();  // repopulate model list for this provider
  document.getElementById('modelSelect').value = state.model;
  document.getElementById('apiKeyInput').value  = state.apiKey;
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
}

/* ──────────────────────────────────────────────────────────
   Prompt generation API (steps 4 → 5)
────────────────────────────────────────────────────────── */

async function callGenerateAPI() {
  goTo(4);
  try {
    const res = await fetch('/api/generate-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: state.apiKey, model: state.model, mode: state.mode,
        question: state.question, context: state.context,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to generate prompt.');
    state.generatedPrompt = data.prompt;
    document.getElementById('promptDisplay').textContent = data.prompt;
    document.getElementById('modelBadge').textContent    = data.model_used;
    resetCopyBtn('copyBtn');
    goTo(5);
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
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      showToast(`"${file.name}" is not a PDF — skipped.`);
      continue;
    }
    if (file.size > 50 * 1024 * 1024) {
      showToast(`"${file.name}" exceeds 50 MB — skipped.`);
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
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function submitUpload() {
  if (state.selectedFiles.length === 0) { showToast('Please select at least one PDF.'); return; }

  // Build papers queue from selected files
  state.papers = state.selectedFiles.map(f => ({
    id: crypto.randomUUID(),
    blob: f.blob,        // ArrayBuffer-backed Blob, safe to upload at any time
    filename: f.name,
    status: 'pending',
    result: '',
    rawResponse: null,   // last raw model text, even if non-JSON
    pageImages: [],
    entries: null,
    entryIndex: 0,
    evidencePages: [],   // sorted page numbers for the current entry
    evidencePageIdx: 0,  // index into evidencePages
    evidenceCount: null, // number of evidence snippets returned by server (null = unknown)
    tokenUsage: null,    // {prompt, completion, total} token counts from API
    pagesProcessed: 0,
    error: null,
    overrides: {},       // { [entryIndex]: { [fieldPath]: {original_value, final_value, human_override} } }
  }));
  state.activePaperId = null;

  // Start processing the queue (runs to completion async)
  processQueue();
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
  let isFirst = true;
  for (const paper of state.papers) {
    if (paper.status !== 'pending') continue;
    // 30s pause between papers — OpenAI's vision TPM budget (tokens per minute) is shared
    // across back-to-back requests; shorter pauses cause the model to refuse silently.
    if (!isFirst) await new Promise(r => setTimeout(r, 30000));
    isFirst = false;
    await processPaper(paper);
  }
}

async function processPaper(paper, attempt = 0) {
  paper.status = 'processing';

  if (state.step !== 8) {
    // First paper (or retry before results are shown) — show loading screen
    document.getElementById('loadingTitle').textContent   = attempt > 0 ? 'Retrying\u2026' : 'Extracting data\u2026';
    document.getElementById('extractingNote').textContent =
      `Processing "${paper.filename}"${state.papers.length > 1 ? ` (1 of ${state.papers.length})` : ''}…`;
    goTo(7);
  } else {
    // Subsequent papers — update sidebar only
    renderPaperSidebar();
  }

  const form = new FormData();
  form.append('api_key',            state.apiKey);
  form.append('model',              state.model);
  form.append('prompt',             state.generatedPrompt);
  form.append('use_text_extraction', state.useTextExtraction ? '1' : '0');
  form.append('pdf',                paper.blob, paper.filename);

  try {
    const res  = await fetch('/api/extract', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Extraction failed.');

    // Always save the raw model text so we can show it on failure
    if (data.result) paper.rawResponse = data.result;

    console.log(
      `[processPaper] "${paper.filename}" attempt=${attempt}`,
      `finish_reason=${data.finish_reason}`,
      `pages=${data.pages_processed}`,
      `blob_size=${paper.blob?.size ?? '?'}`,
      `result_length=${data.result?.length ?? 0}`,
    );

    // content_filter means a policy decision — retrying won't help, fail fast
    if (data.finish_reason === 'content_filter') {
      throw new Error(`Blocked by OpenAI content filter (finish_reason=content_filter). Try a different model or reduce the number of pages.`);
    }

    // If the model returned natural language instead of structured output it
    // almost certainly didn't receive the page images (transient API issue).
    // Treat this as a retriable error — HTTP 200 is not sufficient; we need
    // parseable output.  A legitimate labeling result is ≤ 5 chars ("A"–"D");
    // anything longer that doesn't parse is a model confusion response.
    if (parseFull(data.result) === null && data.result.trim().length > 5) {
      throw new Error('Model returned an unexpected response — retrying…');
    }

    paper.status         = 'done';
    paper.result         = data.result;
    paper.pageImages     = data.page_images || [];
    paper.pagesProcessed = data.pages_processed;
    paper.entries        = parseEntries(data.result);
    paper.parsed         = parseFull(data.result);   // full object for page fallback
    paper.entryIndex     = 0;
    paper.evidenceCount  = data.evidence_count ?? null; // null = not reported (old response)
    paper.tokenUsage     = data.token_usage    ?? null; // {prompt, completion, total}

    if (state.activePaperId === null) {
      // First result ready — switch to results page
      state.activePaperId = paper.id;
      displayPaper(paper);
      goTo(8);
    } else {
      renderPaperSidebar(); // background update
    }

  } catch (err) {
    if (attempt === 0) {
      // Auto-retry after 60 s — OpenAI's vision TPM quota resets over a rolling
      // 60 s window.  A 3 s pause is never enough to clear the budget from a large
      // prior request; the model silently refuses with "I can't process documents".
      const onLoadingScreen = state.step !== 8;
      if (onLoadingScreen) {
        // Ensure loading screen is visible so the countdown is shown
        document.getElementById('extractingNote').textContent =
          `"${paper.filename}" will be retried automatically.`;
        goTo(7);
      }
      await sleepWithCountdown(60000, s => {
        if (onLoadingScreen) {
          document.getElementById('loadingTitle').textContent   = `Retrying in ${s}s\u2026`;
          document.getElementById('extractingNote').textContent = `"${paper.filename}" — waiting for API quota to reset.`;
        }
      });
      return processPaper(paper, 1);
    }

    paper.status = 'error';
    paper.error  = err.message;

    if (state.activePaperId === null) {
      // First paper failed both attempts — go back to upload
      showToast(err.message);
      goTo(6);
    } else {
      // Background failure — make it visible with a toast (sidebar shows ✕ too)
      showToast(`"${paper.filename}": ${err.message}`);
      renderPaperSidebar();
    }
  }
}

/* ──────────────────────────────────────────────────────────
   Step 8 — Papers sidebar
────────────────────────────────────────────────────────── */

function renderPaperSidebar() {
  const sidebar = document.getElementById('papersSidebar');
  sidebar.innerHTML = state.papers.map(p => {
    const isActive  = p.id === state.activePaperId;
    const icon      = { pending: '○', processing: '⟳', done: '✓', error: '✕' }[p.status];
    const clickable = p.status === 'done' || p.status === 'error';
    const cls       = ['paper-item', isActive ? 'active' : '', `status-${p.status}`].filter(Boolean).join(' ');
    const onclick   = clickable ? `onclick="setActivePaper('${p.id}')"` : '';
    return `
      <div class="${cls}" ${onclick}>
        <span class="paper-status-icon">${icon}</span>
        <span class="paper-name">${escHtml(p.filename.replace(/\.pdf$/i, ''))}</span>
      </div>`;
  }).join('');
}

function setActivePaper(id) {
  const paper = state.papers.find(p => p.id === id);
  if (!paper || (paper.status !== 'done' && paper.status !== 'error')) return;
  state.activePaperId = id;
  renderEvidenceWarning(paper);
  displayPaper(paper);
  renderPaperSidebar();
}

/* ──────────────────────────────────────────────────────────
   Step 8 — Display a paper's results
────────────────────────────────────────────────────────── */

function renderEvidenceWarning(paper) {
  const el = document.getElementById('evidenceWarning');
  if (!el) return;
  // Show warning only when server confirmed 0 snippets (evidenceCount=0).
  // null means server didn't report it (loaded from file, old response, etc).
  const show = paper.status === 'done' && paper.evidenceCount === 0;
  el.style.display = show ? 'flex' : 'none';
}

function renderTokenFooter(paper) {
  const el = document.getElementById('tokenFooter');
  if (!el) return;
  const u = paper.tokenUsage;
  if (!u || (!u.prompt && !u.completion && !u.total)) {
    el.style.display = 'none';
    return;
  }
  const fmt = n => n.toLocaleString();
  el.innerHTML =
    `<span class="token-label">Tokens</span>` +
    `<span class="token-stat">${fmt(u.prompt)} prompt</span>` +
    `<span class="token-sep">·</span>` +
    `<span class="token-stat">${fmt(u.completion)} completion</span>` +
    `<span class="token-sep">·</span>` +
    `<span class="token-stat token-total">${fmt(u.total)} total</span>`;
  el.style.display = 'flex';
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
  paper.status = 'pending';
  paper.error  = null;
  renderPaperSidebar();
  processPaper(paper);
}

function renderEntry(paper) {
  const entry = paper.entries[paper.entryIndex];
  const total = paper.entries.length;

  document.getElementById('entryCounter').textContent = `Entry ${paper.entryIndex + 1} of ${total}`;
  document.getElementById('prevBtn').disabled = paper.entryIndex === 0;
  document.getElementById('nextBtn').disabled = paper.entryIndex === total - 1;

  const display = document.getElementById('resultDisplay');
  display.dataset.paperId  = paper.id;
  display.dataset.entryIdx = paper.entryIndex;
  display.innerHTML = renderValueHtml(entry);
  applyOverrides(paper);

  // Collect all page numbers referenced in this entry; fall back to top-level
  const entryPages  = [...findAllEntryPages(entry)].sort((a, b) => a - b);
  const globalPages = entryPages.length ? entryPages
    : [...findAllEntryPages(paper.parsed)].sort((a, b) => a - b);
  paper.evidencePages   = globalPages;
  paper.evidencePageIdx = 0;
  updatePageNav(paper);
  showPageImage(paper, paper.evidencePages[0] ?? null);
}

function showPageImage(paper, pageNum) {
  const img   = document.getElementById('pageDisplayImg');
  const none  = document.getElementById('pageDisplayNone');
  const label = document.getElementById('pageDisplayLabel');
  const n     = paper.pageImages.length;

  // pageNum is 1-indexed; pageImages is 0-indexed
  // Reset zoom whenever we navigate to a new page
  zoomReset();

  if (pageNum && pageNum >= 1 && pageNum <= n) {
    // Normal case — page is within the captured range
    img.src            = paper.pageImages[pageNum - 1];
    img.style.display  = 'block';
    none.style.display = 'none';
    label.textContent  = `Page ${pageNum}`;
  } else if (pageNum && pageNum > n && n > 0) {
    // Page cited by the model is beyond our captured range (document truncated
    // at MAX_PAGES).  Show the last captured page and explain.
    img.src            = paper.pageImages[n - 1];
    img.style.display  = 'block';
    none.style.display = 'none';
    label.textContent  = `Page ${pageNum} (beyond p.\u202f${n} — showing last captured)`;
  } else if (!pageNum && n > 0) {
    img.src            = paper.pageImages[0];
    img.style.display  = 'block';
    none.style.display = 'none';
    label.textContent  = 'Page 1';
  } else {
    img.style.display  = 'none';
    none.style.display = 'block';
    none.innerHTML     = paper.pageImages.length === 0
      ? 'No page preview available.<br>Upload the PDF to see page images.'
      : 'No page reference<br>found in this entry.';
    label.textContent  = 'Page —';
  }
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

function updatePageNav(paper) {
  const pages  = paper.evidencePages;
  const idx    = paper.evidencePageIdx;
  const navEl  = document.getElementById('pageEvidenceNav');
  const prevEl = document.getElementById('pageNavPrev');
  const nextEl = document.getElementById('pageNavNext');
  const cntEl  = document.getElementById('pageNavCounter');

  if (pages.length > 1) {
    navEl.style.display  = 'flex';
    cntEl.textContent    = `${idx + 1}/${pages.length}`;
    prevEl.disabled      = idx === 0;
    nextEl.disabled      = idx === pages.length - 1;
  } else {
    navEl.style.display  = 'none';
  }
}

function prevEvidencePage() {
  const p = getActivePaper();
  if (!p || p.evidencePageIdx === 0) return;
  p.evidencePageIdx--;
  updatePageNav(p);
  showPageImage(p, p.evidencePages[p.evidencePageIdx]);
}

function nextEvidencePage() {
  const p = getActivePaper();
  if (!p || p.evidencePageIdx >= p.evidencePages.length - 1) return;
  p.evidencePageIdx++;
  updatePageNav(p);
  showPageImage(p, p.evidencePages[p.evidencePageIdx]);
}

/* ──────────────────────────────────────────────────────────
   JSON parsing helpers
────────────────────────────────────────────────────────── */

function stripFences(text) {
  // Remove markdown code fences: ```json ... ``` or ``` ... ```
  return text.replace(/^```(?:json)?\s*\n?/i, '').replace(/\n?```\s*$/i, '').trim();
}

function parseFull(text) {
  try { return JSON.parse(stripFences(text)); } catch { return null; }
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
    return `<div class="rv-list">${data.map((item, i) => `
      <div class="rv-list-item">
        <span class="rv-idx">${i + 1}</span>
        <div class="rv-list-body">${renderValueHtml(item, depth + 1, path ? `${path}[${i}]` : `[${i}]`)}</div>
      </div>`).join('')}</div>`;
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
  img.style.transform = `translate(${zoom.x}px,${zoom.y}px) scale(${zoom.scale})`;
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
      const res  = await fetch('/api/pages', { method: 'POST', body: form });
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
    mode: null, provider: 'openai', question: '', context: '', inputMode: 'generate',
    generatedPrompt: '', useTextExtraction: false, selectedFiles: [], papers: [],
    activePaperId: null, loadedFromFile: false, setupReturnStep: null,
  });
  document.getElementById('questionInput').value     = '';
  document.getElementById('contextInput').value      = '';
  document.getElementById('manualPromptInput').value = '';
  showStep3Choice();
  renderFileList();
  cancelLoadOption();
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

let toastTimer = null;
function showToast(message) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('visible'), 4500);
}

// Initialisation is handled by the DOMContentLoaded listener near onProviderChange.
