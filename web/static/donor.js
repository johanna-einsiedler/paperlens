/* Donor flow — the modal that asks "want to share this dataset?" after
   Download All, and the POST /api/donate handler.

   Loaded LAST in index.html so app.js's helpers (showToast, fetchScoped,
   state) are already in scope.  The feature flag is read from
   ``window.__PAPERLENS_CONFIG__.donate`` (populated by loadServerConfig
   in app.js); when ``enabled`` is false the modal is never offered.

   State stored on ``window`` so it survives across donor-modal opens and
   so the dev console can poke at it during debugging. */

(function () {
  // Persistent: "this batch was successfully donated, never offer again".
  const DONATED_KEY  = 'paperlens.donated.v1';            // localStorage  — map batchId -> true
  // Session-only: "this batch's modal was dismissed without donating,
  // don't pop again for it this session".  Cleared by a new browser
  // session so the user is invited again later if they reopen the app.
  const DISMISSED_KEY = 'paperlens.donate.dismissedBatches.v1';  // sessionStorage — map batchId -> true

  function _donateCfg() {
    return (window.__PAPERLENS_CONFIG__ && window.__PAPERLENS_CONFIG__.donate) || {};
  }

  function _readMap(store, key) {
    try { return JSON.parse(store.getItem(key) || '{}'); }
    catch (_) { return {}; }
  }

  function _writeMap(store, key, batchId) {
    try {
      const map = _readMap(store, key);
      if (batchId) map[batchId] = true;
      store.setItem(key, JSON.stringify(map));
    } catch (_) { /* private mode or quota */ }
  }

  function _alreadyDonated(batchId) {
    return !!(batchId && _readMap(localStorage, DONATED_KEY)[batchId]);
  }

  function _markDonated(batchId) {
    _writeMap(localStorage, DONATED_KEY, batchId);
  }

  function _dismissedThisBatch(batchId) {
    return !!(batchId && _readMap(sessionStorage, DISMISSED_KEY)[batchId]);
  }

  function _markDismissed(batchId) {
    _writeMap(sessionStorage, DISMISSED_KEY, batchId);
  }

  /* Offer the donation modal after a successful download.  Called from
     downloadAllPapers / downloadAllPapersCsv in app.js.  Bails silently
     when the flag is off, the batch has already been donated, the user
     already dismissed it for this batch, or there's no batch id to
     attach the donation to.  When all checks pass, opens the modal
     directly — the dimmed-backdrop overlay is the affordance, no
     separate toast.  Done on a short delay so the browser's download
     save-dialog (if any) lands before the overlay takes focus. */
  function donorMaybeOffer(batchId) {
    const cfg = _donateCfg();
    if (!cfg.enabled) return;
    if (!batchId) return;
    if (_alreadyDonated(batchId)) return;
    if (_dismissedThisBatch(batchId)) return;
    setTimeout(() => openDonateModal(batchId), 400);
  }

  /* Open the modal in its initial form state.  Idempotent — re-opening
     clears previous error messages and resets the success-stage view. */
  function openDonateModal(batchId) {
    const overlay = document.getElementById('donateOverlay');
    if (!overlay) return;
    window.__DONATE_BATCH_ID__ = batchId || (window.state && state.batchId) || '';
    document.getElementById('donateError').style.display = 'none';
    document.getElementById('donateFormStage').style.display = '';
    document.getElementById('donateSuccessStage').style.display = 'none';
    overlay.style.display = 'flex';
    document.getElementById('donateTitleInput').focus();
  }

  function closeDonateModal() {
    const overlay = document.getElementById('donateOverlay');
    if (overlay) overlay.style.display = 'none';
    // Per-batch dismissal — don't re-offer for THIS batch this session.
    // Other batches still get offered; a new browser session resets.
    // Successful donations call _markDonated separately (persistent),
    // so the "Done" button here is fine to also mark dismissed.
    if (window.__DONATE_BATCH_ID__) _markDismissed(window.__DONATE_BATCH_ID__);
  }

  function _donateToggleAttribution() {
    const mode = (document.querySelector('input[name="donateAttribution"]:checked') || {}).value;
    document.getElementById('donateAttributedFields').style.display =
      mode === 'attributed' ? '' : 'none';
  }

  function _donateToggleVisibility() {
    const mode = (document.querySelector('input[name="donateVisibility"]:checked') || {}).value;
    document.getElementById('donateGatedFields').style.display =
      mode === 'gated' ? '' : 'none';
  }

  function _showDonateError(msg) {
    const el = document.getElementById('donateError');
    el.textContent = msg;
    el.style.display = '';
  }

  async function submitDonateModal() {
    const title = document.getElementById('donateTitleInput').value.trim();
    if (!title) { _showDonateError('Dataset title is required.'); return; }

    const description = document.getElementById('donateDescInput').value.trim();
    const attrMode    = (document.querySelector('input[name="donateAttribution"]:checked') || {}).value || 'anonymous';
    const name        = document.getElementById('donateNameInput').value.trim();
    const affiliation = document.getElementById('donateAffiliationInput').value.trim();
    if (attrMode === 'attributed' && !name) {
      _showDonateError('Attributed donations need a name.'); return;
    }

    const visMode  = (document.querySelector('input[name="donateVisibility"]:checked') || {}).value || 'public';
    const password = document.getElementById('donatePasswordInput').value;
    if (visMode === 'gated' && password.length < 8) {
      _showDonateError('Gated datasets need a password of at least 8 characters.'); return;
    }

    const shareOk   = document.getElementById('donateConsentShare').checked;
    const licenseOk = document.getElementById('donateConsentLicense').checked;
    if (!shareOk || !licenseOk) {
      _showDonateError('Please confirm both consent checkboxes.'); return;
    }

    const batchId = window.__DONATE_BATCH_ID__;
    if (!batchId) { _showDonateError('No batch id — cannot donate.'); return; }

    const btn = document.getElementById('donateSubmitBtn');
    btn.disabled = true;
    btn.textContent = 'Submitting…';
    document.getElementById('donateError').style.display = 'none';

    try {
      const res = await fetch('/api/donate', {
        method:  'POST',
        headers: {'Content-Type': 'application/json'},
        body:    JSON.stringify({
          batch_id:     batchId,
          title:        title,
          description:  description,
          attribution:  {mode: attrMode, name: name, affiliation: affiliation},
          visibility:   {mode: visMode, password: password},
          consents:     {sharing_rights: shareOk, license_cc_by_4: licenseOk},
        }),
      });
      if (!res.ok) {
        // Try JSON first (FastAPI's HTTPException(detail=...) format),
        // fall back to raw text so we always log SOMETHING the user can act on.
        const ct   = res.headers.get('content-type') || '';
        const raw  = await res.text();
        let detail = `Server returned HTTP ${res.status}.`;
        if (ct.includes('application/json')) {
          try {
            const body = JSON.parse(raw);
            detail = body.detail || body.error || detail;
          } catch (_) { /* fall through */ }
        } else if (raw && raw.length < 300) {
          detail = raw;
        }
        // Log the full response to the browser console so the dev-tools
        // user can read tracebacks even when the modal just shows the
        // summary line.
        console.error('[donate] /api/donate failed', res.status, ct, raw);
        _showDonateError(
          detail +
          (res.status >= 500 ? '  (Check the terminal where you ran python server.py for the traceback.)' : '')
        );
        return;
      }
      const data = await res.json();
      _markDonated(batchId);
      _renderDonateSuccess(data);
    } catch (err) {
      console.error('[donate] network error', err);
      _showDonateError(err.message || 'Network error.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Share dataset';
    }
  }

  function _renderDonateSuccess(data) {
    document.getElementById('donateFormStage').style.display    = 'none';
    document.getElementById('donateSuccessStage').style.display = '';
    const text = document.getElementById('donateSuccessText');

    if (data.mode === 'live' && data.pr_url) {
      // GitHub PR link is always present in live mode.  Zenodo URL is
      // optional (present when the server has PAPERLENS_ZENODO_TOKEN set
      // AND the deposit succeeded); a zenodo_error is shown when the
      // token was set but the deposit failed (the PR still went through).
      let html =
        `<strong>Dataset submitted.</strong>  A pull request has been opened ` +
        `for human review:<br>` +
        `<a href="${data.pr_url}" target="_blank" rel="noopener" style="word-break:break-all">${data.pr_url}</a>`;

      if (data.zenodo_html_url) {
        html +=
          `<br><br><strong>Draft Zenodo deposit:</strong> review and publish ` +
          `when you're ready to mint the DOI &mdash;<br>` +
          `<a href="${data.zenodo_html_url}" target="_blank" rel="noopener" style="word-break:break-all">${data.zenodo_html_url}</a>`;
        if (data.zenodo_doi) {
          html += `<br><span style="font-size:12px;color:var(--text-muted)">Pre-reserved DOI: <code>${data.zenodo_doi}</code> (live after publish)</span>`;
        }
      } else if (data.zenodo_error) {
        html +=
          `<br><br><span style="color:#92400e;font-size:13px">⚠ Zenodo deposit failed: ${escHtmlSafe(data.zenodo_error)}.  ` +
          `The GitHub PR is unaffected; you can mint a DOI manually on Zenodo later.</span>`;
      }

      html +=
        `<br><br><span style="font-size:12px;color:var(--text-muted)">` +
        `Dataset ID: <code>${data.dataset_id}</code> &middot; ` +
        `Papers: ${data.paper_count}</span>`;
      text.innerHTML = html;
    } else if (data.mode === 'dry-run') {
      text.innerHTML =
        `<strong>Bundle built (dry-run mode).</strong>  The donation was ` +
        `staged locally at <code>${data.bundle_path}</code> &mdash; no PR ` +
        `was opened.  Set <code>PAPERLENS_DONATE_LIVE=1</code> to enable ` +
        `live submissions.` +
        `<br><span style="font-size:12px;color:var(--text-muted)">` +
        `Dataset ID: <code>${data.dataset_id}</code> &middot; ` +
        `Papers: ${data.paper_count}</span>`;
    } else {
      text.textContent = 'Submitted.';
    }
  }

  // Local tiny escape so an error string can't smuggle HTML into the
  // success panel.  Mirrors app.js escHtml without depending on its
  // load order.
  function escHtmlSafe(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // Clean up the pre-refactor global 24h-dismiss key.  Earlier versions
  // stored a unix timestamp at paperlens.donate.dismissedAt.v1 that
  // silenced the modal across all batches; the new logic uses a
  // per-batch sessionStorage map instead.  Remove the stale entry on
  // first load so users who closed the modal during yesterday's
  // debugging start fresh.
  try { localStorage.removeItem('paperlens.donate.dismissedAt.v1'); } catch (_) {}

  // Expose the public entry points on window for the inline handlers in
  // index.html and the call from app.js.downloadAllPapers.
  window.openDonateModal   = openDonateModal;
  window.closeDonateModal  = closeDonateModal;
  window.submitDonateModal = submitDonateModal;
  window.donorMaybeOffer   = donorMaybeOffer;
  window._donateToggleAttribution = _donateToggleAttribution;
  window._donateToggleVisibility  = _donateToggleVisibility;
  // Debug helper: force-open the modal against the current batch from
  // the dev console.  Bypasses the offer-gate (enabled / donated /
  // dismissed checks) so you can verify the markup + form even when
  // the feature flag is off or you've already donated.
  window.donateNow = function () {
    const batchId = (window.state && state.batchId) || window.__DONATE_BATCH_ID__ || '';
    openDonateModal(batchId);
  };
})();
