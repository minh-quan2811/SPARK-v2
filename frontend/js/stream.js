/**
 * stream.js
 * SSE controller — connects to the backend event stream and
 * drives all dashboard UI updates.
 *
 * Depends on (loaded before this file):
 *   constants.js  — AGENT_META, NODE_TYPE, NODE_LABEL, THOUGHT_ICON
 *   ui.js         — ensureSection, chipState, addThought, focusAgent, esc
 *   renderers.js  — renderCV, renderJobs, renderCurriculum
 */

'use strict';

(function () {

  /* ── Session ID ─────────────────────────────────────────── */
  const sessionId = localStorage.getItem('session_id');

  const hdrDot = document.getElementById('hdr-dot');
  const hdrTxt = document.getElementById('hdr-txt');

  function setHeader(state, text) {
    hdrDot.className = 'dot ' + state;
    hdrTxt.textContent = text;
  }

  if (!sessionId) {
    setHeader('err', 'No session — please submit the form first.');
    return;
  }

  /* ── Connect ────────────────────────────────────────────── */
  setHeader('live', 'Connecting…');

  const es = new EventSource(`http://127.0.0.1:8000/api/stream/${sessionId}`);

  es.onopen = () => setHeader('live', 'Pipeline running…');

  /* ── Message handler ────────────────────────────────────── */
  es.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    // markdown may be top-level on the event, or nested inside output
    const { agent, node, message, status, output, markdown, type } = data;

    /* ── Pipeline complete ── */
    if (type === 'complete') {
      setHeader('done', 'Pipeline complete');
      es.close();
      return;
    }

    /* ── Agent events ── */
    if (!agent) return;

    ensureSection(agent);

    /* Running — show thought bubble */
    if (status === 'running') {
      chipState(agent, 'run', message || node || '');
      addThought(agent, node, message);

      const stateEl = document.getElementById('astate-' + agent);
      if (stateEl) stateEl.textContent = message || 'running…';
    }

    /* Done — render output card */
    if (status === 'done') {
      chipState(agent, 'done', 'Done');

      const stateEl = document.getElementById('astate-' + agent);
      if (stateEl) stateEl.textContent = 'done';

      const outEl = document.getElementById('out-' + agent);
      if (outEl) {
        let rendered = false;
        try {
          if (agent === 'plan_agent') {
            // plan can arrive as: top-level `markdown`, inside `output` as string/object
            const planContent = markdown || output;
            if (planContent) {
              renderPlan(outEl, planContent);
              rendered = true;
            }
          } else if (output) {
            const parsed = typeof output === 'string' ? JSON.parse(output) : output;
            if      (agent === 'cv_agent')         outEl.innerHTML = renderCV(parsed);
            else if (agent === 'job_agent')        outEl.innerHTML = renderJobs(parsed);
            else if (agent === 'curriculum_agent') outEl.innerHTML = renderCurriculum(parsed);
            rendered = true;
          }
        } catch (err) {
          outEl.innerHTML = `<div class="empty"><i class="ti ti-info-circle"></i> Output received (unparseable).</div>`;
          rendered = true;
        }

        /* done banner — only prepended when something actually rendered */
        if (rendered) {
          const banner = document.createElement('div');
          banner.className = 'done-banner';
          banner.innerHTML = `<i class="ti ti-circle-check"></i> ${esc(AGENT_META[agent]?.label || agent)} completed successfully.`;
          outEl.prepend(banner);
        }
      }
    }

    /* Error */
    if (status === 'error') {
      chipState(agent, '', 'Error');

      const stateEl = document.getElementById('astate-' + agent);
      if (stateEl) stateEl.textContent = 'error';

      addThought(agent, 'agent_error', message || 'Unknown error');
    }
  };

  /* ── Stream error ───────────────────────────────────────── */
  es.onerror = () => {
    setHeader('err', 'Stream disconnected');
    es.close();
  };

  /* ── Plan agent renderer (markdown) ────────────────────── */
  function renderPlan(container, rawOutput) {
    let markdown = '';

    if (typeof rawOutput === 'string') {
      markdown = rawOutput;
    } else if (rawOutput && typeof rawOutput === 'object') {
      // try common field names the backend might use
      markdown = rawOutput.plan
               || rawOutput.markdown
               || rawOutput.content
               || rawOutput.roadmap
               || rawOutput.text
               || JSON.stringify(rawOutput, null, 2);
    }

    const card = document.createElement('div');
    card.className = 'out-card';
    card.innerHTML = `
      <div class="out-card-hd">
        <i class="ti ti-map asec-icon" style="color:var(--plan);font-size:14px" aria-hidden="true"></i>
        <span class="out-card-title">Career Roadmap</span>
      </div>
      <div class="out-card-body">
        <div class="roadmap">${typeof marked !== 'undefined' ? marked.parse(markdown) : esc(markdown)}</div>
      </div>`;
    container.appendChild(card);
  }

}());