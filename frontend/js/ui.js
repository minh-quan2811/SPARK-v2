/**
 * ui.js
 * Low-level DOM helpers for the dashboard:
 *   ensureSection   — creates agent panel if missing
 *   chipState       — updates sidebar chip
 *   addThought      — appends a thought row to a stream
 *   focusAgent      — scrolls main panel to an agent section
 */

'use strict';

/* ── DOM shortcuts ── */
const esc = s => String(s || '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

const now = () =>
  new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

/* ── Section builder ─────────────────────────────────── */
function ensureSection(agent) {
  if (document.getElementById('sec-' + agent)) return;

  const meta = AGENT_META[agent] || { label: agent, icon: 'ti-robot' };
  const sec  = document.createElement('div');
  sec.className   = 'agent-section';
  sec.id          = 'sec-' + agent;
  sec.dataset.agent = agent;
  sec.innerHTML = `
    <div class="asec-header">
      <i class="ti ${meta.icon} asec-icon" aria-hidden="true"></i>
      <span class="asec-title">${esc(meta.label)}</span>
      <span class="asec-state" id="astate-${agent}">running…</span>
    </div>
    <div class="asec-body">
      <div class="stream" id="stream-${agent}"></div>
      <div id="out-${agent}"></div>
    </div>`;

  document.getElementById('main').appendChild(sec);
}

/* ── Sidebar chip state ──────────────────────────────── */
function chipState(agent, state, step) {
  const dot   = document.getElementById('cdot-'   + agent);
  const badge = document.getElementById('cbadge-' + agent);
  const cstep = document.getElementById('cstep-'  + agent);
  if (!dot) return;

  dot.className   = 'cdot ' + state;
  badge.className = 'cbadge ' + state;
  badge.textContent = state === 'run' ? 'running' : state === 'done' ? 'done' : 'wait';
  if (step) cstep.textContent = step;
}

/* ── Add thought row ─────────────────────────────────── */
function addThought(agent, node, message) {
  const stream = document.getElementById('stream-' + agent);
  if (!stream) return;

  const type  = NODE_TYPE[node]  || 'step';
  const label = NODE_LABEL[node] || node || '';
  const icon  = THOUGHT_ICON[type] || 'ti-arrow-right';

  const el = document.createElement('div');
  el.className = `thought ${type}`;
  el.innerHTML = `
    <i class="ti ${icon} t-icon" aria-hidden="true"></i>
    <div class="t-body">
      ${label ? `<div class="t-label">${esc(label)}</div>` : ''}
      <div class="t-text">${esc(message || '')}</div>
    </div>
    <span class="t-time">${now()}</span>`;
  stream.appendChild(el);
}

/* ── Focus / scroll to agent ─────────────────────────── */
function focusAgent(agent) {
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  document.getElementById('chip-' + agent)?.classList.add('active');

  const sec   = document.getElementById('sec-'  + agent);
  const panel = document.getElementById('main');
  if (sec && panel) panel.scrollTop = sec.offsetTop - 20;
}