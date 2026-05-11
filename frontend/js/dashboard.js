const sessionId = localStorage.getItem('session_id');
const agents = ['cv_agent', 'job_agent', 'curriculum_agent', 'plan_agent'];

const agentLabels = {
  cv_agent: '📄 CV Agent',
  job_agent: '💼 Job Agent',
  curriculum_agent: '📚 Curriculum Agent',
  plan_agent: '🗺️ Plan Agent',
};

const cards = document.getElementById('cards');
const logEl = document.getElementById('live-log');

// ── Card helpers ──────────────────────────────────────────────

function ensureCard(name) {
  let el = document.getElementById('card-' + name);
  if (!el) {
    el = document.createElement('div');
    el.id = 'card-' + name;
    el.className = 'card';
    el.innerHTML = `
      <div class="card-header">
        <span class="card-title">${agentLabels[name] || name}</span>
        <span class="card-status" id="status-${name}">waiting</span>
      </div>
      <pre id="output-${name}" class="card-output" style="display:none"></pre>
    `;
    cards.appendChild(el);
  }
}

function setStatus(agent, status) {
  const el = document.getElementById('status-' + agent);
  if (!el) return;
  el.textContent = status;
  el.className = 'card-status status-' + status;
}

function setOutput(agent, data) {
  const pre = document.getElementById('output-' + agent);
  if (!pre) return;
  pre.style.display = 'block';
  pre.textContent = JSON.stringify(data, null, 2);
}

// ── Live log ─────────────────────────────────────────────────

function log(agent, node, message, status) {
  const time = new Date().toLocaleTimeString();
  const label = agentLabels[agent] || agent;
  const line = document.createElement('div');
  line.className = 'log-line log-' + (status || 'info');
  line.innerHTML = `<span class="log-time">${time}</span> <span class="log-agent">${label}</span> <span class="log-node">${node || ''}</span> ${message || ''}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

// ── Init cards ───────────────────────────────────────────────

agents.forEach(a => ensureCard(a));

if (!sessionId) {
  log('system', '', 'No session ID found. Please submit the form first.', 'error');
} else {
  log('system', '', `Connecting to session ${sessionId}…`, 'info');

  const es = new EventSource(`http://127.0.0.1:8000/api/stream/${sessionId}`);

  es.onopen = () => log('system', '', 'Stream connected.', 'info');

  es.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    const { agent, node, message, status, output, markdown, type } = data;

    if (type === 'complete') {
      log('system', '', '✅ Pipeline complete.', 'done');
      es.close();
      return;
    }

    if (agent) {
      ensureCard(agent);

      if (status === 'running') {
        setStatus(agent, 'running');
        log(agent, node, message, 'running');
      }

      if (status === 'done') {
        setStatus(agent, 'done');
        log(agent, '', 'Done.', 'done');
        if (output) setOutput(agent, output);
      }
    }

    if (markdown) {
      document.getElementById('report').innerHTML = marked.parse(markdown);
      document.getElementById('report-section').style.display = 'block';
    }
  };

  es.onerror = () => {
    log('system', '', 'Stream error or closed.', 'error');
    es.close();
  };
}