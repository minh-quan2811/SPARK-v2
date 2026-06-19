/**
 * form.js
 * Handles career planning form submission.
 * Posts multipart/form-data to the backend, stores the
 * returned session_id, then redirects to the dashboard.
 */

'use strict';

(function () {
  const form = document.getElementById('sparkForm');
  if (!form) return;

  let submitting = false;

  /* Builds the plan_preferences string sent to the backend */
  function buildPlanPreferences() {
    const duration = document.getElementById('duration').value.trim();
    const dims = typeof getDimensionSelections === 'function'
      ? getDimensionSelections()
      : {};

    const dimLine = Object.entries(dims)
      .map(([key, value]) => `${key}=${value}`)
      .join(',');

    const parts = [];
    if (duration) parts.push(duration);
    if (dimLine)  parts.push(`dimensions: ${dimLine}`);

    return parts.join('\n\n');
  }

  form.addEventListener('submit', async e => {
    e.preventDefault();
    e.stopImmediatePropagation();

    if (submitting) return;
    submitting = true;

    const btn = form.querySelector('.submit-btn');
    btn.disabled    = true;
    btn.textContent = 'Submitting…';

    try {
      const data = new FormData(form);

      // Remove the raw 'duration' field name (not a backend field) and
      // replace plan_preferences with the assembled string.
      data.delete('duration');
      data.set('plan_preferences', buildPlanPreferences());

      const res = await fetch('http://127.0.0.1:8000/api/submit', {
        method: 'POST',
        body:   data,
      });

      if (!res.ok) throw new Error(`Server error ${res.status}`);

      const json = await res.json();

      if (!json.session_id) throw new Error('No session_id returned from server.');

      localStorage.setItem('session_id', json.session_id);
      window.location.href = 'dashboard.html';

    } catch (err) {
      submitting      = false;
      btn.disabled    = false;
      btn.textContent = 'Generate Career Roadmap';
      alert(`Submission failed: ${err.message}`);
    }
  });
}());