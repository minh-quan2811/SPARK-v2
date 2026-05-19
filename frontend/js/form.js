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

  let submitting = false;          // ← guard against double-submit

  form.addEventListener('submit', async e => {
    e.preventDefault();
    e.stopImmediatePropagation();  // ← block any other listeners

    if (submitting) return;        // ← already in-flight, bail out
    submitting = true;

    const btn = form.querySelector('.submit-btn');
    btn.disabled    = true;
    btn.textContent = 'Submitting…';

    try {
      const data = new FormData(form);

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
      submitting      = false;     // ← allow retry on failure
      btn.disabled    = false;
      btn.textContent = 'Generate Career Roadmap';
      alert(`Submission failed: ${err.message}`);
    }
  });
}());