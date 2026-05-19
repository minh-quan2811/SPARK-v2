/**
 * faculty-dropdown.js
 * Searchable faculty dropdown for the career planning form.
 * Loads options from assets/ten_chuong_trinh.txt relative to the page.
 */

(function () {
  'use strict';

  const input    = document.getElementById('facultyInput');
  const dropdown = document.getElementById('facultyDropdown');

  if (!input || !dropdown) return;

  let options = [];

  /* ── Load options from flat text file ── */
  fetch('assets/ten_chuong_trinh.txt')
    .then(res => res.text())
    .then(text => {
      options = text
        .split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 0);
    })
    .catch(() => {
      /* Silently fail — user can still type manually */
    });

  /* ── Filter and render matches ── */
  function renderDropdown(query) {
    dropdown.innerHTML = '';

    if (!query) {
      dropdown.style.display = 'none';
      return;
    }

    const matches = options.filter(opt =>
      opt.toLowerCase().includes(query.toLowerCase())
    );

    if (!matches.length) {
      dropdown.style.display = 'none';
      return;
    }

    matches.slice(0, 100).forEach(match => {
      const item = document.createElement('div');
      item.className   = 'dropdown-item';
      item.textContent = match;
      item.addEventListener('click', () => {
        input.value            = match;
        dropdown.style.display = 'none';
      });
      dropdown.appendChild(item);
    });

    dropdown.style.display = 'block';
  }

  /* ── Events ── */
  input.addEventListener('input', () => renderDropdown(input.value.trim()));

  document.addEventListener('click', e => {
    if (!e.target.closest('.dropdown')) {
      dropdown.style.display = 'none';
    }
  });
}());