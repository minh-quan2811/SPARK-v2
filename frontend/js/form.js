document.getElementById('sparkForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const btn = e.target.querySelector('button[type="submit"]');
  btn.disabled = true;
  btn.textContent = 'Submitting…';

  try {
    const formData = new FormData(e.target);
    const res = await fetch('http://127.0.0.1:8000/api/submit', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();

    // Write to localStorage first, then navigate
    localStorage.setItem('session_id', data.session_id);

    // Small tick to guarantee the write is visible before navigation
    await new Promise((resolve) => setTimeout(resolve, 0));

    location.href = 'dashboard.html';
  } catch (err) {
    console.error('Submit failed:', err);
    btn.disabled = false;
    btn.textContent = 'Submit';
    alert(`Submission failed: ${err.message}`);
  }
});