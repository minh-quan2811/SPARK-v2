/**
 * renderers.js
 * Pure functions that build HTML strings for each agent's output.
 *   renderCV(data)         → CV extraction card
 *   renderJobs(data)       → Job listings card
 *   renderCurriculum(data) → Curriculum / Cypher card
 *
 * Each function returns an HTML string — call innerHTML on the
 * target container to render.
 */

'use strict';

/* ── CV renderer ──────────────────────────────────────── */
function renderCV(data) {
  const edu  = data.education        || {};
  const tech = data.technical_skills || [];
  const soft = data.soft_skills      || [];
  const cert = data.certifications   || [];
  const exps = data.experience       || [];
  const proj = data.projects         || [];

  let h = `<div class="out-card">
    <div class="out-card-hd">
      <i class="ti ti-file-description" style="color:var(--cv);font-size:14px" aria-hidden="true"></i>
      <span class="out-card-title">Extracted CV Data</span>
    </div>
    <div class="out-card-body">`;

  /* Education */
  if (edu.degree || edu.major || edu.gpa || edu.graduation_year || edu.academic_year) {
    h += `<div class="frow"><span class="fkey">Education</span><div>`;
    if (edu.degree || edu.major)
      h += `<div class="fval">${esc([edu.degree, edu.major].filter(Boolean).join(' · '))}</div>`;
    if (edu.gpa)
      h += `<div style="font-size:11px;color:var(--muted2)">GPA: ${esc(edu.gpa)}</div>`;
    if (edu.academic_year)
      h += `<div style="font-size:11px;color:var(--muted2)">Year ${esc(edu.academic_year)} of study</div>`;
    if (edu.graduation_year)
      h += `<div style="font-size:11px;color:var(--muted2)">Graduating ${esc(edu.graduation_year)}</div>`;
    h += `</div></div>`;
  }

  /* Technical skills */
  if (tech.length) {
    h += `<div class="frow"><span class="fkey">Technical skills</span>
      <div class="tags">${tech.map(s => `<span class="tag skill">${esc(s)}</span>`).join('')}</div>
    </div>`;
  }

  /* Soft skills */
  if (soft.length) {
    h += `<div class="frow"><span class="fkey">Soft skills</span>
      <div class="tags">${soft.map(s => `<span class="tag soft">${esc(s)}</span>`).join('')}</div>
    </div>`;
  }

  /* Experience */
  if (exps.length) {
    h += `<hr class="divd"><div class="section-title-sm">Work Experience (${exps.length})</div>`;
    exps.forEach(e => {
      const skills = (e.skills_used || []).slice(0, 6);
      h += `<div class="xblock">
        <div class="xblock-title">${esc(e.position || 'Role')}</div>
        <div class="xblock-sub">${esc(e.company || '')}${e.duration ? ' · ' + esc(e.duration) : ''}</div>
        ${e.description ? `<div class="xblock-desc">${esc(e.description)}</div>` : ''}
        ${skills.length ? `<div class="tags" style="margin-top:4px">${skills.map(s => `<span class="tag skill">${esc(s)}</span>`).join('')}</div>` : ''}
      </div>`;
    });
  }

  /* Projects */
  if (proj.length) {
    h += `<hr class="divd"><div class="section-title-sm">Projects (${proj.length})</div>`;
    proj.forEach(p => {
      const skills = (p.skills_used || []).slice(0, 6);
      h += `<div class="xblock">
        <div class="xblock-title">${esc(p.name || 'Project')}</div>
        ${p.description ? `<div class="xblock-desc">${esc(p.description)}</div>` : ''}
        ${skills.length ? `<div class="tags" style="margin-top:4px">${skills.map(s => `<span class="tag skill">${esc(s)}</span>`).join('')}</div>` : ''}
      </div>`;
    });
  }

  /* Certifications */
  if (cert.length) {
    h += `<hr class="divd">
      <div class="frow"><span class="fkey">Certifications</span>
        <div class="tags">${cert.map(c => `<span class="tag">${esc(c)}</span>`).join('')}</div>
      </div>`;
  }

  h += `</div></div>`;
  return h;
}

/* ── Jobs renderer ────────────────────────────────────── */
function renderJobs(data) {
  const jobs    = data.jobs           || [];
  const titles  = data.top_job_titles || [];
  const summary = data.summary        || '';

  let h = `<div class="out-card">
    <div class="out-card-hd">
      <i class="ti ti-briefcase" style="color:var(--job);font-size:14px" aria-hidden="true"></i>
      <span class="out-card-title">Job Search Results — ${jobs.length} listings</span>
    </div>
    <div class="out-card-body">`;

  if (summary) {
    h += `<div style="font-size:12px;line-height:1.7;padding:10px 12px;
      background:var(--surface3);border:1px solid var(--border);border-radius:6px;
      color:var(--muted2)">${esc(summary)}</div>`;
  }

  if (titles.length) {
    h += `<div class="frow"><span class="fkey">Top titles</span>
      <div class="tags">${titles.slice(0, 8).map(t => `<span class="tag jtitle">${esc(t)}</span>`).join('')}</div>
    </div>`;
  }

  if (jobs.length) h += `<hr class="divd"><div class="section-title-sm">All Listings</div>`;

  jobs.forEach(j => {
    const tech = j.technical_skills  || [];
    const reqs = j.requirements      || [];
    const resp = j.responsibilities  || [];
    h += `<div class="job-card">
      <div>
        <span class="job-card-title">${esc(j.title   || 'Untitled')}</span>
        <span class="job-card-company">${esc(j.company || '')}</span>
      </div>
      <div class="job-meta">
        ${j.location            ? `<span class="tag"><i class="ti ti-map-pin" style="font-size:10px"></i> ${esc(j.location)}</span>` : ''}
        ${j.seniority           ? `<span class="tag">${esc(j.seniority)}</span>`        : ''}
        ${j.employment_type     ? `<span class="tag">${esc(j.employment_type)}</span>`  : ''}
        ${j.years_of_experience ? `<span class="tag">${esc(j.years_of_experience)}</span>` : ''}
        ${j.remote === true     ? `<span class="tag soft">Remote</span>`                : ''}
      </div>
      ${tech.length ? `<div class="tags">${tech.map(s => `<span class="tag skill">${esc(s)}</span>`).join('')}</div>` : ''}
      ${reqs.length ? `<div class="job-reqs">
        <div class="section-title-sm">Requirements</div>
        <ul>${reqs.slice(0, 5).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </div>` : ''}
      ${resp.length ? `<div class="job-resp">
        <div class="section-title-sm">Responsibilities</div>
        <ul>${resp.slice(0, 5).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      </div>` : ''}
      ${j.salary    ? `<div class="job-salary">Salary: ${esc(j.salary)}</div>` : ''}
      ${j.apply_url ? `<a class="job-url" href="${esc(j.apply_url)}" target="_blank" rel="noopener">
        Apply <i class="ti ti-external-link" style="font-size:11px"></i>
      </a>` : ''}
    </div>`;
  });

  h += `</div></div>`;
  return h;
}

/* ── Curriculum renderer ──────────────────────────────── */
function renderCurriculum(data) {
  const records = data.database_records || data.records || [];
  const cypher  = data.cypher_statement || '';
  const errors  = data.errors           || [];

  let h = `<div class="out-card">
    <div class="out-card-hd">
      <i class="ti ti-school" style="color:var(--cur);font-size:14px" aria-hidden="true"></i>
      <span class="out-card-title">Curriculum — ${records.length} courses found</span>
    </div>
    <div class="out-card-body">`;

  if (cypher) {
    h += `<div class="section-title-sm">Generated Cypher Query</div>
      <pre class="cypher">${esc(cypher)}</pre>`;
  }

  if (records.length) {
    h += `<div class="section-title-sm">Courses</div><div class="tags">`;
    records.forEach(r => {
      const label = r.name || Object.values(r).find(v => typeof v === 'string' && v.length < 150 && v !== r.program);
      if (label) h += `<span class="tag course">${esc(label)}</span>`;
    });
    h += `</div>`;

    h += `<div class="section-title-sm" style="margin-top:6px">Record details (first 3)</div>`;
    records.slice(0, 3).forEach(r => {
      h += `<div class="xblock">`;
      Object.entries(r).forEach(([k, v]) => {
        if (v == null || v === '') return;
        h += `<div class="frow">
          <span class="fkey" style="min-width:100px">${esc(k)}</span>
          <span class="fval" style="font-size:11px">${esc(String(v))}</span>
        </div>`;
      });
      h += `</div>`;
    });
  } else {
    h += `<div class="empty"><i class="ti ti-info-circle"></i> No courses found.</div>`;
  }

  if (errors.length) {
    h += `<div style="color:var(--red);font-size:11px;padding:8px 10px;
      background:rgba(192,57,43,.07);border:1px solid rgba(192,57,43,.2);border-radius:6px">
      ${esc(errors.join(', '))}</div>`;
  }

  h += `</div></div>`;
  return h;
}