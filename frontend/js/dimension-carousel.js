/**
 * dimension-carousel.js
 * Drives the 6 roadmap-style dimensions as a single paginated carousel.
 * One dimension is shown at a time as two choice-cards; arrows page
 * between dimensions; selections persist when navigating back and forth.
 */

'use strict';

const DIMENSIONS = [
  {
    key: 'coverage',
    name: 'Coverage',
    left:  { value: 'focused', label: 'Focused',
      desc: 'Only subjects that directly close your skill gaps — a concise, targeted roadmap.' },
    right: { value: 'broad', label: 'Broad',
      desc: 'Adjacent and supplementary subjects are included too, building transferable skills beyond your immediate target.' },
  },
  {
    key: 'prior_knowledge',
    name: 'Prior Knowledge',
    left:  { value: 'skip_known', label: 'Skip Known',
      desc: "Subjects you've already shown competency in on your CV are left out — no redundancy." },
    right: { value: 'full_coverage', label: 'Full Coverage',
      desc: 'Foundational subjects stay in the plan regardless of prior exposure, so no gap is left unaddressed.' },
  },
  {
    key: 'sequence',
    name: 'Learning Sequence',
    left:  { value: 'prereq_first', label: 'Prereqs First',
      desc: 'Every foundational course is completed before any applied or project-based subject.' },
    right: { value: 'early_exposure', label: 'Early Exposure',
      desc: 'Applied courses can appear earlier, even with some prerequisites still pending, to stay practical and motivating sooner.' },
  },
  {
    key: 'pace',
    name: 'Study Pace',
    left:  { value: 'fast_track', label: 'Fast-Track',
      desc: 'The roadmap is compressed into fewer semesters with a heavier load each term.' },
    right: { value: 'spaced', label: 'Spaced',
      desc: 'The same content is spread across more semesters at a lighter load, giving you more time to absorb each topic.' },
  },
  {
    key: 'theory_practice',
    name: 'Theory–Practice',
    left:  { value: 'theory_first', label: 'Theory-First',
      desc: 'Formal coursework and lectures lead the plan, with hands-on work supporting them.' },
    right: { value: 'project_first', label: 'Project-First',
      desc: 'Practical work, labs, and applied certifications lead the plan, with theory courses playing a supporting role.' },
  },
  {
    key: 'structure',
    name: 'Schedule Structure',
    left:  { value: 'structured', label: 'Structured',
      desc: 'A fixed semester-by-semester plan with an explicit course assignment for each term.' },
    right: { value: 'flexible', label: 'Flexible',
      desc: 'A suggested ordering without courses locked to specific semesters, so you can adjust based on availability and workload.' },
  },
];

const _selections = {};
let _current = 0;

/* Reads current selections → { coverage: 'focused', pace: 'spaced', ... } */
function getDimensionSelections() {
  const out = {};
  DIMENSIONS.forEach(dim => {
    out[dim.key] = _selections[dim.key] || dim.left.value;
  });
  return out;
}

(function () {
  const root      = document.getElementById('dimCarousel');
  if (!root) return;

  const nameEl    = document.getElementById('dimName');
  const progEl    = document.getElementById('dimProgress');
  const dotsEl    = document.getElementById('dimDots');
  const prevBtn   = document.getElementById('dimPrev');
  const nextBtn   = document.getElementById('dimNext');

  const leftCard  = document.getElementById('dimCardLeft');
  const rightCard = document.getElementById('dimCardRight');
  const leftLabel = document.getElementById('dimCardLeftLabel');
  const leftDesc  = document.getElementById('dimCardLeftDesc');
  const rightLabel = document.getElementById('dimCardRightLabel');
  const rightDesc  = document.getElementById('dimCardRightDesc');

  /* Build progress dots once */
  DIMENSIONS.forEach((dim, i) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'dim-dot';
    dot.setAttribute('aria-label', dim.name);
    dot.addEventListener('click', () => goTo(i));
    dotsEl.appendChild(dot);
  });

  function render() {
    const dim = DIMENSIONS[_current];
    const selected = _selections[dim.key];

    nameEl.textContent = dim.name;
    progEl.textContent = `${_current + 1} / ${DIMENSIONS.length}`;

    leftLabel.textContent  = dim.left.label;
    leftDesc.textContent   = dim.left.desc;
    rightLabel.textContent = dim.right.label;
    rightDesc.textContent  = dim.right.desc;

    leftCard.classList.toggle('selected', selected === dim.left.value);
    rightCard.classList.toggle('selected', selected === dim.right.value);

    prevBtn.disabled = _current === 0;
    nextBtn.disabled = _current === DIMENSIONS.length - 1;

    [...dotsEl.children].forEach((dot, i) => {
      dot.classList.toggle('current', i === _current);
      dot.classList.toggle('answered', Boolean(_selections[DIMENSIONS[i].key]));
    });
  }

  function goTo(index) {
    _current = Math.max(0, Math.min(DIMENSIONS.length - 1, index));
    render();
  }

  function select(side) {
    const dim = DIMENSIONS[_current];
    _selections[dim.key] = dim[side].value;
    render();
  }

  leftCard.addEventListener('click', () => select('left'));
  rightCard.addEventListener('click', () => select('right'));
  prevBtn.addEventListener('click', () => goTo(_current - 1));
  nextBtn.addEventListener('click', () => goTo(_current + 1));

  render();
}());