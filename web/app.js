import { predictMatch } from './model.js';
import { distributionChart, calibrationChart, hideTip } from './charts.js';

const TARGETS = ['eng.1', 'esp.1', 'ita.1', 'ger.1', 'fra.1', 'uefa.champions'];
const VIEWS = ['fixtures', 'results', 'tool', 'teams', 'model'];
const RESULT_LINES = { goals: ['1.5', '2.5', '3.5'], corners: ['8.5', '9.5', '10.5', '11.5'] };

// ------------------------------------------------------------------ storage
const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(`gcp:${key}`);
      return v == null ? fallback : JSON.parse(v);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`gcp:${key}`, JSON.stringify(value)); } catch { /* private mode */ }
  },
};

const state = {
  view: 'fixtures',
  comp: store.get('comp', 'all'),
  range: store.get('range', '7'),
  resultsRange: store.get('resultsRange', '14'),
  search: '',
  goalLine: store.get('goalLine', '2.5'),
  cornerLine: store.get('cornerLine', '9.5'),
  sort: store.get('sort', 'date'),
  teamSort: { key: 'gf', dir: -1 },
  season: null,
  tool: store.get('tool', null),
};

let D = null;

// ------------------------------------------------------------------ helpers
function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    e.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return e;
}

const SVGNS = 'http://www.w3.org/2000/svg';
function icon(kind, title) {
  const svg = document.createElementNS(SVGNS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', title ? 'false' : 'true');
  const paths = {
    warn: 'M12 3 2 21h20L12 3zm0 6v6m0 3v.5',
    check: 'M5 12.5 10 17 19 7',
    cross: 'M6 6l12 12M18 6 6 18',
  };
  const p = document.createElementNS(SVGNS, 'path');
  p.setAttribute('d', paths[kind]);
  svg.appendChild(p);
  if (title) {
    const t = document.createElementNS(SVGNS, 'title');
    t.textContent = title;
    svg.appendChild(t);
    svg.setAttribute('role', 'img');
  }
  return svg;
}

const f1 = (x) => (x == null ? '–' : Number(x).toFixed(1));
const f2 = (x) => (x == null ? '–' : Number(x).toFixed(2));
const pct = (p) => (p == null ? '–' : `${Math.round(p * 100)}%`);
const odds = (p) => (p > 0.005 ? (1 / p).toFixed(2) : '–');
const compName = (c) => D.meta.competitions[c]?.name ?? (c === 'other' ? 'Other league' : c);
const compShort = (c) => D.meta.competitions[c]?.short ?? D.meta.competitions[c]?.name ?? (c === 'other' ? 'Other' : c);
const team = (id) => D.teams[id] ?? { name: `Team ${id}`, short: `Team ${id}`, logo: '' };

function logo(id, cls) {
  const t = team(id);
  if (!t.logo) return h('span', { class: `logo-ph ${cls || ''}`, 'aria-hidden': 'true' });
  const img = h('img', { src: t.logo, alt: '', loading: 'lazy', class: cls, width: 22, height: 22 });
  img.addEventListener('error', () => img.replaceWith(h('span', { class: 'logo-ph', 'aria-hidden': 'true' })), { once: true });
  return img;
}

function teamCell(id, side, score) {
  return h('div', { class: `team ${side}` },
    h('span', {}, team(id).name), logo(id),
    score != null ? h('b', { class: 'sc' }, score) : null);
}

const dayKey = (d) => d.toLocaleDateString('en-CA');
function dayLabel(d) {
  const today = new Date();
  const k = dayKey(d);
  if (k === dayKey(today)) return 'Today';
  const tm = new Date(today); tm.setDate(today.getDate() + 1);
  if (k === dayKey(tm)) return 'Tomorrow';
  const yd = new Date(today); yd.setDate(today.getDate() - 1);
  if (k === dayKey(yd)) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' });
}
const timeStr = (d) => d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
const shortDate = (d) => d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' });

function matchesFilter(m) {
  if (state.comp !== 'all' && m.comp !== state.comp) return false;
  if (!state.search) return true;
  const q = state.search.toLowerCase();
  return team(m.home).name.toLowerCase().includes(q) || team(m.away).name.toLowerCase().includes(q);
}

// ------------------------------------------------------------------ data
async function load() {
  if (window.__DATA__) {
    D = window.__DATA__;
    return;
  }
  const names = ['meta', 'model', 'teams', 'fixtures', 'results'];
  const parts = await Promise.all(names.map(async (n) => {
    const r = await fetch(`data/${n}.json`, { cache: 'no-store' });
    if (!r.ok) throw new Error(`data/${n}.json: HTTP ${r.status}`);
    return r.json();
  }));
  D = Object.fromEntries(names.map((n, i) => [n, parts[i]]));
}

// ------------------------------------------------------------------ filters
function chips() {
  const opts = [['all', 'All'], ...TARGETS.map((c) => [c, compName(c)])];
  return h('div', { class: 'chips', role: 'group', 'aria-label': 'Competition' },
    opts.map(([v, label]) => h('button', {
      class: 'chip', type: 'button', 'aria-pressed': String(state.comp === v),
      onclick: () => { state.comp = v; store.set('comp', v); render(); },
    }, label)));
}

function select(label, value, options, onchange) {
  return h('label', { class: 'field' }, label,
    h('select', { onchange: (e) => onchange(e.target.value) },
      options.map(([v, t]) => h('option', { value: v, selected: String(v) === String(value) }, t))));
}

function searchBox() {
  const input = h('input', { type: 'search', placeholder: 'Search team', value: state.search, 'aria-label': 'Search team' });
  input.addEventListener('input', () => {
    state.search = input.value.trim();
    renderView();
  });
  return h('label', { class: 'field' }, input);
}

function renderFilters() {
  const box = document.getElementById('filters');
  box.replaceChildren();
  if (state.view === 'tool') return;
  box.append(chips());
  if (state.view === 'fixtures') {
    box.append(
      select('Show', state.range, [['1', 'Next 24 hours'], ['3', 'Next 3 days'], ['7', 'Next 7 days'], ['14', 'Next 14 days'], ['30', 'Next 30 days'], ['all', 'Rest of season']],
        (v) => { state.range = v; store.set('range', v); renderView(); }),
      select('Goals line', state.goalLine, D.model.goal_lines.map((l) => [String(l), String(l)]),
        (v) => { state.goalLine = v; store.set('goalLine', v); renderView(); }),
      select('Corners line', state.cornerLine, D.model.corner_lines.map((l) => [String(l), String(l)]),
        (v) => { state.cornerLine = v; store.set('cornerLine', v); renderView(); }),
      select('Sort', state.sort, [['date', 'Kick-off'], ['goals', 'Expected goals'], ['corners', 'Expected corners'], ['pgoals', 'Goals over %'], ['pcorners', 'Corners over %']],
        (v) => { state.sort = v; store.set('sort', v); renderView(); }),
      searchBox(),
    );
  } else if (state.view === 'results') {
    const gl = RESULT_LINES.goals.includes(state.goalLine) ? state.goalLine : '2.5';
    const cl = RESULT_LINES.corners.includes(state.cornerLine) ? state.cornerLine : '9.5';
    box.append(
      select('Show', state.resultsRange, [['3', 'Last 3 days'], ['7', 'Last 7 days'], ['14', 'Last 14 days'], ['30', 'Last 30 days'], ['all', 'This season']],
        (v) => { state.resultsRange = v; store.set('resultsRange', v); renderView(); }),
      select('Goals line', gl, RESULT_LINES.goals.map((l) => [l, l]), (v) => { state.goalLine = v; store.set('goalLine', v); renderView(); }),
      select('Corners line', cl, RESULT_LINES.corners.map((l) => [l, l]), (v) => { state.cornerLine = v; store.set('cornerLine', v); renderView(); }),
      searchBox(),
    );
  } else if (state.view === 'teams') {
    box.append(searchBox());
  }
}

// ------------------------------------------------------------------ fixtures
function metricCell(kind, total, unit, p, line) {
  return h('div', { class: `metric c-${kind}` },
    h('div', { class: 'top' },
      h('span', { class: 'val' }, f1(total), h('small', {}, unit)),
      h('span', { class: 'p' }, `O${line} `, h('b', {}, pct(p)))),
    h('div', { class: `meter ${kind}`, role: 'img', 'aria-label': `${pct(p)} chance of over ${line} ${unit}` },
      h('i', { style: `width:${Math.round((p ?? 0) * 100)}%` })));
}

function confCell(f) {
  if (f.confidence === 'low') return h('div', { class: 'conf' }, icon('warn', 'Limited recent data for one of these teams'));
  return h('div', { class: 'conf' });
}

function fixtureRow(f, showDate) {
  const d = new Date(f.date);
  const open = () => openMatch(f);
  return h('div', {
    class: 'fx', role: 'button', tabindex: 0,
    'aria-label': `${team(f.home).name} v ${team(f.away).name}, details`,
    onclick: open, onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } },
  },
  h('div', { class: 'time' }, f.state === 'in' ? 'LIVE' : timeStr(d), h('small', {}, showDate ? `${shortDate(d)} · ${compShort(f.comp)}` : compShort(f.comp))),
  h('div', { class: 'teams-m' },
    teamCell(f.home, 'home'),
    h('div', { class: 'xg', title: 'Expected goals' }, h('b', {}, f1(f.goals.home)), ' – ', h('b', {}, f1(f.goals.away))),
    teamCell(f.away, 'away')),
  metricCell('goals', f.goals.total, 'goals', f.goals.over[state.goalLine], state.goalLine),
  metricCell('corners', f.corners.total, 'corners', f.corners.over[state.cornerLine], state.cornerLine),
  confCell(f));
}

function listHead(cols) {
  return h('div', { class: 'fx-head', 'aria-hidden': 'true' },
    h('div', {}, cols[0]), h('div', { style: 'text-align:right' }, 'Home'), h('div', { style: 'text-align:center' }, cols[1]),
    h('div', {}, 'Away'), h('div', { class: 'c-goals' }, cols[2]), h('div', { class: 'c-corners' }, cols[3]), h('div', {}));
}

function renderFixtures(view) {
  const now = Date.now();
  const horizon = state.range === 'all' ? Infinity : now + Number(state.range) * 864e5;
  let list = D.fixtures.filter((f) => matchesFilter(f) && Date.parse(f.date) <= horizon);
  const key = {
    goals: (f) => -f.goals.total, corners: (f) => -f.corners.total,
    pgoals: (f) => -f.goals.over[state.goalLine], pcorners: (f) => -f.corners.over[state.cornerLine],
  }[state.sort];
  list = [...list].sort(key ? (a, b) => key(a) - key(b) : (a, b) => a.date.localeCompare(b.date));

  if (!list.length) {
    view.append(h('div', { class: 'card empty' }, D.fixtures.length ? 'No fixtures match these filters. Try a longer time window.' : 'No upcoming fixtures in the data yet.'));
    return;
  }
  const avg = (fn) => list.reduce((s, f) => s + fn(f), 0) / list.length;
  view.append(h('div', { class: 'summary-row' },
    tile('Matches', list.length, state.range === 'all' ? 'rest of season' : `next ${state.range === '1' ? '24 hours' : `${state.range} days`}`),
    tile('Avg expected goals', f2(avg((f) => f.goals.total)), `${pct(avg((f) => f.goals.over[state.goalLine]))} avg chance of over ${state.goalLine}`),
    tile('Avg expected corners', f2(avg((f) => f.corners.total)), `${pct(avg((f) => f.corners.over[state.cornerLine]))} avg chance of over ${state.cornerLine}`)));

  const head = listHead(['Time', 'xG', 'Total goals', 'Total corners']);
  if (state.sort !== 'date') {
    view.append(h('div', { class: 'card' }, head, list.slice(0, 400).map((f) => fixtureRow(f, true))));
    return;
  }
  const days = new Map();
  for (const f of list) {
    const k = f.state === 'in' ? 'live' : dayKey(new Date(f.date));
    if (!days.has(k)) days.set(k, []);
    days.get(k).push(f);
  }
  let first = true;
  const groups = [...days].sort(([a], [b]) => (a === 'live' ? -1 : b === 'live' ? 1 : 0));
  for (const [k, fs] of groups) {
    view.append(h('div', { class: 'day' },
      h('h3', {}, k === 'live' ? 'Live now (pre-match prediction)' : dayLabel(new Date(fs[0].date))),
      h('div', { class: 'card' }, first ? head : null, fs.map((f) => fixtureRow(f, false)))));
    first = false;
  }
}

function tile(label, value, sub) {
  return h('div', { class: 'card tile' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value), sub ? h('div', { class: 'sub' }, sub) : null);
}

// ------------------------------------------------------------------ results
function verdict(p, actual, line) {
  if (p == null) return null;
  const callOver = p > 0.5;
  const isOver = actual > Number(line);
  const ok = callOver === isOver;
  return h('span', { class: `verdict ${ok ? 'ok' : 'miss'}` }, icon(ok ? 'check' : 'cross'),
    `${callOver ? 'Over' : 'Under'} ${line} (${pct(callOver ? p : 1 - p)}) · ${ok ? 'correct' : 'wrong'}`);
}

function resultCell(kind, r, line, unit) {
  const x = r[kind];
  if (!x) return h('div', { class: `metric c-${kind}` }, h('span', { class: 'muted' }, 'no data'));
  return h('div', { class: `metric c-${kind}` },
    h('div', { class: 'top' },
      h('span', { class: 'val' }, String(x.actual), h('small', {}, `${unit} · predicted ${f1(x.pred)}`))),
    verdict(x.p_over[line], x.actual, line));
}

function renderResults(view) {
  const gl = RESULT_LINES.goals.includes(state.goalLine) ? state.goalLine : '2.5';
  const cl = RESULT_LINES.corners.includes(state.cornerLine) ? state.cornerLine : '9.5';
  const seasonStart = D.meta.metrics.at(-1)?.start ?? '2026-07-01';
  const from = state.resultsRange === 'all' ? Date.parse(seasonStart) : Date.now() - Number(state.resultsRange) * 864e5;
  const list = D.results.filter((r) => matchesFilter(r) && Date.parse(r.date) >= from).sort((a, b) => b.date.localeCompare(a.date));
  if (!list.length) {
    view.append(h('div', { class: 'card empty' }, 'No finished matches match these filters.'));
    return;
  }
  const stat = (kind, line) => {
    const xs = list.map((r) => r[kind]).filter((x) => x && x.p_over[line] != null);
    const hits = xs.filter((x) => (x.p_over[line] > 0.5) === (x.actual > Number(line))).length;
    const mae = xs.reduce((s, x) => s + Math.abs(x.pred - x.actual), 0) / (xs.length || 1);
    return { n: xs.length, hit: hits / (xs.length || 1), mae };
  };
  const g = stat('goals', gl), c = stat('corners', cl);
  view.append(h('div', { class: 'summary-row' },
    tile('Matches', list.length, 'predicted before kick-off'),
    tile(`Goals O/U ${gl} calls`, pct(g.hit), `${g.n} matches · avg error ${f2(g.mae)} goals`),
    tile(`Corners O/U ${cl} calls`, pct(c.hit), `${c.n} matches · avg error ${f2(c.mae)} corners`)));

  const rows = list.slice(0, 500).map((r) => {
    const d = new Date(r.date);
    const gs = r.goals;
    return h('div', { class: 'fx res', role: 'button', tabindex: 0, onclick: () => openResult(r), onkeydown: (e) => { if (e.key === 'Enter') openResult(r); } },
      h('div', { class: 'time' }, shortDate(d), h('small', {}, compShort(r.comp))),
      h('div', { class: 'teams-m' },
        teamCell(r.home, 'home', gs?.home_actual),
        h('div', { class: 'xg res-score score' }, gs ? `${gs.home_actual} – ${gs.away_actual}` : '–'),
        teamCell(r.away, 'away', gs?.away_actual)),
      resultCell('goals', r, gl, 'goals'),
      resultCell('corners', r, cl, 'corners'),
      h('div', {}));
  });
  view.append(h('div', { class: 'card' }, listHead(['Date', 'Score', 'Goals', 'Corners']), rows));
  view.append(h('p', { class: 'muted', style: 'font-size:12.5px' },
    'Every prediction here was made by a model refitted each week using only matches played before that week (no hindsight). A call is "Over" when the model gave it more than 50%.'));
}

function openResult(r) {
  const neutral = !!r.neutral;
  const pred = predictMatch(D.model, D.teams, r.home, r.away, r.comp, neutral);
  openMatch({ id: r.id, comp: r.comp, date: r.date, home: r.home, away: r.away, neutral, ...pred, result: r, confidence: 'high' });
}

// ------------------------------------------------------------------ detail
function linePicker(lines, current, onpick) {
  const wrap = h('div', { class: 'line-pick', role: 'group', 'aria-label': 'Over/under line' });
  const render = (cur) => {
    wrap.replaceChildren(...lines.map((l) => h('button', {
      type: 'button', 'aria-pressed': String(String(l) === String(cur)),
      onclick: () => { render(l); onpick(String(l)); },
    }, String(l))));
  };
  render(current);
  return wrap;
}

function pOverFromPmf(pmf, line) {
  let s = 0;
  for (let k = Math.floor(Number(line)) + 1; k < pmf.length; k++) s += pmf[k];
  return s;
}

function distPanel(kind, f, lines, initial) {
  const x = f[kind];
  const unit = kind === 'goals' ? 'goals' : 'corners';
  const body = h('div');
  const draw = (line) => {
    // over-probabilities come from the full distribution; the chart's last bucket is "n or more"
    const p = x.over[line] ?? pOverFromPmf(x.pmf, line);
    body.replaceChildren(
      h('div', { class: 'ou' },
        h('span', {}, `Over ${line}: `, h('b', {}, pct(p)), h('span', { class: 'muted' }, `  fair odds ${odds(p)}`)),
        h('span', {}, `Under ${line}: `, h('b', {}, pct(1 - p)), h('span', { class: 'muted' }, `  fair odds ${odds(1 - p)}`))),
      h('div', { class: 'chart-legend' },
        h('span', {}, h('i', { style: `background:var(--${kind})` }), `Over ${line}`),
        h('span', {}, h('i', { style: `background:var(--${kind}-track)` }), `Under ${line}`)),
      distributionChart(x.pmf, { line: Number(line), series: kind, unit, ariaLabel: `Probability of each total ${unit} count` }));
  };
  draw(initial);
  const label = kind === 'goals' ? 'Total goals' : 'Total corners';
  return h('div', { class: 'card panel' },
    h('h4', {}, h('span', { class: `key ${kind}` }), label),
    h('div', { class: 'big' },
      h('span', { class: 'total' }, f2(x.total)),
      h('span', { class: 'split' }, team(f.home).short || team(f.home).name, ' ', h('b', {}, f2(x.home)), ' · ', team(f.away).short || team(f.away).name, ' ', h('b', {}, f2(x.away)))),
    linePicker(lines, initial, (l) => {
      if (kind === 'goals') { state.goalLine = l; store.set('goalLine', l); } else { state.cornerLine = l; store.set('cornerLine', l); }
      draw(l);
    }),
    body);
}

function goalsExtras(f) {
  const g = f.goals;
  const hn = team(f.home).short || team(f.home).name;
  const an = team(f.away).short || team(f.away).name;
  const facts = h('div', { class: 'facts' },
    fact(`${hn} win`, pct(g.hda[0])), fact('Draw', pct(g.hda[1])), fact(`${an} win`, pct(g.hda[2])),
    fact('Both teams score', pct(g.btts)), fact(`${hn} clean sheet`, pct(g.cs[0])), fact(`${an} clean sheet`, pct(g.cs[1])));
  return h('div', {},
    facts,
    h('div', { class: 'section-title', style: 'margin-top:14px' }, 'Most likely scores'),
    h('div', { class: 'scores' }, g.scores.slice(0, 6).map((s) => h('span', {}, h('b', {}, `${s.h}-${s.a}`), pct(s.p)))),
    f.market_goals_line != null ? h('p', { class: 'muted', style: 'font-size:12.5px;margin:10px 0 0' }, `Bookmaker goals line (via ESPN): ${f.market_goals_line}. Model chance of over ${f.market_goals_line}: ${pct(g.over[String(f.market_goals_line)])}.`) : null);
}

function fact(label, value) {
  return h('div', { class: 'fact' }, h('div', { class: 'label' }, label), h('div', { class: 'value' }, value));
}

function cornersExtras(f) {
  const c = f.corners;
  const hn = team(f.home).short || team(f.home).name;
  const an = team(f.away).short || team(f.away).name;
  return h('div', {},
    h('div', { class: 'facts' },
      fact(`${hn} more corners`, pct(c.home_more)), fact('Level', pct(1 - c.home_more - c.away_more)), fact(`${an} more corners`, pct(c.away_more))),
    h('div', { class: 'table-wrap' }, h('table', { class: 'mini-table' },
      h('thead', {}, h('tr', {}, h('th', { class: 'left' }, 'Team corners'), D.model.team_corner_lines.map((l) => h('th', {}, `O${l}`)))),
      h('tbody', {},
        h('tr', {}, h('td', { class: 'left' }, `${hn} (${f2(c.home)})`), D.model.team_corner_lines.map((l) => h('td', {}, pct(c.home_over[String(l)])))),
        h('tr', {}, h('td', { class: 'left' }, `${an} (${f2(c.away)})`), D.model.team_corner_lines.map((l) => h('td', {}, pct(c.away_over[String(l)]))))))));
}

function formTable(id) {
  const t = team(id);
  const form = t.form ?? [];
  const rec = t.recent;
  return h('div', { class: 'card panel' },
    h('h5', {}, logo(id), t.name),
    rec ? h('div', { class: 'muted', style: 'font-size:12.5px;margin-bottom:6px' },
      `Last ${rec.n}: goals ${f1(rec.gf)} for, ${f1(rec.ga)} against · corners ${f1(rec.cf)} for, ${f1(rec.ca)} against`) : null,
    form.length ? h('div', { class: 'table-wrap' }, h('table', { class: 'mini-table' },
      h('thead', {}, h('tr', {}, h('th', { class: 'left' }, 'Date'), h('th', { class: 'left' }, 'Opponent'), h('th', {}, 'Goals'), h('th', {}, 'Corners'))),
      h('tbody', {}, form.map((m) => h('tr', {},
        h('td', { class: 'left muted' }, new Date(m.date).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })),
        h('td', { class: 'left' }, `${m.venue === 'H' ? 'v' : '@'} ${team(m.opp).short || team(m.opp).name}`, h('span', { class: 'comp-tag' }, compShort(m.comp))),
        h('td', {}, `${m.gf}-${m.ga}`),
        h('td', {}, m.cf == null ? '–' : `${m.cf}-${m.ca}`)))))) : h('p', { class: 'muted' }, 'No recent matches in the data.'));
}

export function matchDetail(f, { closable }) {
  const d = new Date(f.date);
  const head = h('div', { class: 'detail-head' },
    h('div', { class: 'meta' },
      h('span', {}, compName(f.comp)),
      h('span', {}, '·'),
      h('span', {}, `${d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })} ${timeStr(d)}`),
      f.venue ? h('span', {}, `· ${f.venue}`) : null,
      f.neutral ? h('span', { class: 'comp-tag' }, 'Neutral venue') : null),
    h('div', { class: 'matchup', id: 'match-title' },
      h('div', { class: 'side home' }, h('span', {}, team(f.home).name), logo(f.home)),
      h('div', { class: 'vs' }, f.result?.goals ? h('b', { style: 'font-size:22px;color:var(--ink)' }, `${f.result.goals.home_actual} – ${f.result.goals.away_actual}`) : 'vs'),
      h('div', { class: 'side away' }, logo(f.away), h('span', {}, team(f.away).name))),
    closable ? h('button', { class: 'btn ghost icon close', type: 'button', 'aria-label': 'Close', onclick: closeMatch }, icon('cross')) : null);

  const notes = [];
  if (f.result) {
    const r = f.result;
    notes.push(h('div', { class: 'note' }, h('span', {},
      `Final: ${r.goals ? r.goals.actual : '–'} goals, ${r.corners ? r.corners.actual : '–'} corners. `,
      `Before kick-off the model expected ${r.goals ? f1(r.goals.pred) : '–'} goals and ${r.corners ? f1(r.corners.pred) : '–'} corners. `,
      'The distributions below use the current ratings.')));
  }
  if (f.confidence === 'low' || f.confidence === 'medium') {
    notes.push(h('div', { class: 'note' }, icon('warn'),
      h('span', {}, f.confidence === 'low'
        ? 'Limited data: at least one of these teams has few recent matches in our data, so this prediction leans on league averages.'
        : 'Moderate data: one of these teams has a modest number of recent matches in our data.')));
  }

  return h('div', {},
    head,
    notes,
    h('div', { class: 'detail-grid' },
      h('div', {}, distPanel('goals', f, D.model.goal_lines, state.goalLine), h('div', { class: 'card panel', style: 'margin-top:16px' }, goalsExtras(f))),
      h('div', {}, distPanel('corners', f, D.model.corner_lines, state.cornerLine), h('div', { class: 'card panel', style: 'margin-top:16px' }, cornersExtras(f)))),
    h('div', { class: 'section-title', style: 'padding:0 20px' }, 'Recent form'),
    h('div', { class: 'form-grid', style: 'padding-top:0' }, formTable(f.home), formTable(f.away)));
}

function openMatch(f) {
  const dlg = document.getElementById('match');
  const body = dlg.querySelector('.dialog-body');
  body.replaceChildren(matchDetail(f, { closable: true }));
  if (!dlg.open) dlg.showModal();
  dlg.scrollTop = 0;
}

function closeMatch() {
  hideTip();
  document.getElementById('match').close();
}

// ------------------------------------------------------------------ match tool
function renderTool(view) {
  const selectable = Object.entries(D.teams).filter(([, t]) => t.selectable && t.r)
    .sort((a, b) => a[1].name.localeCompare(b[1].name));
  const labelOf = (id) => `${team(id).name} (${compShort(team(id).league)})`;
  const byLabel = new Map(selectable.map(([id]) => [labelOf(id).toLowerCase(), id]));
  const byName = new Map(selectable.map(([id, t]) => [t.name.toLowerCase(), id]));
  const resolve = (v) => byLabel.get(v.trim().toLowerCase()) ?? byName.get(v.trim().toLowerCase()) ?? null;

  const first = D.fixtures.find((f) => f.comp === 'uefa.champions') ?? D.fixtures[0];
  const init = state.tool ?? (first ? { home: first.home, away: first.away, comp: first.comp, neutral: false } : { home: null, away: null, comp: 'eng.1', neutral: false });

  const list = h('datalist', { id: 'team-list' }, selectable.map(([id]) => h('option', { value: labelOf(id) })));
  const homeIn = h('input', { list: 'team-list', placeholder: 'Type a team…', value: init.home ? labelOf(init.home) : '', autocomplete: 'off' });
  const awayIn = h('input', { list: 'team-list', placeholder: 'Type a team…', value: init.away ? labelOf(init.away) : '', autocomplete: 'off' });
  const compSel = h('select', {}, TARGETS.map((c) => h('option', { value: c, selected: c === init.comp }, compName(c))));
  const neutralIn = h('input', { type: 'checkbox', checked: !!init.neutral });
  const out = h('div', { class: 'tool-result' });
  const msg = h('p', { class: 'muted', role: 'status' });

  const run = () => {
    const home = resolve(homeIn.value), away = resolve(awayIn.value);
    if (!home || !away) { msg.textContent = 'Pick both teams from the list.'; out.replaceChildren(); return; }
    if (home === away) { msg.textContent = 'Pick two different teams.'; out.replaceChildren(); return; }
    msg.textContent = '';
    state.tool = { home, away, comp: compSel.value, neutral: neutralIn.checked };
    store.set('tool', state.tool);
    const pred = predictMatch(D.model, D.teams, home, away, compSel.value, neutralIn.checked);
    const scheduled = D.fixtures.find((f) => f.home === home && f.away === away);
    const n = Math.min(team(home).n_eff ?? 0, team(away).n_eff ?? 0);
    const f = {
      id: 'tool', comp: compSel.value, date: scheduled?.date ?? new Date().toISOString(), home, away, neutral: neutralIn.checked,
      venue: scheduled?.venue, market_goals_line: null, confidence: n >= 20 ? 'high' : n >= 8 ? 'medium' : 'low', ...pred,
    };
    out.replaceChildren(h('div', { class: 'card' }, matchDetail(f, { closable: false })));
  };
  const autoComp = () => {
    const home = resolve(homeIn.value), away = resolve(awayIn.value);
    if (home && away && team(home).league === team(away).league && TARGETS.includes(team(home).league)) compSel.value = team(home).league;
    else if (home && away && team(home).league !== team(away).league) compSel.value = 'uefa.champions';
  };
  homeIn.addEventListener('change', () => { autoComp(); run(); });
  awayIn.addEventListener('change', () => { autoComp(); run(); });
  compSel.addEventListener('change', run);
  neutralIn.addEventListener('change', run);

  view.append(
    h('div', { class: 'card' },
      h('div', { class: 'tool-form' },
        h('label', {}, 'Home team', homeIn),
        h('button', { class: 'btn', type: 'button', title: 'Swap teams', onclick: () => { const t = homeIn.value; homeIn.value = awayIn.value; awayIn.value = t; run(); } }, '⇄ Swap'),
        h('label', {}, 'Away team', awayIn),
        h('label', {}, 'Competition', compSel),
        h('label', { class: 'check' }, neutralIn, 'Neutral venue'),
        h('button', { class: 'btn primary', type: 'button', onclick: run }, 'Predict'),
        list)),
    h('p', { class: 'muted', style: 'font-size:12.5px' },
      `Any of the ${selectable.length} clubs that played in these six competitions in the last year can be paired, including clubs from different leagues. The competition sets the league's baseline scoring and corner rates.`),
    msg, out);
  if (homeIn.value && awayIn.value) run();
}

// ------------------------------------------------------------------ teams
function renderTeams(view) {
  const inUcl = new Set();
  for (const m of [...D.fixtures, ...D.results]) if (m.comp === 'uefa.champions') { inUcl.add(m.home); inUcl.add(m.away); }
  let rows = Object.entries(D.teams).filter(([id, t]) => t.r && t.selectable && (
    state.comp === 'all' ? TARGETS.slice(0, 5).includes(t.league) || inUcl.has(id)
      : state.comp === 'uefa.champions' ? inUcl.has(id) : t.league === state.comp));
  if (state.search) rows = rows.filter(([, t]) => t.name.toLowerCase().includes(state.search.toLowerCase()));
  const cols = [
    ['name', 'Team', (t) => t.name, 'left'],
    ['league', 'League', (t) => compShort(t.league), 'left'],
    ['gf', 'Goals for', (t) => t.profile.gf],
    ['ga', 'Goals against', (t) => t.profile.ga],
    ['cf', 'Corners for', (t) => t.profile.cf],
    ['ca', 'Corners against', (t) => t.profile.ca],
    ['rgf', 'L10 goals', (t) => t.recent?.gf],
    ['rga', 'L10 conceded', (t) => t.recent?.ga],
    ['rcf', 'L10 corners', (t) => t.recent?.cf],
    ['rca', 'L10 corners agst', (t) => t.recent?.ca],
  ];
  const { key, dir } = state.teamSort;
  const getter = cols.find((c) => c[0] === key)?.[2] ?? cols[2][2];
  rows.sort((a, b) => {
    const x = getter(a[1]), y = getter(b[1]);
    if (typeof x === 'string') return dir * x.localeCompare(y);
    return dir * ((x ?? -1) - (y ?? -1));
  });
  const th = ([k, label, , align]) => h('th', {
    class: `sortable ${align || ''}`, scope: 'col', 'aria-sort': k === key ? (dir > 0 ? 'ascending' : 'descending') : null,
    onclick: () => { state.teamSort = { key: k, dir: k === key ? -dir : (k === 'name' || k === 'league' ? 1 : -1) }; renderView(); },
  }, label, k === key ? (dir > 0 ? ' ↑' : ' ↓') : '');
  view.append(
    h('p', { class: 'muted', style: 'font-size:12.5px;margin-top:0' },
      'Model ratings: expected per-game numbers against an average opponent from the team\'s own league at a neutral venue. L10 = raw averages over the last 10 matches in any competition.'),
    h('div', { class: 'card table-wrap' },
      h('table', {},
        h('thead', {}, h('tr', {}, cols.map(th))),
        h('tbody', {}, rows.map(([id, t]) => h('tr', {},
          h('td', { class: 'left' }, h('div', { class: 'team' }, logo(id), h('span', {}, t.name))),
          h('td', { class: 'left muted' }, compShort(t.league)),
          cols.slice(2).map((c) => h('td', {}, f2(c[2](t))))))))));
  if (!rows.length) view.append(h('div', { class: 'card empty' }, 'No teams match.'));
}

// ------------------------------------------------------------------ model
function renderModel(view) {
  const seasons = D.meta.metrics;
  if (!seasons.length) { view.append(h('div', { class: 'card empty' }, 'Backtest results are not available yet.')); return; }
  const season = seasons.find((s) => s.id === state.season) ?? seasons.at(-1);
  const scope = state.comp === 'all' ? season : season.by_comp[state.comp];
  const g = scope?.goals, c = scope?.corners;

  view.append(h('div', { class: 'season-switch chips', role: 'group', 'aria-label': 'Season' },
    seasons.map((s) => h('button', {
      class: 'chip', type: 'button', 'aria-pressed': String(s.id === season.id),
      onclick: () => { state.season = s.id; renderView(); },
    }, `${s.label} · ${s.note}`))));

  if (!g || !c) {
    view.append(h('div', { class: 'card empty' }, 'Not enough matches in this competition for this season yet.'));
  } else {
    const skill = (m) => (m.base_rps ? `${((1 - m.rps / m.base_rps) * 100).toFixed(1)}%` : '–');
    view.append(h('div', { class: 'summary-row' },
      tile('Goals O/U 2.5 calls correct', pct(g.lines['2.5'].hit_rate), `baseline ${pct(g.lines['2.5'].base_hit_rate)} · ${g.n} matches`),
      tile('Avg error, total goals', f2(g.mae), `baseline ${f2(g.base_mae)} · RPS skill ${skill(g)}`),
      tile('Corners O/U 9.5 calls correct', pct(c.lines['9.5'].hit_rate), `baseline ${pct(c.lines['9.5'].base_hit_rate)} · ${c.n} matches`),
      tile('Avg error, total corners', f2(c.mae), `baseline ${f2(c.base_mae)} · RPS skill ${skill(c)}`)));
  }

  const comps = Object.entries(season.by_comp);
  if (comps.length) {
    view.append(h('div', { class: 'section-title' }, `By competition, ${season.label}`),
      h('div', { class: 'card table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, ['Competition', 'Matches', 'Goals error', 'Baseline', 'O/U 2.5 correct', 'Baseline', 'Corners error', 'Baseline', 'O/U 9.5 correct', 'Baseline']
          .map((t, i) => h('th', { class: i === 0 ? 'left' : '' }, t)))),
        h('tbody', {}, comps.map(([comp, m]) => h('tr', {},
          h('td', { class: 'left' }, compName(comp)), h('td', {}, m.goals.n),
          h('td', {}, f2(m.goals.mae)), h('td', { class: 'muted' }, f2(m.goals.base_mae)),
          h('td', {}, pct(m.goals.lines['2.5'].hit_rate)), h('td', { class: 'muted' }, pct(m.goals.lines['2.5'].base_hit_rate)),
          h('td', {}, f2(m.corners.mae)), h('td', { class: 'muted' }, f2(m.corners.base_mae)),
          h('td', {}, pct(m.corners.lines['9.5'].hit_rate)), h('td', { class: 'muted' }, pct(m.corners.lines['9.5'].base_hit_rate)))))))
    );
  }

  const calib = h('div', { class: 'calib' });
  for (const [kind, line, title] of [['goals', '2.5', 'Over 2.5 goals'], ['corners', '9.5', 'Over 9.5 corners']]) {
    const bins = season[kind]?.lines?.[line]?.calibration ?? [];
    calib.append(h('div', { class: 'card' },
      h('h4', {}, `Calibration: ${title}`),
      h('div', { class: 'muted', style: 'font-size:12px' }, 'Dots on the diagonal mean "60%" really happens 60% of the time. All six competitions.'),
      bins.length ? calibrationChart(bins, { series: kind, ariaLabel: `Calibration of ${title} predictions` }) : h('p', { class: 'muted' }, 'Not enough matches yet.'),
      bins.length ? h('table', { class: 'mini-table' },
        h('thead', {}, h('tr', {}, h('th', { class: 'left' }, 'Predicted'), h('th', {}, 'Observed'), h('th', {}, 'Matches'))),
        h('tbody', {}, bins.map((b) => h('tr', {}, h('td', { class: 'left' }, pct(b.p)), h('td', {}, pct(b.freq)), h('td', {}, b.n))))) : null));
  }
  view.append(h('div', { class: 'section-title' }, 'Are the probabilities honest?'), calib);

  const m = D.meta;
  const support = Object.entries(m.competitions).filter(([, v]) => !v.target).map(([, v]) => v.name);
  view.append(h('div', { class: 'section-title' }, 'How it works'), h('div', { class: 'prose' },
    h('p', {}, `The model is trained on ${m.matches_played.toLocaleString()} finished matches since January 2023 (${m.matches_with_corners.toLocaleString()} with corner counts). The six target competitions are covered, plus ${support.length} others: ${support.join(', ')}. Second divisions rate promoted clubs, and the other leagues and UEFA cups put every Champions League club on one shared scale.`),
    h('h3', {}, 'Goals'),
    h('p', {}, 'Each club gets an attack and a defence rating. Expected goals for a side = exp(baseline + competition effect + home advantage + own attack − opponent defence). Ratings come from a Poisson regression where recent matches count more (half-life ', String(m.params.goals.half_life_days), ' days). Ratings are also pulled toward the club\'s league average, which keeps numbers sensible for clubs with few matches. The two expected-goal numbers give a full scoreline grid, with a Dixon–Coles correction for low scores, and the total-goals distribution comes from that grid.'),
    h('h3', {}, 'Corners'),
    h('p', {}, 'The same structure is fitted separately on corners won: a corner-winning rating and a corner-conceding rating per club. Corner totals vary more than a Poisson count allows, so total corners use a negative-binomial distribution with its spread fitted from the data. ESPN records missing box scores as 0–0 corners; those games are treated as missing.'),
    h('h3', {}, 'Testing'),
    h('p', {}, 'Settings were tuned on 2025-26 only. 2026-27 is held out: each week the model is refitted on matches played before that week and then scored on that week\'s games. The baseline is the competition\'s own distribution of totals over the previous year. RPS skill is the improvement in ranked probability score over that baseline; higher is better.'),
    h('p', {}, 'Predictions cover regulation time only. Line-ups, injuries, weather and motivation are not in the model, so use these numbers as a starting point, not a guarantee.'),
    h('p', { class: 'muted' }, `Last build: ${new Date(m.generated_at).toLocaleString()} (${m.build_seconds}s). ${m.fixtures} upcoming fixtures predicted.`)));
}

// ------------------------------------------------------------------ shell
function renderView() {
  hideTip();
  const view = document.getElementById('view');
  view.replaceChildren();
  ({ fixtures: renderFixtures, results: renderResults, tool: renderTool, teams: renderTeams, model: renderModel })[state.view](view);
}

function render() {
  document.querySelectorAll('.tabs a').forEach((a) => {
    if (a.dataset.view === state.view) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  const updated = document.getElementById('updated');
  updated.textContent = `Updated ${new Date(D.meta.generated_at).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })}`;
  renderFilters();
  renderView();
}

function route() {
  const v = location.hash.replace('#', '');
  state.view = VIEWS.includes(v) ? v : 'fixtures';
  if (document.getElementById('match').open) closeMatch();
  if (D) render();
}

// ------------------------------------------------------------------ refresh
async function setupRefresh() {
  const btn = document.getElementById('refresh');
  let status;
  try {
    const r = await fetch('api/status', { cache: 'no-store' });
    if (!r.ok) return;
    status = await r.json();
  } catch { return; }
  btn.hidden = false;
  const poll = () => {
    btn.disabled = true;
    btn.textContent = 'Refreshing…';
    const timer = setInterval(async () => {
      let s;
      try { s = await (await fetch('api/status', { cache: 'no-store' })).json(); } catch { return; }
      if (s.running) return;
      clearInterval(timer);
      btn.disabled = false;
      btn.textContent = 'Refresh data';
      if (s.last_error) {
        btn.title = `Last refresh failed: ${s.last_error}`;
        btn.textContent = 'Refresh failed, retry';
        return;
      }
      await load();
      render();
    }, 3000);
  };
  btn.addEventListener('click', async () => {
    try { await fetch('api/refresh', { method: 'POST' }); } catch { return; }
    poll();
  });
  if (status.running) poll();
}

function setupTheme() {
  const btn = document.getElementById('theme');
  const apply = (t) => {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  };
  apply(store.get('theme', null));
  btn.addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    const next = dark ? 'light' : 'dark';
    apply(next);
    store.set('theme', next);
    if (D) renderView();
  });
}

async function main() {
  setupTheme();
  window.addEventListener('hashchange', route);
  document.getElementById('match').addEventListener('close', hideTip);
  document.getElementById('match').addEventListener('click', (e) => { if (e.target.id === 'match') closeMatch(); });
  const view = document.getElementById('view');
  view.replaceChildren(h('div', { class: 'card empty' }, 'Loading predictions…'));
  try {
    await load();
  } catch (err) {
    view.replaceChildren(h('div', { class: 'card empty' },
      'No prediction data yet. Run ', h('code', {}, 'python -m predictor update'), ' (or start the server with ', h('code', {}, 'python -m predictor serve'), ') and reload.',
      h('br'), h('small', { class: 'muted' }, String(err.message || err))));
    setupRefresh();
    return;
  }
  route();
  render();
  setupRefresh();
}

main();
