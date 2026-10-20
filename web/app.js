import { predictMatch } from './model.js';
import { distributionChart, calibrationChart, hideTip } from './charts.js';

const TARGETS = ['eng.1', 'esp.1', 'ita.1', 'ger.1', 'fra.1', 'uefa.champions'];
const VIEWS = ['fixtures', 'tips', 'results', 'tool', 'teams', 'model'];
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
  tipsRange: store.get('tipsRange', '3'),
  tipsMarket: store.get('tipsMarket', 'all'),
  tipsSort: store.get('tipsSort', 'date'),
  boat: store.get('boat', []),
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
    plus: 'M12 5v14M5 12h14',
    boat: 'M3 14h18l-2.5 5h-13L3 14zM12 3v11M12 4l6 8h-6',
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
  } else if (state.view === 'tips') {
    box.append(
      select('Show', state.tipsRange, [['1', 'Next 24 hours'], ['3', 'Next 3 days'], ['7', 'Next 7 days'], ['14', 'Next 14 days']],
        (v) => { state.tipsRange = v; store.set('tipsRange', v); renderView(); }),
      select('Market', state.tipsMarket, [['all', 'Goals and corners'], ['goals', 'Goals'], ['corners', 'Corners']],
        (v) => { state.tipsMarket = v; store.set('tipsMarket', v); renderView(); }),
      select('Sort', state.tipsSort, [['date', 'Kick-off'], ['p', 'Most likely'], ['lift', 'Furthest above league rate']],
        (v) => { state.tipsSort = v; store.set('tipsSort', v); renderView(); }),
      searchBox(),
    );
  } else if (state.view === 'teams') {
    box.append(searchBox());
  }
}

// ------------------------------------------------------------------ fixtures
function metricCell(kind, total, unit, p, line, withOdds = false) {
  return h('div', { class: `metric c-${kind}` },
    h('div', { class: 'top' },
      h('span', { class: 'val' }, f1(total), h('small', {}, unit),
        withOdds ? h('span', { class: 'odds-dot', title: 'Includes bookmaker odds', 'aria-label': 'includes bookmaker odds' }) : null),
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
  metricCell('goals', f.goals.total, 'goals', f.goals.over[state.goalLine], state.goalLine, !!f.market),
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

  if (list.some((f) => f.market)) {
    view.append(h('p', { class: 'small muted legend-line' }, h('span', { class: 'odds-dot' }),
      ' Forecast blended with bookmaker odds. League odds appear a few days before kick-off; the Champions League has none.'));
  }
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

// ------------------------------------------------------------------ lên thuyền (picks)
const pickLabel = (t) => `${t.side === 'over' ? 'Over' : 'Under'} ${t.line} ${t.kind}`;
const matchName = (m) => `${team(m.home).name} v ${team(m.away).name}`;
const wantKind = (t) => state.tipsMarket === 'all' || t.kind === state.tipsMarket;
let boatOpen = false;

function allPicks() {
  const out = [];
  for (const f of D.fixtures) for (const t of f.tips ?? []) out.push({ f, t, key: `${f.id}:${t.kind}` });
  return out;
}

/** Bookmaker prices for a pick, when our data has them (over/under 2.5 goals in league games). */
function pickPrices(f, t) {
  if (t.kind !== 'goals' || t.line !== 2.5 || !f.market) return null;
  const o = f.market.odds;
  return t.side === 'over' ? { avg: o.over, best: o.best_over } : { avg: o.under, best: o.best_under };
}

function setBoat(keys) {
  state.boat = keys;
  store.set('boat', keys);
  renderBoat();
}

function boatToggle(key, t, f) {
  return h('button', {
    class: 'boat-toggle', type: 'button', 'data-key': key, 'data-label': `${pickLabel(t)}, ${matchName(f)}`,
    onclick: () => setBoat(state.boat.includes(key) ? state.boat.filter((k) => k !== key) : [...state.boat, key]),
  });
}

function tipRow({ f, t, key }, showDate) {
  const d = new Date(f.date);
  const prices = pickPrices(f, t);
  const ev = prices?.best ? t.p * prices.best - 1 : null;
  const open = () => openMatch(f);
  return h('div', { class: 'tip' },
    boatToggle(key, t, f),
    h('div', { class: 'time' }, timeStr(d), h('small', {}, showDate ? `${shortDate(d)} · ${compShort(f.comp)}` : compShort(f.comp))),
    h('div', {
      class: 'tip-match', role: 'button', tabindex: 0, 'aria-label': `${matchName(f)}, details`,
      onclick: open, onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } },
    },
    h('div', { class: 'team' }, logo(f.home), h('span', {}, team(f.home).name)),
    h('div', { class: 'team' }, logo(f.away), h('span', {}, team(f.away).name))),
    h('div', { class: 'pick' },
      h('div', { class: 'pick-label' }, h('span', { class: `key ${t.kind}` }), pickLabel(t)),
      h('div', { class: 'pick-sub' }, `Expected ${f1(f[t.kind].total)} ${t.kind}`)),
    h('div', { class: 'chance' },
      h('div', { class: 'top' }, h('b', {}, pct(t.p)), h('span', { class: 'muted' }, `league ${pct(t.base)}`)),
      h('div', { class: `meter ${t.kind}`, role: 'img', 'aria-label': `${pct(t.p)} chance; ${pct(t.base)} in a typical ${compShort(f.comp)} match` },
        h('i', { style: `width:${Math.round(t.p * 100)}%` }), h('b', { class: 'base', style: `left:${(t.base * 100).toFixed(1)}%` }))),
    h('div', { class: 'price' },
      h('span', {}, 'Break-even odds ', h('b', {}, odds(t.p))),
      prices?.best
        ? h('span', {}, `Best price ${f2(prices.best)} · `, h('span', { class: ev > VALUE_EDGE ? 'edge-pos' : 'muted' }, signedPct(ev)))
        : h('span', { class: 'muted' }, noPriceReason(f, t))));
}

function noPriceReason(f, t) {
  if (t.kind === 'corners') return 'No corner prices in our data';
  if (f.comp.startsWith('uefa.')) return 'No Champions League prices in our data';
  if (t.line !== 2.5) return 'Our data prices the 2.5 line only';
  return 'No bookmaker price yet';
}

function renderBoat() {
  for (const b of document.querySelectorAll('.boat-toggle')) {
    const on = state.boat.includes(b.dataset.key);
    b.setAttribute('aria-pressed', String(on));
    b.setAttribute('aria-label', `${on ? 'Remove from' : 'Add to'} your boat: ${b.dataset.label}`);
    b.replaceChildren(icon(on ? 'check' : 'plus'));
  }
  const box = document.getElementById('boat');
  if (!box) return;
  const byKey = new Map(allPicks().map((x) => [x.key, x]));
  const legs = state.boat.map((k) => byKey.get(k)).filter(Boolean);
  if (legs.length !== state.boat.length) { state.boat = legs.map((x) => x.key); store.set('boat', state.boat); }
  box.hidden = !legs.length;
  if (!legs.length) { box.replaceChildren(); return; }

  const p = legs.reduce((s, x) => s * x.t.p, 1);
  const price = h('input', { type: 'number', inputmode: 'decimal', min: '1.01', step: '0.01', placeholder: 'e.g. 3.20', 'aria-label': 'Decimal odds for the whole boat' });
  const out = h('span', { role: 'status' });
  price.addEventListener('input', () => {
    const o = Number(price.value);
    out.replaceChildren(o > 1 ? h('span', {}, 'expected return ', h('b', { class: p * o - 1 > VALUE_EDGE ? 'edge-pos' : '' }, signedPct(p * o - 1))) : '');
  });
  const sameMatch = new Set(legs.map((x) => x.f.id)).size < legs.length;
  box.replaceChildren(h('div', {},
    h('div', { class: 'boat-bar' },
      h('span', { class: 'boat-title' }, icon('boat'), h('b', {}, 'Your boat'), ` · ${legs.length} ${legs.length === 1 ? 'pick' : 'picks'}`),
      h('span', {}, legs.length === 1 ? 'Wins ' : 'All win ', h('b', {}, pct(p))),
      h('span', {}, 'Break-even odds ', h('b', {}, odds(p))),
      h('span', { class: 'boat-actions' },
        h('button', { class: 'btn ghost', type: 'button', 'aria-expanded': String(boatOpen), onclick: () => { boatOpen = !boatOpen; renderBoat(); } },
          boatOpen ? 'Hide picks' : 'Show picks'),
        h('button', { class: 'btn ghost', type: 'button', onclick: () => setBoat([]) }, 'Clear'))),
    boatOpen ? h('div', { class: 'boat-body' },
      h('ul', { class: 'boat-legs' }, legs.map((x) => h('li', {},
        h('span', {}, h('span', { class: `key ${x.t.kind}` }), h('b', {}, pickLabel(x.t)),
          h('span', { class: 'muted' }, ` · ${matchName(x.f)} · ${shortDate(new Date(x.f.date))} ${timeStr(new Date(x.f.date))}`)),
        h('span', { class: 'leg-p' }, pct(x.t.p),
          h('button', {
            class: 'btn ghost icon close', type: 'button', 'aria-label': `Remove ${pickLabel(x.t)}, ${matchName(x.f)}`,
            onclick: () => setBoat(state.boat.filter((k) => k !== x.key)),
          }, icon('cross')))))),
      h('div', { class: 'checker' }, h('label', { class: 'field' }, 'Your bookmaker\'s price for the whole boat', price), out),
      h('p', { class: 'small muted' },
        'The chance that all picks win multiplies their chances. ',
        sameMatch ? 'Goals and corners picks from the same match are related, so for those the combined chance is only approximate. ' : '',
        'The bookmaker\'s margin is charged on every leg, so a combo usually pays well under the break-even odds.')) : null));
}

function tipsRecord(view) {
  const seasons = D.meta.metrics.filter((s) => s.tips);
  if (!seasons.length) return;
  const rate = (x) => (x ? `${pct(x.hit)} of ${x.n.toLocaleString()}` : '–');
  const roi = (o, k) => (o ? `${signedPct(o[k].roi)} (±${Math.round((o[k].roi_se ?? 0) * 100)}%)` : '–');
  view.append(h('div', { class: 'section-title' }, 'How past picks did'),
    h('div', { class: 'card table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Season', 'Picks', 'Won', 'We said', 'Goals picks', 'Corners picks', 'With odds', 'Return, avg price', 'Return, best price']
        .map((t, i) => h('th', { class: i === 0 ? 'left' : '' }, t)))),
      h('tbody', {}, seasons.map((s) => {
        const x = s.tips;
        return h('tr', {},
          h('td', { class: 'left' }, s.label, h('span', { class: 'comp-tag' }, s.note)),
          h('td', {}, x.n.toLocaleString()), h('td', {}, `${x.won.toLocaleString()} (${pct(x.hit)})`), h('td', {}, pct(x.p)),
          h('td', {}, rate(x.by_kind.goals)), h('td', {}, rate(x.by_kind.corners)),
          h('td', {}, x.odds ? x.odds.n : 0), h('td', {}, roi(x.odds, 'avg_price')), h('td', {}, roi(x.odds, 'best_price')));
      })))));

  const tiers = seasons[0].tips.tiers.map((t) => [t.lo, t.hi]);
  view.append(h('div', { class: 'card table-wrap', style: 'margin-top:12px;max-width:640px' }, h('table', { class: 'mini-table' },
    h('thead', {}, h('tr', {}, h('th', { class: 'left' }, 'We said'), seasons.map((s) => h('th', {}, `Happened, ${s.label}`)))),
    h('tbody', {}, tiers.map(([lo, hi]) => h('tr', {},
      h('td', { class: 'left' }, hi >= 1 ? `${pct(lo)} or more` : `${pct(lo)}–${pct(hi)}`),
      seasons.map((s) => {
        const t = s.tips.tiers.find((x) => x.lo === lo);
        return h('td', {}, t ? `${pct(t.hit)} of ${t.n}` : '–');
      })))))));
  view.append(h('p', { class: 'small muted' },
    'Past picks come from the weekly backtest: each week the model was refitted on earlier matches only and the same rule was applied to its forecasts. ',
    'When "Happened" is close to "We said", the chances are honest. "Return" is what one-unit bets on every pick returned at the bookmakers\' pre-closing prices; ',
    'only over/under 2.5 goals picks have prices in our data. A likely pick is not the same as a good bet: the bookmakers usually see the same thing and price it short.'));
}

function recentPicks(view) {
  const from = Date.now() - 14 * 864e5;
  const rows = [];
  for (const r of D.results) {
    if (!r.tips || !matchesFilter(r) || Date.parse(r.date) < from) continue;
    for (const t of r.tips) if (wantKind(t) && r[t.kind]) rows.push({ r, t });
  }
  if (!rows.length) return;
  rows.sort((a, b) => b.r.date.localeCompare(a.r.date));
  const won = rows.filter((x) => x.t.won).length;
  const said = rows.reduce((s, x) => s + x.t.p, 0) / rows.length;
  view.append(h('div', { class: 'section-title' }, 'Picks in the last 14 days'),
    h('p', { class: 'small muted', style: 'margin-top:0' }, `${won} of ${rows.length} won (${pct(won / rows.length)}); on average we gave them ${pct(said)}.`),
    h('div', { class: 'card table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Date', 'Match', 'Pick', 'Chance', 'Result'].map((t, i) => h('th', { class: i < 3 ? 'left' : '' }, t)))),
      h('tbody', {}, rows.slice(0, 150).map(({ r, t }) => h('tr', { class: 'clickable', onclick: () => openResult(r) },
        h('td', { class: 'left muted' }, shortDate(new Date(r.date)), h('span', { class: 'comp-tag' }, compShort(r.comp))),
        h('td', { class: 'left' }, matchName(r), r.goals ? h('span', { class: 'muted' }, ` ${r.goals.home_actual}–${r.goals.away_actual}`) : null),
        h('td', { class: 'left' }, h('span', { class: `key ${t.kind}` }), pickLabel(t)),
        h('td', {}, pct(t.p)),
        h('td', {}, h('span', { class: `verdict ${t.won ? 'ok' : 'miss'}` }, icon(t.won ? 'check' : 'cross'), `${r[t.kind].actual} ${r[t.kind].actual === 1 ? t.kind.slice(0, -1) : t.kind}`))))))));
}

function renderTips(view) {
  const horizon = Date.now() + Number(state.tipsRange) * 864e5;
  const picks = allPicks().filter(({ f, t }) => f.state === 'pre' && wantKind(t) && matchesFilter(f) && Date.parse(f.date) <= horizon);
  const key = { p: (x) => -x.t.p, lift: (x) => x.t.base - x.t.p }[state.tipsSort];
  picks.sort(key ? (a, b) => key(a) - key(b) : (a, b) => a.f.date.localeCompare(b.f.date));

  const seasons = D.meta.metrics.filter((s) => s.tips);
  const rule = seasons.at(-1)?.tips.rule ?? { min_prob: 0.6, min_lift: 0.08 };
  view.append(h('p', { class: 'intro' },
    'The model\'s strongest over/under calls for the coming days. A match gets a pick when our chance for one side of a line is at least ',
    pct(rule.min_prob), ` and at least ${Math.round(rule.min_lift * 100)} points above how often that side lands in the competition (last 12 months). `,
    'Tap + to put picks in your boat and see the chance that they all win.'));

  const sum = (fn) => seasons.reduce((s, x) => s + fn(x.tips), 0);
  const n = sum((x) => x.n), nOdds = sum((x) => x.odds?.n ?? 0);
  const matches = new Set(picks.map((x) => x.f.id)).size;
  view.append(h('div', { class: 'summary-row' },
    tile('Picks', picks.length, `${matches} ${matches === 1 ? 'match' : 'matches'} · next ${state.tipsRange === '1' ? '24 hours' : `${state.tipsRange} days`}`),
    n ? tile(`Past picks won, since ${seasons[0].label}`, pct(sum((x) => x.won) / n),
      `${sum((x) => x.won).toLocaleString()} of ${n.toLocaleString()} · we said ${pct(sum((x) => x.p * x.n) / n)}`) : null,
    nOdds ? tile('Return at bookmaker prices', signedPct(sum((x) => (x.odds ? x.odds.avg_price.roi * x.odds.n : 0)) / nOdds),
      `${nOdds} past over/under 2.5 picks, average price`) : null));

  if (!picks.length) {
    view.append(h('div', { class: 'card empty' }, D.fixtures.some((f) => f.tips?.length)
      ? 'No picks match these filters. Try a longer time window.' : 'No picks for the coming fixtures yet.'));
  } else {
    const head = h('div', { class: 'tip-head', 'aria-hidden': 'true' },
      h('div', {}), h('div', {}, 'Time'), h('div', {}, 'Match'), h('div', {}, 'Pick'), h('div', {}, 'Chance'), h('div', {}, 'Price'));
    if (state.tipsSort !== 'date') {
      view.append(h('div', { class: 'card' }, head, picks.map((x) => tipRow(x, true))));
    } else {
      const days = new Map();
      for (const x of picks) {
        const k = dayKey(new Date(x.f.date));
        if (!days.has(k)) days.set(k, []);
        days.get(k).push(x);
      }
      let first = true;
      for (const xs of days.values()) {
        view.append(h('div', { class: 'day' },
          h('h3', {}, dayLabel(new Date(xs[0].f.date))),
          h('div', { class: 'card' }, first ? head : null, xs.map((x) => tipRow(x, false)))));
        first = false;
      }
    }
  }
  view.append(h('div', { id: 'boat', class: 'card boat', role: 'region', 'aria-label': 'Your boat', hidden: true }));
  renderBoat();
  tipsRecord(view);
  recentPicks(view);
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
    h('div', { class: 'scores' }, g.scores.slice(0, 6).map((s) => h('span', {}, h('b', {}, `${s.h}-${s.a}`), pct(s.p)))));
}

// ------------------------------------------------------------------ prices
const signedPct = (x) => (x == null ? '–' : `${x >= 0 ? '+' : '−'}${Math.abs(x * 100).toFixed(1)}%`);
const VALUE_EDGE = 0.03;

function edgeCell(prob, price) {
  if (!price) return h('td', { class: 'muted' }, '–');
  const ev = prob * price - 1;
  return h('td', { class: ev > VALUE_EDGE ? 'edge-pos' : 'muted' }, signedPct(ev));
}

/** Backtest record of betting the edges this page shows, for the latest season that has one. */
function valueRecord() {
  const s = [...D.meta.metrics].reverse().find((x) => x.market?.best_price?.bets);
  return s ? { season: s.label, ...s.market } : null;
}

function marketPanel(f) {
  const m = f.market;
  const g = f.goals;
  const hn = team(f.home).short || team(f.home).name;
  const an = team(f.away).short || team(f.away).name;
  const p = g.over['2.5'];
  const row = (label, ours, theirs, avg, best) => h('tr', {},
    h('td', { class: 'left' }, label), h('td', {}, h('b', {}, pct(ours))), h('td', {}, pct(theirs)),
    h('td', {}, avg ? f2(avg) : '–'), h('td', {}, best ? f2(best) : '–'), h('td', { class: 'muted' }, odds(ours)), edgeCell(ours, best));
  const rec = valueRecord();
  return h('div', { class: 'card panel' },
    h('h4', {}, 'Against the bookmakers'),
    h('div', { class: 'table-wrap' }, h('table', { class: 'mini-table' },
      h('thead', {}, h('tr', {}, ['', 'Ours', 'Bookmakers', 'Avg odds', 'Best odds', 'Fair odds', 'Edge at best']
        .map((t, i) => h('th', { class: i === 0 ? 'left' : '' }, t)))),
      h('tbody', {},
        row('Over 2.5', p, m.p_over, m.odds.over, m.odds.best_over),
        row('Under 2.5', 1 - p, 1 - m.p_over, m.odds.under, m.odds.best_under),
        row(`${hn} win`, g.hda[0], m.hda[0], m.odds.home, null),
        row('Draw', g.hda[1], m.hda[1], m.odds.draw, null),
        row(`${an} win`, g.hda[2], m.hda[2], m.odds.away, null)))),
    h('p', { class: 'small muted' },
      `"Bookmakers" is the average price with the margin (${pct(m.margin)} on this over/under) removed. `,
      `On its own the model expected ${f2(m.model.total)} goals (${pct(m.model.p_over)} over 2.5); the bookmakers imply ${f2(m.xg[0] + m.xg[1])}. `,
      `The forecast on this page takes ${Math.round(D.model.market_blend * 100)}% from the bookmakers, the mix that scored best on the 2025-26 tuning season.`),
    rec ? h('p', { class: 'small muted' },
      `Edges this size are mostly noise. In ${rec.season}, backing every over/under 2.5 edge above ${Math.round(rec.edge * 100)}% at the best price returned `,
      h('b', {}, signedPct(rec.best_price.roi)), ` over ${rec.best_price.bets} bets (±${Math.round((rec.best_price.roi_se ?? 0) * 100)}% standard error).`) : null);
}

function priceChecker(f) {
  const hn = team(f.home).short || team(f.home).name;
  const an = team(f.away).short || team(f.away).name;
  const over = (x, l) => x.over[l] ?? pOverFromPmf(x.pmf, l);
  const markets = [
    { label: 'Total goals', lines: D.model.goal_lines, def: state.goalLine, p: (l) => over(f.goals, l) },
    { label: 'Total corners', lines: D.model.corner_lines, def: state.cornerLine, p: (l) => over(f.corners, l) },
    { label: `${hn} corners`, lines: D.model.team_corner_lines, def: '4.5', p: (l) => f.corners.home_over[l] },
    { label: `${an} corners`, lines: D.model.team_corner_lines, def: '4.5', p: (l) => f.corners.away_over[l] },
    { label: 'Both teams score', lines: null, p: () => f.goals.btts },
  ];
  const mSel = h('select', { 'aria-label': 'Market' }, markets.map((m, i) => h('option', { value: i }, m.label)));
  const lSel = h('select', { 'aria-label': 'Line' });
  const sSel = h('select', { 'aria-label': 'Side' });
  const price = h('input', { type: 'number', inputmode: 'decimal', min: '1.01', step: '0.01', placeholder: 'e.g. 1.90', 'aria-label': 'Decimal odds' });
  const out = h('div', { class: 'checker-out', role: 'status' });
  const fillLines = () => {
    const m = markets[mSel.value];
    lSel.hidden = !m.lines;
    if (m.lines) lSel.replaceChildren(...m.lines.map((l) => h('option', { value: String(l), selected: String(l) === String(m.def) }, String(l))));
    sSel.replaceChildren(...(m.lines ? ['Over', 'Under'] : ['Yes', 'No']).map((s) => h('option', { value: s }, s)));
  };
  const update = () => {
    const m = markets[mSel.value];
    const q0 = m.p(m.lines ? lSel.value : null);
    const q = sSel.value === 'Over' || sSel.value === 'Yes' ? q0 : 1 - q0;
    const o = Number(price.value);
    if (!(o > 1)) {
      out.replaceChildren(h('span', {}, 'Our chance ', h('b', {}, pct(q)), ` · fair odds ${odds(q)}. Enter a bookmaker's price to compare.`));
      return;
    }
    const ev = q * o - 1;
    const verdict = ev > 0.05 ? ['edge-pos', 'Above our fair price'] : ev > 0 ? ['', 'Slightly above our fair price, within the model\'s error'] : ['muted', 'Below our fair price'];
    out.replaceChildren(
      h('span', {}, 'Our chance ', h('b', {}, pct(q)), ` · the price implies ${pct(1 / o)} · expected return `, h('b', { class: verdict[0] }, signedPct(ev))),
      h('span', { class: `checker-verdict ${verdict[0]}` }, verdict[1]));
  };
  mSel.addEventListener('change', () => { fillLines(); update(); });
  for (const el of [lSel, sSel]) el.addEventListener('change', update);
  price.addEventListener('input', update);
  fillLines();
  update();
  return h('div', { class: 'card panel' },
    h('h4', {}, 'Check a price'),
    h('p', { class: 'small muted', style: 'margin-top:0' }, 'Compare any bookmaker\'s decimal odds with our probability.'),
    h('div', { class: 'checker' }, mSel, lSel, sSel, price),
    out,
    h('p', { class: 'small muted' }, 'Bookmakers often take a bigger margin on corner markets than on goals. Our corner forecasts have been tested against results only: corner odds are not in our data.'));
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
      r.goals?.market_p_over != null ? `Over 2.5 goals: ${pct(r.goals.p_over['2.5'])} from us, ${pct(r.goals.market_p_over)} from the bookmakers. ` : '',
      'The distributions below use the current ratings.')));
  }
  if (f.tips?.length && !f.result) {
    notes.push(h('div', { class: 'note' }, h('span', {},
      h('a', { href: '#tips' }, 'Lên thuyền'), ': ', f.tips.map((t) => `${pickLabel(t)} (${pct(t.p)})`).join(' · '))));
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
    f.result ? null : h('div', { class: 'detail-grid' }, f.market ? marketPanel(f) : null, priceChecker(f)),
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
      venue: scheduled?.venue, confidence: n >= 20 ? 'high' : n >= 8 ? 'medium' : 'low', ...pred,
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

  const mk = season.market;
  if (mk) {
    const row = (label, s, strong) => h('tr', {},
      h('td', { class: 'left' }, strong ? h('b', {}, label) : label),
      h('td', {}, s.log_loss.toFixed(4)), h('td', {}, s.brier.toFixed(4)), h('td', {}, pct(s.hit_rate)));
    const bets = (b) => (b.bets ? `${b.bets} bets, ${signedPct(b.roi)} return (±${Math.round((b.roi_se ?? 0) * 100)}%)` : 'no bets');
    view.append(h('div', { class: 'section-title' }, 'Against the bookmakers'),
      h('div', { class: 'card panel' },
        h('p', { class: 'small muted', style: 'margin-top:0' },
          `Over/under 2.5 goals in the ${mk.n.toLocaleString()} big-five league matches of ${season.label} that had odds. Lower log loss and Brier score are better.`),
        h('div', { class: 'table-wrap' }, h('table', { class: 'mini-table' },
          h('thead', {}, h('tr', {}, ['Forecast', 'Log loss', 'Brier', 'O/U 2.5 correct'].map((t, i) => h('th', { class: i === 0 ? 'left' : '' }, t)))),
          h('tbody', {},
            row('Model alone', mk.model),
            row('Model + odds (what this site shows)', mk.final, true),
            row('Bookmakers, margin removed', mk.market)))),
        h('p', { class: 'small' },
          `Betting every over/under 2.5 edge above ${Math.round(mk.edge * 100)}%: at average prices ${bets(mk.avg_price)}; at the best price across bookmakers ${bets(mk.best_price)}. `,
          `The bookmakers' margin on this market averaged ${pct(mk.margin)}.`),
        h('p', { class: 'small muted' },
          'Bookmaker prices already contain team news, line-ups and the money of professional bettors, so beating them over many bets is hard. ',
          'Treat a single "edge" on this site as a hint, not a sure profit.')));
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
    h('p', {}, 'Each club gets an attack and a defence rating. Expected goals for a side = exp(baseline + competition effect + home advantage + own attack − opponent defence). ',
      `Where a past match had bookmaker odds (${(m.matches_with_odds ?? 0).toLocaleString()} matches in 16 leagues), the ratings are fitted to the expected goals implied by its closing 1X2 and over/under prices rather than to the scoreline, because a price measures team strength with far less luck in it. `,
      'Other matches, such as the UEFA cups, are fitted to their scores. Recent matches count more (half-life ', String(m.params.goals.half_life_days), ' days), and ratings are pulled toward the club\'s league average, which keeps numbers sensible for clubs with few matches. The two expected-goal numbers give a full scoreline grid, with a Dixon–Coles correction for low scores, and the total-goals distribution comes from that grid.'),
    h('h3', {}, 'Bookmaker odds'),
    h('p', {}, 'Odds come from football-data.co.uk, which publishes average and best prices across bookmakers for the main European leagues, collected on Tuesday and Friday before each round. The Champions League is not covered. ',
      `When a fixture has odds, its expected goals are a blend of the model's and the bookmakers' (${Math.round((m.params.market_blend ?? 0) * 100)}% bookmakers), the mix that scored best on 2025-26`,
      m.params.corner_market && (m.params.corner_market.own || m.params.corner_market.opp)
        ? ', and a side the bookmakers rate above the model also gets slightly more expected corners.' : '.'),
    h('h3', {}, 'Corners'),
    h('p', {}, 'The same structure is fitted separately on corners won: a corner-winning rating and a corner-conceding rating per club. Corner totals vary more than a Poisson count allows, so total corners use a negative-binomial distribution with its spread fitted from the data. ESPN records missing box scores as 0–0 corners; those games are treated as missing, and filled from football-data.co.uk where it has the count.'),
    h('h3', {}, 'Testing'),
    h('p', {}, 'Settings were tuned on 2025-26 only. 2026-27 is held out: each week the model is refitted on matches played before that week and then scored on that week\'s games, using only the pre-closing odds that were public a day or more before kick-off. The baseline is the competition\'s own distribution of totals over the previous year. RPS skill is the improvement in ranked probability score over that baseline; higher is better.'),
    h('p', {}, 'Predictions cover regulation time only. Line-ups, injuries and motivation reach the forecast only through bookmaker odds, when a fixture has them, so use these numbers as a starting point, not a guarantee.'),
    h('p', { class: 'muted' }, `Last build: ${new Date(m.generated_at).toLocaleString()} (${m.build_seconds}s). ${m.fixtures} upcoming fixtures predicted.`)));
}

// ------------------------------------------------------------------ shell
function renderView() {
  hideTip();
  const view = document.getElementById('view');
  view.replaceChildren();
  ({ fixtures: renderFixtures, tips: renderTips, results: renderResults, tool: renderTool, teams: renderTeams, model: renderModel })[state.view](view);
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
