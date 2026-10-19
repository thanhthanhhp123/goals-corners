// In-browser copy of the prediction maths, used by the match tool.
// Mirrors predictor/dist.py and MatchPredictor.predict in predictor/model.py;
// tests/test_js_parity.py checks this file against the Python output.

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function poissonPmf(lam, kmax) {
  const l = Math.log(Math.max(lam, 1e-12));
  const out = new Array(kmax + 1);
  for (let k = 0; k <= kmax; k++) out[k] = Math.exp(k * l - lam - lgamma(k + 1));
  return out;
}

export function nbPmf(mean, size, kmax) {
  mean = Math.max(mean, 1e-12);
  const p = size / (size + mean);
  const out = new Array(kmax + 1);
  for (let k = 0; k <= kmax; k++) {
    out[k] = Math.exp(lgamma(k + size) - lgamma(size) - lgamma(k + 1) + size * Math.log(p) + k * Math.log1p(-p));
  }
  return out;
}

const normalise = (pmf) => {
  const s = pmf.reduce((a, b) => a + b, 0);
  return pmf.map((v) => v / s);
};

function countPmf(mean, size, kmax) {
  const pmf = !Number.isFinite(size) || size > 1e5 ? poissonPmf(mean, kmax) : nbPmf(mean, size, kmax);
  return normalise(pmf);
}

export function scoreMatrix(lh, la, rho, gmax) {
  const ph = poissonPmf(lh, gmax);
  const pa = poissonPmf(la, gmax);
  const m = ph.map((x) => pa.map((y) => x * y));
  m[0][0] *= Math.max(1 - lh * la * rho, 0);
  m[0][1] *= Math.max(1 + lh * rho, 0);
  m[1][0] *= Math.max(1 + la * rho, 0);
  m[1][1] *= Math.max(1 - rho, 0);
  let s = 0;
  for (const row of m) for (const v of row) s += v;
  return m.map((row) => row.map((v) => v / s));
}

export function totalFromMatrix(m) {
  const n = m.length;
  const total = new Array(2 * n - 1).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) total[i + j] += m[i][j];
  return total;
}

export function pOver(pmf, line) {
  let s = 0;
  for (let k = Math.floor(line) + 1; k < pmf.length; k++) s += pmf[k];
  return s;
}

function trim(pmf, kmax) {
  const head = pmf.slice(0, kmax);
  head.push(pmf.slice(kmax).reduce((a, b) => a + b, 0));
  return head;
}

function pGreater(a, b) {
  let cdf = 0;
  let s = 0;
  for (let k = 1; k < a.length; k++) {
    cdf += b[k - 1];
    s += a[k] * cdf;
  }
  return s;
}

/** Attack/defence strengths of a team for one model ("goals" or "corners"). */
export function teamStrength(model, teams, kind, id) {
  const t = teams[id];
  if (t && t.r) return kind === 'goals' ? [t.r[0], t.r[1]] : [t.r[2], t.r[3]];
  const km = model[kind];
  const g = (t && t.league) || 'other';
  return [km.g_att[g] ?? 0, km.g_dfn[g] ?? 0];
}

export function rates(model, teams, kind, home, away, comp, neutral) {
  const km = model[kind];
  const [ah, dh] = teamStrength(model, teams, kind, home);
  const [aa, da] = teamStrength(model, teams, kind, away);
  const base = km.mu + (km.alpha[comp] ?? 0);
  const homeAdv = neutral ? 0 : km.home + (km.d_home[comp] ?? 0);
  return [Math.exp(base + homeAdv + ah - da), Math.exp(base + aa - dh)];
}

/** Blend with the market's expected goals [home, away] (mirrors with_market in model.py). */
export function withMarket(model, lh, la, ch, ca, market) {
  if (!market || !market.every((x) => Number.isFinite(x) && x > 0)) return [lh, la, ch, ca];
  const w = model.market_blend ?? 0;
  const k = model.corner_market ?? { own: 0, opp: 0 };
  const dh = Math.log(market[0] / lh), da = Math.log(market[1] / la);
  return [
    lh * Math.exp(w * dh), la * Math.exp(w * da),
    ch * Math.exp(k.own * dh + k.opp * da), ca * Math.exp(k.own * da + k.opp * dh),
  ];
}

/** Full prediction in the same shape as fixtures.json entries. */
export function predictMatch(model, teams, home, away, comp, neutral = false, market = null) {
  const [lh0, la0] = rates(model, teams, 'goals', home, away, comp, neutral);
  const [ch0, ca0] = rates(model, teams, 'corners', home, away, comp, neutral);
  const [lh, la, ch, ca] = withMarket(model, lh0, la0, ch0, ca0, market);
  const m = scoreMatrix(lh, la, model.goals.rho, model.max_goals);
  const goalsTotal = totalFromMatrix(m);
  const cornersTotal = countPmf(ch + ca, model.corners.dispersion, model.max_corners);
  const homeC = countPmf(ch, model.corners.side_dispersion, model.max_corners);
  const awayC = countPmf(ca, model.corners.side_dispersion, model.max_corners);

  const cells = [];
  let hw = 0, dr = 0, aw = 0, btts = 0, homeCs = 0, awayCs = 0;
  m.forEach((row, i) => row.forEach((p, j) => {
    cells.push({ h: i, a: j, p });
    if (i > j) hw += p; else if (i === j) dr += p; else aw += p;
    if (i > 0 && j > 0) btts += p;
    if (j === 0) homeCs += p;
    if (i === 0) awayCs += p;
  }));
  cells.sort((x, y) => y.p - x.p);

  // Lines are all x.5, so String(l) matches the Python keys ("2.5").
  const overK = (pmf, lines) => Object.fromEntries(lines.map((l) => [String(l), pOver(pmf, l)]));
  return {
    goals: {
      home: lh, away: la, total: lh + la,
      pmf: trim(goalsTotal, 9),
      over: overK(goalsTotal, model.goal_lines),
      btts, hda: [hw, dr, aw], cs: [homeCs, awayCs],
      scores: cells.slice(0, 8),
    },
    corners: {
      home: ch, away: ca, total: ch + ca,
      pmf: trim(cornersTotal, 22),
      over: overK(cornersTotal, model.corner_lines),
      home_over: overK(homeC, model.team_corner_lines),
      away_over: overK(awayC, model.team_corner_lines),
      home_more: pGreater(homeC, awayC),
      away_more: pGreater(awayC, homeC),
    },
  };
}
