// Small SVG charts: probability distributions and calibration plots.
// Labels are always set with textContent (team and league names come from an API).

const SVG = 'http://www.w3.org/2000/svg';

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, str, attrs = {}) {
  const t = el('text', { x, y, ...attrs }, parent);
  t.textContent = str;
  return t;
}

let tooltip;
function getTooltip() {
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.setAttribute('role', 'status');
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

function showTip(target, value, label) {
  const tip = getTooltip();
  tip.replaceChildren();
  const v = document.createElement('strong');
  v.textContent = value;
  const l = document.createElement('span');
  l.textContent = label;
  tip.append(v, l);
  tip.style.display = 'block';
  const r = target.getBoundingClientRect();
  const tr = tip.getBoundingClientRect();
  let left = r.left + r.width / 2 - tr.width / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - tr.width - 8));
  let top = r.top - tr.height - 8;
  if (top < 8) top = r.bottom + 8;
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}

export function hideTip() {
  if (tooltip) tooltip.style.display = 'none';
}

function barPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  const y0 = y + h;
  if (h <= 0.5) return `M${x},${y0}H${x + w}`;
  return `M${x},${y0}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y0}Z`;
}

const pct = (p) => `${(p * 100).toFixed(p >= 0.095 || p === 0 ? 0 : 1)}%`;

/**
 * Column chart of a count distribution.
 * pmf: probabilities for 0..n-1, last entry is the "n-1 or more" bucket.
 * line: the over/under line (e.g. 2.5); bars above it get the full series colour.
 */
export function distributionChart(pmf, { line, series, unit, ariaLabel }) {
  const n = pmf.length;
  const W = 520, H = 190;
  const m = { t: 14, r: 8, b: 26, l: 34 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const maxP = Math.max(...pmf);
  const step = maxP > 0.3 ? 0.1 : maxP > 0.12 ? 0.05 : 0.02;
  const top = Math.max(step, Math.ceil(maxP / step) * step);

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': ariaLabel });
  const grid = el('g', { class: 'grid' }, svg);
  for (let v = 0; v <= top + 1e-9; v += step) {
    const y = m.t + ih - (v / top) * ih;
    el('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: v === 0 ? 'baseline' : 'gridline' }, grid);
    text(grid, m.l - 6, y + 3.5, `${Math.round(v * 100)}%`, { class: 'tick', 'text-anchor': 'end' });
  }

  const band = iw / n;
  const bw = Math.min(24, band - 2);
  const bars = el('g', {}, svg);
  pmf.forEach((p, k) => {
    const x = m.l + k * band + (band - bw) / 2;
    const h = (p / top) * ih;
    const isOver = line != null && k > line;
    const label = k === n - 1 ? `${k}+` : String(k);
    const g = el('g', { class: `bar ${series} ${isOver ? 'over' : 'under'}`, tabindex: 0 }, bars);
    el('rect', { x: m.l + k * band, y: m.t, width: band, height: ih, class: 'hit' }, g);
    el('path', { d: barPath(x, m.t + ih - h, bw, h, 4), class: 'mark' }, g);
    const showLabel = n <= 12 || k % 2 === 0 || k === n - 1;
    if (showLabel) text(svg, x + bw / 2, H - m.b + 15, label, { class: 'tick', 'text-anchor': 'middle' });
    const tipLabel = `exactly ${label} ${unit}`.replace(`exactly ${k}+`, `${k} or more`);
    const enter = () => showTip(g, pct(p), tipLabel);
    g.addEventListener('pointerenter', enter);
    g.addEventListener('focus', enter);
    g.addEventListener('pointerleave', hideTip);
    g.addEventListener('blur', hideTip);
  });

  if (line != null) {
    const x = m.l + (Math.floor(line) + 1) * band;
    el('line', { x1: x, x2: x, y1: m.t - 6, y2: m.t + ih, class: 'refline' }, svg);
    text(svg, x + 4, m.t + 2, `line ${line}`, { class: 'tick reflabel' });
  }
  return svg;
}

/** Reliability diagram: predicted probability vs observed frequency, one series. */
export function calibrationChart(bins, { series, ariaLabel }) {
  const W = 300, H = 260;
  const m = { t: 12, r: 12, b: 34, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const lo = 0.1, hi = 0.9;
  const sx = (v) => m.l + ((v - lo) / (hi - lo)) * iw;
  const sy = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': ariaLabel });
  const grid = el('g', { class: 'grid' }, svg);
  for (let v = 0.1; v <= 0.9 + 1e-9; v += 0.2) {
    el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: v < 0.11 ? 'baseline' : 'gridline' }, grid);
    el('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ih, class: 'gridline' }, grid);
    text(grid, m.l - 6, sy(v) + 3.5, `${Math.round(v * 100)}%`, { class: 'tick', 'text-anchor': 'end' });
    text(grid, sx(v), H - m.b + 15, `${Math.round(v * 100)}%`, { class: 'tick', 'text-anchor': 'middle' });
  }
  text(svg, m.l + iw / 2, H - 4, 'Predicted probability', { class: 'axis-title', 'text-anchor': 'middle' });
  text(svg, 10, m.t + ih / 2, 'Observed', { class: 'axis-title', 'text-anchor': 'middle', transform: `rotate(-90 10 ${m.t + ih / 2})` });
  el('line', { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi), class: 'refline' }, svg);
  const pts = bins.filter((b) => b.p >= lo && b.p <= hi);
  if (pts.length > 1) {
    el('polyline', { points: pts.map((b) => `${sx(b.p)},${sy(Math.min(hi, Math.max(lo, b.freq)))}`).join(' '), class: `series-line ${series}` }, svg);
  }
  for (const b of pts) {
    const g = el('g', { class: `dot ${series}`, tabindex: 0 }, svg);
    const cx = sx(b.p), cy = sy(Math.min(hi, Math.max(lo, b.freq)));
    el('circle', { cx, cy, r: 12, class: 'hit' }, g);
    el('circle', { cx, cy, r: 4.5, class: 'mark' }, g);
    const enter = () => showTip(g, `${pct(b.freq)} observed`, `predicted ${pct(b.p)} · ${b.n} matches`);
    g.addEventListener('pointerenter', enter);
    g.addEventListener('focus', enter);
    g.addEventListener('pointerleave', hideTip);
    g.addEventListener('blur', hideTip);
  }
  return svg;
}
