"""Pick suggestions for the "Lên thuyền" view: the model's strongest over/under calls.

A pick is one side of a goals or corners over/under line. For each match and each
market we keep the side whose probability is furthest above the competition's own
rate at that line over the previous year, provided it is at least MIN_PROB likely
and at least MIN_LIFT above that rate. Comparing with the competition's rate, rather
than taking the most likely side, skips calls like "under 11.5 corners" that hold in
most matches anyway and keeps the picks that say something about this match.

The same rule runs over the weekly backtest, so the site can show how the picks did
when they were made before kick-off. The thresholds were set for a manageable number
of picks (about 25 a week across the six competitions), not tuned for results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dist
from .backtest import EVAL_LINES, baseline

LINES = EVAL_LINES  # goals 1.5-3.5, corners 8.5-11.5: the lines the backtest records
MIN_PROB = 0.60
MAX_PROB = 0.85  # above this the price is too short to be worth listing
MIN_LIFT = 0.08
TIERS = [0.60, 0.65, 0.70, 0.75]


def pick(kind: str, p_over: dict[float, float], base_over: dict[float, float]) -> dict | None:
    """Best pick for one market, or None. Both dicts map line -> P(total > line)."""
    best = None
    for line in LINES[kind]:
        q, b = p_over.get(line), base_over.get(line)
        if q is None or b is None or not (np.isfinite(q) and np.isfinite(b)):
            continue
        for side, p, pb in (("over", q, b), ("under", 1 - q, 1 - b)):
            lift = p - pb
            if MIN_PROB <= p <= MAX_PROB and lift >= MIN_LIFT and (best is None or lift > best["lift"]):
                best = {"kind": kind, "line": line, "side": side, "p": float(p), "base": float(pb), "lift": float(lift)}
    return best


def won(t: dict, actual: int) -> bool:
    return actual > t["line"] if t["side"] == "over" else actual < t["line"]


# ----------------------------------------------------------------- live fixtures
def base_rates(played: pd.DataFrame, comps, now: pd.Timestamp) -> dict[str, dict[str, dict[float, float]]]:
    """Each competition's over-rate at every pick line, from the 365 days before `now`
    (the same baseline the backtest uses)."""
    history = played[(played["date"] < now) & (played["date"] >= now - pd.Timedelta(days=365))]
    return {comp: {kind: {line: dist.p_over(baseline(history, kind, comp)[0], line) for line in LINES[kind]}
                   for kind in LINES} for comp in comps}


def fixture_tips(pred: dict, base: dict[str, dict[float, float]]) -> list[dict]:
    """Picks for one fixture from its prediction JSON (whose `over` keys are strings)."""
    out = []
    for kind in LINES:
        over = {float(k): v for k, v in pred[kind]["over"].items() if v is not None}
        t = pick(kind, over, base[kind])
        if t:
            out.append(t)
    return out


# ----------------------------------------------------------------- track record
def history(preds: pd.DataFrame) -> pd.DataFrame:
    """The picks the rule would have made in the backtest, with their results.

    Over/under 2.5 goals picks on matches that had odds also carry the average and
    best pre-closing price of the picked side.
    """
    rows = []
    for r in preds.to_dict("records"):
        for kind in LINES:
            actual = r.get(f"{kind}_actual")
            if actual is None or pd.isna(actual):
                continue
            t = pick(kind, {l: r.get(f"{kind}_p_over_{l}") for l in LINES[kind]},
                     {l: r.get(f"{kind}_base_p_over_{l}") for l in LINES[kind]})
            if t is None:
                continue
            price = best = np.nan
            if kind == "goals" and t["line"] == 2.5 and pd.notna(r.get("mkt_odds_over")):
                price = r["mkt_odds_over"] if t["side"] == "over" else r["mkt_odds_under"]
                best = r["mkt_best_over"] if t["side"] == "over" else r["mkt_best_under"]
            rows.append({"event_id": r["event_id"], "date": r["date"], "comp": r["comp"], **t,
                         "actual": int(actual), "won": won(t, int(actual)), "price": price, "best": best})
    return pd.DataFrame(rows)


def _returns(won: np.ndarray, price: np.ndarray) -> dict:
    ret = np.where(won, price - 1, -1.0)
    return {"roi": float(ret.mean()), "roi_se": float(ret.std(ddof=1) / np.sqrt(len(ret))) if len(ret) > 1 else None}


def summarise(h: pd.DataFrame) -> dict | None:
    """Hit rates, stated vs observed chance by tier, and returns where real prices exist."""
    if h.empty:
        return None
    days = (h["date"].max() - h["date"].min()).days + 1
    out = {
        "n": int(len(h)), "won": int(h["won"].sum()), "p": float(h["p"].mean()), "hit": float(h["won"].mean()),
        "per_week": float(len(h) / max(days / 7, 1.0)),
        "by_kind": {k: {"n": int(len(g)), "p": float(g["p"].mean()), "hit": float(g["won"].mean())}
                    for k, g in h.groupby("kind")},
        "tiers": [],
        "rule": {"min_prob": MIN_PROB, "max_prob": MAX_PROB, "min_lift": MIN_LIFT},
    }
    edges = [*TIERS, 1.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = h[(h["p"] >= lo) & (h["p"] < hi)]
        if len(g):
            out["tiers"].append({"lo": lo, "hi": hi, "n": int(len(g)), "p": float(g["p"].mean()), "hit": float(g["won"].mean())})
    priced = h.dropna(subset=["price", "best"])
    if len(priced):
        w = priced["won"].to_numpy()
        out["odds"] = {"n": int(len(priced)), "hit": float(w.mean()),
                       "avg_price": _returns(w, priced["price"].to_numpy()),
                       "best_price": _returns(w, priced["best"].to_numpy())}
    return out
