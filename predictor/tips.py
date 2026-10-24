"""Pick suggestions for the "Lên thuyền" view: the model's strongest calls.

A market is a set of selections, one of which the match settles: the sides of an
over/under line, home/draw/away, both teams to score yes/no and so on. For each
match and each market we keep the selection whose probability is furthest above
how often it lands in the competition over the previous year, provided it is at
least MIN_PROB likely and at least MIN_LIFT above that rate. Comparing with the
competition's rate, rather than taking the most likely selection, skips calls like
"under 11.5 corners" that hold in most matches anyway and keeps the picks that say
something about this match.

The same rule runs over the weekly backtest, so the site can show how the picks did
when they were made before kick-off. The thresholds were set for a manageable number
of picks, not tuned for results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PROB = 0.60
MAX_PROB = 0.85  # above this the price is too short to be worth listing
MIN_LIFT = 0.08
TIERS = [0.60, 0.65, 0.70, 0.75]


def _ou(*lines):
    return [(side, line) for line in lines for side in ("over", "under")]


HDA = [("home", None), ("draw", None), ("away", None)]
# kind -> (what settles it, its selections as (side, line))
MARKETS = {
    "goals": ("ft", _ou(1.5, 2.5, 3.5)),
    "corners": ("corners", _ou(8.5, 9.5, 10.5, 11.5)),
    "result": ("ft", HDA),
    "btts": ("ft", [("yes", None), ("no", None)]),
    # Asian handicap on half-goal lines: "home -1.5" wins when the home side wins by 2 or more
    "handicap": ("ft", [("home", -1.5), ("away", 1.5), ("away", -1.5), ("home", 1.5)]),
    "ht_result": ("ht", HDA),
    "ht_goals": ("ht", _ou(0.5, 1.5)),
    "cards": ("cards", _ou(2.5, 3.5, 4.5, 5.5, 6.5)),  # yellow and red cards, one each
}
# the (home, away) counts that settle each source, as matches/backtest columns
COUNTS = {
    "ft": ("home_goals", "away_goals"),
    "ht": ("home_ht_goals", "away_ht_goals"),
    "corners": ("home_corners", "away_corners"),
    "cards": ("home_cards", "away_cards"),
}
# bookmaker prices in our data (average, best), as `market` columns
PRICES = {
    ("goals", "over", 2.5): ("odds_over", "best_over"), ("goals", "under", 2.5): ("odds_under", "best_under"),
    ("result", "home", None): ("odds_home", "best_home"), ("result", "draw", None): ("odds_draw", "best_draw"),
    ("result", "away", None): ("odds_away", "best_away"),
}


def hit(side: str, line: float | None, h, a):
    """Whether a selection wins given the home and away counts (arrays work too).
    web/app.js mirrors this function."""
    if side == "over":
        return h + a > line
    if side == "under":
        return h + a < line
    if side == "yes":
        return (h > 0) & (a > 0)
    if side == "no":
        return (h == 0) | (a == 0)
    if side == "draw":
        return h == a
    margin = h - a if side == "home" else a - h
    return margin + (-0.5 if line is None else line) > 0


def probs(ft: np.ndarray, ht: np.ndarray | None, corners: np.ndarray | None, cards: np.ndarray | None) -> dict:
    """{(kind, side, line): P(win)} from the full-time and half-time score grids and the
    corner and card total pmfs (None skips the markets that need it)."""
    i, j = np.indices(ft.shape)
    out = {}
    for kind, (src, sels) in MARKETS.items():
        grid = {"ft": ft, "ht": ht, "corners": corners, "cards": cards}[src]
        if grid is None:
            continue
        # a total pmf settles over/unders on its own: treat it as a grid with no away column
        h, a = (i, j) if grid.ndim == 2 else (np.arange(len(grid)), 0)
        for side, line in sels:
            out[(kind, side, line)] = float(grid[hit(side, line, h, a)].sum())
    return out


def base_rates(history: pd.DataFrame, comp: str) -> dict:
    """How often every selection won in `comp` (all competitions if it has under 30 games)."""
    out = {}
    for kind, (src, sels) in MARKETS.items():
        cols = list(COUNTS[src])
        hist = history.loc[history["comp"] == comp, cols].dropna()
        if len(hist) < 30:
            hist = history[cols].dropna()
        if hist.empty:
            continue
        h, a = hist[cols[0]].to_numpy(), hist[cols[1]].to_numpy()
        for side, line in sels:
            out[(kind, side, line)] = float(hit(side, line, h, a).mean())
    return out


def pick(kind: str, p: dict, base: dict) -> dict | None:
    """Best pick for one market, or None. Both dicts map (kind, side, line) -> probability."""
    best = None
    for side, line in MARKETS[kind][1]:
        q, b = p.get((kind, side, line)), base.get((kind, side, line))
        if q is None or b is None or not (np.isfinite(q) and np.isfinite(b)):
            continue
        lift = q - b
        if MIN_PROB <= q <= MAX_PROB and lift >= MIN_LIFT and (best is None or lift > best["lift"]):
            best = {"kind": kind, "line": line, "side": side, "p": float(q), "base": float(b), "lift": float(lift)}
    return best


def picks(p: dict, base: dict) -> list[dict]:
    """At most one pick per market."""
    return [t for t in (pick(kind, p, base) for kind in MARKETS) if t]


def won(t: dict, h, a) -> bool:
    return bool(hit(t["side"], t["line"], h, a))


# ----------------------------------------------------------------- track record
def key(kind: str, side: str, line) -> str:
    """Backtest column suffix for a selection."""
    return f"{kind}|{side}|{line}"


def history(preds: pd.DataFrame) -> pd.DataFrame:
    """The picks the rule would have made in the backtest, with their results.

    Picks on a market with prices in our data (over/under 2.5 goals, 1X2) on matches
    that had odds also carry the average and best pre-closing price. (football-data has
    only the bookmakers' main Asian handicap line, which is never a ±1.5 pick's line.)
    """
    rows = []
    for r in preds.to_dict("records"):
        p = {sel: r.get(f"p|{key(*sel)}") for sel in _selections()}
        base = {sel: r.get(f"b|{key(*sel)}") for sel in _selections()}
        for kind, (src, _) in MARKETS.items():
            h, a = (r.get(c) for c in COUNTS[src])
            if h is None or a is None or pd.isna(h) or pd.isna(a):
                continue
            t = pick(kind, p, base)
            if t is None:
                continue
            price = best = np.nan
            cols = PRICES.get((kind, t["side"], t["line"]))
            if cols and pd.notna(r.get(f"mkt_{cols[0]}")):
                price, best = r[f"mkt_{cols[0]}"], r.get(f"mkt_{cols[1]}", np.nan)
            rows.append({"event_id": r["event_id"], "date": r["date"], "comp": r["comp"], **t,
                         "won": won(t, h, a), "price": price, "best": best})
    return pd.DataFrame(rows)


def _selections():
    return [(kind, side, line) for kind, (_, sels) in MARKETS.items() for side, line in sels]


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
        out["odds"]["by_kind"] = {k: {"n": int(len(g)), **_returns(g["won"].to_numpy(), g["price"].to_numpy())}
                                  for k, g in priced.groupby("kind")}
    return out
