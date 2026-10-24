"""Rolling-origin evaluation: refit on data before each origin, predict the next window.

Every prediction is made only from matches played before it, so the numbers
are honest out-of-sample results. The baseline is the empirical distribution
of match totals in the same competition over the previous 365 days.
"""
from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pandas as pd

from . import config, dist, tips
from .espn import played_mask
from .model import RateModel, _count_pmf, fit_half, half_matrix, team_groups, with_market

log = logging.getLogger(__name__)

KMAX = {"goals": 20, "corners": config.MAX_CORNERS, "cards": config.MAX_CARDS}
COLS = {"goals": ("home_goals", "away_goals"), "corners": ("home_corners", "away_corners"), "cards": ("home_cards", "away_cards")}
MARKET_COLS = {"goals": ("mk_home", "mk_away")}
MARKET_RECORD = ["mkp_home", "mkp_away", "mkp_p_over", "odds_over", "odds_under", "best_over", "best_under", "mk_margin_ou",
                 "odds_home", "odds_draw", "odds_away", "best_home", "best_draw", "best_away"]
EVAL_LINES = {"goals": [1.5, 2.5, 3.5], "corners": [8.5, 9.5, 10.5, 11.5], "cards": [3.5, 4.5, 5.5]}
DEFAULT_PARAMS = {"goals": config.GOALS_PARAMS, "corners": config.CORNERS_PARAMS, "cards": config.CARDS_PARAMS}


def _total_pmf(kind: str, model: RateModel, lh: float, la: float) -> np.ndarray:
    if kind == "goals":
        return dist.total_from_matrix(dist.score_matrix(lh, la, model.rho, config.MAX_GOALS))[: KMAX[kind] + 1]
    return _count_pmf(lh + la, model.dispersion, KMAX[kind])


def _baseline_pmf(totals: np.ndarray, kmax: int) -> np.ndarray:
    counts = np.bincount(np.clip(totals.astype(int), 0, kmax), minlength=kmax + 1).astype(float)
    counts += 0.5
    return counts / counts.sum()


def baseline(history: pd.DataFrame, kind: str, comp: str) -> tuple[np.ndarray, float]:
    """Distribution and mean of match totals in `comp` (all competitions if it has under 30 matches)."""
    yh, ya = COLS[kind]
    hist = history[history["comp"] == comp][[yh, ya]].dropna().sum(axis=1).to_numpy()
    if len(hist) < 30:
        hist = history[[yh, ya]].dropna().sum(axis=1).to_numpy()
    return _baseline_pmf(hist, KMAX[kind]), float(hist.mean())


def _rps(pmf: np.ndarray, k: int) -> float:
    cdf = np.cumsum(pmf)
    obs = (np.arange(len(pmf)) >= k).astype(float)
    return float(np.mean((cdf - obs) ** 2))


def rolling(
    matches: pd.DataFrame,
    start: str,
    end: str | pd.Timestamp,
    step_days: int = 7,
    kinds: tuple[str, ...] = ("goals", "corners", "cards"),
    params: dict[str, dict] | None = None,
    comps: list[str] | None = None,
    blend: float | None = None,
    corner_market: dict | None = None,
) -> pd.DataFrame:
    """Out-of-sample predictions for target-competition matches in [start, end).

    Matches with pre-closing odds (collected a day or more before kick-off) are
    blended with them exactly as live fixtures are; `goals_model_*` columns keep
    the model-only numbers. With all three kinds, `p|...` and `b|...` columns hold every
    pick selection's probability and its competition rate (see tips.MARKETS).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    comps = comps or list(config.TARGET_COMPETITIONS)
    played = matches[played_mask(matches)]
    start_ts = pd.Timestamp(start, tz="UTC") if not isinstance(start, pd.Timestamp) else start
    end_ts = pd.Timestamp(end, tz="UTC") if not isinstance(end, pd.Timestamp) else end
    origins = pd.date_range(start_ts, end_ts, freq=f"{step_days}D")
    warm: dict[str, RateModel | None] = {k: None for k in kinds}
    records = []
    for t0 in origins:
        t1 = min(t0 + pd.Timedelta(days=step_days), end_ts)
        window = played[(played["date"] >= t0) & (played["date"] < t1) & played["comp"].isin(comps)]
        if window.empty:
            continue
        groups = team_groups(matches, t0)
        history = played[(played["date"] < t0) & (played["date"] >= t0 - pd.Timedelta(days=365))]
        baselines: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        sel_bases: dict[str, dict] = {}
        ht = fit_half(played, t0)

        fitted = {}
        for kind in kinds:
            yh, ya = COLS[kind]
            m = RateModel.from_params(params[kind]).fit(played, yh, ya, t0, groups, warm=warm[kind],
                                                        market=MARKET_COLS.get(kind))
            if kind == "goals":
                m.fit_rho()
            fitted[kind] = warm[kind] = m
        for row in window.itertuples(index=False):
            rec = {
                "event_id": row.event_id, "date": row.date, "comp": row.comp,
                "home_id": row.home_id, "away_id": row.away_id, "neutral": row.neutral,
                **{c: getattr(row, c) for cols in tips.COUNTS.values() for c in cols},
            }
            raw = {k: fitted[k].rates(row.home_id, row.away_id, row.comp, bool(row.neutral)) for k in kinds}
            mk = (getattr(row, "mkp_home", np.nan), getattr(row, "mkp_away", np.nan))
            mk = mk if np.all(np.isfinite(mk)) else None
            g0 = raw.get("goals", (1.0, 1.0))
            c0 = raw.get("corners", (1.0, 1.0))
            if "goals" not in raw:
                mk = None  # the corner adjustment is relative to the goals model
            g1, c1 = with_market(*g0, *c0, mk, blend=blend, corner=corner_market)
            adjusted = {**raw, **{k: v for k, v in (("goals", g1), ("corners", c1)) if k in raw}}
            if mk is not None:
                rec.update({f"mkt_{c}": getattr(row, c) for c in MARKET_RECORD})
            for kind in kinds:
                yh, ya = COLS[kind]
                ah, aa = getattr(row, yh), getattr(row, ya)
                if pd.isna(ah) or pd.isna(aa):
                    continue
                model = fitted[kind]
                lh, la = adjusted[kind]
                pmf = _total_pmf(kind, model, lh, la)
                if kind == "goals":
                    pmf0 = _total_pmf(kind, model, *raw[kind])
                    rec["goals_model_home"], rec["goals_model_away"] = raw[kind]
                    rec["goals_model_pred"] = sum(raw[kind])
                    for line in EVAL_LINES[kind]:
                        rec[f"goals_model_p_over_{line}"] = dist.p_over(pmf0, line)
                if (kind, row.comp) not in baselines:
                    baselines[(kind, row.comp)] = baseline(history, kind, row.comp)
                base, base_mean = baselines[(kind, row.comp)]
                actual = int(ah + aa)
                kk = min(actual, KMAX[kind])
                rec.update({
                    f"{kind}_home_pred": lh, f"{kind}_away_pred": la, f"{kind}_pred": lh + la,
                    f"{kind}_base": base_mean, f"{kind}_actual": actual,
                    f"{kind}_home_actual": int(ah), f"{kind}_away_actual": int(aa),
                    f"{kind}_logp": float(np.log(max(pmf[kk], 1e-12))),
                    f"{kind}_base_logp": float(np.log(base[kk])),
                    f"{kind}_rps": _rps(pmf, kk), f"{kind}_base_rps": _rps(base, kk),
                })
                for line in EVAL_LINES[kind]:
                    rec[f"{kind}_p_over_{line}"] = dist.p_over(pmf, line)
                    rec[f"{kind}_base_p_over_{line}"] = dist.p_over(base, line)
            if set(kinds) >= {"goals", "corners", "cards"}:
                (lh, la), (ch, ca), (kh, ka) = adjusted["goals"], adjusted["corners"], adjusted["cards"]
                p = tips.probs(dist.score_matrix(lh, la, fitted["goals"].rho, config.MAX_GOALS), half_matrix(lh, la, ht),
                               _total_pmf("corners", fitted["corners"], ch, ca), _total_pmf("cards", fitted["cards"], kh, ka))
                if row.comp not in sel_bases:
                    sel_bases[row.comp] = tips.base_rates(history, row.comp)
                rec.update({f"p|{tips.key(*k)}": v for k, v in p.items()})
                rec.update({f"b|{tips.key(*k)}": v for k, v in sel_bases[row.comp].items()})
            records.append(rec)
    return pd.DataFrame(records)


def summarise(preds: pd.DataFrame, kind: str) -> dict:
    """Headline accuracy metrics for one target (goals or corners), model vs baseline."""
    df = preds.dropna(subset=[f"{kind}_actual"])
    if df.empty:
        return {}
    y = df[f"{kind}_actual"].to_numpy(float)
    pred, base = df[f"{kind}_pred"].to_numpy(), df[f"{kind}_base"].to_numpy()
    out = {
        "n": int(len(df)),
        "mae": float(np.mean(np.abs(pred - y))), "base_mae": float(np.mean(np.abs(base - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "base_rmse": float(np.sqrt(np.mean((base - y) ** 2))),
        "log_score": float(df[f"{kind}_logp"].mean()), "base_log_score": float(df[f"{kind}_base_logp"].mean()),
        "rps": float(df[f"{kind}_rps"].mean()), "base_rps": float(df[f"{kind}_base_rps"].mean()),
        "mean_pred": float(pred.mean()), "mean_actual": float(y.mean()),
        "corr": float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 else None,
        "lines": {},
    }
    for line in EVAL_LINES[kind]:
        p, pb = df[f"{kind}_p_over_{line}"].to_numpy(), df[f"{kind}_base_p_over_{line}"].to_numpy()
        o = (y > line).astype(float)
        out["lines"][str(line)] = {
            "brier": float(np.mean((p - o) ** 2)), "base_brier": float(np.mean((pb - o) ** 2)),
            "hit_rate": float(np.mean((p > 0.5) == (o > 0.5))),
            "base_hit_rate": float(np.mean((pb > 0.5) == (o > 0.5))),
            "over_rate": float(o.mean()),
            "calibration": calibration(p, o),
        }
    return out


def calibration(p: np.ndarray, o: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() >= 15:
            out.append({"p": float(p[sel].mean()), "freq": float(o[sel].mean()), "n": int(sel.sum())})
    return out


def tune(matches: pd.DataFrame, kind: str, start: str = "2025-08-01", end: str = "2026-06-15",
         step_days: int = 14, grid: dict | None = None) -> pd.DataFrame:
    """Grid search over hyperparameters on a validation window, scored by mean log score."""
    grid = grid or ({
        "market_weight": [3.0, 10.0, 30.0],
        "half_life_days": [60.0, 90.0, 180.0],
        "lam_team": [5.0, 10.0, 25.0],
    } if kind == "goals" else {
        "half_life_days": [180.0, 270.0, 365.0, 540.0],
        "lam_team": [20.0, 40.0, 80.0, 160.0],
    })
    base = dict(DEFAULT_PARAMS[kind])
    keys = list(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**base, **dict(zip(keys, combo))}
        t = time.time()
        preds = rolling(matches, start, end, step_days, kinds=(kind,), params={kind: params})
        s = summarise(preds, kind)
        rows.append({**dict(zip(keys, combo)), "log_score": s["log_score"], "rps": s["rps"], "mae": s["mae"],
                     "base_log_score": s["base_log_score"], "n": s["n"]})
        log.info("%s %s -> log %.4f rps %.4f mae %.3f (%.0fs)", kind, dict(zip(keys, combo)), s["log_score"], s["rps"], s["mae"], time.time() - t)
    return pd.DataFrame(rows).sort_values("log_score", ascending=False)
