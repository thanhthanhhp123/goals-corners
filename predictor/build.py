"""Fetch data, fit models, backtest and write the JSON files the web app reads."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest, config, espn, market
from .model import MatchPredictor, fit_predictor

log = logging.getLogger(__name__)

BACKTEST_START = "2025-08-01"  # first origin of the weekly backtest (Mondays from here)
SEASONS = [
    {"id": "2025-26", "label": "2025-26", "start": "2025-07-01", "end": "2026-07-01", "note": "used to choose model settings"},
    {"id": "2026-27", "label": "2026-27", "start": "2026-07-01", "end": "2027-07-01", "note": "held out (never used for tuning)"},
]
FORM_GAMES = 6


def _r(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def _params_key() -> str:
    blob = json.dumps({"g": config.GOALS_PARAMS, "c": config.CORNERS_PARAMS, "b": config.MARKET_BLEND,
                       "cm": config.CORNER_MARKET, "rho": config.MARKET_RHO, "v": 3}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:10]


# ----------------------------------------------------------------- backtest
def cached_backtest(matches: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Weekly rolling backtest, caching windows that are fully in the past."""
    cache_path = config.DATA_DIR / f"backtest_{_params_key()}.pkl"
    cache = pd.read_pickle(cache_path) if cache_path.exists() else pd.DataFrame()
    done = set(cache["origin"]) if not cache.empty else set()
    complete, partial = [], []
    for t0 in pd.date_range(pd.Timestamp(BACKTEST_START, tz="UTC"), now, freq="7D"):
        t1 = t0 + pd.Timedelta(days=7)
        is_complete = t1 < now - pd.Timedelta(days=2)
        if is_complete and t0 in done:
            continue
        preds = backtest.rolling(matches, t0, min(t1, now), step_days=7)
        if preds.empty:
            continue
        preds["origin"] = t0
        # Windows still in progress are used for this build but never cached.
        (complete if is_complete else partial).append(preds)
    if complete:
        cache = pd.concat([f for f in [cache, *complete] if not f.empty], ignore_index=True)
        cache.to_pickle(cache_path)
    frames = [f for f in [cache, *partial] if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("event_id", keep="last").sort_values("date").reset_index(drop=True)


def season_metrics(preds: pd.DataFrame) -> list[dict]:
    out = []
    for season in SEASONS:
        sel = preds[(preds["date"] >= pd.Timestamp(season["start"], tz="UTC")) & (preds["date"] < pd.Timestamp(season["end"], tz="UTC"))]
        if sel.empty:
            continue
        entry = {**season, "goals": _round_tree(backtest.summarise(sel, "goals")), "corners": _round_tree(backtest.summarise(sel, "corners")),
                 "market": _round_tree(market_metrics(sel)), "by_comp": {}}
        for comp in config.TARGET_COMPETITIONS:
            sc = sel[sel["comp"] == comp]
            if len(sc) >= 10:
                entry["by_comp"][comp] = {
                    "goals": _round_tree(_light(backtest.summarise(sc, "goals"))),
                    "corners": _round_tree(_light(backtest.summarise(sc, "corners"))),
                }
        out.append(entry)
    return out


VALUE_EDGE = 0.03  # expected return a bet needs before the backtest "places" it


def _binary_scores(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": float(-np.mean(np.where(y, np.log(p), np.log(1 - p)))),
            "brier": float(np.mean((p - y) ** 2)), "hit_rate": float(np.mean((p > 0.5) == y))}


def value_bets(p: np.ndarray, y: np.ndarray, over: np.ndarray, under: np.ndarray, edge: float) -> dict:
    """Flat one-unit bets on whichever side of O/U 2.5 has expected return above `edge`."""
    ev_o, ev_u = p * over - 1, (1 - p) * under - 1
    take_o = (ev_o >= ev_u) & (ev_o > edge)
    take_u = (ev_u > ev_o) & (ev_u > edge)
    ret = np.concatenate([np.where(y[take_o], over[take_o] - 1, -1.0), np.where(~y[take_u], under[take_u] - 1, -1.0)])
    if len(ret) == 0:
        return {"bets": 0}
    return {"bets": int(len(ret)), "won": float(np.mean(ret > 0)), "roi": float(ret.mean()),
            "roi_se": float(ret.std(ddof=1) / np.sqrt(len(ret))) if len(ret) > 1 else None}


def market_metrics(sel: pd.DataFrame) -> dict | None:
    """Over/under 2.5 goals: model, model+odds and the bookmakers, on matches that had odds."""
    if "mkt_mkp_p_over" not in sel:
        return None
    df = sel.dropna(subset=["mkt_mkp_p_over", "goals_actual", "goals_model_p_over_2.5"])
    if len(df) < 30:
        return None
    y = df["goals_actual"].to_numpy() > 2.5
    p_final = df["goals_p_over_2.5"].to_numpy()
    out = {
        "n": int(len(df)),
        "margin": float(df["mkt_mk_margin_ou"].mean()),
        "model": _binary_scores(df["goals_model_p_over_2.5"].to_numpy(), y),
        "final": _binary_scores(p_final, y),
        "market": _binary_scores(df["mkt_mkp_p_over"].to_numpy(), y),
        "edge": VALUE_EDGE,
        "avg_price": value_bets(p_final, y, df["mkt_odds_over"].to_numpy(), df["mkt_odds_under"].to_numpy(), VALUE_EDGE),
        "best_price": value_bets(p_final, y, df["mkt_best_over"].to_numpy(), df["mkt_best_under"].to_numpy(), VALUE_EDGE),
    }
    return out


def _light(summary: dict) -> dict:
    s = dict(summary)
    lines = s.pop("lines", {})
    s["lines"] = {k: {kk: vv for kk, vv in v.items() if kk != "calibration"} for k, v in lines.items()}
    return s


def _round_tree(obj, nd=4):
    if isinstance(obj, dict):
        return {k: _round_tree(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_tree(v, nd) for v in obj]
    if isinstance(obj, float):
        return _r(obj, nd)
    return obj


# ----------------------------------------------------------------- exports
def team_form(played: pd.DataFrame) -> dict[str, list[dict]]:
    long = pd.concat(
        [
            pd.DataFrame({"team": played["home_id"], "opp": played["away_id"], "venue": "H", "gf": played["home_goals"], "ga": played["away_goals"],
                          "cf": played["home_corners"], "ca": played["away_corners"], "date": played["date"], "comp": played["comp"]}),
            pd.DataFrame({"team": played["away_id"], "opp": played["home_id"], "venue": "A", "gf": played["away_goals"], "ga": played["home_goals"],
                          "cf": played["away_corners"], "ca": played["home_corners"], "date": played["date"], "comp": played["comp"]}),
        ]
    ).sort_values("date")
    form: dict[str, list[dict]] = {}
    for team, grp in long.groupby("team"):
        rows = grp.tail(FORM_GAMES).iloc[::-1]
        form[team] = [
            {"date": r.date.strftime("%Y-%m-%d"), "comp": r.comp, "opp": r.opp, "venue": r.venue, "gf": int(r.gf), "ga": int(r.ga),
             "cf": None if pd.isna(r.cf) else int(r.cf), "ca": None if pd.isna(r.ca) else int(r.ca)}
            for r in rows.itertuples(index=False)
        ]
    return form


def recent_averages(played: pd.DataFrame, n: int = 10) -> dict[str, dict]:
    long = pd.concat(
        [
            pd.DataFrame({"team": played["home_id"], "gf": played["home_goals"], "ga": played["away_goals"], "cf": played["home_corners"], "ca": played["away_corners"], "date": played["date"]}),
            pd.DataFrame({"team": played["away_id"], "gf": played["away_goals"], "ga": played["home_goals"], "cf": played["away_corners"], "ca": played["home_corners"], "date": played["date"]}),
        ]
    ).sort_values("date")
    out = {}
    for team, grp in long.groupby("team"):
        g = grp.tail(n)
        out[team] = {"n": int(len(g)), "gf": _r(g["gf"].mean(), 2), "ga": _r(g["ga"].mean(), 2), "cf": _r(g["cf"].mean(), 2), "ca": _r(g["ca"].mean(), 2)}
    return out


def _prediction_json(pred: dict) -> dict:
    g, c = pred["goals"], pred["corners"]
    return {
        "goals": {
            "home": _r(g["home"]), "away": _r(g["away"]), "total": _r(g["total"]),
            "pmf": [_r(p, 4) for p in g["pmf"]], "over": {k: _r(v, 4) for k, v in g["over"].items()},
            "btts": _r(g["btts"], 4), "hda": [_r(g["home_win"], 4), _r(g["draw"], 4), _r(g["away_win"], 4)],
            "cs": [_r(g["home_cs"], 4), _r(g["away_cs"], 4)],
            "scores": [{"h": s["h"], "a": s["a"], "p": _r(s["p"], 4)} for s in g["scores"]],
        },
        "corners": {
            "home": _r(c["home"]), "away": _r(c["away"]), "total": _r(c["total"]),
            "pmf": [_r(p, 4) for p in c["pmf"]], "over": {k: _r(v, 4) for k, v in c["over"].items()},
            "home_over": {k: _r(v, 4) for k, v in c["home_over"].items()},
            "away_over": {k: _r(v, 4) for k, v in c["away_over"].items()},
            "home_more": _r(c["home_more"], 4), "away_more": _r(c["away_more"], 4),
        },
    }


def _market_json(predictor: MatchPredictor, row, mk) -> dict | None:
    """Bookmaker view of a fixture next to the model's own (pre-blend) numbers."""
    if mk is None:
        return None
    alone = predictor.predict(row.home_id, row.away_id, row.comp, bool(row.neutral))
    price = lambda x: None if pd.isna(x) else _r(x, 2)  # noqa: E731
    return {
        "xg": [_r(mk[0], 4), _r(mk[1], 4)],
        "p_over": _r(row.mkp_p_over, 4),
        "hda": [_r(row.mkp_p_home, 4), _r(row.mkp_p_draw, 4), _r(row.mkp_p_away, 4)],
        "odds": {"over": price(row.odds_over), "under": price(row.odds_under),
                 "best_over": price(row.best_over), "best_under": price(row.best_under),
                 "home": price(row.odds_home), "draw": price(row.odds_draw), "away": price(row.odds_away)},
        "margin": _r(row.mk_margin_ou, 4),
        "model": {"total": _r(alone["goals"]["total"]), "p_over": _r(alone["goals"]["over"]["2.5"], 4),
                  "corners": _r(alone["corners"]["total"])},
    }


def _confidence(predictor: MatchPredictor, home: str, away: str) -> str:
    n = min(predictor.corners.n_eff.get(home, 0.0), predictor.corners.n_eff.get(away, 0.0),
            predictor.goals.n_eff.get(home, 0.0), predictor.goals.n_eff.get(away, 0.0))
    if n >= 20:
        return "high"
    if n >= 8:
        return "medium"
    return "low"


def _known_team(predictor: MatchPredictor, team_id: str, names: dict[str, str]) -> bool:
    name = names.get(team_id, "")
    placeholder = any(tok in name.lower() for tok in ("tbd", "winner", "loser", "runner-up", "to be determined"))
    return team_id in predictor.goals.att and not placeholder


def build(force_fetch: bool = True) -> dict:
    t_start = time.time()
    now = pd.Timestamp(datetime.now(timezone.utc))
    log.info("fetching data")
    matches, teams = espn.load_dataset(force=force_fetch)
    log.info("fetching odds")
    odds = market.market_table(matches, teams, rho=config.MARKET_RHO, force=force_fetch)
    matches = market.attach(matches, odds)
    names = dict(zip(teams["team_id"], teams["name"]))
    played = matches[espn.played_mask(matches)]

    log.info("fitting models on %d played matches", len(played))
    predictor = fit_predictor(matches, now)

    log.info("predicting fixtures")
    upcoming = matches[
        matches["comp"].isin(config.TARGET_COMPETITIONS)
        & matches["state"].isin(["pre", "in"])
        & ~matches["status"].isin(["STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_ABANDONED"])
        & (matches["date"] >= now - pd.Timedelta(hours=4))
    ]
    fixtures = []
    for row in upcoming.itertuples(index=False):
        if not (_known_team(predictor, row.home_id, names) and _known_team(predictor, row.away_id, names)):
            continue
        mk = (row.mkp_home, row.mkp_away)
        mk = mk if np.all(np.isfinite(mk)) else None
        pred = predictor.predict(row.home_id, row.away_id, row.comp, bool(row.neutral), market=mk)
        fixtures.append({
            "id": row.event_id, "comp": row.comp, "date": row.date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "state": row.state, "status": row.status, "neutral": bool(row.neutral), "venue": row.venue,
            "home": row.home_id, "away": row.away_id,
            "confidence": _confidence(predictor, row.home_id, row.away_id),
            "market": _market_json(predictor, row, mk),
            **_prediction_json(pred),
        })

    log.info("backtesting")
    preds = cached_backtest(matches, now)
    metrics = season_metrics(preds)

    results = []
    recent = preds[preds["date"] >= pd.Timestamp(SEASONS[-1]["start"], tz="UTC") - pd.Timedelta(days=60)]
    # to_dict, not itertuples: columns like "goals_p_over_2.5" are not valid attribute names
    for r in recent.to_dict("records"):
        rec = {"id": r["event_id"], "comp": r["comp"], "date": r["date"].strftime("%Y-%m-%dT%H:%M:%SZ"), "home": r["home_id"], "away": r["away_id"],
               "neutral": bool(r["neutral"])}
        for kind in ("goals", "corners"):
            actual = r.get(f"{kind}_actual")
            if actual is None or pd.isna(actual):
                continue
            rec[kind] = {
                "pred": _r(r[f"{kind}_pred"]), "home_pred": _r(r[f"{kind}_home_pred"]), "away_pred": _r(r[f"{kind}_away_pred"]),
                "actual": int(actual), "home_actual": int(r[f"{kind}_home_actual"]), "away_actual": int(r[f"{kind}_away_actual"]),
                "p_over": {str(l): _r(r[f"{kind}_p_over_{l}"], 4) for l in backtest.EVAL_LINES[kind]},
            }
        p_mkt = r.get("mkt_mkp_p_over")
        if p_mkt is not None and not pd.isna(p_mkt) and "goals" in rec:
            rec["goals"]["market_p_over"] = _r(p_mkt, 4)
        results.append(rec)

    log.info("exporting")
    form = team_form(played)
    averages = recent_averages(played)
    needed = set()
    for f in fixtures:
        needed.update([f["home"], f["away"]])
    for r in results:
        needed.update([r["home"], r["away"]])
    for f in form.values():
        needed.update(x["opp"] for x in f)
    # every club that played in a target competition recently is selectable in the match predictor
    recent_target = played[played["comp"].isin(config.TARGET_COMPETITIONS) & (played["date"] >= now - pd.Timedelta(days=400))]
    selectable = set(recent_target["home_id"]) | set(recent_target["away_id"])
    needed |= selectable
    team_rows = teams.set_index("team_id")
    teams_out = {}
    for tid in sorted(needed):
        if tid not in team_rows.index:
            continue
        info = team_rows.loc[tid]
        group = predictor.goals.groups.get(tid, "other")
        comp_ctx = group if group in config.ALL_COMPETITIONS else "uefa.champions"
        gf, ga = predictor.goals.team_profile(tid, comp_ctx)
        cf, ca = predictor.corners.team_profile(tid, comp_ctx)
        entry = {
            "name": info["name"], "short": info["short_name"], "abbr": info["abbr"], "logo": info["logo"],
            "league": group, "selectable": tid in selectable,
        }
        if tid in selectable or tid in {f["home"] for f in fixtures} | {f["away"] for f in fixtures}:
            ga_att, ga_dfn = predictor.goals.team_strength(tid)
            co_att, co_dfn = predictor.corners.team_strength(tid)
            entry.update({
                "r": [_r(ga_att, 4), _r(ga_dfn, 4), _r(co_att, 4), _r(co_dfn, 4)],
                "profile": {"gf": _r(gf, 2), "ga": _r(ga, 2), "cf": _r(cf, 2), "ca": _r(ca, 2)},
                "n_eff": _r(min(predictor.goals.n_eff.get(tid, 0.0), predictor.corners.n_eff.get(tid, 0.0)), 1),
                "recent": averages.get(tid),
                "form": form.get(tid, []),
            })
        teams_out[tid] = entry

    competitions = {**{k: {**v, "target": True} for k, v in config.TARGET_COMPETITIONS.items()},
                    **{k: {**v, "target": False} for k, v in config.SUPPORT_COMPETITIONS.items()}}
    model_json = {
        "goals": {k: _round_tree(v, 5) for k, v in predictor.goals.export().items()},
        "corners": {k: _round_tree(v, 5) for k, v in predictor.corners.export().items()},
        "market_blend": config.MARKET_BLEND, "corner_market": config.CORNER_MARKET,
        "max_goals": config.MAX_GOALS, "max_corners": config.MAX_CORNERS,
        "goal_lines": config.GOAL_LINES, "corner_lines": config.CORNER_LINES, "team_corner_lines": config.TEAM_CORNER_LINES,
    }
    meta = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_seconds": round(time.time() - t_start, 1),
        "matches_total": int(len(matches)), "matches_played": int(len(played)),
        "matches_with_corners": int(played["home_corners"].notna().sum()),
        "fixtures": len(fixtures), "competitions": competitions,
        "matches_with_odds": int(played["mk_home"].notna().sum()),
        "fixtures_with_odds": sum(1 for f in fixtures if f["market"]),
        "params": {"goals": config.GOALS_PARAMS, "corners": config.CORNERS_PARAMS,
                   "market_blend": config.MARKET_BLEND, "corner_market": config.CORNER_MARKET},
        "metrics": metrics,
    }
    config.WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "model": model_json, "teams": teams_out, "fixtures": fixtures, "results": results}
    for name, obj in payload.items():
        (config.WEB_DATA_DIR / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    log.info("done: %d fixtures, %d results, %d teams in %.0fs", len(fixtures), len(results), len(teams_out), time.time() - t_start)
    return meta
