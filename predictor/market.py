"""Bookmaker odds from football-data.co.uk, matched to ESPN events.

football-data.co.uk publishes free CSVs with results and odds for the main
European leagues: one file per league and season, plus `fixtures.csv` with the
coming round. Each file carries two snapshots of the odds. The "pre-closing"
prices are collected on Friday afternoon for weekend games and Tuesday for
midweek games, and the closing prices are taken just before kick-off.

From the 1X2 and over/under 2.5 prices we back out the expected goals of each
side (the market's lambdas) by fitting the same Dixon-Coles score grid the
model uses. Those numbers are used in two ways:

- past matches: as the goals model's training target, because a price is a far
  less noisy measure of team strength than the scoreline;
- coming matches: blended with the model's own forecast.
"""
from __future__ import annotations

import io
import logging
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
HEADERS = {"User-Agent": "corner-goal-predictor/1.0"}

# ESPN slug -> football-data division code
FD_CODES = {
    "eng.1": "E0", "eng.2": "E1", "esp.1": "SP1", "esp.2": "SP2", "ita.1": "I1", "ita.2": "I2",
    "ger.1": "D1", "ger.2": "D2", "fra.1": "F1", "fra.2": "F2", "ned.1": "N1", "bel.1": "B1",
    "por.1": "P1", "tur.1": "T1", "gre.1": "G1", "sco.1": "SC0",
}
CODE_TO_COMP = {v: k for k, v in FD_CODES.items()}

# Bookmaker columns in order of preference. "Avg" is the market average, PS/P is
# Pinnacle (dropped by the site in 2025), BFE is the Betfair exchange.
X12_SOURCES = {"pre": ["Avg", "B365", "PS", "BFE"], "close": ["AvgC", "B365C", "PSC", "BFEC"]}
OU_SOURCES = {"pre": ["Avg", "B365", "P", "BFE"], "close": ["AvgC", "B365C", "PC", "BFEC"]}

MARKET_COLUMNS = [
    "mk_home", "mk_away",            # expected goals implied by the closing (else pre-closing) odds
    "mkp_home", "mkp_away",          # the same from the pre-closing odds only
    "mkp_p_home", "mkp_p_draw", "mkp_p_away", "mkp_p_over",  # pre-closing no-vig probabilities
    "odds_over", "odds_under", "best_over", "best_under",    # pre-closing average and best O/U 2.5 prices
    "odds_home", "odds_draw", "odds_away",                   # pre-closing average 1X2 prices
    "best_home", "best_draw", "best_away",                   # pre-closing best 1X2 prices
    "ah_line", "odds_ah_home", "odds_ah_away", "best_ah_home", "best_ah_away",  # pre-closing Asian handicap (home line)
    "mk_margin_ou", "fd_home_corners", "fd_away_corners",
]


# ---------------------------------------------------------------- download
def season_codes(now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    current = now.year if now.month >= 7 else now.year - 1
    return [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(config.FIRST_YEAR - 1, current + 1)]


def _read_csv(raw: bytes) -> pd.DataFrame:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    # a few historical files have trailing commas or ragged rows
    return pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)


def _download(url: str) -> bytes | None:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            log.warning("%s failed (%s)", url, exc)
    return None


def fetch(force: bool = False) -> pd.DataFrame:
    """All cached or downloaded odds files as one table (results and coming fixtures)."""
    out_dir = config.DATA_DIR / "market"
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = season_codes()
    jobs = [(s, code) for s in seasons for code in FD_CODES.values()] + [("fixtures", None)]

    def run(job):
        season, code = job
        path = out_dir / (f"{season}_{code}.csv" if code else "fixtures.csv")
        live = code is None or season == seasons[-1]
        if path.exists() and not (force and live):
            return job, path.read_bytes()
        raw = _download(FIXTURES_URL if code is None else SEASON_URL.format(season=season, code=code))
        if raw:
            path.write_bytes(raw)
        elif path.exists():
            raw = path.read_bytes()
        return job, raw

    frames = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for (season, code), raw in pool.map(run, jobs):
            if not raw:
                continue
            df = _read_csv(raw)
            if "Div" not in df.columns:
                df = df.rename(columns={df.columns[0]: "Div"})
            df = df[df["Div"].isin(CODE_TO_COMP)]
            frames.append(df.assign(source=season))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    df["comp"] = df["Div"].map(CODE_TO_COMP)
    df["day"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["day"])
    # a fixture also appears in its season file once played; keep the played copy
    df["played"] = df["FTHG"].notna() if "FTHG" in df else False
    df = df.sort_values("played").drop_duplicates(["comp", "day", "HomeTeam", "AwayTeam"], keep="last")
    return df.reset_index(drop=True)


# ----------------------------------------------------------- odds handling
def _first(df: pd.DataFrame, cols: list[list[str]]) -> np.ndarray:
    """Row-wise first complete set of price columns, as an (n, k) array (NaN if none)."""
    out = np.full((len(df), len(cols[0])), np.nan)
    for group in cols:
        if not all(c in df.columns for c in group):
            continue
        vals = df[group].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ok = np.all(np.isfinite(vals) & (vals > 1.0), axis=1) & ~np.all(np.isfinite(out), axis=1)
        out[ok] = vals[ok]
    return out


def _x12(df, when):
    return _first(df, [[f"{p}H", f"{p}D", f"{p}A"] for p in X12_SOURCES[when]])


def _ou(df, when):
    return _first(df, [[f"{p}>2.5", f"{p}<2.5"] for p in OU_SOURCES[when]])


def no_vig(prices: np.ndarray) -> np.ndarray:
    """Proportional removal of the bookmaker margin."""
    q = 1.0 / prices
    return q / q.sum(axis=1, keepdims=True)


def _grid_probs(lh: np.ndarray, la: np.ndarray, rho: float, gmax: int = config.MAX_GOALS) -> np.ndarray:
    """P(home win), P(draw), P(away win), P(over 2.5) from a Dixon-Coles grid, vectorised."""
    k = np.arange(gmax + 1)
    from scipy.special import gammaln

    ph = np.exp(k * np.log(lh[:, None]) - lh[:, None] - gammaln(k + 1))
    pa = np.exp(k * np.log(la[:, None]) - la[:, None] - gammaln(k + 1))
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= np.maximum(1 - lh * la * rho, 0)
    m[:, 0, 1] *= np.maximum(1 + lh * rho, 0)
    m[:, 1, 0] *= np.maximum(1 + la * rho, 0)
    m[:, 1, 1] *= max(1 - rho, 0)
    m /= m.sum(axis=(1, 2), keepdims=True)
    i, j = np.indices((gmax + 1, gmax + 1))
    home = (m * (i > j)).sum(axis=(1, 2))
    draw = (m * (i == j)).sum(axis=(1, 2))
    over = (m * (i + j > 2)).sum(axis=(1, 2))
    return np.stack([home, draw, 1 - home - draw, over], axis=1)


def implied_goals(p_x12: np.ndarray, p_over: np.ndarray, rho: float, iters: int = 40) -> np.ndarray:
    """Expected goals (home, away) whose score grid best reproduces the market probabilities.

    Rows with no 1X2 prices get NaN. Without an over/under price the fit uses
    the 1X2 alone, which pins the supremacy well and the total only loosely.
    """
    n = len(p_x12)
    target = np.column_stack([p_x12, p_over])
    wts = np.column_stack([np.ones((n, 3)), np.where(np.isfinite(p_over), 1.0, 0.0)])
    target = np.nan_to_num(target)
    ok = np.all(np.isfinite(p_x12), axis=1)
    x = np.tile(np.log([1.45, 1.15]), (n, 1))
    eps = 1e-4
    lam = 1e-3 * np.ones(n)
    for _ in range(iters):
        f0 = _grid_probs(np.exp(x[:, 0]), np.exp(x[:, 1]), rho)
        r = (f0 - target) * wts
        jh = (_grid_probs(np.exp(x[:, 0] + eps), np.exp(x[:, 1]), rho) - f0) / eps * wts
        ja = (_grid_probs(np.exp(x[:, 0]), np.exp(x[:, 1] + eps), rho) - f0) / eps * wts
        a11, a12, a22 = (jh * jh).sum(1) + lam, (jh * ja).sum(1), (ja * ja).sum(1) + lam
        g1, g2 = (jh * r).sum(1), (ja * r).sum(1)
        det = a11 * a22 - a12 ** 2
        step = np.column_stack([(a22 * g1 - a12 * g2) / det, (a11 * g2 - a12 * g1) / det])
        x = np.clip(x - np.clip(step, -0.5, 0.5), np.log(0.05), np.log(8.0))
    out = np.exp(x)
    out[~ok] = np.nan
    return out


# ------------------------------------------------------------ name matching
_DROP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "sv", "vfb", "vfl", "fk", "sk", "cd", "ud", "rc", "club", "de", "1."}


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = s.replace("'", "").replace(".", " ").replace("-", " ")
    return " ".join(t for t in s.split() if t not in _DROP)


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_events(market: pd.DataFrame, matches: pd.DataFrame, teams: pd.DataFrame) -> pd.Series:
    """ESPN event_id for each market row (NaN when no confident match)."""
    names = {}
    for t in teams.itertuples(index=False):
        names[t.team_id] = {_norm(t.name), _norm(t.short_name)} - {""}
    espn = matches[matches["comp"].isin(FD_CODES)].copy()
    espn["day"] = espn["date"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()

    cache: dict[tuple[str, str], float] = {}

    def name_sim(fd_name: str, team_id: str) -> float:
        key = (fd_name, team_id)
        if key not in cache:
            n = _norm(fd_name)
            cache[key] = max((_sim(n, x) for x in names.get(team_id, ())), default=0.0)
        return cache[key]

    # 1) played games: same competition, date within a day and same score -> learn the names
    played = market[market["played"]].copy()
    played["hg"] = pd.to_numeric(played["FTHG"], errors="coerce")
    played["ag"] = pd.to_numeric(played["FTAG"], errors="coerce")
    cand = []
    for shift in (-1, 0, 1):
        p = played.assign(day=played["day"] + pd.Timedelta(days=shift)).reset_index()
        cand.append(p.merge(espn[["event_id", "comp", "day", "home_id", "away_id", "home_goals", "away_goals"]],
                            on=["comp", "day"]))
    cand = pd.concat(cand, ignore_index=True)
    cand = cand[(cand["hg"] == cand["home_goals"]) & (cand["ag"] == cand["away_goals"])]
    votes: dict[str, Counter] = defaultdict(Counter)
    if not cand.empty:
        cand["s"] = [name_sim(h, hi) + name_sim(a, ai)
                     for h, a, hi, ai in zip(cand["HomeTeam"], cand["AwayTeam"], cand["home_id"], cand["away_id"])]
        best = cand.sort_values("s").drop_duplicates("index", keep="last")
        best = best[best["s"] >= 0.8]
        for r in best.itertuples(index=False):
            votes[r.HomeTeam][r.home_id] += 1
            votes[r.AwayTeam][r.away_id] += 1
    alias = {k: v.most_common(1)[0][0] for k, v in votes.items()}

    # 2) names never seen in a played game (new fixtures): best fuzzy match within the competition
    recent = espn[espn["date"] >= espn["date"].max() - pd.Timedelta(days=400)]
    comp_teams = {c: set(g["home_id"]) | set(g["away_id"]) for c, g in recent.groupby("comp")}
    for r in market.itertuples(index=False):
        for nm in (r.HomeTeam, r.AwayTeam):
            if nm in alias:
                continue
            pool = comp_teams.get(r.comp, set())
            scored = sorted(((name_sim(nm, t), t) for t in pool), reverse=True)
            if scored and scored[0][0] >= 0.75:
                alias[nm] = scored[0][1]

    # 3) join on the two clubs and the date (the season files and ESPN may differ by a day)
    m = market.assign(home_id=market["HomeTeam"].map(alias), away_id=market["AwayTeam"].map(alias)).reset_index()
    m = m.dropna(subset=["home_id", "away_id"])
    joined = m.merge(espn[["event_id", "home_id", "away_id", "day"]], on=["home_id", "away_id"], suffixes=("", "_e"))
    joined = joined[(joined["day"] - joined["day_e"]).abs() <= pd.Timedelta(days=3)]
    joined = joined.drop_duplicates("index").set_index("index")
    return joined["event_id"].reindex(market.index)


# ------------------------------------------------------------------ public
def market_table(matches: pd.DataFrame, teams: pd.DataFrame, rho: float = -0.03, force: bool = False) -> pd.DataFrame:
    """Market columns (see MARKET_COLUMNS) indexed by ESPN event_id."""
    market = fetch(force=force)
    if market.empty:
        return pd.DataFrame(columns=MARKET_COLUMNS)
    market["event_id"] = match_events(market, matches, teams)
    market = market.dropna(subset=["event_id"]).drop_duplicates("event_id", keep="last").reset_index(drop=True)

    x12_pre, ou_pre = _x12(market, "pre"), _ou(market, "pre")
    x12_cl, ou_cl = _x12(market, "close"), _ou(market, "close")
    p_x12_pre, p_ou_pre = no_vig(x12_pre), no_vig(ou_pre)[:, 0]
    # closing prices where complete, else pre-closing
    use_cl = np.all(np.isfinite(x12_cl), axis=1)
    x12_best = np.where(use_cl[:, None], x12_cl, x12_pre)
    ou_best = np.where(np.all(np.isfinite(ou_cl), axis=1)[:, None], ou_cl, ou_pre)
    lam_best = implied_goals(no_vig(x12_best), no_vig(ou_best)[:, 0], rho)
    lam_pre = implied_goals(p_x12_pre, p_ou_pre, rho)
    best_ou = _first(market, [["Max>2.5", "Max<2.5"]])
    best_x12 = _first(market, [["MaxH", "MaxD", "MaxA"]])
    ah = _first(market, [["AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA"]])

    out = pd.DataFrame({
        "event_id": market["event_id"].astype(str),
        "mk_home": lam_best[:, 0], "mk_away": lam_best[:, 1],
        "mkp_home": lam_pre[:, 0], "mkp_away": lam_pre[:, 1],
        "mkp_p_home": p_x12_pre[:, 0], "mkp_p_draw": p_x12_pre[:, 1], "mkp_p_away": p_x12_pre[:, 2],
        "mkp_p_over": p_ou_pre,
        "odds_over": ou_pre[:, 0], "odds_under": ou_pre[:, 1],
        "best_over": best_ou[:, 0], "best_under": best_ou[:, 1],
        "odds_home": x12_pre[:, 0], "odds_draw": x12_pre[:, 1], "odds_away": x12_pre[:, 2],
        "best_home": best_x12[:, 0], "best_draw": best_x12[:, 1], "best_away": best_x12[:, 2],
        "ah_line": pd.to_numeric(market.get("AHh"), errors="coerce"),
        "odds_ah_home": ah[:, 0], "odds_ah_away": ah[:, 1], "best_ah_home": ah[:, 2], "best_ah_away": ah[:, 3],
        "mk_margin_ou": (1 / ou_pre).sum(axis=1) - 1,
        # football-data also records corners, which fills ESPN's gaps
        "fd_home_corners": pd.to_numeric(market.get("HC"), errors="coerce"),
        "fd_away_corners": pd.to_numeric(market.get("AC"), errors="coerce"),
    })
    log.info("market odds matched for %d events", len(out))
    return out.set_index("event_id")


def attach(matches: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Matches with the market columns joined on and ESPN's missing corner counts filled in."""
    out = matches.copy()
    key = out["event_id"].astype(str)
    for col in MARKET_COLUMNS:
        out[col] = key.map(table[col]).astype(float) if col in table else np.nan
    fd_ok = out["fd_home_corners"].notna() & out["fd_away_corners"].notna() & \
        (out["fd_home_corners"] + out["fd_away_corners"] > 0)
    fill = fd_ok & (out["home_corners"].isna() | out["away_corners"].isna())
    out.loc[fill, "home_corners"] = out.loc[fill, "fd_home_corners"]
    out.loc[fill, "away_corners"] = out.loc[fill, "fd_away_corners"]
    log.info("filled corners for %d matches from football-data", int(fill.sum()))
    return out
