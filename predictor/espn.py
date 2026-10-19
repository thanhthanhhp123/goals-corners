"""Download ESPN scoreboards and normalise them into match and team tables."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

REQUEST_LIMIT = 1000
HEADERS = {"User-Agent": "corner-goal-predictor/1.0"}


def _cache_path(slug: str, year: int) -> Path:
    return config.RAW_DIR / slug / f"{year}.json"


def _get_json(url: str, params: dict, attempts: int = 4) -> dict:
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == attempts - 1:
                raise
            wait = 2 ** attempt
            log.warning("%s %s failed (%s), retrying in %ss", url, params, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _fetch_events(slug: str, period: str) -> list[dict]:
    """Fetch all events of a competition for a calendar year (YYYY) or month (YYYYMM)."""
    url = config.ESPN_SCOREBOARD.format(slug=slug)
    data = _get_json(url, {"dates": period, "limit": REQUEST_LIMIT})
    events = data.get("events", [])
    if len(events) >= REQUEST_LIMIT and len(period) == 4:
        # The year hit the response cap: fall back to month-by-month queries.
        events = []
        for month in range(1, 13):
            events.extend(_fetch_events(slug, f"{period}{month:02d}"))
    return events


KEEP_STATS = {"wonCorners", "totalShots", "shotsOnTarget", "possessionPct"}


def _slim_event(event: dict) -> dict:
    """Keep only the fields `normalise` reads; raw scoreboards are ~20x larger."""
    comps = []
    for comp in event.get("competitions") or []:
        competitors = []
        for side in comp.get("competitors", []):
            team = side.get("team", {}) or {}
            competitors.append(
                {
                    "homeAway": side.get("homeAway"),
                    "score": side.get("score"),
                    "team": {k: team.get(k) for k in ("id", "displayName", "shortDisplayName", "name", "abbreviation", "logo", "color")},
                    "statistics": [
                        {"name": s.get("name"), "displayValue": s.get("displayValue")}
                        for s in side.get("statistics", []) or []
                        if s.get("name") in KEEP_STATS
                    ],
                }
            )
        comps.append(
            {
                "neutralSite": comp.get("neutralSite"),
                "venue": {"fullName": (comp.get("venue") or {}).get("fullName")},
                "status": {"type": ((comp.get("status") or {}).get("type") or {})},
                "odds": [{"overUnder": (o or {}).get("overUnder")} for o in comp.get("odds", []) or []],
                "competitors": competitors,
            }
        )
    return {
        "id": event.get("id"),
        "date": event.get("date"),
        "season": event.get("season"),
        "status": {"type": ((event.get("status") or {}).get("type") or {})},
        "competitions": comps,
    }


def fetch_year(slug: str, year: int, force: bool = False) -> list[dict]:
    """Return the raw events of one competition-year, using the on-disk cache.

    Years that ended before the current one are cached for good; the current
    and following years are re-downloaded whenever `force` is set.
    """
    path = _cache_path(slug, year)
    current_year = datetime.now(timezone.utc).year
    if path.exists() and not (force and year >= current_year):
        return json.loads(path.read_text(encoding="utf-8"))
    events = [_slim_event(e) for e in _fetch_events(slug, str(year))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events), encoding="utf-8")
    return events


def fetch_all(force: bool = False, workers: int = 8) -> dict[str, list[dict]]:
    """Download every configured competition for every year of the data window."""
    current_year = datetime.now(timezone.utc).year
    years = range(config.FIRST_YEAR, current_year + 2)
    jobs = [(slug, year) for slug in config.ALL_COMPETITIONS for year in years]

    def run(job):
        slug, year = job
        try:
            return job, fetch_year(slug, year, force=force)
        except Exception as exc:  # one bad feed should not sink the whole update
            log.error("could not fetch %s %s: %s", slug, year, exc)
            cached = _cache_path(slug, year)
            if cached.exists():
                return job, json.loads(cached.read_text(encoding="utf-8"))
            return job, []

    by_slug: dict[str, list[dict]] = {slug: [] for slug in config.ALL_COMPETITIONS}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (slug, _year), events in pool.map(run, jobs):
            by_slug[slug].extend(events)
    return by_slug


def _stat(competitor: dict, name: str) -> float | None:
    for stat in competitor.get("statistics", []) or []:
        if stat.get("name") == name:
            value = stat.get("displayValue")
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _score(competitor: dict) -> float | None:
    try:
        return float(competitor.get("score"))
    except (TypeError, ValueError):
        return None


def _goals_line(competition: dict) -> float | None:
    for odds in competition.get("odds", []) or []:
        line = (odds or {}).get("overUnder")
        if isinstance(line, (int, float)):
            return float(line)
    return None


def normalise(by_slug: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn raw ESPN events into a `matches` frame and a `teams` frame."""
    rows = []
    teams: dict[str, dict] = {}
    for slug, events in by_slug.items():
        for event in events:
            comps = event.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
            home, away = sides.get("home"), sides.get("away")
            if not home or not away:
                continue
            status = (comp.get("status") or event.get("status") or {}).get("type", {})
            for side in (home, away):
                team = side.get("team", {})
                tid = str(team.get("id", ""))
                if tid and tid not in teams:
                    teams[tid] = {
                        "team_id": tid,
                        "name": team.get("displayName") or team.get("name") or tid,
                        "short_name": team.get("shortDisplayName") or team.get("displayName") or tid,
                        "abbr": team.get("abbreviation") or "",
                        "logo": team.get("logo") or "",
                        "color": team.get("color") or "",
                    }
            rows.append(
                {
                    "event_id": str(event.get("id")),
                    "comp": slug,
                    "season": (event.get("season") or {}).get("slug") or "",
                    "season_year": (event.get("season") or {}).get("year"),
                    "date": event.get("date") or comp.get("date"),
                    "status": status.get("name", ""),
                    "state": status.get("state", ""),
                    "completed": bool(status.get("completed")),
                    "neutral": bool(comp.get("neutralSite")),
                    "venue": (comp.get("venue") or {}).get("fullName") or "",
                    "home_id": str(home.get("team", {}).get("id", "")),
                    "away_id": str(away.get("team", {}).get("id", "")),
                    "home_goals": _score(home),
                    "away_goals": _score(away),
                    "home_corners": _stat(home, "wonCorners"),
                    "away_corners": _stat(away, "wonCorners"),
                    "home_shots": _stat(home, "totalShots"),
                    "away_shots": _stat(away, "totalShots"),
                    "market_goals_line": _goals_line(comp),
                }
            )
    matches = pd.DataFrame(rows)
    if matches.empty:
        return matches, pd.DataFrame(list(teams.values()))
    matches["date"] = pd.to_datetime(matches["date"], utc=True, errors="coerce")
    matches = matches.dropna(subset=["date"])
    matches = matches[(matches["home_id"] != "") & (matches["away_id"] != "")]
    matches = matches.drop_duplicates("event_id", keep="last").sort_values("date").reset_index(drop=True)
    # ESPN reports 0-0 when it has no box score (e.g. most UEFA qualifiers and
    # whole seasons of some leagues). A real match with no corners at all is
    # vanishingly rare, so treat 0-0 as missing rather than as data.
    for a, b in (("home_corners", "away_corners"), ("home_shots", "away_shots")):
        missing = (matches[a] == 0) & (matches[b] == 0)
        matches.loc[missing, [a, b]] = np.nan
    return matches, pd.DataFrame(list(teams.values()))


def played_mask(matches: pd.DataFrame) -> pd.Series:
    """Matches that finished in regulation time with a valid score."""
    return (
        matches["status"].isin(config.FINISHED_STATUSES)
        & matches["completed"]
        & matches["home_goals"].notna()
        & matches["away_goals"].notna()
    )


def upcoming_mask(matches: pd.DataFrame, now: pd.Timestamp) -> pd.Series:
    return (matches["state"] == "pre") & (matches["status"] == "STATUS_SCHEDULED") & (matches["date"] >= now - pd.Timedelta(hours=3))


def load_dataset(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_slug = fetch_all(force=force)
    matches, teams = normalise(by_slug)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(config.DATA_DIR / "matches.csv", index=False)
    teams.to_csv(config.DATA_DIR / "teams.csv", index=False)
    return matches, teams
