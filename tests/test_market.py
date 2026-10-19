"""Odds handling: margin removal, expected goals backed out of prices, event matching."""
import numpy as np
import pandas as pd

from predictor import market
from predictor.model import with_market


def test_no_vig_sums_to_one():
    prices = np.array([[1.90, 1.90], [1.50, 2.60]])
    p = market.no_vig(prices)
    assert np.allclose(p.sum(axis=1), 1)
    assert p[1, 0] > p[1, 1]


def test_implied_goals_round_trip():
    rng = np.random.default_rng(1)
    lam = np.column_stack([rng.uniform(0.6, 3.0, 40), rng.uniform(0.4, 2.5, 40)])
    probs = market._grid_probs(lam[:, 0], lam[:, 1], -0.03)
    back = market.implied_goals(probs[:, :3], probs[:, 3], -0.03)
    assert np.allclose(back, lam, rtol=2e-3)


def test_implied_goals_without_over_under_keeps_supremacy():
    probs = market._grid_probs(np.array([2.0]), np.array([0.9]), -0.03)
    back = market.implied_goals(probs[:, :3], np.array([np.nan]), -0.03)
    assert back[0, 0] > back[0, 1]


def test_match_events_uses_date_score_and_names():
    teams = pd.DataFrame({"team_id": ["1", "2", "3", "4"],
                          "name": ["Nottingham Forest", "Wolverhampton Wanderers", "Manchester City", "Arsenal"],
                          "short_name": ["Nottm Forest", "Wolves", "Man City", "Arsenal"]})
    when = pd.Timestamp("2026-09-20 14:00", tz="UTC")
    matches = pd.DataFrame({
        "event_id": ["a", "b", "c"], "comp": "eng.1",
        "date": [when, when, when + pd.Timedelta(days=7)],
        "home_id": ["1", "3", "2"], "away_id": ["2", "4", "1"],
        "home_goals": [1.0, 1.0, np.nan], "away_goals": [1.0, 1.0, np.nan],
    })
    fd = pd.DataFrame({
        "comp": "eng.1", "day": [pd.Timestamp("2026-09-20")] * 2 + [pd.Timestamp("2026-09-27")],
        "HomeTeam": ["Nott'm Forest", "Man City", "Wolves"], "AwayTeam": ["Wolves", "Arsenal", "Nott'm Forest"],
        "FTHG": [1, 1, np.nan], "FTAG": [1, 1, np.nan], "played": [True, True, False],
    })
    ids = market.match_events(fd, matches, teams)
    assert list(ids) == ["a", "b", "c"]


def test_with_market_blend_and_corner_nudge():
    (gh, ga), (ch, ca) = with_market(1.0, 1.0, 5.0, 5.0, (2.0, 1.0), blend=0.5, corner={"own": 0.3, "opp": -0.2})
    assert np.isclose(gh, np.sqrt(2.0)) and np.isclose(ga, 1.0)
    assert np.isclose(ch, 5.0 * 2.0 ** 0.3) and np.isclose(ca, 5.0 * 2.0 ** -0.2)
    assert with_market(1.0, 1.2, 5.0, 4.0, None) == ((1.0, 1.2), (5.0, 4.0))
