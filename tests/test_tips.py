import numpy as np
import pandas as pd
import pytest

from predictor import dist, tips


def ou(kind, over):
    """Selection probabilities of an over/under market from {line: P(over)}."""
    out = {}
    for line, q in over.items():
        out[(kind, "over", line)] = q
        out[(kind, "under", line)] = None if q is None else 1 - q
    return out


def test_pick_prefers_the_side_furthest_above_the_league_rate():
    # Over 1.5 is the likeliest call, but it is barely above the league rate.
    p = ou("goals", {1.5: 0.82, 2.5: 0.66, 3.5: 0.40})
    base = ou("goals", {1.5: 0.78, 2.5: 0.52, 3.5: 0.30})
    t = tips.pick("goals", p, base)
    assert (t["line"], t["side"]) == (2.5, "over")
    assert t["p"] == pytest.approx(0.66)
    assert t["lift"] == pytest.approx(0.14)


def test_pick_handles_unders_and_thresholds():
    base_over = {8.5: 0.62, 9.5: 0.52, 10.5: 0.40, 11.5: 0.29}
    base = ou("corners", base_over)
    t = tips.pick("corners", ou("corners", {8.5: 0.45, 9.5: 0.33, 10.5: 0.22, 11.5: 0.14}), base)
    assert (t["line"], t["side"]) == (9.5, "under")
    # close to the league rate everywhere: no pick
    assert tips.pick("corners", ou("corners", {k: v + 0.03 for k, v in base_over.items()}), base) is None
    # far above the league rate but under MIN_PROB: no pick
    assert tips.pick("goals", ou("goals", {1.5: 0.5, 2.5: 0.30, 3.5: 0.12}), ou("goals", {1.5: 0.3, 2.5: 0.1, 3.5: 0.05})) is None
    # missing values are skipped
    assert tips.pick("goals", ou("goals", {1.5: np.nan, 2.5: None}), ou("goals", {1.5: 0.7, 2.5: 0.5})) is None


def test_hit_settles_every_kind_of_selection():
    assert tips.hit("over", 2.5, 2, 1) and not tips.hit("under", 2.5, 2, 1)
    assert tips.hit("home", None, 2, 1) and not tips.hit("draw", None, 2, 1) and not tips.hit("away", None, 2, 1)
    assert tips.hit("draw", None, 1, 1)
    assert tips.hit("yes", None, 2, 1) and tips.hit("no", None, 3, 0)
    assert not tips.hit("home", -1.5, 2, 1) and tips.hit("home", -1.5, 3, 1)
    assert tips.hit("away", 1.5, 2, 1) and not tips.hit("away", 1.5, 3, 1)


def test_probs_agree_with_the_score_grid():
    m = dist.score_matrix(1.8, 0.9, -0.03, 10)
    m1 = dist.score_matrix(0.8, 0.4, 0.0, 10)
    corners = dist.normalise_pmf(dist.poisson_pmf(10.2, 30))
    p = tips.probs(m, m1, corners, None)
    assert p[("result", "home", None)] == pytest.approx(np.tril(m, -1).sum())
    assert sum(p[("result", s, None)] for s in ("home", "draw", "away")) == pytest.approx(1)
    assert p[("goals", "over", 2.5)] == pytest.approx(dist.p_over(dist.total_from_matrix(m), 2.5))
    assert p[("btts", "yes", None)] == pytest.approx(m[1:, 1:].sum())
    # home -1.5 and away +1.5 are the two sides of one line
    assert p[("handicap", "home", -1.5)] + p[("handicap", "away", 1.5)] == pytest.approx(1)
    assert p[("ht_goals", "under", 0.5)] == pytest.approx(m1[0, 0])
    assert p[("corners", "over", 9.5)] == pytest.approx(dist.p_over(corners, 9.5))
    assert not any(k[0] == "cards" for k in p)  # no card pmf, no card market


def test_base_rates_count_each_selection():
    history = pd.DataFrame({"comp": ["eng.1"] * 40, "home_goals": [2, 0] * 20, "away_goals": [1, 0] * 20,
                            "home_ht_goals": [1, 0] * 20, "away_ht_goals": [0, 0] * 20,
                            "home_corners": [6, np.nan] * 20, "away_corners": [5, np.nan] * 20,
                            "home_cards": np.nan, "away_cards": np.nan})
    b = tips.base_rates(history, "eng.1")
    assert b[("result", "home", None)] == pytest.approx(0.5)
    assert b[("result", "draw", None)] == pytest.approx(0.5)
    assert b[("goals", "over", 2.5)] == pytest.approx(0.5)
    assert b[("corners", "over", 9.5)] == pytest.approx(1.0)  # missing corner counts are left out
    assert not any(k[0] == "cards" for k in b)


def test_history_grades_picks_and_attaches_prices():
    row = {"event_id": "1", "date": pd.Timestamp("2026-08-01", tz="UTC"), "comp": "eng.1",
           "home_goals": 2, "away_goals": 1, "home_ht_goals": 0, "away_ht_goals": 0,
           "home_corners": 5, "away_corners": 4, "home_cards": np.nan, "away_cards": np.nan,
           "mkt_odds_over": 1.8, "mkt_odds_under": 2.0, "mkt_best_over": 1.9, "mkt_best_under": 2.1,
           "mkt_odds_home": 1.5, "mkt_best_home": 1.6}
    sel = {**ou("goals", {1.5: 0.85, 2.5: 0.68, 3.5: 0.42}), **ou("corners", {8.5: 0.35, 9.5: 0.20, 10.5: 0.12, 11.5: 0.07}),
           ("result", "home", None): 0.70, ("result", "draw", None): 0.18, ("result", "away", None): 0.12,
           **ou("cards", {4.5: 0.9})}
    base = {**ou("goals", {1.5: 0.78, 2.5: 0.53, 3.5: 0.30}), **ou("corners", {8.5: 0.62, 9.5: 0.52, 10.5: 0.40, 11.5: 0.29}),
            ("result", "home", None): 0.45, ("result", "draw", None): 0.25, ("result", "away", None): 0.30,
            **ou("cards", {4.5: 0.5})}
    row.update({f"p|{tips.key(*k)}": v for k, v in sel.items()})
    row.update({f"b|{tips.key(*k)}": v for k, v in base.items()})
    h = tips.history(pd.DataFrame([row])).set_index("kind")
    assert set(h.index) == {"goals", "corners", "result"}  # cards have no result to grade
    assert (h.loc["goals", "line"], h.loc["goals", "side"], h.loc["goals", "won"]) == (2.5, "over", True)
    assert (h.loc["goals", "price"], h.loc["goals", "best"]) == (1.8, 1.9)
    assert (h.loc["corners", "line"], h.loc["corners", "side"], h.loc["corners", "won"]) == (9.5, "under", True)
    assert np.isnan(h.loc["corners", "price"])
    assert (h.loc["result", "side"], h.loc["result", "won"], h.loc["result", "price"]) == ("home", True, 1.5)

    s = tips.summarise(h.reset_index())
    assert (s["n"], s["won"]) == (3, 3)
    assert s["odds"]["n"] == 2
    assert s["odds"]["avg_price"]["roi"] == pytest.approx((0.8 + 0.5) / 2)
    assert s["odds"]["by_kind"]["result"]["roi"] == pytest.approx(0.5)

