import numpy as np
import pandas as pd
import pytest

from predictor import tips


def test_pick_prefers_the_side_furthest_above_the_league_rate():
    # Over 1.5 is the likeliest call, but it is barely above the league rate.
    p_over = {1.5: 0.82, 2.5: 0.66, 3.5: 0.40}
    base = {1.5: 0.78, 2.5: 0.52, 3.5: 0.30}
    t = tips.pick("goals", p_over, base)
    assert (t["line"], t["side"]) == (2.5, "over")
    assert t["p"] == pytest.approx(0.66)
    assert t["lift"] == pytest.approx(0.14)


def test_pick_handles_unders_and_thresholds():
    base = {8.5: 0.62, 9.5: 0.52, 10.5: 0.40, 11.5: 0.29}
    t = tips.pick("corners", {8.5: 0.45, 9.5: 0.33, 10.5: 0.22, 11.5: 0.14}, base)
    assert (t["line"], t["side"]) == (9.5, "under")
    # close to the league rate everywhere: no pick
    assert tips.pick("corners", {k: v + 0.03 for k, v in base.items()}, base) is None
    # far above the league rate but under MIN_PROB: no pick
    assert tips.pick("goals", {1.5: 0.5, 2.5: 0.30, 3.5: 0.12}, {1.5: 0.3, 2.5: 0.1, 3.5: 0.05}) is None
    # missing values are skipped
    assert tips.pick("goals", {1.5: np.nan, 2.5: None}, {1.5: 0.7, 2.5: 0.5}) is None


def test_history_grades_picks_and_attaches_prices():
    row = {"event_id": "1", "date": pd.Timestamp("2026-08-01", tz="UTC"), "comp": "eng.1",
           "goals_actual": 3, "corners_actual": 9,
           "mkt_odds_over": 1.8, "mkt_odds_under": 2.0, "mkt_best_over": 1.9, "mkt_best_under": 2.1}
    for line, p, b in [(1.5, 0.85, 0.78), (2.5, 0.68, 0.53), (3.5, 0.42, 0.30)]:
        row[f"goals_p_over_{line}"], row[f"goals_base_p_over_{line}"] = p, b
    for line, p, b in [(8.5, 0.35, 0.62), (9.5, 0.20, 0.52), (10.5, 0.12, 0.40), (11.5, 0.07, 0.29)]:
        row[f"corners_p_over_{line}"], row[f"corners_base_p_over_{line}"] = p, b
    h = tips.history(pd.DataFrame([row])).set_index("kind")
    assert (h.loc["goals", "line"], h.loc["goals", "side"], h.loc["goals", "won"]) == (2.5, "over", True)
    assert (h.loc["goals", "price"], h.loc["goals", "best"]) == (1.8, 1.9)
    assert (h.loc["corners", "line"], h.loc["corners", "side"], h.loc["corners", "won"]) == (9.5, "under", True)
    assert np.isnan(h.loc["corners", "price"])

    s = tips.summarise(h.reset_index())
    assert (s["n"], s["won"]) == (2, 2)
    assert s["odds"]["n"] == 1
    assert s["odds"]["avg_price"]["roi"] == pytest.approx(0.8)
