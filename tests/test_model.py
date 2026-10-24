import numpy as np
import pandas as pd

from predictor import dist
from predictor.espn import _slim_event, normalise, played_mask
from predictor.model import MatchPredictor, RateModel


def synthetic_season(n_teams=16, rounds=4, seed=0):
    rng = np.random.default_rng(seed)
    teams = [str(i) for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    rows, day = [], 0
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lh = np.exp(0.2 + 0.25 + att[i] - dfn[j])
                la = np.exp(0.2 + att[j] - dfn[i])
                rows.append({
                    "event_id": f"{day}-{i}-{j}", "comp": "eng.1", "date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(hours=day),
                    "status": "STATUS_FULL_TIME", "state": "post", "completed": True, "neutral": False,
                    "home_id": teams[i], "away_id": teams[j], "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
                })
                day += 1
    return pd.DataFrame(rows), teams, att, dfn


def test_rate_model_recovers_team_strengths():
    df, teams, att, dfn = synthetic_season()
    as_of = df["date"].max() + pd.Timedelta(days=1)
    groups = {t: "eng.1" for t in teams}
    m = RateModel(half_life_days=1e6, lam_team=0.5, lam_group=0.5, lam_comp=50, lam_home=50)
    m.fit(df, "home_goals", "away_goals", as_of, groups)
    est_att = np.array([m.att[t] for t in teams])
    est_dfn = np.array([m.dfn[t] for t in teams])
    assert np.corrcoef(est_att, att)[0, 1] > 0.85
    assert np.corrcoef(est_dfn, dfn)[0, 1] > 0.85
    assert abs(m.home - 0.25) < 0.1


def test_level_correction_follows_a_recent_drift():
    df, teams, *_ = synthetic_season()
    as_of = df["date"].max() + pd.Timedelta(days=1)
    recent = df["date"] >= as_of - pd.Timedelta(days=5)
    df.loc[recent, ["home_goals", "away_goals"]] *= 2  # the last few days produce twice as much
    groups = {t: "eng.1" for t in teams}
    plain = RateModel(half_life_days=1e6).fit(df, "home_goals", "away_goals", as_of, groups)
    level = RateModel(half_life_days=1e6, level_days=5).fit(df, "home_goals", "away_goals", as_of, groups)
    fitted = lambda m: sum(sum(m.rates(h, a, "eng.1")) for h, a in zip(df.loc[recent, "home_id"], df.loc[recent, "away_id"]))  # noqa: E731
    actual = (df.loc[recent, "home_goals"] + df.loc[recent, "away_goals"]).sum()
    assert fitted(plain) < 0.8 * actual
    assert abs(fitted(level) / actual - 1) < 1e-6


def test_predictions_are_proper_distributions():
    df, teams, *_ = synthetic_season(rounds=2)
    df["home_corners"] = df["home_goals"] * 2 + 4
    df["away_corners"] = df["away_goals"] * 2 + 3
    df["home_cards"], df["away_cards"] = df["away_goals"] + 1, df["home_goals"] + 2
    as_of = df["date"].max() + pd.Timedelta(days=1)
    groups = {t: "eng.1" for t in teams}
    goals = RateModel().fit(df, "home_goals", "away_goals", as_of, groups).fit_rho()
    corners = RateModel().fit(df, "home_corners", "away_corners", as_of, groups)
    cards = RateModel().fit(df, "home_cards", "away_cards", as_of, groups)
    pred = MatchPredictor(goals, corners, cards, (0.45, 0.44)).predict("0", "1", "eng.1")
    for kind in ("goals", "corners", "cards"):
        p = pred[kind]
        assert abs(sum(p["pmf"]) - 1) < 1e-3
        overs = [p["over"][k] for k in sorted(p["over"], key=float)]
        assert all(a >= b for a, b in zip(overs, overs[1:])), "P(over) must fall as the line rises"
        assert abs(p["total"] - (p["home"] + p["away"])) < 1e-9
    g = pred["goals"]
    assert abs(g["home_win"] + g["draw"] + g["away_win"] - 1) < 1e-6
    ht = pred["ht"]
    assert abs(sum(ht["hda"]) - 1) < 1e-6
    assert abs(ht["home"] - 0.45 * g["home"]) < 1e-9
    assert ht["hda"][1] > g["draw"], "fewer goals by half-time, so more level scores"


def test_score_matrix_and_totals():
    m = dist.score_matrix(1.4, 1.1, -0.05, 10)
    assert abs(m.sum() - 1) < 1e-12
    total = dist.total_from_matrix(m)
    assert abs(total.sum() - 1) < 1e-12
    mean = (np.arange(len(total)) * total).sum()
    assert abs(mean - 2.5) < 0.05
    assert abs(dist.p_over(total, 2.5) - total[3:].sum()) < 1e-12


def test_negative_binomial_mean_and_variance():
    pmf = dist.nbinom_pmf(10.0, 15.0, 80)
    k = np.arange(len(pmf))
    mean = (k * pmf).sum()
    var = ((k - mean) ** 2 * pmf).sum()
    assert abs(pmf.sum() - 1) < 1e-9
    assert abs(mean - 10) < 1e-6
    assert abs(var - (10 + 100 / 15)) < 1e-4


def _event(eid, home_corners, away_corners, odds=None, details=None):
    def side(ha, tid, goals, corners):
        return {
            "homeAway": ha, "score": str(goals),
            "team": {"id": tid, "displayName": f"Team {tid}", "abbreviation": "T", "logo": ""},
            "statistics": [{"name": "wonCorners", "displayValue": str(corners)}, {"name": "totalShots", "displayValue": "10"}],
        }

    return {
        "id": eid, "date": "2026-09-01T19:00Z", "season": {"year": 2026, "slug": "s"},
        "status": {"type": {"name": "STATUS_FULL_TIME", "state": "post", "completed": True}},
        "competitions": [{
            "neutralSite": False, "venue": {"fullName": "Ground"}, "odds": odds, "details": details,
            "status": {"type": {"name": "STATUS_FULL_TIME", "state": "post", "completed": True}},
            "competitors": [side("home", "1", 2, home_corners), side("away", "2", 1, away_corners)],
        }],
    }


def test_normalise_treats_zero_zero_corners_as_missing():
    matches, teams = normalise({"eng.1": [_event("a", 6, 4, odds=[None, {"overUnder": 2.5}]), _event("b", 0, 0)]})
    assert len(teams) == 2
    a = matches.set_index("event_id").loc["a"]
    b = matches.set_index("event_id").loc["b"]
    assert a["home_corners"] == 6 and a["away_corners"] == 4 and a["market_goals_line"] == 2.5
    assert pd.isna(b["home_corners"]) and pd.isna(b["away_corners"])
    assert played_mask(matches).all()


def _detail(text, team, clock, **flags):
    return {"type": {"text": text}, "clock": {"displayValue": clock}, "team": {"id": team}, **flags}


def test_normalise_reads_half_time_goals_and_cards():
    # the home side wins 2-1; the away goal comes in first-half stoppage time
    timeline = [
        _detail("Goal", "1", "30'", scoringPlay=True), _detail("Yellow Card", "1", "20'", yellowCard=True),
        _detail("Goal", "2", "45'+2'", scoringPlay=True), _detail("Red Card", "2", "70'", redCard=True),
        _detail("Goal", "1", "80'", scoringPlay=True), _detail("Penalty - Scored", "2", "120'", scoringPlay=True, shootout=True),
    ]
    events = [_slim_event(_event("a", 6, 4, details=timeline)),
              _slim_event(_event("b", 6, 4, details=timeline[:2]))]  # goal events missing
    m = normalise({"eng.1": events})[0].set_index("event_id")
    assert m.loc["a", ["home_ht_goals", "away_ht_goals", "home_cards", "away_cards"]].tolist() == [1, 1, 1, 1]
    assert m.loc["b", ["home_ht_goals", "away_ht_goals"]].isna().all()
    assert m.loc["b", "home_cards"] == 1
