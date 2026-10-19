"""Team-strength rate models for goals and corners.

Each model is a penalised Poisson regression on per-side counts:

    log E[count for side s] = mu + alpha[comp] + (H + dH[comp]) * at_home(s)
                              + att[team(s)] - dfn[opponent(s)]

`att`/`dfn` are attack/defence strengths (for corners: how many a team wins
and how many it denies). Every club sits on one shared scale, so a match
between clubs from different leagues (Champions League) is predicted from the
same ratings as domestic games. Clubs are shrunk towards the mean of their
current domestic league ("group"), which gives sensible numbers for clubs with
few matches. Older matches are down-weighted with an exponential half-life.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from scipy.special import gammaln

from . import config, dist

OTHER_GROUP = "other"


def team_groups(matches: pd.DataFrame, as_of: pd.Timestamp, lookahead_days: int = 45) -> dict[str, str]:
    """Map each team to its domestic league around `as_of`.

    Uses the latest domestic fixture up to `lookahead_days` ahead (schedules are
    public before a season starts, so promoted clubs get their new league).
    Clubs never seen in a domestic league we track go to OTHER_GROUP.
    """
    dom = matches[matches["comp"].map(config.is_domestic) & (matches["date"] <= as_of + pd.Timedelta(days=lookahead_days))]
    long = pd.concat(
        [
            dom[["date", "comp", "home_id"]].rename(columns={"home_id": "team"}),
            dom[["date", "comp", "away_id"]].rename(columns={"away_id": "team"}),
        ]
    ).sort_values("date")
    latest = long.groupby("team")["comp"].last()
    groups = latest.to_dict()
    for col in ("home_id", "away_id"):
        for team in matches[col].unique():
            groups.setdefault(team, OTHER_GROUP)
    return groups


def time_weights(dates: pd.Series, as_of: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (as_of - dates).dt.total_seconds().to_numpy() / 86400.0
    return np.power(0.5, np.clip(age, 0, None) / half_life_days)


@dataclass
class RateModel:
    half_life_days: float = 365.0
    lam_team: float = 8.0
    lam_group: float = 2.0
    lam_comp: float = 20.0
    lam_home: float = 20.0
    # Goals only: weight of a match whose target is the market's expected goals
    # instead of the scoreline (0 = ignore the market).
    market_weight: float = 0.0

    # fitted state
    mu: float = 0.0
    home: float = 0.0
    alpha: dict = field(default_factory=dict)
    d_home: dict = field(default_factory=dict)
    att: dict = field(default_factory=dict)
    dfn: dict = field(default_factory=dict)
    g_att: dict = field(default_factory=dict)
    g_dfn: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    n_eff: dict = field(default_factory=dict)
    dispersion: float = float("inf")  # NB size for the match total (inf = Poisson)
    side_dispersion: float = float("inf")  # NB size for one team's count
    rho: float = 0.0  # Dixon-Coles low-score correction (goals only)

    @classmethod
    def from_params(cls, params: dict) -> "RateModel":
        return cls(**params)

    # ------------------------------------------------------------------ fitting
    def fit(
        self,
        played: pd.DataFrame,
        y_home: str,
        y_away: str,
        as_of: pd.Timestamp,
        groups: dict[str, str],
        warm: "RateModel | None" = None,
        min_weight: float = 1e-3,
        market: tuple[str, str] | None = None,
    ) -> "RateModel":
        """Fit by penalised weighted Poisson likelihood.

        With `market` (two columns of market-implied expected counts) and a
        positive `market_weight`, matches that have market numbers are fitted
        to them instead of to the observed counts.
        """
        df = played[(played["date"] < as_of) & played[y_home].notna() & played[y_away].notna()]
        w = time_weights(df["date"], as_of, self.half_life_days)
        keep = w >= min_weight
        df, w = df[keep], w[keep]
        if df.empty:
            raise ValueError("no training data before %s" % as_of)
        obs_h, obs_a = df[y_home].to_numpy(float), df[y_away].to_numpy(float)
        tgt_h, tgt_a, w_fit = obs_h, obs_a, w
        if market and self.market_weight > 0 and market[0] in df:
            mh, ma = df[market[0]].to_numpy(float), df[market[1]].to_numpy(float)
            has = np.isfinite(mh) & np.isfinite(ma)
            tgt_h, tgt_a = np.where(has, mh, obs_h), np.where(has, ma, obs_a)
            w_fit = w * np.where(has, self.market_weight, 1.0)

        teams = pd.Index(pd.unique(pd.concat([df["home_id"], df["away_id"]])))
        comps = pd.Index(sorted(df["comp"].unique()))
        grp_of_team = np.array([groups.get(t, OTHER_GROUP) for t in teams])
        group_index = pd.Index(sorted(set(grp_of_team)))
        tg = group_index.get_indexer(grp_of_team)

        T, C, G = len(teams), len(comps), len(group_index)
        o_alpha, o_dh = 2, 2 + C
        o_att, o_dfn = 2 + 2 * C, 2 + 2 * C + T
        o_gatt, o_gdfn = 2 + 2 * C + 2 * T, 2 + 2 * C + 2 * T + G
        P = o_gdfn + G

        n = len(df)
        hi = teams.get_indexer(df["home_id"])
        ai = teams.get_indexer(df["away_id"])
        ci = comps.get_indexer(df["comp"])
        hflag = (~df["neutral"].to_numpy(dtype=bool)).astype(float)

        rows_h = np.repeat(np.arange(n), 6)
        cols_h = np.stack([np.zeros(n, int), np.ones(n, int), o_alpha + ci, o_dh + ci, o_att + hi, o_dfn + ai], 1).ravel()
        vals_h = np.stack([np.ones(n), hflag, np.ones(n), hflag, np.ones(n), -np.ones(n)], 1).ravel()
        rows_a = np.repeat(np.arange(n, 2 * n), 4)
        cols_a = np.stack([np.zeros(n, int), o_alpha + ci, o_att + ai, o_dfn + hi], 1).ravel()
        vals_a = np.stack([np.ones(n), np.ones(n), np.ones(n), -np.ones(n)], 1).ravel()
        X = sparse.csr_matrix(
            (np.concatenate([vals_h, vals_a]), (np.concatenate([rows_h, rows_a]), np.concatenate([cols_h, cols_a]))),
            shape=(2 * n, P),
        )
        XT = X.T.tocsr()
        y = np.concatenate([tgt_h, tgt_a])
        ww = np.concatenate([w_fit, w_fit])

        lt, lg, lc, lh = self.lam_team, self.lam_group, self.lam_comp, self.lam_home

        def objective(theta):
            eta = np.clip(X @ theta, -15, 15)
            rate = np.exp(eta)
            f = np.dot(ww, rate - y * eta)
            grad = XT @ (ww * (rate - y))
            att, dfn = theta[o_att:o_dfn], theta[o_dfn:o_gatt]
            gatt, gdfn = theta[o_gatt:o_gdfn], theta[o_gdfn:]
            ra, rd = att - gatt[tg], dfn - gdfn[tg]
            f += lt * (ra @ ra + rd @ rd) + lg * (gatt @ gatt + gdfn @ gdfn)
            grad[o_att:o_dfn] += 2 * lt * ra
            grad[o_dfn:o_gatt] += 2 * lt * rd
            grad[o_gatt:o_gdfn] += -2 * lt * np.bincount(tg, ra, minlength=G) + 2 * lg * gatt
            grad[o_gdfn:] += -2 * lt * np.bincount(tg, rd, minlength=G) + 2 * lg * gdfn
            al, dh = theta[o_alpha:o_dh], theta[o_dh:o_att]
            f += lc * (al @ al) + lh * (dh @ dh)
            grad[o_alpha:o_dh] += 2 * lc * al
            grad[o_dh:o_att] += 2 * lh * dh
            return f, grad

        theta0 = np.zeros(P)
        theta0[0] = np.log(max(np.average(y, weights=ww), 1e-3))
        theta0[1] = 0.1
        if warm is not None and warm.att:
            theta0[0], theta0[1] = warm.mu, warm.home
            for i, c in enumerate(comps):
                theta0[o_alpha + i] = warm.alpha.get(c, 0.0)
                theta0[o_dh + i] = warm.d_home.get(c, 0.0)
            for i, t in enumerate(teams):
                theta0[o_att + i] = warm.att.get(t, 0.0)
                theta0[o_dfn + i] = warm.dfn.get(t, 0.0)
            for i, g in enumerate(group_index):
                theta0[o_gatt + i] = warm.g_att.get(g, 0.0)
                theta0[o_gdfn + i] = warm.g_dfn.get(g, 0.0)

        res = optimize.minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-6})
        theta = res.x

        self.mu, self.home = float(theta[0]), float(theta[1])
        self.alpha = dict(zip(comps, theta[o_alpha:o_dh].tolist()))
        self.d_home = dict(zip(comps, theta[o_dh:o_att].tolist()))
        self.att = dict(zip(teams, theta[o_att:o_dfn].tolist()))
        self.dfn = dict(zip(teams, theta[o_dfn:o_gatt].tolist()))
        self.g_att = dict(zip(group_index, theta[o_gatt:o_gdfn].tolist()))
        self.g_dfn = dict(zip(group_index, theta[o_gdfn:].tolist()))
        self.groups = dict(groups)
        w_conf = time_weights(df["date"], as_of, config.CONFIDENCE_HALF_LIFE)
        n_eff = np.bincount(np.concatenate([hi, ai]), np.concatenate([w_conf, w_conf]), minlength=T)
        self.n_eff = dict(zip(teams, n_eff.tolist()))

        rates = np.exp(np.clip(X @ theta, -15, 15))
        rh, ra_ = rates[:n], rates[n:]
        yh, ya = obs_h, obs_a
        self.side_dispersion = _fit_nb_size(np.concatenate([yh, ya]), np.concatenate([rh, ra_]), np.concatenate([w, w]))
        self.dispersion = _fit_nb_size(yh + ya, rh + ra_, w)
        self._train = (yh, ya, rh, ra_, w)
        return self

    def fit_rho(self) -> "RateModel":
        """Estimate the Dixon-Coles correlation for goals from the training fit."""
        yh, ya, rh, ra, w = self._train
        low = (yh <= 1) & (ya <= 1)
        x, y, lh, la, ww = yh[low], ya[low], rh[low], ra[low], w[low]

        def nll(rho):
            tau = dist.dc_tau(x, y, lh, la, rho)
            if np.any(tau <= 0):
                return 1e12
            return -np.dot(ww, np.log(tau))

        upper = float(min(1.0, 1.0 / np.max(lh * la))) - 1e-6
        res = optimize.minimize_scalar(nll, bounds=(-0.3, min(0.3, upper)), method="bounded")
        self.rho = float(res.x)
        return self

    # --------------------------------------------------------------- predicting
    def team_strength(self, team: str) -> tuple[float, float]:
        if team in self.att:
            return self.att[team], self.dfn[team]
        g = self.groups.get(team, OTHER_GROUP)
        return self.g_att.get(g, 0.0), self.g_dfn.get(g, 0.0)

    def rates(self, home: str, away: str, comp: str, neutral: bool = False) -> tuple[float, float]:
        ah, dh = self.team_strength(home)
        aa, da = self.team_strength(away)
        base = self.mu + self.alpha.get(comp, 0.0)
        home_adv = 0.0 if neutral else self.home + self.d_home.get(comp, 0.0)
        return float(np.exp(base + home_adv + ah - da)), float(np.exp(base + aa - dh))

    def team_profile(self, team: str, comp: str) -> tuple[float, float]:
        """Expected count for/against per game vs an average opponent of the team's league."""
        a, d = self.team_strength(team)
        g = self.groups.get(team, OTHER_GROUP)
        base = self.mu + self.alpha.get(comp, 0.0) + 0.5 * (self.home + self.d_home.get(comp, 0.0))
        return float(np.exp(base + a - self.g_dfn.get(g, 0.0))), float(np.exp(base + self.g_att.get(g, 0.0) - d))

    def export(self) -> dict:
        return {
            "mu": self.mu,
            "home": self.home,
            "alpha": self.alpha,
            "d_home": self.d_home,
            "g_att": self.g_att,
            "g_dfn": self.g_dfn,
            "dispersion": _finite(self.dispersion),
            "side_dispersion": _finite(self.side_dispersion),
            "rho": self.rho,
        }


def _finite(x: float, cap: float = 1e6) -> float:
    return float(min(x, cap))


def _fit_nb_size(y: np.ndarray, mean: np.ndarray, w: np.ndarray) -> float:
    """Weighted ML estimate of the negative-binomial size given fitted means."""

    def nll(log_size):
        r = np.exp(log_size)
        p = r / (r + mean)
        ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(p) + y * np.log1p(-p)
        return -np.dot(w, ll)

    res = optimize.minimize_scalar(nll, bounds=(np.log(0.5), np.log(1e6)), method="bounded")
    return float(np.exp(res.x))


@dataclass
class MatchPredictor:
    """Joins the goals and corners models into full match predictions."""

    goals: RateModel
    corners: RateModel

    def predict(self, home: str, away: str, comp: str, neutral: bool = False,
                market: tuple[float, float] | None = None) -> dict:
        """Full prediction; `market` is the match's market-implied expected goals, if known."""
        lh, la = self.goals.rates(home, away, comp, neutral)
        ch, ca = self.corners.rates(home, away, comp, neutral)
        (lh, la), (ch, ca) = with_market(lh, la, ch, ca, market)
        m = dist.score_matrix(lh, la, self.goals.rho, config.MAX_GOALS)
        goals_total = dist.total_from_matrix(m)
        corners_total = _count_pmf(ch + ca, self.corners.dispersion, config.MAX_CORNERS)
        home_c = _count_pmf(ch, self.corners.side_dispersion, config.MAX_CORNERS)
        away_c = _count_pmf(ca, self.corners.side_dispersion, config.MAX_CORNERS)
        flat = np.argsort(m, axis=None)[::-1][:8]
        scores = [{"h": int(i // m.shape[1]), "a": int(i % m.shape[1]), "p": float(m.flat[i])} for i in flat]
        return {
            "goals": {
                "home": lh,
                "away": la,
                "total": lh + la,
                "pmf": _trim(goals_total, 9),
                "over": {str(l): dist.p_over(goals_total, l) for l in config.GOAL_LINES},
                "btts": float(m[1:, 1:].sum()),
                "home_cs": float(m[:, 0].sum()),
                "away_cs": float(m[0, :].sum()),
                "home_win": float(np.tril(m, -1).sum()),
                "draw": float(np.trace(m)),
                "away_win": float(np.triu(m, 1).sum()),
                "scores": scores,
            },
            "corners": {
                "home": ch,
                "away": ca,
                "total": ch + ca,
                "pmf": _trim(corners_total, 22),
                "over": {str(l): dist.p_over(corners_total, l) for l in config.CORNER_LINES},
                "home_over": {str(l): dist.p_over(home_c, l) for l in config.TEAM_CORNER_LINES},
                "away_over": {str(l): dist.p_over(away_c, l) for l in config.TEAM_CORNER_LINES},
                "home_more": float(_p_greater(home_c, away_c)),
                "away_more": float(_p_greater(away_c, home_c)),
            },
        }


def with_market(lh: float, la: float, ch: float, ca: float, market: tuple[float, float] | None,
                blend: float | None = None, corner: dict | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Blend model expected goals with the market's and nudge expected corners to match.

    web/model.js mirrors this function.
    """
    if market is None or not all(np.isfinite(market)) or min(market) <= 0:
        return (lh, la), (ch, ca)
    w = config.MARKET_BLEND if blend is None else blend
    k = config.CORNER_MARKET if corner is None else corner
    mh, ma = market
    d_h, d_a = np.log(mh / lh), np.log(ma / la)
    goals = (float(lh * np.exp(w * d_h)), float(la * np.exp(w * d_a)))
    corners = (float(ch * np.exp(k["own"] * d_h + k["opp"] * d_a)), float(ca * np.exp(k["own"] * d_a + k["opp"] * d_h)))
    return goals, corners


def _count_pmf(mean: float, size: float, kmax: int) -> np.ndarray:
    if not np.isfinite(size) or size > 1e5:
        pmf = dist.poisson_pmf(mean, kmax)
    else:
        pmf = dist.nbinom_pmf(mean, size, kmax)
    return dist.normalise_pmf(pmf)


def _p_greater(a: np.ndarray, b: np.ndarray) -> float:
    cdf_b = np.cumsum(b)
    return float(np.sum(a[1:] * cdf_b[:-1]))


def _trim(pmf: np.ndarray, kmax: int) -> list[float]:
    """Return pmf[0..kmax-1] plus a final bucket holding P(X >= kmax)."""
    head = pmf[:kmax].tolist()
    head.append(float(pmf[kmax:].sum()))
    return [round(p, 5) for p in head]


def fit_predictor(matches: pd.DataFrame, as_of: pd.Timestamp, goals_params: dict | None = None,
                  corners_params: dict | None = None, warm: MatchPredictor | None = None) -> MatchPredictor:
    from .espn import played_mask

    played = matches[played_mask(matches)]
    groups = team_groups(matches, as_of)
    goals = RateModel.from_params(goals_params or config.GOALS_PARAMS)
    goals.fit(played, "home_goals", "away_goals", as_of, groups, warm=warm.goals if warm else None,
              market=("mk_home", "mk_away")).fit_rho()
    corners = RateModel.from_params(corners_params or config.CORNERS_PARAMS)
    corners.fit(played, "home_corners", "away_corners", as_of, groups, warm=warm.corners if warm else None)
    return MatchPredictor(goals, corners)
