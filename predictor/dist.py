"""Count distributions used to turn expected rates into probabilities.

web/model.js mirrors these functions for the in-browser match predictor; keep
the two in sync (tests/test_js_parity.py checks them against each other).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def poisson_pmf(lam: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    return np.exp(k * np.log(max(lam, 1e-12)) - lam - gammaln(k + 1))


def nbinom_pmf(mean: float, size: float, kmax: int) -> np.ndarray:
    """Negative binomial parameterised by mean and size (variance = mean + mean^2/size)."""
    k = np.arange(kmax + 1)
    mean = max(mean, 1e-12)
    p = size / (size + mean)
    logp = gammaln(k + size) - gammaln(size) - gammaln(k + 1) + size * np.log(p) + k * np.log1p(-p)
    return np.exp(logp)


def dc_tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles adjustment factor for low scores."""
    tau = np.ones(np.broadcast(x, y, lh, la).shape)
    tau = np.where((x == 0) & (y == 0), 1 - lh * la * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1 + lh * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1 + la * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
    return tau


def score_matrix(lh: float, la: float, rho: float, gmax: int) -> np.ndarray:
    """P(home = i, away = j) for i, j in 0..gmax, Dixon-Coles corrected and renormalised."""
    m = np.outer(poisson_pmf(lh, gmax), poisson_pmf(la, gmax))
    m[0, 0] *= max(1 - lh * la * rho, 0.0)
    m[0, 1] *= max(1 + lh * rho, 0.0)
    m[1, 0] *= max(1 + la * rho, 0.0)
    m[1, 1] *= max(1 - rho, 0.0)
    return m / m.sum()


def total_from_matrix(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    total = np.zeros(2 * n - 1)
    for i in range(n):
        total[i : i + n] += m[i]
    return total


def p_over(pmf: np.ndarray, line: float) -> float:
    return float(pmf[int(np.floor(line)) + 1 :].sum())


def normalise_pmf(pmf: np.ndarray) -> np.ndarray:
    return pmf / pmf.sum()
