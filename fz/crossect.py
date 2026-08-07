"""Cross-sectional regressions: Fama-MacBeth, standard errors, and IC over time.

The Fama-MacBeth two-pass estimator turns a panel into a *time series of
cross-sectional slopes*: fit one regression per date, then treat the T daily
slopes as T draws of the premium. Inference therefore happens in the time
dimension, which is where the dependence lives -- and where you have to be
careful once forward returns overlap (see `newey_west_var`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FamaMacBethResult:
    """Output of `fama_macbeth`. Columns follow the design matrix order
    (intercept first when `add_intercept=True`, then `scores_list`)."""

    coefficients: np.ndarray   # (n_cols,) mean of the daily slopes
    t_stats: np.ndarray        # (n_cols,) coefficients / std_errors
    std_errors: np.ndarray     # (n_cols,) s.e. of the mean, HAC if hac_lags > 0
    daily_coefs: np.ndarray    # (n_days, n_cols)
    daily_r2: np.ndarray       # (n_days,)
    n_periods: int = 0         # cross-sections that actually estimated


def forward_returns(returns: np.ndarray, horizon: int = 1) -> np.ndarray:
    """Compound the next `horizon` periods of returns, aligned to date t.

    `out[t]` is the simple return from holding a name from the close of day
    t to the close of day t + horizon, so it pairs with a score built from
    data up to and including day t. The last `horizon` rows are NaN (the
    future has not happened yet), and any NaN inside the window propagates
    -- a missing day is missing, not silently a flat one.

    `horizon=1` is exactly the one-day-ahead shift every example needs:

    >>> import numpy as np
    >>> r = np.array([[np.nan], [0.01], [0.02]])
    >>> np.round(forward_returns(r, 1).ravel(), 4)
    array([0.01, 0.02,  nan])
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    r = np.asarray(returns, dtype=float)
    n_days = r.shape[0]
    out = np.full_like(r, np.nan)
    growth = 1.0 + r
    for t in range(n_days - horizon):
        out[t] = np.prod(growth[t + 1: t + 1 + horizon], axis=0) - 1.0
    return out


def newey_west_var(x: np.ndarray, lags: int = 0) -> float:
    """Long-run variance of a scalar series under a Bartlett (NW) kernel.

    Returns the HAC estimate of T * Var(mean(x)):

        S = g_0 + 2 * sum_{l=1..L} (1 - l / (L + 1)) * g_l,

    with autocovariances g_l = sum_t (x_t - xbar)(x_{t-l} - xbar) / (T - 1).
    The T-1 divisor means `lags=0` returns the ordinary sample variance, so
    the classic Fama-MacBeth t-stat is the L=0 special case rather than a
    separate code path. The Bartlett weights guarantee S >= 0.

    At `lags=0` it is exactly `np.var(x, ddof=1)`:

    >>> import numpy as np
    >>> round(newey_west_var(np.array([1.0, 2.0, 3.0, 4.0]), 0), 4)
    1.6667

    A positively autocorrelated series has a long-run variance larger than
    its sample variance -- which is the entire reason the correction exists:

    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    >>> round(newey_west_var(x, 0), 4), round(newey_west_var(x, 3), 4)
    (6.0, 13.1786)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = x.shape[0]
    if T < 2:
        return float("nan")
    lags = min(int(lags), T - 1)
    d = x - x.mean()
    s = float(d @ d) / (T - 1)
    for lag in range(1, lags + 1):
        g = float(d[lag:] @ d[:-lag]) / (T - 1)
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * g
    return max(s, 0.0)


def shanken_factor(premia: np.ndarray, factor_cov: np.ndarray) -> float:
    """Shanken's (1992) errors-in-variables inflation, ``1 + lam' Sigma_f^-1 lam``.

    This is the multiplier that a two-pass cross-sectional regression owes its
    standard errors when the second-pass regressors are *estimated betas*
    rather than observed quantities. The first pass measures each beta with
    error; the second pass then treats a noisy regressor as exact, and the
    resulting understatement of the sampling variance is, asymptotically,
    the cross-sectional term scaled by this factor.

    `premia` is the vector of factor risk premia (same units and frequency as
    `factor_cov`, which is the covariance matrix of the factor returns). The
    size of the correction is governed entirely by the factors' squared
    Sharpe ratio: a market factor at 0.5%/month against 4.5%/month volatility
    gives 1.012, while a factor earning 0.8 standard deviations a period gives
    1.64.

    `fama_macbeth` in this library regresses on *characteristics*, which are
    observed, so this correction does not apply to its output -- see
    `docs/validation.md`, which measures both cases. It ships because the
    distinction is the single most common way a Fama-MacBeth t-stat is
    overstated in practice, and because it is worth being able to compute.

    >>> float(round(shanken_factor([0.008], [[0.04 ** 2]]), 4))
    1.04
    """
    lam = np.atleast_1d(np.asarray(premia, dtype=float))
    cov = np.atleast_2d(np.asarray(factor_cov, dtype=float))
    if cov.shape != (lam.size, lam.size):
        raise ValueError(f"factor_cov must be {(lam.size, lam.size)}, got {cov.shape}")
    return 1.0 + float(lam @ np.linalg.solve(cov, lam))


def _ols_slope(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS slope of y on X (no intercept added implicitly)."""
    # Solve normal equations: beta = (X^T X)^-1 X^T y
    valid = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    Xv = X[valid]; yv = y[valid]
    if Xv.shape[0] < Xv.shape[1] + 2:
        return np.full(X.shape[1], np.nan), np.nan
    XtX = Xv.T @ Xv + 1e-9 * np.eye(Xv.shape[1])
    beta = np.linalg.solve(XtX, Xv.T @ yv)
    resid = yv - Xv @ beta
    tss = np.sum((yv - yv.mean()) ** 2)
    rss = np.sum(resid ** 2)
    r2 = 1.0 - rss / max(tss, 1e-12)
    return beta, float(r2)


def fama_macbeth(scores_list: list[np.ndarray], fwd_returns: np.ndarray,
                 add_intercept: bool = True, hac_lags: int = 0) -> FamaMacBethResult:
    """Daily cross-sectional regressions of forward returns on factor scores.

    Each `scores_list[k]` is (n_days, n_stocks); `fwd_returns` is the matching
    panel of returns earned *after* the scores are known (see
    `forward_returns`). Day t contributes one OLS slope vector; the premium
    estimate is the mean of those slopes and its standard error is
    `sqrt(S / T)` with `S` from `newey_west_var`.

    `hac_lags=0` (the default) gives the textbook Fama-MacBeth t-stat, which
    assumes the daily slopes are serially independent. That is defensible for
    one-day-ahead returns and *wrong* for overlapping horizons: with h-day
    forward returns, consecutive slopes share h-1 days of data, so set
    `hac_lags=h-1` to stop the t-stat from counting the same evidence twice.

    On a panel where the return *is* 2 bp per unit of score, with no noise,
    the estimator returns exactly that and nothing for the intercept:

    >>> import numpy as np
    >>> score = np.tile(np.array([-1.0, 0.0, 1.0, 2.0]), (30, 1))
    >>> res = fama_macbeth([score], 0.002 * score)
    >>> np.round(res.coefficients, 6), res.n_periods
    (array([0.   , 0.002]), 30)
    >>> float(np.round(res.daily_r2[0], 6))
    1.0
    """
    n_days, n_stocks = scores_list[0].shape
    K = len(scores_list)
    n_cols = K + (1 if add_intercept else 0)
    daily_coefs = np.full((n_days, n_cols), np.nan)
    daily_r2 = np.full(n_days, np.nan)

    for t in range(n_days):
        cols = []
        if add_intercept:
            cols.append(np.ones(n_stocks))
        for s in scores_list:
            cols.append(s[t])
        X = np.column_stack(cols)
        b, r2 = _ols_slope(X, fwd_returns[t])
        daily_coefs[t] = b
        daily_r2[t] = r2

    valid_rows = ~np.any(np.isnan(daily_coefs), axis=1)
    coefs = daily_coefs[valid_rows]
    n = coefs.shape[0]
    mean = coefs.mean(axis=0)
    se = np.array([np.sqrt(newey_west_var(coefs[:, j], hac_lags) / max(n, 1))
                   for j in range(n_cols)])
    return FamaMacBethResult(coefficients=mean, t_stats=mean / (se + 1e-12),
                             std_errors=se, daily_coefs=daily_coefs,
                             daily_r2=daily_r2, n_periods=n)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """0-based ranks with average tie handling.

    Tied values share the mean of the ordinal ranks they span, so the result
    does not depend on input order. Winsorised scores are guaranteed to tie
    in the clipped tails -- exactly where long-short portfolios are formed --
    which makes tie handling load-bearing rather than cosmetic.

    The two 30s span ordinal ranks 2 and 3, so both get 2.5:

    >>> import numpy as np
    >>> average_ranks(np.array([10.0, 30.0, 20.0, 30.0]))
    array([0. , 2.5, 1. , 2.5])
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    xs = x[order]
    i, n = 0, x.shape[0]
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def rank_information_coefficient(scores: np.ndarray,
                                 fwd_returns: np.ndarray) -> np.ndarray:
    """Per-day Spearman correlation between score and forward return.

    Returns (n_days,) IC series. NaN if too few valid points (fewer than 20
    names with both a score and a return).

    A perfect ranking scores +1 and its reverse -1, which is the sanity
    check the sign convention hangs on:

    >>> import numpy as np
    >>> s = np.tile(np.arange(24.0), (2, 1))
    >>> rank_information_coefficient(s, np.vstack([s[0], -s[0]]))
    array([ 1., -1.])
    >>> float(np.isnan(rank_information_coefficient(s[:, :19], s[:, :19])[0]))
    1.0
    """
    n_days = scores.shape[0]
    out = np.full(n_days, np.nan)
    for t in range(n_days):
        s = scores[t]; r = fwd_returns[t]
        valid = ~np.isnan(s) & ~np.isnan(r)
        if valid.sum() < 20:
            continue
        s_r = average_ranks(s[valid])
        r_r = average_ranks(r[valid])
        # Spearman = Pearson on ranks
        s_z = (s_r - s_r.mean()) / (s_r.std() + 1e-12)
        r_z = (r_r - r_r.mean()) / (r_r.std() + 1e-12)
        out[t] = float((s_z * r_z).mean())
    return out


def rolling_ic(scores: np.ndarray, fwd_returns: np.ndarray,
               window: int = 60) -> np.ndarray:
    """Trailing-`window` mean of the daily rank IC.

    A factor's per-day IC is noisy; the rolling average is what you actually
    look at to judge whether the signal is alive or decaying over time.
    NaN until enough non-NaN daily ICs accumulate in the window.

    Five days whose daily ICs are +1, +1, -1, -1, +1; a two-day trailing
    mean turns that into the run of zeros where the signal flips:

    >>> import numpy as np
    >>> s = np.tile(np.arange(24.0), (5, 1))
    >>> fwd = s * np.array([1.0, 1.0, -1.0, -1.0, 1.0])[:, None]
    >>> rolling_ic(s, fwd, window=2)
    array([ 1.,  1.,  0., -1.,  0.])
    """
    ic = rank_information_coefficient(scores, fwd_returns)
    out = np.full_like(ic, np.nan)
    for t in range(len(ic)):
        w = ic[max(0, t - window + 1): t + 1]
        valid = w[~np.isnan(w)]
        if valid.size:
            out[t] = valid.mean()
    return out
