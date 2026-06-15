"""Cross-sectional regressions: Fama-MacBeth and IC over time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FamaMacBethResult:
    coefficients: np.ndarray   # (n_factors,) mean of daily betas
    t_stats: np.ndarray        # (n_factors,) Newey-West-style not implemented here
    daily_coefs: np.ndarray    # (n_days, n_factors)
    daily_r2: np.ndarray       # (n_days,)


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
                 add_intercept: bool = True) -> FamaMacBethResult:
    """Daily cross-sectional regressions of forward returns on factor scores.

    `scores_list[k]` shape: (n_days, n_stocks). Returns daily coefficients;
    Fama-MacBeth t-stat is mean / (std / sqrt(N)) of daily coefs.
    """
    n_days, n_stocks = scores_list[0].shape
    K = len(scores_list)
    n_cols = K + (1 if add_intercept else 0)
    daily_coefs = np.full((n_days, n_cols), np.nan)
    daily_r2 = np.full(n_days, np.nan)

    for t in range(n_days):
        # Build design matrix
        cols = []
        if add_intercept:
            cols.append(np.ones(n_stocks))
        for s in scores_list:
            cols.append(s[t])
        X = np.column_stack(cols)
        y = fwd_returns[t]
        b, r2 = _ols_slope(X, y)
        daily_coefs[t] = b
        daily_r2[t] = r2

    # Aggregate Fama-MacBeth point estimate + t-stat
    valid_rows = ~np.any(np.isnan(daily_coefs), axis=1)
    coefs = daily_coefs[valid_rows]
    mean = coefs.mean(axis=0)
    sd = coefs.std(axis=0, ddof=1)
    n = coefs.shape[0]
    t = mean / (sd / np.sqrt(max(n, 1)) + 1e-12)
    return FamaMacBethResult(coefficients=mean, t_stats=t,
                             daily_coefs=daily_coefs, daily_r2=daily_r2)


def rank_information_coefficient(scores: np.ndarray, fwd_returns: np.ndarray):
    """Per-day Spearman correlation between score and forward return.

    Returns (n_days,) IC series. NaN if too few valid points.
    """
    n_days = scores.shape[0]
    out = np.full(n_days, np.nan)
    for t in range(n_days):
        s = scores[t]; r = fwd_returns[t]
        valid = ~np.isnan(s) & ~np.isnan(r)
        if valid.sum() < 20:
            continue
        s_r = _ranks(s[valid])
        r_r = _ranks(r[valid])
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
    """
    ic = rank_information_coefficient(scores, fwd_returns)
    out = np.full_like(ic, np.nan)
    for t in range(len(ic)):
        w = ic[max(0, t - window + 1): t + 1]
        valid = w[~np.isnan(w)]
        if valid.size:
            out[t] = valid.mean()
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)
