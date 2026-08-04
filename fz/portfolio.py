"""Portfolio sorts and long-short return aggregation."""

from __future__ import annotations

import numpy as np

from .crossect import average_ranks


def quintile_sort_returns(scores: np.ndarray, fwd_returns: np.ndarray,
                          n_quantiles: int = 5) -> np.ndarray:
    """For each row of `scores`, sort into `n_quantiles` quantiles and
    compute the mean forward return per quantile.

    Returns shape (n_days, n_quantiles). NaN where scores are all NaN.
    Quantile 0 is bottom (smallest score), quantile n-1 is top.
    Tied scores share an average rank, so quantile membership does not
    depend on ticker order (winsorised scores always tie in the tails).
    """
    n_days, n_stocks = scores.shape
    out = np.full((n_days, n_quantiles), np.nan)
    for t in range(n_days):
        s = scores[t]
        r = fwd_returns[t]
        valid = ~np.isnan(s) & ~np.isnan(r)
        if valid.sum() < n_quantiles * 2:
            continue
        s_v = s[valid]; r_v = r[valid]
        ranks = average_ranks(s_v)
        n_v = ranks.shape[0]
        # Map rank to quantile in 0..n_quantiles-1
        q = np.floor(ranks * n_quantiles / n_v).astype(int).clip(max=n_quantiles - 1)
        for k in range(n_quantiles):
            mask = (q == k)
            if mask.any():
                out[t, k] = r_v[mask].mean()
    return out


def long_short_return(quintile_returns: np.ndarray) -> np.ndarray:
    """Top quantile minus bottom quantile per row."""
    return quintile_returns[:, -1] - quintile_returns[:, 0]


def cumulative(returns_1d: np.ndarray) -> np.ndarray:
    """Cumulative geometric return path (1+r1)(1+r2)... - 1."""
    out = np.full_like(returns_1d, np.nan)
    cum = 1.0
    for t, r in enumerate(returns_1d):
        if np.isnan(r):
            out[t] = cum - 1.0
            continue
        cum *= (1.0 + r)
        out[t] = cum - 1.0
    return out


def sharpe_annualised(returns_1d: np.ndarray, freq: int = 252) -> float:
    """Annualised Sharpe ratio from daily returns."""
    r = returns_1d[~np.isnan(returns_1d)]
    if r.size < 2:
        return float("nan")
    mu = r.mean() * freq
    sd = r.std(ddof=1) * np.sqrt(freq)
    return float(mu / sd) if sd > 0 else float("nan")
