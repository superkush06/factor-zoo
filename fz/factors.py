"""Factor characteristic computation."""

from __future__ import annotations

import numpy as np

from .universe import Universe


def _winsorize(x: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Clip each row (one date's cross-section) to its own (low, high) quantiles.

    Quantiles are computed per date, never over the pooled panel: a score at
    day t must not depend on the distribution of data from day t+1 onward.
    Rows that are entirely NaN (factor warm-up) pass through untouched.
    """
    out = np.asarray(x, dtype=float).copy()
    row_ok = ~np.all(np.isnan(out), axis=-1)
    if not row_ok.any():
        return out
    lo = np.nanquantile(out[row_ok], low, axis=-1, keepdims=True)
    hi = np.nanquantile(out[row_ok], high, axis=-1, keepdims=True)
    out[row_ok] = np.clip(out[row_ok], lo, hi)
    return out


def _standardize(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score (per row), handling NaN.

    All-NaN rows (factor warm-up) stay NaN without tripping numpy's
    mean-of-empty-slice / degrees-of-freedom RuntimeWarnings.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    row_ok = ~np.all(np.isnan(x), axis=-1)
    if not row_ok.any():
        return out
    sub = x[row_ok]
    mu = np.nanmean(sub, axis=-1, keepdims=True)
    sd = np.nanstd(sub, axis=-1, keepdims=True) + 1e-12
    out[row_ok] = (sub - mu) / sd
    return out


def momentum(u: Universe, lookback: int = 252, skip: int = 21) -> np.ndarray:
    """12m-minus-1m momentum (price-driven).

    Returns shape (n_days, n_stocks); first `lookback` rows are NaN.
    """
    log_p = np.log(u.prices)
    n_days = log_p.shape[0]
    raw = np.full_like(log_p, np.nan)
    for t in range(lookback, n_days):
        raw[t] = log_p[t - skip] - log_p[t - lookback]
    return _standardize(_winsorize(raw))


def short_reversal(u: Universe, window: int = 5) -> np.ndarray:
    """1-week mean-reversion: high recent return -> low expected next-day return."""
    n_days, n_stocks = u.returns.shape
    raw = np.full((n_days, n_stocks), np.nan)
    # nancumsum: day 0's return is NaN by contract (no prior price). It never
    # falls inside a t >= window trailing window, so treating it as 0 in the
    # cumulative sum is exact, not an approximation.
    csum = np.nancumsum(u.returns, axis=0)
    for t in range(window, n_days):
        raw[t] = -(csum[t] - csum[t - window])
    return _standardize(_winsorize(raw))


def value_btm(u: Universe) -> np.ndarray:
    """Book-to-market ratio (higher -> "value"). Cross-sectionally z-scored."""
    raw = u.book_value / (u.market_cap + 1e-12)
    return _standardize(_winsorize(np.log(np.maximum(raw, 1e-12))))


def size_factor(u: Universe) -> np.ndarray:
    """Smaller cap -> larger 'size factor' exposure (SMB convention)."""
    raw = -np.log(u.market_cap + 1e-12)
    return _standardize(_winsorize(raw))


def quality_roe(u: Universe) -> np.ndarray:
    """Return-on-equity proxy: earnings / book."""
    raw = u.earnings / (u.book_value + 1e-12)
    return _standardize(_winsorize(raw))


def low_vol(u: Universe, window: int = 60) -> np.ndarray:
    """Negative trailing standard deviation: low-vol -> high exposure."""
    n_days, n_stocks = u.returns.shape
    raw = np.full((n_days, n_stocks), np.nan)
    for t in range(window, n_days):
        sl = u.returns[t - window: t]
        raw[t] = -sl.std(axis=0)
    return _standardize(_winsorize(raw))
