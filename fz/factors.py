"""Factor characteristic computation."""

from __future__ import annotations

import numpy as np

from .universe import Universe


def _winsorize(x: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    """Clip values to (low, high) quantiles, ignoring NaN."""
    lo = np.nanquantile(x, low)
    hi = np.nanquantile(x, high)
    return np.clip(x, lo, hi)


def _standardize(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score (per row), handling NaN."""
    mu = np.nanmean(x, axis=-1, keepdims=True)
    sd = np.nanstd(x, axis=-1, keepdims=True) + 1e-12
    return (x - mu) / sd


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
    csum = np.cumsum(u.returns, axis=0)
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
