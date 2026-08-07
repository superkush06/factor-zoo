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

    Four names on constant log-drifts of 0, +1%, +2% and -1% a day: the
    score orders them by drift, and the winsorised tails keep the extremes
    from running away.

    >>> import numpy as np
    >>> from fz import universe_from_prices
    >>> drift = np.array([0.0, 0.01, 0.02, -0.01])
    >>> u = universe_from_prices(100.0 * np.exp(np.outer(np.arange(10.0), drift)))
    >>> np.round(momentum(u, lookback=5, skip=1)[-1], 3)
    array([-0.455,  0.455,  1.339, -1.339])
    >>> int(np.isnan(momentum(u, lookback=5, skip=1)[:, 0]).sum())
    5
    """
    log_p = np.log(u.prices)
    n_days = log_p.shape[0]
    raw = np.full_like(log_p, np.nan)
    for t in range(lookback, n_days):
        raw[t] = log_p[t - skip] - log_p[t - lookback]
    return _standardize(_winsorize(raw))


def short_reversal(u: Universe, window: int = 5) -> np.ndarray:
    """1-week mean-reversion: high recent return -> low expected next-day return.

    Two names, and over the last two days the first one is down 3% while the
    second is up 5%. The loser scores high, which is the whole sign
    convention:

    >>> import numpy as np
    >>> from fz import universe_from_prices
    >>> r = np.array([[0.0, 0.0], [0.05, -0.05], [0.01, 0.01], [-0.04, 0.04]])
    >>> u = universe_from_prices(100.0 * np.cumprod(1.0 + r, axis=0))
    >>> np.round(short_reversal(u, window=2)[-1], 3)
    array([ 1., -1.])
    """
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
    """Book-to-market ratio (higher -> "value"). Cross-sectionally z-scored.

    Four names on the same book value and market caps of 1x, 2x, 4x, 8x, so
    book-to-market halves along the row and the cheapest name scores highest:

    >>> import numpy as np
    >>> from fz import universe_from_prices
    >>> mcap = 1e9 * np.array([[1.0, 2.0, 4.0, 8.0]] * 3)
    >>> u = universe_from_prices(np.full((3, 4), 100.0), market_cap=mcap,
    ...                          book_value=np.full((3, 4), 5e8))
    >>> np.round(value_btm(u)[0], 3)
    array([ 1.339,  0.455, -0.455, -1.339])

    Supply no book values and the score is NaN rather than an error, so a
    price-only panel still runs the price-driven factors:

    >>> bool(np.all(np.isnan(value_btm(universe_from_prices(np.full((3, 4), 100.0))))))
    True
    """
    raw = u.book_value / (u.market_cap + 1e-12)
    return _standardize(_winsorize(np.log(np.maximum(raw, 1e-12))))


def size_factor(u: Universe) -> np.ndarray:
    """Smaller cap -> larger 'size factor' exposure (SMB convention).

    >>> import numpy as np
    >>> from fz import universe_from_prices
    >>> mcap = 1e9 * np.array([[1.0, 2.0, 4.0, 8.0]] * 3)
    >>> u = universe_from_prices(np.full((3, 4), 100.0), market_cap=mcap)
    >>> np.round(size_factor(u)[0], 3)
    array([ 1.339,  0.455, -0.455, -1.339])

    Market cap is shares times price, so on a real panel this score carries
    accumulated return as well as size -- see the IC table in README.
    """
    raw = -np.log(u.market_cap + 1e-12)
    return _standardize(_winsorize(raw))


def quality_roe(u: Universe) -> np.ndarray:
    """Return-on-equity proxy: earnings / book.

    Four names on the same book and ROEs of 2%, 4%, 6%, 8%:

    >>> import numpy as np
    >>> from fz import universe_from_prices
    >>> book = np.full((3, 4), 5e8)
    >>> u = universe_from_prices(np.full((3, 4), 100.0), book_value=book,
    ...                          earnings=book * np.array([0.02, 0.04, 0.06, 0.08]))
    >>> np.round(quality_roe(u)[0], 3)
    array([-1.339, -0.455,  0.455,  1.339])
    """
    raw = u.earnings / (u.book_value + 1e-12)
    return _standardize(_winsorize(raw))


def low_vol(u: Universe, window: int = 60) -> np.ndarray:
    """Negative trailing standard deviation: low-vol -> high exposure.

    Three names alternating up and down at amplitudes 1%, 3% and 0.5%: the
    quietest one scores highest, the wildest lowest.

    >>> import numpy as np
    >>> from fz import universe_from_prices
    >>> flip = (-1.0) ** np.arange(9.0)
    >>> r = flip[:, None] * np.array([0.01, 0.03, 0.005])
    >>> u = universe_from_prices(100.0 * np.cumprod(1.0 + r, axis=0),
    ...                          tickers=["CALM", "WILD", "QUIET"])
    >>> np.round(low_vol(u, window=4)[-1], 3)
    array([ 0.463, -1.389,  0.926])
    """
    n_days, n_stocks = u.returns.shape
    raw = np.full((n_days, n_stocks), np.nan)
    for t in range(window, n_days):
        sl = u.returns[t - window: t]
        raw[t] = -sl.std(axis=0)
    return _standardize(_winsorize(raw))
