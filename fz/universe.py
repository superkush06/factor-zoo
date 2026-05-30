"""Synthetic equity universe generator.

We can't ship licensed market data, so this builds a realistic-looking
universe with:
  - latent factors (market, momentum, value, size, quality, vol)
  - per-stock factor exposures (slowly time-varying)
  - idiosyncratic returns
  - tradable prices derived from returns
  - "fundamentals" snapshots used to compute value/quality/etc

The cross-section is large enough (~500 stocks x ~5 years daily) that
factor premia recovered from this data approximate the true exposures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Universe:
    """A synthetic panel of (n_stocks x n_days) returns and characteristics."""
    prices: np.ndarray       # (n_days, n_stocks)
    returns: np.ndarray      # (n_days, n_stocks)
    market_cap: np.ndarray   # (n_days, n_stocks)  for size
    book_value: np.ndarray   # (n_days, n_stocks)  for value
    earnings: np.ndarray     # (n_days, n_stocks)  for quality
    dates: np.ndarray = field(default=None)  # type: ignore[assignment]
    tickers: list[str] = field(default_factory=list)


def make_universe(n_stocks: int = 200, n_days: int = 1260,
                  seed: int = 0) -> Universe:
    """Generate a synthetic universe of ~5y daily data for ~200 stocks.

    True factor structure:
      r_{i,t} = beta_i * r_M,t + lambda_mom * x_mom_i + lambda_val * x_val_i
                + lambda_size * x_size_i + lambda_vol * x_vol_i
                + idio noise
    With small "lambdas" so factors are detectable but noisy.
    """
    rng = np.random.default_rng(seed)

    # ---- Latent exposures (per stock, slowly time-varying) -----------
    base_size = rng.lognormal(mean=2.0, sigma=1.0, size=n_stocks)
    base_value = rng.normal(loc=0.0, scale=1.0, size=n_stocks)
    base_quality = rng.normal(loc=0.0, scale=1.0, size=n_stocks)
    base_low_vol = rng.normal(loc=0.0, scale=1.0, size=n_stocks)
    betas = rng.normal(loc=1.0, scale=0.3, size=n_stocks)

    # Market returns
    market_ret = rng.normal(loc=0.0003, scale=0.012, size=n_days)

    # True factor premia per day (cross-sectionally constant)
    # Values are realistic-ish per-day premia (annualised ~3-7%)
    prem_value   = rng.normal(loc=0.0001, scale=0.001, size=n_days)
    prem_size    = rng.normal(loc=0.00005, scale=0.0008, size=n_days)
    prem_quality = rng.normal(loc=0.0001, scale=0.001, size=n_days)
    prem_lowvol  = rng.normal(loc=0.0001, scale=0.001, size=n_days)

    # Idio noise — larger so the test cross-sections aren't trivial
    idio_sigma = rng.uniform(0.012, 0.04, size=n_stocks)
    idio = rng.normal(size=(n_days, n_stocks)) * idio_sigma[None, :]

    # Standardise base exposures cross-sectionally
    def _std(x): return (x - x.mean()) / (x.std() + 1e-12)

    s_value = _std(base_value)
    s_size = _std(np.log(base_size))   # smaller-cap -> high exposure to "size" factor
    s_quality = _std(base_quality)
    s_lowvol = _std(base_low_vol)

    # Daily returns
    returns = (betas[None, :] * market_ret[:, None]
               + np.outer(prem_value, s_value)
               + np.outer(-prem_size, s_size)       # SMB: small minus big
               + np.outer(prem_quality, s_quality)
               + np.outer(-prem_lowvol, s_lowvol)   # low-vol premium
               + idio)

    # Prices from log-returns starting at 100
    log_p = np.log(100.0) + np.cumsum(returns, axis=0)
    prices = np.exp(log_p)

    # Market cap drifts with price (proportional)
    market_cap = base_size[None, :] * prices

    # Book value: noisy lag of cap
    book_value = market_cap * np.exp(0.5 * base_value[None, :])

    # Earnings: noisy quality-weighted slice
    earnings = market_cap * 0.05 * (1.0 + 0.5 * base_quality[None, :])

    tickers = [f"SYN{i:04d}" for i in range(n_stocks)]
    return Universe(prices=prices, returns=returns,
                    market_cap=market_cap, book_value=book_value,
                    earnings=earnings, dates=np.arange(n_days),
                    tickers=tickers)
