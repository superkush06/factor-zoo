"""Synthetic equity universe generator.

We can't ship licensed market data, so this builds a realistic-looking
universe with:
  - latent factors (market, momentum, value, size, quality, vol)
  - per-stock factor exposures
  - idiosyncratic returns
  - tradable prices derived from returns
  - "fundamentals" snapshots used to compute value/quality/etc

Every advertised premium is *identifiable*: it flows through the
characteristic that measures it (momentum through trailing returns,
low-vol through realized volatility, value through book-to-market,
quality through ROE). Zero any entry of `DEFAULT_PREMIA` via the
`premia` argument to build a placebo universe where the characteristic
exists but is unpriced -- the acceptance tests use exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Mean daily *arithmetic* premium per factor (~7.6%/yr for value/quality/
#: low_vol, ~2.5%/yr for size). "momentum" is parameterised differently: it is the
#: stationary cross-sectional dispersion (daily sd) of a persistent AR(1)
#: expected-return drift -- persistence is what makes trailing 12-1 month
#: returns predictive, so dispersion of the drift *is* the momentum premium.
DEFAULT_PREMIA: dict[str, float] = {
    "value": 3e-4,
    "size": 1e-4,
    "quality": 3e-4,
    "low_vol": 4e-4,
    "momentum": 9e-4,
}


@dataclass
class Universe:
    """A synthetic panel of (n_stocks x n_days) returns and characteristics."""
    prices: np.ndarray       # (n_days, n_stocks)
    returns: np.ndarray      # (n_days, n_stocks)  simple returns; row 0 is NaN
    market_cap: np.ndarray   # (n_days, n_stocks)  for size
    book_value: np.ndarray   # (n_days, n_stocks)  for value
    earnings: np.ndarray     # (n_days, n_stocks)  for quality
    dates: np.ndarray = field(default=None)  # type: ignore[assignment]
    tickers: list[str] = field(default_factory=list)


def make_universe(n_stocks: int = 200, n_days: int = 1260, seed: int = 0,
                  premia: dict[str, float] | None = None) -> Universe:
    """Generate a synthetic universe of ~5y daily data for ~200 stocks.

    True factor structure (daily log-return shocks):

        r_{i,t} = beta_i * m_t + drift_{i,t}
                  + lam_val,t * v_i - lam_size,t * s_i
                  + lam_qual,t * q_i + lam_lowvol,t * l_i
                  + eps_{i,t}

    Each premium is wired to the characteristic that measures it:

      - drift_{i,t} is a slow AR(1) per-stock expected return (phi = 0.995),
        so the trailing 12-1 month return -- the momentum characteristic --
        genuinely predicts forward returns;
      - sd(eps_i) decreases monotonically in l_i, so trailing realized
        volatility carries the low-vol premium instead of an invisible
        latent variable;
      - log book-to-market is monotone in v_i, and ROE is monotone in q_i
        with earnings derived from *book value*, not market cap, so the
        quality score is not a disguised (anti-)value bet.

    `premia` overrides entries of `DEFAULT_PREMIA`. Setting one to 0.0
    yields a placebo universe: the characteristic still exists, it just
    isn't priced (the daily factor-return noise is retained).

    Contract: `returns` are simple returns with `returns[0] = NaN` (no
    prior price); `prices` start at 100.0 and compound `returns` exactly,
    so `prices[1:] / prices[:-1] - 1 == returns[1:]`.

    >>> import numpy as np
    >>> u = make_universe(n_stocks=50, n_days=100, seed=0)
    >>> u.returns.shape, float(u.prices[0, 0]), bool(np.isnan(u.returns[0, 0]))
    ((100, 50), 100.0, True)
    >>> bool(np.allclose(u.prices[1:] / u.prices[:-1] - 1.0, u.returns[1:]))
    True

    The seed is the whole panel, so a re-run is the same universe:

    >>> bool(np.array_equal(u.prices, make_universe(50, 100, seed=0).prices))
    True
    >>> sorted(DEFAULT_PREMIA)
    ['low_vol', 'momentum', 'quality', 'size', 'value']
    >>> make_universe(50, 100, seed=0, premia={"nope": 0.0})
    Traceback (most recent call last):
        ...
    ValueError: unknown premia keys: ['nope']
    """
    rng = np.random.default_rng(seed)
    lam = dict(DEFAULT_PREMIA)
    if premia:
        unknown = set(premia) - set(lam)
        if unknown:
            raise ValueError(f"unknown premia keys: {sorted(unknown)}")
        lam.update(premia)

    # ---- Latent exposures (per stock) ---------------------------------
    base_size = rng.lognormal(mean=2.0, sigma=1.0, size=n_stocks)
    base_value = rng.normal(loc=0.0, scale=1.0, size=n_stocks)
    base_quality = rng.normal(loc=0.0, scale=1.0, size=n_stocks)
    base_low_vol = rng.normal(loc=0.0, scale=1.0, size=n_stocks)
    betas = rng.normal(loc=1.0, scale=0.3, size=n_stocks)

    # Standardise base exposures cross-sectionally
    def _std(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    s_value = _std(base_value)
    s_size = _std(np.log(base_size))   # smaller-cap -> high exposure to "size"
    s_quality = _std(base_quality)
    s_lowvol = _std(base_low_vol)

    # Market returns
    market_ret = rng.normal(loc=0.0003, scale=0.012, size=n_days)

    # Daily premia: priced mean + factor-return noise. A placebo universe
    # zeroes the mean but keeps the noise (factor returns still wiggle,
    # they just don't pay).
    prem_value = lam["value"] + 0.7e-3 * rng.standard_normal(n_days)
    prem_size = lam["size"] + 0.6e-3 * rng.standard_normal(n_days)
    prem_quality = lam["quality"] + 0.7e-3 * rng.standard_normal(n_days)
    prem_lowvol = lam["low_vol"] + 0.7e-3 * rng.standard_normal(n_days)

    def _slow_ar1(scale: float, rho: float) -> np.ndarray:
        """Stationary per-stock AR(1) panel with stationary sd = `scale`."""
        w = np.zeros((n_days, n_stocks))
        innov = rng.standard_normal((n_days, n_stocks)) * (scale * np.sqrt(1.0 - rho ** 2))
        w[0] = scale * rng.standard_normal(n_stocks)
        for t in range(1, n_days):
            w[t] = rho * w[t - 1] + innov[t]
        return w

    # Momentum premium: persistent expected-return drift. Trailing 12-1
    # returns aggregate the drift, and the drift persists, so past winners
    # keep drifting up: the momentum characteristic is priced.
    drift = _slow_ar1(lam["momentum"], rho=0.995)

    # Idiosyncratic noise: quiet stocks are *genuinely* quiet. The premium-
    # bearing exposure is the standardized (negative) TRUE total volatility
    # -- idio plus beta-times-market -- which is precisely the quantity the
    # low_vol characteristic (trailing realized vol) estimates.
    idio_sigma = 0.019 - 0.009 * np.tanh(s_lowvol)
    idio = rng.standard_normal((n_days, n_stocks)) * idio_sigma[None, :]
    true_vol = np.sqrt(idio_sigma ** 2 + (betas * 0.012) ** 2)
    s_lv_exposure = _std(-true_vol)

    # Ito / Jensen correction: premia are specified in *arithmetic* returns,
    # so subtract each stock's half-variance drift. Without it, the +sigma^2/2
    # convexity of exp() would hand high-vol stocks a spurious ~5bp/day edge
    # that silently cancels the low-vol premium.
    half_var = 0.5 * (true_vol ** 2 + lam["momentum"] ** 2)

    # Daily log-return shocks
    log_r = (betas[None, :] * market_ret[:, None]
             + drift
             + np.outer(prem_value, s_value)
             + np.outer(-prem_size, s_size)      # SMB: small minus big
             + np.outer(prem_quality, s_quality)
             + np.outer(prem_lowvol, s_lv_exposure)  # quiet stocks earn the premium
             - half_var[None, :]
             + idio)

    # Simple-returns contract: prices start at 100 and compound `returns`
    # exactly; day 0 has no prior price so returns[0] is NaN.
    returns = np.exp(log_r) - 1.0
    returns[0] = np.nan
    prices = np.empty((n_days, n_stocks))
    prices[0] = 100.0
    prices[1:] = 100.0 * np.cumprod(1.0 + returns[1:], axis=0)

    # Market cap drifts with price (proportional)
    market_cap = base_size[None, :] * prices

    # Fundamentals: slow AR(1) wiggle keeps daily cross-sections from being
    # verbatim repeats without burying the signal.
    # log(B/M) = 0.5 v + wiggle  ->  value characteristic measures v.
    book_value = market_cap * np.exp(0.5 * base_value[None, :]
                                     + _slow_ar1(0.10, rho=0.99))
    # ROE = 6% + 2% q + wiggle  ->  quality measures q; earnings come from
    # book value, not market cap, so quality is decoupled from value.
    roe = (0.06 + 0.02 * base_quality[None, :] + _slow_ar1(0.008, rho=0.99))
    earnings = book_value * roe

    tickers = [f"SYN{i:04d}" for i in range(n_stocks)]
    return Universe(prices=prices, returns=returns,
                    market_cap=market_cap, book_value=book_value,
                    earnings=earnings, dates=np.arange(n_days),
                    tickers=tickers)
