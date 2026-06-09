"""Factor characteristic tests."""

import numpy as np

from fz.factors import (
    low_vol,
    momentum,
    quality_roe,
    short_reversal,
    size_factor,
    value_btm,
)
from fz.universe import make_universe


def test_momentum_skip_window_is_nan():
    u = make_universe(n_stocks=40, n_days=300, seed=0)
    s = momentum(u, lookback=100, skip=10)
    # First 100 rows should be all NaN
    assert np.isnan(s[:100]).all()


def test_momentum_is_zero_centered():
    u = make_universe(n_stocks=80, n_days=500, seed=0)
    s = momentum(u, lookback=120, skip=10)
    last = s[300]
    assert abs(last.mean()) < 1e-6
    assert abs(last.std() - 1.0) < 1e-3


def test_value_is_finite_and_centered():
    u = make_universe(n_stocks=60, n_days=200, seed=1)
    s = value_btm(u)
    assert np.isfinite(s).all()
    assert abs(s.mean(axis=-1)).max() < 1e-6


def test_low_vol_higher_for_quiet_stocks():
    """Stocks with smaller returns dispersion should rank higher in low_vol."""
    u = make_universe(n_stocks=50, n_days=300, seed=0)
    lv = low_vol(u, window=60)
    # Last row vs actual return std over the last 60 days
    actual_vol = u.returns[-60:].std(axis=0)
    score_last = lv[-1]
    # Negative correlation between actual vol and "low_vol" score
    z = (actual_vol - actual_vol.mean()) / actual_vol.std()
    s_z = (score_last - score_last.mean()) / score_last.std()
    corr = float((z * s_z).mean())
    assert corr < -0.9


def test_size_factor_uses_log_market_cap():
    """Smaller market cap -> larger size factor exposure."""
    u = make_universe(n_stocks=60, n_days=120, seed=0)
    s = size_factor(u)[-1]
    mc = u.market_cap[-1]
    # Negative correlation between log(market cap) and size factor
    z = (np.log(mc) - np.log(mc).mean()) / np.log(mc).std()
    s_z = (s - s.mean()) / s.std()
    corr = float((z * s_z).mean())
    assert corr < -0.9


def test_quality_short_reversal_finite():
    u = make_universe(n_stocks=40, n_days=200, seed=0)
    assert np.isfinite(quality_roe(u)).all()
    s = short_reversal(u, window=5)
    assert np.isfinite(s[10:]).all()
