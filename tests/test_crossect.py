"""Fama-MacBeth and IC tests."""

import numpy as np

from fz.crossect import fama_macbeth, rank_information_coefficient
from fz.factors import value_btm
from fz.universe import make_universe


def test_fama_macbeth_recovers_loaded_factor():
    """Build a synthetic universe with a strong VALUE premium and check
    that Fama-MacBeth on the value characteristic recovers a positive coefficient.
    """
    u = make_universe(n_stocks=200, n_days=600, seed=1)
    s_val = value_btm(u)
    # Shift returns forward by one day
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]
    res = fama_macbeth([s_val], fwd, add_intercept=True)
    # coef[1] is the value factor
    assert res.coefficients[1] > 0
    assert res.t_stats[1] > 1.5  # crude — premium is loaded into the data


def test_ic_in_unit_interval():
    rng = np.random.default_rng(0)
    n_days, n_stocks = 50, 100
    scores = rng.standard_normal((n_days, n_stocks))
    fwd = rng.standard_normal((n_days, n_stocks))
    ic = rank_information_coefficient(scores, fwd)
    assert np.all(np.isnan(ic) | ((ic >= -1) & (ic <= 1)))


def test_fama_macbeth_intercept_recovers_market():
    """Daily intercept should approximate average market return."""
    u = make_universe(n_stocks=150, n_days=400, seed=2)
    s = value_btm(u)
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]
    res = fama_macbeth([s], fwd, add_intercept=True)
    # The intercept averages cross-sectional means each day
    assert abs(res.coefficients[0]) < 0.01  # ~daily return scale


def test_rolling_ic_smooths_daily_ic():
    import numpy as np

    from fz.crossect import rank_information_coefficient, rolling_ic
    rng = np.random.default_rng(0)
    n_days, n_stocks = 120, 80
    scores = rng.standard_normal((n_days, n_stocks))
    fwd = rng.standard_normal((n_days, n_stocks))
    ric = rolling_ic(scores, fwd, window=30)
    daily = rank_information_coefficient(scores, fwd)
    assert ric.shape == (n_days,)
    # rolling series has lower variance than the raw daily IC
    assert np.nanstd(ric) < np.nanstd(daily)
    # values stay in [-1, 1]
    assert np.all(np.isnan(ric) | ((ric >= -1) & (ric <= 1)))
