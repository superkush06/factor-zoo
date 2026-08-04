"""Forward-return alignment and Fama-MacBeth standard errors.

The interesting case is overlapping returns: hold a name for h days and
consecutive cross-sections share h-1 days of data, so the daily slopes are
autocorrelated by construction and the iid standard error understates the
truth. These tests pin the HAC estimator against an analytic AR(1) answer
and then show it doing its job on the real pipeline.
"""

import numpy as np
import pytest

from fz.crossect import (
    fama_macbeth,
    forward_returns,
    newey_west_var,
    shanken_factor,
)
from fz.factors import momentum, value_btm
from fz.universe import make_universe


def test_forward_returns_horizon_one_is_a_shift():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=(50, 7))
    r[0] = np.nan
    fwd = forward_returns(r, horizon=1)
    np.testing.assert_allclose(fwd[:-1], r[1:])
    assert np.all(np.isnan(fwd[-1]))


def test_forward_returns_compounds_geometrically():
    r = np.array([[np.nan], [0.10], [0.10], [0.10]])
    fwd = forward_returns(r, horizon=3)
    assert fwd[0, 0] == pytest.approx(1.10 ** 3 - 1.0)
    assert np.all(np.isnan(fwd[1:]))


def test_forward_returns_rejects_zero_horizon():
    with pytest.raises(ValueError):
        forward_returns(np.zeros((5, 2)), horizon=0)


def test_newey_west_zero_lags_is_the_sample_variance():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    assert newey_west_var(x, lags=0) == pytest.approx(np.var(x, ddof=1))


def test_newey_west_matches_analytic_ar1_long_run_variance():
    """For a stationary AR(1) with autocorrelation rho^l, the Bartlett
    long-run variance is sigma^2 * (1 + 2 * sum_l (1 - l/(L+1)) rho^l)."""
    rho, sigma, lags, T = 0.7, 1.0, 12, 400_000
    rng = np.random.default_rng(7)
    innov = rng.standard_normal(T) * sigma * np.sqrt(1.0 - rho ** 2)
    x = np.empty(T)
    x[0] = sigma * rng.standard_normal()
    for t in range(1, T):
        x[t] = rho * x[t - 1] + innov[t]

    weights = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    analytic = sigma ** 2 * (1.0 + 2.0 * np.sum(weights * rho ** np.arange(1, lags + 1)))
    assert newey_west_var(x, lags=lags) == pytest.approx(analytic, rel=0.03)


def test_fama_macbeth_standard_errors_reproduce_the_classic_tstat():
    u = make_universe(n_stocks=150, n_days=500, seed=4)
    fwd = forward_returns(u.returns)
    res = fama_macbeth([value_btm(u)], fwd, add_intercept=True)
    daily = res.daily_coefs[~np.any(np.isnan(res.daily_coefs), axis=1)]
    classic = daily.std(axis=0, ddof=1) / np.sqrt(daily.shape[0])
    np.testing.assert_allclose(res.std_errors, classic, rtol=1e-10)
    np.testing.assert_allclose(res.t_stats, res.coefficients / classic, rtol=1e-6)
    assert res.n_periods == daily.shape[0]


def test_overlapping_horizons_need_hac_errors():
    """A 21-day holding period reuses 20 of every 21 days. The iid standard
    error treats those as independent evidence; the HAC one does not."""
    u = make_universe(n_stocks=200, n_days=1200, seed=5)
    scores = [momentum(u, lookback=252, skip=21), value_btm(u)]
    fwd = forward_returns(u.returns, horizon=21)

    naive = fama_macbeth(scores, fwd, hac_lags=0)
    hac = fama_macbeth(scores, fwd, hac_lags=20)

    np.testing.assert_allclose(hac.coefficients, naive.coefficients, rtol=1e-12)
    assert np.all(hac.std_errors[1:] > 1.5 * naive.std_errors[1:])
    assert np.all(hac.t_stats[1:] < naive.t_stats[1:])


def test_shanken_factor_is_one_plus_the_squared_sharpe():
    """1 + lam' Sigma_f^-1 lam. A zero premium needs no correction; a premium
    of 0.8 factor standard deviations needs 1.64; and because the quantity is
    a squared Sharpe ratio it is invariant to rescaling premia and volatility
    together."""
    assert shanken_factor([0.0], [[0.04 ** 2]]) == pytest.approx(1.0)
    assert shanken_factor([0.04], [[0.05 ** 2]]) == pytest.approx(1.64)
    assert shanken_factor([0.08], [[0.1 ** 2]]) == pytest.approx(1.64)


def test_shanken_factor_adds_squared_sharpes_across_orthogonal_factors():
    two = shanken_factor([0.01, 0.005], [[0.04 ** 2, 0.0], [0.0, 0.03 ** 2]])
    assert two == pytest.approx(1.0 + 0.01 ** 2 / 0.04 ** 2 + 0.005 ** 2 / 0.03 ** 2)
    # correlated factors are not the sum: the inverse does the work
    cov = [[0.04 ** 2, 0.5 * 0.04 * 0.03], [0.5 * 0.04 * 0.03, 0.03 ** 2]]
    lam = np.array([0.01, 0.005])
    assert shanken_factor(lam, cov) == pytest.approx(
        1.0 + float(lam @ np.linalg.solve(np.array(cov), lam)))


def test_shanken_factor_rejects_a_mismatched_covariance():
    with pytest.raises(ValueError):
        shanken_factor([0.01, 0.02], [[1.0]])
