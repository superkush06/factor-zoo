"""Portfolio sort tests."""

import numpy as np
import pytest

from fz.portfolio import (
    cumulative,
    long_short_return,
    quintile_sort_returns,
    sharpe_annualised,
)


def test_quintile_sort_shape():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((20, 100))
    fwd = rng.standard_normal((20, 100))
    out = quintile_sort_returns(scores, fwd, n_quantiles=5)
    assert out.shape == (20, 5)


def test_long_short_picks_extremes():
    """Top quintile minus bottom quintile."""
    scores = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).astype(float)
    fwd = np.array([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]).astype(float)
    out = quintile_sort_returns(scores, fwd, n_quantiles=5)
    ls = long_short_return(out)
    # Top quintile mean (90, 100) = 95; bottom (10, 20) = 15; spread = 80
    assert ls[0] == pytest.approx(80.0, abs=1e-6)


def test_cumulative_compounds():
    daily = np.array([0.01, 0.02, -0.01])
    cum = cumulative(daily)
    expected = [0.01, 1.01 * 1.02 - 1.0, 1.01 * 1.02 * 0.99 - 1.0]
    np.testing.assert_allclose(cum, expected)


def test_sharpe_makes_sense():
    rng = np.random.default_rng(0)
    # Returns with mean 0.001/day, std 0.01/day -> Sharpe ~ 0.001/0.01*sqrt(252) ~ 1.59
    returns = rng.normal(loc=0.001, scale=0.01, size=10_000)
    s = sharpe_annualised(returns)
    assert 1.3 < s < 1.9
