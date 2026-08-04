"""Synthetic universe sanity tests."""

import numpy as np
import pytest

from fz.universe import DEFAULT_PREMIA, make_universe


def test_universe_shape():
    u = make_universe(n_stocks=50, n_days=300, seed=0)
    assert u.prices.shape == (300, 50)
    assert u.returns.shape == (300, 50)
    assert u.market_cap.shape == (300, 50)
    assert len(u.tickers) == 50


def test_universe_deterministic_under_seed():
    a = make_universe(n_stocks=30, n_days=100, seed=42)
    b = make_universe(n_stocks=30, n_days=100, seed=42)
    np.testing.assert_allclose(a.returns, b.returns, equal_nan=True)


def test_universe_prices_positive():
    u = make_universe(n_stocks=30, n_days=100, seed=1)
    assert (u.prices > 0).all()


def test_universe_returns_finite_after_day_zero():
    u = make_universe(n_stocks=30, n_days=200, seed=2)
    assert np.isnan(u.returns[0]).all()      # no prior price on day 0
    assert np.isfinite(u.returns[1:]).all()


def test_prices_start_at_100_and_compound_returns_exactly():
    """The Universe contract: simple returns, prices[0] = 100, and the price
    path is exactly the compounded return path (no log/simple mixing)."""
    u = make_universe(n_stocks=40, n_days=250, seed=3)
    np.testing.assert_allclose(u.prices[0], 100.0)
    implied = u.prices[1:] / u.prices[:-1] - 1.0
    np.testing.assert_allclose(implied, u.returns[1:], rtol=1e-10)


def test_premia_override_and_validation():
    base = make_universe(n_stocks=30, n_days=120, seed=5)
    placebo = make_universe(n_stocks=30, n_days=120, seed=5,
                            premia={"value": 0.0})
    # same seed, different priced structure -> different returns
    assert not np.allclose(base.returns[1:], placebo.returns[1:])
    with pytest.raises(ValueError):
        make_universe(n_stocks=10, n_days=50, premia={"vol": 0.0})
    assert set(DEFAULT_PREMIA) == {"value", "size", "quality", "low_vol", "momentum"}
