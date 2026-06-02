"""Synthetic universe sanity tests."""

import numpy as np

from fz.universe import make_universe


def test_universe_shape():
    u = make_universe(n_stocks=50, n_days=300, seed=0)
    assert u.prices.shape == (300, 50)
    assert u.returns.shape == (300, 50)
    assert u.market_cap.shape == (300, 50)
    assert len(u.tickers) == 50


def test_universe_deterministic_under_seed():
    a = make_universe(n_stocks=30, n_days=100, seed=42)
    b = make_universe(n_stocks=30, n_days=100, seed=42)
    np.testing.assert_allclose(a.returns, b.returns)


def test_universe_prices_positive():
    u = make_universe(n_stocks=30, n_days=100, seed=1)
    assert (u.prices > 0).all()


def test_universe_returns_finite():
    u = make_universe(n_stocks=30, n_days=200, seed=2)
    assert np.isfinite(u.returns).all()
