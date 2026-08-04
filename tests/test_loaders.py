"""Real-data loader tests."""

import pathlib

import numpy as np
import pytest

from fz.factors import momentum
from fz.loaders import load_prices_csv, universe_from_prices

SAMPLE = pathlib.Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"


def test_universe_from_prices_shapes_and_returns():
    prices = np.array([[100.0, 50.0], [110.0, 55.0], [99.0, 60.5]])
    u = universe_from_prices(prices, tickers=["X", "Y"])
    assert u.prices.shape == (3, 2)
    assert u.returns.shape == (3, 2)
    assert np.isnan(u.returns[0]).all()                             # row 0: no prior price
    np.testing.assert_allclose(u.returns[1], [0.10, 0.10])          # +10% both
    assert u.tickers == ["X", "Y"]


def test_universe_from_prices_fundamentals_default_nan():
    u = universe_from_prices(np.array([[1.0], [2.0]]))
    assert np.isnan(u.market_cap).all()
    assert np.isnan(u.book_value).all()
    assert np.isnan(u.earnings).all()


def test_universe_from_prices_validates():
    with pytest.raises(ValueError):
        universe_from_prices(np.array([1.0, 2.0]))              # not 2-D
    with pytest.raises(ValueError):
        universe_from_prices(np.array([[1.0, 2.0]]))            # <2 rows
    with pytest.raises(ValueError):
        universe_from_prices(np.array([[1.0], [2.0]]), tickers=["A", "B"])


def test_price_factor_works_on_loaded_universe():
    """A price-based factor (momentum) should produce finite scores from a
    loaded real-data universe, even with no fundamentals."""
    u = load_prices_csv(str(SAMPLE))
    s = momentum(u, lookback=20, skip=2)
    # last row should be finite + cross-sectionally standardized (~zero mean)
    last = s[-1]
    assert np.isfinite(last).all()
    assert abs(last.mean()) < 1e-6


def test_load_prices_csv_dimensions():
    u = load_prices_csv(str(SAMPLE))
    assert u.prices.shape[1] == 5                  # 5 tickers
    assert u.tickers[0] == "AAPL"
    assert len(u.dates) == u.prices.shape[0]
