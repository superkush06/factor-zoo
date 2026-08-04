"""No-look-ahead (point-in-time) invariance tests.

Every score at day t must depend only on data up to day t. The check:
compute scores on the full universe, then on the universe truncated at
day T -- the first T rows must be identical. The old panel-global
winsorisation failed this by up to ~0.9 z-units for momentum.
"""

import dataclasses

import numpy as np
import pytest

from fz.crossect import rolling_ic
from fz.factors import (
    low_vol,
    momentum,
    quality_roe,
    short_reversal,
    size_factor,
    value_btm,
)
from fz.portfolio import quintile_sort_returns
from fz.universe import make_universe

T_CUT = 400

FACTOR_FNS = [
    lambda u: momentum(u, lookback=252, skip=21),
    lambda u: short_reversal(u, window=5),
    value_btm,
    size_factor,
    quality_roe,
    lambda u: low_vol(u, window=60),
]
FACTOR_IDS = ["momentum", "short_reversal", "value", "size", "quality", "low_vol"]


def _truncate(u, T):
    return dataclasses.replace(
        u,
        prices=u.prices[:T],
        returns=u.returns[:T],
        market_cap=u.market_cap[:T],
        book_value=u.book_value[:T],
        earnings=u.earnings[:T],
        dates=u.dates[:T],
    )


@pytest.fixture(scope="module")
def universe():
    return make_universe(n_stocks=120, n_days=600, seed=3)


@pytest.mark.parametrize("fn", FACTOR_FNS, ids=FACTOR_IDS)
def test_factor_scores_are_point_in_time(universe, fn):
    full = fn(universe)
    part = fn(_truncate(universe, T_CUT))
    np.testing.assert_allclose(part, full[:T_CUT], equal_nan=True, atol=1e-12)


def test_rolling_ic_is_point_in_time(universe):
    fwd = np.full_like(universe.returns, np.nan)
    fwd[:-1] = universe.returns[1:]
    scores = value_btm(universe)
    full = rolling_ic(scores, fwd, window=60)
    part = rolling_ic(scores[:T_CUT], fwd[:T_CUT], window=60)
    np.testing.assert_allclose(part, full[:T_CUT], equal_nan=True, atol=1e-12)


def test_quintile_sort_is_point_in_time(universe):
    fwd = np.full_like(universe.returns, np.nan)
    fwd[:-1] = universe.returns[1:]
    scores = value_btm(universe)
    full = quintile_sort_returns(scores, fwd, n_quantiles=5)
    part = quintile_sort_returns(scores[:T_CUT], fwd[:T_CUT], n_quantiles=5)
    np.testing.assert_allclose(part, full[:T_CUT], equal_nan=True, atol=1e-12)
