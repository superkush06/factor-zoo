"""Acceptance matrix: every advertised premium is recoverable, and nothing
else is. For each factor in the four-factor design, Fama-MacBeth must find
t > 2 for its own loaded premium, and |t| < 2 in a placebo universe where
that premium is zeroed (the characteristic still exists, it just isn't
priced)."""

import numpy as np
import pytest

from fz.crossect import fama_macbeth
from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.universe import make_universe

DESIGN = ["momentum", "value", "quality", "low_vol"]
N_STOCKS, N_DAYS, SEED = 300, 1000, 11


def _fm_tstats(premia=None):
    u = make_universe(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED, premia=premia)
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]
    scores = [
        momentum(u, lookback=252, skip=21),
        value_btm(u),
        quality_roe(u),
        low_vol(u, window=120),
    ]
    res = fama_macbeth(scores, fwd, add_intercept=True)
    return dict(zip(DESIGN, res.t_stats[1:], strict=True))


@pytest.fixture(scope="module")
def loaded_tstats():
    return _fm_tstats()


@pytest.mark.parametrize("factor", DESIGN)
def test_loaded_premium_is_recovered(loaded_tstats, factor):
    assert loaded_tstats[factor] > 2.0, (
        f"{factor}: t={loaded_tstats[factor]:.2f} — the premium this universe "
        f"advertises is not recoverable from its own characteristic"
    )


@pytest.mark.parametrize("factor", DESIGN)
def test_placebo_premium_is_not_recovered(factor):
    t = _fm_tstats(premia={factor: 0.0})[factor]
    assert abs(t) < 2.0, (
        f"{factor}: placebo t={t:.2f} — an unpriced characteristic must not "
        f"show up as a premium"
    )
