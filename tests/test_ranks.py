"""Rank tie-handling tests: winsorised tails tie by construction."""

import numpy as np

from fz.crossect import average_ranks, rank_information_coefficient
from fz.portfolio import long_short_return, quintile_sort_returns


def test_tied_values_share_average_rank():
    x = np.array([7.0, 1.0, 7.0, 3.0, 1.0])
    r = average_ranks(x)
    # the two 1.0s span ordinal ranks 0,1 -> both 0.5; the 7.0s span 3,4 -> 3.5
    np.testing.assert_allclose(r, [3.5, 0.5, 3.5, 2.0, 0.5])


def test_ranks_are_order_independent():
    rng = np.random.default_rng(0)
    x = np.repeat([1.0, 2.0, 2.0, 5.0], 5)  # heavy ties
    perm = rng.permutation(x.size)
    r = average_ranks(x)
    r_perm = average_ranks(x[perm])
    np.testing.assert_allclose(r_perm, r[perm])


def test_winsorised_tails_get_symmetric_ranks():
    # A clipped vector: three values tied at each tail, like a 1%/99%
    # winsorised cross-section.
    x = np.array([-2.0, -2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    r = average_ranks(x)
    assert r[0] == r[1] == r[2] == 1.0
    assert r[-1] == r[-2] == r[-3] == 7.0
    # symmetric: distance from the ends matches
    np.testing.assert_allclose(r[:3], (len(x) - 1) - r[-3:])


def test_quintile_membership_ignores_ticker_order():
    """With ties straddling a quantile boundary, the long-short return must
    not depend on column (ticker) order."""
    scores = np.array([[1.0, 1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    fwd = np.array([[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]])
    perm = np.array([2, 0, 1, 5, 4, 3, 9, 8, 7, 6])
    ls_a = long_short_return(quintile_sort_returns(scores, fwd, n_quantiles=5))
    ls_b = long_short_return(quintile_sort_returns(scores[:, perm], fwd[:, perm],
                                                   n_quantiles=5))
    np.testing.assert_allclose(ls_a, ls_b)


def test_spearman_ic_ignores_ticker_order():
    rng = np.random.default_rng(1)
    scores = np.round(rng.standard_normal((5, 60)), 1)  # rounding forces ties
    fwd = rng.standard_normal((5, 60))
    perm = rng.permutation(60)
    ic_a = rank_information_coefficient(scores, fwd)
    ic_b = rank_information_coefficient(scores[:, perm], fwd[:, perm])
    np.testing.assert_allclose(ic_a, ic_b, atol=1e-12)
