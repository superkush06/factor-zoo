"""Invariants that hold for every valid input, not just the fixtures.

A fixture test says "on this panel, with this seed, the answer was 7". A
property test says "whatever panel you hand it, this identity holds" — which
is the only kind of statement that survives a refactor. Each test below draws
its own random inputs from a seeded generator and asserts something that is
true by construction, by algebra, or by the definition of the estimator.

Draw counts are chosen so a violation on a randomly-shaped input would be
caught, not so the suite takes a minute: these all run in a couple of
seconds.
"""

import dataclasses
import math

import numpy as np
import pytest

from fz import (
    average_ranks,
    cumulative,
    fama_macbeth,
    forward_returns,
    long_short_return,
    low_vol,
    make_universe,
    momentum,
    newey_west_var,
    quality_roe,
    quintile_sort_returns,
    rank_information_coefficient,
    sharpe_annualised,
    short_reversal,
    size_factor,
    value_btm,
)

SEED = 20260727
FACTORS = {
    "momentum": lambda u: momentum(u, lookback=60, skip=5),
    "short_reversal": lambda u: short_reversal(u, window=5),
    "value": value_btm,
    "size": size_factor,
    "quality": quality_roe,
    "low_vol": lambda u: low_vol(u, window=20),
}


def _truncate(u, T):
    return dataclasses.replace(
        u, prices=u.prices[:T], returns=u.returns[:T],
        market_cap=u.market_cap[:T], book_value=u.book_value[:T],
        earnings=u.earnings[:T], dates=u.dates[:T])


# --------------------------------------------------------------------------
# Ranks
# --------------------------------------------------------------------------
def test_ranks_conserve_their_total():
    """Sum of average ranks is n(n-1)/2 for any input.

    Ties redistribute rank mass among the tied names but cannot create or
    destroy it, so the total is fixed by n alone. This is the invariant that
    breaks first if tie handling is written as anything other than "share the
    mean of the ordinal ranks you span".
    """
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        n = int(rng.integers(2, 120))
        # deliberately coarse, so ties are common
        x = np.round(rng.standard_normal(n) * rng.choice([0.1, 1.0, 10.0]), 1)
        r = average_ranks(x)
        assert r.sum() == pytest.approx(n * (n - 1) / 2.0)
        assert r.min() >= 0.0 and r.max() <= n - 1


def test_ranks_are_equivariant_under_permutation_and_monotone_maps():
    """Ranks follow the data, not its storage order, and any strictly
    increasing transform of the values leaves them unchanged — that is what
    makes a rank statistic a rank statistic."""
    rng = np.random.default_rng(SEED + 1)
    for _ in range(200):
        n = int(rng.integers(3, 80))
        x = np.round(rng.standard_normal(n), 1)
        perm = rng.permutation(n)
        np.testing.assert_allclose(average_ranks(x[perm]), average_ranks(x)[perm])
        for f in (np.exp, lambda v: 3.0 * v - 7.0, lambda v: v ** 3):
            np.testing.assert_allclose(average_ranks(f(x)), average_ranks(x))


# --------------------------------------------------------------------------
# Characteristics
# --------------------------------------------------------------------------
def test_every_score_row_is_a_unit_z_score():
    """Each date's cross-section is standardised on its own: mean 0, sd 1.

    Not "approximately, on average" — every row that is not warm-up NaN. A
    factor that pools its moments across dates fails this the moment the
    panel is not stationary.
    """
    rng = np.random.default_rng(SEED + 2)
    for _ in range(12):
        u = make_universe(n_stocks=int(rng.integers(40, 160)),
                          n_days=int(rng.integers(120, 260)),
                          seed=int(rng.integers(0, 10_000)))
        for name, fn in FACTORS.items():
            s = fn(u)
            live = ~np.all(np.isnan(s), axis=1)
            assert live.any(), name
            rows = s[live]
            assert np.abs(np.nanmean(rows, axis=1)).max() < 1e-9, name
            assert np.abs(np.nanstd(rows, axis=1) - 1.0).max() < 1e-6, name


def test_momentum_ignores_the_units_a_price_is_quoted_in():
    """Momentum is a difference of logs, so multiplying any stock's whole
    price series by a positive constant — a currency change, a share split
    applied retroactively — must leave the score bit-identical."""
    rng = np.random.default_rng(SEED + 3)
    for _ in range(15):
        u = make_universe(n_stocks=60, n_days=200,
                          seed=int(rng.integers(0, 10_000)))
        scale = np.exp(rng.normal(0.0, 1.5, size=u.prices.shape[1]))
        v = dataclasses.replace(u, prices=u.prices * scale[None, :])
        np.testing.assert_allclose(momentum(v, lookback=60, skip=5),
                                   momentum(u, lookback=60, skip=5),
                                   equal_nan=True, atol=1e-12)


def test_scoring_never_reorders_a_cross_section():
    """Winsorise-then-z-score is a monotone map of the raw characteristic, so
    it may merge names in the clipped tails but may never swap two of them.

    This is what licenses using the score for a rank-based sort: the ordering
    the portfolio trades is the ordering of the underlying quantity.
    """
    rng = np.random.default_rng(SEED + 4)
    for _ in range(10):
        u = make_universe(n_stocks=80, n_days=90,
                          seed=int(rng.integers(0, 10_000)))
        raw = np.log(u.book_value / u.market_cap)
        score = value_btm(u)
        for t in rng.integers(0, 90, size=8):
            r, s = raw[t], score[t]
            d_raw = np.sign(r[:, None] - r[None, :])
            d_s = np.sign(s[:, None] - s[None, :])
            assert not np.any(d_raw * d_s < 0)      # no inversions anywhere


def test_scores_never_look_at_a_day_that_has_not_happened(random_cuts=8):
    """f(panel[:T]) == f(panel)[:T] for every factor and every cut T.

    tests/test_no_lookahead.py pins one cut; this draws them at random, which
    is what catches a window that only leaks near a boundary.
    """
    rng = np.random.default_rng(SEED + 5)
    u = make_universe(n_stocks=70, n_days=320, seed=5)
    for T in rng.integers(80, 320, size=random_cuts):
        part = _truncate(u, int(T))
        for name, fn in FACTORS.items():
            np.testing.assert_allclose(fn(part), fn(u)[:T], equal_nan=True,
                                       atol=1e-12, err_msg=f"{name} at T={T}")


# --------------------------------------------------------------------------
# Sorts and books
# --------------------------------------------------------------------------
def test_quintile_buckets_partition_the_cross_section():
    """With n divisible by 5 and no ties, the buckets are equal-sized and
    disjoint, so the unweighted mean of the five bucket means is exactly the
    cross-sectional mean return. Nothing may be double-counted or dropped."""
    rng = np.random.default_rng(SEED + 6)
    for _ in range(120):
        n_days, n_stocks = int(rng.integers(2, 12)), int(rng.integers(2, 40)) * 5
        s = rng.standard_normal((n_days, n_stocks))
        r = rng.standard_normal((n_days, n_stocks))
        q = quintile_sort_returns(s, r, 5)
        np.testing.assert_allclose(q.mean(axis=1), r.mean(axis=1), atol=1e-12)


def test_flipping_a_signal_flips_the_book():
    """Sorting on -s is sorting on s upside down: bucket k of one is bucket
    (4-k) of the other, and the long-short return changes sign. A sort that
    is not antisymmetric is quietly favouring one side."""
    rng = np.random.default_rng(SEED + 7)
    for _ in range(120):
        n_days, n_stocks = int(rng.integers(2, 10)), int(rng.integers(2, 30)) * 5
        s = rng.standard_normal((n_days, n_stocks))
        r = rng.standard_normal((n_days, n_stocks))
        q, q_flip = quintile_sort_returns(s, r, 5), quintile_sort_returns(-s, r, 5)
        np.testing.assert_allclose(q_flip, q[:, ::-1], atol=1e-12)
        np.testing.assert_allclose(long_short_return(q_flip),
                                   -long_short_return(q), atol=1e-12)


def test_cumulative_growth_telescopes():
    """cumulative(r)[-1] == prod(1 + r) - 1, with NaN days held flat.

    A compounding routine that drifts from the product it is meant to be is
    the single easiest way to publish an equity curve nobody can reproduce.
    """
    rng = np.random.default_rng(SEED + 8)
    for _ in range(150):
        n = int(rng.integers(1, 200))
        r = rng.normal(0.0, 0.02, size=n)
        r[rng.random(n) < 0.1] = np.nan
        assert cumulative(r)[-1] == pytest.approx(
            float(np.prod(1.0 + r[~np.isnan(r)])) - 1.0, rel=1e-12)


def test_sharpe_is_scale_free_and_sign_flipping():
    """Sharpe(c r) == Sharpe(r) for c > 0 and Sharpe(-r) == -Sharpe(r):
    leverage does not create a better strategy, and shorting a book negates
    its ratio exactly."""
    rng = np.random.default_rng(SEED + 9)
    for _ in range(150):
        r = rng.normal(rng.normal(0, 0.001), 0.01, size=int(rng.integers(20, 500)))
        c = float(rng.uniform(0.01, 100.0))
        base = sharpe_annualised(r)
        assert sharpe_annualised(c * r) == pytest.approx(base, rel=1e-10)
        assert sharpe_annualised(-r) == pytest.approx(-base, rel=1e-10)


# --------------------------------------------------------------------------
# Forward returns and the estimator
# --------------------------------------------------------------------------
def test_multi_day_forward_returns_are_the_product_of_daily_ones():
    """1 + fwd_h[t] == prod_k (1 + fwd_1[t+k]). Holding for h days is holding
    for one day, h times; if the two disagree the horizon machinery is
    silently mixing simple and log returns."""
    rng = np.random.default_rng(SEED + 10)
    for _ in range(40):
        n_days, n_stocks = int(rng.integers(20, 60)), int(rng.integers(1, 6))
        r = rng.normal(0.0, 0.02, size=(n_days, n_stocks))
        r[0] = np.nan
        f1 = forward_returns(r, 1)
        h = int(rng.integers(2, 6))
        fh = forward_returns(r, h)
        chained = np.prod([1.0 + f1[t: n_days - h + t] for t in range(h)], axis=0) - 1.0
        np.testing.assert_allclose(fh[: n_days - h], chained, rtol=1e-12)


def _truncated_long_run_var(x: np.ndarray, lags: int) -> float:
    """The same estimator with every kernel weight set to 1.

    This is the truncated (rectangular) kernel Newey and West's triangular one
    replaces. It is *not* positive semi-definite, which is the whole point of
    using it as a foil below.
    """
    d = x - x.mean()
    T = d.shape[0]
    s = float(d @ d) / (T - 1)
    for lag in range(1, min(lags, T - 1) + 1):
        s += 2.0 * float(d[lag:] @ d[:-lag]) / (T - 1)
    return s


def test_newey_west_output_is_a_variance():
    """Shift-invariant, scales with c^2, never negative.

    Non-negativity is asserted here as an output property only —
    `newey_west_var` ends in `max(s, 0.0)`, so this assertion would hold under
    any kernel whatsoever. Crediting it to the Bartlett weights is the job of
    the next test.
    """
    rng = np.random.default_rng(SEED + 11)
    for _ in range(300):
        n = int(rng.integers(5, 400))
        kind = rng.integers(0, 3)
        x = rng.standard_normal(n)
        if kind == 1:                       # strongly persistent
            x = np.cumsum(x) / math.sqrt(n)
        elif kind == 2:                     # alternating, negative autocov
            x = x * (-1.0) ** np.arange(n)
        lags = int(rng.integers(0, min(n, 40)))
        s = newey_west_var(x, lags)
        assert s >= 0.0
        c = float(rng.uniform(-5, 5))
        assert newey_west_var(c * x, lags) == pytest.approx(c * c * s, rel=1e-9)
        assert newey_west_var(x + rng.normal(0, 10), lags) == pytest.approx(s, rel=1e-9)


def test_bartlett_weights_and_not_the_clamp_keep_the_variance_positive():
    """The triangular weights are what make the long-run variance non-negative.

    The kernel's Fourier transform is the Fejer kernel, which is non-negative,
    so the weighted autocovariance sum is a spectral average of a periodogram
    and cannot go below zero. `newey_west_var` also clamps at zero, so a bare
    `s >= 0` assertion proves nothing about the weights. This test removes the
    clamp from the argument twice over.

    The foil is an over-differenced series, x_t = e_t - e_{t-1}: gamma_0 =
    2 sigma^2, gamma_1 = -sigma^2, and nothing beyond. Summing those
    *unweighted* gives 0 in population, so on a finite sample the truncated
    kernel lands below zero about half the time. The Bartlett sum is
    2 sigma^2 (1 - L/(L+1)) = gamma_0 / (L+1), comfortably positive.

    So: on every draw where the truncated kernel is negative, the shipped
    estimator must be *strictly* positive — a clamped zero fails that — and on
    a long series it must hit gamma_0/(L+1), which pins the weights themselves
    rather than just their sign.
    """
    rng = np.random.default_rng(SEED + 11)
    negatives = 0
    for _ in range(400):
        n = int(rng.integers(40, 300))
        e = rng.standard_normal(n + 1)
        x = e[1:] - e[:-1]
        lags = int(rng.integers(1, 9))
        if _truncated_long_run_var(x, lags) < 0.0:
            negatives += 1
            assert newey_west_var(x, lags) > 0.0, (
                "the Bartlett kernel must stay positive where the truncated "
                "kernel goes negative; a zero here is the clamp talking"
            )
    assert negatives >= 50, (
        f"only {negatives} draws exercised a negative truncated kernel; the "
        "foil is not doing its job"
    )

    e = rng.standard_normal(400_001)
    x = e[1:] - e[:-1]
    gamma_0 = float(np.var(x, ddof=1))
    for lags in (1, 2, 5, 12, 20):
        assert newey_west_var(x, lags) == pytest.approx(gamma_0 / (lags + 1), rel=0.02)


def test_fama_macbeth_is_exact_when_the_model_is_exact():
    """If forward returns are literally X b with no noise, the estimator must
    return b — every day, and therefore on average. Cross-sectional OLS is a
    projection; feeding it a point already in the column space has to return
    that point."""
    rng = np.random.default_rng(SEED + 12)
    for _ in range(60):
        n_days, n_stocks = int(rng.integers(5, 40)), int(rng.integers(30, 120))
        k = int(rng.integers(1, 4))
        scores = [rng.standard_normal((n_days, n_stocks)) for _ in range(k)]
        b = rng.normal(0.0, 0.01, size=k + 1)
        fwd = b[0] + sum(bi * s for bi, s in zip(b[1:], scores, strict=True))
        res = fama_macbeth(scores, fwd)
        np.testing.assert_allclose(res.coefficients, b, atol=1e-8)
        assert np.allclose(res.daily_r2[~np.isnan(res.daily_r2)], 1.0, atol=1e-8)


def test_fama_macbeth_is_linear_in_the_premium_it_is_handed():
    """Add c*z to the forward returns and the coefficient on z rises by
    exactly c, whatever the noise. OLS is linear in y, so the estimation error
    is the same draw in both fits: this is an identity, not a limit."""
    rng = np.random.default_rng(SEED + 13)
    for _ in range(40):
        n_days, n_stocks = int(rng.integers(40, 120)), int(rng.integers(40, 200))
        z = rng.standard_normal((n_days, n_stocks))
        noise = rng.normal(0.0, 0.02, size=(n_days, n_stocks))
        c = float(rng.uniform(-0.01, 0.01))
        a = fama_macbeth([z], noise)
        b = fama_macbeth([z], noise + c * z)
        assert b.coefficients[1] - a.coefficients[1] == pytest.approx(c, abs=1e-12)
        np.testing.assert_allclose(b.std_errors, a.std_errors, rtol=1e-9)


def test_fama_macbeth_rescales_with_its_regressor():
    """Multiply a score column by c and its premium divides by c: the fitted
    contribution to expected return is unchanged. A coefficient that does not
    move this way is not the slope of anything."""
    rng = np.random.default_rng(SEED + 14)
    for _ in range(40):
        n_days, n_stocks = 60, 100
        s1 = rng.standard_normal((n_days, n_stocks))
        s2 = rng.standard_normal((n_days, n_stocks))
        fwd = rng.normal(0.0, 0.02, size=(n_days, n_stocks))
        c = float(rng.uniform(0.1, 10.0))
        base = fama_macbeth([s1, s2], fwd)
        scaled = fama_macbeth([c * s1, s2], fwd)
        assert scaled.coefficients[1] == pytest.approx(base.coefficients[1] / c,
                                                       rel=1e-8)
        assert scaled.coefficients[2] == pytest.approx(base.coefficients[2], rel=1e-8)
        assert scaled.t_stats[1] == pytest.approx(base.t_stats[1], rel=1e-6)


def test_information_coefficient_is_a_correlation():
    """Bounded in [-1, 1], antisymmetric in the score, and invariant to any
    strictly increasing transform of the forward returns — because Spearman
    only ever sees ranks."""
    rng = np.random.default_rng(SEED + 15)
    for _ in range(40):
        n_days, n_stocks = int(rng.integers(3, 15)), int(rng.integers(25, 150))
        s = rng.standard_normal((n_days, n_stocks))
        r = rng.standard_normal((n_days, n_stocks))
        ic = rank_information_coefficient(s, r)
        assert np.all(np.isnan(ic) | ((ic >= -1.0) & (ic <= 1.0)))
        np.testing.assert_allclose(rank_information_coefficient(-s, r), -ic,
                                   equal_nan=True, atol=1e-12)
        for f in (np.exp, lambda v: 2.0 * v + 5.0, lambda v: v ** 3):
            np.testing.assert_allclose(rank_information_coefficient(s, f(r)), ic,
                                       equal_nan=True, atol=1e-12)


# --------------------------------------------------------------------------
# The generator's own contract
# --------------------------------------------------------------------------
def test_the_universe_contract_holds_for_any_seed_and_shape():
    """prices[0] == 100, returns[0] is NaN, and prices compound returns
    exactly. Everything downstream — the loaders, `cumulative`, every
    price-driven factor — assumes exactly one returns convention."""
    rng = np.random.default_rng(SEED + 16)
    for _ in range(20):
        u = make_universe(n_stocks=int(rng.integers(10, 90)),
                          n_days=int(rng.integers(30, 200)),
                          seed=int(rng.integers(0, 100_000)))
        assert np.isnan(u.returns[0]).all()
        np.testing.assert_allclose(u.prices[0], 100.0)
        # atol covers the handful of near-zero returns where a 1e-16 rounding
        # difference is a large *relative* one
        np.testing.assert_allclose(u.prices[1:] / u.prices[:-1] - 1.0,
                                   u.returns[1:], rtol=1e-10, atol=1e-14)
        assert np.isfinite(u.returns[1:]).all()
        assert (u.prices > 0).all()


def test_zeroing_a_premium_only_touches_that_premium():
    """A placebo universe is the same draw with one term deleted. Because
    `premia` rescales already-drawn numbers, the difference between the loaded
    and placebo return panels must lie entirely along that factor's exposure
    vector — a rank-one change, not a fresh sample."""
    rng = np.random.default_rng(SEED + 17)
    for factor in ("value", "size", "quality", "low_vol"):
        seed = int(rng.integers(0, 10_000))
        base = make_universe(n_stocks=60, n_days=120, seed=seed)
        placebo = make_universe(n_stocks=60, n_days=120, seed=seed,
                                premia={factor: 0.0})
        d = np.log1p(base.returns[1:]) - np.log1p(placebo.returns[1:])
        # rank one: every date's difference is the same vector, scaled
        sv = np.linalg.svd(d, compute_uv=False)
        assert sv[0] > 1e-12
        assert sv[1] / sv[0] < 1e-9, factor
