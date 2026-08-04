"""Fama-MacBeth on the four-factor design, then the same fit at a 21-day
holding period to show what overlapping returns do to the t-stat.

Run:  python3 examples/fama_macbeth.py
"""

import numpy as np

from fz.crossect import fama_macbeth, forward_returns
from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.universe import make_universe

NAMES = ["intercept", "mom", "val", "qual", "lvol"]


def main() -> None:
    u = make_universe(n_stocks=300, n_days=1500, seed=0)
    scores = [
        momentum(u, lookback=252, skip=21),
        value_btm(u),
        quality_roe(u),
        low_vol(u, window=60),
    ]

    res = fama_macbeth(scores, forward_returns(u.returns), add_intercept=True)
    print(f"one-day-ahead returns, {res.n_periods} cross-sections\n")
    print(f"{'factor':<10} {'coef':>12} {'s.e.':>12} {'t-stat':>9}")
    for name, c, se, t in zip(NAMES, res.coefficients, res.std_errors,
                              res.t_stats, strict=True):
        print(f"{name:<10} {c:>12.6f} {se:>12.6f} {t:>9.2f}")
    print(f"\nmean daily R^2: {np.nanmean(res.daily_r2):.4f}")

    # Hold for a month instead of a day. Consecutive cross-sections now share
    # 20 of their 21 days, so the daily slopes are autocorrelated and the iid
    # standard error counts the same evidence over and over.
    h = 21
    fwd_h = forward_returns(u.returns, horizon=h)
    naive = fama_macbeth(scores, fwd_h, hac_lags=0)
    hac = fama_macbeth(scores, fwd_h, hac_lags=h - 1)

    print(f"\n{h}-day overlapping returns — same estimates, honest errors\n")
    print(f"{'factor':<10} {'coef':>12} {'t (iid)':>9} {'t (NW)':>9} {'overstated':>12}")
    for name, c, t0, t1 in zip(NAMES, hac.coefficients, naive.t_stats,
                               hac.t_stats, strict=True):
        print(f"{name:<10} {c:>12.6f} {t0:>9.2f} {t1:>9.2f} {t0 / t1:>11.2f}x")


if __name__ == "__main__":
    main()
