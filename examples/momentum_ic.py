"""Information coefficient of every characteristic, one day ahead.

The IC is the per-date Spearman correlation between a score and the return
that follows it. Daily ICs are tiny and noisy — 0.03 is a good factor — so
the number that matters is the information ratio, mean IC over its own
standard deviation, which says how reliably the sign repeats.

Only four of the six characteristics are in the generator's pricing design.
Nothing here pays for short reversal, and the size score is contaminated by
accumulated price — market cap is shares times price, so -log(mcap) is part
characteristic and part past return, which drags it to the wrong sign. Both
are printed anyway: it is worth seeing what no signal, and what a leaky
characteristic, look like beside a working one.

Run:  python3 examples/momentum_ic.py
"""

import argparse

import numpy as np

from fz.crossect import forward_returns, rank_information_coefficient, rolling_ic
from fz.factors import (
    low_vol,
    momentum,
    quality_roe,
    short_reversal,
    size_factor,
    value_btm,
)
from fz.universe import make_universe

FACTORS = {
    "momentum": lambda u: momentum(u, lookback=252, skip=21),
    "value": value_btm,
    "quality": quality_roe,
    "low_vol": lambda u: low_vol(u, window=60),
    "size": size_factor,
    "short_reversal": lambda u: short_reversal(u, window=5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    u = make_universe(n_stocks=300, n_days=1500, seed=0)
    fwd = forward_returns(u.returns)

    print(f"{'factor':<15} {'mean IC':>9} {'IR':>8} {'% > 0':>8} "
          f"{f'{args.window}d IC range':>16}")
    for name, fn in FACTORS.items():
        s = fn(u)
        ic = rank_information_coefficient(s, fwd)
        roll = rolling_ic(s, fwd, window=args.window)
        valid = ic[~np.isnan(ic)]
        ir = valid.mean() / valid.std()
        print(f"{name:<15} {valid.mean():>+9.4f} {ir:>+8.3f} "
              f"{(valid > 0).mean():>7.1%} "
              f"{np.nanmin(roll):>+7.3f} {np.nanmax(roll):>+7.3f}")


if __name__ == "__main__":
    main()
