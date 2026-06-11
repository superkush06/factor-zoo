"""Run a multi-factor portfolio sort + long-short backtest.

Trains 5 factors on synthetic data, sorts the universe daily into
quintiles by each factor's score, and reports long-short Sharpe ratios.

Run:  PYTHONPATH=. python3 examples/factor_backtest.py
"""

import argparse

import numpy as np

from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.portfolio import (
    long_short_return,
    quintile_sort_returns,
    sharpe_annualised,
)
from fz.universe import make_universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stocks", type=int, default=300)
    ap.add_argument("--n-days", type=int, default=1500)
    args = ap.parse_args()

    u = make_universe(n_stocks=args.n_stocks, n_days=args.n_days, seed=0)
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]

    factor_fns = {
        "momentum": lambda u: momentum(u, lookback=252, skip=21),
        "value":    value_btm,
        "quality":  quality_roe,
        "low_vol":  lambda u: low_vol(u, window=60),
    }

    print(f"{'factor':<12} {'L-S Sharpe':>12} {'L-S annual ret':>16}")
    for name, fn in factor_fns.items():
        s = fn(u)
        qr = quintile_sort_returns(s, fwd, n_quantiles=5)
        ls = long_short_return(qr)
        ann = np.nanmean(ls) * 252
        sr = sharpe_annualised(ls)
        print(f"{name:<12} {sr:>12.3f} {ann:>16.4f}")


if __name__ == "__main__":
    main()
