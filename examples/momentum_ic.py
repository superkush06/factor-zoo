"""Compute and plot rolling factor IC over time."""

import argparse

import numpy as np

from fz.crossect import rank_information_coefficient
from fz.factors import momentum
from fz.universe import make_universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60)
    ap.parse_args()

    u = make_universe(n_stocks=300, n_days=1500, seed=0)
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]
    s = momentum(u, lookback=252, skip=21)

    ic = rank_information_coefficient(s, fwd)
    valid = ~np.isnan(ic)
    print(f"Mean IC:        {np.nanmean(ic):+.4f}")
    print(f"IR (mean/std):  {np.nanmean(ic) / (np.nanstd(ic) + 1e-9):+.4f}")
    print(f"%-positive IC:  {(ic[valid] > 0).mean():.1%}")


if __name__ == "__main__":
    main()
