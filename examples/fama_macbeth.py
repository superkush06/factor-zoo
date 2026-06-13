"""Run a Fama-MacBeth multi-factor regression and report point estimates.

Run:  PYTHONPATH=. python3 examples/fama_macbeth.py
"""

import numpy as np

from fz.crossect import fama_macbeth
from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.universe import make_universe


def main() -> None:
    u = make_universe(n_stocks=300, n_days=1500, seed=0)
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]

    scores = [
        ("mom",   momentum(u, lookback=252, skip=21)),
        ("val",   value_btm(u)),
        ("qual",  quality_roe(u)),
        ("lvol",  low_vol(u, window=60)),
    ]
    res = fama_macbeth([s for _, s in scores], fwd, add_intercept=True)

    print(f"{'factor':<10} {'coef':>14} {'t-stat':>10}")
    print(f"{'intercept':<10} {res.coefficients[0]:>14.6f} {res.t_stats[0]:>10.2f}")
    for i, (name, _) in enumerate(scores, start=1):
        print(f"{name:<10} {res.coefficients[i]:>14.6f} {res.t_stats[i]:>10.2f}")
    print(f"\nMean daily R^2 across days: {np.nanmean(res.daily_r2):.4f}")


if __name__ == "__main__":
    main()
