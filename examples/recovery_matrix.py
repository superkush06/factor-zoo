"""The claim the whole library rests on, printed as a table.

For each factor, run the full four-factor Fama-MacBeth twice: once on the
universe as generated, and once on a placebo universe in which that single
premium has been set to zero. The characteristic still exists in the placebo
-- the same stocks are still cheap, still quiet, still trending -- it simply
is not paid for. A pipeline that is wired correctly finds the premium in the
first column and nothing in the second.

Because `premia` only rescales already-drawn random numbers, the two runs
share every random draw: this is one universe with a term removed, not two
samples.

Run:  python3 examples/recovery_matrix.py
"""

import numpy as np

from fz.crossect import fama_macbeth, forward_returns
from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.universe import make_universe

DESIGN = ["momentum", "value", "quality", "low_vol"]


def _t_stats(**kwargs) -> np.ndarray:
    u = make_universe(n_stocks=300, n_days=1500, seed=0, **kwargs)
    scores = [
        momentum(u, lookback=252, skip=21),
        value_btm(u),
        quality_roe(u),
        low_vol(u, window=60),
    ]
    return fama_macbeth(scores, forward_returns(u.returns)).t_stats[1:]


def main() -> None:
    loaded = _t_stats()
    print(f"{'factor':<10} {'t loaded':>10} {'t placebo':>11} {'verdict':>9}")
    for i, name in enumerate(DESIGN):
        placebo = _t_stats(premia={name: 0.0})[i]
        ok = loaded[i] > 2.0 and abs(placebo) < 2.0
        print(f"{name:<10} {loaded[i]:>10.2f} {placebo:>11.2f} "
              f"{'pass' if ok else 'FAIL':>9}")


if __name__ == "__main__":
    main()
