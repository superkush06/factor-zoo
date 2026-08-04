"""Daily quintile sort of every factor, reported as a long-short book.

Rank the universe by each factor's score, hold the top quintile against the
bottom for one day, repeat. If the characteristic carries information the
spread is positive and the three buckets in between line up in order.

Annualisation here is arithmetic: a bucket's mean daily forward return times
252. `spread` is Q5 - Q1 in those units, which by linearity is the long-short
book's own mean daily return times 252. Compounding that same daily mean over
252 days instead -- (1 + mean)**252 - 1 -- gives a larger number, and that is
the convention `docs/validation.md` row 1 quotes for momentum. Same book, same
daily mean, two conventions; `annualised_compounded` converts between them.

Run:  python3 examples/factor_backtest.py
"""

import argparse
from dataclasses import dataclass

import numpy as np

from fz.crossect import forward_returns
from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.portfolio import (
    long_short_return,
    quintile_sort_returns,
    sharpe_annualised,
)
from fz.universe import make_universe

FACTORS = {
    "momentum": lambda u: momentum(u, lookback=252, skip=21),
    "value": value_btm,
    "quality": quality_roe,
    "low_vol": lambda u: low_vol(u, window=60),
}

HEADER = (f"{'factor':<10} {'Q1 %/yr':>9} {'Q5 %/yr':>9} "
          f"{'spread':>9} {'Sharpe':>8} {'monotone':>9}")


@dataclass(frozen=True)
class SortRow:
    """One printed row; every percentage is an arithmetic annualisation."""
    name: str
    q1: float
    q5: float
    spread: float
    sharpe: float
    monotone: bool


def annualised_compounded(arithmetic_pct: float, freq: int = 252) -> float:
    """Compounded annual return of a book quoted at `arithmetic_pct` %/yr.

    The columns below are a daily mean times 252; compounding that same daily
    mean gives (1 + pct/100/252)**252 - 1, which is the convention row 1 of
    docs/validation.md uses. Neither is more correct, and the gap between them
    is not extra return.
    """
    return ((1.0 + arithmetic_pct / 100.0 / freq) ** freq - 1.0) * 100.0


def sort_table(n_stocks: int = 300, n_days: int = 1500,
               seed: int = 0) -> list[SortRow]:
    """The table `main` prints, as data, so a test can pin it."""
    u = make_universe(n_stocks=n_stocks, n_days=n_days, seed=seed)
    fwd = forward_returns(u.returns)
    rows = []
    for name, fn in FACTORS.items():
        q = quintile_sort_returns(fn(u), fwd, n_quantiles=5)
        rungs = np.nanmean(q, axis=0) * 252 * 100
        ls = long_short_return(q)
        rows.append(SortRow(name, float(rungs[0]), float(rungs[-1]),
                            float(rungs[-1] - rungs[0]),
                            sharpe_annualised(ls),
                            bool(np.all(np.diff(rungs) > 0))))
    return rows


def format_row(r: SortRow) -> str:
    return (f"{r.name:<10} {r.q1:>9.2f} {r.q5:>9.2f} "
            f"{r.spread:>9.2f} {r.sharpe:>8.2f} "
            f"{'yes' if r.monotone else 'no':>9}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stocks", type=int, default=300)
    ap.add_argument("--n-days", type=int, default=1500)
    args = ap.parse_args()

    print(HEADER)
    for row in sort_table(n_stocks=args.n_stocks, n_days=args.n_days):
        print(format_row(row))


if __name__ == "__main__":
    main()
