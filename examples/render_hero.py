"""Render the README hero image: a factor tearsheet.

Run:  python examples/render_hero.py   ->  writes docs/demo.png

Top panel: cumulative long-short return of each factor's quintile sort.
Bottom panel: rolling 60-day information coefficient (IC) of momentum.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fz import (  # noqa: E402
    cumulative,
    long_short_return,
    low_vol,
    make_universe,
    momentum,
    quality_roe,
    quintile_sort_returns,
    rolling_ic,
    value_btm,
)


def main() -> None:
    u = make_universe(n_stocks=300, n_days=1000, seed=0)
    fwd = np.full_like(u.returns, np.nan)
    fwd[:-1] = u.returns[1:]

    factors = {
        "momentum": momentum(u, lookback=252, skip=21),
        "value": value_btm(u),
        "quality": quality_roe(u),
        "low_vol": low_vol(u, window=60),
    }

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[2, 1], sharex=True
    )

    for name, scores in factors.items():
        ls = long_short_return(quintile_sort_returns(scores, fwd, n_quantiles=5))
        curve = cumulative(np.nan_to_num(ls))
        ax1.plot(curve, lw=1.2, label=name)
    ax1.axhline(0.0, color="0.5", lw=0.6)
    ax1.set_ylabel("cumulative long-short return")
    ax1.set_title("factor-zoo — long-short factor performance on a synthetic universe")
    ax1.legend(loc="upper left", ncol=4, fontsize=9)

    ric = rolling_ic(factors["momentum"], fwd, window=60)
    ax2.plot(ric, color="#7b3294", lw=0.9)
    ax2.axhline(0.0, color="0.5", lw=0.6)
    ax2.set_ylabel("momentum 60d IC")
    ax2.set_xlabel("trading day")

    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "demo.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
