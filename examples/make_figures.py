"""Regenerate every figure in the README from the shipped code.

    python3 examples/make_figures.py            # every figure
    python3 examples/make_figures.py recovery   # just one

Four claims, four pictures:

  docs/recovery.png   the sorts are monotone, they compound, and the
                      regression reads back the premium that was written
                      into the universe -- and reads back nothing when the
                      premium is switched off.
  docs/overlap.png    what overlapping holding periods do to a t-stat, and
                      what a Newey-West correction does about it.
  docs/lookahead.png  where a panel-wide winsorisation leaks the future
                      into a point-in-time score, and how much.
  docs/calibration.png  plant a premium of known size and ask for it back:
                      the estimate against the truth, and the distribution of
                      the errors against the interval that claims to cover them.

Requires matplotlib (`pip install -e ".[plot]"`); nothing in `fz` does.

The font family and the PNG metadata are pinned, which is what makes a re-run
reproduce the committed images in `docs/` byte for byte on the matplotlib they
were drawn with (3.11). It does not survive a version change: on matplotlib
3.9 the same code draws the same content, but the layout rounds three of the
four canvases to 459 pixels tall instead of 460 and the text rasterises
differently. Do not diff bytes across builds -- regenerate and look at the
picture.
"""

from __future__ import annotations

import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

from fz import (  # noqa: E402
    cumulative,
    fama_macbeth,
    forward_returns,
    long_short_return,
    low_vol,
    make_universe,
    momentum,
    quality_roe,
    quintile_sort_returns,
    value_btm,
)
from fz.factors import _standardize  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from validate import planted_panel, premium_for_annual_spread  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
N_STOCKS, N_DAYS, SEED = 300, 1500, 0

COLOR = {
    "momentum": "#2b6cb0",
    "value": "#b7791f",
    "quality": "#2f855a",
    "low_vol": "#805ad5",
}
GREY = "#718096"
INK = "#1a202c"

STYLE = {
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cbd2d9",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#eceef1",
    "grid.linewidth": 0.8,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 9,
    "axes.labelsize": 9.5,
    "axes.labelcolor": "#3d4852",
    "text.color": INK,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#606f7b",
    "ytick.color": "#606f7b",
    "legend.frameon": False,
    "legend.fontsize": 9,
}


plt.rcParams.update(STYLE)


def _scores(u):
    return {
        "momentum": momentum(u, lookback=252, skip=21),
        "value": value_btm(u),
        "quality": quality_roe(u),
        "low_vol": low_vol(u, window=60),
    }


def _save(fig, name: str) -> None:
    DOCS.mkdir(exist_ok=True)
    out = DOCS / name
    # Software defaults to a matplotlib version string, which would make the
    # bytes depend on the machine that ran this rather than on the code.
    fig.savefig(out, dpi=100, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {out.relative_to(DOCS.parent)}")


# --------------------------------------------------------------------------
# 1. Recovery: the pipeline reads back what the generator wrote.
# --------------------------------------------------------------------------
def recovery() -> None:
    u = make_universe(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED)
    scores = _scores(u)
    fwd = forward_returns(u.returns)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.8),
                                        constrained_layout=True)

    # (a) quintile ladders
    for name, s in scores.items():
        rungs = np.nanmean(quintile_sort_returns(s, fwd, 5), axis=0) * 252 * 100
        ax1.plot(range(1, 6), rungs, marker="o", ms=5, lw=1.6,
                 color=COLOR[name], label=name)
    ax1.axhline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax1.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)])
    ax1.set_ylabel("mean forward return, %/yr")
    ax1.set_title("Sorts are monotone")
    ax1.legend(loc="upper left", ncol=2)

    # (b) compounded long-short, log scale: constant Sharpe is a straight line
    for name, s in scores.items():
        ls = long_short_return(quintile_sort_returns(s, fwd, 5))
        ax2.plot(1.0 + cumulative(np.nan_to_num(ls)), lw=1.4, color=COLOR[name])
    ax2.axhline(1.0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax2.set_yscale("log")
    ax2.set_yticks([1, 2, 3, 5, 7], ["1x", "2x", "3x", "5x", "7x"])
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("trading day")
    ax2.set_ylabel("growth of 1 unit in Q5 - Q1 (log scale)")
    ax2.set_title("They compound")

    # (c) estimated premium, loaded universe vs placebo
    names = list(scores)
    loaded = fama_macbeth(list(scores.values()), fwd)
    y = np.arange(len(names))[::-1]
    for i, name in enumerate(names):
        up = make_universe(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED,
                           premia={name: 0.0})
        rp = fama_macbeth(list(_scores(up).values()), forward_returns(up.returns))
        ax3.errorbar(rp.coefficients[i + 1] * 1e4, y[i] - 0.16,
                     xerr=2 * rp.std_errors[i + 1] * 1e4, fmt="o", ms=6,
                     mfc="white", mec=GREY, ecolor=GREY, elinewidth=1.2, capsize=3)
        ax3.errorbar(loaded.coefficients[i + 1] * 1e4, y[i] + 0.16,
                     xerr=2 * loaded.std_errors[i + 1] * 1e4, fmt="o", ms=6,
                     color=COLOR[name], ecolor=COLOR[name], elinewidth=1.6,
                     capsize=3)
    ax3.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 3)))
    ax3.set_yticks(y, names)
    ax3.set_ylim(-0.7, len(names) - 0.3)
    ax3.set_xlabel("Fama-MacBeth premium, bp/day (bars = ±2 s.e.)")
    ax3.set_title("And the regression agrees")
    ax3.plot([], [], "o", color=INK, label="premium loaded")
    ax3.plot([], [], "o", mfc="white", mec=GREY, ls="none", label="placebo (premium = 0)")
    ax3.legend(loc="lower right")

    fig.suptitle(
        f"factor-zoo — {N_STOCKS} synthetic names, {N_DAYS} days, daily quintile sorts",
        fontsize=12, fontweight="bold", x=0.007, ha="left")
    _save(fig, "recovery.png")


# --------------------------------------------------------------------------
# 2. Overlapping holding periods and Newey-West.
# --------------------------------------------------------------------------
def overlap(horizons=(1, 2, 3, 5, 10, 21, 42, 63)) -> None:
    u = make_universe(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED)
    scores = _scores(u)
    iid = {k: [] for k in scores}
    hac = {k: [] for k in scores}
    for h in horizons:
        fwd = forward_returns(u.returns, horizon=h)
        a = fama_macbeth(list(scores.values()), fwd, hac_lags=0)
        b = fama_macbeth(list(scores.values()), fwd, hac_lags=h - 1)
        for i, name in enumerate(scores):
            iid[name].append(a.t_stats[i + 1])
            hac[name].append(b.t_stats[i + 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6),
                                   constrained_layout=True)
    for name in scores:
        ax1.plot(horizons, iid[name], ls=(0, (4, 2)), lw=1.4, color=COLOR[name])
        ax1.plot(horizons, hac[name], lw=1.8, color=COLOR[name], label=name)
    ax1.set_xscale("log")
    ax1.set_xticks(list(horizons), [str(h) for h in horizons])
    ax1.set_xlabel("holding period, trading days")
    ax1.set_ylabel("Fama-MacBeth t-stat")
    ax1.set_title("Same evidence, counted once or counted h times")
    ax1.annotate("assuming independent days", (horizons[-2], iid["momentum"][-2]),
                 textcoords="offset points", xytext=(-8, 6), ha="right",
                 fontsize=9, color=GREY)
    ax1.annotate("Newey-West, lag h-1", (horizons[-2], hac["momentum"][-2]),
                 textcoords="offset points", xytext=(-8, 8), ha="right",
                 fontsize=9, color=INK)
    ax1.legend(loc="upper left", ncol=2)

    bartlett = [math.sqrt(1.0 + (h - 1) * (2 * h - 1) / (3.0 * h)) for h in horizons]
    ax2.plot(horizons, bartlett, color="#cbd2d9", lw=6, zorder=1,
             solid_capstyle="round")
    for name in scores:
        ratio = np.array(iid[name]) / np.array(hac[name])
        ax2.plot(horizons, ratio, marker="o", ms=4, lw=1.6, color=COLOR[name],
                 zorder=3)
    # What the inflation *should* be. An h-day overlap makes the slope series
    # an equally weighted MA(h-1); a truncated kernel would then recover the
    # full sqrt(h), but Bartlett's triangular weights discount the longest
    # lags and leave sqrt(1 + (h-1)(2h-1)/(3h)) -- about 0.82 sqrt(h).
    ax2.plot(horizons, np.sqrt(horizons), color="#b8c0c8", lw=1.2, ls=(0, (2, 3)),
             zorder=2)
    ax2.annotate(r"$\sqrt{h}$, if the kernel did not taper",
                 (horizons[-2], np.sqrt(horizons[-2])),
                 textcoords="offset points", xytext=(-10, 2), ha="right",
                 color="#95a0aa")
    ax2.annotate("Bartlett at $L=h-1$:\n"
                 r"$\sqrt{1 + (h-1)(2h-1)/3h}$",
                 (0.55, 0.10), xycoords="axes fraction", color="#5a6773")
    ax2.set_xscale("log")
    ax2.set_xticks(list(horizons), [str(h) for h in horizons])
    ax2.set_xlabel("holding period, trading days")
    ax2.set_ylabel("t-stat inflation, iid / Newey-West")
    ax2.set_title("How much the naive t-stat overstates")
    _save(fig, "overlap.png")


# --------------------------------------------------------------------------
# 3. Look-ahead: what a panel-wide winsorisation leaks.
# --------------------------------------------------------------------------
def _pooled_momentum(u, lookback=252, skip=21, low=0.01, high=0.99):
    """Momentum scored the wrong way: clip against quantiles of the *whole*
    panel, so day t borrows the distribution of every later day.

    It reuses the shipped z-scorer on purpose -- the winsorisation is then
    the only difference between this and `fz.factors.momentum`.
    """
    log_p = np.log(u.prices)
    raw = np.full_like(log_p, np.nan)
    for t in range(lookback, log_p.shape[0]):
        raw[t] = log_p[t - skip] - log_p[t - lookback]
    lo, hi = np.nanquantile(raw, low), np.nanquantile(raw, high)
    return _standardize(np.clip(raw, lo, hi))


def _truncate(u, T):
    import dataclasses
    return dataclasses.replace(
        u, prices=u.prices[:T], returns=u.returns[:T],
        market_cap=u.market_cap[:T], book_value=u.book_value[:T],
        earnings=u.earnings[:T], dates=u.dates[:T])


def lookahead(cut=500) -> None:
    u = make_universe(n_stocks=200, n_days=N_DAYS, seed=3)
    ut = _truncate(u, cut)

    z_full, z_cut = _pooled_momentum(u)[:cut], _pooled_momentum(ut)
    ok = ~np.isnan(z_full) & ~np.isnan(z_cut)
    drift = (z_cut - z_full)[ok]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.6),
                                   constrained_layout=True)
    ax1.axhline(0, color=INK, lw=2.0, zorder=3)
    ax1.scatter(z_full[ok], drift, s=3, alpha=0.18, lw=0,
                color=COLOR["momentum"], rasterized=True)
    ax1.set_xlabel("momentum z-score on day t")
    ax1.set_ylabel("score change when the future is removed")
    ax1.set_title(f"Truncating the panel at day {cut}")
    ax1.annotate(f"pooled winsorisation: up to {np.abs(drift).max():.2f} z-units",
                 (0.03, 0.94), xycoords="axes fraction", fontsize=9,
                 color=COLOR["momentum"])
    ax1.annotate("per-date winsorisation: exactly zero, everywhere",
                 (0.03, 0.06), xycoords="axes fraction", fontsize=9, color=INK)

    cuts = np.arange(300, N_DAYS, 100)
    pooled_max, perdate_max = [], []
    for c in cuts:
        uc = _truncate(u, int(c))
        pooled_max.append(np.nanmax(np.abs(_pooled_momentum(uc) - _pooled_momentum(u)[:c])))
        perdate_max.append(np.nanmax(np.abs(momentum(uc) - momentum(u)[:c])))
    ax2.plot(cuts, pooled_max, marker="o", ms=4, lw=1.6,
             color=COLOR["momentum"], label="quantiles pooled over the panel")
    ax2.plot(cuts, perdate_max, marker="o", ms=4, lw=2.0, color=INK,
             label="quantiles per date (shipped)")
    ax2.set_xlabel("day the panel is truncated at")
    ax2.set_ylabel("max |score change| over all earlier days")
    ax2.set_title("A score that changes when tomorrow arrives is not a score")
    ax2.legend(loc="upper right")
    _save(fig, "lookahead.png")

# --------------------------------------------------------------------------
# 4. Calibration: plant a premium of known size and ask for it back.
# --------------------------------------------------------------------------
def calibration(n_rep: int = 400) -> None:
    """Left: estimate against truth on panels where the characteristic is
    observed exactly, so the only thing between the planted number and the
    estimate is sampling error. Right: the standardised errors of `n_rep`
    independent panels against the normal the interval assumes."""
    grid = [0.0] + [premium_for_annual_spread(x) for x in (0.04, 0.1201, 0.25, 0.45)]
    est, err = [], []
    for i, lam in enumerate(grid):
        z, r = planted_panel(lam, seed=500 + i, n_days=1500, n_stocks=400)
        res = fama_macbeth([z], forward_returns(r))
        est.append((res.coefficients[1] * 1e4, 2 * res.std_errors[1] * 1e4))
    for s_ in range(n_rep):
        lam = premium_for_annual_spread(0.1201)
        z, r = planted_panel(lam, seed=2000 + s_, n_days=600, n_stocks=250)
        res = fama_macbeth([z], forward_returns(r))
        err.append((res.coefficients[1] - lam) / res.std_errors[1])
    err = np.array(err)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   constrained_layout=True)
    x = np.array(grid) * 1e4
    y = np.array([e for e, _ in est])
    e2 = np.array([s_ for _, s_ in est])
    lim = (-1.05, x.max() * 1.12)
    ax1.plot(lim, lim, color=GREY, lw=1.0, ls=(0, (4, 3)))
    ax1.errorbar(x, y, yerr=e2, fmt="o", ms=6, color=COLOR["momentum"],
                 ecolor=COLOR["momentum"], elinewidth=1.5, capsize=4)
    jt = premium_for_annual_spread(0.1201) * 1e4
    ax1.annotate("Jegadeesh-Titman scale\n(12.01%/yr)", (jt, jt),
                 textcoords="offset points", xytext=(10, -26), fontsize=9,
                 color=INK)
    ax1.annotate("nothing planted,\nnothing found", (0.0, y[0]),
                 textcoords="offset points", xytext=(12, -26), fontsize=9,
                 color=GREY)
    ax1.set_xlim(*lim); ax1.set_ylim(*lim)
    ax1.set_xlabel("premium written into the panel, bp/day")
    ax1.set_ylabel("Fama-MacBeth estimate, bp/day (bars = $\\pm$2 s.e.)")
    ax1.set_title("Ask for a known number back")

    ax2.hist(err, bins=28, density=True, color=COLOR["momentum"], alpha=0.35,
             edgecolor="white", linewidth=0.6)
    g = np.linspace(-4, 4, 400)
    ax2.plot(g, np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi), color=INK, lw=1.8)
    for c in (-1.959964, 1.959964):
        ax2.axvline(c, color=GREY, lw=1.0, ls=(0, (4, 3)))
    inside = float(np.mean(np.abs(err) < 1.959964))
    ax2.annotate(f"{inside:.1%} of {n_rep} panels inside $\\pm$1.96 s.e.\n"
                 f"(nominal 95.0%)", (0.03, 0.92), xycoords="axes fraction",
                 fontsize=9, va="top", color=INK)
    ax2.set_xlabel("(estimate $-$ truth) / reported standard error")
    ax2.set_ylabel("density")
    ax2.set_title("The interval means what it says")
    fig.suptitle("factor-zoo — calibration on panels with a planted premium",
                 fontsize=12, fontweight="bold", x=0.007, ha="left")
    _save(fig, "calibration.png")


FIGURES = {"recovery": recovery, "overlap": overlap, "lookahead": lookahead,
           "calibration": calibration}


def main(argv: list[str]) -> None:
    wanted = argv[1:] or list(FIGURES)
    for name in wanted:
        if name not in FIGURES:
            raise SystemExit(f"unknown figure {name!r}; pick from {list(FIGURES)}")
        FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv)
