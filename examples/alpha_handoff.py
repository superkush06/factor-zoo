"""One rebalance cycle, end to end: scores in, a holdable book out.

A cross-sectional factor library produces one thing the rest of a stack
actually consumes: a vector of expected returns, dated. Everything else --
the sorts, the t-stats, the placebo matrix -- exists to decide whether that
vector is worth believing. This example builds it and then does the smallest
honest thing you can do with it.

    scores  ->  Fama-MacBeth premia (trailing only)  ->  mu_t
    returns ->  one-factor risk model (trailing only) ->  Sigma_t
    mu, Sigma -> dollar-neutral mean-variance weights -> daily P&L

The optimiser here is four lines of algebra, not a portfolio library: the
covariance is one market factor plus a diagonal, so Sherman-Morrison gives
Sigma^-1 in closed form and the weights never need a 300x300 solve. A real
stack replaces this step with constrained optimisation (sibling repo
`portopt`) and hands the resulting book to a risk engine (`risk`). The point
of this file is the handoff, not the optimiser.

The signal and the risk model are point-in-time. Premia are re-estimated on
an expanding window that ends at the rebalance date, the risk model uses a
trailing window, and the book is held until the next rebalance -- so the
*shape* of the P&L below is the P&L a run of this pipeline would have earned,
not a fit.

The leverage is not point-in-time, and the printed table says so. Both books
are scaled to `TARGET_VOL` using the realised volatility of the whole
backtest, which is not knowable at the start of it. Sharpe and the
correlation between the two books are scale-invariant and therefore unaffected;
`ret %/yr` and `maxDD %` are proportional to that constant, so read them as a
comparison between the two sizings at a common risk level rather than as
returns anyone could have earned. Scaling on a trailing volatility estimate
instead would change both columns and is a portfolio-construction decision
this file deliberately does not make -- see the sibling `portopt`.

Run:  python3 examples/alpha_handoff.py
"""

from __future__ import annotations

import argparse

import numpy as np

from fz.crossect import fama_macbeth, forward_returns
from fz.factors import low_vol, momentum, quality_roe, value_btm
from fz.portfolio import long_short_return, quintile_sort_returns, sharpe_annualised
from fz.universe import make_universe

REBALANCE = 21          # trading days between refits
RISK_WINDOW = 250       # trailing days for the risk model
WARMUP = 400            # first rebalance: enough history to estimate anything
TARGET_VOL = 0.10       # annualised, both books scaled to it


def _scores(u) -> list[np.ndarray]:
    return [momentum(u, lookback=252, skip=21), value_btm(u),
            quality_roe(u), low_vol(u, window=60)]


def _risk_model(returns: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """One-factor covariance from a trailing window: beta, market variance,
    idiosyncratic variances. The market is the equal-weighted mean return."""
    mkt = returns.mean(axis=1)
    mkt_c = mkt - mkt.mean()
    var_m = float(mkt_c @ mkt_c) / (len(mkt) - 1)
    beta = (mkt_c @ (returns - returns.mean(axis=0))) / (var_m * (len(mkt) - 1))
    resid = returns - np.outer(mkt, beta)
    idio = resid.var(axis=0, ddof=1)
    return beta, var_m, np.maximum(idio, 1e-8)


def _dollar_neutral_weights(mu, beta, var_m, idio):
    """w proportional to Sigma^-1 (mu - c 1) with c set so the book is
    dollar neutral, and Sigma = var_m * beta beta' + diag(idio).

    Sherman-Morrison: (D + v b b')^-1 x = D^-1 x - v (b'D^-1 x) / (1 + v b'D^-1 b) * D^-1 b.
    """
    def solve(x):
        dx = x / idio
        db = beta / idio
        return dx - var_m * float(beta @ dx) / (1.0 + var_m * float(beta @ db)) * db

    s_mu, s_one = solve(mu), solve(np.ones_like(mu))
    w = s_mu - (s_mu.sum() / s_one.sum()) * s_one
    gross = np.abs(w).sum()
    return w / gross if gross > 0 else w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stocks", type=int, default=300)
    ap.add_argument("--n-days", type=int, default=1500)
    args = ap.parse_args()

    u = make_universe(n_stocks=args.n_stocks, n_days=args.n_days, seed=0)
    scores = _scores(u)
    fwd = forward_returns(u.returns)
    n_days = u.returns.shape[0]

    pnl = np.full(n_days, np.nan)
    turnover, w_prev, n_fits = [], None, 0
    premia_path = []

    for t0 in range(WARMUP, n_days - 1, REBALANCE):
        # --- alpha: premia from history that ends at t0 -------------------
        # fwd[t] is earned between t and t+1, so the last row that is known
        # at the close of t0 is fwd[t0 - 1]: slice to t0, not t0 + 1.
        res = fama_macbeth([s[:t0] for s in scores], fwd[:t0])
        lam = res.coefficients[1:]
        n_fits += 1
        premia_path.append(lam)
        z = np.array([s[t0] for s in scores])              # (K, n_stocks)
        live = ~np.any(np.isnan(z), axis=0)
        mu = np.zeros(z.shape[1])
        mu[live] = lam @ z[:, live]

        # --- risk: trailing window, same cut-off --------------------------
        win = u.returns[t0 - RISK_WINDOW + 1: t0 + 1]
        beta, var_m, idio = _risk_model(win)

        w = np.zeros(z.shape[1])
        w[live] = _dollar_neutral_weights(mu[live], beta[live], var_m, idio[live])
        if w_prev is not None:
            turnover.append(float(np.abs(w - w_prev).sum()))
        w_prev = w

        # --- hold it until the next rebalance -----------------------------
        for t in range(t0 + 1, min(t0 + 1 + REBALANCE, n_days)):
            pnl[t] = float(w @ np.nan_to_num(u.returns[t]))

    live_pnl = pnl[~np.isnan(pnl)]
    # Ex-post leverage: the divisor is the realised volatility of the whole
    # backtest, which was not knowable at the start of it. Sharpe and the
    # correlation below do not depend on it; ret %/yr and maxDD % do.
    scale = TARGET_VOL / (live_pnl.std(ddof=1) * np.sqrt(252))
    mv = live_pnl * scale

    # --- the naive book: equal-weighted Q5-Q1 of the four scores, blended --
    legs = np.array([long_short_return(quintile_sort_returns(s, fwd, 5))
                     for s in scores])
    # leg[t] is earned between t and t+1; pnl[t] is earned between t-1 and t.
    ls = legs[:, WARMUP: n_days - 1].mean(axis=0)
    ls = ls[~np.isnan(ls)]
    ew = ls * (TARGET_VOL / (ls.std(ddof=1) * np.sqrt(252)))

    lam_last = premia_path[-1] * 1e4
    print(f"{n_fits} rebalances, {len(live_pnl)} days held, "
          f"{args.n_stocks} names\n")
    print("premia at the last refit, bp/day (trailing estimate, not the truth)")
    print("   mom {:.2f}   val {:.2f}   qual {:.2f}   lvol {:.2f}\n".format(*lam_last))
    print(f"{'book':<28} {'ret %/yr':>9} {'vol %/yr':>9} {'Sharpe':>8} {'maxDD %':>9}")
    for name, r in (("mean-variance on mu, Sigma", mv), ("equal-weight Q5-Q1 blend", ew)):
        cum = np.cumprod(1.0 + r)
        dd = float((cum / np.maximum.accumulate(cum) - 1.0).min()) * 100
        print(f"{name:<28} {r.mean() * 252 * 100:>9.2f} "
              f"{r.std(ddof=1) * np.sqrt(252) * 100:>9.2f} "
              f"{sharpe_annualised(r):>8.2f} {dd:>9.2f}")
    n = min(len(mv), len(ew))
    print(f"\ncorrelation of the two books: {np.corrcoef(mv[:n], ew[:n])[0, 1]:.3f}")
    print(f"mean turnover per rebalance: {np.mean(turnover):.2f} of gross book")
    print(f"\nboth books are levered to {TARGET_VOL:.0%} using full-sample realised "
          "volatility, so ret %/yr\nand maxDD % are ex-post normalised; Sharpe "
          "and the correlation are scale-invariant.")


if __name__ == "__main__":
    main()
