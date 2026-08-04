"""Check this library against things that were not written in this repository.

Two kinds of outside authority are used here, and they are not
interchangeable.

*Published results.* The cross-sectional literature fixes the **sign** of
each premium and its rough size in real markets. A synthetic universe cannot
confirm or refute those findings -- the premia in it were put there on
purpose -- but a pipeline that reports the value premium with the sign Fama
and French found, and the volatility effect with the sign Ang, Hodrick, Xing
and Zhang found, is at least wired the way the literature is wired. Where our
answer disagrees with theirs, the row says so.

Every `agrees` verdict below is computed from the numbers in the same row --
the sign of the estimate and whether its t-statistic clears `T_BAR`, or the
sign of the rank IC -- and never written down as a constant. Negate a
characteristic and the affected rows print `DIFF` and
`tests/test_validation.py` goes red; that property is what makes the column
worth printing.

*Closed forms.* The rest of the table compares the estimators against
mathematics: the conditional means of a standard normal inside its own
quintiles, the Spearman correlation implied by a bivariate normal, the
Bartlett long-run variance of a moving average, and the Shanken
errors-in-variables inflation. These have exact answers, so agreement is
measurable rather than rhetorical, and every number below is printed by this
script rather than typed into it.

Run:  python3 examples/validate.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from fz import (
    fama_macbeth,
    forward_returns,
    long_short_return,
    low_vol,
    make_universe,
    momentum,
    newey_west_var,
    quality_roe,
    quintile_sort_returns,
    rank_information_coefficient,
    shanken_factor,
    size_factor,
    value_btm,
)

ND = NormalDist()
N_STOCKS, N_DAYS, SEED = 300, 1500, 0
T_BAR = 2.0     # significance bar the literature rows are judged against

# Rows 15-16. The first triple is a deliberately extreme factor Sharpe, so the
# Shanken inflation is visible in 2000 replications; the second is a
# market-like premium and volatility, where it is not. Both are quoted in the
# prose of docs/validation.md, so both live here rather than inline.
SHANKEN_LAM, SHANKEN_SIG_F, SHANKEN_SIG_E = 0.04, 0.05, 0.06
MARKET_LAM, MARKET_SIG_F = 0.005, 0.045


@dataclass
class Check:
    """One row of docs/validation.md.

    `agrees` is always derived from the measured numbers in the same row and
    never written down as a constant; it is coerced to a plain `bool` here so
    that a numpy scalar coming out of a comparison still compares as one.
    """
    claim: str
    ours: str
    reference: str
    ref_value: str
    agrees: bool
    note: str = ""

    def __post_init__(self) -> None:
        self.agrees = bool(self.agrees)


# --------------------------------------------------------------------------
# Closed forms used as ground truth
# --------------------------------------------------------------------------
def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_quintile_means() -> np.ndarray:
    """E[z | z in quintile k] for a standard normal, k = 1..5.

    For a < z < b the truncated mean is (phi(a) - phi(b)) / (Phi(b) - Phi(a)).
    """
    cuts = [-math.inf] + [ND.inv_cdf(k / 5) for k in (1, 2, 3, 4)] + [math.inf]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        pa = 0.0 if a == -math.inf else _phi(a)
        pb = 0.0 if b == math.inf else _phi(b)
        Pa = 0.0 if a == -math.inf else ND.cdf(a)
        Pb = 1.0 if b == math.inf else ND.cdf(b)
        out.append((pa - pb) / (Pb - Pa))
    return np.array(out)


Q_SPREAD = float(normal_quintile_means()[-1] - normal_quintile_means()[0])


def premium_for_annual_spread(spread: float) -> float:
    """Premium per unit z-score per day whose Q5-Q1 book compounds to `spread`/yr."""
    return math.log1p(spread) / 252.0 / Q_SPREAD


def bartlett_inflation(h: int) -> float:
    """se(HAC, L = h-1) / se(iid) for an equally weighted MA(h-1) slope series.

    With lam_t = (u_t + ... + u_{t+h-1}) / h and u iid, gamma_l = sigma^2 (h-l)/h^2,
    so the Bartlett sum at L = h-1 is sigma^2/h * (1 + (h-1)(2h-1)/(3h)) while the
    iid variance is gamma_0 = sigma^2/h. The truncated kernel would give exactly
    h; Bartlett's triangular weights discount the longest lags, which is why the
    inflation is ~0.82 sqrt(h) rather than sqrt(h).
    """
    return math.sqrt(1.0 + (h - 1) * (2 * h - 1) / (3.0 * h))


def planted_panel(premium: float, seed: int, n_days: int, n_stocks: int,
                  noise: float = 0.015):
    """A panel whose characteristic is observed exactly, so there is no
    errors-in-variables attenuation and the Fama-MacBeth coefficient has to
    come back as the number that was planted."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_days, n_stocks))
    r = np.full((n_days, n_stocks), np.nan)
    r[1:] = premium * z[:-1] + noise * rng.standard_normal((n_days - 1, n_stocks))
    return z, r


# --------------------------------------------------------------------------
# Group A: the published cross-section
# --------------------------------------------------------------------------
def _sign_note(agrees: bool) -> str:
    """Opening clause of a literature row's note, taken from its verdict."""
    return ("Sign and significance agree." if agrees
            else f"Sign or significance disagrees (bar: t > {T_BAR:.0f}).")


def volatility_decomposition(u) -> tuple[float, float]:
    """(idiosyncratic share of variance, corr(total vol, idiosyncratic vol)).

    Ang et al. sort on idiosyncratic volatility against a three-factor
    benchmark; `low_vol` is trailing *total* volatility. These two numbers are
    how far apart the two characteristics are in this universe, and row 3's
    note and the prose of docs/validation.md both quote them, so they are
    computed once here.
    """
    r = u.returns[1:]
    mkt = r.mean(axis=1)
    beta = ((mkt - mkt.mean()) @ (r - r.mean(axis=0))) / ((mkt - mkt.mean()) @ (mkt - mkt.mean()))
    total = r.std(axis=0)
    idio = np.sqrt(np.maximum(total ** 2 - (beta * mkt.std()) ** 2, 0.0))
    return (float(np.mean(idio ** 2 / total ** 2)),
            float(np.corrcoef(total, idio)[0, 1]))


def momentum_size_correlation(u) -> float:
    """Pooled correlation between the momentum and size scores.

    Both pages explain row 4's disagreement with this number: market cap is
    shares times price, so `-log(mcap)` carries accumulated past return, and
    in a universe with persistent momentum that leakage points the wrong way.
    Scores are already z-scored by date, so pooling and averaging the per-date
    correlations give the same figure.
    """
    m, s = momentum(u, lookback=252, skip=21), size_factor(u)
    ok = ~np.isnan(m) & ~np.isnan(s)
    return float(np.corrcoef(m[ok], s[ok])[0, 1])


def literature_checks() -> list[Check]:
    u = make_universe(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED)
    scores = [momentum(u, lookback=252, skip=21), value_btm(u),
              quality_roe(u), low_vol(u, window=60)]
    fwd = forward_returns(u.returns)
    res = fama_macbeth(scores, fwd)
    bp, t = res.coefficients[1:] * 1e4, res.t_stats[1:]

    def spread(s):
        """Q5-Q1 annualised by compounding, to match how the published
        long-short numbers below are quoted."""
        ls = long_short_return(quintile_sort_returns(s, fwd, 5))
        return (1.0 + np.nanmean(ls)) ** 252 - 1.0

    size_ic = float(np.nanmean(rank_information_coefficient(size_factor(u), fwd)))

    # Verdicts are derived from the numbers above, never asserted. A row
    # "agrees" with the published cross-section when our estimate has the sign
    # that literature reports and clears the conventional t > 2 bar; the size
    # row agrees when its rank IC is positive, because the reference prices
    # small-minus-big positively. Break a characteristic and the verdict has
    # to flip -- that is the whole point of publishing the column.
    priced_up = [bool(bp[k] > 0.0 and t[k] > T_BAR) for k in range(4)]
    size_agrees = bool(size_ic > 0.0)

    # How close is total volatility to idiosyncratic volatility *here*, and how
    # badly does market cap leak past returns into the size score?
    idio_share, vol_corr = volatility_decomposition(u)
    mom_size_corr = momentum_size_correlation(u)

    return [
        Check(
            "A 12-month-minus-1-month momentum characteristic earns a positive premium",
            f"{bp[0]:+.2f} bp/day (t = {t[0]:+.2f}), "
            f"Q5-Q1 = {spread(scores[0]) * 100:+.1f}%/yr compounded",
            "Jegadeesh & Titman (1993), relative-strength strategy, 6-month "
            "formation and 6-month holding, NYSE/AMEX 1965-1989",
            "+12.01%/yr compounded excess return",
            priced_up[0],
            f"{_sign_note(priced_up[0])} The magnitude does not and is not "
            "meant to: this universe's premia are deliberately an order of "
            "magnitude larger than anything in CRSP so that recovery is "
            "unambiguous in 1500 days.",
        ),
        Check(
            "High book-to-market earns more than low book-to-market",
            f"{bp[1]:+.2f} bp/day (t = {t[1]:+.2f})",
            "Fama & French (1992), month-by-month cross-sectional regressions, "
            "1963-1990",
            "average slope on ln(BE/ME) = +0.50 %/month (t = 5.71)",
            priced_up[1],
            f"{_sign_note(priced_up[1])} The magnitudes are not comparable: "
            "their regressor is ln(BE/ME) in levels, ours is a z-score, and the "
            "conversion needs the cross-sectional dispersion of ln(BE/ME) in "
            "their sample.",
        ),
        Check(
            "Quiet stocks earn more than volatile ones, per unit of characteristic",
            f"{bp[3]:+.2f} bp/day (t = {t[3]:+.2f}), "
            f"Q5-Q1 = {spread(scores[3]) * 100:+.1f}%/yr compounded",
            "Ang, Hodrick, Xing & Zhang (2006), quintiles on idiosyncratic "
            "volatility relative to the Fama-French three-factor model",
            "highest-vol quintile underperforms the lowest by about "
            "1.06%/month",
            priced_up[3],
            f"{_sign_note(priced_up[3])} Their sort is on *idiosyncratic* "
            "volatility against a three-factor benchmark; `low_vol` here is "
            "trailing *total* return volatility. In this universe the two are "
            f"close cousins -- idiosyncratic variance is {idio_share:.0%} of "
            f"total on average and the two volatilities correlate {vol_corr:.2f} "
            "across names -- which is not true of real equities.",
        ),
        Check(
            "Small stocks earn more than large ones",
            f"rank IC of `size_factor` = {size_ic:+.4f}",
            "Fama & French (1992), month-by-month cross-sectional regressions, "
            "1963-1990",
            "average slope on ln(ME) = -0.15 %/month (t = -2.58), i.e. small "
            "earns more",
            size_agrees,
            "The reference prices small-minus-big positively, so this row "
            "agrees exactly when the rank IC of `size_factor` is positive; "
            f"measured, it is {size_ic:+.4f}. Market cap is shares times "
            "price, so -log(mcap) is part characteristic and part accumulated "
            "return; in a universe with persistent momentum that contamination "
            f"points the wrong way -- the size score correlates "
            f"{mom_size_corr:.2f} with the momentum score. `size_factor` ships "
            "because it is the standard construction, not because it works "
            "here.",
        ),
    ]


# --------------------------------------------------------------------------
# Group B: closed forms
# --------------------------------------------------------------------------
def quintile_ladder_check() -> Check:
    """The sort machinery against truncated-normal conditional means."""
    lam = premium_for_annual_spread(0.1201)
    rng = np.random.default_rng(101)
    n_days, n_stocks = 500, 2000
    z = rng.standard_normal((n_days, n_stocks))
    fwd = np.full_like(z, np.nan)
    fwd[:-1] = lam * z[:-1]                     # noiseless: pure sorting test
    rungs = np.nanmean(quintile_sort_returns(z, fwd, 5), axis=0) / lam
    analytic = normal_quintile_means()
    err = float(np.abs(rungs - analytic).max())
    return Check(
        "Quintile bucket means equal the conditional means of the score",
        "[" + ", ".join(f"{x:+.4f}" for x in rungs) + "]",
        "E[z | quintile] for a standard normal, "
        "(phi(a) - phi(b)) / (Phi(b) - Phi(a))",
        "[" + ", ".join(f"{x:+.4f}" for x in analytic) + "]",
        err < 0.01,
        f"max absolute error {err:.4f} over {n_days * n_stocks:,} score-days.",
    )


def calibration_check() -> list[Check]:
    """Plant a premium at a literature magnitude; demand it back within its
    own standard error, and demand the interval to have the coverage it
    advertises."""
    out = []
    targets = [
        ("A premium planted at the Jegadeesh-Titman magnitude (12.01%/yr) "
         "comes back", premium_for_annual_spread(0.1201)),
        ("A premium planted at the Ang et al. magnitude (1.06%/month) "
         "comes back", premium_for_annual_spread((1.0106 ** 12) - 1.0)),
        ("A universe with no premium in it yields no premium", 0.0),
    ]
    for i, (claim, lam) in enumerate(targets):
        z, r = planted_panel(lam, seed=200 + i, n_days=1500, n_stocks=400)
        res = fama_macbeth([z], forward_returns(r))
        dev = (res.coefficients[1] - lam) / res.std_errors[1]
        out.append(Check(
            claim,
            f"{res.coefficients[1] * 1e4:+.4f} bp/day "
            f"(s.e. {res.std_errors[1] * 1e4:.4f}), {dev:+.2f} s.e. from truth",
            "the planted value, exactly known",
            f"{lam * 1e4:+.4f} bp/day",
            abs(dev) < 2.0,
        ))

    lam = premium_for_annual_spread(0.1201)
    n_rep = 300
    hits = 0
    for s in range(n_rep):
        z, r = planted_panel(lam, seed=1000 + s, n_days=600, n_stocks=250)
        res = fama_macbeth([z], forward_returns(r))
        hits += abs(res.coefficients[1] - lam) < 1.959964 * res.std_errors[1]
    cover = hits / n_rep
    out.append(Check(
        "The 95% Fama-MacBeth interval covers the truth 95% of the time",
        f"{cover:.3f} ({hits}/{n_rep} replications)",
        "nominal coverage of a two-sided Gaussian interval",
        "0.950",
        abs(cover - 0.95) < 3 * math.sqrt(0.95 * 0.05 / n_rep),
        f"Monte-Carlo standard error {math.sqrt(0.95 * 0.05 / n_rep):.3f}.",
    ))
    return out


def spearman_check() -> Check:
    """Rank IC against the bivariate-normal Spearman identity."""
    rng = np.random.default_rng(303)
    rows, ok = [], True
    for rho in (0.02, 0.05, 0.10):
        n_days, n_stocks = 3000, 400
        a = rng.standard_normal((n_days, n_stocks))
        b = rho * a + math.sqrt(1.0 - rho ** 2) * rng.standard_normal((n_days, n_stocks))
        ic = rank_information_coefficient(a, b)
        m = float(np.nanmean(ic))
        se = float(np.nanstd(ic)) / math.sqrt(np.sum(~np.isnan(ic)))
        analytic = (6.0 / math.pi) * math.asin(rho / 2.0)
        ok &= abs(m - analytic) < 3 * se
        rows.append((rho, m, se, analytic))
    devs = [(m - a) / se for _, m, se, a in rows]
    return Check(
        "Rank IC of a bivariate normal matches the Spearman identity",
        ", ".join(f"{m:.5f}+-{se:.5f}" for _, m, se, _ in rows),
        "rho_s = (6/pi) arcsin(rho/2) at Pearson rho = 0.02, 0.05, 0.10",
        ", ".join(f"{a:.5f}" for *_, a in rows),
        ok,
        "Deviations of " + ", ".join(f"{d:+.1f}" for d in devs)
        + " Monte-Carlo standard errors.",
    )


def newey_west_checks() -> list[Check]:
    rng = np.random.default_rng(404)

    # (i) L = 0 is the ordinary sample variance, to machine precision.
    x = rng.standard_normal(5000)
    exact = float(abs(newey_west_var(x, lags=0) - np.var(x, ddof=1)))

    # (ii) MA(1): gamma_0 = s^2 (1 + th^2), gamma_1 = s^2 th, gamma_l = 0 beyond.
    theta, lags, n = 0.6, 5, 2_000_000
    e = rng.standard_normal(n + 1)
    ma1 = e[1:] + theta * e[:-1]
    ma1_ana = (1 + theta ** 2) + 2 * (1 - 1 / (lags + 1)) * theta
    ma1_ours = newey_west_var(ma1, lags=lags)

    # (iii) AR(1): gamma_l = s^2 rho^l.
    rho, n = 0.7, 500_000
    innov = rng.standard_normal(n) * math.sqrt(1.0 - rho ** 2)
    ar1 = np.empty(n)
    ar1[0] = rng.standard_normal()
    for t in range(1, n):
        ar1[t] = rho * ar1[t - 1] + innov[t]
    lags = 12
    w = 1.0 - np.arange(1, lags + 1) / (lags + 1.0)
    ar1_ana = 1.0 + 2.0 * float(np.sum(w * rho ** np.arange(1, lags + 1)))
    ar1_ours = newey_west_var(ar1, lags=lags)

    # (iv) the overlap inflation curve.
    u = rng.standard_normal(400_000 + 70)
    infl_ours, infl_ana = [], []
    for h in (5, 21, 63):
        lam_t = np.convolve(u, np.ones(h) / h, mode="valid")
        infl_ours.append(math.sqrt(newey_west_var(lam_t, h - 1)
                                   / newey_west_var(lam_t, 0)))
        infl_ana.append(bartlett_inflation(h))

    return [
        Check("`hac_lags=0` is the ordinary sample variance",
              # the exact value here is a 1-ULP artefact that differs between
              # interpreters (2.22e-16 on 3.11, 0.00e+00 on 3.12), so report
              # it at a stable resolution rather than pinning the last bit
              f"abs difference = {'< 1e-15' if exact < 1e-15 else f'{exact:.2e}'}",
              "same code path, L = 0 term only",
              "0", exact < 1e-12, "Exact, not approximate: same estimator."),
        Check("Bartlett long-run variance of an MA(1)",
              f"{ma1_ours:.4f}",
              "gamma_0 + 2 sum_l (1 - l/(L+1)) gamma_l with theta = 0.6, L = 5",
              f"{ma1_ana:.4f}", abs(ma1_ours / ma1_ana - 1) < 0.01,
              f"relative error {abs(ma1_ours / ma1_ana - 1):.2%} at n = 2,000,000."),
        Check("Bartlett long-run variance of an AR(1)",
              f"{ar1_ours:.4f}",
              "gamma_l = rho^l with rho = 0.7, L = 12",
              f"{ar1_ana:.4f}", abs(ar1_ours / ar1_ana - 1) < 0.02,
              f"relative error {abs(ar1_ours / ar1_ana - 1):.2%} at n = 500,000."),
        Check("t-stat inflation from h-day overlap, at h = 5, 21, 63",
              ", ".join(f"{v:.3f}" for v in infl_ours),
              "sqrt(1 + (h-1)(2h-1)/(3h)) for an equally weighted MA(h-1)",
              ", ".join(f"{v:.3f}" for v in infl_ana),
              all(abs(a / b - 1) < 0.01 for a, b in zip(infl_ours, infl_ana, strict=True)),
              "Not sqrt(h): the Bartlett weights discount the longest lags, "
              "which costs about 18% at h = 63 "
              f"(sqrt(63) = {math.sqrt(63):.3f}).",
              ),
    ]


def shanken_checks(n_rep: int = 2000) -> list[Check]:
    """Estimated betas need Shanken's correction; observed characteristics
    do not. Both halves are measured, because the second is the reason this
    library regresses on characteristics."""
    n_stocks, n_days = 20, 4000
    lam, sig_f, sig_e = SHANKEN_LAM, SHANKEN_SIG_F, SHANKEN_SIG_E
    betas = np.linspace(0.5, 1.5, n_stocks).reshape(-1, 1)
    c = shanken_factor([lam], [[sig_f ** 2]])
    cross = sig_e ** 2 / (n_stocks * betas.var())      # cross-sectional term
    predicted = math.sqrt((cross * c + sig_f ** 2) / (cross + sig_f ** 2))

    def run(estimate_betas: bool) -> float:
        rng = np.random.default_rng(41)
        est = np.empty(n_rep)
        ses = np.empty(n_rep)
        for i in range(n_rep):
            f = rng.normal(lam, sig_f, size=(n_days, 1))
            r = f @ betas.T + rng.normal(0.0, sig_e, size=(n_days, n_stocks))
            if estimate_betas:
                X = np.column_stack([np.ones(n_days), f])
                b = np.linalg.lstsq(X, r, rcond=None)[0][1:].T
            else:
                b = betas
            Z = np.column_stack([np.ones(n_stocks), b])
            lam_t = (np.linalg.solve(Z.T @ Z, Z.T) @ r.T).T[:, 1]
            est[i] = lam_t.mean()
            ses[i] = lam_t.std(ddof=1) / math.sqrt(n_days)
        return float(est.std(ddof=1) / ses.mean())

    measured, control = run(True), run(False)
    return [
        Check("Estimated betas inflate the true sampling error of a two-pass "
              "premium",
              f"sd(estimate) / mean(reported s.e.) = {measured:.4f} over "
              f"{n_rep} replications",
              "Shanken (1992) errors-in-variables correction, "
              "sqrt((A(1 + lam' Sf^-1 lam) + Sf) / (A + Sf))",
              f"{predicted:.4f}",
              abs(measured / predicted - 1) < 0.02,
              f"1 + lam' Sf^-1 lam = {c:.3f} here, a deliberately high factor "
              "Sharpe; at market-like premia "
              f"({shanken_factor([MARKET_LAM], [[MARKET_SIG_F ** 2]]):.3f}) "
              "the correction is under 1%."),
        Check("Observed regressors do not need the correction",
              f"sd(estimate) / mean(reported s.e.) = {control:.4f}",
              "no errors in variables, so the reported s.e. is already right",
              "1.0000", abs(control - 1.0) < 0.03,
              "This is why `fama_macbeth` regresses on characteristics and "
              "ships no Shanken adjustment."),
    ]


def all_checks() -> tuple[list[Check], list[Check]]:
    group_b = [quintile_ladder_check(), *calibration_check(), spearman_check(),
               *newey_west_checks(), *shanken_checks()]
    return literature_checks(), group_b


def _print(title: str, checks: list[Check]) -> None:
    print(f"\n{title}\n" + "=" * len(title))
    for c in checks:
        print(f"\n  {'ok ' if c.agrees else 'DIFF'}  {c.claim}")
        print(f"        ours      {c.ours}")
        print(f"        reference {c.ref_value}")
        print(f"        source    {c.reference}")
        if c.note:
            print(f"        note      {c.note}")


def main() -> None:
    lit, closed = all_checks()
    _print("Against the published cross-section", lit)
    _print("Against closed forms", closed)
    n_diff = sum(not c.agrees for c in lit + closed)
    print(f"\n{len(lit) + len(closed)} checks, {n_diff} disagreement(s) — "
          "see docs/validation.md for what each one means.")


if __name__ == "__main__":
    main()
