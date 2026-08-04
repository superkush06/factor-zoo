# Validation

Everything else in this repository is the library checking itself. A
truncation test says the scores are point-in-time; a placebo says the premium
is not an artefact; a version test says the two version strings agree. All of
that is necessary and none of it is evidence that the estimator computes the
right thing, because the standard it is held to was written in the same
repository.

This page is the part that is not. Sixteen claims, each measured against
something outside `fz`: a published result, or a closed form.

Every cell of the two tables below — claim, our value, reference value,
source, verdict — is printed by `examples/validate.py`. Run it and you get
this page back:

```bash
python3 examples/validate.py        # ~15 s
```

`tests/test_validation.py` runs the same code under pytest and compares the
published tables against a live run, cell for cell, so a number cannot be
typed into a table on this page by hand and cannot drift when an estimator
changes. The prose between the tables quotes figures that no column holds —
the 68% idiosyncratic share and the 0.91 volatility correlation behind row 3,
the −0.46 momentum/size correlation behind row 4, the 1.64 and 1.012 Shanken
factors behind rows 15–16, and the −29.07 / +10.36 / 39.43 that the README's
sort table prints — and the same test file recomputes every one of them and
looks for it here. What is left in the prose is a rounding or a restatement of
a pinned cell (95.3% is row 9's 0.953, 1.13 is row 15's 1.1298, 6.48 is row
14's 6.488), or arithmetic on one.

The `agrees` column is computed inside `validate.py` from the numbers in its
own row — the sign of the estimate and whether its t-statistic clears 2, or
the sign of the rank IC — so breaking a characteristic flips a verdict and
turns CI red rather than leaving a green table that says nothing.

---

## 1. Against the published cross-section

A synthetic universe cannot confirm or refute a published finding — the
premia in it were put there on purpose. What it *can* do is show that the
pipeline is wired the way the literature is wired: that a characteristic the
literature prices positively comes out of `fama_macbeth` positive, and that
where the library's answer points the other way there is a stated reason.

Magnitudes are not comparable and no attempt is made to pretend otherwise.
The premia in `make_universe` are roughly an order of magnitude larger than
anything in CRSP, deliberately, so that recovery is unambiguous in 1500 days.
The sign, the significance, and the ordering are the content.

| # | claim | our value | reference value | source | agrees |
|---|---|---|---|---|:--:|
| 1 | A 12-month-minus-1-month momentum characteristic earns a positive premium | +3.18 bp/day (t = +7.22), Q5−Q1 = +48.3%/yr compounded | +12.01%/yr compounded excess return | Jegadeesh & Titman (1993), relative-strength strategy, 6-month formation and 6-month holding, NYSE/AMEX 1965-1989 | yes |
| 2 | High book-to-market earns more than low book-to-market | +2.90 bp/day (t = +7.30) | average slope on ln(BE/ME) = +0.50 %/month (t = 5.71) | Fama & French (1992), month-by-month cross-sectional regressions, 1963-1990 | yes |
| 3 | Quiet stocks earn more than volatile ones, per unit of characteristic | +3.50 bp/day (t = +6.45), Q5−Q1 = +43.1%/yr compounded | highest-vol quintile underperforms the lowest by about 1.06%/month | Ang, Hodrick, Xing & Zhang (2006), quintiles on idiosyncratic volatility relative to the Fama-French three-factor model | yes |
| 4 | Small stocks earn more than large ones | rank IC of `size_factor` = −0.0174 | average slope on ln(ME) = −0.15 %/month (t = −2.58), i.e. small earns more | Fama & French (1992), month-by-month cross-sectional regressions, 1963-1990 | **no** |

Rows 1–3 read `yes` because the measured premium is positive *and* its
t-statistic clears 2; row 4 reads `no` because the measured rank IC is
negative where the reference says it should be positive. Negate any of those
four signals and the corresponding verdict changes on the next run.

### What each row does and does not establish

**Row 1.** Jegadeesh and Titman's headline strategy is not this one — theirs
forms on six months and holds for six, ours forms on twelve months skipping
the most recent one and holds for a day — so the two numbers are not
measurements of the same object. What transfers is the sign and the fact that
skipping the most recent month is required for it: momentum in this universe
is a persistent expected-return drift, and a trailing window that includes
last week picks up short-horizon reversal instead. The premium is large here
because it was made large.

The `Q5−Q1 = +48.3%/yr` in that row is the long-short book's *mean daily*
return compounded over 252 days. The README's factor-sort table quotes a
`spread` of 39.43%/yr for the same signal, which is the same daily mean
annualised the other way: its Q1 and Q5 columns are mean daily returns times
252 (−29.07 and +10.36), so their difference is the long-short mean times 252
as well. The two do convert into each other — (1 + 0.3943/252)²⁵² − 1 =
0.483 — and `examples/factor_backtest.py::annualised_compounded` is that
conversion. Nothing is in tension; the labels just have to say which
convention is on screen, because the compounded number is the bigger one and
it is not the better result.

**Row 2.** Fama and French regress on ln(BE/ME) in levels; `value_btm`
z-scores it by date. Converting between the two needs the cross-sectional
dispersion of ln(BE/ME) in their sample, which we do not have, so only the
sign and the significance are compared. Both agree, and the t-statistics
happen to land in the same neighbourhood (7.30 against 5.71) — which is a
coincidence of sample size, not a validation.

**Row 3.** Ang, Hodrick, Xing and Zhang sort on *idiosyncratic* volatility
measured against the Fama-French three-factor model. `low_vol` is trailing
*total* return volatility. In real equities those are meaningfully different
characteristics — total volatility loads on market beta, which carries its
own premium — but in this universe they are close cousins: idiosyncratic
variance is 68% of total variance on average and the two volatilities
correlate 0.91 across names. The sign agrees; a like-for-like comparison
would need a residual-volatility characteristic this library does not ship.
The construction `low_vol` actually implements — a sort on total volatility —
is the one in Blitz and van Vliet (2007); [`theory.md`](theory.md) section 7
says which paper goes with which characteristic, and which neighbouring
papers are about something else.

**Row 4 — the disagreement.** The literature says small beats large.
`size_factor` in this universe says the opposite, at IC = −0.0174, and that
is not a bug in the estimator: it is a bug in the characteristic, and one
that exists in real data too. Market capitalisation is shares outstanding
times price, so `-log(mcap)` is part cross-sectional characteristic and part
accumulated past return. In a universe with persistent momentum, yesterday's
winners are today's large caps, and the size score ends up correlated −0.46
with the momentum score. The generator does price size positively
(`DEFAULT_PREMIA["size"] = 1e-4`); the contaminated proxy simply cannot see
it past the momentum leakage. `size_factor` ships because it is the standard
construction, and its IC is printed in `examples/momentum_ic.py` next to the
factors that work, because that contrast is the lesson. A version that
neutralised past returns out of market cap would recover the sign; that is a
different characteristic and it is not in this release.

---

## 2. Against closed forms

These have exact answers. Agreement is measurable rather than rhetorical, and
a disagreement here would be a defect rather than a difference of setting.

| # | claim | our value | reference value | source | agrees |
|---|---|---|---|---|:--:|
| 5 | Quintile bucket means equal the conditional means of the score | [−1.3986, −0.5318, +0.0002, +0.5318, +1.3979] | [−1.3998, −0.5319, +0.0000, +0.5319, +1.3998] | E[z \| quintile] for a standard normal, (φ(a) − φ(b)) / (Φ(b) − Φ(a)) | yes |
| 6 | A premium planted at the Jegadeesh-Titman magnitude (12.01%/yr) comes back | +1.4612 bp/day (s.e. 0.1999), −0.73 s.e. from truth | +1.6076 bp/day | the planted value, exactly known | yes |
| 7 | A premium planted at the Ang et al. magnitude (1.06%/month) comes back | +1.9346 bp/day (s.e. 0.1930), +0.73 s.e. from truth | +1.7935 bp/day | the planted value, exactly known | yes |
| 8 | A universe with no premium in it yields no premium | +0.0513 bp/day (s.e. 0.1929), +0.27 s.e. from truth | +0.0000 bp/day | the planted value, exactly known | yes |
| 9 | The 95% Fama-MacBeth interval covers the truth 95% of the time | 0.953 (286/300 replications) | 0.950 | nominal coverage of a two-sided Gaussian interval | yes |
| 10 | Rank IC of a bivariate normal matches the Spearman identity | 0.02000±0.00090, 0.04590±0.00092, 0.09463±0.00091 | 0.01910, 0.04775, 0.09553 | ρ_s = (6/π) arcsin(ρ/2) at Pearson ρ = 0.02, 0.05, 0.10 | yes |
| 11 | `hac_lags=0` is the ordinary sample variance | abs difference = < 1e-15 | 0 | same code path, L = 0 term only | yes |
| 12 | Bartlett long-run variance of an MA(1) | 2.3551 | 2.3600 | γ_0 + 2 Σ_l (1 − l/(L+1)) γ_l with θ = 0.6, L = 5 | yes |
| 13 | Bartlett long-run variance of an AR(1) | 4.4947 | 4.4817 | γ_l = ρ^l with ρ = 0.7, L = 12 | yes |
| 14 | t-stat inflation from h-day overlap, at h = 5, 21, 63 | 1.843, 3.748, 6.488 | 1.844, 3.744, 6.481 | √(1 + (h−1)(2h−1)/(3h)) for an equally weighted MA(h−1) | yes |
| 15 | Estimated betas inflate the true sampling error of a two-pass premium | sd(estimate) / mean(reported s.e.) = 1.1298 over 2000 replications | 1.1317 | Shanken (1992) errors-in-variables correction, √((A(1 + λ' Sf^−1 λ) + Sf) / (A + Sf)) | yes |
| 16 | Observed regressors do not need the correction | sd(estimate) / mean(reported s.e.) = 1.0103 | 1.0000 | no errors in variables, so the reported s.e. is already right | yes |

Each `agrees` here is a tolerance on the row's own error, evaluated at run
time: row 11 to 1e-12 absolute, rows 12 and 13 to 1% and 2% relative, row 14
to 1% at every horizon, row 5 to 0.01 absolute, rows 6–8 to two standard
errors of the planted value, rows 9 and 10 to three Monte-Carlo standard
errors, and rows 15–16 to 2% and 3%. The script prints the realised error
next to each.

![calibration](calibration.png)

### The calibration test (rows 6–9)

The recovery matrix elsewhere in this repository is *directional*: it asks
whether a premium shows up with t > 2 and a placebo does not. That is a weak
question. The strong one is whether the estimate lands where it should, and
whether the interval around it means what it claims.

So rows 6–9 use a panel the generator does not build: the characteristic is
drawn as a standard normal and *observed exactly*, and forward returns are
`λ·z + noise`. There is no errors-in-variables attenuation, no proxy, nothing
between the planted number and the estimate except sampling error. The
estimator therefore has to return λ, and it does — within 0.73 of its own
standard error at both literature magnitudes, and within 0.27 when λ is zero.

λ is set by inverting the quintile spread. For a standard normal score, the
top and bottom quintiles have conditional means ±1.3998, so a long-short book
earning a compounded 12.01%/yr — Jegadeesh and Titman's number — implies

    λ = ln(1.1201) / 252 / 2.7996 = 1.6076 bp/day per unit z.

Row 9 is the one that matters most. Three hundred independent panels, each
producing an estimate and a standard error: 286 of the 300 intervals — 95.3% —
contained the truth, against a Monte-Carlo standard error of 1.3 points. The
right panel of the figure above is that experiment: the standardised errors
against the normal density the interval assumes.

### Why the overlap inflation is not √h (row 14)

Hold a name for h days and consecutive cross-sections share h−1 days of
return, so the daily slope series behaves like an equally weighted MA(h−1) of
iid shocks. For that process γ_l = σ²(h−l)/h², so the true long-run variance
is σ² against an iid variance of γ₀ = σ²/h: a truncated kernel would recover
a factor of exactly h, and the t-stat would be overstated by √h.

Bartlett's triangular weights do not recover all of it. Summing the weighted
autocovariances at L = h−1 gives

    S / γ₀ = 1 + (h−1)(2h−1) / 3h,

so the measured inflation is √(1 + (h−1)(2h−1)/3h) ≈ 0.82√h for large h —
6.48 rather than 7.94 at h = 63. The right panel of `overlap.png` plots both
curves; the measured factor-by-factor inflations sit on the Bartlett one, not
on √h. This is a property of the kernel, not a shortcoming of the
implementation: a bandwidth chosen larger than h−1 recovers more of the tail
at the cost of noisier estimates.

Those same triangular weights are what keep the estimate non-negative: the
kernel's Fourier transform is the Fejér kernel, which is non-negative, so the
weighted autocovariance sum is a spectral average of a periodogram and cannot
go below zero. `tests/test_properties.py` checks that on series where an
*unweighted* truncated kernel does go negative, so the property is credited
to the weights rather than to the `max(s, 0.0)` guard in the implementation.

### Shanken, and why this library does not need him (rows 15–16)

The classic two-pass procedure estimates betas in a first pass and then
regresses returns on those estimates. The second pass treats a noisy
regressor as exact, and Shanken (1992) showed the resulting standard errors
are understated by a factor governed by the factors' squared Sharpe ratio,
`1 + λ'Σ_f⁻¹λ`. `shanken_factor` computes it.

Row 15 measures the effect rather than asserting it. Two thousand simulated
panels, 20 assets, one factor, betas estimated from the same sample used for
the cross-sectional regressions: the standard deviation of the estimate
across replications is 1.13 times the standard error the procedure reports,
against the 1.1317 the correction predicts. The setup uses a deliberately
extreme factor Sharpe (`1 + λ'Σ_f⁻¹λ` = 1.64) so the effect is visible; at
market-like premia — 0.5%/month against 4.5%/month volatility — the factor is
1.012 and the correction is worth less than 1%, which is why it is so often
skipped without consequence.

Row 16 is the same experiment with the betas handed over rather than
estimated, and the ratio comes back at 1.01. That is the case `fama_macbeth`
is in: it regresses on **characteristics**, which are observed. No Shanken
adjustment applies to anything this library prints, and `shanken_factor`
exists so that the distinction can be computed rather than argued about.

What *does* apply to characteristics is classical errors-in-variables
attenuation: `momentum` is a noisy proxy for the true expected-return drift,
so its coefficient is biased toward zero by Cov(x, z)/Var(z). That is derived
in [`theory.md`](theory.md) and it is exactly why the acceptance matrix is
directional — the planted premium is not the number the regression should
return on `make_universe` output, which is what rows 6–9 use a clean panel
for.

---

## What is not validated here

- **Nothing about live markets.** Every premium recovered above was written
  into the data first. This page shows the estimator is correct, not that
  momentum pays.
- **Transaction costs, capacity, borrow, survivorship.** The sorts are
  frictionless and equal-weighted. A 4-Sharpe long-short book that rebalances
  daily is an arithmetic result, not a strategy.
- **Bandwidth selection.** `hac_lags` is a parameter; nothing here says how to
  pick it on data whose autocorrelation you do not know.
- **Cross-sectional dependence within a day.** Fama-MacBeth absorbs it into
  the daily slope rather than modelling it. If the daily slopes are themselves
  correlated across factors, the univariate intervals above are still right
  and a joint test would need more.
- **The universe's realism.** `make_universe` has Gaussian returns, no
  sectors, no earnings announcements, no delistings, and constant premia. It
  is a test harness with a known answer, and that is all it claims to be.

## References

- Fama, E. & MacBeth, J. (1973). *Risk, Return, and Equilibrium: Empirical Tests.* Journal of Political Economy.
- Newey, W. & West, K. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica.
- Lehmann, B. (1990). *Fads, Martingales, and Market Efficiency.* Quarterly Journal of Economics.
- Shanken, J. (1992). *On the Estimation of Beta-Pricing Models.* Review of Financial Studies.
- Fama, E. & French, K. (1992). *The Cross-Section of Expected Stock Returns.* Journal of Finance.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency.* Journal of Finance.
- Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006). *The Cross-Section of Volatility and Expected Returns.* Journal of Finance.
- Blitz, D. & van Vliet, P. (2007). *The Volatility Effect: Lower Risk Without Lower Return.* Journal of Portfolio Management.
