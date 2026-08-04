# Theory

Everything here is implemented in `fz` in a few hundred lines of NumPy. This
document is the reasoning behind those lines: why the estimator is shaped the
way it is, where its standard error comes from, and what has to be true of a
synthetic data generating process before "we recovered the premium" means
anything at all.

## 1. The two-pass estimator

Write the cross-sectional model for stock $i$ on date $t$:

$$
r_{i,t+1} \;=\; \alpha_t \;+\; \sum_k \lambda_t^k \, z_{i,t}^k \;+\; \varepsilon_{i,t+1},
$$

where $z_{i,t}^k$ is the (winsorised, z-scored) value of characteristic $k$
known at the close of day $t$. Two things make this awkward for pooled OLS.
Returns on the same day are strongly cross-correlated — everything moves with
the market — and the premium $\lambda^k$ is not obviously constant.

Fama and MacBeth's (1973) answer is to stop fighting it. Fit one
cross-sectional regression per date, giving a *time series* of slopes
$\{\hat\lambda_t^k\}_{t=1}^T$, and do the inference in the time dimension:

$$
\bar\lambda^k = \frac{1}{T}\sum_t \hat\lambda_t^k,
\qquad
\widehat{\mathrm{se}}(\bar\lambda^k) = \sqrt{S_k / T}.
$$

Cross-sectional dependence within a day is absorbed into $\hat\lambda_t$ and
never has to be modelled: whatever the correlation structure on a given day,
it produces one number, and that number's variability across days is what the
standard error measures. `fama_macbeth` returns the whole `daily_coefs` panel
precisely because those slopes are the object of study, not an intermediate.

The classic estimator takes $S_k = \operatorname{Var}(\hat\lambda_t^k)$, the
ordinary sample variance — that is, it assumes the daily slopes are serially
independent.

## 2. When the slopes are not independent

That assumption is defensible for one-day-ahead returns and false the moment
the forward window is longer than the rebalance interval. With $h$-day
forward returns, the regressions on days $t$ and $t+1$ share $h-1$ days of
realised return, so $\hat\lambda_t$ and $\hat\lambda_{t+1}$ are mechanically
correlated: the slope series behaves like an MA($h-1$) process even if the
underlying premium is iid.

The fix is a heteroskedasticity- and autocorrelation-consistent long-run
variance. `newey_west_var` implements the Bartlett-kernel estimator of Newey
and West (1987):

$$
S \;=\; \gamma_0 \;+\; 2\sum_{l=1}^{L}\Bigl(1 - \frac{l}{L+1}\Bigr)\gamma_l,
\qquad
\gamma_l = \frac{1}{T-1}\sum_{t=l+1}^{T}(x_t - \bar x)(x_{t-l} - \bar x).
$$

Two details matter. The triangular weights are not cosmetic — they make the
kernel positive semi-definite, so $S \ge 0$ always, which a raw truncated sum
of autocovariances does not guarantee. And the $T-1$ divisor means $L = 0$
returns exactly the ordinary sample variance, so the classic Fama-MacBeth
t-stat is the $L = 0$ special case of the same code path rather than a
separate branch. `hac_lags = h - 1` is the natural default for $h$-day
overlap.

How wrong does the naive version get? Take the slope series to be an
equally weighted moving average of $h$ independent shocks,
$\hat\lambda_t = h^{-1}\sum_{j=0}^{h-1} u_{t+j}$. Then
$\gamma_l = \sigma^2 (h-l)/h^2$ for $l < h$ and zero beyond, so

$$
\gamma_0 = \frac{\sigma^2}{h},
\qquad
\gamma_0 + 2\sum_{l=1}^{h-1}\gamma_l = \sigma^2 .
$$

The naive variance is $\gamma_0$ and the true long-run variance is
$\sigma^2$, a factor of exactly $h$: an untapered (truncated) kernel would
recover all of it and the t-stat would be overstated by $\sqrt{h}$. Bartlett
does not recover all of it, because the triangular weights discount precisely
the longest lags. Summing the weighted autocovariances at $L = h-1$,

$$
\frac{S}{\gamma_0}
= 1 + \frac{2}{h^2}\sum_{l=1}^{h-1}(h-l)^2
= 1 + \frac{(h-1)(2h-1)}{3h},
$$

so the measured inflation is $\sqrt{1 + (h-1)(2h-1)/3h}$, which approaches
$\sqrt{2h/3} \approx 0.816\sqrt{h}$ rather than $\sqrt{h}$ — 6.48 against
7.94 at $h = 63$. Both curves are drawn in `docs/overlap.png`, and the
factor-by-factor measurements sit on the Bartlett one. This is a property of
the kernel, not a defect: a longer bandwidth recovers more of the tail at the
cost of a noisier estimate. The identity is checked against simulation in
[`validation.md`](validation.md), row 14.

## 3. Portfolio sorts

The non-parametric sibling of the regression. Rank stocks by $z^k$ each day
into quintiles $Q_1,\dots,Q_5$ and record each bucket's mean forward return;
the long-short portfolio $Q_5 - Q_1$ earns the premium if the characteristic
is priced. A sort discards the magnitude of the score and keeps only the
order, which makes it robust to outliers and to any monotone
mis-specification of the characteristic — and blind to whether the
relationship is linear.

That is exactly why the *shape* of the ladder is worth plotting. A positive
$Q_5 - Q_1$ spread with a lumpy middle usually means the score is only
working in the tails. All four factors in the shipped universe are monotone
across all five buckets.

Winsorising before ranking guarantees ties in the clipped tails — precisely
the names that define the long and short legs. `average_ranks` gives tied
values the mean of the ordinal ranks they span, so bucket membership and the
Spearman IC never depend on the order in which tickers happen to be listed.

## 4. Information coefficient

$$
\mathrm{IC}_t = \operatorname{Spearman}\bigl(z_{\cdot,t},\, r_{\cdot,t+1}\bigr),
$$

computed as a Pearson correlation on average ranks. Daily ICs are tiny: 0.02
to 0.05 is a good factor. The summary statistic that matters is the
information ratio $\overline{\mathrm{IC}} / \sigma_{\mathrm{IC}}$, which says
how reliably the sign repeats.

Grinold's (1989) fundamental law supplies the intuition for why a 0.03
correlation is worth anything at all: the achievable information ratio of a
strategy is approximately $\mathrm{IC}\sqrt{N}$ for $N$ independent bets. A
whisper of predictive correlation, repeated across 300 names and 1247 days,
is a t-stat of 7.

## 5. Point-in-time scores

A score is *point-in-time* if the value it takes on day $t$ depends only on
data from day $t$ and earlier. State that as an invariant on the panel
computation: for every truncation $T$,

$$
f(\text{panel}[:T]) \;=\; f(\text{panel})[:T].
$$

This is testable, and `tests/test_no_lookahead.py` tests it — for every
factor, for `rolling_ic`, and for `quintile_sort_returns`.

The interesting failure mode is subtle. Winsorisation implemented as
`np.nanquantile(panel, 0.01)` over the flattened panel computes clip bounds
from *every* date, so a score in year one is shaped by year five. It does not
change ranks — clipping is monotone — so portfolio sorts look fine and the
leak hides; what it changes is z-score magnitudes, which is exactly what the
regression reads. Computing quantiles per row costs nothing and makes the
invariant hold to the bit.

## 6. What identifiability means for a synthetic universe

The repository's headline claim is that recovery on synthetic data proves the
pipeline. That claim is empty unless each premium actually flows through the
characteristic that is supposed to measure it. The generator's log-return
equation is

$$
\log(1 + r_{i,t}) = \beta_i m_t + \delta_{i,t}
  + \lambda^{\mathrm{val}}_t v_i - \lambda^{\mathrm{size}}_t s_i
  + \lambda^{\mathrm{qual}}_t q_i + \lambda^{\mathrm{lvol}}_t \ell_i
  - \tfrac{1}{2}\sigma_i^2 + \epsilon_{i,t},
$$

and every term is wired to a measurable quantity:

- **Momentum** is $\delta_{i,t}$, a per-stock AR(1) expected-return drift with
  $\phi = 0.995$ (half-life $\approx 138$ days). Trailing 12-1 month returns
  aggregate the drift and the drift persists, so past winners keep drifting up.
  A premium attached to an iid shock would leave momentum unpredictable by
  construction.
- **Low volatility** enters through $\sigma_i$ itself: idiosyncratic
  volatility is a decreasing function of the exposure, and the premium loads
  on standardised *negative true total* volatility,
  $\sqrt{\sigma_{\mathrm{idio},i}^2 + (\beta_i\sigma_m)^2}$ — exactly the
  quantity a trailing realised-vol estimator targets. Attach the premium to a
  latent variable that never touches realised volatility and no estimator can
  find it, however correct the estimator is.
- **Value and quality** are decoupled on purpose. $\log(B/M)$ is monotone in
  $v_i$, and ROE is monotone in $q_i$ with earnings derived from *book value*
  rather than market cap. Deriving earnings from market cap makes ROE a
  disguised short-value bet, and the "quality" result is then a value result
  wearing a hat.
- Slow AR(1) wiggle on the fundamentals keeps consecutive daily
  cross-sections from being verbatim repeats of each other.

**The Ito correction.** Premia are specified as arithmetic returns, but the
generator builds prices from log-returns, and
$\mathbb{E}[e^{X}] = e^{\mu + \sigma^2/2}$ for Gaussian $X$. Without
subtracting $\sigma_i^2/2$, the convexity of exponentiation hands
high-volatility stocks a spurious drift of about 5 bp/day — enough to cancel
the low-volatility premium outright and make the factor look dead.

**Attenuation.** Even with everything wired correctly, the estimated
coefficient is not the injected one. The score $z$ is a noisy proxy for the
true exposure $x$; regressing on $z$ rather than $x$ gives, in the classical
errors-in-variables limit,

$$
\operatorname{plim} \hat\lambda \;=\; \lambda \cdot \frac{\operatorname{Cov}(x, z)}{\operatorname{Var}(z)},
$$

which is smaller in magnitude than $\lambda$ whenever the proxy is imperfect.
The right acceptance test is therefore directional and comparative, not a
point equality against the injected number.

**Estimated regressors and Shanken's correction.** The attenuation above is
the characteristic-regression version of a problem the beta-pricing
literature met first. In the classic two-pass procedure the second-stage
regressors are *estimated betas*, and treating them as exact understates the
sampling variance of the premium. Shanken (1992) showed the multiplier is

$$
1 + \lambda' \Sigma_f^{-1} \lambda,
$$

governed entirely by the factors' squared Sharpe ratio: negligible for a
market factor, substantial for a high-Sharpe one. `shanken_factor` computes
it. It does **not** apply to anything `fama_macbeth` prints, because the
regressors here are characteristics, which are observed — a claim that is
measured both ways in [`validation.md`](validation.md), rows 15 and 16,
rather than asserted.

**The placebo.** Hence `premia={"low_vol": 0.0}`. Because the override only
scales already-drawn random numbers, the placebo universe is the *same* draw
with one term deleted rather than a fresh sample — a genuine counterfactual.
The acceptance matrix in `tests/test_recovery.py` demands $t > 2$ for each
loaded premium and $|t| < 2$ for the same coefficient once its premium is
switched off. A pipeline that leaks the future, or a characteristic that is
secretly a different factor, fails one half or the other.

## 7. Factors implemented

| factor | construction | reference |
|---|---|---|
| momentum | 12-month return skipping the most recent month | Jegadeesh & Titman (1993) |
| short reversal | negative 1-week trailing return | Lehmann (1990) |
| value | log book-to-market | Fama & French (1992) |
| size | negative log market cap (SMB convention) | Fama & French (1993) |
| quality | return on equity, earnings / book | Hou, Xue & Zhang (2015) |
| low volatility | negative trailing 60-day return standard deviation | Blitz & van Vliet (2007) |

Two of those references are chosen narrowly, because the nearest-sounding
paper is about a different characteristic.

`quality_roe` is earnings over book equity, which is the q-factor model's ROE
leg (Hou, Xue and Zhang 2015); Fama and French's RMW (2015) is the other
standard home for it. It is *not* Novy-Marx (2013): the gross profitability
premium is gross profits over assets, and the argument of that paper is
precisely that earnings-based measures such as ROE are the contaminated
alternative it improves on. Citing it here would invert its conclusion.

`low_vol` is the negative of trailing *total* return volatility, which is
what Blitz and van Vliet (2007) sort on; Baker, Bradley and Wurgler (2011)
sort on both total volatility and beta. It is not Frazzini and Pedersen
(2014), whose betting-against-beta portfolios sort on market beta under a
leverage-constraint story — a different quantity with a different mechanism.
Ang, Hodrick, Xing and Zhang (2006), used as the sign reference in
[`validation.md`](validation.md) row 3, sort on *idiosyncratic* volatility
against a three-factor benchmark, which is a third quantity again; that row
measures how close it is to total volatility in this universe rather than
assuming they are interchangeable.

## 8. Caveats worth repeating

The premia are also treated as independent, which they are not: Asness,
Moskowitz and Pedersen (2013) document value and momentum jointly across
eight markets and asset classes and find them consistently negatively
correlated with each other, so a combined book is worth more than the sum of
its legs. Nothing in `make_universe` reproduces that structure.

Real research runs on CRSP and Compustat, which are licence-walled. The
synthetic universe exists so that the *code* can be falsified, and nothing
more: recovering the value premium from data in which the value premium was
planted says the estimator works, not that value works. The literature on how
easily that line gets crossed is worth reading — Harvey, Liu and Zhu (2016) on
the multiple-testing problem in the published factor zoo, and Hou, Xue and
Zhang (2020) on how many published anomalies survive careful replication.

## References

- Fama, E. & MacBeth, J. (1973). *Risk, Return, and Equilibrium: Empirical Tests.* JPE.
- Newey, W. & West, K. (1987). *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica.
- Grinold, R. (1989). *The Fundamental Law of Active Management.* JPM.
- Lehmann, B. (1990). *Fads, Martingales, and Market Efficiency.* QJE.
- Shanken, J. (1992). *On the Estimation of Beta-Pricing Models.* RFS.
- Fama, E. & French, K. (1992). *The Cross-Section of Expected Stock Returns.* JF.
- Fama, E. & French, K. (1993). *Common Risk Factors in the Returns on Stocks and Bonds.* JFE.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* JF.
- Ang, A., Hodrick, R., Xing, Y. & Zhang, X. (2006). *The Cross-Section of Volatility and Expected Returns.* JF.
- Blitz, D. & van Vliet, P. (2007). *The Volatility Effect: Lower Risk Without Lower Return.* JPM.
- Baker, M., Bradley, B. & Wurgler, J. (2011). *Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly.* FAJ.
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium.* JFE. (Gross profits / assets — *not* the reference for `quality_roe`; see section 7.)
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere.* JF.
- Frazzini, A. & Pedersen, L. (2014). *Betting Against Beta.* JFE. (Sorts on market beta — *not* the reference for `low_vol`; see section 7.)
- Hou, K., Xue, C. & Zhang, L. (2015). *Digesting Anomalies: An Investment Approach.* RFS.
- Fama, E. & French, K. (2015). *A Five-Factor Asset Pricing Model.* JFE.
- Harvey, C., Liu, Y. & Zhu, H. (2016). *... and the Cross-Section of Expected Returns.* RFS.
- Hou, K., Xue, C. & Zhang, L. (2020). *Replicating Anomalies.* RFS.
