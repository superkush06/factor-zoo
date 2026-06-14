# Factor zoo theory

## Cross-sectional return models

Stocks have characteristics — momentum, valuation ratios, quality
metrics, recent volatility. **Cross-sectional asset pricing** asks:
do future returns line up with these characteristics?

For each day \(t\), build factor scores \(z_i^k\) (one per stock \(i\),
factor \(k\)), then regress next-day returns on the scores:
\[
r_{i, t+1} = \alpha_t + \sum_k \lambda_t^k z_{i, t}^k + \varepsilon_{i, t+1}.
\]
This is **Fama-MacBeth** (1973). The daily coefficients \(\lambda_t^k\)
estimate the per-period premium. The **Fama-MacBeth point estimate** is
\(\bar\lambda^k = \frac{1}{T}\sum_t \lambda_t^k\); the **t-stat** is
\(\bar\lambda^k / (s_\lambda / \sqrt T)\).

## Portfolio sorts

Alternative: rank stocks each day by \(z^k\) into quintiles \(Q_1, \ldots, Q_5\).
The **long-short** portfolio \(Q_5 - Q_1\) earns \(\bar\lambda^k\) on average if
the characteristic is priced. Sharpe ratio of this portfolio is the
standard "is the factor real?" diagnostic.

## Information Coefficient

Per-day Spearman correlation between factor score and forward returns:
\[
\mathrm{IC}_t = \mathrm{Spearman}(z_{\cdot, t}, r_{\cdot, t+1}).
\]
Good factors: \(\bar{\mathrm{IC}} \sim 0.02\)-\(0.05\). The **Information
Ratio** is \(\bar{\mathrm{IC}} / \sigma_{\mathrm{IC}}\).

## Factors implemented

| factor       | construction                                            | reference                             |
|--------------|----------------------------------------------------------|---------------------------------------|
| momentum     | 12m return excluding most recent month                  | Jegadeesh-Titman 1993                 |
| short reversal | -1w trailing return (mean reversion)                  | Lehmann 1990                          |
| value        | log book-to-market                                       | Fama-French 1992                      |
| size         | -log market cap (SMB convention)                         | Fama-French 1993                      |
| quality (ROE)| earnings / book value                                    | Novy-Marx 2013                        |
| low vol      | -trailing 60d return std                                 | Frazzini-Pedersen 2014                |

## Synthetic-data caveats

Real research uses CRSP / Compustat. This repo ships a **synthetic
universe** that injects each factor's "true premium" into the data
generating process. Recovering positive premia therefore **proves
that the factor pipeline is wired correctly**, not that any of these
factors actually work in live markets.

To use real data, replace `fz.universe.make_universe` with a loader
that returns the same `Universe` dataclass populated from your CRSP /
yfinance / etc. download.

## References

- Fama & MacBeth (1973), *Risk, Return, and Equilibrium: Empirical Tests*.
- Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*.
- Fama & French (1993), *Common risk factors in the returns on stocks and bonds*.
- Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*.
- Frazzini & Pedersen (2014), *Betting Against Beta*.
