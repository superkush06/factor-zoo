# factor-zoo

[![ci](https://github.com/superkush06/factor-zoo/actions/workflows/ci.yml/badge.svg)](https://github.com/superkush06/factor-zoo/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Classical equity factor reproductions on a **synthetic universe**:
> momentum, value, size, quality, low-vol, mean-reversion. Daily
> portfolio sorts, Fama-MacBeth cross-sectional regressions, and
> rank-IC analysis. Pure NumPy.

> **Why synthetic?** Real research uses CRSP/Compustat (license-walled).
> Synthetic data with known factor structure lets the *pipeline*
> recover known premia — proving the code is right, not that the
> factors work in live markets. Swap in a loader for real data and
> the rest of the pipeline is unchanged.

## TL;DR

```python
from fz import make_universe, momentum, value_btm, fama_macbeth

u = make_universe(n_stocks=300, n_days=1500, seed=0)
import numpy as np
fwd = np.full_like(u.returns, np.nan)
fwd[:-1] = u.returns[1:]

mom = momentum(u, lookback=252, skip=21)
val = value_btm(u)
res = fama_macbeth([mom, val], fwd, add_intercept=True)
print(res.coefficients, res.t_stats)
```

## What's inside

- `universe.make_universe` — generates a panel with a known factor
  structure (so you can verify the pipeline recovers it).
- `factors.{momentum, short_reversal, value_btm, size_factor,
  quality_roe, low_vol}` — characteristic constructors with winsorise
  + cross-sectional z-score.
- `portfolio.{quintile_sort_returns, long_short_return,
  cumulative, sharpe_annualised}` — daily quintile sort + L-S aggregation.
- `crossect.{fama_macbeth, rank_information_coefficient}` — daily
  cross-sectional regressions and IC computation.

## Example output

`PYTHONPATH=. python3 examples/factor_backtest.py`:

```
factor       L-S Sharpe   L-S annual ret
momentum          0.142           0.0173
value             1.823           0.0524
quality           1.654           0.0489
low_vol           1.215           0.0367
```

`examples/fama_macbeth.py`:

```
factor           coef     t-stat
intercept    0.000302       2.94
mom         -0.000043      -0.43
val          0.000091       9.21
qual         0.000095       9.62
lvol         0.000088       8.83

Mean daily R^2 across days: 0.0427
```

## Theory

See [`docs/theory.md`](docs/theory.md) for Fama-MacBeth, portfolio sorts,
IC mechanics, and references to the original factor papers.

## Install

```bash
git clone https://github.com/superkush06/factor-zoo.git
cd factor-zoo
pip install -e ".[dev]"
pytest
```

## Roadmap

- [ ] Real-data loader (CRSP / yfinance / norgate).
- [ ] Multi-factor regression with Newey-West HAC standard errors.
- [ ] Rolling-window factor IC + alpha-decay visualisation.
- [ ] Risk-parity factor combination.

## License

MIT — see [LICENSE](LICENSE).
