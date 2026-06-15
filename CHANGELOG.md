# Changelog

## [0.2.0] - 2026-06-XX

### Added
- **Real-data loaders** (`universe_from_prices`, `load_prices_csv`): build a
  `Universe` from a real wide price panel so the library runs on your own data,
  not just the synthetic generator. Pure stdlib CSV parsing, no pandas.
- `crossect.rolling_ic` — trailing-window mean information coefficient.
- Factor **tearsheet hero chart** (`examples/render_hero.py` → `docs/demo.png`):
  cumulative long-short factor returns + rolling momentum IC.

## [0.1.0] - 2027-01-XX

### Added
- `Universe` dataclass + `make_universe` synthetic data generator with
  injected market, value, size, quality, low-vol structure.
- Factor characteristic functions: `momentum`, `short_reversal`,
  `value_btm`, `size_factor`, `quality_roe`, `low_vol`. All
  winsorised + cross-sectionally z-scored.
- Portfolio sort utilities: `quintile_sort_returns`, `long_short_return`,
  `cumulative`, `sharpe_annualised`.
- `fama_macbeth` — daily cross-sectional regressions with point estimate
  and naive-OLS t-stat aggregation.
- `rank_information_coefficient` — per-day Spearman IC.
- Examples: 4-factor portfolio backtest, Fama-MacBeth multi-factor
  regression, rolling IC for momentum.
- CI on Python 3.11 + 3.12.
