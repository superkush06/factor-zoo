# Changelog

## [0.5.4] - 2026-08-04

### Fixed
- **README said every number in the tour block is pinned by a test, and
  nothing pinned any of them.** `pyproject.toml` sets `testpaths = ["tests"]`
  with no `--doctest-glob`, so README.md was never collected, and three of the
  block's outputs — `array([-3.28, 3.18, 2.9, 2.44, 3.5])`,
  `array([-0.95, 7.22, 7.3, 6.48, 6.45])` and `res.n_periods` → 1247 — appear
  nowhere under `tests/`. The nearest test, `tests/test_recovery.py`, runs a
  different universe (300 × 1000, seed 11, `low_vol(window=120)`) and asserts
  only `t > 2` / `|t| < 2`, so any of the printed values could have drifted
  with CI green. `tests/test_readme_tour.py` now extracts both `pycon` blocks
  and runs them as doctests — 14 examples and 6 — and diffs every line of
  output against a live run. Retyping 1247 as 1248 and 7.22 as 7.23 fails the
  suite; both passed before this commit. `--doctest-glob=*.md` on its own does
  not do the job: doctest reads the closing fence into the last example's
  expected output, and both blocks fail on that alone.
- **"The injected premia (2.4–3.5 bp/day per unit z, roughly 6–9%/yr)"
  described the recovered coefficients, not the injected ones.**
  `DEFAULT_PREMIA` asks for 3, 3 and 4 bp/day per unit exposure for
  value/quality/low_vol, and momentum's entry is a drift dispersion rather
  than a per-unit-z premium at all; 2.44–3.50 is what Fama-MacBeth reads back
  at `n_days=1500, seed=0`. The bullet says recovered now, and the annualised
  range is the measured 6.1–8.8%/yr rather than a widened 6–9.
- **"Sherman-Morrison gives Σ⁻¹ in four lines."** The closure in
  `examples/alpha_handoff.py` that applies Σ⁻¹ is three statements; four is
  the count only if the `def` line is algebra. README and that file's
  docstring both say three, and name the closure.

### Changed
- "Sharpes near 4" is now the measured 3.6 to 5.1. The four sort Sharpes are
  5.09, 4.14, 3.90 and 3.60, and only one of them is near 4.
- "every row clears it" now gives the margin it is asserting: against a bar of
  two standard errors the loaded column's worst is 6.45 and the placebo
  column's worst is 1.39.

## [0.5.3] - 2026-08-03

### Fixed
- **`docs/validation.md` claimed every prose figure is recomputed; three of
  them were not.** The paragraph under row 1 restates the momentum row of
  README's sort table — the −29.07 and +10.36 bucket columns, the 39.43%/yr
  `spread`, and the (1 + 0.3943/252)²⁵² − 1 conversion — and the page said
  `tests/test_validation.py` "recomputes every one of them and looks for it
  here". It looked for them in README only. Four separate edits to the doc's
  copies — +10.36 to +10.37, −29.07 to −29.08, the spread to 39.44%/yr, the
  conversion to 0.3944 — each left the whole suite green. All four are red
  now: the three restatements are asserted against the same live
  `sort_table()` run that pins README's block.
- **The prose quoted row 1's cell with an ASCII hyphen.** The table cell is
  typeset `Q5−Q1` with U+2212, and both the cell diff and the prose lookups
  normalise that character to a hyphen before comparing, so nothing on the
  page could tell the two spellings apart. The sentence uses the page's own
  minus sign now, and a test that reads the file unnormalised keeps it there.

### Changed
- `test_docs_prose_figures_are_recomputed_too` said two of its five figures
  are also in README. One is: the −0.46 momentum/size correlation. The
  docstring says one, and the test asserts the other four are absent from
  README rather than leaving the count as an unchecked claim about a file it
  never opened.

## [0.5.2] - 2026-08-03

### Fixed
- **The factor-sort table is annualised arithmetically, and both pages called
  it compounded.** `examples/factor_backtest.py` prints each bucket's mean
  daily forward return times 252, but README called `spread` "two separately
  compounded annual returns, differenced" and `docs/validation.md` called the
  same columns "Q5's compounded annual return minus Q1's compounded annual
  return". Compounding Q1 and Q5 separately gives −25.24 and +10.91, not the
  −29.07 and +10.36 on the page. Both descriptions now say what the code does,
  and both give the conversion the old text asserted did not exist: the same
  daily mean compounded is (1 + 0.3943/252)²⁵² − 1 = 48.3%/yr, which is
  precisely the +48.3%/yr `docs/validation.md` row 1 quotes for the same book.
  0.5.1's changelog entry reconciling those two figures named the same wrong
  arithmetic.
- **README claimed the figures are byte-reproducible.** They are, on the
  matplotlib build they were drawn with (3.11); regenerating on 3.9 changes
  every one of the four PNGs, and rounds three of the four canvases to 459
  pixels tall instead of 460. Pinning the font family and the PNG metadata
  inside the script cannot reach that, so README and `make_figures.py` now
  claim reproducibility from the script and not identity across builds.
- **`docs/validation.md` said a number could not be typed into the page by
  hand.** True of its two tables, which are diffed cell for cell; not true of
  the prose around them, which quoted the 68% idiosyncratic share, the 0.91
  volatility correlation, the −0.46 momentum/size correlation, the 1.64 and
  1.012 Shanken factors and the README sort table's −29.07 / +10.36 / 39.43
  with nothing checking any of them. All of those are recomputed by
  `tests/test_validation.py` now, and the paragraph says which figures are
  recomputed and which are a rounding of a pinned cell.
- **README's excerpt of the closed-form table pinned only our column.** The
  reference column is as computed as the rest of the page — 1.1317 comes out
  of `shanken_factor`, 0.01910 out of the Spearman identity — so it could have
  drifted in README while staying right in `docs/validation.md`. Both columns
  are pinned now, along with the 0.29% relative error and the 0.73 / 0.27
  calibration deviations README quotes in prose.

### Added
- `factor_backtest.sort_table`, `format_row` and `annualised_compounded`: the
  printed table as data, the one formatter that renders it, and the conversion
  between the two annualisation conventions. `main` prints what `sort_table`
  returns, so README's block is a live run rather than a transcript.
- `validate.volatility_decomposition` and `validate.momentum_size_correlation`,
  and module-level constants for the two Shanken setups. Row 3's and row 4's
  notes and the prose of `docs/validation.md` now read those figures from one
  definition instead of three.
- `test_readme_sort_table_is_a_live_run_and_reconciles_with_row_1` and
  `test_docs_prose_figures_are_recomputed_too`.

### Changed
- `docs/validation.md` gets its typography back in the tables — φ, Φ, ρ, π, γ,
  θ, λ, √, ± and a real minus sign. Pinning the cells in 0.5.1 had forced the
  page to be written in `validate.py`'s terminal ASCII; the comparison now
  reduces the page to that ASCII before diffing, which is the same guarantee
  without the readability cost.

## [0.5.1] - 2026-07-29

### Fixed
- **The `agrees` column of `docs/validation.md` rows 1-4 was a set of Python
  literals**, not a comparison. Negating `momentum` left the momentum row
  printing `ok` next to a premium of −3.18 bp/day, and the test suite stayed
  green. Those four verdicts are now derived from the measured sign and
  t-statistic (`bp > 0 and t > 2`, and the sign of the rank IC for the size
  row), the `(wrong sign)` annotation and the forced leading `+` in the value
  strings are gone, and each row's note opens with a clause taken from its own
  verdict. Corrupting a characteristic now flips the printed verdict and turns
  `tests/test_validation.py` red.
- **Wrong papers for two characteristics in `docs/theory.md`.** `quality_roe`
  is earnings over book equity, which is the q-factor ROE leg of Hou, Xue &
  Zhang (2015) — not Novy-Marx (2013), whose gross-profitability premium is
  gross profits over assets and whose argument is specifically that
  earnings-based measures are the contaminated alternative. `low_vol` is a
  sort on trailing *total* volatility, which is Blitz & van Vliet (2007) — not
  Frazzini & Pedersen (2014), who sort on market beta. Section 7 now says what
  each of the neighbouring papers is actually about.
- `README.md` no longer claims the published table cannot drift: it could,
  because only the booleans were pinned. See below.
- `examples/alpha_handoff.py` said "everything is point-in-time" while setting
  the leverage of both books from full-sample realised volatility. The
  docstring, an inline comment, the printed output and the README now say
  which two columns that affects (`ret %/yr`, `maxDD %`) and which two it does
  not (Sharpe, correlation).
- `README.md` quoted momentum's Q5−Q1 spread as 39.43%/yr while
  `docs/validation.md` quoted 48.3%/yr for the same signal. Both were right
  for what they measured — differencing two compounded annual returns versus
  compounding the long-short leg's mean daily return — and neither said so.
  Both pages now define the arithmetic and point at the other number.

### Added
- `tests/test_validation.py::test_docs_validation_table_is_the_scripts_output`
  parses the two markdown tables in `docs/validation.md` and demands every
  claim, value, reference value, source and verdict be character-for-character
  what a live run of `examples/validate.py` produces. A number typed into the
  page by hand, or one that moved when an estimator changed, now fails CI.
  `test_readme_excerpt_quotes_the_live_numbers` does the same for the seven
  results README reprints.
- `tests/test_properties.py::test_bartlett_weights_and_not_the_clamp_keep_the_variance_positive`.
  The old non-negativity property was satisfied by the `max(s, 0.0)` at the end
  of `newey_west_var` no matter what the kernel weights were: replacing the
  Bartlett weight with 1.0 left every property test green. The new test uses an
  over-differenced series as a foil — the truncated kernel goes negative on
  roughly half the draws — and demands the shipped estimator be *strictly*
  positive there, plus hit the closed form γ₀/(L+1) on a long sample. The
  old test keeps the shift/scale invariance and now says plainly that its
  non-negativity assertion is an output property, not evidence about the
  weights.

## [0.5.0] - 2026-07-27

### Added
- **`docs/validation.md`** — sixteen claims measured against sources outside
  this repository: the sign of the momentum, value and volatility premia in
  Jegadeesh & Titman (1993), Fama & French (1992) and Ang et al. (2006), and
  closed forms for the Gaussian quintile conditional mean, the
  bivariate-normal Spearman identity, the Bartlett long-run variance of an
  AR(1) and an MA(1), the exact overlap inflation, and the Shanken (1992)
  errors-in-variables factor. One row disagrees — `size_factor` gets the sign
  of the size premium wrong here — and the page says why rather than dropping
  it.
- `examples/validate.py` prints every number on that page; the calibration
  half plants a premium at a published magnitude on a panel whose
  characteristic is observed exactly and demands it back within its own
  standard error, with 95.3% coverage measured over 300 replications.
  `tests/test_validation.py` runs the same code under pytest.
- `shanken_factor(premia, factor_cov)` — the `1 + lam' Sigma_f^-1 lam`
  inflation a two-pass regression on *estimated betas* owes its standard
  errors. Measured both ways in the validation table: 1.130 against a
  predicted 1.132 with estimated betas, 1.010 with observed regressors, which
  is the case `fama_macbeth` is in.
- `tests/test_properties.py` — randomised invariants rather than fixtures:
  rank totals are conserved, every score row is a unit z-score, momentum
  ignores the units a price is quoted in, scoring never reorders a
  cross-section, quintile buckets partition the cross-section, flipping a
  signal flips the book, `newey_west_var` is non-negative for every series and
  lag length, Fama-MacBeth is exact on an exact model and linear in the
  premium it is handed, the rank IC is invariant to monotone maps of the
  forward return, and zeroing a premium changes the return panel by exactly
  rank one.
- `examples/alpha_handoff.py` — the library doing its job as one stage of a
  pipeline: trailing-window premia into expected returns, a one-factor
  covariance into Sherman-Morrison, out comes a dollar-neutral book. Risk
  weighting the same four signals buys a third more Sharpe and halves the
  drawdown against an equal-weighted quintile blend.
- `docs/calibration.png` — the estimate against the truth, and the
  distribution of standardised errors against the interval that claims to
  cover them.

### Changed
- `docs/theory.md` now derives the overlap inflation exactly:
  `sqrt(1 + (h-1)(2h-1)/3h)`, roughly `0.82 sqrt(h)`, not `sqrt(h)`. The
  README said `sqrt(h)` while `overlap.png` plotted something visibly below
  it; the figure now draws both curves and the prose matches the measurement.
- `examples/make_figures.py` pins the font family and strips the matplotlib
  version out of the PNG metadata, so a re-run reproduces the committed
  images byte for byte. It no longer requests a font weight DejaVu does not
  have, which was printing `findfont` warnings on a clean install.
- README says which extra `make_figures.py` needs (`.[plot]`) and drops the
  redundant `PYTHONPATH=.` prefix from every example invocation — the
  quickstart already installs the package.

## [0.4.0] - 2026-07-27

### Added
- **Standard errors on the Fama-MacBeth estimate.** `FamaMacBethResult` now
  carries `std_errors` and `n_periods` alongside the coefficients, so the
  premium can be reported with an interval instead of a bare t-stat.
- **Newey-West HAC inference** for overlapping forward returns:
  `newey_west_var` (Bartlett kernel, `lags=0` reproduces the ordinary sample
  variance exactly) and `fama_macbeth(..., hac_lags=h-1)`. At a 21-day
  holding period the iid t-stat on momentum reads 32.9 against a HAC value of
  8.9; `tests/test_standard_errors.py` checks the estimator against the
  analytic long-run variance of an AR(1).
- `forward_returns(returns, horizon)` — compounds the next h days and aligns
  them to date t, replacing the hand-rolled shift every example was doing.
- `examples/recovery_matrix.py` — the recovery/placebo comparison as a
  runnable table rather than only a pytest assertion.
- `examples/make_figures.py` regenerates every README figure from the shipped
  code: `docs/recovery.png` (quintile ladders, compounded spreads, premium
  vs placebo), `docs/overlap.png` (t-stat inflation under overlapping
  horizons), `docs/lookahead.png` (what pooled winsorisation leaks).

### Changed
- README rewritten around the three claims the library can actually defend,
  with every printed number re-run from the shipped code.
- `docs/theory.md` now derives the Fama-MacBeth standard error and the
  Bartlett long-run variance, spells out what identifiability requires of the
  data generating process (including the Ito correction and the
  errors-in-variables attenuation), and cites the replication literature.
- `examples/factor_backtest.py` reports the full quintile ladder and flags
  monotonicity; `examples/momentum_ic.py` covers all six characteristics and
  shows what an unpriced and a contaminated one look like.

### Removed
- `examples/render_hero.py` and `docs/demo.png`, superseded by
  `examples/make_figures.py`.

## [0.3.0] - 2026-07-09

### Fixed
- **Point-in-time winsorisation**: factor scores were clipped against
  quantiles of the whole panel, so a day-t score depended on data from
  day t+1 onward (up to ~0.9 z-units of drift for momentum). Quantiles
  are now computed per cross-section; `tests/test_no_lookahead.py`
  asserts truncation invariance for every factor, `rolling_ic`, and
  `quintile_sort_returns`.
- **One returns convention**: `make_universe` built prices from
  log-returns while everything else consumed simple returns (~0.4%/day
  self-disagreement). The `Universe` contract is now simple returns with
  `returns[0] = NaN` and `prices[0] = 100`, compounded exactly;
  `universe_from_prices` matches (no more fake 0% on day 0).
- **Average-rank tie handling** in Spearman IC and quantile sorts:
  winsorised scores always tie in the clipped tails, exactly where
  long-short portfolios are formed; results no longer depend on ticker
  order.
- Factor warm-up rows no longer spew nanmean/nanstd RuntimeWarnings.

### Added
- **Identifiable premia** in `make_universe`: momentum via a persistent
  AR(1) expected-return drift, low-vol via genuinely-quieter stocks
  (premium loads on true total volatility), quality decoupled from value
  (earnings derived from book value), with an Ito correction so premia
  are arithmetic. `premia={...}` overrides `DEFAULT_PREMIA`; zero an
  entry for a placebo universe.
- **Recovery/placebo acceptance matrix** (`tests/test_recovery.py`):
  each factor must show t > 2 for its own loaded premium and |t| < 2
  when unpriced.
- `average_ranks` exported; README numbers and `docs/demo.png`
  regenerated from the shipped code.

## [0.2.0] - 2026-06-XX

### Added
- **Real-data loaders** (`universe_from_prices`, `load_prices_csv`): build a
  `Universe` from a real wide price panel so the library runs on your own data,
  not just the synthetic generator. Pure stdlib CSV parsing, no pandas.
- `crossect.rolling_ic` — trailing-window mean information coefficient.
- Factor **tearsheet hero chart** (`examples/render_hero.py` → `docs/demo.png`):
  cumulative long-short factor returns + rolling momentum IC.

## [0.1.0] - 2026-05-XX

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
