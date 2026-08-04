"""The validation table has to keep being true.

`examples/validate.py` is what produces every number in `docs/validation.md`.
This module runs the same code and checks the published page against it, so
the table cannot quietly drift away from the library: if an estimator changes,
either the page changes with it or this test goes red.

Four things are pinned, because pinning only the last one was not enough:

* **the cells.** `test_docs_validation_table_is_the_scripts_output` parses the
  two markdown tables in `docs/validation.md` and demands every claim, value,
  reference value, source and verdict be character-for-character what a live
  run of `validate.py` produces, once the page's typography is reduced to the
  ASCII a terminal gets (`φ` for `phi`, `−` for `-`, `±` for `+-`). Edit a
  number into the page by hand, or let an estimator drift, and the comparison
  fails. That reduction is one-way, so
  `test_doc_prose_keeps_the_pages_own_minus_sign` guards the one character it
  would otherwise let the page lose.
* **the prose.** The paragraphs between the tables quote figures no column
  holds. `test_docs_prose_figures_are_recomputed_too` recomputes them, and
  `test_readme_sort_table_is_a_live_run_and_reconciles_with_row_1` recomputes
  the sort table the prose reconciles row 1 against — every figure of it that
  either page quotes, not only the ones README prints.
* **README's excerpt.** `test_readme_excerpt_quotes_the_live_numbers` covers
  both columns of the six closed-form rows README reprints. The reference
  column is computed too — 1.1317 comes out of `shanken_factor` — so it can
  drift like any other number.
* **the verdicts.** The closed-form checks must all agree. The literature
  checks are allowed to disagree — one of them does, deliberately — so the
  pattern of agreement is pinned by name. Those verdicts are computed inside
  `validate.py` from the measured premia, so breaking a characteristic flips
  one and this test notices.
"""

import math
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "examples"))

from factor_backtest import (  # noqa: E402
    HEADER,
    annualised_compounded,
    format_row,
    sort_table,
)
from validate import (  # noqa: E402
    MARKET_LAM,
    MARKET_SIG_F,
    N_DAYS,
    N_STOCKS,
    SEED,
    SHANKEN_LAM,
    SHANKEN_SIG_F,
    all_checks,
    bartlett_inflation,
    momentum_size_correlation,
    normal_quintile_means,
    planted_panel,
    premium_for_annual_spread,
    volatility_decomposition,
)

from fz import (  # noqa: E402
    fama_macbeth,
    forward_returns,
    make_universe,
    newey_west_var,
    shanken_factor,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "validation.md"
README = ROOT / "README.md"
_NUMBERED_ROW = re.compile(r"^\|\s*\d+\s*\|")

# What README reprints from each result: the figures it quotes from our value,
# from the reference value, and from the row's note. Pinning only `ours` left
# the reference column of README's closed-form table free to drift -- 1.1317 in
# particular is computed at run time from `shanken_factor`, not a constant.
_README_QUOTES = {
    "Quintile bucket means equal the conditional means of the score":
        {"ours": ("-1.3986", "+1.3979"), "ref": ("-1.3998", "+1.3998")},
    "Rank IC of a bivariate normal matches the Spearman identity":
        {"ours": ("0.02000", "0.04590", "0.09463"),
         "ref": ("0.01910", "0.04775", "0.09553")},
    "Bartlett long-run variance of an AR(1)":
        {"ours": ("4.4947",), "ref": ("4.4817",), "note": ("0.29%",)},
    "t-stat inflation from h-day overlap, at h = 5, 21, 63":
        {"ours": ("6.488",), "ref": ("6.481",)},
    "Estimated betas inflate the true sampling error of a two-pass premium":
        {"ours": ("1.1298",), "ref": ("1.1317",)},
    "Observed regressors do not need the correction":
        {"ours": ("1.0103",), "ref": ("1.0000",)},
    "The 95% Fama-MacBeth interval covers the truth 95% of the time":
        {"ours": ("0.953", "286/300")},
    "A premium planted at the Jegadeesh-Titman magnitude (12.01%/yr) "
    "comes back": {"ours": ("0.73",)},
    "A universe with no premium in it yields no premium": {"ours": ("0.27",)},
}


def _flat(path: pathlib.Path) -> str:
    """A page as one long line, with the unicode minus normalised.

    Prose checks below look for a phrase with a number in it; whether the
    published paragraph happens to wrap in the middle of that phrase is not
    something a test should have an opinion about.
    """
    return " ".join(path.read_text(encoding="utf-8").replace("−", "-").split())


# The published tables are typeset; `validate.py` prints ASCII to a terminal.
# These substitutions undo the typesetting and nothing else, so a cell still
# has to match the script character for character afterwards -- a moved digit
# is still a failure -- but the page is allowed to spell a Greek letter with a
# Greek letter instead of writing out `phi`.
_TYPOGRAPHY = {
    "−": "-",       # U+2212 minus, not a hyphen
    "±": "+-",
    "√": "sqrt",
    "Σ": "sum",
    "φ": "phi", "Φ": "Phi", "ρ": "rho", "π": "pi",
    "γ": "gamma", "θ": "theta", "λ": "lam",
}


def _plain(cell: str) -> str:
    for typeset, ascii_ in _TYPOGRAPHY.items():
        cell = cell.replace(typeset, ascii_)
    return cell


def _doc_table_rows() -> list[list[str]]:
    """Every numbered row of every markdown table in docs/validation.md.

    Cells are split on unescaped pipes, so a cell may contain a literal `|`
    written as `\\|` (row 5's `E[z \\| quintile]` does), and are then reduced
    to the ASCII the script prints.
    """
    rows = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if not _NUMBERED_ROW.match(line):
            continue
        cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
        rows.append([_plain(c.strip().replace("\\|", "|")) for c in cells])
    return rows


@pytest.fixture(scope="module")
def checks():
    lit, closed = all_checks()
    return {c.claim: c for c in lit + closed}, lit, closed


def test_docs_validation_table_is_the_scripts_output(checks):
    """The published page is the script's output, cell for cell.

    Verdicts alone are not enough: a number typed into the page by hand, or a
    number that quietly moved when an estimator changed, is invisible to a
    boolean assertion. So compare the strings.
    """
    _, lit, closed = checks
    expected = lit + closed
    rows = _doc_table_rows()
    assert len(rows) == len(expected), (
        f"docs/validation.md has {len(rows)} numbered table rows, "
        f"validate.py produces {len(expected)} checks"
    )
    for i, (cells, c) in enumerate(zip(rows, expected, strict=True), start=1):
        where = f"docs/validation.md row {i}"
        assert len(cells) == 6, f"{where}: expected 6 columns, got {len(cells)}"
        assert cells[0] == str(i), f"{where}: rows must be numbered in order"
        assert cells[1] == c.claim, f"{where}: claim"
        assert cells[2] == c.ours, f"{where}: our value"
        assert cells[3] == c.ref_value, f"{where}: reference value"
        assert cells[4] == c.reference, f"{where}: source"
        verdict = cells[5].strip("*").lower()
        assert verdict in ("yes", "no"), f"{where}: verdict must be yes or no"
        assert (verdict == "yes") is c.agrees, f"{where}: verdict"


def test_doc_prose_keeps_the_pages_own_minus_sign():
    """The prose is typeset like the tables it quotes, not like the script.

    `_plain` rewrites U+2212 to a hyphen before the tables are diffed, and
    `_flat` does the same before a prose phrase is looked up, so neither check
    can tell the two characters apart. Nothing else on the page would notice a
    sentence quoting `Q5-Q1` back at a table cell that reads `Q5−Q1`.
    """
    raw = DOC.read_text(encoding="utf-8")
    assert "Q5−Q1" in raw
    assert "Q5-Q1" not in raw, (
        "docs/validation.md quotes the quintile spread with an ASCII hyphen "
        "somewhere; the page writes it with U+2212"
    )


def test_readme_excerpt_quotes_the_live_numbers(checks):
    """README reprints nine of the sixteen results in its own words.

    Both columns of its closed-form table are checked, not just ours: the
    reference column is as computed as the rest of the page (1.1317 comes out
    of `shanken_factor`, 0.01910 out of the Spearman identity), so a stale
    reference is just as much a drift as a stale estimate. The check runs both
    ways -- the number must still be what `validate.py` prints, and it must
    still be in README.
    """
    by_claim, _, _ = checks
    text = README.read_text(encoding="utf-8").replace("−", "-")
    for claim, quoted in _README_QUOTES.items():
        c = by_claim[claim]
        for column, printed in (("ours", c.ours), ("ref", c.ref_value),
                                ("note", c.note)):
            for num in quoted.get(column, ()):
                assert num in printed, (
                    f"README quotes {num} from the {column} column of "
                    f"{claim!r}, but validate.py now prints {printed!r}"
                )
                assert num in text, (
                    f"README no longer quotes {num} for {claim!r}"
                )


def test_docs_prose_figures_are_recomputed_too(checks):
    """The tables are diffed cell for cell; the prose is not a table.

    Five figures are quoted in the prose of docs/validation.md that no column
    holds: the idiosyncratic share and the volatility correlation behind row
    3, the momentum/size correlation behind row 4, and the two Shanken factors
    behind rows 15-16. They are recomputed here from the same constants
    `validate.py` uses, so the paragraph cannot keep a number the estimators
    have moved away from.

    Exactly one of the five is repeated in README -- the -0.46 momentum/size
    correlation, in the paragraph under the IC table -- and that copy is
    asserted too. The other four appear nowhere in README.
    """
    doc, readme = _flat(DOC), _flat(README)
    u = make_universe(n_stocks=N_STOCKS, n_days=N_DAYS, seed=SEED)

    idio_share, vol_corr = volatility_decomposition(u)
    assert f"variance is {idio_share:.0%} of total variance on average" in doc
    assert f"correlate {vol_corr:.2f} across names" in doc

    corr = momentum_size_correlation(u)
    assert f"correlated {corr:.2f} with the momentum score" in doc
    assert f"correlates {corr:.2f} with the momentum score" in readme

    extreme = shanken_factor([SHANKEN_LAM], [[SHANKEN_SIG_F ** 2]])
    market = shanken_factor([MARKET_LAM], [[MARKET_SIG_F ** 2]])
    assert f"λ` = {extreme:.2f}) so the effect is visible" in doc
    assert (f"{MARKET_LAM:.1%}/month against {MARKET_SIG_F:.1%}/month "
            f"volatility — the factor is {market:.3f}") in doc

    # The remaining four live on this page and nowhere else, which is why only
    # the correlation is looked up in README above. Copy one into README and
    # this goes red rather than leaving the copy unpinned.
    for page_only in (f"{idio_share:.0%}", f"{vol_corr:.2f}",
                      f"{extreme:.2f}", f"{market:.3f}"):
        assert page_only not in readme, (
            f"README now quotes {page_only} too; it needs a check of its own "
            f"here, the way the momentum/size correlation has one"
        )


def test_readme_sort_table_is_a_live_run_and_reconciles_with_row_1(checks):
    """README's factor-sort block, and both pages' prose about it.

    The two figures are one mean daily return under two annualisations: the
    printed table multiplies by 252, row 1 compounds. Pinning the block line
    for line, and then converting one number into the other, is what stops the
    two pages from describing the same book with incompatible arithmetic
    again.

    `docs/validation.md` restates the momentum row in words rather than
    reprinting the block -- the two bucket columns, the spread, and the
    conversion between the two annualisations -- and those restatements are
    recomputed here as well. Checking them only where README prints them left
    the doc's copies loose: `+10.36` there could become `+10.37`, or the
    `spread` become 39.44%/yr, with the whole suite still green.
    """
    by_claim, _, _ = checks
    readme = README.read_text(encoding="utf-8").replace("−", "-")
    assert HEADER in readme
    rows = {r.name: r for r in sort_table()}
    for r in rows.values():
        assert format_row(r) in readme, (
            f"README's sort table no longer matches a live run at {r.name!r}; "
            f"expected {format_row(r)!r}"
        )

    mom = rows["momentum"]
    quoted = f"{mom.spread:.2f}"                       # 39.43, the README cell
    conversion = f"(1 + {float(quoted) / 100:.4f}/252)"
    compounded = f"+{annualised_compounded(mom.spread):.1f}%/yr"   # +48.3%/yr
    assert conversion in readme
    assert compounded in readme

    # The doc restates all four in prose: row 1's compounded figure, which it
    # already had to quote, plus the spread, the bucket columns and the
    # conversion, which nothing looked for here before. `_flat` is what lets a
    # phrase be found across a line wrap, and normalises the page's U+2212 to
    # the hyphen a float formats with.
    doc = _flat(DOC)
    for phrase in (compounded,
                   f"`spread` of {quoted}%/yr",
                   f"({mom.q1:+.2f} and {mom.q5:+.2f})",
                   conversion):
        assert phrase in doc, (
            f"docs/validation.md no longer says {phrase!r} about the momentum "
            f"row of the sort table"
        )

    row_1 = by_claim["A 12-month-minus-1-month momentum characteristic earns "
                     "a positive premium"]
    assert compounded + " compounded" in row_1.ours, (
        f"row 1 prints {row_1.ours!r}, which no longer compounds to the "
        f"sort table's spread of {quoted}%/yr"
    )


def test_every_closed_form_check_agrees(checks):
    _, _, closed = checks
    failed = [c.claim for c in closed if not c.agrees]
    assert not failed, f"closed-form checks that stopped agreeing: {failed}"


def test_literature_agreement_pattern_is_the_documented_one(checks):
    _, lit, _ = checks
    verdicts = {c.claim: c.agrees for c in lit}
    disagreeing = [k for k, v in verdicts.items() if not v]
    assert disagreeing == ["Small stocks earn more than large ones"], (
        "docs/validation.md documents exactly one disagreement with the "
        f"published cross-section; found {disagreeing}"
    )


def test_the_table_is_not_empty(checks):
    _, lit, closed = checks
    assert len(lit) >= 4 and len(closed) >= 10


# --- the same ground truths, recomputed here rather than trusted ----------
def test_normal_quintile_means_are_symmetric_and_sum_to_zero():
    """E[z | quintile] for a standard normal: the buckets are a partition, so
    the five conditional means (equal weight) must average to E[z] = 0, and
    symmetry of the density forces q1 = -q5."""
    m = normal_quintile_means()
    assert m.mean() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(m, -m[::-1], atol=1e-12)
    assert m[0] == pytest.approx(-1.3998096, abs=1e-6)   # phi(z_0.2) / 0.2


def test_planted_premium_is_inside_its_own_interval():
    lam = premium_for_annual_spread(0.1201)          # Jegadeesh-Titman scale
    z, r = planted_panel(lam, seed=77, n_days=1200, n_stocks=300)
    res = fama_macbeth([z], forward_returns(r))
    assert abs(res.coefficients[1] - lam) < 2.0 * res.std_errors[1]


def test_bartlett_inflation_matches_a_simulated_overlap():
    """An h-day overlapping slope series is an equally weighted MA(h-1) of
    iid shocks; its Bartlett-at-(h-1) inflation has a closed form."""
    rng = np.random.default_rng(9)
    u = rng.standard_normal(200_000 + 70)
    for h in (5, 21, 63):
        lam_t = np.convolve(u, np.ones(h) / h, mode="valid")
        got = math.sqrt(newey_west_var(lam_t, h - 1) / newey_west_var(lam_t, 0))
        assert got == pytest.approx(bartlett_inflation(h), rel=0.02)
        assert got < math.sqrt(h)        # Bartlett always discounts
