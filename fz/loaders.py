"""Build a `Universe` from real market data instead of the synthetic generator.

The synthetic `make_universe` is great for validating the pipeline (it injects
known factor premia you can recover), but the whole point of a factor library
is to run it on *your* data. These loaders turn a plain price panel — the kind
you get from yfinance, Norgate, CRSP, or a CSV export — into the same
`Universe` the rest of `fz` consumes.

Price-based factors (momentum, short reversal, low-vol) work from prices alone.
The fundamental factors (value, quality, size) additionally need
`market_cap` / `book_value` / `earnings`; pass them if you have them, otherwise
those factors return NaN and the rest still work.
"""

from __future__ import annotations

import numpy as np

from .universe import Universe


def universe_from_prices(
    prices: np.ndarray,
    *,
    tickers: list[str] | None = None,
    dates: np.ndarray | None = None,
    market_cap: np.ndarray | None = None,
    book_value: np.ndarray | None = None,
    earnings: np.ndarray | None = None,
) -> Universe:
    """Assemble a `Universe` from a (n_days, n_stocks) price panel.

    Returns are simple period-over-period pct changes; row 0 is NaN
    (there is no prior price), matching `make_universe`'s contract. Factor
    warm-up windows therefore see a missing value instead of a fake 0% day.
    Fundamental panels default to NaN (their factors then yield NaN scores).
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 2:
        raise ValueError("prices must be a 2-D (n_days, n_stocks) array")
    n_days, n_stocks = prices.shape
    if n_days < 2:
        raise ValueError("need at least 2 rows of prices to form returns")

    returns = np.full_like(prices, np.nan)
    returns[1:] = prices[1:] / prices[:-1] - 1.0

    def _panel(x):
        if x is None:
            return np.full((n_days, n_stocks), np.nan)
        x = np.asarray(x, dtype=float)
        if x.shape != (n_days, n_stocks):
            raise ValueError(f"fundamental panel must be {(n_days, n_stocks)}")
        return x

    if tickers is None:
        tickers = [f"A{i:04d}" for i in range(n_stocks)]
    if len(tickers) != n_stocks:
        raise ValueError("len(tickers) must equal number of price columns")

    return Universe(
        prices=prices,
        returns=returns,
        market_cap=_panel(market_cap),
        book_value=_panel(book_value),
        earnings=_panel(earnings),
        dates=dates if dates is not None else np.arange(n_days),
        tickers=list(tickers),
    )


def load_prices_csv(path: str, *, has_date_column: bool = True) -> Universe:
    """Load a wide price CSV into a `Universe`.

    Expected layout (one row per date, one column per ticker):

        date,AAPL,MSFT,GOOG
        2020-01-02,75.1,160.6,68.4
        2020-01-03,74.4,158.6,68.0
        ...

    With `has_date_column=False`, every column is treated as a ticker price.
    Pure stdlib parsing — no pandas dependency.
    """
    rows: list[list[str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([c.strip() for c in line.split(",")])
    if len(rows) < 2:
        raise ValueError("CSV needs a header plus at least one data row")

    header = rows[0]
    start = 1 if has_date_column else 0
    tickers = header[start:]
    dates_list, price_rows = [], []
    for r in rows[1:]:
        if has_date_column:
            dates_list.append(r[0])
        price_rows.append([float(v) for v in r[start:]])

    prices = np.array(price_rows, dtype=float)
    dates = np.array(dates_list) if has_date_column else np.arange(len(price_rows))
    return universe_from_prices(prices, tickers=tickers, dates=dates)
