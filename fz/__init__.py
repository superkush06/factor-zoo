"""factor-zoo — equity factors on synthetic data."""

from .crossect import (
    FamaMacBethResult,
    average_ranks,
    fama_macbeth,
    forward_returns,
    newey_west_var,
    rank_information_coefficient,
    rolling_ic,
    shanken_factor,
)
from .factors import (
    low_vol,
    momentum,
    quality_roe,
    short_reversal,
    size_factor,
    value_btm,
)
from .loaders import load_prices_csv, universe_from_prices
from .portfolio import (
    cumulative,
    long_short_return,
    quintile_sort_returns,
    sharpe_annualised,
)
from .universe import DEFAULT_PREMIA, Universe, make_universe

__version__ = "0.5.3"
__all__ = [
    "Universe", "make_universe", "DEFAULT_PREMIA",
    "universe_from_prices", "load_prices_csv",
    "momentum", "short_reversal", "value_btm", "size_factor",
    "quality_roe", "low_vol",
    "quintile_sort_returns", "long_short_return", "cumulative",
    "sharpe_annualised",
    "FamaMacBethResult", "fama_macbeth", "forward_returns",
    "newey_west_var", "shanken_factor", "rank_information_coefficient",
    "rolling_ic", "average_ranks",
    "__version__",
]
