"""factor-zoo — equity factors on synthetic data."""

from .crossect import (
    FamaMacBethResult,
    fama_macbeth,
    rank_information_coefficient,
    rolling_ic,
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
from .universe import Universe, make_universe

__version__ = "0.2.0"
__all__ = [
    "Universe", "make_universe",
    "universe_from_prices", "load_prices_csv",
    "momentum", "short_reversal", "value_btm", "size_factor",
    "quality_roe", "low_vol",
    "quintile_sort_returns", "long_short_return", "cumulative",
    "sharpe_annualised",
    "FamaMacBethResult", "fama_macbeth", "rank_information_coefficient",
    "rolling_ic",
    "__version__",
]
