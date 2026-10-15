"""Empirical level, slope and curvature of the yield curve.

Diebold and Li (2006) proxies built from three maturities, used as benchmarks
for the Nelson-Siegel factors.
"""

import pandas as pd

from yieldcurve.spreads import spread

FACTOR_COLUMNS = ["level", "slope", "curvature"]


def empirical_factors(table: pd.DataFrame) -> pd.DataFrame:
    """Return level, slope and curvature for each date in `table`.

    level = mean of 3M, 2Y and 10Y; slope = 10Y - 3M; curvature = 2*2Y - 3M - 10Y.
    """
    short, mid, long = table["3M"], table["2Y"], table["10Y"]
    return pd.DataFrame(
        {
            "level": (short + mid + long) / 3,
            "slope": spread(table, "10Y", "3M"),
            "curvature": 2 * mid - short - long,
        }
    )
