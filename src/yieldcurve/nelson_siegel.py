"""Nelson-Siegel yield curve model, implemented from scratch.

    y(tau) = b0 + b1 * (1 - exp(-lam*tau)) / (lam*tau)
                + b2 * ((1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau))

Following Diebold and Li (2006), maturities `tau` are in months and lam is
fixed at 0.0609, which puts the peak of the curvature loading at 30 months.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

DIEBOLD_LI_LAMBDA = 0.0609

FACTOR_COLUMNS = ["level", "slope", "curvature"]


def ns_loadings(
    maturities: Sequence[float] | np.ndarray, lam: float = DIEBOLD_LI_LAMBDA
) -> pd.DataFrame:
    """Return the level, slope and curvature loadings for each maturity (in months).

    The slope and curvature loadings are taken at their limits (1 and 0) for a
    maturity of exactly zero. NaN maturities give NaN slope and curvature loadings.
    """
    tau = np.asarray(maturities, dtype=float)
    x = lam * tau
    safe_x = np.where(x == 0, 1.0, x)  # avoid 0/0; the limit is patched in below
    slope = np.where(x == 0, 1.0, -np.expm1(-safe_x) / safe_x)  # expm1: accurate for small x
    curvature = slope - np.exp(-x)
    return pd.DataFrame(
        {"level": np.ones_like(tau), "slope": slope, "curvature": curvature},
        index=pd.Index(maturities, name="maturity_months"),
    )
