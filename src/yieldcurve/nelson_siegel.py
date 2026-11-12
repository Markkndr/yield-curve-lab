"""Nelson-Siegel yield curve model, implemented from scratch.

    y(tau) = b0 + b1 * (1 - exp(-lam*tau)) / (lam*tau)
                + b2 * ((1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau))

Following Diebold and Li (2006), maturities `tau` are in months and lam is
fixed at 0.0609, which puts the peak of the curvature loading at 30 months.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

DIEBOLD_LI_LAMBDA = 0.0609

FACTOR_COLUMNS = ["level", "slope", "curvature"]


@dataclass(frozen=True)
class NSFit:
    """A Nelson-Siegel fit of one curve.

    `betas` holds b0, b1, b2 under the labels level, slope and curvature, in the
    units of the yields. With these loadings the short rate is b0 + b1, so the
    slope beta is negative when the curve slopes upward. `fitted` and
    `residuals` (observed minus fitted) are indexed by the maturities passed in;
    `residuals` is NaN where no yield was observed.
    """

    betas: pd.Series
    lam: float
    fitted: pd.Series
    residuals: pd.Series

    @property
    def rmse(self) -> float:
        """Root mean squared residual over the observed points."""
        return float(np.sqrt(np.nanmean(self.residuals.to_numpy() ** 2)))


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


def fit_ns_fixed(
    maturities: Sequence[float] | np.ndarray,
    yields: Sequence[float] | np.ndarray,
    lam: float = DIEBOLD_LI_LAMBDA,
) -> NSFit:
    """Fit the three betas by ordinary least squares with lam held fixed.

    With lam fixed the model is linear in the betas, so the fit is a regression
    of the yields on the three loadings. Points where the maturity or the yield
    is NaN are left out; at least three must remain.
    """
    tau = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    if tau.shape != y.shape or tau.ndim != 1:
        raise ValueError("maturities and yields must be 1-D and the same length")
    observed = ~np.isnan(tau) & ~np.isnan(y)
    if observed.sum() < len(FACTOR_COLUMNS):
        raise ValueError(f"need at least 3 observed points to fit, got {observed.sum()}")

    loadings = ns_loadings(tau, lam)
    X = loadings.to_numpy()
    betas, *_ = np.linalg.lstsq(X[observed], y[observed], rcond=None)
    fitted = pd.Series(X @ betas, index=loadings.index, name="fitted")
    return NSFit(
        betas=pd.Series(betas, index=FACTOR_COLUMNS, name="beta"),
        lam=float(lam),
        fitted=fitted,
        residuals=(y - fitted).rename("residual"),
    )
