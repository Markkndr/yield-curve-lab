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
from scipy.optimize import minimize_scalar

DIEBOLD_LI_LAMBDA = 0.0609

# Search range for an optimized lam. The curvature loading peaks at about
# 1.79 / lam months, so these bounds keep the hump between 3 months and 15
# years; outside that range the slope and curvature loadings become nearly
# collinear over the maturities we observe and the betas explode.
LAMBDA_BOUNDS = (0.01, 0.6)
LAMBDA_GRID_SIZE = 60

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
    return pd.DataFrame(
        _loadings_matrix(np.asarray(maturities, dtype=float), lam),
        columns=FACTOR_COLUMNS,
        index=pd.Index(maturities, name="maturity_months"),
    )


def _loadings_matrix(tau: np.ndarray, lam: float) -> np.ndarray:
    """The loadings as a plain `(len(tau), 3)` array, for the fitting loops."""
    x = lam * tau
    safe_x = np.where(x == 0, 1.0, x)  # avoid 0/0; the limit is patched in below
    slope = np.where(x == 0, 1.0, -np.expm1(-safe_x) / safe_x)  # expm1: accurate for small x
    return np.column_stack([np.ones_like(tau), slope, slope - np.exp(-x)])


def _observed_points(
    maturities: Sequence[float] | np.ndarray,
    yields: Sequence[float] | np.ndarray,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return maturities, yields and the mask of points where both are present."""
    tau = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    if tau.shape != y.shape or tau.ndim != 1:
        raise ValueError("maturities and yields must be 1-D and the same length")
    observed = ~np.isnan(tau) & ~np.isnan(y)
    if observed.sum() < min_points:
        raise ValueError(f"need at least {min_points} observed points to fit, got {observed.sum()}")
    return tau, y, observed


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
    tau, y, observed = _observed_points(maturities, yields, min_points=3)

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


def _sse(lam: float, tau: np.ndarray, y: np.ndarray) -> float:
    """Sum of squared residuals of the OLS fit at `lam` (the profiled objective)."""
    X = _loadings_matrix(tau, lam)
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sum((y - X @ betas) ** 2))


def fit_ns(
    maturities: Sequence[float] | np.ndarray,
    yields: Sequence[float] | np.ndarray,
    bounds: tuple[float, float] = LAMBDA_BOUNDS,
) -> NSFit:
    """Fit the betas and lam, with lam searched inside `bounds`.

    For any given lam the best betas come from OLS, so only lam needs a
    numerical search: the objective is the sum of squared residuals of the
    fixed-lam fit. That objective can have several local minima, so it is
    first evaluated on a log-spaced grid and then refined with a bounded
    scalar minimizer around the best grid point.

    The grid includes `DIEBOLD_LI_LAMBDA` when it lies inside `bounds`, so the
    result never fits worse than `fit_ns_fixed` at the default lam. Needs at
    least four observed points, one more than the three betas.
    """
    low, high = bounds
    if not 0 < low < high:
        raise ValueError(f"bounds must satisfy 0 < low < high, got {bounds}")
    tau, y, observed = _observed_points(maturities, yields, min_points=4)
    tau_obs, y_obs = tau[observed], y[observed]

    grid = np.geomspace(low, high, LAMBDA_GRID_SIZE)
    if low < DIEBOLD_LI_LAMBDA < high:
        grid = np.sort(np.append(grid, DIEBOLD_LI_LAMBDA))
    sse = np.array([_sse(lam, tau_obs, y_obs) for lam in grid])
    best = int(sse.argmin())

    bracket = (grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)])
    refined = minimize_scalar(
        _sse, bounds=bracket, args=(tau_obs, y_obs), method="bounded", options={"xatol": 1e-10}
    )
    lam = refined.x if refined.fun < sse[best] else grid[best]
    return fit_ns_fixed(maturities, yields, lam)
