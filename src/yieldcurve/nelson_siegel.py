"""Nelson-Siegel yield curve model, implemented from scratch.

    y(tau) = b0 + b1 * (1 - exp(-lam*tau)) / (lam*tau)
                + b2 * ((1 - exp(-lam*tau)) / (lam*tau) - exp(-lam*tau))

Following Diebold and Li (2006), maturities `tau` are in months and lam is
fixed at 0.0609, which puts the peak of the curvature loading at 30 months.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from yieldcurve.config import DATA_DIR
from yieldcurve.data import MATURITY_YEARS, load_us_yields

logger = logging.getLogger(__name__)

DIEBOLD_LI_LAMBDA = 0.0609

# Search range for an optimized lam. The curvature loading peaks at about
# 1.79 / lam months, so these bounds keep the hump between 3 months and 15
# years; outside that range the slope and curvature loadings become nearly
# collinear over the maturities we observe and the betas explode.
LAMBDA_BOUNDS = (0.01, 0.6)
LAMBDA_GRID_SIZE = 60

FACTOR_COLUMNS = ["level", "slope", "curvature"]
# Columns of a daily fit table: the betas, the lam used, and the fit quality.
DAILY_FIT_COLUMNS = [*FACTOR_COLUMNS, "lam", "rmse_bp", "n_points"]

# The 1M Treasury starts on this day, so every US maturity exists from here on.
FULL_CURVE_START = "2001-07-31"
US_NS_FACTORS_PATH = DATA_DIR / "us_ns_factors.parquet"


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


def maturity_months(labels: Sequence[str]) -> np.ndarray:
    """Maturity labels such as "3M" or "10Y" as months, the unit the model uses."""
    return np.array([MATURITY_YEARS[label] * 12 for label in labels])


def fit_ns_daily(table: pd.DataFrame, lam: float | None = DIEBOLD_LI_LAMBDA) -> pd.DataFrame:
    """Fit every row (trading day) of a `dates x maturities` yield table.

    With a number for `lam` each day is fitted by `fit_ns_fixed`; with
    `lam=None` each day gets its own optimized lam from `fit_ns`. Yields are
    expected in percent. Returns one row per day with `DAILY_FIT_COLUMNS`:
    the three betas, the lam used, the RMSE in basis points and the number of
    maturities fitted. Days with too few quoted maturities are left out.
    """
    months = maturity_months(table.columns)
    rows, dates = [], []
    for date, yields in zip(table.index, table.to_numpy(dtype=float), strict=True):
        try:
            fit = fit_ns(months, yields) if lam is None else fit_ns_fixed(months, yields, lam)
        except ValueError:
            continue  # not enough maturities quoted that day
        n_points = int(fit.residuals.notna().sum())
        rows.append([*fit.betas, fit.lam, fit.rmse * 100, n_points])
        dates.append(date)
    result = pd.DataFrame(rows, columns=DAILY_FIT_COLUMNS, index=pd.Index(dates, name="date"))
    return result.astype({"n_points": int})


def fitted_yields(factors: pd.DataFrame, labels: Sequence[str]) -> pd.DataFrame:
    """Rebuild the fitted curve for each day of a daily fit table.

    `factors` needs the three beta columns and `lam`; `labels` are the maturity
    columns to evaluate. Subtract the result from the observed table to get
    residuals per maturity.
    """
    months = maturity_months(labels)
    betas = factors[FACTOR_COLUMNS].to_numpy()
    curves = [
        _loadings_matrix(months, lam) @ beta
        for lam, beta in zip(factors["lam"], betas, strict=True)
    ]
    return pd.DataFrame(curves, index=factors.index, columns=list(labels))


def ns_curve(
    maturities: Sequence[float] | np.ndarray,
    betas: Sequence[float] | pd.Series,
    lam: float = DIEBOLD_LI_LAMBDA,
) -> pd.Series:
    """Evaluate the model curve at `maturities` (in months) for given betas and lam."""
    loadings = ns_loadings(maturities, lam)
    return pd.Series(loadings.to_numpy() @ np.asarray(betas, dtype=float), index=loadings.index)


def load_ns_factors(
    refresh: bool = False,
    cache_path: Path = US_NS_FACTORS_PATH,
    yields: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return daily fixed-lam fits of the US curve, using the Parquet cache when present.

    Fits every trading day from `FULL_CURVE_START` with the Diebold-Li lam.
    `yields` defaults to the cached US table; `refresh=True` refits it.
    """
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    table = load_us_yields() if yields is None else yields
    factors = fit_ns_daily(table.loc[FULL_CURVE_START:])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(cache_path)
    logger.info("Cached %d daily fits to %s", len(factors), cache_path)
    return factors


def get_ns_factors(date: str | pd.Timestamp, factors: pd.DataFrame | None = None) -> pd.Series:
    """Return the fit for the last trading day on or before `date`.

    The Series is named by the trading day actually used, like `data.curve_on`.
    `factors` defaults to `load_ns_factors()`.
    """
    table = load_ns_factors() if factors is None else factors
    target = pd.Timestamp(date)
    pos = table.index.searchsorted(target, side="right") - 1
    if pos < 0:
        raise KeyError(f"no Nelson-Siegel fit on or before {target.date()}")
    return table.iloc[pos].rename(table.index[pos])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    daily = load_ns_factors(refresh=True)
    print(daily.tail())
    print(daily["rmse_bp"].describe())
