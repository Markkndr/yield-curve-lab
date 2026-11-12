import numpy as np
import pandas as pd
import pytest

from yieldcurve import nelson_siegel as ns

# The eleven US maturities in months, and an upward-sloping, humped test curve.
MONTHS = np.array([1, 3, 6, 12, 24, 36, 60, 84, 120, 240, 360], dtype=float)
BETAS = np.array([5.0, -2.0, 1.5])


def _curve(months=MONTHS, betas=BETAS, lam=ns.DIEBOLD_LI_LAMBDA):
    return ns.ns_loadings(months, lam).to_numpy() @ betas


def test_loadings_columns_and_index():
    result = ns.ns_loadings([3, 24, 120])

    assert list(result.columns) == ns.FACTOR_COLUMNS
    assert result.index.tolist() == [3, 24, 120]
    assert (result["level"] == 1.0).all()


def test_loadings_match_closed_form():
    lam, tau = 0.0609, 24.0
    decay = np.exp(-lam * tau)
    slope = (1 - decay) / (lam * tau)

    row = ns.ns_loadings([tau], lam).iloc[0]

    assert row["slope"] == pytest.approx(slope, rel=1e-12)
    assert row["curvature"] == pytest.approx(slope - decay, rel=1e-12)


def test_loadings_limits_at_short_and_long_end():
    result = ns.ns_loadings([0.0, 1e-9, 1e6])

    # tau -> 0: slope loading -> 1, curvature loading -> 0 (finite at exactly 0)
    assert result.loc[0.0, "slope"] == 1.0
    assert result.loc[0.0, "curvature"] == 0.0
    assert result.loc[1e-9, "slope"] == pytest.approx(1.0)
    assert result.loc[1e-9, "curvature"] == pytest.approx(0.0, abs=1e-9)
    # tau -> infinity: both decay to 0, only the level survives
    assert result.loc[1e6, "slope"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc[1e6, "curvature"] == pytest.approx(0.0, abs=1e-4)


def test_diebold_li_lambda_puts_curvature_peak_near_30_months():
    months = np.arange(1, 121, 0.1)
    result = ns.ns_loadings(months)

    # 0.0609 is Diebold-Li's rounded value: the exact peak is at ~29.4 months
    assert result["curvature"].idxmax() == pytest.approx(30, abs=1)


def test_nan_maturity_gives_nan_loadings():
    result = ns.ns_loadings([12, np.nan])

    assert result.iloc[1][["slope", "curvature"]].isna().all()
    assert result.iloc[0].notna().all()


def test_fit_fixed_recovers_betas_from_noiseless_curve():
    fit = ns.fit_ns_fixed(MONTHS, _curve())

    assert list(fit.betas.index) == ns.FACTOR_COLUMNS
    np.testing.assert_allclose(fit.betas.to_numpy(), BETAS, rtol=0, atol=1e-10)
    np.testing.assert_allclose(fit.fitted.to_numpy(), _curve(), atol=1e-10)
    np.testing.assert_allclose(fit.residuals.to_numpy(), 0.0, atol=1e-10)
    assert fit.rmse == pytest.approx(0.0, abs=1e-10)
    assert fit.lam == ns.DIEBOLD_LI_LAMBDA
    assert fit.fitted.index.tolist() == MONTHS.tolist()


@pytest.mark.parametrize("lam", [0.02, 0.0609, 0.2])
def test_fit_fixed_recovers_betas_at_other_lambdas(lam):
    fit = ns.fit_ns_fixed(MONTHS, _curve(lam=lam), lam=lam)

    np.testing.assert_allclose(fit.betas.to_numpy(), BETAS, atol=1e-9)


def test_fit_fixed_short_rate_and_long_rate_limits():
    fit = ns.fit_ns_fixed(MONTHS, _curve())

    # y(0) = b0 + b1 and y(infinity) = b0
    assert fit.betas["level"] + fit.betas["slope"] == pytest.approx(3.0)
    assert fit.betas["level"] == pytest.approx(5.0)


def test_fit_fixed_ignores_missing_yields_and_maturities():
    yields = _curve()
    yields[[0, 9]] = np.nan  # 1M and 20Y not quoted
    months = MONTHS.copy()
    months[4] = np.nan

    fit = ns.fit_ns_fixed(months, yields)

    np.testing.assert_allclose(fit.betas.to_numpy(), BETAS, atol=1e-10)
    assert fit.residuals.isna().to_numpy().tolist() == [i in (0, 4, 9) for i in range(11)]
    # a maturity without a yield still gets a fitted value
    assert fit.fitted.iloc[0] == pytest.approx(_curve()[0])
    assert np.isnan(fit.fitted.iloc[4])


def test_fit_fixed_matches_least_squares_on_noisy_curve():
    rng = np.random.default_rng(0)
    yields = _curve() + 0.05 * rng.standard_normal(len(MONTHS))

    fit = ns.fit_ns_fixed(MONTHS, yields)

    # OLS residuals are orthogonal to every loading
    X = ns.ns_loadings(MONTHS).to_numpy()
    np.testing.assert_allclose(X.T @ fit.residuals.to_numpy(), 0.0, atol=1e-10)
    assert fit.rmse == pytest.approx(np.sqrt(np.mean(fit.residuals**2)))
    assert 0 < fit.rmse < 0.1


def test_fit_fixed_rejects_too_few_points_and_bad_shapes():
    with pytest.raises(ValueError, match="at least 3"):
        ns.fit_ns_fixed([3, 24, 120], [4.0, np.nan, 4.5])
    with pytest.raises(ValueError, match="same length"):
        ns.fit_ns_fixed([3, 24, 120], [4.0, 4.2])


@pytest.mark.parametrize("lam", [0.015, 0.04, 0.0609, 0.15, 0.4])
def test_fit_recovers_lambda_and_betas_from_noiseless_curve(lam):
    fit = ns.fit_ns(MONTHS, _curve(lam=lam))

    assert fit.lam == pytest.approx(lam, rel=1e-5)
    np.testing.assert_allclose(fit.betas.to_numpy(), BETAS, atol=1e-4)
    assert fit.rmse < 1e-7


def test_fit_recovers_lambda_with_missing_maturities():
    yields = _curve(lam=0.09)
    yields[[0, 5, 9]] = np.nan  # 1M, 3Y and 20Y not quoted

    fit = ns.fit_ns(MONTHS, yields)

    assert fit.lam == pytest.approx(0.09, rel=1e-5)
    np.testing.assert_allclose(fit.betas.to_numpy(), BETAS, atol=1e-4)
    assert fit.residuals.isna().sum() == 3


@pytest.mark.parametrize("seed", range(20))
def test_fit_never_worse_than_fixed_lambda(seed):
    rng = np.random.default_rng(seed)
    betas = rng.normal([4.0, -1.0, 0.0], [2.0, 2.0, 3.0])
    yields = _curve(betas=betas, lam=rng.uniform(0.02, 0.5))
    yields += 0.1 * rng.standard_normal(len(MONTHS))

    optimized = ns.fit_ns(MONTHS, yields)
    fixed = ns.fit_ns_fixed(MONTHS, yields)

    assert optimized.rmse <= fixed.rmse
    assert ns.LAMBDA_BOUNDS[0] <= optimized.lam <= ns.LAMBDA_BOUNDS[1]


def test_fit_keeps_lambda_inside_custom_bounds():
    fit = ns.fit_ns(MONTHS, _curve(lam=0.3), bounds=(0.02, 0.1))

    assert 0.02 <= fit.lam <= 0.1
    assert fit.lam == pytest.approx(0.1, rel=1e-3)  # true lam is above the range


def test_fit_rejects_too_few_points_and_bad_bounds():
    with pytest.raises(ValueError, match="at least 4"):
        ns.fit_ns([3, 24, 120], [4.0, 4.2, 4.5])
    with pytest.raises(ValueError, match="bounds"):
        ns.fit_ns(MONTHS, _curve(), bounds=(0.5, 0.1))


LABELS = ["3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]


def _daily_table(lam=ns.DIEBOLD_LI_LAMBDA):
    """Five days of synthetic curves whose betas drift, and the betas used."""
    index = pd.bdate_range("2024-01-01", periods=5, name="date")
    betas = pd.DataFrame(
        {"level": np.linspace(4, 5, 5), "slope": np.linspace(-2, 1, 5), "curvature": 1.0},
        index=index,
    )
    months = ns.maturity_months(LABELS)
    curves = betas.to_numpy() @ ns.ns_loadings(months, lam).to_numpy().T
    return pd.DataFrame(curves, index=index, columns=LABELS), betas


def test_maturity_months_converts_labels():
    assert ns.maturity_months(["1M", "3M", "2Y", "10Y"]).tolist() == pytest.approx([1, 3, 24, 120])


def test_fit_daily_recovers_betas_for_every_day():
    table, betas = _daily_table()

    daily = ns.fit_ns_daily(table)

    assert list(daily.columns) == ns.DAILY_FIT_COLUMNS
    assert daily.index.equals(table.index)
    np.testing.assert_allclose(daily[ns.FACTOR_COLUMNS].to_numpy(), betas.to_numpy(), atol=1e-9)
    assert (daily["lam"] == ns.DIEBOLD_LI_LAMBDA).all()
    assert daily["rmse_bp"].max() < 1e-6
    assert (daily["n_points"] == len(LABELS)).all()


def test_fit_daily_handles_missing_maturities_and_skips_thin_days():
    table, betas = _daily_table()
    table.loc[table.index[1], "30Y"] = np.nan  # one maturity missing: still fitted
    table.loc[table.index[3], LABELS[2:]] = np.nan  # two points left: skipped

    daily = ns.fit_ns_daily(table)

    assert daily.index.tolist() == table.index[[0, 1, 2, 4]].tolist()
    assert daily.loc[table.index[1], "n_points"] == len(LABELS) - 1
    np.testing.assert_allclose(
        daily.loc[table.index[1], ns.FACTOR_COLUMNS].to_numpy(dtype=float),
        betas.iloc[1].to_numpy(),
        atol=1e-9,
    )


def test_fit_daily_with_optimized_lambda_recovers_it():
    table, betas = _daily_table(lam=0.1)

    daily = ns.fit_ns_daily(table, lam=None)

    np.testing.assert_allclose(daily["lam"].to_numpy(), 0.1, rtol=1e-4)
    np.testing.assert_allclose(daily[ns.FACTOR_COLUMNS].to_numpy(), betas.to_numpy(), atol=1e-3)


def test_fitted_yields_rebuilds_the_curves():
    table, _ = _daily_table()
    daily = ns.fit_ns_daily(table)

    fitted = ns.fitted_yields(daily, LABELS)

    pd.testing.assert_frame_equal(fitted, table, check_freq=False, atol=1e-9)


def test_load_ns_factors_fits_full_curve_era_and_caches(tmp_path):
    table, _ = _daily_table()
    early = table.copy()
    early.index = pd.bdate_range("2001-07-02", periods=5, name="date")  # before the 1M exists
    yields = pd.concat([early, table])
    cache = tmp_path / "ns.parquet"

    first = ns.load_ns_factors(cache_path=cache, yields=yields)
    second = ns.load_ns_factors(cache_path=cache, yields=yields.iloc[:0])  # served from cache

    assert cache.exists()
    assert first.index.min() == table.index[0]  # nothing before FULL_CURVE_START
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_get_ns_factors_rolls_back_to_last_trading_day():
    table, betas = _daily_table()  # Mon 2024-01-01 .. Fri 2024-01-05
    daily = ns.fit_ns_daily(table)

    saturday = ns.get_ns_factors("2024-01-06", daily)

    assert saturday.name == pd.Timestamp("2024-01-05")
    assert saturday["level"] == pytest.approx(betas["level"].iloc[-1])
    assert list(saturday.index) == ns.DAILY_FIT_COLUMNS
    with pytest.raises(KeyError):
        ns.get_ns_factors("2023-12-29", daily)


def test_ns_curve_evaluates_the_model():
    curve = ns.ns_curve(MONTHS, BETAS)

    np.testing.assert_allclose(curve.to_numpy(), _curve())
    assert curve.index.tolist() == MONTHS.tolist()
