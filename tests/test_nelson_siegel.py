import numpy as np
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
