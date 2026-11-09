import numpy as np
import pytest

from yieldcurve import nelson_siegel as ns


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
