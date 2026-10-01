import numpy as np
import pandas as pd
import pytest

from yieldcurve import data

NAN = np.nan


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


class FakeFetcher:
    """Stands in for FRED: returns canned series and counts calls."""

    def __init__(self, series_by_id):
        self.series_by_id = series_by_id
        self.calls = 0

    def __call__(self, series_id):
        self.calls += 1
        return self.series_by_id[series_id]


@pytest.fixture
def fake_us_fetch():
    rng = np.random.default_rng(0)
    return FakeFetcher(
        {
            series_id: _series(3 + 0.1 * rng.standard_normal(10))
            for series_id in data.US_TREASURY_SERIES.values()
        }
    )


def test_read_fred_csv_parses_missing_markers(tmp_path):
    path = tmp_path / "DGS10.csv"
    path.write_text("observation_date,DGS10\n2024-01-01,\n2024-01-02,3.95\n2024-01-03,.\n")

    s = data.read_fred_csv(path)

    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.dtype == float
    assert s.isna().tolist() == [True, False, True]
    assert s.iloc[1] == 3.95


def test_fill_short_gaps_fills_only_short_interior_runs():
    s = _series([NAN, 1.0, NAN, NAN, 2.0, NAN, NAN, NAN, NAN, 3.0])

    filled = data.fill_short_gaps(s, max_gap=2)

    expected = [NAN, 1.0, 1.0, 1.0, 2.0, NAN, NAN, NAN, NAN, 3.0]
    np.testing.assert_array_equal(filled.to_numpy(), expected)


def test_clean_yield_table_sorts_dedupes_and_drops_holidays():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"])
    raw = pd.DataFrame({"2Y": [4.3, 4.1, NAN, 4.4], "10Y": [4.0, 3.9, NAN, 4.05]}, index=idx)

    clean = data.clean_yield_table(raw)

    assert clean.index.tolist() == list(pd.to_datetime(["2024-01-01", "2024-01-03"]))
    assert clean.index.name == "date"
    assert clean.loc["2024-01-03", "2Y"] == 4.4  # last duplicate wins
    assert list(clean.columns) == ["2Y", "10Y"]


def test_clean_yield_table_keeps_structural_gap():
    # 20Y missing for a long stretch while other maturities trade.
    ten = _series(np.linspace(3, 4, 20))
    twenty = ten.copy() + 0.3
    twenty.iloc[5:15] = NAN
    raw = pd.DataFrame({"10Y": ten, "20Y": twenty})

    clean = data.clean_yield_table(raw, max_gap=5)

    assert clean["20Y"].iloc[5:15].isna().all()
    assert clean["10Y"].notna().all()


def test_build_yield_table_aligns_series_in_maturity_order():
    fetch = FakeFetcher({"A": _series([1.0, 2.0]), "B": _series([5.0], start="2024-01-02")})

    table = data.build_yield_table({"2Y": "A", "10Y": "B"}, fetch)

    assert list(table.columns) == ["2Y", "10Y"]
    assert len(table) == 2
    assert np.isnan(table["10Y"].iloc[0])


def test_validate_yield_table_rejects_bad_tables():
    good = pd.DataFrame({"10Y": [4.0, 4.1]}, index=pd.bdate_range("2024-01-01", periods=2))
    data.validate_yield_table(good)

    with pytest.raises(ValueError, match="outside"):
        data.validate_yield_table(good.assign(**{"10Y": [4.0, 410.0]}))
    with pytest.raises(ValueError, match="unknown maturity"):
        data.validate_yield_table(good.rename(columns={"10Y": "11Y"}))
    with pytest.raises(ValueError, match="sorted"):
        data.validate_yield_table(good.iloc[::-1])


def test_load_us_yields_uses_cache(tmp_path, fake_us_fetch):
    cache = tmp_path / "us.parquet"

    first = data.load_us_yields(cache_path=cache, fetch=fake_us_fetch)
    second = data.load_us_yields(cache_path=cache, fetch=fake_us_fetch)

    assert cache.exists()
    assert fake_us_fetch.calls == len(data.US_TREASURY_SERIES)  # second call hit cache
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert list(first.columns) == list(data.US_TREASURY_SERIES)

    data.load_us_yields(refresh=True, cache_path=cache, fetch=fake_us_fetch)
    assert fake_us_fetch.calls == 2 * len(data.US_TREASURY_SERIES)


def test_maturity_labels_match_series():
    assert list(data.US_TREASURY_SERIES) == list(data.MATURITY_YEARS)
    years = list(data.MATURITY_YEARS.values())
    assert years == sorted(years)


def test_curve_on_rolls_back_to_last_trading_day():
    index = pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08"])
    table = pd.DataFrame({"1M": [NAN, 5.0, 5.1], "10Y": [4.0, 4.1, 4.2]}, index=index)

    saturday = data.curve_on(table, "2024-01-06")
    assert saturday.name == pd.Timestamp("2024-01-05")
    assert saturday.to_dict() == {"1M": 5.0, "10Y": 4.1}

    assert data.curve_on(table, "2024-01-04").to_dict() == {"10Y": 4.0}  # unissued 1M dropped
    with pytest.raises(KeyError):
        data.curve_on(table, "2024-01-03")


@pytest.mark.network
def test_fetch_fred_series_live():
    s = data.fetch_fred_series("DGS10")
    assert s.index.min() <= pd.Timestamp("1962-01-02")
    assert s.dropna().between(*data.YIELD_BOUNDS).all()
