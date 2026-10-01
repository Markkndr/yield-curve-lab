import pandas as pd
import pytest

from yieldcurve import spreads


def _series(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def _raw(s):
    """Every negative run, no merging or length filter."""
    return spreads.find_inversions(s, min_days=1, merge_gap=0)


def test_spread_is_long_minus_short_and_named():
    table = pd.DataFrame({"2Y": [4.0, 4.5], "10Y": [4.2, 4.1]})
    s = spreads.spread(table, "10Y", "2Y")
    assert s.name == "10Y-2Y"
    assert s.round(2).tolist() == [0.2, -0.4]


def test_find_inversions_reports_episode_stats():
    s = _series([0.3, -0.1, -0.5, -0.2, 0.1])
    (episode,) = _raw(s).itertuples(index=False)

    assert episode.start == s.index[1]
    assert episode.end == s.index[3]
    assert episode.trading_days == 3
    assert episode.min_spread == -0.5
    assert episode.min_date == s.index[2]
    assert not episode.ongoing


def test_find_inversions_at_start_and_end_of_sample():
    s = _series([-0.2, -0.1, 0.3, 0.2, -0.4, -0.1])
    episodes = _raw(s)

    assert episodes["start"].tolist() == [s.index[0], s.index[4]]
    assert episodes["end"].tolist() == [s.index[1], s.index[5]]
    assert episodes["ongoing"].tolist() == [False, True]


def test_find_inversions_drops_single_day_blips():
    s = _series([0.2, -0.05, 0.2, -0.1, -0.1, -0.1, 0.2])
    episodes = spreads.find_inversions(s, min_days=2, merge_gap=0)

    assert episodes["start"].tolist() == [s.index[3]]
    assert episodes["trading_days"].tolist() == [3]


def test_find_inversions_bridges_short_positive_gaps_only():
    s = _series([-0.1, -0.2, 0.01, -0.3, 0.1, 0.1, 0.1, -0.1])

    merged = spreads.find_inversions(s, min_days=1, merge_gap=1)
    assert merged["start"].tolist() == [s.index[0], s.index[7]]
    assert merged["end"].tolist() == [s.index[3], s.index[7]]
    assert merged["trading_days"].tolist() == [4, 1]
    assert merged["min_spread"].tolist() == [-0.3, -0.1]

    assert len(_raw(s)) == 3


def test_find_inversions_does_not_bridge_leading_or_trailing_positive_runs():
    s = _series([0.1, -0.2, -0.2, 0.1])
    episodes = spreads.find_inversions(s, min_days=1, merge_gap=5)
    assert episodes[["start", "end"]].to_numpy().tolist() == [[s.index[1], s.index[2]]]


@pytest.mark.parametrize("values", [[], [0.1, 0.2], [float("nan")] * 3])
def test_find_inversions_empty_when_never_inverted(values):
    episodes = _raw(_series(values))
    assert episodes.empty
    assert list(episodes.columns) == spreads.EPISODE_COLUMNS


def test_find_inversions_ignores_missing_days():
    s = _series([-0.1, float("nan"), -0.2, 0.1])
    (episode,) = _raw(s).itertuples(index=False)
    assert (episode.start, episode.end, episode.trading_days) == (s.index[0], s.index[2], 2)


def test_recession_periods_follow_fred_convention():
    periods = spreads.recession_periods()
    covid = periods.iloc[-1]
    assert covid["start"] == pd.Timestamp("2020-03-01")
    assert covid["end"] == pd.Timestamp("2020-04-30")
    assert (periods["start"] < periods["end"]).all()


@pytest.mark.network
def test_2022_24_inversion_in_live_fred_data():
    from yieldcurve.data import fetch_fred_series

    s = (fetch_fred_series("DGS10") - fetch_fred_series("DGS2")).loc["2022":"2024"]
    episodes = spreads.find_inversions(s)
    longest = episodes.loc[episodes["trading_days"].idxmax()]

    # The unbroken run ends 2024-08-26; the 2024-09-03 and 09-05 dips are bridged.
    assert longest["start"] == pd.Timestamp("2022-07-06")
    assert longest["end"] == pd.Timestamp("2024-09-05")
    assert longest["trading_days"] > 500
    assert _raw(s)["trading_days"].max() == 537
