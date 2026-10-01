import numpy as np
import pandas as pd

from yieldcurve import plots

NAN = np.nan


def _table():
    index = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {m: 3 + 0.1 * rng.standard_normal(len(index)) for m in ["1M", "2Y", "10Y"]},
        index=index,
    )
    table.loc[:"2024-01-31", "1M"] = NAN  # 1M not yet issued in January
    return table


def test_curve_snapshots_one_trace_per_date_with_rolled_back_label():
    fig = plots.curve_snapshots(_table(), {"Start": "2024-01-06", "Later": "2024-03-01"})

    assert [t.name for t in fig.data] == ["Start (2024-01-05)", "Later (2024-03-01)"]
    assert list(fig.data[0].customdata) == ["2Y", "10Y"]  # unissued 1M dropped
    assert list(fig.data[1].x) == [1 / 12, 2.0, 10.0]


def test_yield_heatmap_resamples_and_keeps_gaps():
    fig = plots.yield_heatmap(_table(), freq="ME")
    (heatmap,) = fig.data

    assert list(heatmap.y) == ["1M", "2Y", "10Y"]
    assert np.asarray(heatmap.z).shape == (3, 3)  # 3 maturities x Jan-Mar
    assert np.isnan(heatmap.z[0][0])  # 1M blank in January
