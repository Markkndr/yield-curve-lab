import pandas as pd

from yieldcurve import factors


def test_empirical_factors_on_known_curves():
    table = pd.DataFrame({"3M": [1.0, 3.0], "2Y": [3.0, 3.0], "10Y": [2.0, 3.0]})
    result = factors.empirical_factors(table)

    assert list(result.columns) == factors.FACTOR_COLUMNS
    assert result.iloc[0].tolist() == [2.0, 1.0, 3.0]  # humped curve
    assert result.iloc[1].tolist() == [3.0, 0.0, 0.0]  # flat curve
