"""Term spreads and yield curve inversion episodes.

Spreads are in percentage points (the table's yields are in percent), long
minus short, so a negative value means the curve is inverted.
"""

import pandas as pd

# NBER business-cycle peaks and troughs since 1962 (nber.org/research/business-cycle-dating).
# Matches FRED's USREC flags as of its 2026-08 observation; append new cycles as NBER dates them.
NBER_RECESSIONS: list[tuple[str, str]] = [
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

# Defaults that keep noise around zero from counting as separate episodes.
MIN_INVERSION_DAYS = 10  # an episode must last two trading weeks
MERGE_GAP_DAYS = 5  # positive stretches this short do not end an episode

EPISODE_COLUMNS = ["start", "end", "trading_days", "min_spread", "min_date", "ongoing"]


def spread(table: pd.DataFrame, long: str = "10Y", short: str = "2Y") -> pd.Series:
    """Return `long` minus `short` yield in percentage points, named e.g. "10Y-2Y"."""
    return (table[long] - table[short]).rename(f"{long}-{short}")


def recession_periods() -> pd.DataFrame:
    """NBER recessions as `start`/`end` dates, from the month after the peak
    through the trough month (the convention behind FRED's recession shading)."""
    return pd.DataFrame(
        [
            {
                "start": (pd.Period(peak, "M") + 1).start_time,
                "end": pd.Period(trough, "M").end_time.normalize(),
            }
            for peak, trough in NBER_RECESSIONS
        ]
    )


def find_inversions(
    spread: pd.Series,
    min_days: int = MIN_INVERSION_DAYS,
    merge_gap: int = MERGE_GAP_DAYS,
) -> pd.DataFrame:
    """Find episodes where `spread` is negative.

    Runs of at most `merge_gap` non-negative days between two inverted runs are
    bridged into one episode, and episodes shorter than `min_days` trading days
    are dropped (pass `min_days=1, merge_gap=0` for every raw run). Days are
    counted in observations, after dropping NaNs.

    Returns one row per episode: `start` and `end` (first and last inverted
    day), `trading_days` (observations from start to end), `min_spread` and
    `min_date` (the deepest inversion), and `ongoing` (still inverted on the
    last observation, so `end` is not final). An episode at the very start of
    the sample may have begun earlier than `start`.
    """
    s = spread.dropna()
    if s.empty:
        return pd.DataFrame(columns=EPISODE_COLUMNS)

    inverted = s < 0
    runs = (inverted != inverted.shift()).cumsum()
    run_length = inverted.groupby(runs).transform("size")
    interior = (runs != runs.iloc[0]) & (runs != runs.iloc[-1])
    in_episode = inverted | (interior & (run_length <= merge_gap))

    episode_id = (in_episode != in_episode.shift()).cumsum()[in_episode]
    rows = [
        {
            "start": days.index[0],
            "end": days.index[-1],
            "trading_days": len(days),
            "min_spread": days.min(),
            "min_date": days.idxmin(),
            "ongoing": days.index[-1] == s.index[-1],
        }
        for _, days in s[in_episode].groupby(episode_id)
        if len(days) >= min_days
    ]
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)
