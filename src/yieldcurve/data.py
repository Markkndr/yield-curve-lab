"""Yield data pipeline: fetch, clean, and cache `dates x maturities` tables.

Yields are kept in percent, as published. Columns are maturity labels ordered
from shortest to longest; use `MATURITY_YEARS` to map them to year fractions.
"""

import json
import logging
import sys
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import fredapi
import pandas as pd

from yieldcurve.config import DATA_DIR, get_fred_api_key

logger = logging.getLogger(__name__)

# FRED constant-maturity Treasury series. DGS1MO starts 2001-07-31 and DGS20
# is missing 1987-1993 (not issued); the rest cover at least 1981 onward.
US_TREASURY_SERIES: dict[str, str] = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

MATURITY_YEARS: dict[str, float] = {
    "1M": 1 / 12,
    "3M": 3 / 12,
    "6M": 6 / 12,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "15Y": 15.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

US_CACHE_PATH = DATA_DIR / "us_treasury_yields.parquet"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# AKK (Hungarian Government Debt Management Agency) daily benchmark yields:
# the tenor labels AKK publishes, mapped to this project's maturity labels.
# There is no 2Y benchmark, and the 20Y only exists in recent years.
HU_BENCHMARK_TENORS: dict[str, str] = {
    "3M": "3M",
    "6M": "6M",
    "12M": "1Y",
    "3Y": "3Y",
    "5Y": "5Y",
    "10Y": "10Y",
    "15Y": "15Y",
    "20Y": "20Y",
}

HU_CACHE_PATH = DATA_DIR / "hu_benchmark_yields.parquet"

# The JSON endpoint behind akk.hu's benchmark-yields page. It is not a
# documented public API, so the Parquet cache is the dependable copy.
AKK_BENCHMARK_URL = (
    "https://www.akk.hu/api/data/xpose3-dwh/data?query=getBenchmark&dateStart={start}&dateEnd={end}"
)
HU_FIRST_YEAR = 2002  # AKK's benchmark history starts on 2002-01-03
AKK_TIMEOUT_SECONDS = 60
AKK_COLUMNS = ["date", "tenor", "yield"]

# Longest run of missing days (after dropping market holidays) that gets
# forward-filled. Longer runs are structural (series not issued) and stay NaN.
MAX_FILL_GAP = 5

# Plausible bounds for a government yield in percent; outside means bad data.
YIELD_BOUNDS = (-5.0, 30.0)

Fetcher = Callable[[str], pd.Series]
# Takes ISO start and end dates, returns long rows with `AKK_COLUMNS`.
BenchmarkFetcher = Callable[[str, str], pd.DataFrame]


def read_fred_csv(source: str | Path) -> pd.Series:
    """Parse a FRED graph CSV download (date column + one value column)."""
    frame = pd.read_csv(source, index_col=0, parse_dates=True, na_values=["."])
    return frame.iloc[:, 0].astype(float)


def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch one FRED series via the API, or the public CSV if no key is set."""
    try:
        api_key = get_fred_api_key()
    except RuntimeError:
        logger.info("No FRED_API_KEY; using public CSV download for %s", series_id)
        return read_fred_csv(FRED_CSV_URL.format(series_id=series_id))
    return fredapi.Fred(api_key=api_key).get_series(series_id).astype(float)


def build_yield_table(
    series: Mapping[str, str], fetch: Fetcher = fetch_fred_series
) -> pd.DataFrame:
    """Fetch each series and align them into one raw `dates x maturities` table."""
    columns = {label: fetch(series_id) for label, series_id in series.items()}
    return pd.concat(columns, axis=1, sort=True)


def fill_short_gaps(s: pd.Series, max_gap: int = MAX_FILL_GAP) -> pd.Series:
    """Forward-fill runs of at most `max_gap` NaNs that follow a valid value.

    Forward-fill (not interpolation) avoids look-ahead. Leading NaNs (before a
    series starts) and longer runs are left missing.
    """
    missing = s.isna()
    run_length = missing.groupby((~missing).cumsum()).transform("sum")
    started = s.notna().cummax()
    fillable = missing & started & (run_length <= max_gap)
    return s.where(~fillable, s.ffill())


def clean_yield_table(raw: pd.DataFrame, max_gap: int = MAX_FILL_GAP) -> pd.DataFrame:
    """Sort, dedupe, drop all-missing dates (market holidays), fill short gaps."""
    table = raw.copy()
    table.index = pd.DatetimeIndex(table.index).normalize()
    table.index.name = "date"
    table = table[~table.index.duplicated(keep="last")].sort_index()
    table = table.astype(float).dropna(how="all")
    return table.apply(fill_short_gaps, max_gap=max_gap)


def validate_yield_table(table: pd.DataFrame) -> None:
    """Raise ValueError if the table breaks the pipeline's invariants."""
    if not isinstance(table.index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")
    if not table.index.is_monotonic_increasing or table.index.has_duplicates:
        raise ValueError("index must be sorted and unique")
    unknown = set(table.columns) - set(MATURITY_YEARS)
    if unknown:
        raise ValueError(f"unknown maturity columns: {sorted(unknown)}")
    low, high = YIELD_BOUNDS
    bad_rows = ((table < low) | (table > high)).any(axis=1)
    if bad_rows.any():
        raise ValueError(f"yields outside {YIELD_BOUNDS}%:\n{table[bad_rows].head()}")


def curve_on(table: pd.DataFrame, date: str | pd.Timestamp) -> pd.Series:
    """Return the curve on the last trading day on or before `date`.

    The Series is named by the trading day actually used, so callers can tell
    when a weekend or holiday was rolled back. Maturities without data that
    day (not yet issued) are dropped.
    """
    target = pd.Timestamp(date)
    pos = table.index.searchsorted(target, side="right") - 1
    if pos < 0:
        raise KeyError(f"no data on or before {target.date()}")
    return table.iloc[pos].dropna().rename(table.index[pos])


def read_akk_benchmark(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Parse an AKK `getBenchmark` JSON response into long rows (`AKK_COLUMNS`).

    AKK quotes one benchmark security per tenor per day; `yield` is its
    benchmark yield in percent.
    """
    meta = payload.get("meta", {})
    if meta.get("status") != "OK":
        raise ValueError(f"AKK request failed: {meta.get('message') or meta.get('status')}")
    rows = pd.DataFrame(payload.get("data", []), columns=AKK_COLUMNS)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["yield"] = rows["yield"].astype(float)
    return rows


def fetch_akk_benchmark(start: str, end: str) -> pd.DataFrame:
    """Fetch AKK benchmark yields for the dates `start`..`end` (ISO, inclusive)."""
    url = AKK_BENCHMARK_URL.format(start=start, end=end)
    with urllib.request.urlopen(url, timeout=AKK_TIMEOUT_SECONDS) as response:
        return read_akk_benchmark(json.load(response))


def build_hu_yield_table(
    fetch: BenchmarkFetcher = fetch_akk_benchmark,
    first_year: int = HU_FIRST_YEAR,
    last_year: int | None = None,
) -> pd.DataFrame:
    """Fetch AKK benchmark yields one calendar year at a time and pivot them
    into a raw `dates x maturities` table. Tenors this project does not model
    are dropped with a warning."""
    if last_year is None:
        last_year = pd.Timestamp.today().year
    rows = pd.concat(
        [fetch(f"{year}-01-01", f"{year}-12-31") for year in range(first_year, last_year + 1)],
        ignore_index=True,
    )
    unknown = set(rows["tenor"].dropna()) - set(HU_BENCHMARK_TENORS)
    if unknown:
        logger.warning("Dropping unknown AKK tenors: %s", sorted(unknown))
    wide = rows.pivot_table(index="date", columns="tenor", values="yield", aggfunc="last")
    wide = wide.reindex(columns=list(HU_BENCHMARK_TENORS)).rename(columns=HU_BENCHMARK_TENORS)
    wide.columns.name = None
    return wide


def _load_cached(
    cache_path: Path, refresh: bool, build: Callable[[], pd.DataFrame]
) -> pd.DataFrame:
    """Read the Parquet cache, or build, clean, validate and cache the table."""
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    table = clean_yield_table(build())
    validate_yield_table(table)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(cache_path)
    logger.info("Cached %d rows to %s", len(table), cache_path)
    return table


def load_us_yields(
    refresh: bool = False,
    cache_path: Path = US_CACHE_PATH,
    fetch: Fetcher = fetch_fred_series,
) -> pd.DataFrame:
    """Return the clean US Treasury table, using the Parquet cache when present."""
    return _load_cached(cache_path, refresh, lambda: build_yield_table(US_TREASURY_SERIES, fetch))


def load_hu_yields(
    refresh: bool = False,
    cache_path: Path = HU_CACHE_PATH,
    fetch: BenchmarkFetcher = fetch_akk_benchmark,
) -> pd.DataFrame:
    """Return the clean Hungarian benchmark table, using the Parquet cache when present."""
    return _load_cached(cache_path, refresh, lambda: build_hu_yield_table(fetch))


if __name__ == "__main__":
    # `python -m yieldcurve.data` refreshes both caches; pass `us` or `hu` for one.
    logging.basicConfig(level=logging.INFO)
    loaders = {"us": load_us_yields, "hu": load_hu_yields}
    for market in sys.argv[1:] or list(loaders):
        yields = loaders[market](refresh=True)
        print(f"== {market.upper()}")
        print(yields.tail())
        print(yields.notna().sum().rename("observations"))
