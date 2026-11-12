<div align="center">

# 📈 Yield Curve Lab

### Model government bond yield curves, decompose their moves, and ask questions about them in plain English

A Python research project on US and Hungarian government bond yield curves: a clean data pipeline, Nelson-Siegel curve fitting built from scratch, PCA on daily yield changes, and a GenAI layer (RAG + LangGraph agent) that combines the numbers with what the Fed and the MNB actually said.

<br>

<img src="https://img.shields.io/badge/Status-In_Progress-orange?style=for-the-badge" alt="Status: In Progress">
<a href="https://github.com/Markkndr/yield-curve-lab/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Markkndr/yield-curve-lab/ci.yml?branch=main&style=for-the-badge&label=CI" alt="CI"></a>
<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy">
<img src="https://img.shields.io/badge/Data-FRED-2A5A8C?style=for-the-badge" alt="Data: FRED">
<img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT">

</div>

> 🚧 **This project is under active development.** The sections below describe the full project vision and roadmap. See [**Current Status**](#-current-status) for what's already implemented.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Who It's For](#-who-its-for)
- [Roadmap](#-roadmap)
- [Questions It Answers](#-questions-it-answers)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Current Status](#-current-status)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Further Reading](#-further-reading)
- [License](#-license)

---

## 🎯 Overview

The yield curve (government bond yields across maturities, from 1 month to 30 years) is one of the most watched signals in finance. Its **level** tracks the rate environment, its **slope** has preceded every modern US recession when it inverted, and its **shape** encodes what the market expects central banks to do next.

**This project turns that curve into something you can measure and query:**

- 📊 **Collect** daily US Treasury yields (FRED) and Hungarian benchmark yields (ÁKK) into one clean, tested `dates × maturities` table
- 📐 **Fit** the Nelson-Siegel model from scratch and track its level, slope, and curvature factors over time
- 🔬 **Decompose** daily yield changes with PCA and compare the statistical factors to the Nelson-Siegel ones
- 🔮 **Forecast** the curve with the Diebold-Li approach and test it against a random walk out of sample
- 🤖 **Ask** questions like *"When did the 10Y–2Y spread invert in 2022, and what did the Fed say about inflation then?"*, answered by an agent that combines computed numbers with cited central-bank statements

---

## 👥 Who It's For

| Audience | What they get |
|----------|---------------|
| **Quant / data science reviewers** | A from-scratch Nelson-Siegel implementation, PCA factor analysis, and an out-of-sample forecasting test |
| **GenAI / ML engineering reviewers** | A RAG pipeline with citations and guardrails, a tool-calling LangGraph agent, and an evaluation suite running in CI |
| **Macro & fixed-income readers** | Spread and inversion history, key-date curve snapshots, and a Hungary 2022–23 case study |

---

## 🗺 Roadmap

### Phase 0 — ⚙️ Setup
- `pyproject.toml`, virtual environment, pytest, ruff
- pandas, NumPy, SciPy, scikit-learn, statsmodels, Plotly, fredapi
- FRED API key in `.env`

### Phase 1 — 📦 Data Pipeline
- FRED constant-maturity Treasury series (1M–30Y); ÁKK benchmark yields
- Clean `dates × maturities` table, Parquet cache, gap handling
- Pipeline tests

### Phase 2 — 🔎 Exploratory Analysis
- Curve snapshots on key dates (2007, 2008, 2020, 2022) and a yield heatmap
- 10Y–2Y and 10Y–3M spreads and inversion periods
- Hungary 2022–23 case study

### Phase 3 — 📐 Nelson-Siegel from Scratch
- Fixed λ (Diebold-Li, OLS), then optimized λ
- Daily fits, RMSE in basis points, factor time series
- Tests on synthetic curves

### Phase 4 — 🔬 PCA on Daily Yield Changes
- Explained variance and loadings (level / slope / curvature)
- Rolling PCA across rate eras
- **Headline finding:** PCA factors vs. Nelson-Siegel factors

### Phase 5 — 🔮 Application
- Diebold-Li AR(1) forecasting vs. random walk, out of sample, **or**
- PCA-based bond portfolio risk and scenarios

### Phase 6 — 🇭🇺 Hungary vs. US
- Side-by-side PCA comparison of the two curves

### Phase 7 — 🖥️ Presentation
- Streamlit dashboard
- Findings write-up
- Dockerfile

### Phase 8 — 🤖 GenAI Layer
- **8a Setup:** LangChain, LangGraph, LLM provider SDK, Chroma, FastAPI, Langfuse; API key with a provider cost cap
- **8b Document corpus:** Fed FOMC statements and MNB rate-decision press releases (2007 onward), chunked with metadata into a persisted Chroma index
- **8c RAG chain:** metadata-filtered retrieval, cited answers with Pydantic structured output, and a "not enough information" guardrail
- **8d Tools:** `get_curve`, `get_spread`, `find_inversions`, `get_ns_factors`, `search_statements`, each with a Pydantic input schema and unit tests
- **8e Agent:** a LangGraph tool-calling loop with an iteration limit that combines numbers with cited statements
- **8f Evaluation:** a golden question set; retrieval hit rate, numeric correctness, faithfulness (LLM-as-judge), latency, and cost per query; Langfuse tracing; runs as a regression test in GitHub Actions
- **8g Serving:** FastAPI (`/ask`, `/health`), Docker Compose, and a chat tab in the dashboard

---

## 🧠 Questions It Answers

```text
• What did the US curve look like before Lehman, at the COVID low, and at the 2022 hiking peak?
• When was the 10Y–2Y spread inverted, and for how long each time?
• How much of daily yield variance do the first three principal components explain, and is that stable across eras?
• Do the PCA factors line up with Nelson-Siegel's level, slope, and curvature?
• Does a Diebold-Li forecast beat a random walk out of sample?
• How did Hungary's curve behave through the 2022–23 inflation shock and MNB's rate hikes?
• "When did the 10Y–2Y spread invert in 2022, and what did the Fed say about inflation then?"
```

---

## 🏗 Architecture

### Current architecture

What runs today is the **data pipeline**: a single Python module that fetches every Treasury maturity from FRED, aligns and cleans them, validates the result, and caches it as Parquet. Later analysis reads the cached table instead of calling FRED again. Hungarian benchmark yields from ÁKK go through the same clean, validate and cache steps into `data/hu_benchmark_yields.parquet` (`load_hu_yields()`).

```
  FRED  ──►  fetch_fred_series  ──►  build_yield_table  ──►  clean_yield_table  ──►  validate_yield_table
  (API with key, or                  (align 11 series)       (drop holidays,         (sorted dates, known
   public CSV fallback)                                       fill short gaps)        maturities, sane range)
                                                                                              │
                                    load_us_yields()  ◄──  data/us_treasury_yields.parquet  ◄─┘
```

### Target architecture (vision)

The full project adds the components below. **None of these are implemented yet.** They describe the roadmap, not the current build.

- **Analysis modules:** Nelson-Siegel fitting, PCA, and forecasting as tested library code, with notebooks on top
- **Document index:** Fed and MNB statements embedded into Chroma
- **Agent:** a LangGraph loop calling analysis tools and the retriever
- **Serving:** a FastAPI backend and a Streamlit dashboard with a chat tab

```
  FRED / ÁKK ──► data pipeline ──► Parquet cache ──► nelson_siegel · pca · forecast ──► notebooks
                                                            │                            Streamlit
                                                            ▼                            dashboard
  Fed / MNB  ──► ingest ──► Chroma index ──► retriever ──► tools ──► LangGraph agent ──► FastAPI /ask
  statements                                                              │
                                                              Langfuse tracing · eval suite (CI)
```

---

## 🛠 Tech Stack

**In use today:**

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ (developed on 3.14) |
| **Data** | pandas, NumPy, PyArrow (Parquet) |
| **Data source** | FRED via `fredapi`, with a public-CSV fallback when no key is set; ÁKK benchmark yields over HTTPS (standard library) |
| **Modeling** | NumPy least squares and SciPy (bounded scalar optimization) for Nelson-Siegel |
| **Visualization** | Plotly |
| **Config** | `python-dotenv` (`.env`) |
| **Quality** | pytest, ruff, GitHub Actions CI (Python 3.11 + 3.14) |

**Planned (roadmap):**

| Layer | Technology |
|-------|-----------|
| **Modeling** | scikit-learn (PCA), statsmodels (AR models) |
| **Visualization** | Streamlit |
| **GenAI** | LangChain, LangGraph, an LLM provider SDK, Chroma |
| **Serving** | FastAPI, Uvicorn, Docker Compose |
| **Observability & eval** | Langfuse, GitHub Actions |

> scikit-learn and statsmodels are already installed with the project, but no module uses them yet.

---

## ✅ Current Status

Phases 0 to 3 are in place: the data pipeline for both markets, the exploratory analysis, and Nelson-Siegel fitting. Implemented so far:

- ⚙️ **Project setup:** `pyproject.toml` with a `src/` layout, pytest, ruff, and `.env`-based configuration; GitHub Actions runs lint and the offline tests on every push and PR
- 📥 **FRED ingestion:** all 11 constant-maturity Treasury series (`DGS1MO` … `DGS30`), through the API when a key is set, otherwise through FRED's public CSV download
- 🧹 **Cleaning:**
  - Drops market holidays (dates where every maturity is missing)
  - Forward-fills gaps of up to 5 trading days, avoiding look-ahead
  - Leaves structural gaps missing: the 20Y was not issued in 1987–1993, and the 1M only starts in 2001
- ✔️ **Validation:** sorted, unique dates; known maturity columns; yields within −5% to 30%
- 💾 **Parquet cache:** 16,000+ trading days from 1962 onward, with the full 1M–30Y curve available from July 2001
- 🇭🇺 **Hungarian benchmark yields:** ÁKK's daily benchmark yields (3M, 6M, 1Y, 3Y, 5Y, 10Y, 15Y from 2002-01-03; 20Y from 2020-05-20) through the same clean → validate → Parquet steps. There is no 2Y benchmark, so Hungarian slopes use 10Y–3Y or 10Y–3M
- 🧪 **Tests:** offline unit tests with fake FRED and ÁKK fetchers, plus live checks of both sources behind a `network` marker
- 📐 **Nelson-Siegel (Phase 3):** factor loadings, a fixed-λ OLS fit and an optimized-λ fit written from scratch and tested by recovering known betas and λ from synthetic curves. [`notebooks/02_nelson_siegel.ipynb`](notebooks/02_nelson_siegel.ipynb) fits all 6,290 trading days since 2001-07-31:
  - Median RMSE is 9.5 bp with the Diebold-Li λ and 4.7 bp with λ optimized each day
  - The worst fits are bill-market dislocations (April 2023, August 2007), the zero-rate years 2011–13, and the inverted curves of 2022–23
  - Residuals are systematic: a wave of about ±10 bp along the curve, including a persistent 20Y kink
  - The slope and curvature factors match their empirical proxies (correlations 0.996 and 0.979); the level factor tracks the 30Y yield (0.99), not the 3M/2Y/10Y average (0.60)
- 🔎 **EDA (Phase 2):** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) covers data coverage, the curve on key dates (2007, 2008, the 2020 low, the 2022 hiking peak, latest) a yield heatmap since 1962, and 10Y–2Y / 10Y–3M inversion episodes against NBER recessions (the 2022–24 10Y–2Y inversion: 2022-07-06 to 2024-09-05, the longest in the sample), and a Hungary 2022–23 case study (long yields peaked on 2022-10-14, the day of the MNB's emergency move; the 10Y–3M inversion reached −685 bp in January 2023, against −189 bp in the US), using reusable Plotly helpers in `yieldcurve.plots`

> 📝 Note: ÁKK has no documented download API. The loader calls the JSON endpoint behind the [benchmark-yields page](https://www.akk.hu/hu/statisztika/hozamok-indexek-forgalmi-adatok/referenciahozamok), which could change without notice, so the Parquet cache is the dependable copy.

### Python API — `yieldcurve.data`

| Function / constant | Description |
|---------------------|-------------|
| `load_us_yields(refresh=False)` | The clean US Treasury table, served from the Parquet cache when present |
| `load_hu_yields(refresh=False)` | The clean Hungarian benchmark table (ÁKK), served from the Parquet cache when present |
| `fetch_akk_benchmark(start, end)` | ÁKK benchmark yields for a date range as long `date`, `tenor`, `yield` rows |
| `build_hu_yield_table(fetch)` | Fetch ÁKK year by year and pivot into a raw `dates × maturities` table |
| `fetch_fred_series(series_id)` | One FRED series via the API, or the public CSV if no key is set |
| `build_yield_table(series, fetch)` | Fetch and align several series into a raw table |
| `clean_yield_table(raw, max_gap=5)` | Sort, dedupe, drop holidays, fill short gaps |
| `fill_short_gaps(s, max_gap=5)` | Forward-fill only short gaps that follow a valid value |
| `validate_yield_table(table)` | Raise `ValueError` if the table breaks the pipeline's rules |
| `curve_on(table, date)` | The curve on the last trading day on or before `date` (named by the day used) |
| `MATURITY_YEARS` | Maturity label → years (e.g. `"3M"` → `0.25`), for curve fitting |

### Python API — `yieldcurve.plots`

| Function | Description |
|----------|-------------|
| `curve_snapshots(table, dates)` | Curves on several labelled dates, maturity on a log axis |
| `yield_heatmap(table, freq="ME")` | Time × maturity heatmap of period-average yields |
| `spread_chart(spreads, recessions, freq=None)` | Spreads over time with a zero line and recession shading |
| `yield_history(table, freq=None)` | One line per maturity column over time |
| `line_chart(series, title, y_title)` | Named time series as lines, for measures that are not yields (RMSE, factors) |
| `curve_fit(observed, fits)` | One day's observed yields as markers with fitted curves as lines |

### Python API — `yieldcurve.spreads`

| Function / constant | Description |
|---------------------|-------------|
| `spread(table, long="10Y", short="2Y")` | Long minus short yield in percentage points (negative = inverted) |
| `find_inversions(spread, min_days=10, merge_gap=5)` | Inversion episodes: start, end, trading days, deepest spread and its date, ongoing flag |
| `recession_periods()` | NBER recessions as start/end dates (FRED shading convention) |

### Python API — `yieldcurve.nelson_siegel`

Maturities are in months, as in Diebold and Li (2006).

| Function / constant | Description |
|---------------------|-------------|
| `ns_loadings(maturities, lam=0.0609)` | Level, slope and curvature loadings for each maturity |
| `fit_ns_fixed(maturities, yields, lam=0.0609)` | OLS fit of the three betas with λ fixed; returns an `NSFit` (betas, λ, fitted curve, residuals, `rmse`) |
| `fit_ns(maturities, yields, bounds=(0.01, 0.6))` | Also optimizes λ: log-spaced grid, then a bounded SciPy refinement, with OLS betas at every step. Never fits worse than the fixed-λ fit |
| `fit_ns_daily(table, lam=0.0609)` | Fit every trading day of a yield table (`lam=None` optimizes λ per day): betas, λ, RMSE in bp, points fitted |
| `fitted_yields(factors, labels)` | Rebuild the fitted curves from a daily fit table, e.g. to get residuals per maturity |
| `ns_curve(maturities, betas, lam)` | Evaluate the model curve for given betas and λ |
| `load_ns_factors(refresh=False)` | Daily fixed-λ fits of the US curve since 2001-07-31, served from `data/us_ns_factors.parquet` |
| `get_ns_factors(date)` | The fit on the last trading day on or before `date` |
| `DIEBOLD_LI_LAMBDA` | 0.0609, which puts the curvature loading's peak near 30 months |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- A **FRED API key** is optional. Without one, the pipeline falls back to FRED's public CSV download. You can get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html).

### Install

```bash
git clone https://github.com/Markkndr/yield-curve-lab.git
cd yield-curve-lab

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -e ".[dev]"
```

### Configure (optional)

```bash
cp .env.example .env
# then set FRED_API_KEY=... in .env
```

> 🔐 `.env` is git-ignored. Keep API keys out of version control.

### Build the data cache

```bash
python -m yieldcurve.data      # downloads both markets into data/*.parquet
python -m yieldcurve.data hu   # or just one: us | hu
python -m yieldcurve.nelson_siegel   # daily Nelson-Siegel fits into data/us_ns_factors.parquet
```

```python
from yieldcurve.data import load_hu_yields, load_us_yields

yields = load_us_yields()      # dates × maturities, in percent
yields.loc["2022-10-24"]       # the curve on one day

hu = load_hu_yields()          # same shape, Hungarian benchmark yields
```

### Run the notebooks

```bash
jupyter execute --inplace notebooks/01_eda.ipynb   # re-run top to bottom from the cache
jupyter execute --inplace notebooks/02_nelson_siegel.ipynb
```

Or open them in VS Code / JupyterLab. Figures are interactive there; a static PNG copy is saved alongside so they also render on GitHub.

### Run the tests

```bash
pytest                         # offline tests
pytest -m network              # live FRED and ÁKK checks
ruff check .
```

---

## 📂 Project Structure

```
yield-curve-lab/
├── src/yieldcurve/
│   ├── config.py          # Project paths and FRED API key loading
│   ├── data.py            # Fetch → clean → validate → Parquet cache
│   ├── factors.py         # Empirical level, slope and curvature proxies
│   ├── nelson_siegel.py   # Nelson-Siegel loadings and fits, from scratch
│   ├── plots.py           # Plotly figures shared by notebooks and the dashboard
│   └── spreads.py         # Term spreads, inversion episodes, NBER recessions
├── tests/                 # One test module per source module
├── data/                  # Parquet cache (git-ignored, rebuilt by the pipeline)
├── notebooks/
│   ├── 01_eda.ipynb       # Coverage, key-date curves, heatmap, spreads and inversions, Hungary 2022–23
│   └── 02_nelson_siegel.ipynb  # Loadings, daily fits, fit error, factors vs. empirical proxies
├── .env.example           # Template for FRED_API_KEY
├── pyproject.toml
└── LICENSE
```

Planned additions: `pca.py`, `forecast.py`, and a `genai/` package under `src/yieldcurve/`, plus `api/`, `app/`, and `eval/` at the top level.

---

## 📚 Further Reading

- Diebold, F. X. & Li, C. (2006). *Forecasting the term structure of government bond yields.* Journal of Econometrics.
- Litterman, R. & Scheinkman, J. (1991). *Common factors affecting bond returns.* Journal of Fixed Income.
- Gürkaynak, R. S., Sack, B. & Wright, J. H. (2007). *The U.S. Treasury yield curve: 1961 to the present.* The Fed's fitted curve, used here as a benchmark.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

Reading the curve, one basis point at a time. 📈

</div>
