<div align="center">

# 📈 Yield Curve Lab

### Model government bond yield curves, decompose their moves, and ask questions about them in plain English

A Python research project on US and Hungarian government bond yield curves: a clean data pipeline, Nelson-Siegel curve fitting built from scratch, PCA on daily yield changes, and a GenAI layer (RAG + LangGraph agent) that combines the numbers with what the Fed and the MNB actually said.

<br>

<img src="https://img.shields.io/badge/Status-In_Progress-orange?style=for-the-badge" alt="Status: In Progress">
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

What runs today is the **data pipeline**: a single Python module that fetches every Treasury maturity from FRED, aligns and cleans them, validates the result, and caches it as Parquet. Later analysis reads the cached table instead of calling FRED again.

```
  FRED  ──►  fetch_fred_series  ──►  build_yield_table  ──►  clean_yield_table  ──►  validate_yield_table
  (API with key, or                  (align 11 series)       (drop holidays,         (sorted dates, known
   public CSV fallback)                                       fill short gaps)        maturities, sane range)
                                                                                              │
                                    load_us_yields()  ◄──  data/us_treasury_yields.parquet  ◄─┘
```

### Target architecture (vision)

The full project adds the components below. **None of these are implemented yet.** They describe the roadmap, not the current build.

- **Hungarian data source:** ÁKK benchmark yields through the same clean-and-cache pipeline
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
| **Data source** | FRED via `fredapi`, with a public-CSV fallback when no key is set |
| **Config** | `python-dotenv` (`.env`) |
| **Quality** | pytest, ruff |

**Planned (roadmap):**

| Layer | Technology |
|-------|-----------|
| **Modeling** | SciPy (optimization), scikit-learn (PCA), statsmodels (AR models) |
| **Visualization** | Plotly, Streamlit |
| **GenAI** | LangChain, LangGraph, an LLM provider SDK, Chroma |
| **Serving** | FastAPI, Uvicorn, Docker Compose |
| **Observability & eval** | Langfuse, GitHub Actions |

> The modeling and plotting libraries are already installed with the project, but no analysis modules use them yet.

---

## ✅ Current Status

The foundation (Phases 0 and 1, US data) is in place. Implemented so far:

- ⚙️ **Project setup:** `pyproject.toml` with a `src/` layout, pytest, ruff, and `.env`-based configuration
- 📥 **FRED ingestion:** all 11 constant-maturity Treasury series (`DGS1MO` … `DGS30`), through the API when a key is set, otherwise through FRED's public CSV download
- 🧹 **Cleaning:**
  - Drops market holidays (dates where every maturity is missing)
  - Forward-fills gaps of up to 5 trading days, avoiding look-ahead
  - Leaves structural gaps missing: the 20Y was not issued in 1987–1993, and the 1M only starts in 2001
- ✔️ **Validation:** sorted, unique dates; known maturity columns; yields within −5% to 30%
- 💾 **Parquet cache:** 16,000+ trading days from 1962 onward, with the full 1M–30Y curve available from July 2001
- 🧪 **Tests:** offline unit tests with a fake FRED fetcher, plus a live FRED check behind a `network` marker
- 🔎 **EDA (Phase 2, in progress):** [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) covers data coverage, the curve on key dates (2007, 2008, the 2020 low, the 2022 hiking peak, latest) and a yield heatmap since 1962, using reusable Plotly helpers in `yieldcurve.plots`

> 📝 Note: Hungarian (ÁKK) yields are not in yet. ÁKK's statistics pages load their data in the browser, so a reliable download source is still being worked out.

### Python API — `yieldcurve.data`

| Function / constant | Description |
|---------------------|-------------|
| `load_us_yields(refresh=False)` | The clean US Treasury table, served from the Parquet cache when present |
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
python -m yieldcurve.data      # downloads all series and writes data/us_treasury_yields.parquet
```

```python
from yieldcurve.data import load_us_yields

yields = load_us_yields()      # dates × maturities, in percent
yields.loc["2022-10-24"]       # the curve on one day
```

### Run the notebooks

```bash
jupyter execute --inplace notebooks/01_eda.ipynb   # re-run top to bottom from the cache
```

Or open them in VS Code / JupyterLab. Figures are interactive there; a static PNG copy is saved alongside so they also render on GitHub.

### Run the tests

```bash
pytest                         # offline tests
pytest -m network              # live FRED check
ruff check .
```

---

## 📂 Project Structure

```
yield-curve-lab/
├── src/yieldcurve/
│   ├── config.py          # Project paths and FRED API key loading
│   ├── data.py            # Fetch → clean → validate → Parquet cache
│   └── plots.py           # Plotly figures shared by notebooks and the dashboard
├── tests/
│   ├── test_config.py
│   ├── test_data.py
│   └── test_plots.py
├── data/                  # Parquet cache (git-ignored, rebuilt by the pipeline)
├── notebooks/
│   └── 01_eda.ipynb       # Coverage, key-date curves, yield heatmap
├── .env.example           # Template for FRED_API_KEY
├── pyproject.toml
└── LICENSE
```

Planned additions: `nelson_siegel.py`, `pca.py`, `forecast.py`, and a `genai/` package under `src/yieldcurve/`, plus `api/`, `app/`, and `eval/` at the top level.

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
