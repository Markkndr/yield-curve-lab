"""Plotly figures for `dates x maturities` yield tables.

Shared by the EDA notebooks and the dashboard. Every function returns a
`go.Figure` styled with the "yieldcurve" template, so callers only decide
what to plot, not how it looks.
"""

from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from yieldcurve.data import MATURITY_YEARS, curve_on

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Categorical slots in fixed order (CVD-validated as a sequence; never reorder).
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# One-hue blue ramp, light -> dark, for magnitude (heatmaps).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]

_AXIS = dict(
    gridcolor=GRID,
    gridwidth=1,
    linecolor=BASELINE,
    zeroline=False,
    tickfont=dict(color=INK_MUTED),
    title=dict(font=dict(color=INK_SECONDARY)),
)

pio.templates["yieldcurve"] = go.layout.Template(
    layout=dict(
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13),
        title=dict(x=0, xref="paper", xanchor="left", font=dict(size=17)),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=SERIES_COLORS,
        xaxis=_AXIS,
        yaxis=_AXIS,
        legend=dict(font=dict(color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BASELINE, font=dict(color=INK)),
        margin=dict(l=60, r=30, t=70, b=50),
    )
)
TEMPLATE = "yieldcurve"


def curve_snapshots(
    table: pd.DataFrame,
    dates: Mapping[str, str | pd.Timestamp],
    title: str = "Yield curve on key dates",
) -> go.Figure:
    """Plot the curve on each date (rolled back to the last trading day).

    `dates` maps a legend label to a date; the trading day actually used is
    appended to the label. Maturity is on a log axis so the short end is legible.
    """
    fig = go.Figure()
    for label, date in dates.items():
        curve = curve_on(table, date)
        fig.add_trace(
            go.Scatter(
                x=[MATURITY_YEARS[m] for m in curve.index],
                y=curve.to_numpy(),
                customdata=curve.index,
                name=f"{label} ({curve.name:%Y-%m-%d})",
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=8, line=dict(width=2, color=SURFACE)),
                hovertemplate="%{customdata}: %{y:.2f}%",
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        hovermode="x unified",
        xaxis=dict(
            title="Maturity",
            type="log",
            tickvals=list(MATURITY_YEARS.values()),
            ticktext=list(MATURITY_YEARS),
        ),
        yaxis=dict(title="Yield (%)", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
        margin=dict(b=120),
    )
    return fig


def yield_heatmap(
    table: pd.DataFrame,
    freq: str = "ME",
    title: str = "Yields by maturity over time",
) -> go.Figure:
    """Heatmap of yields (time x maturity), averaged per `freq` period.

    Averaging keeps the figure light; maturities not yet issued stay blank.
    """
    resampled = table.resample(freq).mean()
    fig = go.Figure(
        go.Heatmap(
            x=resampled.index,
            y=list(resampled.columns),
            z=resampled.T.to_numpy(),
            colorscale=SEQUENTIAL_BLUE,
            zmin=0,
            colorbar=dict(title=dict(text="Yield (%)"), outlinewidth=0, thickness=12),
            hoverongaps=False,
            hovertemplate="%{x|%b %Y} · %{y}: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="Maturity", showgrid=False, type="category"),
    )
    return fig
