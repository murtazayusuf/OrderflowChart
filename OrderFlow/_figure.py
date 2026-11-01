"""Plotly figure construction.

Layout of one candle slot (x is the integer bar index i, all in data units so
it stays aligned at any zoom level):

    i-0.45             i+0.12 i+0.16 i+0.20        i+0.46
      |  bid  |  ask cells  |  candle  | volume profile |

The candle sits beside the cells rather than on top of them so it never hides
the numbers.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CELL_LEFT, CELL_RIGHT = -0.45, 0.12
CANDLE_X = 0.16
PROFILE_BASE, PROFILE_MAX = 0.20, 0.26
_GRID_WARN_CELLS = 4_000_000

BG = "#1e1f24"
GREEN, RED = "#26a69a", "#ef5350"


def _fmt_size(v: float) -> str:
    if not np.isfinite(v):
        return ""
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}k"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _decimals(tick: float) -> int:
    for d in range(0, 9):
        if abs(round(tick, d) - tick) < 1e-9:
            return d
    return 8


def _tick_labels(bars: pd.DataFrame):
    times = pd.to_datetime(bars["time"])
    if times.isna().all():
        return [str(k) for k in bars["key"]]
    t = times.dropna()
    fmt = "%H:%M" if t.dt.normalize().nunique() <= 1 else "%m-%d %H:%M"
    if (t.dt.second != 0).any():
        fmt += ":%S"
    labels = times.dt.strftime(fmt).fillna("").tolist()
    # Range/volume bars share timestamps: label repeats with an ordinal.
    seen = {}
    out = []
    for lab in labels:
        n = seen.get(lab, 0)
        seen[lab] = n + 1
        out.append(lab if n == 0 else f"{lab} #{n + 1}")
    return out


def _footprint_heatmap(fp: pd.DataFrame, n_bars: int, tick: float):
    pmin, pmax = fp["price"].min(), fp["price"].max()
    n_rows = int(round((pmax - pmin) / tick)) + 1
    n_cols = 2 * n_bars - 1
    if n_rows * n_cols > _GRID_WARN_CELLS:
        warnings.warn(
            f"Footprint grid is {n_rows} price levels x {n_bars} candles; the figure will be "
            "large and slow. Plot a slice of the data or use a coarser tick_size.",
            stacklevel=4,
        )
    rows = np.round((fp["price"].values - pmin) / tick).astype(int)
    cols = 2 * fp["bar"].values

    z = np.full((n_rows, n_cols), np.nan)
    text = np.full((n_rows, n_cols), "", dtype=object)
    z[rows, cols] = fp["imbalance"].values
    bid_txt = [_fmt_size(v) for v in fp["bid_size"].values]
    ask_txt = [_fmt_size(v) for v in fp["ask_size"].values]
    text[rows, cols] = [f"{b:>5} {a:<5}" for b, a in zip(bid_txt, ask_txt)]

    x_edges = []
    for i in range(n_bars):
        x_edges += [i + CELL_LEFT, i + CELL_RIGHT]
    x_edges = np.array(x_edges)
    y_edges = pmin + (np.arange(n_rows + 1) - 0.5) * tick

    return go.Heatmap(
        x=x_edges, y=y_edges, z=z, text=text,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Courier New, monospace"},
        colorscale="icefire_r", zmin=-1, zmax=1, showscale=False,
        hoverongaps=False, name="Bid x Ask", showlegend=True,
        hovertemplate=(f"Price %{{y:.{_decimals(tick)}f}}<br>Bid x Ask %{{text}}"
                       "<br>Imbalance %{z:.2f}<extra></extra>"),
    )


def _candle_traces(bars: pd.DataFrame):
    traces = []
    for up, color in ((True, GREEN), (False, RED)):
        sel = bars[(bars["close"] >= bars["open"]) == up]
        if sel.empty:
            continue
        x = np.repeat(sel["bar"].values + CANDLE_X, 3).astype(float)
        wick = np.column_stack([sel["low"], sel["high"], np.full(len(sel), np.nan)]).ravel()
        body = np.column_stack([sel["open"], sel["close"], np.full(len(sel), np.nan)]).ravel()
        common = dict(x=x, mode="lines", hoverinfo="skip", legendgroup="candle")
        traces.append(go.Scatter(y=wick, line=dict(color=color, width=1), name="Candle",
                                 showlegend=up, **common))
        traces.append(go.Scatter(y=body, line=dict(color=color, width=5), name="Candle",
                                 showlegend=False, **common))
    return traces


def _profile_trace(fp: pd.DataFrame, stats: pd.DataFrame, tick: float):
    vol = fp["bid_size"] + fp["ask_size"]
    vmax = vol.groupby(fp["bar"]).transform("max").replace(0, np.nan)
    length = (vol / vmax).fillna(0).values * PROFILE_MAX
    poc = stats.set_index("bar")["poc"]
    is_poc = np.isclose(fp["price"].values, poc.reindex(fp["bar"]).values)
    colors = np.where(is_poc, "rgba(255, 213, 79, 0.85)", "rgba(144, 164, 174, 0.55)")
    return go.Bar(
        orientation="h", y=fp["price"], x=length, base=fp["bar"].values + PROFILE_BASE,
        width=tick * 0.85, marker=dict(color=colors, line=dict(width=0)),
        name="Volume profile", hovertemplate="Volume %{customdata}<extra></extra>",
        customdata=[_fmt_size(v) for v in vol.values],
    )


STAT_ROWS = [("volume", "Volume"), ("delta", "Delta"),
             ("cum_delta", "Cum. delta"), ("delta_pct", "Delta %")]


def _stats_heatmap(stats: pd.DataFrame):
    zs, texts, ys = [], [], []
    for col, label in STAT_ROWS:
        v = stats[col].astype(float).values
        scale = np.nanmax(np.abs(v)) if len(v) and np.nanmax(np.abs(v)) > 0 else 1.0
        zs.append(v / scale)
        texts.append([f"{x:.0f}%" if col == "delta_pct" else _fmt_size(x) for x in v])
        ys.append(label)
    return go.Heatmap(
        x=stats["bar"].values, y=ys, z=np.vstack(zs), text=np.array(texts, dtype=object),
        texttemplate="%{text}", textfont={"size": 10}, colorscale="RdYlGn",
        zmin=-1, zmax=1, showscale=False, xgap=2, ygap=2, name="Stats",
        hovertemplate="%{y}: %{text}<extra></extra>", showlegend=False,
    )


def build_figure(bars: pd.DataFrame, fp: pd.DataFrame, stats: pd.DataFrame, tick: float,
                 visible_bars: int = 8, visible_levels: int = 40,
                 title: str = "Order Flow", height: int = 780) -> go.Figure:
    n = len(bars)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.86, 0.14])
    fig.add_trace(_footprint_heatmap(fp, n, tick), row=1, col=1)
    fig.add_trace(_profile_trace(fp, stats, tick), row=1, col=1)
    for t in _candle_traces(bars):
        fig.add_trace(t, row=1, col=1)
    fig.add_trace(_stats_heatmap(stats), row=2, col=1)

    # Initial viewport: the last `visible_bars` candles and at most
    # `visible_levels` price levels (more than ~45 rows makes the numbers
    # unreadable), centred on the latest candle. Pan/scroll to see the rest.
    first = max(0, n - visible_bars)
    view = bars.iloc[first:]
    last = bars.iloc[-1]
    lo, hi = view["low"].min(), view["high"].max()
    span = min(hi - lo + 4 * tick, visible_levels * tick)
    span = max(span, min(visible_levels, 20) * tick)
    mid = (last["high"] + last["low"]) / 2
    y_range = [mid - span / 2, mid + span / 2]
    x_range = [first - 0.5, n - 0.5]

    labels = _tick_labels(bars)
    dec = _decimals(tick)
    fig.update_layout(
        title=title, height=height, template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG, dragmode="pan",
        margin=dict(l=10, r=10, t=50, b=20), bargap=0, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hovermode="closest", spikedistance=1000, hoverdistance=50,
        # Keeps the user's pan/zoom when the figure is re-sent with new data
        # (Dash, FigureWidget), instead of snapping back to the default view.
        uirevision="orderflowchart",
    )
    fig.update_xaxes(showgrid=False, tickmode="array", tickvals=bars["bar"].tolist(),
                     ticktext=labels, range=x_range, showspikes=True, spikecolor="white",
                     spikesnap="cursor", spikemode="across", spikethickness=0.5)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=1, title="Price", range=y_range,
                     tickformat=f".{dec}f", showspikes=True, spikecolor="white",
                     spikesnap="cursor", spikemode="across", spikethickness=0.5)
    fig.update_yaxes(showgrid=False, fixedrange=True, row=2, col=1, autorange="reversed")
    return fig


PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["zoomIn", "zoomOut", "autoScale"],
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
}
