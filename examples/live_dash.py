"""Live-updating footprint chart with Dash.

OrderflowChart itself only builds figures; for live data, rebuild the figure
from your latest window of data on a timer and send it to the browser. The
figure sets `uirevision`, so the viewer's pan/zoom survives each refresh.

This demo replays the bundled range-bar sample, revealing one candle per
second. Replace `fetch_latest()` with reads from your own feed (a websocket
buffer, a database, a KiviDB/Redis stream, ...).

    pip install dash
    python examples/live_dash.py      # then open http://127.0.0.1:8050
"""
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from OrderFlow import OrderFlowChart
from OrderFlow._figure import PLOTLY_CONFIG

DATA = Path(__file__).resolve().parent.parent / "data"
WINDOW = 60  # candles kept on screen

ORDERFLOW = pd.read_csv(DATA / "range_candles.csv",
                        names=["bid_size", "price", "ask_size", "identifier"],
                        index_col=0, parse_dates=True)
OHLC = pd.read_csv(DATA / "range_ohlc.csv",
                   names=["open", "high", "low", "close", "identifier"],
                   index_col=0, parse_dates=True)


def fetch_latest(tick: int):
    """Return (orderflow, ohlc) for the most recent WINDOW candles."""
    end = min(20 + tick, len(OHLC))
    ohlc = OHLC.iloc[max(0, end - WINDOW):end]
    orderflow = ORDERFLOW[ORDERFLOW["identifier"].isin(ohlc["identifier"])]
    return orderflow, ohlc


chart = OrderFlowChart(*fetch_latest(0), identifier_col="identifier")

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id="footprint", config=PLOTLY_CONFIG, style={"height": "95vh"}),
    dcc.Interval(id="tick", interval=1000),
], style={"backgroundColor": "#1e1f24"})


@app.callback(Output("footprint", "figure"), Input("tick", "n_intervals"))
def refresh(n):
    chart.update(*fetch_latest(n or 0))
    return chart.build_figure(title="Live order flow (replay)")


if __name__ == "__main__":
    app.run(debug=False)
