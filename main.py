"""Quick start: plot the bundled samples. Run `python main.py [range|time]`."""
import sys

import pandas as pd

from OrderFlow import OrderFlowChart

which = sys.argv[1] if len(sys.argv) > 1 else "range"

if which == "range":
    # Range bars: several candles can share a timestamp, so an identifier
    # column links each footprint row to its candle.
    orderflow_data = pd.read_csv("data/range_candles.csv",
                                 names=["bid_size", "price", "ask_size", "identifier"],
                                 index_col=0, parse_dates=True)
    ohlc_data = pd.read_csv("data/range_ohlc.csv",
                            names=["open", "high", "low", "close", "identifier"],
                            index_col=0, parse_dates=True)
    chart = OrderFlowChart(orderflow_data, ohlc_data, identifier_col="identifier")
else:
    # Time bars: the timestamp links footprint rows to candles; no identifier needed.
    orderflow_data = pd.read_csv("data/time_candles.csv", index_col=0, parse_dates=True)
    ohlc_data = pd.read_csv("data/time_ohlc.csv", index_col=0, parse_dates=True)
    chart = OrderFlowChart(orderflow_data, ohlc_data)

chart.plot()
