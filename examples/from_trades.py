"""Build a footprint chart straight from a trade tape.

Each trade needs a timestamp, price, size and aggressor side. Here we
generate a synthetic tape; with real data, load your trades into the same
shape (DatetimeIndex + price/size/side columns).
"""
import numpy as np
import pandas as pd

from OrderFlow import OrderFlowChart

rng = np.random.default_rng(7)
n = 30_000
trades = pd.DataFrame(
    {
        "price": np.round((4500 + np.cumsum(rng.normal(0, 0.15, n))) / 0.25) * 0.25,
        "size": rng.integers(1, 20, n),
        "side": rng.choice(["buy", "sell"], n),
    },
    index=pd.date_range("2024-03-01 09:30", periods=n, freq="200ms", tz="America/New_York"),
)

chart = OrderFlowChart.from_trades(trades, timeframe="5min", tick_size=0.25)
print(chart.stats.tail())
chart.plot()
