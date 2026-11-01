"""OrderflowChart: footprint (order flow) charts with Plotly.

    from OrderFlow import OrderFlowChart
    chart = OrderFlowChart(orderflow_df, ohlc_df)          # time bars
    chart = OrderFlowChart(orderflow_df, ohlc_df, identifier_col="identifier")  # range bars
    chart.plot()
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ._data import (
    OrderFlowDataError,
    bars_and_footprint_from_legacy,
    compute_bar_stats,
    compute_imbalance,
    footprint_from_trades,
    infer_tick_size,
    normalize_bars,
    normalize_footprint,
)
from ._figure import PLOTLY_CONFIG, build_figure

__version__ = "0.2.0"
__all__ = ["OrderFlowChart", "OrderFlowDataError", "footprint_from_trades", "__version__"]

_FORMAT = "orderflowchart/2"


class OrderFlowChart:
    """Footprint chart built from per-price bid/ask volume and OHLC candles.

    Parameters
    ----------
    orderflow_data : DataFrame
        One row per (candle, price level) with columns ``price``, ``bid_size``,
        ``ask_size``. Rows are linked to candles by ``identifier_col`` or, if
        that is None, by timestamp (DatetimeIndex or a ``time`` column).
    ohlc_data : DataFrame
        One row per candle with ``open``, ``high``, ``low``, ``close`` and either a
        timestamp or ``identifier_col``. Any row order is accepted; candles are
        sorted by time, and candles sharing a timestamp keep their input order.
    identifier_col : str, optional
        Column present in both frames that uniquely identifies a candle. Needed
        for range/tick/volume bars, where several candles share a timestamp.
    imbalance_col : str, optional
        Column in orderflow_data with your own imbalance values in [-1, 1].
        If None, imbalance is computed (see ``imbalance``).
    tick_size : float, optional
        Price increment between levels. Inferred from the data if omitted.
    imbalance : {'diagonal', 'horizontal'}
        How to compute imbalance when imbalance_col is None.
    time_col : str, optional
        Name of the timestamp column if it is neither the index nor called
        time/timestamp/datetime/date.
    """

    def __init__(self, orderflow_data: Optional[pd.DataFrame] = None,
                 ohlc_data: Optional[pd.DataFrame] = None,
                 identifier_col: Optional[str] = None,
                 imbalance_col: Optional[str] = None,
                 tick_size: Optional[float] = None,
                 imbalance: str = "diagonal",
                 time_col: Optional[str] = None,
                 **kwargs):
        self.identifier_col = identifier_col
        self.imbalance_col = imbalance_col
        self.imbalance_mode = imbalance
        self.time_col = time_col
        self._tick_size = tick_size
        self.is_processed = False

        if "data" in kwargs:  # v0.1 path used by from_preprocessed_data
            self._load_processed(kwargs.pop("data"))
            return
        if kwargs:
            raise TypeError(f"Unexpected arguments: {sorted(kwargs)}")
        if orderflow_data is None or ohlc_data is None:
            raise OrderFlowDataError("orderflow_data and ohlc_data are both required.")
        self.orderflow_data = orderflow_data
        self.ohlc_data = ohlc_data

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_trades(cls, trades: pd.DataFrame, timeframe: str = "1min", *,
                    price_col: str = "price", size_col: str = "size",
                    side_col: str = "side", time_col: Optional[str] = None,
                    tick_size: Optional[float] = None, **kwargs) -> "OrderFlowChart":
        """Build a chart from a raw trade tape (one row per trade).

        ``side_col`` is the aggressor side: 'buy'/'sell' or +1/-1.
        ``timeframe`` is any pandas frequency string ('30s', '1min', '5min', '1h').
        """
        orderflow, ohlc = footprint_from_trades(
            trades, timeframe, price_col=price_col, size_col=size_col,
            side_col=side_col, time_col=time_col, tick_size=tick_size)
        return cls(orderflow, ohlc, tick_size=tick_size, **kwargs)

    @classmethod
    def from_preprocessed_data(cls, data: dict) -> "OrderFlowChart":
        """Restore a chart from ``get_processed_data()`` output (v0.2 or v0.1 format)."""
        return cls(data=data)

    def update(self, orderflow_data: pd.DataFrame, ohlc_data: pd.DataFrame) -> "OrderFlowChart":
        """Replace the data (e.g. with a newer window from a live feed) and reprocess lazily."""
        self.orderflow_data = orderflow_data
        self.ohlc_data = ohlc_data
        self.is_processed = False
        return self

    # ------------------------------------------------------------------ #
    # processing
    # ------------------------------------------------------------------ #
    def process_data(self) -> None:
        bars = normalize_bars(self.ohlc_data, self.identifier_col, self.time_col)
        fp = normalize_footprint(self.orderflow_data, bars, self.identifier_col,
                                 self.time_col, self.imbalance_col)
        self._finish(bars, fp, self._tick_size)

    def _finish(self, bars: pd.DataFrame, fp: pd.DataFrame, tick_size: Optional[float]) -> None:
        self._tick = float(tick_size) if tick_size else infer_tick_size(fp)
        if "imbalance" not in fp.columns:
            fp["imbalance"] = compute_imbalance(fp, self._tick, self.imbalance_mode)
        fp["imbalance"] = fp["imbalance"].astype(float).clip(-1, 1).fillna(0.0)
        self._bars = bars
        self._footprint = fp
        self._stats = compute_bar_stats(fp, bars)
        self.is_processed = True

    def _ensure_processed(self) -> None:
        if not self.is_processed:
            self.process_data()

    @property
    def tick_size(self) -> float:
        """Price increment between levels (given or inferred)."""
        self._ensure_processed()
        return self._tick

    @property
    def granularity(self) -> float:
        """Alias of tick_size kept for v0.1 compatibility."""
        return self.tick_size

    @property
    def bars(self) -> pd.DataFrame:
        """One row per candle, oldest first: bar, key, time, open, high, low, close."""
        self._ensure_processed()
        return self._bars

    @property
    def footprint(self) -> pd.DataFrame:
        """One row per (bar, price): bid_size, ask_size, imbalance."""
        self._ensure_processed()
        return self._footprint

    @property
    def stats(self) -> pd.DataFrame:
        """Per-candle volume, delta, cumulative delta, delta % and point of control."""
        self._ensure_processed()
        return self._stats

    # ------------------------------------------------------------------ #
    # serialisation
    # ------------------------------------------------------------------ #
    def get_processed_data(self) -> dict:
        """JSON-serialisable snapshot that ``from_preprocessed_data`` can restore."""
        self._ensure_processed()
        bars = self.bars.copy()
        bars["time"] = pd.to_datetime(bars["time"]).map(
            lambda t: None if pd.isna(t) else t.isoformat())
        bars["key"] = bars["key"].map(lambda k: k.isoformat() if hasattr(k, "isoformat") else str(k))
        return {
            "format": _FORMAT,
            "tick_size": self.tick_size,
            "imbalance_mode": self.imbalance_mode,
            "bars": bars.to_dict(orient="list"),
            "footprint": self.footprint.to_dict(orient="list"),
        }

    def _load_processed(self, data: dict) -> None:
        try:
            if data.get("format") == _FORMAT:
                bars = pd.DataFrame(data["bars"])
                bars["time"] = pd.to_datetime(bars["time"], utc=False)
                fp = pd.DataFrame(data["footprint"])
                self.imbalance_mode = data.get("imbalance_mode", self.imbalance_mode)
                self._finish(bars, fp, data.get("tick_size"))
            elif {"orderflow", "ohlc"} <= set(data):
                bars, fp = bars_and_footprint_from_legacy(data)
                self._finish(bars, fp, None)
            else:
                raise KeyError("unrecognised layout")
        except OrderFlowDataError:
            raise
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise OrderFlowDataError(
                "Invalid preprocessed data. Pass the dict returned by get_processed_data()."
            ) from exc

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def build_figure(self, visible_bars: int = 8, visible_levels: int = 40,
                     title: str = "Order Flow", height: int = 780):
        """Return the plotly Figure without displaying it."""
        self._ensure_processed()
        return build_figure(self.bars, self.footprint, self.stats, self.tick_size,
                            visible_bars=visible_bars, visible_levels=visible_levels,
                            title=title, height=height)

    def plot(self, return_figure: bool = False, **figure_kwargs):
        """Show the chart. With ``return_figure=True`` return the Figure instead."""
        fig = self.build_figure(**figure_kwargs)
        if return_figure:
            return fig
        fig.show(config=PLOTLY_CONFIG)
        return None
