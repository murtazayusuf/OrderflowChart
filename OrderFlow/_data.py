"""Input normalisation and footprint metrics.

Everything here is plain pandas/numpy and independent of plotly, so it can be
tested and reused (e.g. for backtests) without rendering anything.

Internal model
--------------
bars       one row per candle, ordered oldest -> newest
           columns: bar (0..N-1), key, time, open, high, low, close
footprint  one row per (bar, price level)
           columns: bar, price, bid_size, ask_size, [imbalance]
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

OHLC_COLS = ["open", "high", "low", "close"]
FOOTPRINT_COLS = ["price", "bid_size", "ask_size"]
TIME_COL_CANDIDATES = ("time", "timestamp", "datetime", "date")

# Prices are rounded to this many decimals before comparisons so that float
# noise (14928.250000001) does not create phantom price levels.
_PRICE_DECIMALS = 10


class OrderFlowDataError(ValueError):
    """Raised when input data cannot be interpreted. The message says why."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise OrderFlowDataError(
            f"{name} is missing required column(s) {missing}. "
            f"Found columns: {list(df.columns)}."
        )


def _extract_time(df: pd.DataFrame, time_col: Optional[str], name: str) -> Optional[pd.Series]:
    """Return a datetime Series aligned to df rows, or None if no time info exists.

    Looks at, in order: explicit time_col, a DatetimeIndex, a column named
    time/timestamp/datetime/date.
    """
    if time_col is not None:
        if time_col in df.columns:
            src = df[time_col]
        elif df.index.name == time_col:
            src = df.index.to_series()
        else:
            raise OrderFlowDataError(f"{name}: time column '{time_col}' not found.")
        return _as_time_series(src, df.index)

    if isinstance(df.index, pd.DatetimeIndex):
        return _as_time_series(df.index, df.index)

    for cand in TIME_COL_CANDIDATES:
        if cand in df.columns:
            try:
                return _as_time_series(df[cand], df.index)
            except (ValueError, TypeError):
                continue
    return None


def _as_time_series(values, index) -> pd.Series:
    # Built from a DatetimeIndex so timezone information survives
    # (Series.values would silently convert tz-aware data to naive UTC).
    return pd.Series(pd.DatetimeIndex(pd.to_datetime(values)), index=index)


def round_prices(prices) -> np.ndarray:
    return np.round(np.asarray(prices, dtype=float), _PRICE_DECIMALS)


def infer_tick_size(footprint: pd.DataFrame) -> float:
    """Smallest positive gap between adjacent price levels within any bar.

    Using within-bar gaps (rather than the first two rows, as before) makes the
    result independent of row order and robust to empty levels.
    """
    fp = footprint[["bar", "price"]].drop_duplicates().sort_values(["bar", "price"])
    diffs = fp.groupby("bar")["price"].diff().dropna()
    diffs = round_prices(diffs)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        # Every bar has a single price level; fall back to global levels.
        levels = np.unique(round_prices(footprint["price"]))
        gaps = np.diff(levels)
        gaps = gaps[gaps > 0]
        if len(gaps) == 0:
            raise OrderFlowDataError(
                "Cannot infer tick size: only one price level in the data. "
                "Pass tick_size explicitly."
            )
        return float(gaps.min())
    return float(diffs.min())


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def normalize_bars(ohlc: pd.DataFrame, identifier_col: Optional[str] = None,
                   time_col: Optional[str] = None) -> pd.DataFrame:
    if ohlc is None or len(ohlc) == 0:
        raise OrderFlowDataError("ohlc_data is empty.")
    _require_columns(ohlc, OHLC_COLS, "ohlc_data")
    times = _extract_time(ohlc, time_col, "ohlc_data")

    if identifier_col is not None:
        _require_columns(ohlc, [identifier_col], "ohlc_data")
        keys = ohlc[identifier_col].astype(str)
        if keys.duplicated().any():
            dup = keys[keys.duplicated()].iloc[0]
            raise OrderFlowDataError(
                f"ohlc_data['{identifier_col}'] must be unique per candle; "
                f"'{dup}' appears more than once."
            )
    else:
        if times is None:
            raise OrderFlowDataError(
                "No identifier_col given and ohlc_data has no timestamps. Either give "
                "ohlc_data a DatetimeIndex (or a 'time' column), or pass identifier_col."
            )
        if times.duplicated().any():
            raise OrderFlowDataError(
                "ohlc_data has duplicate timestamps, so timestamps cannot identify "
                "candles (typical for range/tick/volume bars). Add a column that is "
                "unique per candle and pass it as identifier_col."
            )
        keys = times

    bars = pd.DataFrame({
        "key": keys.reset_index(drop=True),
        "time": times.reset_index(drop=True) if times is not None else pd.NaT,
        **{c: ohlc[c].astype(float).reset_index(drop=True) for c in OHLC_COLS},
    })
    bars["_pos"] = np.arange(len(bars))
    if times is not None:
        # Stable sort: candles sharing a timestamp keep their input order.
        bars = bars.sort_values(["time", "_pos"], kind="mergesort")
    bars = bars.drop(columns="_pos").reset_index(drop=True)
    bars.insert(0, "bar", np.arange(len(bars)))
    return bars


def normalize_footprint(orderflow: pd.DataFrame, bars: pd.DataFrame,
                        identifier_col: Optional[str] = None,
                        time_col: Optional[str] = None,
                        imbalance_col: Optional[str] = None) -> pd.DataFrame:
    if orderflow is None or len(orderflow) == 0:
        raise OrderFlowDataError("orderflow_data is empty.")
    _require_columns(orderflow, FOOTPRINT_COLS, "orderflow_data")
    if imbalance_col is not None:
        _require_columns(orderflow, [imbalance_col], "orderflow_data")

    if identifier_col is not None:
        _require_columns(orderflow, [identifier_col], "orderflow_data")
        keys = orderflow[identifier_col].astype(str)
    else:
        keys = _extract_time(orderflow, time_col, "orderflow_data")
        if keys is None:
            raise OrderFlowDataError(
                "No identifier_col given and orderflow_data has no timestamps. Give it "
                "a DatetimeIndex (or a 'time' column) matching ohlc_data, or pass "
                "identifier_col."
            )

    key_to_bar = pd.Series(bars["bar"].values, index=pd.Index(bars["key"]))
    bar_idx = keys.map(key_to_bar)

    fp = pd.DataFrame({
        "bar": bar_idx.values,
        "price": round_prices(orderflow["price"]),
        "bid_size": orderflow["bid_size"].astype(float).fillna(0).values,
        "ask_size": orderflow["ask_size"].astype(float).fillna(0).values,
    })
    if imbalance_col is not None:
        fp["imbalance"] = orderflow[imbalance_col].astype(float).values

    unmatched = fp["bar"].isna()
    if unmatched.all():
        sample_of = list(pd.unique(keys))[:3]
        sample_bars = list(bars["key"].head(3))
        raise OrderFlowDataError(
            "None of the orderflow rows match a candle. "
            f"orderflow keys look like {sample_of}, ohlc keys look like {sample_bars}. "
            "Check that both use the same identifier, or the same timestamps "
            "(including timezone)."
        )
    if unmatched.any():
        warnings.warn(
            f"{int(unmatched.sum())} orderflow rows did not match any candle and were dropped.",
            stacklevel=3,
        )
        fp = fp[~unmatched]

    fp["bar"] = fp["bar"].astype(int)
    agg = {"bid_size": "sum", "ask_size": "sum"}
    if imbalance_col is not None:
        agg["imbalance"] = "mean"
    fp = fp.groupby(["bar", "price"], as_index=False).agg(agg)
    return fp.sort_values(["bar", "price"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def compute_imbalance(fp: pd.DataFrame, tick_size: float, mode: str = "diagonal") -> pd.Series:
    """Imbalance in [-1, 1] per footprint row. Positive = sellers dominate.

    diagonal   (bid[P] - ask[P+tick]) / (bid[P] + ask[P+tick])
               Sellers hitting the bid at P are compared with buyers lifting the
               offer one tick above, which is how footprint imbalances are read.
               This is the formula the original version intended; it is now
               computed within each candle and by price, not by row position.
    horizontal (bid[P] - ask[P]) / (bid[P] + ask[P])
    """
    if mode == "horizontal":
        num = fp["bid_size"] - fp["ask_size"]
        den = fp["bid_size"] + fp["ask_size"]
    elif mode == "diagonal":
        above = fp[["bar", "price", "ask_size"]].copy()
        above["price"] = round_prices(above["price"] - tick_size)
        above = above.rename(columns={"ask_size": "ask_above"})
        merged = fp[["bar", "price", "bid_size"]].merge(above, on=["bar", "price"], how="left")
        ask_above = merged["ask_above"].fillna(0).values
        num = fp["bid_size"].values - ask_above
        den = fp["bid_size"].values + ask_above
        num, den = pd.Series(num, index=fp.index), pd.Series(den, index=fp.index)
    else:
        raise ValueError("imbalance mode must be 'diagonal' or 'horizontal'")
    out = np.where(den > 0, num / den.where(den > 0, 1), 0.0)
    return pd.Series(out, index=fp.index, dtype=float)


def compute_bar_stats(fp: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Per-candle statistics shown in the lower panel."""
    g = fp.groupby("bar")
    bid = g["bid_size"].sum().reindex(bars["bar"], fill_value=0.0)
    ask = g["ask_size"].sum().reindex(bars["bar"], fill_value=0.0)
    volume = bid + ask
    delta = ask - bid
    poc = (fp.assign(vol=fp["bid_size"] + fp["ask_size"])
             .sort_values(["bar", "vol", "price"], ascending=[True, False, True])
             .drop_duplicates("bar").set_index("bar")["price"]
             .reindex(bars["bar"]))
    stats = pd.DataFrame({
        "bar": bars["bar"].values,
        "volume": volume.values,
        "delta": delta.values,
        "cum_delta": delta.cumsum().values,
        "delta_pct": np.where(volume.values > 0, delta.values / np.where(volume.values > 0, volume.values, 1) * 100, 0.0),
        "poc": poc.values,
    })
    return stats


def footprint_from_trades(trades: pd.DataFrame, timeframe: str = "1min",
                          price_col: str = "price", size_col: str = "size",
                          side_col: str = "side", time_col: Optional[str] = None,
                          tick_size: Optional[float] = None):
    """Aggregate a trade tape into (orderflow, ohlc) frames for OrderFlowChart.

    side_col holds the aggressor side: 'buy'/'sell' (any case, 'b'/'s' also
    accepted), or +1/-1, or booleans where True means buy. Aggressive buys
    are counted in ask_size, aggressive sells in bid_size.
    """
    _require_columns(trades, [price_col, size_col, side_col], "trades")
    times = _extract_time(trades, time_col, "trades")
    if times is None:
        raise OrderFlowDataError("trades need a DatetimeIndex or a time column.")

    side = trades[side_col]
    if side.dtype == bool:
        is_buy = side.values
    elif pd.api.types.is_numeric_dtype(side):
        is_buy = side.values > 0
    else:
        s = side.astype(str).str.strip().str.lower()
        buy_vals, sell_vals = {"buy", "b", "long"}, {"sell", "s", "short"}
        unknown = ~s.isin(buy_vals | sell_vals)
        if unknown.any():
            raise OrderFlowDataError(
                f"Unrecognised values in trades['{side_col}']: {sorted(s[unknown].unique())[:5]}. "
                "Use 'buy'/'sell' or +1/-1."
            )
        is_buy = s.isin(buy_vals).values

    price = trades[price_col].astype(float).values
    if tick_size:
        price = np.round(price / tick_size) * tick_size
    df = pd.DataFrame({
        "time": times.reset_index(drop=True),
        "price": round_prices(price),
        "size": trades[size_col].astype(float).values,
        "is_buy": is_buy,
    })
    df["bar_time"] = df["time"].dt.floor(timeframe)

    ohlc = (df.sort_values("time", kind="mergesort")
              .groupby("bar_time")["price"].agg(["first", "max", "min", "last"]))
    ohlc.columns = OHLC_COLS
    ohlc["volume"] = df.groupby("bar_time")["size"].sum()
    ohlc.index.name = "time"

    df["ask_size"] = np.where(df["is_buy"], df["size"], 0.0)
    df["bid_size"] = np.where(df["is_buy"], 0.0, df["size"])
    orderflow = (df.groupby(["bar_time", "price"])[["bid_size", "ask_size"]].sum()
                   .reset_index().rename(columns={"bar_time": "time"})
                   .set_index("time")[["bid_size", "price", "ask_size"]])
    return orderflow, ohlc


# --------------------------------------------------------------------------- #
# legacy (v0.1) preprocessed JSON
# --------------------------------------------------------------------------- #
def _legacy_frame(block: dict) -> pd.DataFrame:
    block = {k: v for k, v in block.items() if k != "dtypes"}
    return pd.DataFrame(block).replace("nan", np.nan)


def bars_and_footprint_from_legacy(data: dict):
    """Rebuild raw inputs from the JSON written by version 0.1's get_processed_data."""
    of = _legacy_frame(data["orderflow"])
    ohlc = _legacy_frame(data["ohlc"])
    ohlc["index"] = pd.to_datetime(ohlc["index"])
    for c in OHLC_COLS:
        ohlc[c] = ohlc[c].astype(float)
    for c in FOOTPRINT_COLS:
        of[c] = of[c].astype(float)
    # v0.1 stored a 'sequence' column (identifier string length) that it used
    # to order candles sharing a timestamp. That is not a real ordering, so it
    # is ignored; the stored row order is the input order.
    ohlc = ohlc.set_index("index")
    bars = normalize_bars(ohlc, identifier_col="identifier")
    fp = normalize_footprint(of, bars, identifier_col="identifier")
    return bars, fp
