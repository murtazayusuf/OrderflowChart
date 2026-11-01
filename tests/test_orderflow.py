import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from OrderFlow import OrderFlowChart, OrderFlowDataError

DATA = Path(__file__).resolve().parent.parent / "data"


def range_data():
    of = pd.read_csv(DATA / "range_candles.csv", names=["bid_size", "price", "ask_size", "identifier"],
                     index_col=0, parse_dates=True)
    ohlc = pd.read_csv(DATA / "range_ohlc.csv", names=["open", "high", "low", "close", "identifier"],
                       index_col=0, parse_dates=True)
    return of, ohlc


def time_data():
    of = pd.read_csv(DATA / "time_candles.csv", index_col=0, parse_dates=True)
    ohlc = pd.read_csv(DATA / "time_ohlc.csv", index_col=0, parse_dates=True)
    return of, ohlc


def tiny(ascending=True):
    """Two 1-minute candles, two price levels each."""
    t0, t1 = pd.Timestamp("2024-01-02 09:30"), pd.Timestamp("2024-01-02 09:31")
    ohlc = pd.DataFrame({"open": [10.0, 10.01], "high": [10.01, 10.02],
                         "low": [10.0, 10.01], "close": [10.01, 10.02]}, index=[t0, t1])
    of = pd.DataFrame({"bid_size": [5, 1, 0, 4], "price": [10.0, 10.01, 10.01, 10.02],
                       "ask_size": [2, 3, 0, 6]}, index=[t0, t0, t1, t1])
    if not ascending:
        ohlc, of = ohlc.iloc[::-1], of.iloc[::-1]
    return of, ohlc


# ---------------------------------------------------------------- examples
def test_range_example_renders():
    of, ohlc = range_data()
    c = OrderFlowChart(of, ohlc, identifier_col="identifier")
    fig = c.plot(return_figure=True)
    assert c.tick_size == 0.25
    assert len(c.bars) == 476
    # input order is preserved for candles sharing a timestamp
    assert list(c.bars["key"][:3]) == list(ohlc["identifier"][:3])
    assert fig.layout.xaxis.range[1] == pytest.approx(len(c.bars) - 0.5)


def test_time_indexed_without_identifier_issue_4():
    of, ohlc = time_data()  # no identifier column, tz-aware, ohlc sorted descending
    c = OrderFlowChart(of, ohlc, identifier_col=None)
    c.plot(return_figure=True)
    assert c.bars["time"].is_monotonic_increasing
    assert c.bars["time"].dt.tz is not None
    assert len(c.bars) == 60


def test_candle_order_independent_of_input_order_issue_6():
    a = OrderFlowChart(*tiny(True))
    b = OrderFlowChart(*tiny(False))
    pd.testing.assert_frame_equal(a.bars, b.bars)
    pd.testing.assert_frame_equal(a.footprint, b.footprint)


def test_small_tick_has_nonempty_y_range_issue_2():
    c = OrderFlowChart(*tiny())
    assert c.tick_size == pytest.approx(0.01)
    lo, hi = c.build_figure().layout.yaxis.range
    assert hi - lo >= 10 * c.tick_size


# ---------------------------------------------------------------- metrics
def test_diagonal_imbalance_stays_within_candle():
    c = OrderFlowChart(*tiny())
    fp = c.footprint.set_index(["bar", "price"])["imbalance"]
    # bar0 @10.00: bid 5 vs ask@10.01 3 -> (5-3)/8
    assert fp[(0, 10.0)] == pytest.approx(0.25)
    # top level of bar0 has nothing above it in the SAME candle -> bid only
    assert fp[(0, 10.01)] == pytest.approx(1.0)
    # zero volume level -> 0, not NaN or a forward-filled neighbour
    assert fp[(1, 10.01)] == pytest.approx((0 - 6) / 6)
    assert c.footprint["imbalance"].between(-1, 1).all()


def test_horizontal_imbalance_and_custom_column():
    of, ohlc = tiny()
    h = OrderFlowChart(of, ohlc, imbalance="horizontal")
    assert h.footprint["imbalance"].iloc[0] == pytest.approx((5 - 2) / 7)
    of = of.assign(my_imb=0.5)
    c = OrderFlowChart(of, ohlc, imbalance_col="my_imb")
    assert (c.footprint["imbalance"] == 0.5).all()


def test_stats():
    c = OrderFlowChart(*tiny())
    s = c.stats
    assert list(s["volume"]) == [11, 10]
    assert list(s["delta"]) == [-1, 2]
    assert list(s["cum_delta"]) == [-1, 1]
    assert list(s["poc"]) == [10.0, 10.02]


# ---------------------------------------------------------------- errors
def test_duplicate_timestamps_need_identifier():
    of, ohlc = range_data()
    with pytest.raises(OrderFlowDataError, match="identifier_col"):
        OrderFlowChart(of, ohlc.drop(columns="identifier")).process_data()


def test_unmatched_keys_message():
    of, ohlc = tiny()
    of.index = of.index.tz_localize("UTC")
    with pytest.raises(OrderFlowDataError, match="timezone"):
        OrderFlowChart(of, ohlc).process_data()


def test_missing_columns_message():
    of, ohlc = tiny()
    with pytest.raises(OrderFlowDataError, match="bid_size"):
        OrderFlowChart(of.drop(columns="bid_size"), ohlc).process_data()


def test_partial_match_warns_and_drops():
    of, ohlc = tiny()
    extra = pd.DataFrame({"bid_size": [1], "price": [9.0], "ask_size": [1]},
                         index=[pd.Timestamp("2030-01-01")])
    with pytest.warns(UserWarning, match="1 orderflow rows"):
        c = OrderFlowChart(pd.concat([of, extra]), ohlc)
        c.process_data()
    assert len(c.footprint) == 4


def test_time_column_instead_of_index():
    of, ohlc = tiny()
    c = OrderFlowChart(of.rename_axis("time").reset_index(), ohlc.rename_axis("time").reset_index())
    assert len(c.bars) == 2


# ---------------------------------------------------------------- serialisation
def test_processed_roundtrip_through_json():
    of, ohlc = time_data()
    c = OrderFlowChart(of, ohlc)
    d = json.loads(json.dumps(c.get_processed_data()))
    c2 = OrderFlowChart.from_preprocessed_data(d)
    pd.testing.assert_frame_equal(c.footprint, c2.footprint)
    assert c2.tick_size == c.tick_size
    c2.build_figure()


def test_legacy_v01_json_still_loads():
    with open(DATA / "preprocessed_data.json") as f:
        legacy = json.load(f)
    c = OrderFlowChart.from_preprocessed_data(legacy)
    assert len(c.bars) == 10
    assert list(c.bars["key"][:2]) == ["zlEKR", "BsDeTP"]
    c.build_figure()


def test_invalid_preprocessed_data():
    with pytest.raises(OrderFlowDataError):
        OrderFlowChart.from_preprocessed_data({"nope": 1})


# ---------------------------------------------------------------- trades
def test_from_trades():
    idx = pd.date_range("2024-01-02 09:30", periods=6, freq="20s", tz="America/New_York")
    trades = pd.DataFrame({"price": [10.0, 10.01, 10.0, 10.02, 10.02, 10.01],
                           "size": [1, 2, 3, 4, 5, 6],
                           "side": ["buy", "SELL", "b", "s", 1, -1]}, index=idx)
    with pytest.raises(OrderFlowDataError):  # mixed str/int sides are rejected
        OrderFlowChart.from_trades(trades, "1min").process_data()
    trades["side"] = ["buy", "SELL", "b", "s", "buy", "sell"]
    c = OrderFlowChart.from_trades(trades, "1min")
    assert len(c.bars) == 2
    assert c.bars["time"].dt.tz is not None
    b0 = c.bars.iloc[0]
    assert (b0.open, b0.high, b0.low, b0.close) == (10.0, 10.01, 10.0, 10.0)
    assert c.stats["volume"].tolist() == [6, 15]
    assert c.stats["delta"].tolist() == [1 + 3 - 2, 5 - 4 - 6]


def test_update_reprocesses():
    of, ohlc = tiny()
    c = OrderFlowChart(of.iloc[:2], ohlc.iloc[:1])
    assert len(c.bars) == 1
    c.update(of, ohlc)
    assert len(c.bars) == 2
