# Sample data

| File | Contents |
|---|---|
| `range_candles.csv`, `range_ohlc.csv` | Range bars. No header; columns are timestamp, bid_size, price, ask_size, identifier (footprint) and timestamp, open, high, low, close, identifier (OHLC). Several candles share a timestamp, so `identifier` links the two files. |
| `time_candles.csv`, `time_ohlc.csv` | 1-minute bars with a header row and UTC timestamps. Linked by timestamp; no identifier needed. |
| `preprocessed_data.json` | Output of `get_processed_data()` from version 0.1, kept to test that old files still load. |

## Preprocessed format (0.2)

`OrderFlowChart.get_processed_data()` returns a JSON-serialisable dict:

```
{
  "format": "orderflowchart/2",
  "tick_size": 0.25,
  "imbalance_mode": "diagonal",
  "bars":      {"bar": [...], "key": [...], "time": [ISO 8601 or null], "open": [...], "high": [...], "low": [...], "close": [...]},
  "footprint": {"bar": [...], "price": [...], "bid_size": [...], "ask_size": [...], "imbalance": [...]}
}
```

Each key in `bars` / `footprint` is a column. `bar` is the candle's position
(0 = oldest) and links footprint rows to candles.
