from OrderFlow import OrderFlowChart
import pandas as pd

OHLC_CSV = 'orderflow_ohlc.csv'
ORDERFLOW_CSV = 'orderflow_candles.csv'

# Read orderflow data from CSV
orderflow_data = pd.read_csv(
ORDERFLOW_CSV,
names=['bid_size', 'price', 'ask_size', 'identifier'],
index_col=0,
parse_dates=True
)


# Read OHLC data from CSV
ohlc_data = pd.read_csv(
OHLC_CSV,
index_col=0,
parse_dates=True,
names=['open', 'high', 'low', 'close', 'identifier']
)

# Create an instance of OrderFlowChart
orderflowchart = OrderFlowChart(
orderflow_data,
ohlc_data,
identifier_col='identifier'
)

processed_data = orderflowchart.get_processed_data()

# Plot the orderflow chart
orderflowchart.plot()