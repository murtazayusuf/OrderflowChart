<h1>OrderflowChart</h1><p>Welcome to the <b>OrderflowChart</b> project! This project empowers you to visualize orderflow footprint charts effortlessly using Python and Plotly.</p><h2>Usage</h2><p>The heart of the project is the <code>OrderFlowChart</code> class constructor. It's designed to simplify the process of plotting orderflow data on footprint charts, leveraging Plotly's capabilities.</p><h3>Constructor</h3><pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">from</span> OrderFlow <span class="hljs-keyword">import</span> OrderFlowChart

<span class="hljs-comment"># Read orderflow data from CSV</span>
orderflow_data = pd.read_csv(
    ORDERFLOW_CSV,
    names=[<span class="hljs-string">'bid_size'</span>, <span class="hljs-string">'price'</span>, <span class="hljs-string">'ask_size'</span>, <span class="hljs-string">'identifier'</span>],
    index_col=<span class="hljs-number">0</span>,
    parse_dates=<span class="hljs-literal">True</span>
)

<span class="hljs-comment"># Read OHLC data from CSV</span>
ohlc_data = pd.read_csv(
    OHLC_CSV,
    index_col=<span class="hljs-number">0</span>,
    parse_dates=<span class="hljs-literal">True</span>,
    names=[<span class="hljs-string">'open'</span>, <span class="hljs-string">'high'</span>, <span class="hljs-string">'low'</span>, <span class="hljs-string">'close'</span>, <span class="hljs-string">'identifier'</span>]
)

<span class="hljs-comment"># Create an instance of OrderFlowChart</span>
orderflowchart = OrderFlowChart(
    orderflow_data,
    ohlc_data,
    identifier_col=<span class="hljs-string">'identifier'</span>
)

<span class="hljs-comment"># Plot the orderflow chart</span>
orderflowchart.plot()
</code></div></div></pre>
<h3>Parameters</h3>
<ul>
<li><p>
<code>orderflow_data</code>: Your orderflow data, containing columns like 'bid_size', 'price', 'ask_size', and 'identifier'. If the 'imbalance' column needs to be calculated, simply provide it along with the previous mentioned columns</p></li>
<li><p>
<code>ohlc_data</code>: Your OHLC data with columns 'open', 'high', 'low', 'close', and 'identifier'. The 'identifier' column bridges the gap between orderflow and OHLC data.</p></li>
<li><p>
<code>identifier_col</code>: The column that uniquely identifies candles in both datasets. Incase your data is time-indexed i.e. each candle has a unique timestamp that acts as index, pass <i>None</i>.</p></li>
<li><p>
<code>imbalance_col</code>: The column name that contains imbalance for each price level. Provide None if to be calculated.</p></li>
</ul><h3>Output</h3><p>The above code snippet generates a stunning orderflow chart like this:</p><p><img src="image.png" alt="OrderFlowChart Example"></p><p>With OrderflowChart, you can effortlessly transform complex orderflow data into visually appealing and insightful footprint charts. Feel free to explore, customize, and gain new perspectives from your data with this powerful tool.</p>
<h2>Alternative Usage with Preprocessed Data</h2>
<p>If you have your data preprocessed and stored in a JSON format, you can use the <code>OrderFlowChart.from_preprocessed_data</code> class method to simplify the process further. This method allows you to directly load and plot your orderflow chart without manually reading and parsing CSV files.</p>
<h3>Using Preprocessed Data</h3>
<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python"><span class="hljs-keyword">from</span> OrderFlow <span class="hljs-keyword">import</span> OrderFlowChart
<span class="hljs-keyword">import</span> pandas <span class="hljs-keyword">as</span> pd
<span class="hljs-keyword">import</span> json
<span class="hljs-comment"># Load preprocessed data from JSON</span>
with open(<span class="hljs-string">'data/preprocessed_data.json'</span>, <span class="hljs-string">'r'</span>) <span class="hljs-keyword">as</span> f:
    preprocessed_data = json.load(f)

<span class="hljs-comment"># Create an OrderFlowChart instance using preprocessed data</span>
orderflowchart = OrderFlowChart.from_preprocessed_data(preprocessed_data)

<span class="hljs-comment"># Plot the orderflow chart</span>
orderflowchart.plot()
</code></div></div></pre>

<p>This approach is particularly useful when dealing with datasets that have been previously cleaned, aggregated, or transformed, allowing for a streamlined visualization process. Ensure your preprocessed data adheres to the expected format as described in the provided Pydantic model documentation. For detailed information on the data structure and the Pydantic model used for preprocessing, please refer to the <a href='data/README.md'>Data Model Documentation<a>.</p>
