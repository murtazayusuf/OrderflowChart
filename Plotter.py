import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from plotly.graph_objs import *
from plotly.subplots import make_subplots
import warnings
import time

warnings.filterwarnings('ignore')


def candle_proc(df):
    df = df.sort_values(by=['time', 'sequence', 'price'])
    df = df.reset_index()
    df_dp = df.iloc[1::2].copy()
    df = pd.concat([df, df_dp])
    df = df.sort_index()
    df = df.set_index('identifier')
    df = df.sort_values(by=['time', 'sequence'])
    # df = df.sort_index()
    df[2::3] = np.nan
    return df

# Apply robust scaling to size by grouping by time by subtracting median
# and dividing by IQR


def calc_imbalance(df):
    df['sum'] = df['bid_size'] + df['ask_size']
    df['time'] = df.index.astype(str)
    bids, asks = [], []
    for b, a in zip(df['bid_size'].astype(int).astype(str),
                    df['ask_size'].astype(int).astype(str)):
        dif = 4 - len(a)
        a = a + (' ' * dif)
        dif = 4 - len(b)
        b = (' ' * dif) + b
        bids.append(b)
        asks.append(a)

    df['text'] = pd.Series(bids, index=df.index) + '  ' + \
        pd.Series(asks, index=df.index)
    df.index = df['identifier']
    df['size'] = (df['bid_size'] - df['ask_size']) / \
        (df['bid_size'] + df['ask_size'])
    # df = df.drop(['bid_size', 'ask_size'], axis=1)
    return df


def annotate(df2):
    df2 = df2.drop(['size'], axis=1)
    df2['sum'] = df2['sum'] / df2.groupby(df2.index)['sum'].transform('max')
    df2['text'] = ''
    df2['time'] = df2['time'].astype(str)
    df2['text'] = ['█' * int(sum_ * 10) for sum_ in df2['sum']]
    df2['text'] = '                    ' + df2['text']
    df2['time'] = df2['time'].astype(str)
    return df2


def calc_params(of, ohlc):
    delta = of.groupby(of['identifier']).sum()['ask_size'] - \
        of.groupby(of['identifier']).sum()['bid_size']
    delta = delta[ohlc['identifier']]
    cum_delta = delta.rolling(10).sum()
    roc = cum_delta.diff()/cum_delta.shift(1) * 100
    roc = roc.fillna(0).round(2)
    volume = of.groupby(of['identifier']).sum()['ask_size'] + of.groupby(of['identifier']).sum()['bid_size']
    delta = pd.DataFrame(delta, columns=['value'])
    delta['type'] = 'delta'
    cum_delta = pd.DataFrame(cum_delta, columns=['value'])
    cum_delta['type'] = 'cum_delta'
    roc = pd.DataFrame(roc, columns=['value'])
    roc['type'] = 'roc'
    volume = pd.DataFrame(volume, columns=['value'])
    volume['type'] = 'volume'

    labels = pd.concat([delta, cum_delta, roc, volume])
    labels = labels.sort_index()
    labels['text'] = labels['value'].astype(str)

    labels['value'] = np.tanh(labels['value'])
    # raise Exception
    return labels


def calc_vwap(of, df):
    df = df.reset_index(drop=True)
    max_imb_price = df.loc[df.groupby('identifier')['size'].idxmax(
    ), ['price', 'identifier']].set_index('identifier').squeeze()
    min_imb_price = df.loc[df.groupby('identifier')['size'].idxmin(
    ), ['price', 'identifier']].set_index('identifier').squeeze()
    buy_vol = of.groupby('identifier')['bid_size'].sum().T
    sell_vol = of.groupby('identifier')['ask_size'].sum().T

    buy_vwap = max_imb_price.mul(buy_vol).rolling(
        10).sum() / buy_vol.rolling(10).sum()
    sell_vwap = min_imb_price.mul(sell_vol).rolling(
        10).sum() / sell_vol.rolling(10).sum()

    id_ = df.sort_values(['time', 'sequence'])['identifier'].unique()
    buy_vwap, sell_vwap = buy_vwap[id_].round(2), sell_vwap[id_].round(2)

    return buy_vwap, sell_vwap


def range_proc(ohlc, type_='hl'):
    if type_ == 'hl':
        seq = ohlc['low'].append(ohlc['high'])
    if type_ == 'oc':
        seq = ohlc['open'].append(ohlc['close'])
    id_seq = ohlc['identifier'].append(ohlc['identifier'])
    seq_hl = ohlc['sequence'].append(ohlc['sequence'])
    seq = pd.DataFrame(seq, columns=['price'])
    seq['identifier'] = id_seq
    seq['sequence'] = seq_hl
    seq['time'] = seq.index
    seq = seq.sort_index()
    seq = seq.set_index('identifier')
    return seq


def proc_data(ORDERFLOW_CSV, OHLC_CSV):
    # Read orderflow data
    of = pd.read_csv(
        ORDERFLOW_CSV,
        names=[
            'bid_size',
            'price',
            'ask_size',
            'identifier'],
        index_col=0,
        parse_dates=True)
    of['sequence'] = of['identifier'].str.len()
    of.index = of.index - pd.Timedelta(hours=5)
    # Read ohlc data
    ohlc = pd.read_csv(OHLC_CSV, index_col=0, parse_dates=True, names=[
                       'open', 'high', 'low', 'close', 'identifier'])
    ohlc['sequence'] = ohlc['identifier'].str.len()
    ohlc.index = ohlc.index - pd.Timedelta(hours=5)
    ohlc.loc[ohlc['open'] == 0, 'open'] = ohlc.loc[ohlc['open'] == 0, 'low']

    df = calc_imbalance(of.copy())

    df2 = annotate(df.copy())

    # Seperate green and red candles
    green_id = ohlc.loc[ohlc['close'] >= ohlc['open']]['identifier']
    red_id = ohlc.loc[ohlc['close'] < ohlc['open']]['identifier']

    high_low = range_proc(ohlc, type_='hl')
    green_hl = high_low.loc[green_id]
    green_hl = candle_proc(green_hl)

    red_hl = high_low.loc[red_id]
    red_hl = candle_proc(red_hl)

    open_close = range_proc(ohlc, type_='oc')

    green_oc = open_close.loc[green_id]
    green_oc = candle_proc(green_oc)

    red_oc = open_close.loc[red_id]
    red_oc = candle_proc(red_oc)

    labels = calc_params(of, ohlc)

    # buy_vwap, sell_vwap = calc_vwap(of, df)
    buy_vwap, sell_vwap = pd.Series(), pd.Series()

    return of, ohlc, df, df2, green_hl, red_hl, green_oc, red_oc, labels, buy_vwap, sell_vwap



def plot_ranges(ohlc):
    ymin = ohlc['high'][-1] + 1
    ymax = ymin - 12
    xmax = ohlc.shape[0]
    xmin = xmax - 9
    tickvals = [i for i in ohlc['identifier']]
    ticktext = [i for i in ohlc.index]
    return ymin, ymax, xmin, xmax, tickvals, ticktext


def plot_orderflow(ohlc, df, df2, green_hl, red_hl, green_oc,
                   red_oc, labels):
    ymin, ymax, xmin, xmax, tickvals, ticktext = plot_ranges(ohlc)
    print("Total candles: ", ohlc.shape[0])
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.0, row_heights=[9, 1])

    fig.add_trace(go.Scatter(x=df2['identifier'], y=df2['price'], text=df2['text'],
                             name='VolumeProfile', textposition='middle right',
                             textfont=dict(size=8, color='rgb(0, 0, 255, 0.0)'), hoverinfo='none',
                             mode='text', showlegend=True,
                             marker=dict(
                             sizemode='area',
                             sizeref=0.1,  # Adjust the size scaling factor as needed
                             )), row=1, col=1)

    # Add trace for orderflow data
    fig.add_trace(
        go.Heatmap(
            x=df['identifier'],
            y=df['price'],
            z=df['size'],
            text=df['text'],
            colorscale='icefire_r',
            showscale=False,
            showlegend=True,
            name='BidAsk',
            texttemplate="%{text}",
            textfont={
                "size": 11,
                "family": "Courier New"},
            hovertemplate="Price: %{y}<br>Size: %{text}<br>Imbalance: %{z}<extra></extra>",
            xgap=60),
        row=1,
        col=1)

    fig.add_trace(
        go.Scatter(
            x=green_hl.index,
            y=green_hl['price'],
            name='Candle',
            legendgroup='group',
            showlegend=True,
            line=dict(
                color='green',
                width=1.5)),
        row=1,
        col=1)

    fig.add_trace(
        go.Scatter(
            x=red_hl.index,
            y=red_hl['price'],
            name='Candle',
            legendgroup='group',
            showlegend=False,
            line=dict(
                color='red',
                width=1.5)),
        row=1,
        col=1)

    fig.add_trace(
        go.Scatter(
            x=green_oc.index,
            y=green_oc['price'],
            name='Candle',
            legendgroup='group',
            showlegend=False,
            line=dict(
                color='green',
                width=6)),
        row=1,
        col=1)

    fig.add_trace(
        go.Scatter(
            x=red_oc.index,
            y=red_oc['price'],
            name='Candle',
            legendgroup='group',
            showlegend=False,
            line=dict(
                color='red',
                width=6)),
        row=1,
        col=1)

    # fig.add_trace(go.Scatter(x=buy_vwap.index, y=buy_vwap, name='VWAP', legendgroup='group', showlegend=True, line=dict(color='green', width=1.5)), row=1, col=1)

    # fig.add_trace(go.Scatter(x=sell_vwap.index, y=sell_vwap, name='VWAP', legendgroup='group', showlegend=False, line=dict(color='red', width=1.5)), row=1, col=1)
    
    fig.add_trace(
        go.Heatmap(
            x=labels.index,
            y=labels['type'],
            z=labels['value'],
            colorscale='rdylgn',
            showscale=False,
            showlegend=True,
            name='Parameters',
            text=labels['text'],
            texttemplate="%{text}",
            textfont={
                "size": 10},
            hovertemplate="%{x}<br>%{text}<extra></extra>",
            xgap=4,
            ygap=4),
        row=2,
        col=1)

    fig.update_layout(title='Order Book Chart',
                      yaxis=dict(title='Price', showgrid=False, range=[
                                 ymax, ymin], tickformat='.2f'),
                      yaxis2=dict(fixedrange=True, showgrid=False),
                      xaxis2=dict(title='Time', showgrid=False),
                      xaxis=dict(showgrid=False, range=[xmin, xmax]),
                      height=780,
                      template='plotly_dark',
                      paper_bgcolor='#222', plot_bgcolor='#222',
                      dragmode='pan', margin=dict(l=10, r=0, t=40, b=20),)

    fig.update_xaxes(
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=0.25,
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext)
    fig.update_yaxes(
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikemode="across",
        spikethickness=0.25)
    fig.update_layout(spikedistance=1000, hoverdistance=100)

    config = {
        'modeBarButtonsToRemove': ['zoomIn', 'zoomOut', 'zoom', 'autoScale'],
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline',
                                'drawopenpath',
                                'drawclosedpath',
                                'drawcircle',
                                'drawrect',
                                'eraseshape'
                                ]
    }
    # Show figure
    fig.show(config=config)


if __name__ == '__main__':
    ORDERFLOW_CSV = './orderflow_candles.csv'
    OHLC_CSV = './orderflow_ohlc.csv'
    
    start_time = time.time()

    of, ohlc, df, df2, green_hl, red_hl, green_oc, red_oc, labels, buy_vwap, sell_vwap = proc_data(
        ORDERFLOW_CSV, OHLC_CSV)

    print('Time taken: {:.2f} sec'.format(time.time() - start_time))
    plot_orderflow(ohlc, df, df2, green_hl, red_hl, green_oc,
                   red_oc, labels)
