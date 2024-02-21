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
import string
import random

warnings.filterwarnings('ignore')

class OrderFlowChart():
    def __init__(self, orderflow_data, ohlc_data, identifier_col=None, imbalance_col=None, **kwargs):
        """
        The constructor for OrderFlowChart class.
        It takes in the orderflow data and the ohlc data and creates a unique identifier for each candle if not provided.
        It also calculates the imbalance if not provided.
        The data should have datetime index and should have the following columns:
        orderflow_data: ['bid_size', 'price', 'ask_size', 'identifier']
        ohlc_data: ['open', 'high', 'low', 'close', 'identifier']

        The identifier column is used to map the orderflow data to the ohlc data.
        """

        if 'data' in kwargs:
            try:
                self.use_processed_data(kwargs['data'])
            except:
                raise Exception("Invalid data structure found. Please provide a valid processed data dictionary. Refer to documentation for more information.")
        else:
            self.orderflow_data = orderflow_data
            self.ohlc_data = ohlc_data
            self.identifier_col = identifier_col
            self.imbalance_col = imbalance_col
            self.is_processed = False
            self.granularity = abs(self.orderflow_data.iloc[0]['price'] - self.orderflow_data.iloc[1]['price'])

    def generate_random_string(self, length):
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for _ in range(length))

    def create_identifier(self):
        """
        This method will generate a unique gibberish string for each candle based on the timestamp and the price.
        """
        identifier = [self.generate_random_string(5) for i in range(self.ohlc_data.shape[0])]
        self.ohlc_data['identifier'] = identifier
        self.orderflow_data.loc[:, 'identifier'] = self.ohlc_data['identifier']
        
    def create_sequence(self):    
        self.ohlc_data['sequence'] = self.ohlc_data[self.identifier_col].str.len()
        self.orderflow_data['sequence'] = self.orderflow_data[self.identifier_col].str.len()

    def calc_imbalance(self, df):
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
        
        if self.imbalance_col is None:
            print("Calculating imbalance, as no imbalance column was provided.")
            df['size'] = (df['bid_size'] - df['ask_size'].shift().bfill()) / \
                (df['bid_size'] + df['ask_size'].shift().bfill())
            df['size'] = df['size'].ffill().bfill()
        else:
            print("Using imbalance column: {}".format(self.imbalance_col))
            df['size'] = df[self.imbalance_col]
            df = df.drop([self.imbalance_col], axis=1)
        # df = df.drop(['bid_size', 'ask_size'], axis=1)
        return df

    def annotate(self, df2):
        df2 = df2.drop(['size'], axis=1)
        df2['sum'] = df2['sum'] / df2.groupby(df2.index)['sum'].transform('max')
        df2['text'] = ''
        df2['time'] = df2['time'].astype(str)
        df2['text'] = ['█' * int(sum_ * 10) for sum_ in df2['sum']]
        df2['text'] = '                    ' + df2['text']
        df2['time'] = df2['time'].astype(str)
        return df2

    def range_proc(self, ohlc, type_='hl'):
        if type_ == 'hl':
            seq = pd.concat([ohlc['low'], ohlc['high']])
        if type_ == 'oc':
            seq = pd.concat([ohlc['open'], ohlc['close']])
        id_seq = pd.concat([ohlc['identifier'], ohlc['identifier']])
        seq_hl = pd.concat([ohlc['sequence'], ohlc['sequence']])
        seq = pd.DataFrame(seq, columns=['price'])
        seq['identifier'] = id_seq
        seq['sequence'] = seq_hl
        seq['time'] = seq.index
        seq = seq.sort_index()
        seq = seq.set_index('identifier')
        return seq

    def candle_proc(self, df):
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

    def calc_params(self, of, ohlc):
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

    def plot_ranges(self, ohlc):
        ymin = ohlc['high'][-1] + 1
        ymax = ymin - int(48*self.granularity)
        xmax = ohlc.shape[0]
        xmin = xmax - 9
        tickvals = [i for i in ohlc['identifier']]
        ticktext = [i for i in ohlc.index]
        return ymin, ymax, xmin, xmax, tickvals, ticktext

    def process_data(self):
        if self.identifier_col is None:
            self.identifier_col = 'identifier'
            self.create_identifier()

        self.create_sequence()
    
        self.df = self.calc_imbalance(self.orderflow_data)

        self.df2 = self.annotate(self.df.copy())

        self.green_id = self.ohlc_data.loc[self.ohlc_data['close'] >= self.ohlc_data['open']]['identifier']
        self.red_id = self.ohlc_data.loc[self.ohlc_data['close'] < self.ohlc_data['open']]['identifier']

        self.high_low = self.range_proc(self.ohlc_data, type_='hl')
        self.green_hl = self.high_low.loc[self.green_id]
        self.green_hl = self.candle_proc(self.green_hl)

        self.red_hl = self.high_low.loc[self.red_id]
        self.red_hl = self.candle_proc(self.red_hl)

        self.open_close = self.range_proc(self.ohlc_data, type_='oc')

        self.green_oc = self.open_close.loc[self.green_id]
        self.green_oc = self.candle_proc(self.green_oc)

        self.red_oc = self.open_close.loc[self.red_id]
        self.red_oc = self.candle_proc(self.red_oc)

        self.labels = self.calc_params(self.orderflow_data, self.ohlc_data)

        self.is_processed = True

    def get_processed_data(self):
        if not self.is_processed:
            try:
                self.process_data()
            except:
                raise Exception("Data processing failed. Please check the data types and the structure of the data. Refer to documentation for more information.")

        datas = [self.df, self.labels, self.green_hl, self.red_hl, self.green_oc, self.red_oc, self.df2, self.ohlc_data]
        datas2 = []
        # Convert all timestamps to utc float
        temp = ''
        for data in datas:
            temp = data.copy()
            temp.index.name = 'index'
            try:
                temp = temp.reset_index()
            except:
                pass
            dtype_dict = {i:str(j) for i, j in temp.dtypes.items()}
            temp = temp.astype('str')
            temp = temp.fillna('nan')
            temp = temp.to_dict(orient='list')
            temp['dtypes'] = dtype_dict
            datas2.append(temp)

        

        out_dict = {
            'orderflow': datas2[0],
            'labels': datas2[1],
            'green_hl': datas2[2],
            'red_hl': datas2[3],
            'green_oc': datas2[4],
            'red_oc': datas2[5],
            'orderflow2': datas2[6],
            'ohlc': datas2[7]
        }

        return out_dict
    
    @classmethod
    def from_preprocessed_data(cls, data):
        self = cls(None, None, data=data)
        return self

    def use_processed_data(self, data):
        # pop the dtypes
        dtypes = data['orderflow'].pop('dtypes')
        self.df = pd.DataFrame(data['orderflow']).replace('nan', np.nan)
        self.df = self.df.astype(dtypes)
        try:
            self.df = self.df.set_index('index')
        except:
            pass

        dtypes = data['labels'].pop('dtypes')
        self.labels = pd.DataFrame(data['labels']).replace('nan', np.nan)
        self.labels = self.labels.astype(dtypes)
        try:
            self.labels = self.labels.set_index('index')
        except:
            pass
        
        dtypes = data['green_hl'].pop('dtypes')
        self.green_hl = pd.DataFrame(data['green_hl']).replace('nan', np.nan)
        self.green_hl = self.green_hl.astype(dtypes)
        try:
            self.green_hl = self.green_hl.set_index('index')
        except: 
            pass        
        
        dtypes = data['red_hl'].pop('dtypes')
        self.red_hl = pd.DataFrame(data['red_hl']).replace('nan', np.nan)
        self.red_hl = self.red_hl.astype(dtypes)
        try:
            self.red_hl = self.red_hl.set_index('index')
        except:
            pass
        
        dtypes = data['green_oc'].pop('dtypes')
        self.green_oc = pd.DataFrame(data['green_oc']).replace('nan', np.nan)
        self.green_oc = self.green_oc.astype(dtypes)
        try:
            self.green_oc = self.green_oc.set_index('index')
        except:
            pass
        
        dtypes = data['red_oc'].pop('dtypes')
        self.red_oc = pd.DataFrame(data['red_oc']).replace('nan', np.nan)
        self.red_oc = self.red_oc.astype(dtypes)
        try:
            self.red_oc = self.red_oc.set_index('index')
        except:
            pass
        
        dtypes = data['orderflow2'].pop('dtypes')
        self.df2 = pd.DataFrame(data['orderflow2']).replace('nan', np.nan)
        self.df2 = self.df2.astype(dtypes)
        try:
            self.df2 = self.df2.set_index('index')
        except:
            pass
        
        dtypes = data['ohlc'].pop('dtypes')
        self.ohlc_data = pd.DataFrame(data['ohlc']).replace('nan', np.nan)
        self.ohlc_data = self.ohlc_data.astype(dtypes)
        try:
            self.ohlc_data = self.ohlc_data.set_index('index')
        except:
            pass
        self.granularity = abs(self.df.iloc[0]['price'] - self.df.iloc[1]['price'])
        self.is_processed = True

    def plot(self, return_figure=False):
        if not self.is_processed:
            try:
                self.process_data()
            except:
                raise Exception("Data processing failed. Please check the data types and the structure of the data. Refer to documentation for more information.")
        
        ymin, ymax, xmin, xmax, tickvals, ticktext = self.plot_ranges(self.ohlc_data)
        print("Total candles: ", self.ohlc_data.shape[0])
        # Create figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.0, row_heights=[9, 1])

        fig.add_trace(go.Scatter(x=self.df2['identifier'], y=self.df2['price'], text=self.df2['text'],
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
                x=self.df['identifier'],
                y=self.df['price'],
                z=self.df['size'],
                text=self.df['text'],
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
                x=self.green_hl.index,
                y=self.green_hl['price'],
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
                x=self.red_hl.index,
                y=self.red_hl['price'],
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
                x=self.green_oc.index,
                y=self.green_oc['price'],
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
                x=self.red_oc.index,
                y=self.red_oc['price'],
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
                x=self.labels.index,
                y=self.labels['type'],
                z=self.labels['value'],
                colorscale='rdylgn',
                showscale=False,
                showlegend=True,
                name='Parameters',
                text=self.labels['text'],
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

        if return_figure:
            return fig

        # Show figure
        fig.show(config=config)
