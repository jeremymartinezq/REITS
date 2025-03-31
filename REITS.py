from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

app = Flask(__name__)

# List of REITs with full names and tickers
REIT_SYMBOLS = {
    'FTSE NAREIT All Equity': '^FNAR',
    'Apartment': 'APRT',
    'Data Centers': 'DCREIT',
    'Healthcare': 'HCN',
    'Industrial': 'PLD',
    'Office': 'BXP',
    'Residential': 'EQR',
    'Retail': 'SPG',
    'Self Storage': 'PSA',
    'Specialty': 'SPG',
    'Timber': 'WY'
}


# Download historical data
def download_data(symbol, start='2023-01-01', end='2024-12-31', interval='1d'):
    try:
        data = yf.download(symbol, start=start, end=end, interval=interval)
        if data.empty:
            print(f"Failed to download data for {symbol}")
            return None
        data['Datetime'] = data.index.tz_localize(None)
        data.reset_index(drop=True, inplace=True)
        return data
    except Exception as e:
        print(f"Data download error: {e}")
        return None


# Prepare and fit models for each symbol
def process_symbol(symbol, data):
    data.rename(columns={'Datetime': 'ds', 'Close': 'y'}, inplace=True)
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(data[['ds', 'y']])

    last_date = data['ds'].max()
    target_date = datetime(2030, 12, 31)
    periods = (target_date - last_date).days

    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    forecast_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Projected_Close'})
    forecast_data['SMA_24'] = forecast_data['Projected_Close'].rolling(window=24).mean()

    data.set_index('ds', inplace=True)
    data['SMA_24'] = data['y'].rolling(window=24).mean()

    generate_signals(data)
    generate_signals(forecast_data, historical=False)

    calculate_two(data, column_name='y')
    forecast_data.set_index('Date', inplace=True)
    calculate_two(forecast_data, column_name='Projected_Close')

    return data, forecast_data


# Define buy/sell signals
def generate_signals(df, historical=True):
    df['Signal'] = 0
    if historical:
        df.loc[df['y'] > df['SMA_24'], 'Signal'] = 1
        df.loc[df['y'] < df['SMA_24'], 'Signal'] = -1
    else:
        df['SMA_24'] = df['Projected_Close'].rolling(window=24).mean()
        df.loc[df['Projected_Close'] > df['SMA_24'], 'Signal'] = 1
        df.loc[df['Projected_Close'] < df['SMA_24'], 'Signal'] = -1


# Calculate Trend Wave Oscillator (TWO)
def calculate_two(df, short_window=24, long_window=96, column_name='y'):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    df['TWO'] = df[column_name].rolling(window=short_window).mean() - df[column_name].rolling(window=long_window).mean()
    df['TWO_Buy_Marker'] = np.where(df['TWO'] > 0, df['TWO'], np.nan)
    df['TWO_Sell_Marker'] = np.where(df['TWO'] < 0, df['TWO'], np.nan)


# Generate summary chart
def generate_summary_chart(data_dict):
    plt.figure(figsize=(14, 10))
    growth_data = pd.DataFrame()

    for symbol, data in data_dict.items():
        growth = calculate_growth(data)
        growth['Symbol'] = symbol
        growth_data = pd.concat([growth_data, growth])

    plt.subplot(2, 1, 1)
    for symbol in data_dict.keys():
        subset = growth_data[growth_data['Symbol'] == symbol]
        plt.plot(subset.index, subset['YTD'], label=f'{symbol} YTD')
        plt.plot(subset.index, subset['YOY'], label=f'{symbol} YOY')

    plt.title('YTD and YOY Growth of Assets')
    plt.xlabel('Date')
    plt.ylabel('Growth (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    heatmap_data = pd.pivot_table(growth_data, values='YOY', index='Symbol', columns='Date', aggfunc=np.mean)
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.1f', linewidths=0.5)
    plt.title('Heatmap of YOY Growth')
    plt.xlabel('Date')
    plt.ylabel('Symbol')

    plt.tight_layout()
    plt.show()


# Calculate growth metrics
def calculate_growth(data):
    data = data.copy()
    data['Date'] = pd.to_datetime(data.index)
    data.set_index('Date', inplace=True)

    data['YTD'] = data['Close'].pct_change().resample('Y').sum() * 100
    data['YOY'] = data['Close'].pct_change(252).resample('Y').sum() * 100
    return data[['YTD', 'YOY']]


# Plot buy/sell signals
def plot_signals(ax, df, historical=True):
    color_map = {'buy': ('g', 'lime'), 'sell': ('r', 'darkred')}
    suffix = '(Historical)' if historical else '(Projected)'
    price_col = 'y' if historical else 'Projected_Close'
    ax.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1][price_col],
               marker='^', color=color_map['buy'][0 if historical else 1], label=f'Buy Signal {suffix}')
    ax.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1][price_col],
               marker='v', color=color_map['sell'][0 if historical else 1], label=f'Sell Signal {suffix}')


# Plot data
def plot_data(data, forecast_data, symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.5})

    ax1.plot(data.index, data['y'], label=f'Historical {symbol} Prices', color='blue')
    ax1.plot(forecast_data.index, forecast_data['Projected_Close'], label=f'Projected {symbol} Prices', color='red')
    ax1.plot(data.index, data['SMA_24'], label='24-Hour SMA (Historical)', color='green', linestyle='--')
    ax1.plot(forecast_data.index, forecast_data['SMA_24'], label='24-Hour SMA (Projected)', color='orange',
             linestyle='--')

    plot_signals(ax1, data, historical=True)
    plot_signals(ax1, forecast_data, historical=False)

    ax1.axvline(x=data.index.max(), color='red', linestyle='--', label='Projection Start')
    ax1.set_title(f'{symbol} Price Projection Until 2030 with Buy/Sell Signals (Daily Data)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{symbol} Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(data.index, data['TWO'], label=f'Trend Wave Oscillator (Historical)', color='blue')
    ax2.plot(forecast_data.index, forecast_data['TWO'], label=f'Trend Wave Oscillator (Projected)', color='red')

    ax2.scatter(data.index, data['TWO_Buy_Marker'], color='green', marker='^', label='Buy Marker (Historical)')
    ax2.scatter(data.index, data['TWO_Sell_Marker'], color='red', marker='v', label='Sell Marker (Historical)')
    ax2.scatter(forecast_data.index, forecast_data['TWO_Buy_Marker'], color='lime', marker='^',
                label='Buy Marker (Projected)')
    ax2.scatter(forecast_data.index, forecast_data['TWO_Sell_Marker'], color='darkred', marker='v',
                label='Sell Marker (Projected)')

    ax2.axhline(0, color='black', linestyle='--', label='Zero Line')
    ax2.set_title('Trend Wave Oscillator with Markers')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('TWO')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


@app.route('/')
def index():
    return render_template('index.html', reits=REIT_SYMBOLS)


@app.route('/data', methods=['POST'])
def data():
    symbol = request.form.get('symbol')
    if not symbol or symbol not in REIT_SYMBOLS:
        return "Symbol not found", 400

    ticker = REIT_SYMBOLS[symbol]
    data = download_data(ticker)
    if data is None:
        return "Failed to retrieve data", 500

    data, forecast_data = process_symbol(symbol, data)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=data.index, y=data['y'], mode='lines', name=f'{symbol} Historical Prices'), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['Projected_Close'], mode='lines',
                             name=f'{symbol} Projected Prices'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_24'], mode='lines', name='SMA 24-Hour Historical'), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=forecast_data.index, y=forecast_data['SMA_24'], mode='lines', name='SMA 24-Hour Projected'), row=1,
        col=1)

    fig.add_trace(go.Scatter(x=data.index, y=data['TWO'], mode='lines', name='TWO Historical'), row=2, col=1)
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['TWO'], mode='lines', name='TWO Projected'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['TWO_Buy_Marker'], mode='markers',
                             marker=dict(color='green', symbol='triangle-up'), name='TWO Buy Marker Historical'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['TWO_Sell_Marker'], mode='markers',
                             marker=dict(color='red', symbol='triangle-down'), name='TWO Sell Marker Historical'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['TWO_Buy_Marker'], mode='markers',
                             marker=dict(color='lime', symbol='triangle-up'), name='TWO Buy Marker Projected'), row=2,
                  col=1)
    fig.add_trace(go.Scatter(x=forecast_data.index, y=forecast_data['TWO_Sell_Marker'], mode='markers',
                             marker=dict(color='darkred', symbol='triangle-down'), name='TWO Sell Marker Projected'),
                  row=2, col=1)

    fig.update_layout(title_text=f'{symbol} Financial Analysis', xaxis_title='Date', yaxis_title='Price',
                      yaxis2_title='TWO', template='plotly_dark')

    return fig.to_html(full_html=False)


@app.route('/summary')
def summary():
    data_dict = {}
    for name, ticker in REIT_SYMBOLS.items():
        data = download_data(ticker)
        if data is not None:
            data_dict[name] = process_symbol(name, data)[0]

    generate_summary_chart(data_dict)
    return "Summary chart generated!"


@app.route('/theme')
def theme():
    return render_template('theme.html')


if __name__ == '__main__':
    app.run(debug=True)
