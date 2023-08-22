import pandas as pd
import requests
from algotrading.secrecy import API_TOKEN

def get_data_from_url(url: str):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the request was unsuccessful
    return response.json()

def get_technical_data(symbol: str, feature: str):
    url = f'https://www.alphavantage.co/query?function={feature}&symbol={symbol}&interval=daily&time_period=60&series_type=open&apikey={API_TOKEN}'
    data = get_data_from_url(url)
    return pd.DataFrame(data['Technical Analysis: ' + feature]).T

def get_stock_data(symbol: str):
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={API_TOKEN}'

    # Obtain and process stock data
    data = get_data_from_url(api_url)
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df.columns = ['open','high','low','close','adj_close','volume','dividend_amount','split_coeff']
    df.columns = df.columns.str.lower()
    df.drop(columns=['split_coeff'], inplace=True)
    df.dropna(inplace=True)

    # Obtain technical data
    features = ['SMA', 'RSI', 'ADX', 'CCI', 'ATR', 'EMA']
    for feature in features:
        df[feature] = get_technical_data(symbol, feature)

    # Get SPY data and add to DataFrame
    spy_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&outputsize=full&apikey={API_TOKEN}'
    spy_data = get_data_from_url(spy_url)
    df['SPY'] = pd.DataFrame.from_dict(spy_data['Time Series (Daily)']).T['5. adjusted close']

    # Get VIXY data and add to DataFrame
    vixy_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=VIXY&outputsize=full&apikey={API_TOKEN}'
    vixy_data = get_data_from_url(vixy_url)
    df['VIXY'] = pd.DataFrame.from_dict(vixy_data['Time Series (Daily)']).T['5. adjusted close']

    # Sort Index in ascending order
    df = df.sort_index(ascending=True)

    # Create target column
    df['target'] = df['adj_close'].shift(-1)

    # Drop NA values
    df.dropna(inplace=True)

    # Convert all columns to float
    df = df.astype(float)

    return df
