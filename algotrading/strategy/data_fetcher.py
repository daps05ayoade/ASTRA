import requests
import numpy as np
import pandas as pd
from scipy import stats
from statistics import mean
from ratelimiter import RateLimiter
import requests
from algotrading.secrecy import API_TOKEN

rate_limiter = RateLimiter(max_calls=100, period=61) # Create a rate limiter

class DataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.last_request_time = None

    def get_data_from_url(self, url: str):
        """
        Send a GET request to a URL and return the JSON response.
        """
        with rate_limiter: # Use the rate limiter
            response = requests.get(url)
            response.raise_for_status()
            return response.json()

    def get_technical_data(self, symbol: str, feature: str):
        """
        Fetch technical data for a specific stock symbol and feature from Alpha Vantage.
        """
        url = f'https://www.alphavantage.co/query?function={feature}&symbol={symbol}&interval=daily&time_period=60&series_type=open&apikey={self.api_key}'
        data = self.get_data_from_url(url)
        return pd.DataFrame(data['Technical Analysis: ' + feature]).T if 'Technical Analysis: ' + feature in data else None


    def get_stock_data(self, symbol: str):
        """
        Fetch daily adjusted time series data for a specific stock symbol from Alpha Vantage.
        """
        api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={self.api_key}'
        data = self.get_data_from_url(api_url)
        if 'Time Series (Daily)' not in data:
            raise ValueError("Time Series (Daily) data not found in response.")

        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend_amount', 'split_coeff']
        df.columns = df.columns.str.lower()
        df.drop(columns=['split_coeff'], inplace=True)
        df.dropna(inplace=True)

        features = ['SMA', 'RSI', 'ADX', 'CCI', 'ATR', 'EMA']
        for feature in features:
                feature_data = self.get_technical_data(symbol, feature)
                if feature_data is not None:
                    df[feature] = feature_data if not df.empty else feature_data

        spy_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=SPY&outputsize=full&apikey={self.api_key}'
        spy_data = self.get_data_from_url(spy_url)
        if 'Time Series (Daily)' in spy_data:
            spy_df = pd.DataFrame(spy_data['Time Series (Daily)']).T
            spy_df = spy_df['5. adjusted close'].rename('SPY')
            df = df.merge(spy_df, left_index=True, right_index=True)

        vixy_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=VIXY&outputsize=full&apikey={API_TOKEN}'
        vixy_data = self.get_data_from_url(vixy_url)
        df['VIXY'] = pd.DataFrame.from_dict(vixy_data['Time Series (Daily)']).T['5. adjusted close']

        df = df.sort_index(ascending=True)
        df.dropna(inplace=True)
        df = df.astype(float)

        return df

    @staticmethod
    def window_data(data, window_size):
        """
        Convert the input data into a series of windows of a specified size.
        """
        windows = []
        for i in range(len(data) - window_size):
            windows.append(data[i:i + window_size])
        return np.array(windows)


    def get_metrics(self, ticker):
        """
        Get the key metrics for a specific ticker from Alpha Vantage.
        """
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}'
        data = self.get_data_from_url(url)

        df_metrics = pd.DataFrame(index=[0])
        df_metrics['PERatio'] = data['PERatio']
        df_metrics['PriceToBookRatio'] = data['PriceToBookRatio']
        df_metrics['PriceToSalesRatioTTM'] = data['PriceToSalesRatioTTM']
        df_metrics['EVToEBITDA'] = data['EVToEBITDA']
        df_metrics['EVToRevenue'] = data['EVToRevenue']
        df_metrics['Beta'] = data['Beta']

        return df_metrics

    @staticmethod
    def compute_qv_score(df_metrics):
        """
        Compute the Quality-Value (QV) score for the metrics in the input DataFrame.
        """
        # Define the metrics dictionary
        metrics = {
            'PERatio': 'PEPercentile',
            'PriceToBookRatio': 'PBPercentile',
            'PriceToSalesRatioTTM': 'PSPercentile',
            'EVToEBITDA': 'EV/EBITDAPercentile',
            'EVToRevenue': 'EV/RevenuePercentile'
        }

        # Calculating the percentiles for each metric
        for row in df_metrics.index:
            for metric in metrics.keys():
                df_metrics.loc[row, metrics[metric]] = stats.percentileofscore(df_metrics[metric], df_metrics.loc[row, metric])/100

        for row in df_metrics.index:
            value_percentiles = []
            for metric in metrics.keys():
                value_percentiles.append(df_metrics.loc[row, metrics[metric]])
            df_metrics.loc[row, 'QuantitativeValueScore'] = mean(value_percentiles)

        return df_metrics['QuantitativeValueScore']

    def get_sentiment_data(self, ticker):
        """
        Fetches and processes sentiment data for a given ticker, returning a dataframe with averaged sentiment scores.
        """
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={self.api_key}'
        data = self.get_data_from_url(url)
        ticker_data_list = []

        for feed_item in data['feed']:
            for ticker_data in feed_item["ticker_sentiment"]:
                if ticker_data['ticker'] == ticker:
                    ticker_item = {
                        'ticker': ticker,
                        'relevance_score': float(ticker_data['relevance_score']),
                        'ticker_sentiment_score': float(ticker_data['ticker_sentiment_score']),
                        'ticker_sentiment_label': ticker_data['ticker_sentiment_label'],
                    }
                    ticker_data_list.append(ticker_item)

        sentiment_df = pd.DataFrame(ticker_data_list)

        sentiment_mapping = {
            "Very-Bullish": 5,
            "Bullish": 4,
            "Somewhat-Bullish": 3,
            "Neutral": 2,
            "Somewhat-Bearish": 1,
            "Bearish": 0,
            "Very-Bearish": -1
        }

        sentiment_df['SentimentScore'] = sentiment_df['ticker_sentiment_label'].map(sentiment_mapping)
        sentiment_df['SentimentScore'] = sentiment_df['SentimentScore'].astype(int)
        sentiment_df['WeightedAverageSentimentScore'] = (sentiment_df['ticker_sentiment_score'] * sentiment_df['relevance_score']).sum() / sentiment_df['relevance_score'].sum()

        sentiment_df.drop(columns=['relevance_score', 'ticker_sentiment_score', 'ticker_sentiment_label'], inplace=True)

        sentiment_df = sentiment_df.groupby('ticker').mean().reset_index()

        sentiment_df['SentimentScore'] = sentiment_df['SentimentScore']

        return sentiment_df
