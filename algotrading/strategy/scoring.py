import math
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Scoring:
    @staticmethod
    def rank_and_sort_dataframe(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
        """
        Assigns a composite score to each row in the dataframe and assigns a signal based on this score.
        """
        scaler = MinMaxScaler()
        df[['SentimentScore', 'QuantitativeValueScore', 'PercentChange']] = scaler.fit_transform(df[['SentimentScore', 'QuantitativeValueScore', 'PercentChange']])

        df['CompositeScore'] = (weights['SentimentScore'] * df['SentimentScore']
                                + weights['PercentChange'] * df['PercentChange']
                                + weights['QuantitativeValueScore'] / (df['QuantitativeValueScore'] + 0.0001))

        # Compute thresholds based on percentiles of 'CompositeScore'
        thresholds = [df['CompositeScore'].quantile(0.2), df['CompositeScore'].quantile(0.8)]

        df['PrimarySignal'] = pd.cut(df['CompositeScore'], bins=[-np.inf, thresholds[0], thresholds[1], np.inf], labels=['Sell', 'Hold', 'Buy'])

        df = df.sort_values('CompositeScore', ascending=False)

        return df

    @staticmethod
    def process_single_ticker(ticker, api_token, final_df, position_size, max_position_size, high_risk_threshold, portfolio_size):
        """
        Fetches current price for a ticker and updates the dataframe accordingly.
        """
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_token}'
        r = requests.get(url)
        data = r.json()
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            current_price = float(data['Global Quote']['05. price'])
            final_df.loc[final_df['Ticker'] == ticker, 'Price'] = current_price

            signal = final_df.loc[final_df['Ticker'] == ticker, 'PrimarySignal'].values[0]
            shares_to_trade = math.floor(position_size / current_price)
            if shares_to_trade * current_price > max_position_size:
                shares_to_trade = math.floor(max_position_size / current_price)
            weighted_avg_sentiment_score = final_df.loc[final_df['Ticker'] == ticker, 'WeightedAverageSentimentScore'].values[0]
            if weighted_avg_sentiment_score < high_risk_threshold:
                shares_to_trade = math.floor(shares_to_trade * 0.5)  # Reduce the position size by 50%

            if signal == 'Buy':
                final_df.loc[final_df['Ticker'] == ticker, 'Number of Shares to Buy/Sell'] = shares_to_trade
            elif signal == 'Sell':
                final_df.loc[final_df['Ticker'] == ticker, 'Number of Shares to Buy/Sell'] = shares_to_trade
            else:
                final_df.loc[final_df['Ticker'] == ticker, 'Number of Shares to Buy/Sell'] = 0

            # Calculate weight of each ticker in the portfolio
            weight = (shares_to_trade * current_price) / portfolio_size if portfolio_size != 0 else 0
            # Make the weight negative if the signal is 'Sell'
            weight = -weight if signal == 'Sell' else weight
            final_df.loc[final_df['Ticker'] == ticker, 'Weight'] = weight
        else:
            raise Exception(f"Failed to fetch price for {ticker}")
