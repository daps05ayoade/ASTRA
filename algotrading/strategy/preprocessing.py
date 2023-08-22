import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class DataProcessing:
    def __init__(self, portfolio_size=0):
        self.portfolio_size = portfolio_size

    def portfolio_input(self):
        """
        Asks the user for their portfolio size and ensures it is a valid float value.
        """
        while True:
            self.portfolio_size = input("Enter the value of your Cash: $")
            try:
                val = float(self.portfolio_size)
                if val < 0:
                    print("Invalid input! Portfolio size can't be negative.")
                    continue
                break
            except ValueError:
                print("That's not a number! \n Try again:")
        return self.portfolio_size

    @staticmethod
    def get_cash_balance(api):
        """
        Gets the cash balance from the user's Alpaca account.
        """
        account = api.get_account()
        if account.status == 'ACTIVE':
            # Get cash balance
            portfolio_size = float(account.cash)
            logging.info(f"Received portfolio size of: {portfolio_size}")
        else:
            print('Account is not active')
        return portfolio_size

    @staticmethod
    def split_list(lst, n):
        """
        Yields successive n-sized chunks from a list.
        """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def calculate_percent_change_and_qv_score(predictions_df, data_fetcher, all_metrics_df):
        """
        Calculates percent change and quantitative value score for a given dataframe.
        """
        qv_scores = data_fetcher.compute_qv_score(all_metrics_df)
        predictions_df['PercentChange'] = ((predictions_df['NextDayPrice'] - predictions_df['PreviousDayPrice']) / predictions_df['PreviousDayPrice']) * 100
        predictions_df['QuantitativeValueScore'] = qv_scores
        predictions_df['Beta'] = all_metrics_df['Beta'].astype(float)

        return predictions_df

    @staticmethod
    def merge_dataframes_on_ticker(df1, df2):
        """
        Merges two dataframes on 'Ticker'/'ticker' column and drops the redundant 'ticker' column.
        """
        merged_df = pd.merge(df1, df2, left_on='Ticker', right_on='ticker', how='inner')
        merged_df.drop(columns=['ticker'], inplace=True)
        return merged_df

    @staticmethod
    def fetch_predictions(ticker_model_pair, data_fetcher, num_threads, window_size):
        """
        Fetches predictions for a single ticker in parallel.
        """
        ticker, model = ticker_model_pair
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            predictions_df = list(executor.map(lambda _: model.predict_prices(ticker, window_size, data_fetcher), [ticker]))
        return predictions_df

    @staticmethod
    def fetch_metrics(tickers, data_fetcher):
        """
        Fetches metrics for a list of tickers.
        """
        metrics_dfs = []
        for ticker in tqdm(tickers, desc="Fetching metrics"):
            try:
                metrics_df = data_fetcher.get_metrics(ticker)
                metrics_dfs.append(metrics_df)
            except Exception as e:
                logging.error(f"Error getting metrics for {ticker}: {e}")

        return metrics_dfs

    @staticmethod
    def fetch_sentiment(tickers, data_fetcher):
        """
        Fetches sentiment data for a list of tickers.
        """
        sentiment_dfs = []
        for ticker in tqdm(tickers, desc="Fetching sentiment data"):
            try:
                sentiment_df = data_fetcher.get_sentiment_data(ticker)
                sentiment_dfs.append(sentiment_df)
            except Exception as e:
                logging.error(f"Error getting sentiment data for {ticker}: {e}")

        return sentiment_dfs
