import os
import logging
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from algotrading.secrecy import API_TOKEN, API_KEY, SECRET_KEY
from alpaca.trading.client import TradingClient
from algotrading.strategy.algorithm import Model
from algotrading.strategy.data_fetcher import DataFetcher
from algotrading.strategy.preprocessing import DataProcessing
from algotrading.strategy.scoring import Scoring

# Constants
DATA_DIR = "data"
FILE_NAME = "200_large_cap_stocks.csv"
OUTPUT_DIR = "/home/adedapo/personal_project/daps05ayoade/disseration/algotrading/output/Trades"
MODEL_PATH = '/home/adedapo/personal_project/daps05ayoade/disseration/trained-model/'
SCALER_PATH = '/home/adedapo/personal_project/daps05ayoade/disseration/fitted-scaler/'
WINDOW_SIZE = 60
NUM_THREADS = 5

logging.basicConfig(level=logging.INFO)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
current_date = datetime.now()
date_str = current_date.strftime("%d-%m-%Y")


def get_portfolio_size(data_processing):
    return data_processing.get_cash_balance(trading_client)



def get_trading_signals():
    logging.info("Initializing get_trading_signals function")

    data_fetcher = DataFetcher(API_TOKEN)
    data_processing = DataProcessing()
    scoring = Scoring()

    portfolio_size = float(get_portfolio_size(data_processing))
    print(f'Using portfolio size of: ${portfolio_size}')

    file_path = os.path.join(DATA_DIR, FILE_NAME)
    symbols_df = pd.read_csv(file_path)
    tickers = symbols_df['Ticker'].tolist()
    # tickers = ["AAPL", "MSFT", "AMZN", "GOOG"] # Used for Testing

    logging.info(f"Fetching data for {len(tickers)} tickers")

    # Fetch all predictions
    ticker_model_pairs = [
        (ticker, Model(f'{MODEL_PATH}best_model_{ticker}.h5', f'{SCALER_PATH}scaler_{ticker}.pkl'))
        for ticker in tickers
    ]

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Fetch all predictions
        fetch_predictions = lambda pair: data_processing.fetch_predictions(pair, data_fetcher, NUM_THREADS, WINDOW_SIZE)
        predictions_dfs = list(tqdm(executor.map(fetch_predictions, ticker_model_pairs), total=len(ticker_model_pairs), desc="Fetching predictions"))

    # Flatten the list of predictions DataFrames
    predictions_dfs = [df for sublist in predictions_dfs for df in sublist]

    # Fetch all metrics
    metrics_dfs = data_processing.fetch_metrics(tickers, data_fetcher)

    # Fetch all sentiment data
    sentiment_dfs = data_processing.fetch_sentiment(tickers, data_fetcher)

    # Merge and process dataframes
    logging.info("Merging and processing dataframes")
    all_metrics_df = pd.concat(metrics_dfs, ignore_index=True)
    predictions_df = pd.concat(predictions_dfs, ignore_index=True)
    all_sentiment_df = pd.concat(sentiment_dfs, ignore_index=True)

    predictions_df = data_processing.calculate_percent_change_and_qv_score(predictions_df, data_fetcher, all_metrics_df)
    merged_df = data_processing.merge_dataframes_on_ticker(predictions_df, all_sentiment_df)

    ss_w = 1/3
    qv_w = 1/3
    pc_w = 1/3
    weights = {'SentimentScore': ss_w, 'QuantitativeValueScore': qv_w, 'PercentChange': pc_w}
    final_df = scoring.rank_and_sort_dataframe(merged_df, weights)

    symbol_groups = list(data_processing.split_list(final_df['Ticker'], 6))

    logging.info("Calculating position sizes and updating dataframe")

    max_position_size = portfolio_size * 0.10
    num_signals = len(final_df[final_df['PrimarySignal'].isin(['Buy', 'Sell'])])
    position_size = portfolio_size / num_signals if num_signals > 0 else 0
    high_risk_threshold = 0  # Define your own risk threshold

    for symbol_group in tqdm(symbol_groups, desc="Updating dataframe"):
        for ticker in symbol_group:
            try:
                scoring.process_single_ticker(ticker, API_TOKEN, final_df, position_size, max_position_size, high_risk_threshold, portfolio_size)
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")

    logging.info("Cleaning and saving dataframe")

    final_df['BetaExposure'] = final_df['Beta'] * final_df['Weight']
    final_df.drop(columns=['PreviousDayPrice', 'NextDayPrice', 'SentimentScore', 'QuantitativeValueScore', 'PercentChange', 'CompositeScore', 'WeightedAverageSentimentScore', 'Beta', 'Weight'], inplace=True)
    final_df.drop(final_df[final_df['PrimarySignal'] == 'Hold'].index, inplace=True)
    final_df = final_df.reset_index(drop=True)
    final_df = final_df.dropna()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{date_str} Strategy.csv")
    final_df.to_csv(output_path, index=False)

    logging.info(f"Data saved to {output_path}")

    logging.info(final_df)

    return final_df, portfolio_size


if __name__ == '__main__':
    get_trading_signals()
