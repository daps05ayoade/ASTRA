import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from algotrading.strategy.data_fetcher import DataFetcher

class Model:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        with open(scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

    def predict_prices(self, ticker: str, window_size: int, data_fetcher: DataFetcher) -> pd.DataFrame:
        """
        Predict the next day's closing price for a specific ticker using a trained model and scaler.
        """
        stock_data = data_fetcher.get_stock_data(ticker)
        if stock_data.empty:
            print(f"Warning: No data fetched for ticker: {ticker}")
            return None

        scaled_stock_data = self.scaler.transform(stock_data)
        scaled_windows = data_fetcher.window_data(scaled_stock_data, window_size)


        prediction = self.model.predict(scaled_windows)
        prev_day_price = prediction[-2][0]
        next_day_price = prediction[-1][0]

        df_prediction = pd.DataFrame({'Ticker': [ticker],
                                    'PreviousDayPrice': [prev_day_price],
                                    'NextDayPrice': [next_day_price]})
        return df_prediction
