import os
import pickle
import pandas as pd
from algotrading.ml_logic.scraping_data import get_stock_data
from algotrading.ml_logic.preprocessing import window
from algotrading.ml_logic.model import TimeSeriesModel

def train_model(file_path, batch_size=15, epochs=1000):
    # Load the stock symbols
    symbols_df = pd.read_csv(file_path)

    # Iterate over all symbols in the CSV
    for symbol in symbols_df['Ticker']:
        print(f"Training model for {symbol}...")

        # Get the stock data
        stock_data = get_stock_data(symbol)

        # Preprocess the data
        X_train, y_train, X_val, y_val, X_test, y_test, sc = window(stock_data)

        # Initialize the model
        model = TimeSeriesModel((X_train.shape[1], X_train.shape[2]))

        # Compile the model
        model.compile()

        # Train the model
        model.train(X_train, y_train, X_val, y_val, batch_size, epochs)

        # Evaluate the model
        y_pred = model.evaluate(X_test, y_test)

        print(pd.DataFrame(data={'Model Prediction': y_pred, 'Actuals': y_test}))

        # Save the trained model for the ticker
        model.model.save(f'/home/adedapo/personal_project/daps05ayoade/disseration/trained-model/best_model_{symbol}.h5')

        # Save the scaler for the ticker
        with open(f'/home/adedapo/personal_project/daps05ayoade/disseration/fitted-scaler/scaler_{symbol}.pkl', 'wb') as file:
            pickle.dump(sc, file)

    # Return dictionary of trained models
    return model, y_pred

if __name__ == '__main__':
    data_dir = "data"
    file_name = "200_large_cap_stocks.csv"
    file_path = os.path.join(data_dir, file_name)
    train_model(file_path)
