import yfinance as yf

def calculate_macd(data, short_window, long_window):
    """
    Calculate MACD, signal line, and MACD histogram.
    """
    # Short-term EMA
    short_ema = data.ewm(span=short_window, adjust=False).mean()

    # Long-term EMA
    long_ema = data.ewm(span=long_window, adjust=False).mean()

    # MACD line
    macd_line = short_ema - long_ema

    # Signal line
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    return macd_line, signal_line

def calculate_signals(df):
    # Loop over all rows in the DataFrame
    for index, row in df.iterrows():
        ticker = row['Ticker']
        signal = row['PrimarySignal']

        # Skip this ticker if signal is 'Hold'
        if signal in ['Buy', 'Sell']:
            print(f'Getting Data for {ticker}...')  # Log the ticker

            data = yf.download(ticker, start='2020-01-01')

            # Calculate MACD
            macd_line, signal_line = calculate_macd(data['Close'], short_window=12, long_window=26)

            # Create a crossover signal based on the most recent MACD values
            if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
                df.loc[index, 'MACD_Signal'] = 'Buy'
            elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
                df.loc[index, 'MACD_Signal'] = 'Sell'
            else:
                df.loc[index, 'MACD_Signal'] = 'Hold'

    # Fill NaN values in 'MACD_Signal' column with 'Hold'
    df['MACD_Signal'].fillna('Hold', inplace=True)

    return df
