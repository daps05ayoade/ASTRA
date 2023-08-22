import time
import logging
import pandas as pd
import datetime
from pytz import timezone
import alpaca_trade_api as tradeapi

from algotrading.interface.main import get_trading_signals
from algotrading.interface.report import main
from algotrading.secrecy import API_KEY, SECRET_KEY, BASE_URL

trading_client = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

def place_order(symbol, qty, side, order_type, time_in_force, trail_percent=None):
    if trail_percent:
        return trading_client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            trail_percent=trail_percent,
            time_in_force=time_in_force
        )
    else:
        return trading_client.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=time_in_force
        )

def wait_for_order_fill(order):
    while True:
        order_status = trading_client.get_order(order.id).status
        if order_status == 'filled':
            break
        time.sleep(1)


def main_process():
    # Record the start time
    start_time = time.time()

    # Get trading signals
    # df, portfolio_value = get_trading_signals()

    # Record the end time
    end_time = time.time()

    # Print out how long it took to get trading signals
    print(f"It took {(end_time - start_time)/60} minutes to get trading signals.")

    # Check the time again
    ny_time = datetime.datetime.now(timezone('America/New_York'))

    # If timer doesn't work
    df = pd.read_csv("/home/adedapo/personal_project/daps05ayoade/disseration/algotrading/output/Trades/18-08-2023 Strategy.csv")
    portfolio_value = 984576.62

    # # # If it's not yet 9:40, wait
    # while ny_time.hour < 9 or (ny_time.hour == 9 and ny_time.minute < 40):
    #     time.sleep(1)  # Wait for 1 second before checking the time again
    #     ny_time = datetime.datetime.now(timezone('America/New_York'))

    portfolio_beta = df['BetaExposure'].sum()
    amount_spy = portfolio_beta * portfolio_value
    spy_price = trading_client.get_latest_trade('SPY').price
    buy_qty_spy = round(amount_spy / spy_price, 2)
    sell_qty_spy = int(round(amount_spy / spy_price))

    logging.info(f'Beta-Adjusted Exposure: {portfolio_beta:.2f}')

    for index, row in df.iterrows():
        symbol = row['Ticker']
        primary_signal = row['PrimarySignal']
        qty = int(row['Number of Shares to Buy/Sell'])

        if primary_signal == 'Buy':
            order = place_order(symbol, qty, 'buy', 'market', 'day')
            logging.info(f"Submitted BUY order for {qty} shares of {symbol}.")
            wait_for_order_fill(order)
            place_order(symbol, qty, 'sell', 'trailing_stop', 'gtc', trail_percent=1.0) # Define Trailing Stop
            logging.info(f"Submitted TRAILING STOP SELL order for {qty} shares of {symbol} with trail percent of 1.0%.")
        elif primary_signal == 'Sell':
            order = place_order(symbol, qty, 'sell', 'market', 'day')
            logging.info(f"Submitted SELL order for {qty} shares of {symbol}.")
            wait_for_order_fill(order)
            place_order(symbol, qty, 'buy', 'trailing_stop', 'gtc', trail_percent=1.0) # Define Trailing Stop
            logging.info(f"Submitted TRAILING STOP BUY order for {qty} shares of {symbol} with trail percent of 1.0%.")

    if portfolio_beta > 0:
        place_order('SPY', sell_qty_spy, 'sell', 'market', 'day')
        logging.info(f"Submitted SELL order for {sell_qty_spy} shares of SPY.")
    elif portfolio_beta < 0:
        place_order('SPY', buy_qty_spy, 'buy', 'market', 'day')
        logging.info(f"Submitted BUY order for {buy_qty_spy} shares of SPY.")

if __name__ == '__main__':
    main_process()
