import os
import time
import logging
import pandas as pd
from datetime import datetime
import alpaca_trade_api as tradeapi
from algotrading.secrecy import API_KEY, SECRET_KEY, BASE_URL

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
OUTPUT_DIR = "/home/adedapo/personal_project/daps05ayoade/disseration/algotrading/output/Detailed Report"
SUM_OUTPUT_DIR = "/home/adedapo/personal_project/daps05ayoade/disseration/algotrading/output/Summary Report"
PROFIT_SUMMARY_PATH = "/home/adedapo/personal_project/daps05ayoade/disseration/algotrading/output/profit_summary.csv"
MAX_ORDER_HISTORY = 500
current_date = datetime.now()
date_str = current_date.strftime("%d-%m-%Y")

# Initialize trading client
trading_client = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)

def cancel_open_orders():
    """
    Cancels all open orders with the trading client.
    """
    open_orders = trading_client.list_orders(status='open')
    for order in open_orders:
        trading_client.cancel_order(order.id)
    logging.info(f"Canceled {len(open_orders)} open orders.")

def close_active_positions():
    """
    Closes all active positions with the trading client.
    """
    open_positions = trading_client.list_positions()
    closed_orders = []

    for position in open_positions:
        closed_order = trading_client.close_position(position.asset_id)
        if closed_order:
            closed_orders.append(closed_order)
        else:
            logging.info("No positions were closed.")

    logging.info(f"Sent orders to close {len(closed_orders)} positions.")
    return closed_orders

def wait_for_orders_to_fill(closed_orders):
    """
    Waits for all closed orders to be filled by the trading client.
    """
    for order in closed_orders:
        while True:
            try:
                order_status = trading_client.get_order(order.id).status
                if order_status == 'filled':
                    break
            except tradeapi.rest.APIError as e:
                if 'order not found' in str(e):
                    logging.warning(f"Order {order.id} not found.")
                    break
            time.sleep(1)  # Sleep for a while to wait for the order to be filled

def save_report(orders, output_path):
    """
    Saves report of all orders to a CSV file.
    """
    df = pd.DataFrame([{
        "Symbol": order.symbol,
        "Quantity": order.qty,
        "Filled Qty": float(order.filled_qty) if order.filled_qty else None,  # Ensure this is a float
        "Type": order.type,
        "Side": order.side,
        "Stop Price": order.stop_price,
        "Filled Avg Price": float(order.filled_avg_price) if order.filled_avg_price else None,  # Ensure this is a float
        "Status": order.status,
        "Submitted At": order.submitted_at,
        "Filled At": order.filled_at
    } for order in orders])

    df.to_csv(output_path, index=False)
    logging.info(f"Report saved to {output_path}")
    return df



def analyze_and_save_summary(df, summary_output_path):
    """
    Analyzes order data and saves summary to a CSV file.
    """
    df['Filled At'] = pd.to_datetime(df['Filled At'])
    df = df.sort_values('Filled At')

    # Initialize a new column for profit
    df['Profit'] = 0

    # For each symbol
    for symbol in df['Symbol'].unique():
        # Get the transactions for the symbol, excluding canceled transactions
        symbol_df = df.loc[(df['Symbol'] == symbol) & (df['Status'] != 'canceled')].copy()

        # Make sure the transactions are sorted by time
        symbol_df = symbol_df.sort_values('Filled At')

        # For each transaction of the symbol
        for i in range(1, len(symbol_df)):
            # If the transaction is a buy (assume this closes a short position)
            if symbol_df.iloc[i]['Side'] == 'buy':
                # Profit = Quantity * (Selling price - Buying price) for short positions
                df.loc[symbol_df.index[i], 'Profit'] = symbol_df.iloc[i]['Filled Qty'] * (symbol_df.iloc[i-1]['Filled Avg Price'] - symbol_df.iloc[i]['Filled Avg Price'])

            # If the transaction is a sell (assume this closes a long position)
            elif symbol_df.iloc[i]['Side'] == 'sell':
                # Profit = Quantity * (Selling price - Buying price) for long positions
                df.loc[symbol_df.index[i], 'Profit'] = symbol_df.iloc[i]['Filled Qty'] * (symbol_df.iloc[i]['Filled Avg Price'] - symbol_df.iloc[i-1]['Filled Avg Price'])

    summary = df.groupby('Symbol').agg({
        'Filled At': ['first', 'last'],
        'Filled Qty': 'sum',
        'Profit': 'sum'
    })

    summary.columns = ['First Action', 'Last Action', 'Total Quantity', 'Profit']
    summary.to_csv(summary_output_path)
    total_profit = summary['Profit'].sum()

    logging.info(summary)
    logging.info(f"Total Profit: ${total_profit:.2f}")
    logging.info(f"Summary report saved to {summary_output_path}")

    # Append or update total profit with current date in the CSV file.
    if os.path.isfile(PROFIT_SUMMARY_PATH):
        profit_summary_df = pd.read_csv(PROFIT_SUMMARY_PATH)
    else:
        profit_summary_df = pd.DataFrame(columns=['Date', 'Total Profit'])

    if date_str in profit_summary_df['Date'].values:
        profit_summary_df.loc[profit_summary_df['Date'] == date_str, 'Total Profit'] = total_profit
    else:
        profit_summary_df = profit_summary_df.append({'Date': date_str, 'Total Profit': total_profit}, ignore_index=True)

    profit_summary_df.to_csv(PROFIT_SUMMARY_PATH, index=False)
    logging.info(f"Total profit for the day {date_str} is appended to or updated in {PROFIT_SUMMARY_PATH}")



from pytz import timezone
import datetime

def main():
    """
    Main function that executes the trading operations and saves reports.
    """

    # Get the current time in the NYSE timezone
    ny_time = datetime.datetime.now(timezone('America/New_York'))

    # If it's not yet 15:55, wait
    # while ny_time.hour < 15 or (ny_time.hour == 15 and ny_time.minute < 55):
    #     time.sleep(1)  # Wait for 1 second before checking the time again
    #     ny_time = datetime.datetime.now(timezone('America/New_York'))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cancel_open_orders()
    closed_orders = close_active_positions()
    wait_for_orders_to_fill(closed_orders)

    # Get account information
    account = trading_client.get_account()

    # Check if account is restricted from trading
    if account.trading_blocked:
        logging.error('Account is currently restricted from trading.')
        return

    # Check how much money we can use to open new positions
    logging.info(f'${account.buying_power} is available as buying power.')

    # Check our current balance vs. our balance at the last market close
    balance_change = float(account.equity) - float(account.last_equity)
    logging.info(f"Today's portfolio balance change: ${balance_change}")

    # Get a list of all orders.
    orders = trading_client.list_orders(status='closed', limit=MAX_ORDER_HISTORY)

    # Write the positions to the CSV file
    output_path = os.path.join(OUTPUT_DIR, f"{date_str} report.csv")
    df = save_report(orders, output_path)

    # Analyze and save summary
    summary_output_path = os.path.join(SUM_OUTPUT_DIR, f"{date_str} summary_report.csv")
    analyze_and_save_summary(df, summary_output_path)

if __name__ == "__main__":
    main()
