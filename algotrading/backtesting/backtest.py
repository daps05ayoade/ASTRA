import backtrader as bt
import yfinance as yf
import pandas as pd
import os

# Define MACDStrategy class
class Strategy(bt.Strategy):
    """
    This class defines a trading strategy using MACD as a signal and considering a primary signal from a DataFrame.
    """
    params = (
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
        ('df', None),  # add DataFrame as a parameter
    )

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime[0]
        dt = bt.num2date(dt)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.params.macd1,
                                       period_me2=self.params.macd2,
                                       period_signal=self.params.macdsig)
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
            else:
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def next(self):
        signal = self.params.df.loc[self.params.df['Ticker'] == self.data._name, 'Signal'].values[0]

        if not self.position:
            if self.crossover > 0 and signal == 'Buy':
                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.order = self.buy()
            elif self.crossover < 0 and signal == 'Sell':
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.order = self.sell()

        else:
            if self.crossover < 0 and signal == 'Buy':
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.order = self.sell()
            elif self.crossover > 0 and signal == 'Sell':
                self.log('BUY CREATE, %.2f' % self.data.close[0])
                self.order = self.buy()


def run_backtest(df):
    """
    This function performs backtesting on multiple stocks based on the MACDStrategy and a primary signal from a DataFrame.
    """
    initial_cash = 10000.0  # Initial cash

    # Loop over all rows in the DataFrame
    for index, row in df.iterrows():
        ticker = row['Ticker']
        signal = row['Signal']

        # Skip this ticker if signal is 'Hold'
        if signal == 'Hold':
            continue

        print(f'Backtesting {ticker}...')  # Log the ticker currently being backtested

        # Download the data
        data = yf.download(ticker, start='2020-01-01')

        # Instantiate Cerebro engine
        cerebro = bt.Cerebro()

        # Convert the DataFrame to a Backtrader data feed
        datafeeds = bt.feeds.PandasData(dataname=data)

        # Add data feed to Cerebro
        cerebro.adddata(datafeeds, name=ticker)

        # Add strategy and pass DataFrame as a parameter
        cerebro.addstrategy(Strategy, df=df)

        # Set commission and initial cash
        cerebro.broker.setcommission(commission=0.01)  # 1% commission
        cerebro.broker.setcash(initial_cash)

        # Run the backtest
        test = cerebro.run()

        # Calculate percentage return
        final_portfolio_value = cerebro.broker.getvalue()
        perc_return = (final_portfolio_value - initial_cash) / initial_cash * 100

        # Print out final portfolio value and the percentage return
        print(f'Final portfolio value for {ticker}: {cerebro.broker.getvalue():.2f}')
        print(f'Return for {ticker}: {perc_return:.2f}%')


data_path = '/home/adedapo/personal_project/daps05ayoade/disseration/algotrading/output'
final_df = pd.read_excel(os.path.join(data_path, 'Strategy.xlsx'))
run_backtest(final_df)
