# AlgoTrading System

This repository contains an algorithmic trading system built to interact with the Alpaca Trade API. It retrieves trading signals, places market orders, sets trailing stop orders, and adjusts beta exposure with the SPY ETF.

## Table of Contents
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Disclaimer](#disclaimer)

## Features

1. **Automatic Trading**: Based on the trading signals, the system will either buy or sell assets.
2. **Beta-adjusted Exposure**: The system calculates the beta-adjusted exposure to inform trading decisions.
3. **Reporting**: Detailed reports about executed trades are generated and saved as CSV files.
4. **Order Management**: Ability to cancel all open orders and close all active positions.
5. **Model Training**: The system includes a training module for machine learning models used in trading signal generation.

## Setup and Installation

1. **API Secrets**: Ensure you have `API_KEY`, `SECRET_KEY`, and `BASE_URL` saved securely. These will be imported from `algotrading.secrecy`.
2. **Folder Structure**: Create and maintain a specific directory structure for saving reports (`Detailed Report` and `Summary Report`).

## Usage

- **Trading**: Execute the `main_process` function. This function will go through the process of getting trading signals, placing necessary market and trailing stop orders, and then adjusting beta exposure.

- **Reporting**: Call the `main` function from the reporting module. This will generate a detailed report and a summary report based on recent trading activity.

- **Model Training**: Use the `train_model` function to train your machine learning models. Ensure your data is prepared and passed as required parameters.

## Dependencies

1. `alpaca_trade_api`: To interact with the Alpaca trading platform.
2. `pandas`: For data manipulation.
3. `logging`: For logging system events and trade activities.
4. `datetime`: To work with dates.
5. `os`: For file and directory operations.
6. `tensorflow`: For building and training machine learning models.

## Disclaimer

Trading involves financial risks. Please understand the system thoroughly and do thorough backtesting and paper trading before deploying real capital. Always consult with a financial advisor before making any trading decisions.

## Author
Adedapo Ayoade
