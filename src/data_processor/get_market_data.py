"""
Market Data Retrieval Module

This module provides functionality to fetch historical stock market data from
the Alpha Vantage API. It handles requesting time series data for multiple
stock symbols and stores the results in JSON format.
"""

import json
import os
import pandas as pd
from typing import List, Dict
import requests
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def fetch_multiple_stocks_data(
    ticker_symbols: List[str], outputsize: str = "compact"
) -> Dict[str, pd.DataFrame]:
    """
    Fetches daily stock data for multiple ticker symbols using the Alpha Vantage API.

    The function retrieves time series data for each symbol, saves the raw JSON response
    to disk, and handles any errors that occur during the API calls.

    Args:
        ticker_symbols: List of stock ticker symbols to retrieve data for.
        outputsize: Data size parameter for Alpha Vantage API.
                   'compact' returns the latest 100 data points.
                   'full' returns 20+ years of historical data when available.

    Returns:
        Dictionary mapping ticker symbols to their corresponding DataFrame of historical data.
        Note: Currently returns an empty dictionary as DataFrame conversion is not implemented.

    Raises:
        No exceptions are raised as errors are caught and logged internally.
    """
    # Create data directory if it doesn't exist
    data_dir = "./data/20_year_daily"
    os.makedirs(data_dir, exist_ok=True)

    stock_data = {}
    failed_tickers = []

    # Alpha Vantage API endpoint
    base_url = "https://www.alphavantage.co/query"

    # Process each ticker symbol
    for symbol in ticker_symbols:
        try:
            # Set up request parameters
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
                "apikey": ALPHA_VANTAGE_API_KEY,
            }

            # Make API request
            response = requests.get(base_url, params=params)
            data = response.json()

            # Check if response contains expected data
            if "Time Series (Daily)" not in data:
                failed_tickers.append(symbol)
                continue

            # Save raw JSON response to file
            output_filename = os.path.join(data_dir, f"{symbol.replace('.', '_')}.json")
            with open(output_filename, "w") as f:
                json.dump(data, f)

            print(f"Successfully fetched data for {symbol}")

        except Exception as e:
            # Log any errors that occur
            failed_tickers.append(symbol)
            print(f"Failed to fetch data for {symbol}: {str(e)}")

    # Report any failures
    if failed_tickers:
        print("\nFailed to fetch data for the following tickers:")
        print(", ".join(failed_tickers))

    return stock_data


if __name__ == "__main__":
    # List of ticker symbols to retrieve data for
    tickers = [
        "TCEHY",  # Tencent Holdings
        "300750.SHZ",  # Contemporary Amperex Technology (CATL)
        "BYDDY",  # BYD Company
        "MC.PA",  # LVMH Moët Hennessy Louis Vuitton
        "NEE",  # NextEra Energy
        "HTHIY",  # Hitachi
    ]

    # Fetch stock data with full historical range
    stock_data = fetch_multiple_stocks_data(tickers, outputsize="full")
