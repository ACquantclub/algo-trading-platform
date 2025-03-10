import json
import os
import pandas as pd
from typing import Dict, Optional

# Dictionary mapping stock symbols to their local currency codes
CURRENCY_CONVERSIONS = {
    "RELIANCE_BSE": "INR",  # Reliance Industries (Bombay Stock Exchange) - Indian Rupee
    "ADANIENT_BSE": "INR",  # Adani Enterprises (Bombay Stock Exchange) - Indian Rupee
    "300750_SHZ": "CNY",  # Contemporary Amperex Technology (Shenzhen) - Chinese Yuan
    "MC_PA": "EUR",  # LVMH Moët Hennessy (Paris) - Euro
}

# Exchange rates to USD as of March 2025
EXCHANGE_RATES = {
    "INR": 83.677,  # Indian Rupee to USD
    "CNY": 7.189,  # Chinese Yuan to USD
    "EUR": 0.924,  # Euro to USD
}


def load_stock_data(file_path: str) -> dict:
    """
    Load stock data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing stock data

    Returns:
        dict: Parsed JSON data containing stock price information

    Raises:
        FileNotFoundError: If the specified file does not exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    with open(file_path, "r") as f:
        return json.load(f)


def convert_to_usd(
    price: float, from_currency: str, exchange_rates: Dict[str, float]
) -> float:
    """
    Convert price from local currency to USD.

    Args:
        price (float): Price value in local currency
        from_currency (str): Currency code to convert from (e.g., 'INR', 'CNY')
        exchange_rates (Dict[str, float]): Dictionary of exchange rates relative to USD

    Returns:
        float: Price converted to USD, rounded to 3 decimal places

    Notes:
        - Returns original price if from_currency is 'USD' or None
        - Returns original price if from_currency is not found in exchange_rates
    """
    if from_currency == "USD" or from_currency is None:
        return price
    if from_currency not in exchange_rates:
        return price  # If no exchange rate found, assume price is in USD
    return round(price / exchange_rates[from_currency], 3)


def format_market_data(
    start_date: str = "2024-10-10",
    end_date: str = "2025-02-07",
    currency_conversions: Optional[Dict[str, str]] = None,
    exchange_rates: Optional[Dict[str, float]] = None,
):
    """
    Format market data with optional currency conversion for a specific date range.

    This function processes raw stock data files, filters them by date range,
    converts prices to USD if specified, and saves the formatted data to a JSON file.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        currency_conversions (Optional[Dict[str, str]]): Dict mapping stock symbols to
                                                       their local currency codes
        exchange_rates (Optional[Dict[str, float]]): Dict mapping currency codes to
                                                    USD exchange rates

    Returns:
        None

    Side Effects:
        - Creates directory './src/gradescope_autograder/data' if it doesn't exist
        - Writes the formatted data to './src/gradescope_autograder/data/training.json'

    Notes:
        - Stocks are assigned column names alphabetically (A, B, C, etc.)
        - Only dates with complete data across all stocks are included in the output
        - All prices are converted to USD based on provided exchange rates
    """
    # Define path to raw data directory
    data_dir = "./raw_data/20_year_daily"

    # Initialize dictionaries to store data by category
    open_data = {}
    high_data = {}
    low_data = {}
    close_data = {}
    volume_data = {}

    # Get list of all JSON files in the data directory
    stock_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    # Use default empty dictionaries if not provided
    currency_conversions = currency_conversions or {}
    exchange_rates = exchange_rates or {}

    # Process each stock file
    for i, file_name in enumerate(stock_files):
        file_path = os.path.join(data_dir, file_name)
        stock_data = load_stock_data(file_path)
        time_series = stock_data["Time Series (Daily)"]
        col_name = chr(65 + i)  # A=65 in ASCII, so stocks will be named A, B, C, etc.

        # Get currency for this stock, None if not specified
        stock_symbol = file_name.replace(".json", "")
        stock_currency = (
            currency_conversions.get(stock_symbol) if currency_conversions else None
        )

        # Process each date in time series
        for date_str, values in time_series.items():
            # Skip if date is not in our desired range
            if date_str < start_date or date_str > end_date:
                continue

            # Convert prices to USD if needed
            open_price = convert_to_usd(
                float(values["1. open"]), stock_currency, exchange_rates
            )
            high_price = convert_to_usd(
                float(values["2. high"]), stock_currency, exchange_rates
            )
            low_price = convert_to_usd(
                float(values["3. low"]), stock_currency, exchange_rates
            )
            close_price = convert_to_usd(
                float(values["4. close"]), stock_currency, exchange_rates
            )

            # Store data in respective dictionaries, creating date entries as needed
            open_data.setdefault(date_str, {})[col_name] = open_price
            high_data.setdefault(date_str, {})[col_name] = high_price
            low_data.setdefault(date_str, {})[col_name] = low_price
            close_data.setdefault(date_str, {})[col_name] = close_price
            volume_data.setdefault(date_str, {})[col_name] = int(values["5. volume"])

    # Convert nested dictionaries to pandas DataFrames for easier processing
    df_open = pd.DataFrame.from_dict(open_data, orient="index")
    df_high = pd.DataFrame.from_dict(high_data, orient="index")
    df_low = pd.DataFrame.from_dict(low_data, orient="index")
    df_close = pd.DataFrame.from_dict(close_data, orient="index")
    df_volume = pd.DataFrame.from_dict(volume_data, orient="index")

    # Sort indices (dates) in chronological order
    for df in [df_open, df_high, df_low, df_close, df_volume]:
        df.sort_index(ascending=True, inplace=True)

    # Create mask to identify rows with complete data across all DataFrames
    complete_data_mask = df_open.notna().all(axis=1)
    for df in [df_high, df_low, df_close, df_volume]:
        complete_data_mask &= df.notna().all(axis=1)

    # Filter out rows with incomplete data
    df_open = df_open[complete_data_mask]
    df_high = df_high[complete_data_mask]
    df_low = df_low[complete_data_mask]
    df_close = df_close[complete_data_mask]
    df_volume = df_volume[complete_data_mask]

    # Create final dictionary structure for output
    formatted_data = {
        "open": df_open.to_dict(orient="index"),
        "high": df_high.to_dict(orient="index"),
        "low": df_low.to_dict(orient="index"),
        "close": df_close.to_dict(orient="index"),
        "volume": df_volume.to_dict(orient="index"),
    }

    # Ensure output directory exists
    output_dir = "./src/gradescope_autograder/data"
    os.makedirs(output_dir, exist_ok=True)

    # Write formatted data to JSON file
    output_path = os.path.join(output_dir, "training.json")
    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)


if __name__ == "__main__":
    # Execute the data formatting process with predefined currency mappings and exchange rates
    format_market_data(
        currency_conversions=CURRENCY_CONVERSIONS, exchange_rates=EXCHANGE_RATES
    )
