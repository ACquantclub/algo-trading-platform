src/utils/format_market_data.py takes in all the data in data/20_year_daily and converts it into a dict of dict of dict, with the following structure:

{
    "open": {
        "DATE": {
            "STOCK_A": 200,
            "STOCK_B": 300
        },
        "DATE": {
            "STOCK_A": 600,
            "STOCK_B": 900
        }
    },
    "close": {
        "DATE": {
            "STOCK_A": 200,
            "STOCK_B": 300
        },
        "DATE": {
            "STOCK_A": 600,
            "STOCK_B": 900
        }
    },
    "high": {
        "DATE": {
            "STOCK_A": 200,
            "STOCK_B": 300
        },
        "DATE": {
            "STOCK_A": 600,
            "STOCK_B": 900
        }
    },
    "low": {
        "DATE": {
            "STOCK_A": 200,
            "STOCK_B": 300
        },
        "DATE": {
            "STOCK_A": 600,
            "STOCK_B": 900
        }
    },
    "volume": {
        "DATE": {
            "STOCK_A": 200,
            "STOCK_B": 300
        },
        "DATE": {
            "STOCK_A": 600,
            "STOCK_B": 900
        }
    }
}

If a stock is not traded in USD, the data formater will convert it using the average exchange rate of 2024.

training.json starts in 2024-10-10 and ends in 2025-02-10 with 72 trading days.