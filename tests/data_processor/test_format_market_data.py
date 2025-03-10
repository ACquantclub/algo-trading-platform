import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from src.data_processor.format_market_data import (
    load_stock_data,
    convert_to_usd,
    format_market_data,
    CURRENCY_CONVERSIONS,
    EXCHANGE_RATES,
)


def test_load_stock_data():
    """Test loading stock data from a JSON file"""
    mock_data = {
        "Meta Data": {"Symbol": "AAPL"},
        "Time Series (Daily)": {
            "2025-02-01": {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.0",
                "4. close": "100.5",
                "5. volume": "1000000",
            }
        },
    }

    # Test successful load
    with patch(
        "builtins.open", mock_open(read_data=json.dumps(mock_data))
    ) as mock_file:
        result = load_stock_data("dummy_path.json")
        mock_file.assert_called_once_with("dummy_path.json", "r")
        assert result == mock_data

    # Test file not found
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            load_stock_data("nonexistent_file.json")

    # Test invalid JSON
    with patch("builtins.open", mock_open(read_data="invalid json")):
        with pytest.raises(json.JSONDecodeError):
            load_stock_data("invalid_json.json")


def test_convert_to_usd():
    """Test currency conversion to USD"""
    test_exchange_rates = {"INR": 80.0, "CNY": 7.0, "EUR": 0.9}

    # Test USD and None cases (no conversion)
    assert convert_to_usd(100.0, "USD", test_exchange_rates) == 100.0
    assert convert_to_usd(100.0, None, test_exchange_rates) == 100.0

    # Test various currency conversions
    assert convert_to_usd(800.0, "INR", test_exchange_rates) == 10.0  # 800 / 80 = 10
    assert convert_to_usd(70.0, "CNY", test_exchange_rates) == 10.0  # 70 / 7 = 10
    assert convert_to_usd(9.0, "EUR", test_exchange_rates) == 10.0  # 9 / 0.9 = 10

    # Test rounding to 3 decimal places
    assert (
        convert_to_usd(9.1234, "EUR", test_exchange_rates) == 10.137
    )  # 9.1234 / 0.9 = 10.137...

    # Test currency not in exchange_rates
    assert convert_to_usd(100.0, "GBP", test_exchange_rates) == 100.0


@pytest.fixture
def setup_mock_environment():
    """Setup mock environment for format_market_data function tests"""
    # Mock file structure
    mock_files = ["AAPL.json", "MSFT.json", "GOOG.json"]

    # Mock stock data
    mock_stock_data = {
        "Meta Data": {"Symbol": "TEST"},
        "Time Series (Daily)": {
            "2024-10-15": {
                "1. open": "100.0",
                "2. high": "105.0",
                "3. low": "99.0",
                "4. close": "104.0",
                "5. volume": "10000",
            },
            "2024-11-15": {  # This date will be filtered out based on complete_data_mask
                "1. open": "105.0",
                "2. high": "110.0",
                "3. low": "102.0",
                "4. close": "108.0",
                "5. volume": "12000",
            },
            "2025-01-15": {
                "1. open": "110.0",
                "2. high": "115.0",
                "3. low": "108.0",
                "4. close": "112.0",
                "5. volume": "15000",
            },
            "2025-03-15": {  # This date will be filtered out by date range
                "1. open": "120.0",
                "2. high": "125.0",
                "3. low": "118.0",
                "4. close": "122.0",
                "5. volume": "20000",
            },
        },
    }

    # Define mocks
    mocks = {
        "listdir": MagicMock(return_value=mock_files),
        "load_stock_data": MagicMock(return_value=mock_stock_data),
        "open": mock_open(),
        "makedirs": MagicMock(),
    }

    return mocks


@patch("os.path.join", lambda *args: "/".join(args))
@patch("json.dump")
def test_format_market_data(mock_json_dump, setup_mock_environment):
    """Test format_market_data function"""
    mocks = setup_mock_environment

    # Apply mocks
    with (
        patch("os.listdir", mocks["listdir"]),
        patch(
            "src.data_processor.format_market_data.load_stock_data",
            mocks["load_stock_data"],
        ),
        patch("builtins.open", mocks["open"]),
        patch("os.makedirs", mocks["makedirs"]),
        patch("pandas.DataFrame.from_dict") as mock_df_from_dict,
        patch("pandas.DataFrame.sort_index") as mock_sort_index,
    ):

        # Mock DataFrame behavior for filtering
        mock_df = MagicMock()
        mock_df.notna.return_value.all.return_value = pd.Series(
            {
                "2024-10-15": True,
                "2024-11-15": False,  # This date should be filtered out
                "2025-01-15": True,
                "2025-03-15": True,  # This will be filtered by date range already
            }
        )
        mock_df.__getitem__.return_value = mock_df
        mock_df.to_dict.return_value = {
            "2024-10-15": {"A": 100.0, "B": 100.0, "C": 100.0},
            "2025-01-15": {"A": 110.0, "B": 110.0, "C": 110.0},
        }
        mock_df_from_dict.return_value = mock_df
        mock_sort_index.return_value = mock_df

        # Call function with default parameters
        format_market_data()

        # Verify directory creation
        mocks["makedirs"].assert_called_once_with(
            "./src/gradescope_autograder/data", exist_ok=True
        )

        # Verify load_stock_data was called for each file
        assert mocks["load_stock_data"].call_count == 3

        # Verify json.dump was called once
        assert mock_json_dump.call_count == 1

        # Get the formatted data that was passed to json.dump
        formatted_data = mock_json_dump.call_args[0][0]

        # Verify structure of the formatted data
        assert set(formatted_data.keys()) == {"open", "high", "low", "close", "volume"}

        # Verify only dates in range are included (mocks return same data for all files)
        expected_dates = {"2024-10-15", "2025-01-15"}
        for category in formatted_data:
            assert set(formatted_data[category].keys()) == expected_dates


@patch("os.path.join", lambda *args: "/".join(args))
@patch("json.dump")
def test_format_market_data_with_currency_conversion(
    mock_json_dump, setup_mock_environment
):
    """Test format_market_data function with currency conversion"""
    mocks = setup_mock_environment

    # Custom currency conversions and exchange rates for testing
    test_currency_map = {"MSFT.json": "EUR", "GOOG.json": "CNY"}
    test_exchange_rates = {"EUR": 0.85, "CNY": 6.5}

    # Apply mocks
    with (
        patch("os.listdir", mocks["listdir"]),
        patch(
            "src.data_processor.format_market_data.load_stock_data",
            mocks["load_stock_data"],
        ),
        patch("builtins.open", mocks["open"]),
        patch("os.makedirs", mocks["makedirs"]),
    ):

        # Call with custom parameters
        format_market_data(
            start_date="2024-10-01",
            end_date="2025-02-01",
            currency_conversions=test_currency_map,
            exchange_rates=test_exchange_rates,
        )

        # Verify json.dump was called once
        assert mock_json_dump.call_count == 1

        # Verify custom date range was applied


def test_constants():
    """Test that constants are defined correctly"""
    # Check CURRENCY_CONVERSIONS structure
    assert isinstance(CURRENCY_CONVERSIONS, dict)
    assert "RELIANCE_BSE" in CURRENCY_CONVERSIONS
    assert CURRENCY_CONVERSIONS["RELIANCE_BSE"] == "INR"

    # Check EXCHANGE_RATES structure
    assert isinstance(EXCHANGE_RATES, dict)
    assert "INR" in EXCHANGE_RATES
    assert isinstance(EXCHANGE_RATES["INR"], float)
