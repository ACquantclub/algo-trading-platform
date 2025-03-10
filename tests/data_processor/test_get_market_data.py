import os
import json
import pytest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import requests

from src.data_processor.get_market_data import fetch_multiple_stocks_data


@pytest.fixture
def mock_env_variables():
    with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test_api_key"}):
        yield


@pytest.fixture
def mock_response():
    mock = MagicMock()

    # Sample successful response
    successful_response = {
        "Meta Data": {"1. Information": "Daily Prices", "2. Symbol": "TEST"},
        "Time Series (Daily)": {
            "2023-01-01": {
                "1. open": "100.0",
                "2. high": "105.0",
                "3. low": "98.0",
                "4. close": "102.0",
                "5. volume": "1000000",
            }
        },
    }

    # Sample failed response
    failed_response = {"Error Message": "Invalid API call"}

    # Configure the mock to return different responses for different symbols
    def get_side_effect(*args, **kwargs):
        response_mock = MagicMock()
        if kwargs.get("params", {}).get("symbol") == "TCEHY":
            response_mock.json.return_value = successful_response
            return response_mock
        elif kwargs.get("params", {}).get("symbol") == "ERROR":
            raise requests.exceptions.RequestException("API request failed")
        else:
            response_mock.json.return_value = failed_response
            return response_mock

    mock.get.side_effect = get_side_effect
    return mock


def test_fetch_multiple_stocks_data(mock_env_variables, mock_response):
    """Test fetching multiple stocks data from Alpha Vantage API"""

    test_tickers = ["TCEHY", "BYDDY", "ERROR"]

    # Apply mocks - add patch for the API key
    with (
        patch("requests.get", mock_response.get),
        patch("os.makedirs") as mock_makedirs,
        patch("builtins.open", mock_open()) as mock_file,
        patch("json.dump") as mock_json_dump,
        patch("builtins.print") as mock_print,
        patch(
            "src.data_processor.get_market_data.ALPHA_VANTAGE_API_KEY", "test_api_key"
        ),
    ):

        # Call the function
        result = fetch_multiple_stocks_data(test_tickers, outputsize="full")

        # Verify function behavior
        assert mock_makedirs.call_count == 1
        assert mock_makedirs.call_args[0][0] == "./data/20_year_daily"
        assert mock_makedirs.call_args[1] == {"exist_ok": True}

        # Verify requests were made for each ticker
        assert mock_response.get.call_count == 3

        # Verify json.dump was called for successful fetches
        assert mock_json_dump.call_count == 1  # Only TCEHY should succeed

        # Verify file operations
        assert mock_file.call_count == 1

        # Verify print statements
        assert (
            mock_print.call_count == 4
        )  # Success message + error message + 2 failed tickers messages

        # Verify the correct parameters were passed to requests.get
        for i, call in enumerate(mock_response.get.call_args_list):
            args, kwargs = call
            assert kwargs["params"]["function"] == "TIME_SERIES_DAILY"
            assert kwargs["params"]["symbol"] in test_tickers
            assert kwargs["params"]["outputsize"] == "full"
            assert kwargs["params"]["apikey"] == "test_api_key"
