import os
import json
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock, call
from datetime import datetime

from src.strategy_evaluator.grader import Grader
from src.strategy_evaluator.portfolio_evaluator import PortfolioEvaluator


class MockStrategy:
    def allocate(self, prices, **kwargs):
        return [0.5, 0.3, 0.2]  # Mock allocation weights


class MockInvalidStrategy:
    # No allocate method
    pass


@pytest.fixture
def mock_market_data():
    # Create mock price data similar to what would be in testing.json
    dates = pd.date_range(start="2023-01-01", periods=10)
    mock_data = {
        "open": {str(d): [100.0, 200.0, 300.0] for d in dates},
        "close": {str(d): [110.0, 220.0, 330.0] for d in dates},
        "high": {str(d): [115.0, 225.0, 335.0] for d in dates},
        "low": {str(d): [95.0, 195.0, 295.0] for d in dates},
        "volume": {str(d): [1000, 2000, 3000] for d in dates},
    }
    return json.dumps(mock_data)


@pytest.fixture
def mock_evaluator():
    evaluator = MagicMock(spec=PortfolioEvaluator)
    evaluator.evaluate_strategy.return_value = {
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "annual_return": 0.15,
        "max_drawdown": 0.05,
        "volatility": 0.10,
        "win_rate": 0.60,
        "calmar_ratio": 3.0,
    }
    return evaluator


# Patch the setup_market_data method to avoid file access
@patch("builtins.open", new_callable=mock_open)
@patch("pandas.read_json")
@patch("src.strategy_evaluator.grader.PortfolioEvaluator")
def test_init_and_setup_market_data(
    mock_portfolio_evaluator, mock_read_json, mock_file, mock_market_data
):
    """Test initialization and market data setup"""
    # Setup mock read_json to return proper DataFrames with required fields
    mock_dates = pd.date_range(start="2023-01-01", periods=10)
    mock_dfs = {}

    for field in ["open", "close", "high", "low", "volume"]:
        mock_df = pd.DataFrame(
            data=[[100.0, 200.0, 300.0]] * 10,
            index=mock_dates,
            columns=["AAPL", "MSFT", "GOOG"],
        )
        mock_dfs[field] = mock_df

    # Mock read_json to return a DataFrame with our data structure
    mock_read_json.return_value = pd.DataFrame(
        columns=["open", "close", "high", "low", "volume"]
    )
    mock_read_json.return_value.to_dict = lambda: {
        field: mock_dfs[field].to_dict()
        for field in ["open", "close", "high", "low", "volume"]
    }

    # Mock evaluator instance
    mock_evaluator_instance = MagicMock()
    mock_portfolio_evaluator.return_value = mock_evaluator_instance

    # Patch os.path.join to return a consistent path
    with patch("os.path.join", return_value="/autograder/source/data/testing.json"):
        # Create Grader instance without patching setup_market_data
        grader = Grader()

    # Assertions
    assert hasattr(grader, "evaluator")


@patch.object(Grader, "setup_market_data")
@patch("os.path.join", return_value="/autograder/submission/strategy.py")
@patch("os.listdir")
@patch("importlib.util.spec_from_file_location")
@patch("importlib.util.module_from_spec")
def test_find_strategy_file_success(
    mock_module_from_spec, mock_spec_from_file, mock_listdir, mock_path_join, mock_setup
):
    """Test finding a valid strategy file"""
    # Setup mock listdir
    mock_listdir.return_value = ["strategy.py", "README.md"]

    # Setup mock module with a valid strategy class
    mock_module = MagicMock()
    mock_module_from_spec.return_value = mock_module

    # Mock spec_from_file_location
    mock_spec = MagicMock()
    mock_spec_from_file.return_value = mock_spec

    # Instead of mocking __dir__, set dir(mock_module) to return the desired result
    # This correctly replaces the behavior of dir() on the mock object
    type(mock_module).__dir__ = lambda self: ["PortfolioStrategy", "other_item"]

    # Create a proper class mock instead of using __getattr__
    mock_module.PortfolioStrategy = MockStrategy

    # Create Grader instance and find strategy
    grader = Grader()
    strategy_class = grader.find_strategy_file()

    # Assertions
    mock_listdir.assert_called_once_with("/autograder/submission")
    mock_spec_from_file.assert_called_once_with(
        "strategy_module", "/autograder/submission/strategy.py"
    )
    mock_module_from_spec.assert_called_once_with(mock_spec)
    mock_spec.loader.exec_module.assert_called_once_with(mock_module)
    assert strategy_class == MockStrategy


@patch.object(Grader, "setup_market_data")
@patch("os.path.join", return_value="/autograder/submission/invalid.py")
@patch("os.listdir")
def test_find_strategy_file_failure(mock_listdir, mock_path_join, mock_setup):
    """Test when no valid strategy file is found"""
    # Setup mock listdir
    mock_listdir.return_value = ["invalid.py", "README.md"]

    # Create Grader instance
    grader = Grader()

    # Mock import that raises exception
    with patch("importlib.util.spec_from_file_location") as mock_spec:
        mock_spec.side_effect = Exception("Import error")

        # Assert that FileNotFoundError is raised
        with pytest.raises(FileNotFoundError, match="No valid strategy file found"):
            grader.find_strategy_file()

    mock_listdir.assert_called_once_with("/autograder/submission")


@patch.object(Grader, "setup_market_data")
@patch("src.strategy_evaluator.grader.datetime")
def test_grade_submission_success(mock_datetime, mock_setup, mock_evaluator):
    """Test successful grading of a submission"""
    # Setup datetime.now mock to return consistent values
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    end_time = datetime(2023, 1, 1, 12, 0, 10)  # 10 seconds later
    mock_datetime.now.side_effect = [start_time, end_time]

    # Create Grader instance
    grader = Grader()

    # Set the evaluator attribute directly instead of patching it
    grader.evaluator = mock_evaluator

    # Mock methods to avoid actual file operations
    with patch.object(grader, "find_strategy_file", return_value=MockStrategy):
        # Call grade_submission
        results = grader.grade_submission()

    # Assertions for the success path
    assert results["score"] == 2
    assert results["execution_time"] == 10  # 10 seconds difference
    assert len(results["tests"]) == 2
    assert all(test["status"] == "passed" for test in results["tests"])
    assert len(results["leaderboard"]) == 7  # All metrics added


@patch.object(Grader, "setup_market_data")
@patch("src.strategy_evaluator.grader.datetime")
def test_grade_submission_failure_find_strategy(mock_datetime, mock_setup):
    """Test grading when find_strategy_file raises an exception"""
    # Setup datetime.now mock
    start_time = datetime(2023, 1, 1, 12, 0, 0)
    end_time = datetime(2023, 1, 1, 12, 0, 5)
    mock_datetime.now.side_effect = [start_time, end_time]

    # Create Grader instance
    grader = Grader()

    # Mock methods
    with patch.object(
        grader, "find_strategy_file", side_effect=FileNotFoundError("No valid strategy")
    ):
        # Call grade_submission
        results = grader.grade_submission()

    # Assertions for the failure path
    assert results["score"] == 0
    assert results["execution_time"] == 5
    assert len(results["tests"]) == 1
    assert results["tests"][0]["status"] == "failed"
    assert "No valid strategy" in results["output"]
