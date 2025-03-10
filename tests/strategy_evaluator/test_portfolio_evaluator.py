import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import importlib.util
from src.strategy_evaluator.portfolio_evaluator import PortfolioEvaluator


class MockStrategy:
    def allocate(self, market_data):
        # Simple strategy that returns equal weights
        return np.ones(len(market_data["close"]))


class NegativeWeightsStrategy:
    def allocate(self, market_data):
        # Strategy that returns negative weights to test validation
        return np.array([-1, -1, -1])


class NonNumpyStrategy:
    def allocate(self, market_data):
        # Strategy that returns a list instead of numpy array
        return [0.5, 0.5, 0]


class ErrorStrategy:
    def allocate(self, market_data):
        # Strategy that raises an exception
        raise ValueError("Intentional error")


@pytest.fixture
def sample_market_data():
    # Create sample market data with 3 assets and 10 days
    dates = pd.date_range(start="2023-01-01", periods=10)
    assets = ["AAPL", "GOOGL", "MSFT"]

    # Create random data for each OHLCV field
    np.random.seed(42)  # for reproducibility
    data = {}

    for field in ["open", "close", "high", "low"]:
        base_prices = np.random.rand(1, 3) * 100 + 50  # Base prices around $50-$150
        daily_changes = np.random.normal(
            0, 0.01, (10, 3)
        )  # Daily changes with 1% std dev

        # Generate price sequences
        prices = np.cumsum(daily_changes, axis=0) + base_prices

        # Ensure high >= open/close and low <= open/close
        if field == "high":
            prices = prices * 1.01  # High is 1% above
        elif field == "low":
            prices = prices * 0.99  # Low is 1% below

        data[field] = pd.DataFrame(prices, index=dates, columns=assets)

    # Volume data
    volumes = np.random.randint(100000, 1000000, (10, 3))
    data["volume"] = pd.DataFrame(volumes, index=dates, columns=assets)

    # Calculate returns
    returns = data["close"].pct_change().dropna()

    return data, returns


def test_initialization(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    assert evaluator.market_data == market_data
    assert evaluator.returns.equals(returns)


def test_initialization_missing_fields():
    incomplete_data = {
        "open": pd.DataFrame(),
        "close": pd.DataFrame(),
    }  # Missing required fields
    with pytest.raises(ValueError, match="market_data must contain all fields"):
        PortfolioEvaluator(incomplete_data, pd.DataFrame())


def test_validate_and_normalize(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    # Test valid weights
    weights = np.array([0.5, 0.3, 0.2])
    normalized = evaluator._validate_and_normalize(weights)
    assert np.allclose(normalized, np.array([0.5, 0.3, 0.2]))
    assert np.isclose(np.sum(normalized), 1.0)

    # Test weights that need normalization
    weights = np.array([5, 3, 2])
    normalized = evaluator._validate_and_normalize(weights)
    assert np.allclose(normalized, np.array([0.5, 0.3, 0.2]))

    # Test invalid length
    with pytest.raises(ValueError, match="Invalid weight vector length"):
        evaluator._validate_and_normalize(np.array([0.5, 0.5]))


def test_negative_weights_validation(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    # Test negative weights
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        evaluator._validate_and_normalize(np.array([-0.5, 0.8, 0.7]))


def test_evaluate_strategy(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    result = evaluator.evaluate_strategy(MockStrategy)

    # Check if all expected metrics are present
    expected_metrics = [
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "volatility",
        "annual_return",
        "calmar_ratio",
        "win_rate",
        "returns",
        "weights",
        "cumulative_returns",
        "drawdown_series",
        "rolling_sharpe",
        "rolling_volatility",
        "rolling_var_95",
    ]

    for metric in expected_metrics:
        assert metric in result

    # Test dimensions of time series outputs
    assert len(result["returns"]) == len(returns)
    assert result["weights"].shape == (len(returns), len(market_data["close"].columns))


def test_non_numpy_weights(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    # Test strategy that returns lists instead of numpy arrays
    result = evaluator.evaluate_strategy(NonNumpyStrategy)
    assert isinstance(result["weights"], pd.DataFrame)


def test_error_in_strategy(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    # Test strategy that raises an error
    with pytest.raises(ValueError, match="Error in strategy allocate"):
        evaluator.evaluate_strategy(ErrorStrategy)


def test_sharpe_ratio_calculation(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    result = evaluator.evaluate_strategy(MockStrategy)

    # Manual calculation of Sharpe ratio for verification
    portfolio_returns = result["returns"]
    expected_sharpe = portfolio_returns.mean() / portfolio_returns.std()
    assert np.isclose(result["sharpe_ratio"], expected_sharpe)


def test_rolling_metrics(sample_market_data):
    market_data, returns = sample_market_data
    evaluator = PortfolioEvaluator(market_data, returns)

    result = evaluator.evaluate_strategy(MockStrategy)

    # Check rolling metrics
    assert isinstance(result["rolling_sharpe"], pd.Series)
    assert isinstance(result["rolling_volatility"], pd.Series)
    assert isinstance(result["rolling_var_95"], pd.Series)


def test_load_submission(tmp_path):
    # Create a temporary strategy file
    strategy_file = tmp_path / "test_strategy.py"
    with open(strategy_file, "w") as f:
        f.write(
            """
class TestStrategy:
    def allocate(self, market_data):
        import numpy as np
        return np.array([0.5, 0.5])
        """
        )

    # Test loading the strategy
    strategy_class = PortfolioEvaluator.load_submission(str(strategy_file))
    assert hasattr(strategy_class, "allocate")

    # Test loading non-Python file
    with pytest.raises(ValueError, match="must be a Python file"):
        PortfolioEvaluator.load_submission("not_a_python_file.txt")

    # Test loading file without strategy class
    invalid_file = tmp_path / "invalid.py"
    with open(invalid_file, "w") as f:
        f.write("x = 1")

    with pytest.raises(ValueError, match="Missing strategy class"):
        PortfolioEvaluator.load_submission(str(invalid_file))


def test_import_error(monkeypatch):
    # Mock spec_from_file_location to return None
    def mock_spec_from_file_location(*args, **kwargs):
        return None

    monkeypatch.setattr(
        importlib.util, "spec_from_file_location", mock_spec_from_file_location
    )

    with pytest.raises(ImportError, match="Could not load submission file"):
        PortfolioEvaluator.load_submission("test.py")
