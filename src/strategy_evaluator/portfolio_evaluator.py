"""
Portfolio Evaluator Module

This module provides functionality for evaluating the performance of portfolio allocation strategies
using historical market data and calculating standard financial performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Type, Any
import importlib.util
import sys


class PortfolioEvaluator:
    """
    Evaluates performance of portfolio allocation strategies against historical market data.

    The class calculates various performance metrics such as Sharpe ratio, Sortino ratio,
    maximum drawdown, volatility, and other risk-adjusted return metrics.

    Attributes:
        market_data (dict): Dictionary with price and volume DataFrames (OHLCV)
        returns (pd.DataFrame): Asset returns with datetime index and asset columns
    """

    def __init__(self, market_data: dict[str, pd.DataFrame], returns: pd.DataFrame):
        """
        Initialize the portfolio evaluator with market data and returns.

        Args:
            market_data: Dictionary containing DataFrames for different data types:
                - 'open': Opening prices
                - 'close': Closing prices
                - 'high': High prices
                - 'low': Low prices
                - 'volume': Trading volumes
                All DataFrames must have datetime index and columns as asset names
            returns: DataFrame of asset returns with datetime index and asset columns

        Raises:
            ValueError: If market_data is missing required fields
        """
        required_fields = ["open", "close", "high", "low", "volume"]
        if not all(field in market_data for field in required_fields):
            raise ValueError(f"market_data must contain all fields: {required_fields}")

        self.market_data = market_data
        self.returns = returns

    def _validate_and_normalize(self, weights: np.ndarray) -> np.ndarray:
        """
        Validate and normalize portfolio weights.

        Args:
            weights: Array of portfolio weights

        Returns:
            Normalized weights that sum to 1.0

        Raises:
            ValueError: If weights have invalid length or contain negative values
        """
        if len(weights) != len(self.market_data["close"].columns):
            raise ValueError("Invalid weight vector length")

        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        return weights / np.sum(np.abs(weights))

    def evaluate_strategy(self, strategy_class: Type) -> Dict[str, Any]:
        """
        Run strategy and calculate comprehensive performance metrics.

        This method iterates through historical data, allocates portfolio weights
        using the provided strategy, and computes performance metrics.

        Args:
            strategy_class: Class implementing the allocation strategy
                           (must have an allocate method)

        Returns:
            Dictionary containing performance metrics:
                - sharpe_ratio: Risk-adjusted return metric
                - sortino_ratio: Downside risk-adjusted return metric
                - max_drawdown: Maximum peak to trough decline
                - volatility: Standard deviation of returns
                - annual_return: Annualized return
                - calmar_ratio: Annual return divided by max drawdown
                - win_rate: Proportion of positive return periods
                - returns: Time series of portfolio returns
                - weights: Time series of portfolio weights
                - cumulative_returns: Time series of cumulative returns
                - drawdown_series: Time series of drawdowns
                - rolling_sharpe: Rolling Sharpe ratio (3-month window)
                - rolling_volatility: Rolling volatility (3-month window)
                - rolling_var_95: Rolling 5% value-at-risk (3-month window)

        Raises:
            ValueError: If strategy evaluation fails
        """
        try:
            n_periods = len(self.market_data["close"]) - 1
            n_assets = len(self.market_data["close"].columns)
            weights_history = np.zeros((n_periods, n_assets))

            # Create single instance of strategy class
            strategy_instance = strategy_class()

            # Prepare market data arrays
            market_arrays = {
                field: data.values for field, data in self.market_data.items()
            }

            # Get allocation for each time step
            for t in range(n_periods):
                current_data = {
                    field: arrays[t] for field, arrays in market_arrays.items()
                }
                try:
                    weights = strategy_instance.allocate(current_data)
                except Exception as e:
                    raise RuntimeError(f"Error in strategy allocate() method: {str(e)}")
                if not isinstance(weights, np.ndarray):
                    weights = np.array(weights)

                try:
                    weights_history[t] = self._validate_and_normalize(weights)
                except Exception as e:
                    raise ValueError(f"Error in strategy weights: {str(e)}")

            weights_df = pd.DataFrame(
                weights_history,
                index=self.returns.index,
                columns=self.market_data["close"].columns,
            )

            # Calculate daily portfolio returns
            portfolio_returns = np.sum(weights_df.values * self.returns.values, axis=1)
            portfolio_returns = pd.Series(portfolio_returns, index=self.returns.index)

            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()

            # Calculate drawdown series
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1

            # Calculate risk metrics (assuming 2% annual risk-free rate)
            excess_returns = portfolio_returns - 0.02 / 252
            negative_returns = portfolio_returns[portfolio_returns < 0]

            metrics = {
                "sharpe_ratio": portfolio_returns.mean() / portfolio_returns.std(),
                "sortino_ratio": excess_returns.mean() / negative_returns.std(),
                "max_drawdown": drawdowns.min(),
                "volatility": portfolio_returns.std(),
                "annual_return": np.prod(1 + portfolio_returns) - 1,
                "calmar_ratio": (np.prod(1 + portfolio_returns) - 1)
                / abs(drawdowns.min()),
                "win_rate": len(portfolio_returns[portfolio_returns > 0])
                / len(portfolio_returns),
                "returns": portfolio_returns,
                "weights": weights_df,
                "cumulative_returns": cumulative_returns,
                "drawdown_series": drawdowns,
            }

            # Calculate rolling metrics (3-month window)
            WINDOW = 63  # ~3 months of trading days
            metrics.update(
                {
                    "rolling_sharpe": portfolio_returns.rolling(WINDOW).mean()
                    / portfolio_returns.rolling(WINDOW).std()
                    * np.sqrt(252),
                    "rolling_volatility": portfolio_returns.rolling(WINDOW).std()
                    * np.sqrt(252),
                    "rolling_var_95": portfolio_returns.rolling(WINDOW).quantile(0.05),
                }
            )

            return metrics
        except Exception as e:
            raise ValueError(f"Error evaluating strategy: {str(e)}")

    @staticmethod
    def load_submission(submission_path: str) -> Type:
        """
        Load a submitted strategy class from a Python file.

        Args:
            submission_path: Path to the Python file containing the strategy class

        Returns:
            The strategy class (that has an allocate method)

        Raises:
            ValueError: If the file doesn't contain a valid strategy class
            ImportError: If the file cannot be loaded
        """
        if not submission_path.endswith(".py"):
            raise ValueError("Submission must be a Python file")

        spec = importlib.util.spec_from_file_location("submission", submission_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not load submission file")

        module = importlib.util.module_from_spec(spec)
        sys.modules["submission"] = module
        spec.loader.exec_module(module)

        # Look for a class that has an allocate method
        strategy_class = None
        for name, obj in module.__dict__.items():
            if isinstance(obj, type) and hasattr(obj, "allocate"):
                strategy_class = obj
                break

        if strategy_class is None:
            raise ValueError("Missing strategy class with allocate method")

        return strategy_class
