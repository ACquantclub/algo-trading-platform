"""
Pseudoname: Cloudy Threnody (cloudythrenody@gmail.com)
Strategy Name: Momentum-Volatility Optimizer

This is the strategy that GitHub Copilot: Claude 3.5 Sonnet
generated from a one-shot query with the following prompt:
"Given the description of the competition in 1a_2025_algo_case_packet, the sample code, and the training data, create an advanced strategy does as good as possible in the competition."
The model was given access to the Latex version of the case packet, the sample code, and the training data.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioStrategy:
    def __init__(self):
        """
        Initialize strategy parameters and historical data storage
        """
        # Strategy parameters
        self.lookback_period = 20  # Days to look back for momentum
        self.volatility_period = 60  # Days for volatility calculation
        self.momentum_weight = 0.6  # Weight given to momentum signal
        self.vol_weight = 0.4  # Weight given to volatility signal

        # Storage for historical data
        self.historical_closes = []
        self.historical_highs = []
        self.historical_lows = []
        self.historical_volumes = []

        # Risk parameters
        self.max_weight = 0.3  # Maximum weight for any single asset
        self.min_weight = 0.01  # Minimum weight if asset is selected

    def calculate_momentum_signal(self):
        """Calculate momentum signals based on price trends"""
        if len(self.historical_closes) < self.lookback_period:
            return np.ones(len(self.historical_closes[-1])) / len(
                self.historical_closes[-1]
            )

        # Calculate multiple momentum factors
        closes = np.array(self.historical_closes)

        # 1. Price momentum (returns over lookback period)
        returns = closes[-1] / closes[-self.lookback_period] - 1

        # 2. Moving average crossover
        ma_short = np.mean(closes[-10:], axis=0)
        ma_long = np.mean(closes[-self.lookback_period :], axis=0)
        ma_signal = ma_short / ma_long - 1

        # 3. Volume-weighted price momentum
        volumes = np.array(self.historical_volumes[-self.lookback_period :])
        vwap = np.sum(closes[-self.lookback_period :] * volumes, axis=0) / np.sum(
            volumes, axis=0
        )
        vwap_signal = closes[-1] / vwap - 1

        # Combine signals
        momentum_signal = 0.4 * returns + 0.3 * ma_signal + 0.3 * vwap_signal
        return momentum_signal

    def calculate_volatility_signal(self):
        """Calculate volatility-based signals"""
        if len(self.historical_closes) < self.volatility_period:
            return np.ones(len(self.historical_closes[-1])) / len(
                self.historical_closes[-1]
            )

        closes = np.array(self.historical_closes[-self.volatility_period :])
        highs = np.array(self.historical_highs[-self.volatility_period :])
        lows = np.array(self.historical_lows[-self.volatility_period :])

        # 1. Historical volatility
        returns = np.diff(np.log(closes), axis=0)
        volatility = np.std(returns, axis=0) * np.sqrt(252)

        # 2. True Range volatility
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
        )
        atr = np.mean(tr, axis=0)

        # Combine into inverse volatility signal (lower volatility = higher signal)
        vol_signal = 1 / (0.5 * volatility + 0.5 * atr)
        return vol_signal

    def optimize_weights(self, combined_signal):
        """Optimize portfolio weights with constraints"""
        n_assets = len(combined_signal)

        def objective(weights):
            return -np.sum(weights * combined_signal)

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Weights sum to 1
            {"type": "ineq", "fun": lambda x: x - self.min_weight},  # Minimum weight
            {"type": "ineq", "fun": lambda x: self.max_weight - x},  # Maximum weight
        ]

        bounds = [(0, self.max_weight) for _ in range(n_assets)]

        initial_weights = np.ones(n_assets) / n_assets
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    def allocate(self, market_data: dict) -> np.ndarray:
        """
        Generate portfolio weights based on market data
        """
        # Update historical data
        self.historical_closes.append(market_data["close"])
        self.historical_highs.append(market_data["high"])
        self.historical_lows.append(market_data["low"])
        self.historical_volumes.append(market_data["volume"])

        # Calculate signals
        momentum_signal = self.calculate_momentum_signal()
        volatility_signal = self.calculate_volatility_signal()

        # Combine signals
        combined_signal = (
            self.momentum_weight * momentum_signal + self.vol_weight * volatility_signal
        )

        # If not enough historical data, return equal weights
        if len(self.historical_closes) < max(
            self.lookback_period, self.volatility_period
        ):
            return np.ones(len(market_data["close"])) / len(market_data["close"])

        # Optimize weights based on combined signal
        weights = self.optimize_weights(combined_signal)

        # Ensure weights sum to 1 and are non-negative
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)

        return 1 - weights
