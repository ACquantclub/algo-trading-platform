"""
Pseudoname: Chad G. Petty (ChadG.Petty@proton.me)
Strategy Name: Risk-adjusted momentum

This is the strategy that GitHub Copilot: GPT-4o
generated from a one-shot query with the following prompt:
"Given the description of the competition in 1a_2025_algo_case_packet, the sample code, and the training data, create an advanced strategy does as good as possible in the competition."
The model was given access to the to the Latex version of the case packet, the sample code, and the training data.
"""

import numpy as np
import pandas as pd


class PortfolioStrategy:
    def __init__(self, lookback_period=20):
        """
        Initialize any strategy parameters here
        """
        self.lookback_period = lookback_period
        self.price_history = {
            "open": [],
            "close": [],
            "high": [],
            "low": [],
            "volume": [],
        }

    def allocate(self, market_data: dict) -> np.ndarray:
        """
        market_data: Dictionary containing numpy arrays for:
            - 'open': Opening prices
            - 'close': Closing prices
            - 'high': High prices
            - 'low': Low prices
            - 'volume': Trading volumes
        for the current trading day

        Returns: numpy array of portfolio weights
        """
        # Update price history
        for key in self.price_history:
            self.price_history[key].append(market_data[key])
            if len(self.price_history[key]) > self.lookback_period:
                self.price_history[key].pop(0)

        # Ensure we have enough data to calculate momentum and volatility
        if len(self.price_history["close"]) < self.lookback_period:
            n_assets = len(market_data["close"])
            return np.ones(n_assets) / n_assets

        # Convert lists to numpy arrays for easier manipulation
        closes = np.array(self.price_history["close"])

        # Calculate momentum (percentage change over the lookback period)
        momentum = (closes[-1] - closes[0]) / closes[0]

        # Calculate volatility (standard deviation of returns over the lookback period)
        returns = np.diff(closes, axis=0) / closes[:-1]
        volatility = np.std(returns, axis=0)

        # Adjust momentum by volatility to get risk-adjusted momentum
        risk_adjusted_momentum = momentum / (
            volatility + 1e-8
        )  # Add small value to avoid division by zero

        # Normalize weights to sum to 1
        weights = np.maximum(risk_adjusted_momentum, 0)  # Ensure non-negative weights
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            n_assets = len(market_data["close"])
            weights = np.ones(n_assets) / n_assets

        return 1 - weights
