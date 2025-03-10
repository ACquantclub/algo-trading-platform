"""
Pseudoname: Twin Spark (ttwinsspark@proton.me)
Strategy Name: Mean-Variance

This is the strategy that GitHub Copilot: Gemini 2.0 Flash
generated from a one-shot query with the following prompt:
"Given the description of the competition in 1a_2025_algo_case_packet, the sample code, and the training data, create an advanced strategy does as good as possible in the competition."
The model was given access to the Latex version of the case packet, the sample code, and the training data.
"""

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

#########################################################################
## Implement your portfolio allocation strategy as a class with an      ##
## allocate method that takes in market data for one day and returns   ##
## portfolio weights.                                                  ##
#########################################################################


class PortfolioStrategy:
    def __init__(self, lookback_window=20):
        """
        Initialize any strategy parameters here
        """
        self.lookback_window = lookback_window
        self.historical_data = {}  # Store historical market data
        self.asset_list = None
        self.mu = None
        self.sigma = None

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
        # 1. Data Preparation
        closes = market_data["close"]  # Now this is directly a numpy array
        if self.asset_list is None:
            self.asset_list = list(
                range(closes.shape[0])
            )  # Use array indices as asset IDs

        # Update historical data
        if not self.historical_data:
            for key in self.asset_list:
                self.historical_data[key] = []

        for key in self.asset_list:
            self.historical_data[key].append(
                closes[key]
            )  # Directly access array elements

        # Check if enough historical data is available
        if len(self.historical_data[self.asset_list[0]]) < self.lookback_window:
            n_assets = len(self.asset_list)
            weights = np.ones(n_assets) / n_assets
            return weights

        # 2. Calculate Returns
        returns = pd.DataFrame(self.historical_data).pct_change().dropna()

        # 3. Estimate Expected Returns and Covariance Matrix
        self.mu = returns.mean() * 252  # Annualize returns
        self.sigma = returns.cov() * 252  # Annualize covariance

        # 4. Optimization
        weights = self.efficient_frontier()

        return weights

    def efficient_frontier(self):
        """
        Calculates portfolio weights based on efficient frontier optimization.
        """
        n_assets = len(self.asset_list)
        args = (self.mu.values, self.sigma.values)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for asset in range(n_assets))  # Weights between 0 and 1
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Define the objective function (negative Sharpe Ratio)
        def neg_sharpe_ratio(weights, mu, sigma):
            portfolio_return = np.sum(mu * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            sharpe_ratio = (
                portfolio_return
            ) / portfolio_std  # Risk-free rate is assumed to be 0
            return -sharpe_ratio

        # Minimize the negative Sharpe Ratio
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            args=args,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Optimized weights
        weights = result.x
        return weights
