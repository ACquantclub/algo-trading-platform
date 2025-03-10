"""
Pseudoname: Château Petito (chateaupetito@yahoo.com)
Strategy Name: Momentum Risk Optimization

This is the strategy that perplexity: o3-mini
generated from a one-shot query with the following prompt:
"Given the description of the competition in 1a_2025_algo_case_packet and the training data and using any external information you can find as help, create an advanced strategy does as good as possible in the competition"
The model was given access to the pdf version of the case packet, the training data, and the ability to search for external resources (all 4 perplexity source categories were turned on).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class AdvancedPortfolioOptimizer:
    def __init__(self, lookback=60, momentum_window=10, lambda_reg=0.5, gamma=0.5):
        # Set hyperparameters for estimation windows and risk trade-offs.
        self.lookback = lookback
        self.momentum_window = momentum_window
        self.lambda_reg = lambda_reg  # trade-off for total variance
        self.gamma = gamma  # trade-off for downside risk
        self.history = []  # store daily prices for each stock

    def update_history(self, prices):
        # prices: numpy array of shape (n_stocks,)
        self.history.append(prices)
        if len(self.history) > self.lookback:
            self.history.pop(0)

    def compute_returns(self):
        # Compute log returns from price history
        prices = np.array(self.history)
        returns = np.log(prices[1:] / prices[:-1])
        return returns

    def estimate_parameters(self):
        returns = self.compute_returns()
        # Estimate expected returns via EWMA momentum over the momentum_window
        if returns.shape[0] < self.momentum_window:
            # Not enough history: use equal weights or zero signal
            n = returns.shape[1]
            return np.zeros(n), np.eye(n)

        # Compute momentum as the recent average return (exponential weighting)
        weights = np.exp(np.linspace(-1, 0, self.momentum_window))
        weights /= np.sum(weights)
        momentum = np.average(returns[-self.momentum_window :], axis=0, weights=weights)

        # Estimate full covariance matrix over the lookback period
        cov = np.cov(returns.T)

        # Estimate downside risk for each asset: only consider negative returns
        downside_std = np.sqrt(np.mean(np.minimum(returns, 0) ** 2, axis=0))

        return momentum, cov, downside_std

    def objective(self, w, mu, cov, downside_std):
        # Objective: -expected return + lambda * variance + gamma * weighted downside risk
        # Here downside risk contribution is computed as the portfolio's exposure
        ret_term = -np.dot(w, mu)
        vol_term = self.lambda_reg * np.dot(w, np.dot(cov, w))
        # Downside risk (semi deviation) contribution: a weighted sum of individual downside risks
        down_term = self.gamma * np.dot(np.abs(w), downside_std)
        return ret_term + vol_term + down_term

    def allocate(self, market_data):
        # Extract closing prices from market data dictionary
        prices_today = market_data["close"]

        # Update historical prices
        self.update_history(prices_today)

        # If insufficient history, return equal weights
        if len(self.history) < self.lookback:
            n = len(prices_today)
            eq_weights = np.ones(n) / n
            return eq_weights

        # Estimate parameters (expected momentum, covariance, downside risk)
        mu, cov, downside_std = self.estimate_parameters()

        n = len(mu)
        # Start from an equal weight guess
        x0 = np.ones(n) / n

        # Define constraints for non-negative weights that sum to 1
        bounds = tuple((0, 1) for _ in range(n))  # Non-negative constraints
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Sum to 1

        # Optimize with constraints
        result = minimize(
            self.objective,
            x0,
            args=(mu, cov, downside_std),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            w = x0
        else:
            w = result.x

        # Ensure non-negative weights and normalization
        w = np.maximum(w, 0)  # Clip negative values to 0
        norm = np.sum(w)
        if norm > 0:
            w = w / norm
        else:
            w = x0

        return w


# Example of usage within the backtester:
# optimizer = AdvancedPortfolioOptimizer()
# For each day, call:
# weights = optimizer.allocate(today_data)
