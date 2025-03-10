"""
Pseudoname: P. Lex & Co (p.lexco@yahoo.com)
Strategy Name: Resampled Momentum

This is the strategy that perplexity: Deepseek-r1
generated from a one-shot query with the following prompt:
"Given the description of the competition in 1a_2025_algo_case_packet and the training data and using any external information you can find as help, create an advanced strategy does as good as possible in the competition. Please give the code as a single block."
The model was given access to the pdf version of the case packet, the training data, and the ability to search for external resources (all 4 perplexity source categories were turned on).
"""

import numpy as np
from scipy.optimize import minimize


class AdvancedTradingStrategy:
    def __init__(self):
        self.momentum_window = 30  # Momentum calculation period
        self.volatility_window = 20  # Volatility calculation period
        self.risk_aversion = 0.7  # Risk penalty coefficient
        self.lookback = 60  # Covariance estimation window
        self.resample_count = 100  # Number of resampling iterations
        self.historical_prices = []

    def _compute_momentum(self, returns):
        """Calculate volatility-adjusted momentum scores"""
        mom = returns[-self.momentum_window :].mean(axis=0)
        vol = returns[-self.volatility_window :].std(axis=0, ddof=1)
        vol[vol < 1e-8] = 1e-8  # Prevent division by zero
        return mom / vol

    def _robust_covariance(self, returns):
        """Compute shrunk covariance matrix using Ledoit-Wolf approach"""
        n, p = returns.shape
        X = returns - returns.mean(axis=0)
        sample_cov = X.T @ X / (n - 1)

        # Shrinkage target (diagonal matrix with average variance)
        mu = np.trace(sample_cov) / p
        target = mu * np.eye(p)

        # Shrinkage intensity calculation
        delta = (np.linalg.norm(sample_cov - target, "fro") ** 2) / (
            p * (np.trace(sample_cov**2) / p - mu**2)
        )
        shrinkage = max(0, min(1, delta))

        return shrinkage * sample_cov + (1 - shrinkage) * target

    def _optimize_weights(self, expected_returns, covariance):
        """Constrained optimization for portfolio weights"""
        n_assets = len(expected_returns)

        def objective(weights):
            port_return = weights @ expected_returns
            port_risk = weights @ covariance @ weights
            return -(port_return - self.risk_aversion * port_risk)

        # Initial guess (equal weight inverse volatility portfolio)
        vol = np.diag(covariance) ** 0.5
        initial_weights = 1 / (vol + 1e-8)
        initial_weights /= np.sum(np.abs(initial_weights))

        # Modified constraints for long-only positions
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # weights sum to 1
        ]
        bounds = [
            (0, 1) for _ in range(n_assets)
        ]  # Changed bounds to (0,1) for long-only

        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x if result.success else initial_weights

    def allocate(self, daily_data):
        """Main allocation function called by backtester"""
        # Extract closing prices from the daily data
        daily_prices = daily_data["close"]
        self.historical_prices.append(daily_prices)

        if len(self.historical_prices) < self.lookback + 1:
            return np.zeros_like(daily_prices)

        # Convert prices to log returns
        price_matrix = np.array(self.historical_prices)
        returns = np.diff(np.log(price_matrix), axis=0)[-self.lookback :]

        # Rest of the code remains the same...
        # Step 1: Compute momentum-based expected returns
        expected_returns = self._compute_momentum(returns)

        # Step 2: Calculate robust covariance matrix
        covariance = self._robust_covariance(returns)

        # Step 3: Resampled portfolio optimization
        weight_samples = []
        for _ in range(self.resample_count):
            weights = np.random.exponential(scale=1.0, size=returns.shape[0])
            weights /= weights.sum()
            weighted_returns = (returns.T * weights).T

            perturbed_cov = self._robust_covariance(weighted_returns)
            w = self._optimize_weights(expected_returns, perturbed_cov)
            weight_samples.append(w)

        # Average weights across all samples
        final_weights = np.mean(weight_samples, axis=0)

        return final_weights
