"""
Pseudoname: R1D3EP (R1D3EP@proton.me)
Strategy Name: Momentum Risk Parity

This is the strategy that perplexity: Deepseek-r1
generated from a one-shot query with the following prompt:
"Given the description of the competition in 1a_2025_algo_case_packet and the training data and using any external information you can find as help, create an advanced strategy does as good as possible in the competition. Please give the code as a single block."
The model was given access to the pdf version of the case packet, the training data, and the ability to search for external resources.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class OptimalAllocator:
    def __init__(self):
        self.lookback_momentum = 30
        self.lookback_vol = 60
        self.vol_target = 0.15
        self.ewma_alpha = 0.06
        self.hist_prices = []

    def allocate(self, daily_data):
        # Update price history
        self.hist_prices.append(daily_data["close"])
        df = pd.DataFrame(self.hist_prices).ffill().bfill()

        if len(df) < self.lookback_vol + 5:
            return np.ones(26) / 26

        # Calculate returns and volatility
        returns = np.log(df).diff().dropna()
        vol = returns.ewm(alpha=self.ewma_alpha).std().iloc[-1]

        # Momentum signal (exponential weighted moving average)
        momentum = df.pct_change(self.lookback_momentum).iloc[-1]

        # Risk parity weights
        risk_weights = 1 / (vol + 1e-8)
        risk_weights /= risk_weights.sum()

        # Combine momentum and risk parity (ensure non-negative)
        combined_weights = np.maximum(momentum * risk_weights, 0)

        # Volatility targeting optimization
        def portfolio_vol(w):
            cov = returns.cov()
            return np.sqrt(w.T @ cov @ w)

        # Optimization constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum to 1
        ]

        # Solve optimization with non-negative bounds
        res = minimize(
            lambda w: -np.dot(combined_weights, w),  # Maximize momentum exposure
            x0=combined_weights,
            method="SLSQP",
            constraints=constraints,
            bounds=[(0, 1) for _ in range(26)],  # Non-negative bounds
        )

        weights = res.x
        return weights
