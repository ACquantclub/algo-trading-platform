import numpy as np
import pandas as pd
import scipy

#########################################################################
## Implement your portfolio allocation strategy as a class with an      ##
## allocate method that takes in market data for one day and returns   ##
## portfolio weights.                                                  ##
#########################################################################


class PortfolioStrategy:
    def __init__(self):
        """
        Initialize any strategy parameters here
        """
        pass

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
        # Access different data types
        opens = market_data["open"]
        closes = market_data["close"]
        highs = market_data["high"]
        lows = market_data["low"]
        volumes = market_data["volume"]

        # This example strategy equally weights all assets every period
        n_assets = len(market_data["close"])
        weights = np.ones(n_assets) / n_assets
        return weights
