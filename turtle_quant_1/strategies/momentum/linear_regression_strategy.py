"""Simple linear regression strategy implementation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from turtle_quant_1.strategies.base import BaseStrategy


class LinearRegressionStrategy(BaseStrategy):
    """A naive, simple linear regression strategy that uses linear regression to determine trend direction.

    The strategy fits a linear regression line to recent closing prices and uses the slope
    to determine trend direction:
    - Positive slope (upward trend) -> positive score (buy signal)
    - Negative slope (downward trend) -> negative score (sell signal)
    - Slope near zero -> neutral score (hold signal)
    """

    def __init__(
        self, name: str = "LinearRegressionStrategy", lookback_candles: int = 120
    ):
        """Initialize the linear regression strategy.

        Args:
            lookback_candles: Number of recent periods to use for trend calculation.
            name: Name of the strategy.
        """
        super().__init__(name)
        self.lookback_candles = lookback_candles

    def generate_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a trading score based on linear regression analysis.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score between -1.0 (strong sell) and +1.0 (strong buy).
        """
        # Validate input data
        self.validate_data(data)

        # Ensure data is sorted by datetime
        data_sorted = data.sort_values("datetime").copy()

        # Use the specified number of recent periods, or all available data if less
        periods_to_use = min(self.lookback_candles, len(data_sorted))
        recent_data = data_sorted.tail(periods_to_use).copy()

        if len(recent_data) < 2:
            return 0.0  # Not enough data, return neutral

        # Create time index for regression (0, 1, 2, ...)
        recent_data = recent_data.reset_index(drop=True)
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data["Close"].values

        # Fit linear regression to get trend slope
        reg = LinearRegression()
        reg.fit(X, y)
        slope = reg.coef_[0]

        # Calculate relative slope (normalize by average price to make it scale-invariant)
        mean_price = np.mean(y)  # pyrefly: ignore[no-matching-overload]
        if mean_price == 0:
            return 0.0

        relative_slope = slope / mean_price

        # Scale the relative slope to generate score between -1 and +1
        # We use a sigmoid-like function to map slope to score
        # Adjust the scaling factor based on typical price movements
        scaling_factor = 1000.0  # This can be tuned based on the asset volatility

        # Use tanh to map to [-1, 1] range smoothly
        score = np.tanh(relative_slope * scaling_factor)

        # Ensure score is within bounds
        return max(-1.0, min(1.0, float(score)))
