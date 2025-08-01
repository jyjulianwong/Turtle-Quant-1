"""Simple linear regression strategy implementation."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from turtle_quant_1.strategies.base import BaseStrategy


class LinearRegression(BaseStrategy):
    """A naive, simple linear regression strategy that uses linear regression to determine trend direction.

    The strategy fits a linear regression line to recent closing prices and uses the slope
    to determine trend direction:
    - Positive slope (upward trend) -> positive score (buy signal)
    - Negative slope (downward trend) -> negative score (sell signal)
    - Slope near zero -> neutral score (hold signal)
    """

    def __init__(self, lookback_candles: int = 120):
        """Initialize the linear regression strategy.

        Args:
            lookback_candles: Number of recent periods to use for trend calculation.
        """
        super().__init__()
        self.lookback_candles = lookback_candles

    def _get_coefficients(self, data: pd.DataFrame, symbol: str) -> Tuple[float, float]:
        """Get the coefficients of the linear regression model.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Tuple of (slope, intercept)
        """

        self.validate_data(data)

        data_sorted = data.sort_values("datetime").copy()

        # Use the specified number of recent periods, or all available data if less
        periods_to_use = min(self.lookback_candles, len(data_sorted))
        recent_data = data_sorted.tail(periods_to_use).copy()

        if len(recent_data) < 2:
            return 0.0, 0.0  # Not enough data, return neutral

        # Create time index for regression (0, 1, 2, ...)
        recent_data = recent_data.reset_index(drop=True)
        X = np.arange(len(recent_data)).reshape(-1, 1)
        y = recent_data["Close"].values

        # Fit linear regression to get trend slope
        reg = SklearnLinearRegression()
        reg.fit(X, y)
        slope = reg.coef_[0]

        # Calculate relative slope (normalize by average price to make it scale-invariant)
        mean_price = np.mean(y)  # pyrefly: ignore[no-matching-overload]
        if mean_price == 0:
            return 0.0, 0.0

        return slope, mean_price

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a historical score array for a symbol based on market data.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with each value between -1.0 and +1.0, indexed by datetime
        """
        slope, mean_price = self._get_coefficients(data, symbol)

        data_sorted = data.sort_values("datetime").copy()

        relative_slope = slope / mean_price

        # Scale the relative slope to generate score between -1 and +1
        # We use a sigmoid-like function to map slope to score
        # Adjust the scaling factor based on typical price movements
        scaling_factor = 1000.0  # This can be tuned based on the asset volatility

        # Use tanh to map to [-1, 1] range smoothly
        score = np.tanh(relative_slope * scaling_factor)

        # Calculate trend line values
        x = np.arange(len(data_sorted))
        y = score * x

        return pd.Series(
            data=y,  # TODO: This is not correct.
            index=pd.to_datetime(data_sorted["datetime"]),
        ).fillna(0)

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a trading score based on linear regression analysis.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score between -1.0 (strong sell) and +1.0 (strong buy).
        """
        slope, mean_price = self._get_coefficients(data, symbol)

        relative_slope = slope / mean_price

        # Scale the relative slope to generate score between -1 and +1
        # We use a sigmoid-like function to map slope to score
        # Adjust the scaling factor based on typical price movements
        scaling_factor = 1000.0  # This can be tuned based on the asset volatility

        # Use tanh to map to [-1, 1] range smoothly
        score = np.tanh(relative_slope * scaling_factor)

        # Ensure score is within bounds
        return max(-1.0, min(1.0, float(score)))
