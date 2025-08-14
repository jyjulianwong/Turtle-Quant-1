"""Stationary pivot point support and resistance strategy."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from turtle_quant_1.strategies.helpers.helpers import round_to_sig_fig

from .base import BaseSupResStrategy


class StnryPivotPoint(BaseSupResStrategy):
    """Stationary pivot point support and resistance strategy.

    This strategy calculates quarterly pivot points and their associated
    support (S1, S2) and resistance (R1, R2) levels based on the previous
    quarter's high, low, and close prices.

    Formulas:
    - PP (Pivot Point) = (High + Low + Close) / 3
    - R1 = 2 * PP - Low
    - R2 = PP + (High - Low)
    - S1 = 2 * PP - High
    - S2 = PP - (High - Low)
    """

    def __init__(self):
        """Initialize the StnryPivotPoint strategy."""
        super().__init__()

        self.sup_res_zone_threshold = 0.01  # Within 2% of the level

    def _calc_pivot_levels(self, data: pd.DataFrame) -> list[float]:
        """Calculate PP, S1, S2, R1, R2 levels for a quarter's data.

        Args:
            quarter_data: DataFrame with OHLC data for a single quarter.

        Returns:
            List containing [PP, S1, S2, R1, R2] values.
        """
        # Get the high, low, and close for the quarter
        high = data["High"].max()
        low = data["Low"].min()
        close = data["Close"].iloc[-1]  # Last close of the quarter

        # Calculate pivot point
        pp = (high + low + close) / 3

        # Calculate support levels
        s1 = 2 * pp - high
        s2 = pp - (high - low)

        # Calculate resistance levels
        r1 = 2 * pp - low
        r2 = pp + (high - low)

        # Return in order: PP, S1, S2, R1, R2
        return [pp, s1, s2, r1, r2]

    def _calc_sup_res_levels(
        self,
        data: pd.DataFrame,
        lookback: int = 12,  # TODO: Respect CANDLE_UNIT.
    ) -> list[np.ndarray]:
        level_values = [np.full(128, 0.0) for _ in range(len(data))]
        # TODO: Vectorize.
        for i in range(lookback, len(data)):
            i_start = i - lookback
            levels = self._calc_pivot_levels(data.iloc[i_start:i])
            # Round values to reduce number of unique values
            # This is needed for the MultiLabelBinarizer to work later on
            rounded_levels = round_to_sig_fig(levels, 4)
            # Create fixed-size array with padding
            fixed_array = np.full(128, 0.0)
            fixed_array[: len(rounded_levels)] = rounded_levels
            level_values[i] = fixed_array

        return level_values

    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate quarterly pivot point levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with pivot point levels.
            The columns are: ['datetime', 'level_values'].
            The 'level_values' column contains [PP, S1, S2, R1, R2] for each timestamp.
        """
        # Ensure we have the required columns
        required_cols = ["High", "Low", "Close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain {required_cols} columns")

        level_values = self._calc_sup_res_levels(data)

        # Create output DataFrame with 1-to-1 mapping to original data
        result = pd.DataFrame(
            {
                "datetime": pd.to_datetime(data["datetime"]).reset_index(drop=True),
                "level_values": level_values,
            }
        )

        return result
