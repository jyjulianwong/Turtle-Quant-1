"""Stationary pivot point support and resistance strategy."""

import pandas as pd

from turtle_quant_1.strategies.helpers.helpers import convert_to_weekly_data

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

    def _get_quarterly_dfs(self, data: pd.DataFrame) -> pd.api.typing.DataFrameGroupBy:
        """Get the quarterly data for a given DataFrame.

        Args:
            data: DataFrame with OHLC data.

        Returns:
            DataFrame GroupBy with quarterly data.
        """
        # Ensure datetime column exists or use index
        if "datetime" in data.columns:
            data = data.copy()
            data["datetime"] = pd.to_datetime(data["datetime"])
            data.set_index("datetime", inplace=True)
        else:
            data = data.copy()
            # pyrefly: ignore
            data.index = pd.to_datetime(data.index)

        # Group by quarter to calculate quarterly levels
        quarterly_groups = data.groupby(pd.Grouper(freq="3ME"))
        return quarterly_groups

    def _calc_pivot_levels(self, quarter_data: pd.DataFrame) -> list[float]:
        """Calculate PP, S1, S2, R1, R2 levels for a quarter's data.

        Args:
            quarter_data: DataFrame with OHLC data for a single quarter.

        Returns:
            List containing [PP, S1, S2, R1, R2] values.
        """
        # Get the high, low, and close for the quarter
        high = quarter_data["High"].max()
        low = quarter_data["Low"].min()
        close = quarter_data["Close"].iloc[-1]  # Last close of the quarter

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

    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate quarterly pivot point levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with quarterly pivot point levels.
            The columns are: ['datetime_beg', 'datetime_end', 'level_values'].
            The 'level_values' column contains [PP, S1, S2, R1, R2] for each quarter.
        """
        # Ensure we have the required columns
        required_cols = ["High", "Low", "Close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain {required_cols} columns")

        weekly_data = convert_to_weekly_data(data)
        # Group by quarter to calculate quarterly levels
        quarterly_groups = self._get_quarterly_dfs(weekly_data)

        results = []

        for quarter_end, quarter_data in quarterly_groups:
            if len(quarter_data) == 0:
                continue

            # Calculate pivot point levels for this quarter
            level_values = self._calc_pivot_levels(quarter_data)

            # Get quarter start and end dates
            quarter_start_actual = quarter_data.index[0]
            quarter_end_actual = quarter_data.index[-1]

            quarter_start_buffered = max(
                quarter_start_actual - pd.Timedelta(days=30),
                data.iloc[0]["datetime"],
            )
            quarter_end_buffered = min(
                quarter_end_actual + pd.Timedelta(days=30),
                data.iloc[-1]["datetime"],
            )

            results.append(
                {
                    "datetime_beg": quarter_start_buffered,
                    "datetime_end": quarter_end_buffered,
                    "level_values": level_values,
                }
            )

        return pd.DataFrame(results)
