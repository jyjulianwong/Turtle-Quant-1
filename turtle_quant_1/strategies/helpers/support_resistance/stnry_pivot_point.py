"""Stationary pivot point support and resistance strategy."""

import pandas as pd

from turtle_quant_1.strategies.helpers.helpers import convert_to_weekly_data

from .base import BaseSupResStrategy


class StnryPivotPoint(BaseSupResStrategy):
    """Stationary pivot point support and resistance strategy.

    This strategy calculates monthly pivot points and their associated
    support (S1, S2) and resistance (R1, R2) levels based on the previous
    month's high, low, and close prices.

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

    def _get_monthly_dfs(self, data: pd.DataFrame) -> pd.api.typing.DataFrameGroupBy:
        """Get the monthly data for a given DataFrame.

        Args:
            data: DataFrame with OHLC data.

        Returns:
            DataFrame with monthly data.
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

        # Group by month to calculate monthly levels
        monthly_groups = data.groupby(pd.Grouper(freq="ME"))
        return monthly_groups

    def _calc_pivot_levels(self, month_data: pd.DataFrame) -> list[float]:
        """Calculate PP, S1, S2, R1, R2 levels for a month's data.

        Args:
            month_data: DataFrame with OHLC data for a single month.

        Returns:
            List containing [PP, S1, S2, R1, R2] values.
        """
        # Get the high, low, and close for the month
        high = month_data["High"].max()
        low = month_data["Low"].min()
        close = month_data["Close"].iloc[-1]  # Last close of the month

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
        """Generate monthly pivot point levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with monthly pivot point levels.
            The columns are: ['datetime_start', 'datetime_end', 'values'].
            The 'values' column contains [PP, S1, S2, R1, R2] for each month.
        """
        # Ensure we have the required columns
        required_cols = ["High", "Low", "Close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain {required_cols} columns")

        daily_data = convert_to_weekly_data(data)
        # Group by month to calculate monthly levels
        monthly_groups = self._get_monthly_dfs(daily_data)

        results = []

        for month_end, month_data in monthly_groups:
            if len(month_data) == 0:
                continue

            # Calculate pivot point levels for this month
            levels = self._calc_pivot_levels(month_data)

            # Get month start and end dates
            month_start = month_data.index[0]
            month_end_actual = month_data.index[-1]

            results.append(
                {
                    "datetime_start": month_start,
                    "datetime_end": month_end_actual,
                    "values": levels,
                }
            )

        return pd.DataFrame(results)
