"""Stationary Fibonacci retracement support and resistance strategy."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.signal as sp

from turtle_quant_1.strategies.helpers.helpers import (
    convert_to_weekly_data,
    round_to_sig_fig,
)

from .base import BaseSupResStrategy


class StnryFibonacciRetrace(BaseSupResStrategy):
    """Stationary Fibonacci retracement support and resistance strategy.

    This strategy identifies support and resistance levels by calculating
    Fibonacci retracement levels between significant swing highs and lows
    across the entire dataset. It finds the most significant swing points
    and computes the classic Fibonacci ratios: 23.6%, 38.2%, 50%, 61.8%, and 78.6%.
    """

    def __init__(
        self,
        peak_distance: int = 10,  # NOTE: This is a magic number. Depends on resampling.
        peak_prominence_pct: float = 0.02,
        fib_levels: List[float] = [],
    ):
        """Initialize the StnryFibonacciRetrace strategy.

        Args:
            peak_distance: Minimum distance between peaks (in candlesticks).
            peak_prominence_pct: Minimum prominence for peaks (as fraction of price range).
            fib_levels: List of Fibonacci ratios to use. Defaults to standard levels.
        """
        super().__init__()
        self.peak_distance = peak_distance
        self.peak_prominence_pct = peak_prominence_pct

        # Standard Fibonacci retracement levels
        if not fib_levels:
            self.fib_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        else:
            self.fib_levels = fib_levels

    def _find_swing_highs_lows(
        self, data: pd.DataFrame
    ) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows using scipy.signal.find_peaks.

        Args:
            data: DataFrame with OHLC data.

        Returns:
            List of tuples (index, price, type) where type is 'high' or 'low'.
        """
        swings = []
        high_prices = data["High"].values
        low_prices = data["Low"].values
        close_prices = data["Close"].values

        # Calculate dynamic prominence based on price range
        # pyrefly: ignore
        price_range = np.max(high_prices) - np.min(low_prices)
        prominence_threshold = price_range * self.peak_prominence_pct

        # Find swing highs using find_peaks
        peaks, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            close_prices,
            distance=self.peak_distance,
            prominence=prominence_threshold,
        )

        # Add swing highs to results
        for peak_idx in peaks:
            swings.append((peak_idx, close_prices[peak_idx], "high"))

        # Find swing lows by finding peaks in inverted low prices
        troughs, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            # Invert to find troughs as peaks
            -close_prices,  # pyrefly: ignore[unsupported-operation]
            distance=self.peak_distance,
            prominence=prominence_threshold,
        )

        # Add swing lows to results
        for peak_idx in troughs:
            swings.append((peak_idx, close_prices[peak_idx], "low"))

        # Sort by index to maintain chronological order
        swings.sort(key=lambda x: x[0])
        return swings

    def _calc_fibonacci_levels_for_swings(
        self, high_price: float, low_price: float
    ) -> List[float]:
        """Calculate Fibonacci retracement levels between high and low.

        Args:
            high_price: The swing high price.
            low_price: The swing low price.

        Returns:
            List of Fibonacci retracement levels.
        """
        price_range = high_price - low_price
        fib_levels = []

        for ratio in self.fib_levels:
            # Calculate retracement from high (for downtrend)
            retracement_from_high = high_price - (price_range * ratio)
            fib_levels.append(retracement_from_high)

            # Calculate retracement from low (for uptrend)
            retracement_from_low = low_price + (price_range * ratio)
            if retracement_from_low != retracement_from_high:
                fib_levels.append(retracement_from_low)

        # Add the swing high and low themselves as significant levels
        fib_levels.extend([high_price, low_price])

        # Remove duplicates and sort
        fib_levels = list(set(fib_levels))
        fib_levels.sort()

        return fib_levels

    def _calc_fibonacci_levels_for_candle(
        self, data: pd.DataFrame, idx: int
    ) -> list[float]:
        # Find all swing highs and lows for the entire dataset
        all_swings = self._find_swing_highs_lows(data.iloc[:idx])

        if len(all_swings) < 2:
            # Need at least some swings to calculate Fibonacci levels
            return []

        all_levels = []

        # Get the most significant swings across the entire dataset
        highs = [swing for swing in all_swings if swing[2] == "high"]
        lows = [swing for swing in all_swings if swing[2] == "low"]

        if highs and lows:
            # Find the absolute highest high and lowest low
            highest_swing = max(highs, key=lambda x: x[1])
            lowest_swing = min(lows, key=lambda x: x[1])

            # Calculate Fibonacci levels between the absolute extremes
            primary_fib_levels = self._calc_fibonacci_levels_for_swings(
                highest_swing[1], lowest_swing[1]
            )
            all_levels.extend(primary_fib_levels)

            # Also calculate levels between major swing pairs to capture intermediate levels
            # Sort swings by significance (price level)
            significant_highs = sorted(highs, key=lambda x: x[1], reverse=True)[:3]
            significant_lows = sorted(lows, key=lambda x: x[1])[:3]

            # Calculate Fibonacci levels between significant swing pairs
            for high_swing in significant_highs:
                for low_swing in significant_lows:
                    fib_levels = self._calc_fibonacci_levels_for_swings(
                        high_swing[1], low_swing[1]
                    )
                    all_levels.extend(fib_levels)

        if not all_levels:
            return []

        # Remove duplicates and sort
        all_levels = list(set(all_levels))
        all_levels.sort()

        return all_levels

    def _calc_sup_res_levels(
        self,
        data: pd.DataFrame,
        lookback: int = 14,  # NOTE: This is a magic number. Depends on resampling.
    ) -> list[np.ndarray]:
        level_values = [np.full(128, 0.0) for _ in range(len(data))]
        # TODO: Vectorize.
        for i in range(lookback, len(data)):
            i_start = i - lookback
            levels = self._calc_fibonacci_levels_for_candle(data.iloc[i_start:i], i)
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
        """Generate Fibonacci retracement levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with Fibonacci retracement levels.
            The columns are: ['datetime', 'level_values'].
        """
        # Ensure we have the required columns
        required_cols = ["High", "Low"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain {required_cols} columns")

        data_resampled = convert_to_weekly_data(data)
        level_values = self._calc_sup_res_levels(data_resampled)

        # Create output DataFrame with 1-to-1 mapping to original data
        result = pd.DataFrame(
            {
                "datetime": pd.to_datetime(data_resampled["datetime"]).reset_index(
                    drop=True
                ),
                "level_values": level_values,
            }
        )

        result = data[["datetime"]].merge(result, on="datetime", how="left")
        result["level_values"] = result["level_values"].bfill().ffill()

        return result
