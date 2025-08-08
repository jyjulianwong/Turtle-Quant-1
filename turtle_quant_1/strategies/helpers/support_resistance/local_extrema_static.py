"""Local extrema static support and resistance strategy."""

import pandas as pd
import numpy as np
import scipy.signal as sp
from turtle_quant_1.strategies.helpers.support_resistance.base import BaseSupResStrategy
from turtle_quant_1.strategies.helpers.helpers import convert_to_daily_data


class LocalExtremaStatic(BaseSupResStrategy):
    """Local extrema static support and resistance strategy.

    This strategy identifies support and resistance levels by finding local extrema
    (peaks and troughs) in the price data and ranking them based on their frequency
    and proximity to other extrema.
    """

    def __init__(
        self,
        strong_peak_distance: int = 60,
        strong_peak_prominence: int = 20,
        peak_distance: int = 5,
        peak_rank_width: float = 2.0,
        resistance_min_pivot_rank: int = 3,
    ):
        """Initialize the LocalExtremaStatic strategy.

        Args:
            strong_peak_distance: Distance between strong peaks (in days).
            strong_peak_prominence: Prominence for strong peaks.
            peak_distance: Distance between general peaks (in days).
            peak_rank_width: Width for grouping nearby peaks.
            resistance_min_pivot_rank: Minimum rank for resistance levels.
        """
        super().__init__()
        self.strong_peak_distance = strong_peak_distance
        self.strong_peak_prominence = strong_peak_prominence
        self.peak_distance = peak_distance
        self.peak_rank_width = peak_rank_width
        self.resistance_min_pivot_rank = resistance_min_pivot_rank

    def _calc_support_levels(self, data: pd.DataFrame) -> list[float]:
        """Calculate support levels from low prices using inverted peak finding."""
        low_prices = data["Low"].values

        # Find troughs by finding peaks in inverted low prices
        troughs, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            -low_prices,  # pyrefly: ignore[unsupported-operation]
            distance=self.peak_distance,
        )

        # Initialize a dictionary to track the rank of each trough
        trough_to_rank = {trough: 0 for trough in troughs}

        # Loop through all troughs to compare their proximity and rank them
        for i, current_trough in enumerate(troughs):
            # pyrefly: ignore[index-error]
            current_low = data.iloc[current_trough]["Low"]

            # Compare the current trough with previous troughs
            for previous_trough in troughs[:i]:
                if (
                    # pyrefly: ignore
                    abs(current_low - data.iloc[previous_trough]["Low"])
                    <= self.peak_rank_width
                ):
                    trough_to_rank[current_trough] += 1

        # Initialize support levels
        supports = []

        # Add troughs that meet the minimum rank threshold
        for trough, rank in trough_to_rank.items():
            if rank >= self.resistance_min_pivot_rank:
                # pyrefly: ignore[index-error]
                supports.append(data.iloc[trough]["Low"] - 1e-3)

        # Sort and bin nearby support levels
        supports.sort()
        return self._get_binned_levels(supports)

    def _calc_resistance_levels(self, data: pd.DataFrame) -> list[float]:
        """Calculate resistance levels from high prices."""
        high_prices = data["High"].values

        # Find strong peaks in the 'high' price data
        strong_peaks, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            high_prices,
            distance=self.strong_peak_distance,
            prominence=self.strong_peak_prominence,
        )

        # Extract the corresponding high values of the strong peaks
        # pyrefly: ignore[index-error]
        strong_peaks_values = data.iloc[strong_peaks]["High"].values.tolist()

        # Include the yearly high as an additional strong peak
        if len(data) >= 252:
            yearly_high = data["High"].iloc[-252:].max()
            strong_peaks_values.append(yearly_high)

        # Find general peaks in the stock's 'high' prices
        peaks, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            high_prices,
            distance=self.peak_distance,
        )

        # Initialize a dictionary to track the rank of each peak
        peak_to_rank = {peak: 0 for peak in peaks}

        # Loop through all general peaks to compare their proximity and rank them
        for i, current_peak in enumerate(peaks):
            # pyrefly: ignore[index-error]
            current_high = data.iloc[current_peak]["High"]

            # Compare the current peak with previous peaks to calculate rank based on proximity
            for previous_peak in peaks[:i]:
                if (
                    # pyrefly: ignore[index-error]
                    abs(current_high - data.iloc[previous_peak]["High"])
                    <= self.peak_rank_width
                ):
                    peak_to_rank[current_peak] += 1

        # Initialize the list of resistance levels with the strong peaks
        resistances = strong_peaks_values.copy()

        # Add general peaks that meet the minimum rank threshold
        for peak, rank in peak_to_rank.items():
            if rank >= self.resistance_min_pivot_rank:
                # pyrefly: ignore[index-error]
                resistances.append(data.iloc[peak]["High"] + 1e-3)

        # Sort and bin nearby resistance levels
        resistances.sort()
        return self._get_binned_levels(resistances)

    def _get_binned_levels(self, levels: list[float]) -> list[float]:
        """Bin nearby levels together and return their averages."""
        if not levels:
            return []

        # Initialize bins
        level_bins = []
        current_bin = [levels[0]]

        # Loop through sorted levels and bin them
        for level in levels[1:]:
            if level - current_bin[-1] < self.peak_rank_width:
                current_bin.append(level)
            else:
                level_bins.append(current_bin)
                current_bin = [level]

        # Append the last bin
        level_bins.append(current_bin)

        # Calculate average for each bin
        return [float(np.mean(bin_levels)) for bin_levels in level_bins]

    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate all support and resistance levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with one row containing all support and resistance levels.
        """
        # Ensure we have the required columns
        if not all(col in data.columns for col in ["High", "Low"]):
            raise ValueError("Data must contain 'High' and 'Low' columns")

        daily_data = convert_to_daily_data(data)

        # Calculate resistance levels
        resistances = self._calc_resistance_levels(daily_data)

        # Calculate support levels using the same logic on inverted low prices
        supports = self._calc_support_levels(daily_data)

        # Combine all levels
        all_levels = resistances + supports
        all_levels.sort()

        # Create output DataFrame - single row since these are static levels
        result = pd.DataFrame(
            {
                "datetime_start": [
                    data["datetime"].iloc[0]
                    if "datetime" in data.columns
                    else data.index[0]
                ],
                "datetime_end": [
                    data["datetime"].iloc[-1]
                    if "datetime" in data.columns
                    else data.index[-1]
                ],
                "values": [all_levels],
            }
        )

        return result
