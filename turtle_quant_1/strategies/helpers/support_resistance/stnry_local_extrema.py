"""Stationary local extrema static support and resistance strategy."""

import numpy as np
import pandas as pd
import scipy.signal as sp

from turtle_quant_1.strategies.helpers.helpers import round_to_sig_fig

from .base import BaseSupResStrategy


class StnryLocalExtrema(BaseSupResStrategy):
    """Stationary local extrema static support and resistance strategy.

    Refer to https://medium.com/@itay1542/how-to-calculate-support-and-resistance-levels-using-python-a-step-by-step-guide-e94a33c6cbda.

    This strategy identifies support and resistance levels by finding local extrema
    (peaks and troughs) in the price data and ranking them based on their frequency
    and proximity to other extrema.
    """

    def __init__(
        self,
        strong_peak_distance: int = 10,  # TODO: Respect CANDLE_UNIT.
        strong_peak_prominence_pct: float = 0.01,
        peak_distance: int = 5,
        peak_rank_width: float = 2.0,
        res_min_pivot_rank: int = 3,
    ):
        """Initialize the StnryLocalExtrema strategy.

        Args:
            strong_peak_distance: Distance between strong peaks (in days).
            strong_peak_prominence_pct: Prominence for strong peaks (as fraction of price range).
            peak_distance: Distance between general peaks (in days).
            peak_rank_width: Width for grouping nearby peaks.
            res_min_pivot_rank: Minimum rank for resistance levels.
        """
        super().__init__()

        self.sup_res_zone_threshold = 0.01  # Within 2% of the level

        self.strong_peak_distance = strong_peak_distance
        self.strong_peak_prominence_pct = strong_peak_prominence_pct
        self.peak_distance = peak_distance
        self.peak_rank_width = peak_rank_width
        self.res_min_pivot_rank = res_min_pivot_rank

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

    def _calc_sup_levels(self, data: pd.DataFrame) -> list[float]:
        """Calculate support levels from low prices using inverted peak finding."""
        close_prices = data["Close"].values

        # Find troughs by finding peaks in inverted low prices
        troughs, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            -close_prices,  # pyrefly: ignore[unsupported-operation]
            distance=self.peak_distance,
        )

        # Initialize a dictionary to track the rank of each trough
        trough_to_rank = {trough: 0 for trough in troughs}

        # Loop through all troughs to compare their proximity and rank them
        for i, current_trough in enumerate(troughs):
            # pyrefly: ignore[index-error]
            current_low = data.iloc[current_trough]["Close"]

            # Compare the current trough with previous troughs
            for previous_trough in troughs[:i]:
                if (
                    # pyrefly: ignore
                    abs(current_low - data.iloc[previous_trough]["Close"])
                    <= self.peak_rank_width
                ):
                    trough_to_rank[current_trough] += 1

        # Initialize support levels
        supports = []

        # Add troughs that meet the minimum rank threshold
        for trough, rank in trough_to_rank.items():
            if rank >= self.res_min_pivot_rank:
                # pyrefly: ignore[index-error]
                supports.append(data.iloc[trough]["Close"] - 1e-3)

        # Sort and bin nearby support levels
        supports.sort()
        return self._get_binned_levels(supports)

    def _calc_res_levels(self, data: pd.DataFrame) -> list[float]:
        """Calculate resistance levels from high prices."""
        high_prices = data["High"].values
        low_prices = data["Low"].values
        close_prices = data["Close"].values

        # Calculate dynamic prominence based on price range
        # pyrefly: ignore
        price_range = np.max(high_prices) - np.min(low_prices)
        prominence_threshold = price_range * self.strong_peak_prominence_pct

        # Find strong peaks in the 'high' price data
        strong_peaks, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            close_prices,
            distance=self.strong_peak_distance,
            prominence=prominence_threshold,
        )

        # Extract the corresponding high values of the strong peaks
        # pyrefly: ignore[index-error]
        strong_peaks_values = data.iloc[strong_peaks]["Close"].values.tolist()

        # Include the yearly high as an additional strong peak
        if len(data) >= 252:
            yearly_high = data["High"].iloc[-252:].max()
            strong_peaks_values.append(yearly_high)

        # Find general peaks in the stock's 'high' prices
        peaks, _ = sp.find_peaks(  # pyrefly: ignore[missing-attribute]
            close_prices,
            distance=self.peak_distance,
        )

        # Initialize a dictionary to track the rank of each peak
        peak_to_rank = {peak: 0 for peak in peaks}

        # Loop through all general peaks to compare their proximity and rank them
        for i, current_peak in enumerate(peaks):
            # pyrefly: ignore[index-error]
            current_high = data.iloc[current_peak]["Close"]

            # Compare the current peak with previous peaks to calculate rank based on proximity
            for previous_peak in peaks[:i]:
                if (
                    # pyrefly: ignore[index-error]
                    abs(current_high - data.iloc[previous_peak]["Close"])
                    <= self.peak_rank_width
                ):
                    peak_to_rank[current_peak] += 1

        # Initialize the list of resistance levels with the strong peaks
        resistances = strong_peaks_values.copy()

        # Add general peaks that meet the minimum rank threshold
        for peak, rank in peak_to_rank.items():
            if rank >= self.res_min_pivot_rank:
                # pyrefly: ignore[index-error]
                resistances.append(data.iloc[peak]["Close"] + 1e-3)

        # Sort and bin nearby resistance levels
        resistances.sort()
        return self._get_binned_levels(resistances)

    def _calc_sup_res_levels(
        self,
        data: pd.DataFrame,
        lookback: int = 14,  # NOTE: This is a magic number. Lookback is agnostic to CANDLE_UNIT.
    ) -> list[np.ndarray]:
        level_values = [np.full(128, 0.0) for _ in range(len(data))]
        # TODO: Vectorize.
        for i in range(lookback, len(data)):
            i_start = i - lookback
            # Calculate resistance levels
            ress = self._calc_res_levels(data.iloc[i_start:i])
            # Calculate support levels using the same logic on inverted low prices
            sups = self._calc_sup_levels(data.iloc[i_start:i])
            # Combine all levels
            levels = ress + sups
            levels.sort()
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
        """Generate all support and resistance levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with support and resistance levels.
            The columns are: ['datetime', 'level_values'].
        """
        # Ensure we have the required columns
        if not all(col in data.columns for col in ["High", "Low"]):
            raise ValueError("Data must contain 'High' and 'Low' columns")

        level_values = self._calc_sup_res_levels(data)

        # Create output DataFrame with 1-to-1 mapping to original data
        result = pd.DataFrame(
            {
                "datetime": pd.to_datetime(data["datetime"]).reset_index(drop=True),
                "level_values": level_values,
            }
        )

        return result
