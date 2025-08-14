"""Base class for support and resistance strategies."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from turtle_quant_1.strategies.helpers.multiprocessing import ProcessSafeCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global SupResIndicator instance
_global_sup_res_indicator = ProcessSafeCache()

# Global cache instance
_global_cache = ProcessSafeCache()


def get_global_cache():
    """Get the global cache instance."""
    return _global_cache


class SupResIndicator:
    """Entrypoint for determining support and resistance zones for other strategies."""

    @classmethod
    def get_global_instance(cls) -> "SupResIndicator":
        """Get the global SupResIndicator instance.

        To reduce duplicated calculations, the SupResIndicator can be instantiated as a global singleton.
        NOTE: As of now, doing this has little benefit as the SupResIndicator cache is also a global singleton.
        """
        if _global_sup_res_indicator.get("sup_res_indicator") is None:
            # Initialize the global SupResIndicator instance
            _global_sup_res_indicator.set("sup_res_indicator", SupResIndicator())

        # pyrefly: ignore
        return _global_sup_res_indicator.get("sup_res_indicator")

    @classmethod
    def preload_global_instance_cache(cls, symbol: str, data: pd.DataFrame) -> None:
        """Preload the global SupResIndicator instance with data for a symbol.

        This reduces duplicated calculations across all SupResIndicator instances,
        because the cache is a global singleton.
        The calculations are done at the point of loading.
        See the implementation of `_load_sup_res_levels_data` for more details.

        Args:
            symbol: The symbol to preload the data for.
            data: The data to preload the data for.
        """
        for strategy in cls.get_global_instance().strategies:
            cls.get_global_instance()._load_sup_res_levels_data(
                strategy=strategy, symbol=symbol, data=data
            )
            logger.info(
                f"Preloaded {strategy.__class__.__name__} level data for {symbol}"
            )

    def __init__(self, strategies: list["BaseSupResStrategy"] = []):
        """Initialize the SupResIndicator with a list of strategies.

        Args:
            strategies: List of support/resistance strategies to use for consensus.
                If None, will use default strategies.
        """
        if not strategies:
            # Include all strategies by default
            # Import here to avoid circular imports
            from .stnry_fibonacci_retrace import (
                StnryFibonacciRetrace,
            )
            from .stnry_gaussian_kde import (
                StnryGaussianKde,
            )
            from .stnry_local_extrema import (
                StnryLocalExtrema,
            )
            from .stnry_pivot_point import (
                StnryPivotPoint,
            )

            self.strategies = [
                StnryFibonacciRetrace(),
                StnryGaussianKde(),
                StnryLocalExtrema(),
                StnryPivotPoint(),
            ]  # pyrefly: ignore[bad-assignment]
        else:
            self.strategies = strategies  # pyrefly: ignore[bad-assignment]

    def _load_sup_res_levels_data(
        self, strategy: "BaseSupResStrategy", symbol: str, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the support and resistance data for a symbol.

        Args:
            strategy: The strategy to get the support and resistance data for.
            symbol: The symbol to get the support and resistance data for.
            data: The data to get the support and resistance data for.
        """
        cache_key = f"{symbol}_{strategy.__class__.__name__}"
        cache = get_global_cache()

        # Check if data exists in cache
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            logger.debug(
                f"GLOBAL_SUP_RES_DATA_CACHE for {cache_key} already exists. Using cached data."
            )
            return cached_data

        # Generate new data and cache it
        logger.debug(
            f"GLOBAL_SUP_RES_DATA_CACHE for {cache_key} does not exist. Generating data..."
        )
        levels_data = strategy.generate_historical_levels(data, symbol)
        cache.set(cache_key, levels_data)
        return levels_data

    def _is_price_in_sup_res_zone(
        self,
        historical_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        levels_df: pd.DataFrame,
        threshold: float,
    ) -> bool:
        """Check if current price is near any support/resistance levels.

        Args:
            historical_data: Historical data up to current point.
            timestamp: Current timestamp in the data.
            levels_df: DataFrame with support/resistance levels.
            threshold: Price proximity threshold (as fraction of price).

        Returns:
            True if price is near any level, False otherwise.
        """
        if len(levels_df) == 0:
            return False

        # Find the row that corresponds to the current timestamp
        levels_df_copy = levels_df.copy()
        levels_df_copy["datetime"] = pd.to_datetime(levels_df_copy["datetime"])

        # Filter levels that match the current timestamp (or closest available)
        time_mask = levels_df_copy["datetime"] <= timestamp
        if not time_mask.any():
            return False

        # Get the most recent levels up to the current timestamp
        relevant_levels_df = levels_df_copy[time_mask]
        latest_row = relevant_levels_df.iloc[-1]  # Get the most recent row

        # Extract level values from the latest row
        level_values = latest_row["level_values"]

        # Handle case where level_values might be empty or None
        if level_values is None:
            return False
        if isinstance(level_values, (list, tuple)) and len(level_values) == 0:
            return False
        if hasattr(level_values, "size") and level_values.size == 0:
            return False

        # Handle NumPy array format (new fixed-size arrays)
        if isinstance(level_values, np.ndarray):
            # Filter out NaN values from the fixed-size array
            relevant_levels_array = level_values[~np.isnan(level_values)]
        elif isinstance(level_values, list):
            # Legacy list format support
            filtered_values = [
                float(x) for x in level_values if x is not None and not pd.isna(x)
            ]
            if len(filtered_values) == 0:
                return False
            relevant_levels_array = np.array(filtered_values)
        else:
            if pd.isna(level_values):
                return False
            relevant_levels_array = np.array([float(level_values)])

        if len(relevant_levels_array) == 0:
            return False

        # Vectorized distance calculation and threshold check
        price = historical_data[historical_data["datetime"] == timestamp]["Close"].iloc[
            0
        ]
        price_threshold = price * threshold
        distances = np.abs(relevant_levels_array - price)
        # pyrefly: ignore
        return np.any(distances <= price_threshold)

    def _is_price_in_sup_res_zone_vectorized(
        self,
        data: pd.DataFrame,
        levels_df: pd.DataFrame,
        threshold: float,
    ) -> pd.Series:
        """Vectorized version of _is_price_in_sup_res_zone.

        Args:
            data: Historical data containing 'datetime' and 'Close' columns.
            levels_df: DataFrame with support/resistance levels containing 'datetime' and 'level_values'.
            threshold: Price proximity threshold (as fraction of price).

        Returns:
            pd.Series of bool values indicating if each timestamp is in a support/resistance zone.
        """
        if len(levels_df) == 0:
            return pd.Series([False] * len(data), index=data.index)

        _levels_df = levels_df.copy()
        _levels_df.set_index("datetime", inplace=True)
        _levels_df = _levels_df.loc[data["datetime"]]

        # This cumsum must be clarified — assuming you mean per-element cumulative sum across rows,
        # you may need to rethink this if 'level_values' is an array already.
        _levels_df["level_values"] = _levels_df["level_values"].cumsum()

        # Join prices
        _levels_df = _levels_df.join(data[["datetime", "Close"]].set_index("datetime"))
        _levels_df.rename(columns={"Close": "price"}, inplace=True)

        # Stack directly since all are fixed-size arrays
        # pyrefly: ignore
        padded_levels = np.stack(
            _levels_df["level_values"].to_numpy()
        )  # shape: (n_rows, n_levels)

        # Compute nearest level
        prices = _levels_df["price"].to_numpy()
        diffs = np.abs(padded_levels - prices[:, None])
        nearest_idxs = np.argmin(diffs, axis=1)
        nearest_vals = padded_levels[np.arange(len(_levels_df)), nearest_idxs]

        # Distance & zone check
        nearest_distances = nearest_vals - prices
        is_in_zone = np.abs(nearest_distances) <= (nearest_vals * threshold)

        # Assign back if needed
        _levels_df["nearest_level_value"] = nearest_vals
        _levels_df["nearest_level_distance"] = nearest_distances
        _levels_df["is_in_zone"] = is_in_zone

        return _levels_df["is_in_zone"]

    def is_sup_res_zone(
        self,
        data: pd.DataFrame,
        timestamp: pd.Timestamp,
        symbol: str,
        min_consensus: float = 0.5,
    ) -> bool:
        """Check if the current price is in a support or resistance zone.

        Args:
            data: The data to check, must contain 'High', 'Low', 'Close' columns.
            timestamp: The timestamp of the current price to check.
            symbol: The symbol being analyzed.
            min_consensus: Minimum fraction of strategies that must agree (0.0 to 1.0).

        Returns:
            True if the current price is in a support or resistance zone, False otherwise.
        """
        if (
            timestamp > data["datetime"].iloc[-1]
            or timestamp < data["datetime"].iloc[0]
        ):
            raise ValueError(
                f"Timestamp {timestamp} is out of bounds for data of length {len(data)}"
            )

        n_strategies = len(self.strategies)
        if n_strategies == 0:
            raise ValueError("No strategies provided")

        # Early exit if minimum consensus cannot be met
        n_required_agreements = int(np.ceil(min_consensus * n_strategies))
        if n_required_agreements == 0:
            return True  # If min_consensus is very low, always return True

        # Get data up to current point - Don't look into the future
        historical_data = data.copy()
        if len(historical_data) < 2:
            return False  # Not enough data to determine support/resistance

        # Count how many strategies detect a support/resistance zone
        n_agreements = 0
        max_possible_agreements = n_strategies

        for strategy in self.strategies:
            try:
                # Generate levels for this strategy
                levels_df = self._load_sup_res_levels_data(
                    strategy, symbol, historical_data
                )
                if len(levels_df) == 0:
                    max_possible_agreements -= 1
                    # Early exit if consensus can't be reached even with remaining strategies
                    if (
                        n_agreements + (max_possible_agreements - n_agreements)
                        < n_required_agreements
                    ):
                        return False
                    continue

                # Check if current price is near any support/resistance levels
                if self._is_price_in_sup_res_zone(
                    historical_data,
                    timestamp,
                    levels_df,
                    strategy.sup_res_zone_threshold,
                ):
                    n_agreements += 1
                    # Early exit if minimum consensus is already met
                    if n_agreements >= n_required_agreements:
                        return True

            except Exception as e:
                # Skip strategy if it fails (e.g., not enough data)
                logger.error(
                    f"Error calculating support/resistance levels for strategy {strategy.__class__.__name__}: {e}"
                )
                max_possible_agreements -= 1
                # Early exit if consensus can't be reached even with remaining strategies
                if (
                    n_agreements + (max_possible_agreements - n_agreements)
                    < n_required_agreements
                ):
                    return False
                continue

        # Calculate consensus
        consensus_ratio = n_agreements / n_strategies if n_strategies > 0 else 0
        return consensus_ratio >= min_consensus

    def is_sup_res_zone_vectorized(
        self,
        data: pd.DataFrame,
        symbol: str,
        min_consensus: float = 0.5,
    ) -> pd.Series:
        """Check if prices are in support or resistance zones for all timestamps (vectorized).

        Args:
            data: The data to check, must contain 'High', 'Low', 'Close', 'datetime' columns.
            symbol: The symbol being analyzed.
            min_consensus: Minimum fraction of strategies that must agree (0.0 to 1.0).

        Returns:
            pd.Series of bool values indicating if each timestamp is in a support/resistance zone.
            Index matches the input data index.
        """
        n_strategies = len(self.strategies)
        if n_strategies == 0:
            raise ValueError("No strategies provided")

        if len(data) < 2:
            return pd.Series([False] * len(data), index=data.index)

        # Early exit if minimum consensus cannot be met
        n_required_agreements = int(np.ceil(min_consensus * n_strategies))
        if n_required_agreements == 0:
            return pd.Series([True] * len(data), index=data.index)

        # Initialize result array
        sup_res_zones = np.zeros(len(data), dtype=int)
        max_possible_agreements = np.full(len(data), n_strategies)

        # Process each strategy
        for strategy in self.strategies:
            try:
                # Generate levels for this strategy
                levels_df = self._load_sup_res_levels_data(strategy, symbol, data)

                if len(levels_df) == 0:
                    max_possible_agreements -= 1
                    continue

                # Vectorized processing for this strategy
                strategy_zones = self._is_price_in_sup_res_zone_vectorized(
                    data, levels_df, strategy.sup_res_zone_threshold
                )

                # Add to the consensus count
                # pyrefly: ignore
                sup_res_zones += strategy_zones.astype(int)

            except Exception as e:
                # Skip strategy if it fails
                logger.error(
                    f"Error calculating support/resistance levels for strategy {strategy.__class__.__name__}: {e}"
                )
                max_possible_agreements -= 1
                continue

        # Calculate consensus ratio for each timestamp
        # pyrefly: ignore
        consensus_ratios = np.divide(
            sup_res_zones,
            max_possible_agreements,
            out=np.zeros_like(sup_res_zones, dtype=float),
            where=max_possible_agreements != 0,
        )

        # Return boolean series based on consensus threshold
        result = consensus_ratios >= min_consensus
        return pd.Series(result, index=data.index)


class BaseSupResStrategy(ABC):
    """Base class for support and resistance strategies."""

    def __init__(self):
        """Initialize the strategy."""
        # The threshold for determining if price is near a support/resistance level
        self.sup_res_zone_threshold = 0.005  # Within 1% of the level

    @abstractmethod
    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate all support and resistance levels for every historical point for a symbol based on market data.

        The calculation of each level at each timestamp must be autoregressive.
        This means that the calculation of a level at a timestamp must only use data up to that timestamp.
        This is to prevent look-ahead bias whilst allowing for vectorization for efficiency.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with support and resistance levels.
            The columns are: ['datetime', 'level_values'].
            The 'level_values' column contains fixed-size NumPy arrays (128 elements) with support and resistance levels.
        """
        raise NotImplementedError()

    def generate_prediction_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate all support and resistance levels for the latest point for a symbol based on market data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with support and resistance levels.
            The columns are: ['datetime', 'level_values'].
            The 'level_values' column contains fixed-size NumPy arrays (128 elements) with support and resistance levels.
        """
        # TODO: Not being used.
        return self.generate_historical_levels(data, symbol).iloc[[-1]]

    def pivoted_levels(self, levels_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot the `levels_df` and collect common levels into a single row for easier visualization.

        This is a helper method for strategies that generate point-wise levels for every timestamp.

        Args:
            levels_df: DataFrame with timestamps as index and lists of support and resistance levels as columns.

        Returns:
            DataFrame with support and resistance levels.
            The columns are: ['datetime', 'level_values'].
            The 'level_values' column contains fixed-size NumPy arrays (128 elements) with support and resistance levels.
        """
        if (
            "datetime" not in levels_df.columns
            or "level_values" not in levels_df.columns
        ):
            raise ValueError(
                "levels_df must contain 'datetime' and 'level_values' columns"
            )

        result = levels_df.copy()

        mlb = MultiLabelBinarizer(sparse_output=True)

        result = result.join(
            # pyrefly: ignore
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(result.pop("level_values")),
                index=result.index,
                columns=mlb.classes_,
            )
        )

        numeric_cols = result.select_dtypes(include="number").columns

        result_rows = []
        for col in numeric_cols:
            nonzero_mask = result[col] != 0
            if nonzero_mask.any():
                # NOTE: This is an approximation and does not account for gaps between levels
                min_dt = result.loc[nonzero_mask, "datetime"].min()
                max_dt = result.loc[nonzero_mask, "datetime"].max()
                result_rows.append(
                    {
                        "datetime": min_dt,
                        "level_values": [col],
                    }
                )

        result = pd.DataFrame(result_rows)
        return result
