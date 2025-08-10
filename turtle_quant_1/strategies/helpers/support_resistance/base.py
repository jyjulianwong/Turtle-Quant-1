"""Base class for support and resistance strategies."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from turtle_quant_1.strategies.helpers.multiprocessing import ProcessSafeCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global cache instance
_global_cache = ProcessSafeCache()


def get_global_cache():
    """Get the global cache instance."""
    return _global_cache


class SupResIndicator:
    """Entrypoint for determining support and resistance zones for other strategies."""

    def __init__(self, strategies: Optional[List["BaseSupResStrategy"]] = None):
        """Initialize the SupResIndicator with a list of strategies.

        Args:
            strategies: List of support/resistance strategies to use for consensus.
                If None, will use default strategies.
        """
        if strategies is None:
            # Import here to avoid circular imports
            from .stnry_fibonacci_retrace import (
                StnryFibonacciRetrace,
            )
            from .stnry_gaussian_kde import (
                StnryGaussianKDE,
            )
            from .stnry_local_extrema import (
                StnryLocalExtrema,
            )
            from .stnry_pivot_point import (
                StnryPivotPoint,
            )

            self.strategies = [
                StnryFibonacciRetrace(),
                StnryGaussianKDE(),
                StnryLocalExtrema(),
                StnryPivotPoint(),
            ]  # pyrefly: ignore[bad-assignment]
        else:
            self.strategies = strategies  # pyrefly: ignore[bad-assignment]

    def _is_price_near_levels(
        self,
        historical_data: pd.DataFrame,
        current_idx: int,
        current_price: float,
        levels_df: pd.DataFrame,
        threshold: float,
    ) -> bool:
        """Check if current price is near any support/resistance levels.

        Args:
            historical_data: Historical data up to current point.
            current_idx: Current index in the data.
            current_price: The current price to check.
            levels_df: DataFrame with support/resistance levels.
            threshold: Price proximity threshold (as fraction of price).

        Returns:
            True if price is near any level, False otherwise.
        """
        if len(levels_df) == 0:
            return False

        # Get current timestamp for filtering relevant levels
        if "datetime" in historical_data.columns:
            current_time = historical_data.iloc[current_idx]["datetime"]
        else:
            current_time = historical_data.index[current_idx]

        current_time = pd.to_datetime(current_time)

        # Filter levels that are relevant for the current time
        relevant_levels = []

        for _, row in levels_df.iterrows():
            start_time = pd.to_datetime(row["datetime_beg"])
            end_time = pd.to_datetime(row["datetime_end"])

            # Check if current time falls within the level's validity period
            if start_time <= current_time <= end_time:
                if isinstance(row["level_values"], list):
                    relevant_levels.extend(row["level_values"])
                else:
                    relevant_levels.append(row["level_values"])

        if not relevant_levels:
            return False

        # Check if current price is within threshold of any level
        price_threshold = current_price * threshold

        for level in relevant_levels:
            if abs(current_price - level) <= price_threshold:
                return True

        return False

    def _load_sup_res_levels_data(
        self, strategy: "BaseSupResStrategy", symbol: str, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the support and resistance data for a symbol.

        Args:
            strategy: The strategy to get the support and resistance data for.
            symbol: The symbol to get the support and resistance data for.
            data: The data to get the support and resistance data for.
        """
        cache_key = f"{symbol}_{strategy.__class__.__name__}_{len(data)}"
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

    def is_sup_res_zone(
        self,
        data: pd.DataFrame,
        idx: int,
        symbol: str,
        min_consensus: float = 0.5,
    ) -> bool:
        """Check if the current price is in a support or resistance zone.

        Args:
            data: The data to check, must contain 'High', 'Low', 'Close' columns.
            idx: The index of the current price to check.
            symbol: The symbol being analyzed.
            min_consensus: Minimum fraction of strategies that must agree (0.0 to 1.0).

        Returns:
            True if the current price is in a support or resistance zone, False otherwise.
        """
        if idx >= len(data) or idx < 0:
            raise ValueError(
                f"Index {idx} is out of bounds for data of length {len(data)}"
            )

        total_strategies = len(self.strategies)
        if total_strategies == 0:
            raise ValueError("No strategies provided")

        # Get the current price (using Close price)
        current_price = data.iloc[idx]["Close"]

        # Get data up to current point (don't look into the future)
        historical_data = data.iloc[: idx + 1].copy()

        if len(historical_data) < 2:
            return False  # Not enough data to determine support/resistance

        # Count how many strategies detect a support/resistance zone
        strategies_agree = 0

        for strategy in self.strategies:
            try:
                # Generate levels for this strategy
                levels_df = self._load_sup_res_levels_data(
                    strategy, symbol, historical_data
                )
                if len(levels_df) == 0:
                    continue

                # Check if current price is near any support/resistance levels
                if self._is_price_near_levels(
                    historical_data,
                    idx,
                    current_price,
                    levels_df,
                    strategy.sup_res_zone_threshold,
                ):
                    strategies_agree += 1

            except Exception as e:
                # Skip strategy if it fails (e.g., not enough data)
                logger.error(
                    f"Error calculating support/resistance levels for strategy {strategy.__class__.__name__}: {e}"
                )
                continue

        # Calculate consensus
        consensus_ratio = strategies_agree / total_strategies
        return consensus_ratio >= min_consensus


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

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with timestamps as index and lists of support and resistance levels as columns.
            The columns are: ['datetime_beg', 'datetime_end', 'level_values'].
            The 'level_values' column is a list of support and resistance levels of type list[float].
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
            DataFrame with timestamps as index and lists of support and resistance levels as columns.
            The columns are: ['datetime_beg', 'datetime_end', 'level_values'].
            The 'level_values' column is a list of support and resistance levels of type list[float].
        """
        # TODO: Not being used.
        return self.generate_historical_levels(data, symbol).iloc[[-1]]
