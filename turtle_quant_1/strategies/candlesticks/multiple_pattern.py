import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.helpers import get_wick_direction
from turtle_quant_1.strategies.helpers.support_resistance import SupResIndicator


class MultiplePattern(BaseStrategy):
    """Multiple pattern strategy."""

    def __init__(self):
        """Initialize the MultiplePattern strategy."""
        super().__init__()

        self.sup_res_indicator = SupResIndicator()

    def _get_score_for_candle(self, data: pd.DataFrame, idx: int, symbol: str) -> float:
        """Get a single score for the multiple pattern.

        This checks if 3 consecutivecandles have the same direction and if they are in a support or resistance zone.

        Args:
            data: The data to get the score for.
            idx: The index of the current row.

        Returns:
            The score for the multiple pattern.
        """

        wick_dirs = [
            get_wick_direction(row) for _, row in data.iloc[idx - 3 : idx].iterrows()
        ]

        if wick_dirs.count("down") >= 3 and self.sup_res_indicator.is_sup_res_zone(
            data, idx, symbol
        ):
            return +1.0  # bullish reversal
        elif wick_dirs.count("up") >= 3 and self.sup_res_indicator.is_sup_res_zone(
            data, idx, symbol
        ):
            return -1.0  # bearish reversal
        return 0.0

    def _get_score(self, data: pd.DataFrame, idx: int, symbol: str) -> float:
        """Uses _get_score_for_candle to get the maximum score for the last 6 candles.

        Checks if this candlestick pattern has occurred in the last 6 timestamps.

        Args:
            data: The data to get the score for.
            idx: The index of the current row.

        Returns:
            The score for the multiple pattern.
        """
        if idx - 6 < 0:
            return 0.0  # Data out of range

        recent_data = data.iloc[idx - 6 : idx]  # TODO: Respect CANDLE_UNIT.
        scores = recent_data.apply(
            lambda row: self._get_score_for_candle(data, row.name, symbol), axis=1
        )
        return scores.max()

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores = data.apply(lambda row: self._get_score(data, row.name, symbol), axis=1)
        return pd.Series(
            data=scores.values, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        return float(self._get_score(data, len(data) - 1, symbol))
