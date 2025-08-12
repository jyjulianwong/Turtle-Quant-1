import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.support_resistance import SupResIndicator


class EngulfingPattern(BaseStrategy):
    """Engulfing pattern strategy."""

    def __init__(self):
        """Initialize the EngulfingPattern strategy."""
        super().__init__()

        self.sup_res_indicator = SupResIndicator()

    def _get_score_for_candle(self, data: pd.DataFrame, idx: int, symbol: str) -> float:
        """Get a single score for the engulfing pattern.

        This checks if the last candle is an engulfing pattern and if it is in a support or resistance zone.

        Args:
            data: The data to get the score for.
            idx: The index of the current row.

        Returns:
            The score for the engulfing pattern.
        """

        if idx < 1:
            return 0.0

        prev: pd.Series = data.iloc[idx - 1]
        curr: pd.Series = data.iloc[idx]

        if (
            prev["Close"] < prev["Open"]
            and curr["Close"] > curr["Open"]
            and curr["Open"] < prev["Close"]
            and curr["Close"] > prev["Open"]
        ):
            return (
                +1.0
                if self.sup_res_indicator.is_sup_res_zone(data, idx, symbol)
                else 0.0
            )  # Bullish
        if (
            prev["Close"] > prev["Open"]
            and curr["Close"] < curr["Open"]
            and curr["Open"] > prev["Close"]
            and curr["Close"] < prev["Open"]
        ):
            return (
                -1.0
                if self.sup_res_indicator.is_sup_res_zone(data, idx, symbol)
                else 0.0
            )  # Bearish
        return 0.0

    def _get_score(self, data: pd.DataFrame, idx: int, symbol: str) -> float:
        """Uses _get_score_for_candle to get the maximum score for the last 6 candles.

        Checks if this candlestick pattern has occurred in the last 6 timestamps.

        Args:
            data: The data to get the score for.
            idx: The index of the current row.

        Returns:
            The score for the engulfing pattern.
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
