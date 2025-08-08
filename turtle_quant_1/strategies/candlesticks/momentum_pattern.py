import pandas as pd
import numpy as np

from turtle_quant_1.strategies.base import BaseStrategy


class MomentumPattern(BaseStrategy):
    def _is_sideways_market(
        self, data: pd.DataFrame, window: int = 20, flat_threshold: float = 0.015
    ) -> bool:
        recent: pd.DataFrame = data["Close"].rolling(window).agg(["max", "min"])
        flat_range: pd.Series = recent["max"] - recent["min"]
        mean_price: pd.Series = data["Close"].rolling(window).mean()
        ratio: pd.Series = flat_range / mean_price
        return ratio.iloc[-1] < flat_threshold

    def _get_score_for_candle(self, data: pd.DataFrame, idx: int) -> float:
        """Get a single score for the momentum pattern.

        This checks if the last candle has a strong body and if the market is sideways.

        Args:
            data: The data to get the score for.
            idx: The index of the current row.

        Returns:
            The score for the momentum pattern.
        """
        candle_body_range = abs(float(data["Close"].iloc[idx] - data["Open"].iloc[idx]))
        candle_wick_range = float(data["High"].iloc[idx] - data["Low"].iloc[idx])
        has_strong_body = candle_body_range / candle_wick_range > 0.7

        if has_strong_body and self._is_sideways_market(data.iloc[:idx]):
            return float(np.sign(data["Close"].iloc[idx] - data["Open"].iloc[idx]))

        return 0.0

    def _get_score(self, data: pd.DataFrame, idx: int) -> float:
        """Uses _get_score_for_candle to get the maximum score for the last 6 candles.

        Checks if this candlestick pattern has occurred in the last 6 timestamps.

        Args:
            data: The data to get the score for.
            idx: The index of the current row.

        Returns:
            The score for the momentum pattern.
        """
        if idx - 6 < 0:
            return 0.0  # Data out of range

        recent_data = data.iloc[idx - 6 : idx]  # TODO: Respect CANDLE_UNIT.
        scores = recent_data.apply(
            lambda row: self._get_score_for_candle(data, row.name), axis=1
        )
        return scores.max()

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores = data.apply(lambda row: self._get_score(data, row.name), axis=1)
        return pd.Series(
            data=scores.values, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        return float(self._get_score(data, len(data) - 1))
