import numpy as np
import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy


class MomentumPattern(BaseStrategy):
    def _is_sideways_region_vecd(
        self, data: pd.DataFrame, window: int = 20, flat_threshold: float = 0.015
    ) -> pd.Series:
        """Vectorized check if the market is in a sideways pattern for all points."""
        recent: pd.DataFrame = data["Close"].rolling(window).agg(["max", "min"])
        flat_range: pd.Series = recent["max"] - recent["min"]
        mean_price: pd.Series = data["Close"].rolling(window).mean()
        ratio: pd.Series = flat_range / mean_price
        return ratio < flat_threshold

    def _get_score_for_candles_vecd(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Vectorized detection of momentum patterns.

        This checks if candles have strong bodies and if the market is sideways.
        """
        # Calculate candle body and wick ranges vectorized
        candle_body_range = (data["Close"] - data["Open"]).abs()
        candle_wick_range = data["High"] - data["Low"]

        # Avoid division by zero for doji candles
        candle_wick_range = candle_wick_range.replace(0, np.nan)

        # Detect strong body candles (avoid NaN results)
        has_strong_body = (candle_body_range / candle_wick_range) > 0.7
        has_strong_body = has_strong_body.fillna(False)

        # Detect sideways market for each candle using vectorized method
        is_sideways = self._is_sideways_region_vecd(data)

        # Calculate direction of the candle (bullish=1, bearish=-1)
        candle_direction = np.sign(data["Close"] - data["Open"])

        # Pattern score: Strong body in sideways region gets direction score
        pattern_scores = has_strong_body & is_sideways

        return pattern_scores.astype(float) * candle_direction

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores = self._get_score_for_candles_vecd(data, symbol)
        # Check for any occurrence of the pattern in last 6 candles
        scores = scores.fillna(0).rolling(window=6).sum().clip(-1, 1)

        return pd.Series(
            data=scores.values, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        scores = self._get_score_for_candles_vecd(data, symbol)
        # Check for any occurrence of the pattern in last 6 candles
        scores = scores.fillna(0).rolling(window=6).sum().clip(-1, 1)

        return float(scores.iloc[-1])
