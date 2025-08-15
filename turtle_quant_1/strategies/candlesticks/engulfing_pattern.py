import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import CANDLE_UNIT, convert_units
from turtle_quant_1.strategies.helpers.support_resistance import SupResIndicator


class EngulfingPattern(BaseStrategy):
    """Engulfing pattern strategy."""

    def __init__(self):
        """Initialize the EngulfingPattern strategy."""
        super().__init__()

        self.sup_res_indicator = SupResIndicator()

    def _get_score_for_candles_vectorized(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.Series:
        """Vectorized detection of engulfing patterns."""
        # Shift data to get previous candle
        prev_open = data["Open"].shift(1)
        prev_close = data["Close"].shift(1)
        curr_open = data["Open"]
        curr_close = data["Close"]

        # Vectorized bullish engulfing detection
        bullish = (
            (prev_close < prev_open)  # Previous candle is bearish
            & (curr_close > curr_open)  # Current candle is bullish
            & (curr_open < prev_close)  # Current opens below prev close
            & (curr_close > prev_open)  # Current closes above prev open
        )

        # Vectorized bearish engulfing detection
        bearish = (
            (prev_close > prev_open)  # Previous candle is bullish
            & (curr_close < curr_open)  # Current candle is bearish
            & (curr_open > prev_close)  # Current opens above prev close
            & (curr_close < prev_open)  # Current closes below prev open
        )

        sup_res_zones = (
            self.sup_res_indicator.is_sup_res_zone_vectorized(data, symbol)
            .astype(float)
            .fillna(0.0)
        )

        # Return pattern scores
        return (bullish.astype(float) - bearish.astype(float)) * sup_res_zones

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores = self._get_score_for_candles_vectorized(data, symbol)
        # Check for any occurrence of the pattern in last 6 candles
        scores = (
            scores.fillna(0)
            .rolling(window=convert_units(2, "DAY", CANDLE_UNIT))
            .sum()
            .clip(-1, 1)
        )

        return pd.Series(
            data=scores.values, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        scores = self._get_score_for_candles_vectorized(data, symbol)
        # Check for any occurrence of the pattern in last 6 candles
        scores = (
            scores.fillna(0)
            .rolling(window=convert_units(2, "DAY", CANDLE_UNIT))
            .sum()
            .clip(-1, 1)
        )

        return float(scores.iloc[-1])
