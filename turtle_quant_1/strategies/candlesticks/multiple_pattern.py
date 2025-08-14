import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.helpers import get_wick_directions_vectorized
from turtle_quant_1.strategies.helpers.support_resistance import SupResIndicator


class MultiplePattern(BaseStrategy):
    """Multiple pattern strategy."""

    def __init__(self):
        """Initialize the MultiplePattern strategy."""
        super().__init__()

        self.sup_res_indicator = SupResIndicator()

    def _get_score_for_candles_vectorized(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.Series:
        """Vectorized detection of multiple patterns.

        This checks if 3 consecutive candles have the same direction and if they are
        in a support or resistance zone.
        """
        # Get vectorized wick directions for all candles
        wick_directions = get_wick_directions_vectorized(data)

        # Rolling count of consecutive up wicks (value = 1) in last 3 candles
        up_count = (wick_directions == +1).astype(int).rolling(window=3).sum()
        # Rolling count of consecutive down wicks (value = -1) in last 3 candles
        down_count = (wick_directions == -1).astype(int).rolling(window=3).sum()

        # Pattern detected when we have 3 or more of the same direction
        bullish_pattern = down_count >= 3  # 3+ down wicks suggest bullish reversal
        bearish_pattern = up_count >= 3  # 3+ up wicks suggest bearish reversal

        # Get support/resistance zones (vectorized)
        sup_res_zones = self.sup_res_indicator.is_sup_res_zone_vectorized(
            data, symbol
        ).astype(float)

        # Return pattern scores (only in sup/res zones)
        return (
            bullish_pattern.astype(float) - bearish_pattern.astype(float)
        ) * sup_res_zones

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores = self._get_score_for_candles_vectorized(data, symbol)
        # Check for any occurrence of the pattern in last 6 candles
        scores = scores.fillna(0).rolling(window=6).sum().clip(-1, 1)

        return pd.Series(
            data=scores.values, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        scores = self._get_score_for_candles_vectorized(data, symbol)
        # Check for any occurrence of the pattern in last 6 candles
        scores = scores.fillna(0).rolling(window=6).sum().clip(-1, 1)

        return float(scores.iloc[-1])
