from typing import List

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers import is_support_resistance_zone


class EngulfingPattern(BaseStrategy):
    def _get_engulfing_score(self, data: pd.DataFrame, idx: int) -> int:
        if idx < 1:
            return 0

        prev: pd.Series = data.iloc[idx - 1]
        curr: pd.Series = data.iloc[idx]

        if (
            prev["Close"] < prev["Open"]
            and curr["Close"] > curr["Open"]
            and curr["Open"] < prev["Close"]
            and curr["Close"] > prev["Open"]
        ):
            return 1  # Bullish
        if (
            prev["Close"] > prev["Open"]
            and curr["Close"] < curr["Open"]
            and curr["Open"] > prev["Close"]
            and curr["Close"] < prev["Open"]
        ):
            return -1  # Bearish
        return 0

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores: List[float] = []
        for i in range(len(data)):
            score = self._get_engulfing_score(data, i)
            if score != 0 and is_support_resistance_zone(data, i):
                scores.append(float(score))
            else:
                scores.append(0.0)

        return pd.Series(
            data=scores, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        i = len(data) - 1
        score = self._get_engulfing_score(data, i)
        if score != 0 and is_support_resistance_zone(data, i):
            return float(score)
        return 0.0
