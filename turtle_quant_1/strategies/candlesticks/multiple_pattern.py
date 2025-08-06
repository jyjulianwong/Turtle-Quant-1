from typing import List

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers import (
    is_support_resistance_zone,
    get_wick_direction,
)


class MultiplePattern(BaseStrategy):
    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores: List[float] = [0.0] * len(data)
        for i in range(3, len(data)):
            wick_dirs: List[str] = [
                get_wick_direction(row) for _, row in data.iloc[i - 3 : i].iterrows()
            ]
            if wick_dirs.count("down") >= 3 and is_support_resistance_zone(data, i):
                scores[i] = 1.0  # bullish reversal
            elif wick_dirs.count("up") >= 3 and is_support_resistance_zone(data, i):
                scores[i] = -1.0  # bearish reversal

        return pd.Series(
            data=scores, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        wick_dirs: List[str] = [
            get_wick_direction(row) for _, row in data.iloc[-3:].iterrows()
        ]
        i: int = len(data) - 1
        if wick_dirs.count("down") >= 3 and is_support_resistance_zone(data, i):
            return 1.0
        elif wick_dirs.count("up") >= 3 and is_support_resistance_zone(data, i):
            return -1.0
        return 0.0
