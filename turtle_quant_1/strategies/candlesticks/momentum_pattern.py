from typing import List

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

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        scores: List[float] = []
        for i in range(1, len(data)):
            body: float = abs(float(data["Close"].iloc[i] - data["Open"].iloc[i]))
            range_: float = float(data["High"].iloc[i] - data["Low"].iloc[i])
            strong_body: bool = body / range_ > 0.7
            if strong_body and self._is_sideways_market(data.iloc[:i]):
                direction: float = float(
                    np.sign(data["Close"].iloc[i] - data["Open"].iloc[i])
                )
                scores.append(direction)
            else:
                scores.append(0.0)

        return pd.Series(
            data=[0.0] + scores, index=pd.to_datetime(data["datetime"]), dtype=float
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        i = len(data) - 1
        wick_body: float = abs(float(data["Close"].iloc[i] - data["Open"].iloc[i]))
        wick_range: float = float(data["High"].iloc[i] - data["Low"].iloc[i])
        strong_body: bool = wick_body / wick_range > 0.7
        if strong_body and self._is_sideways_market(data):
            return float(np.sign(data["Close"].iloc[i] - data["Open"].iloc[i]))
        return 0.0
