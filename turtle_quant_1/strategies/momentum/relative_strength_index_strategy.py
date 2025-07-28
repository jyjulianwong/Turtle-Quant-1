"""Relative Strength Index (RSI) strategy implementation."""

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy


class RelativeStrengthIndexStrategy(BaseStrategy):
    """A strategy that uses the RSI to generate buy and sell signals."""

    def __init__(self, name: str = "RelativeStrengthIndexStrategy", candles: int = 120):
        """Initialize the RSI strategy.

        Args:
            name: The name of the strategy.
            candles: The number of periods to use for the RSI.
        """
        super().__init__(name)
        self.candles = candles

    def generate_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(self.candles).mean()
        avg_loss = loss.rolling(self.candles).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        score = ((50 - rsi) / 50).clip(-1, 1)
        # RSI < 30 -> Score near +1 -> BUY
        # RSI > 70 -> Score near -1 -> SELL
        return score.fillna(0).iloc[-1]
