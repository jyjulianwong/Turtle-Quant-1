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

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a historical score array for a symbol based on market data."""
        self.validate_data(data)

        delta = data["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(self.candles).mean()
        avg_loss = loss.rolling(self.candles).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return ((50 - rsi) / 50).clip(-1, 1).fillna(0)

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        # RSI < 30 -> Score near +1 -> BUY
        # RSI > 70 -> Score near -1 -> SELL
        return self.generate_historical_scores(data, symbol).iloc[-1]
