"""Moving average crossover strategy implementation."""

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """A strategy that uses moving averages to generate buy and sell signals."""

    def __init__(
        self,
        name: str = "MovingAverageCrossoverStrategy",
        sma_candles: int = 30,
        lma_candles: int = 120,
    ):
        """Initialize the moving average crossover strategy.

        Args:
            name: The name of the strategy.
            sma_candles: The number of candles to use for the short moving average.
            lma_candles: The number of candles to use for the long moving average.
        """

        super().__init__(name)
        self.sma_candles = sma_candles
        self.lma_candles = lma_candles

    def generate_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """

        sma = data["Close"].rolling(self.sma_candles).mean()
        lma = data["Close"].rolling(self.lma_candles).mean()

        score = (sma - lma) / data["Close"]

        return score.fillna(0).clip(-1, 1).iloc[-1]
