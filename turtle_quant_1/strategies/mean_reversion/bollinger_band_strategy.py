"""Bollinger band strategy implementation."""

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy


class BollingerBandStrategy(BaseStrategy):
    """A strategy that uses the Bollinger bands to generate buy and sell signals."""

    def __init__(
        self, name: str = "BollingerBandStrategy", window: int = 120, n_std: int = 2
    ):
        """Initialize the Bollinger band strategy.

        Args:
            name: The name of the strategy.
            window: The number of periods to use for the Bollinger bands.
            n_std: The number of standard deviations to use for the Bollinger bands.
        """
        super().__init__(name)
        self.window = window
        self.n_std = n_std

    def generate_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        ma = data["Close"].rolling(self.window).mean()
        std = data["Close"].rolling(self.window).std()

        upper = ma + self.n_std * std
        lower = ma - self.n_std * std

        score = ((data["Close"] - ma) / (upper - lower)).clip(-1, 1)
        # Score near +1 -> BUY
        # Score near -1 -> SELL
        return score.fillna(0).iloc[-1]
