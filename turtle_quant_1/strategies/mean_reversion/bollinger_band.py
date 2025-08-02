"""Bollinger band strategy implementation."""

import pandas as pd

from turtle_quant_1.strategies.base import BaseStrategy


class BollingerBand(BaseStrategy):
    """A strategy that uses the Bollinger bands to generate buy and sell signals."""

    def __init__(self, window: int = 180, n_std: int = 3):
        """Initialize the Bollinger band strategy.

        Args:
            window: The number of periods to use for the Bollinger bands.
            n_std: The number of standard deviations to use for the Bollinger bands.
        """
        super().__init__()
        self.window = window
        self.n_std = n_std

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a historical score array for a symbol based on market data.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.

        Returns:
            Score array with each value between -1.0 and +1.0, indexed by datetime
        """
        self.validate_data(data)

        data_sorted = data.sort_values("datetime").copy()

        ma = data_sorted["Close"].rolling(self.window).mean()
        std = data_sorted["Close"].rolling(self.window).std()

        upper = ma + self.n_std * std
        lower = ma - self.n_std * std

        return pd.Series(
            data=((data_sorted["Close"] - ma) / (upper - lower))
            .clip(-1, 1)
            .fillna(0)
            .values,
            index=pd.to_datetime(data_sorted["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        # Score near +1 -> BUY
        # Score near -1 -> SELL
        return self.generate_historical_scores(data, symbol).iloc[-1]
