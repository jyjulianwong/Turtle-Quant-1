"""Bollinger band strategy implementation."""

import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units


class BollingerBand(BaseStrategy):
    """A strategy that uses the Bollinger bands to generate buy and sell signals."""

    def __init__(self, lookback_candles: int = 180, n_std: int = 3):
        """Initialize the Bollinger band strategy.

        Args:
            lookback_candles: The number of periods to use for the Bollinger bands.
            n_std: The number of standard deviations to use for the Bollinger bands.
        """
        super().__init__()
        self.lookback_candles = lookback_candles
        self.n_std = n_std

        if (
            lookback_candles
            > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "DAY", CANDLE_UNIT) * 0.5
        ):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({lookback_candles}) for meaningful evaluation to be done. "
                f"Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

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

        ma = data_sorted["Close"].rolling(self.lookback_candles).mean()
        std = data_sorted["Close"].rolling(self.lookback_candles).std()

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
