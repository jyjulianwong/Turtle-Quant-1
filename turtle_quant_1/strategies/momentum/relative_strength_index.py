"""Relative Strength Index (RSI) strategy implementation."""

import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units


class RelativeStrengthIndex(BaseStrategy):
    """A strategy that uses the RSI to generate buy and sell signals."""

    def __init__(self, lookback_candles: int = 60):
        """Initialize the RSI strategy.

        Args:
            lookback_candles: The number of periods to use for the RSI.
        """
        super().__init__()
        self.lookback_candles = lookback_candles

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
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with each value between -1.0 and +1.0, indexed by datetime
        """
        self.validate_data(data)

        data_sorted = data.sort_values("datetime").copy()

        delta = data_sorted["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        mean_gain = gain.rolling(self.lookback_candles).mean()
        mean_loss = loss.rolling(self.lookback_candles).mean()

        rs = mean_gain / (mean_loss + 1e-9)  # Prevent division by zero
        rsi = 100 - (100 / (1 + rs))  # The standard RSI formula

        return pd.Series(
            data=((50 - rsi) / 50).clip(-1, 1).fillna(0).values,  # Rescale to -1 to +1
            index=pd.to_datetime(data_sorted["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        # RSI < 30 -> Score near +1 -> BUY
        # RSI > 70 -> Score near -1 -> SELL
        return self.generate_historical_scores(data, symbol).iloc[-1]
