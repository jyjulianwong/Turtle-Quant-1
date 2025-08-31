"""Relative Strength Index (RSI) strategy implementation."""

import numpy as np
import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.data_units import DataUnitConverter


class RelativeStrengthIndex(BaseStrategy):
    """A strategy that uses the RSI to generate buy and sell signals.

    Leading / Lagging: Lagging in construction. Leading in interpretation.
    Lag period (in candles): (`lookback_candles` - 1) / 2
    Effect: Delay in recognizing that a new trend has begun.
    """

    def __init__(self, lookback_candles: int = 14):
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

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with each value between -1.0 and +1.0, indexed by datetime
        """
        self.validate_data(data)

        data_resampled = DataUnitConverter.convert_to_daily_data(symbol, data)

        delta = data_resampled["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        mean_gain = gain.rolling(self.lookback_candles).mean()
        mean_loss = loss.rolling(self.lookback_candles).mean()

        rs = mean_gain / (mean_loss + 1e-9)  # Prevent division by zero
        rsi = 100 - (100 / (1 + rs))  # The standard RSI formula
        if np.isnan(rsi.iloc[-1]):
            raise ValueError("Last score should not be NaN")

        rsi = rsi.reindex(data.index)
        rsi = rsi.bfill().ffill()

        return pd.Series(
            data=((50 - rsi) / 50).clip(-1, 1).fillna(0).values,  # Rescale to -1 to +1
            index=pd.to_datetime(data["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a score for the strategy.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.
        """
        # RSI < 30 -> Score near +1 -> BUY
        # RSI > 70 -> Score near -1 -> SELL
        return self.generate_historical_scores(
            # TODO: Using * 2.0 here to give buffer zone for any miscalculations.
            # This depends on the resampling of the data
            data.iloc[
                -(
                    round(
                        convert_units(self.lookback_candles, "DAY", CANDLE_UNIT) * 2.0
                    )
                ) :
            ],
            symbol,
        ).iloc[-1]
