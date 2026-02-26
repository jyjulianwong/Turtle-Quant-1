"""Moving average convergence divergence (MACD) strategy implementation."""

import numpy as np
import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.data_units import DataUnitConverter


class MovingAverageConDiv(BaseStrategy):
    """Moving average convergence divergence strategy implementation.

    Leading / Lagging: Lagging when converging. Leading when diverging.
    Lag period (in candles): (`slow_candles` - 1) / 2 - (`fast_candles` - 1) / 2 + (`signal_candles` - 1) / 2
    Effect: Delay in recognizing that a new trend has begun.
    """

    def __init__(
        self,
        fast_candles: int = 12,
        slow_candles: int = 26,
        signal_candles: int = 9,
    ):
        """Initialize the MACD strategy.

        Args:
            fast_candles: The number of candles for the fast EMA.
            slow_candles: The number of candles for the slow EMA.
            signal_candles: The smoothing window applied to the MACD line to form the signal line.
        """
        super().__init__()
        self.fast_candles = fast_candles
        self.slow_candles = slow_candles
        self.signal_candles = signal_candles

        if fast_candles >= slow_candles:
            raise ValueError(
                f"Fast EMA period ({fast_candles}) must be less than slow EMA period ({slow_candles})."
            )

        # This depends on the resampling of the data
        if (
            slow_candles
            > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "1D", "1H") * 0.5
        ):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({slow_candles}) for meaningful evaluation. "
                f"Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate a historical score array for a symbol based on MACD.

        NOTE: Assume that the data is sorted by datetime.

        This strategy works best when resampled to daily data.
        Refer to: https://www.investopedia.com/terms/m/macd.asp

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with each value between -1.0 and +1.0, indexed by datetime
        """
        self.validate_data(data)

        # TODO: This is a magic number.
        data_resampled = DataUnitConverter.convert_to_1h_data(symbol, data)

        # Calculate EMAs
        ema_fast = (
            data_resampled["Close"].ewm(span=self.fast_candles, adjust=False).mean()
        )
        ema_slow = (
            data_resampled["Close"].ewm(span=self.slow_candles, adjust=False).mean()
        )

        # MACD line and Signal line
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_candles, adjust=False).mean()
        macd_hist = macd_line - signal_line

        # Normalize score
        scaling_factor = (
            100.0  # Smaller scaling factor since histogram is usually small
        )
        score = macd_hist * scaling_factor / data_resampled["Close"]
        if np.isnan(score.iloc[-1]):
            raise ValueError("Last score should not be NaN")

        score = score.reindex(data.index)
        score = score.bfill().ffill()

        return pd.Series(
            data=score.fillna(0).clip(-1, 1).values,
            index=pd.to_datetime(data["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate a prediction score based on the latest MACD values.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol to use for the strategy.

        Returns:
            A float between -1.0 and +1.0 representing the signal strength.
        """
        n_candles_required = max(
            self.fast_candles, self.slow_candles, self.signal_candles
        )
        return self.generate_historical_scores(
            # TODO: Using * 4.0 here to give buffer zone for any miscalculations.
            # TODO: Work out why this needs more than * 2.0.
            # This depends on the resampling of the data
            data.iloc[
                -(round(convert_units(n_candles_required, "1H", CANDLE_UNIT) * 4.0)) :
            ],
            symbol,
        ).iloc[-1]
