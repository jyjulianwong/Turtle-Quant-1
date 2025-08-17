"""Slow Stochastic Oscillator crossover strategy implementation."""

import numpy as np
import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.helpers import convert_to_daily_data


class SlowStochOscCrossover(BaseStrategy):
    """A strategy that uses Slow Stochastic Oscillator %K/%D crossovers to generate buy and sell signals.

    Can be intuitively interpreted as whether a stock is overbought or oversold.
    Refer to: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """

    def __init__(
        self,
        k_period: int = 14,
        k_smooth: int = 3,
        d_period: int = 3,
    ):
        """Initialize the Slow Stochastic Oscillator strategy.

        Args:
            k_period: The number of candles to calculate %K.
            k_smooth: The smoothing period applied to %K to create Slow %K.
            d_period: The smoothing period applied to Slow %K to create Slow %D.
        """
        super().__init__()
        self.k_period = k_period
        self.k_smooth = k_smooth
        self.d_period = d_period

        # This depends on the resampling of the data
        if k_period > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "DAY", "DAY"):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({k_period}) "
                f"for meaningful evaluation. Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate historical scores based on Slow %K/%D crossovers.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with values between -1.0 and +1.0, indexed by datetime.
        """
        self.validate_data(data)

        data_resampled = convert_to_daily_data(data)

        # Fast %K
        lowest_low = data_resampled["Low"].rolling(self.k_period).min()
        highest_high = data_resampled["High"].rolling(self.k_period).max()
        fast_k = (
            100 * (data_resampled["Close"] - lowest_low) / (highest_high - lowest_low)
        )

        # Slow %K (smoothed Fast %K)
        slow_k = fast_k.rolling(self.k_smooth).mean()

        # Slow %D (smoothed Slow %K)
        slow_d = slow_k.rolling(self.d_period).mean()

        # Crossovers: +1 when Slow %K > Slow %D, -1 when Slow %K < Slow %D
        crossover_signal = (slow_k > slow_d).astype(int) - (slow_k < slow_d).astype(int)
        if np.isnan(crossover_signal.iloc[-1]):
            raise ValueError("Last score should not be NaN")

        crossover_signal = crossover_signal.reindex(data.index)
        crossover_signal = crossover_signal.bfill().ffill()

        return pd.Series(
            data=crossover_signal.fillna(0).clip(-1, 1).values,
            index=pd.to_datetime(data["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate the latest prediction score.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol being analyzed.

        Returns:
            A float between -1.0 and +1.0 representing the most recent crossover signal.
        """
        return self.generate_historical_scores(
            # TODO: Using * 2.0 here to give buffer zone for any miscalculations.
            # This depends on the resampling of the data
            data.iloc[
                -(round(convert_units(self.k_period, "DAY", CANDLE_UNIT) * 2.0)) :
            ],
            symbol,
        ).iloc[-1]
