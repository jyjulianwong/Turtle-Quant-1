"""Fast Stochastic Oscillator crossover strategy implementation."""

import numpy as np
import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.data_units import DataUnitConverter


class FastStochOscCrossover(BaseStrategy):
    """A strategy that uses Fast Stochastic Oscillator %K/%D crossovers to generate buy and sell signals.

    Can be intuitively interpreted as whether a stock is overbought or oversold.
    Refer to: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """

    def __init__(
        self,
        k_period: int = 14,
        d_period: int = 3,
    ):
        """Initialize the Stochastic Oscillator strategy.

        Leading / Lagging: Lagging in construction. Leading in interpretation.
        Lag period (in candles): (`k_period` - 1) / 2 + (`d_period` - 1) / 2
        Effect: Delay in recognizing that price has deviated far enough from equilibrium.

        Args:
            k_period: The number of candles to calculate %K.
            d_period: The number of candles to smooth %K into %D.
        """
        super().__init__()
        self.k_period = k_period
        self.d_period = d_period

        # This depends on the resampling of the data
        if k_period > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "1D", "1H"):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({k_period}) "
                f"for meaningful evaluation. Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate historical scores based on %K/%D crossovers.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with values between -1.0 and +1.0, indexed by datetime.
        """
        self.validate_data(data)

        # TODO: This is a magic number.
        data_resampled = DataUnitConverter.convert_to_1h_data(symbol, data)

        # Calculate %K
        lowest_low = data_resampled["Low"].rolling(self.k_period).min()
        highest_high = data_resampled["High"].rolling(self.k_period).max()
        percent_k = (
            100 * (data_resampled["Close"] - lowest_low) / (highest_high - lowest_low)
        )

        # Calculate %D (smoothed %K)
        percent_d = percent_k.rolling(self.d_period).mean()

        # Crossovers: +1 when %K > %D, -1 when %K < %D
        crossover_signal = (percent_k > percent_d).astype(int) - (
            percent_k < percent_d
        ).astype(int)
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
                -(round(convert_units(self.k_period, "1H", CANDLE_UNIT) * 2.0)) :
            ],
            symbol,
        ).iloc[-1]
