"""Fast Stochastic Oscillator crossover strategy implementation."""

import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.helpers import convert_to_daily_data


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

        Args:
            k_period: The number of candles to calculate %K.
            d_period: The number of candles to smooth %K into %D.
        """
        super().__init__()
        self.k_period = k_period
        self.d_period = d_period

        # This depends on the resampling of the data
        if k_period > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "DAY", "DAY"):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({k_period}) "
                f"for meaningful evaluation. Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate historical scores based on %K/%D crossovers.

        Args:
            data: DataFrame with OHLCV data.
            symbol: The symbol being analyzed.

        Returns:
            Score array with values between -1.0 and +1.0, indexed by datetime.
        """
        self.validate_data(data)

        data_sorted = data.sort_values("datetime").copy()
        data_resampled = convert_to_daily_data(data_sorted)

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

        crossover_signal = crossover_signal.reindex(data_sorted.index)
        crossover_signal = crossover_signal.bfill().ffill()

        return pd.Series(
            data=crossover_signal.fillna(0).clip(-1, 1).values,
            index=pd.to_datetime(data_sorted["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate the latest prediction score.

        Args:
            data: The data to use for the strategy.
            symbol: The symbol being analyzed.

        Returns:
            A float between -1.0 and +1.0 representing the most recent crossover signal.
        """
        return self.generate_historical_scores(data, symbol).iloc[-1]
