"""Rate of Change (RoC) Multi-Condition Crossover strategy with trend filter."""

import numpy as np
import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.data_units import DataUnitConverter


class RocMultiConditionCrossover(BaseStrategy):
    """A strategy using RoC crossover with optional zero-line, threshold, and trend filters.

    This strategy is a combination of the RoC crossover strategy and the trend filter strategy.
    The following filters are applied:
    - Zero-line filter: Ignore signals if RoC > 0 for buys and RoC < 0 for sells.
    - Threshold filter: Ignore signals if `|RoC| < min_threshold`.
    - Trend filter: Only take long signals if price > SMA(trend_period),
                    short signals if price < SMA(trend_period).
    The filters are put in place to mute signal noise, thus reduce false signals.

    Leading / Lagging: TODO
    Lag period (in candles): TODO
    Effect: TODO

    Refer to: https://www.investopedia.com/terms/r/rateofchange.asp
    """

    def __init__(
        self,
        roc_period: int = 14,
        signal_period: int = 9,
        trend_period: int = 90,  # This is a magic number.
        min_threshold: float = 1.0,
        use_zero_filter: bool = True,
        use_trend_filter: bool = True,
    ):
        """Initialize the RoC multi-condition crossover strategy.

        Args:
            roc_period: The number of candles to calculate RoC.
            signal_period: The number of candles to smooth RoC into a signal line.
            trend_period: Lookback period for trend filter (SMA).
            min_threshold: As a percentage. Ignore signals if `|RoC| < min_threshold`.
            use_zero_filter: If True, require RoC > 0 for buys and RoC < 0 for sells.
            use_trend_filter: If True, only take long signals if price > SMA(trend_period),
                              short signals if price < SMA(trend_period).
        """
        super().__init__()
        self.roc_period = roc_period
        self.signal_period = signal_period
        self.trend_period = trend_period
        self.min_threshold = min_threshold
        self.use_zero_filter = use_zero_filter
        self.use_trend_filter = use_trend_filter

        # This depends on the resampling of the data
        if (
            roc_period > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "1D", "1H")
            or (
                signal_period > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "1D", "1H")
            )
            or trend_period > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "1D", "1H")
        ):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({max(roc_period, signal_period, trend_period)}) "
                f"for meaningful evaluation. Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate historical scores based on RoC crossovers with filters.

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

        # === Step 1: Compute RoC and smoothed signal line ===
        roc = (
            data_resampled["Close"].diff(self.roc_period)
            / data_resampled["Close"].shift(self.roc_period)
        ) * 100
        roc_signal = roc.rolling(self.signal_period).mean()

        # Base crossover condition
        crossover_signal = (roc > roc_signal).astype(int) - (roc < roc_signal).astype(
            int
        )

        # === Step 2: Zero-line filter ===
        if self.use_zero_filter:
            long_mask = (crossover_signal == 1) & (roc > 0)
            short_mask = (crossover_signal == -1) & (roc < 0)
            crossover_signal = long_mask.astype(int) - short_mask.astype(int)

        # === Step 3: Threshold filter ===
        if self.min_threshold > 0:
            crossover_signal = crossover_signal.where(roc.abs() > self.min_threshold, 0)

        # === Step 4: Trend filter ===
        if self.use_trend_filter:
            trend_sma = data_resampled["Close"].rolling(self.trend_period).mean()
            long_mask = (crossover_signal == 1) & (data_resampled["Close"] > trend_sma)
            short_mask = (crossover_signal == -1) & (
                data_resampled["Close"] < trend_sma
            )
            crossover_signal = long_mask.astype(int) - short_mask.astype(int)

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
            A float between -1.0 and +1.0 representing the most recent signal.
        """
        n_candles_required = max(self.roc_period, self.signal_period, self.trend_period)
        return self.generate_historical_scores(
            # TODO: Using * 2.0 here to give buffer zone for any miscalculations.
            # This depends on the resampling of the data
            data.iloc[
                -(round(convert_units(n_candles_required, "1H", CANDLE_UNIT) * 2.0)) :
            ],
            symbol,
        ).iloc[-1]
