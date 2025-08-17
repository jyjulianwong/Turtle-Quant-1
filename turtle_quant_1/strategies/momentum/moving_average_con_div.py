"""Moving average convergence divergence (MACD) strategy implementation."""

import pandas as pd

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.helpers import convert_to_daily_data


class MovingAverageConDiv(BaseStrategy):
    """Moving average convergence divergence strategy implementation."""

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
            > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "DAY", "DAY") * 0.5
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

        data_resampled = convert_to_daily_data(data)

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
        return self.generate_historical_scores(data, symbol).iloc[-1]
