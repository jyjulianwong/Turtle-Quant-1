"""RSI Support Resistance Divergence strategy implementation."""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema  # pyrefly: ignore[missing-module-attribute]

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units


class RsiSupResDivergence(BaseStrategy):
    def __init__(self, lookback_candles: int = 60, local_extrema_window: int = 5):
        """Initialize the RSI strategy.

        Leading / Lagging: Lagging in construction. Leading in interpretation.
        Lag period (in candles): (`lookback_candles` - 1) / 2
        Effect: Delay in recognizing that price has deviated far enough from equilibrium.

        Args:
            lookback_candles: The number of periods to use for the RSI.
            local_extrema_window: The size of the window for the local maxima and minima.
        """

        super().__init__()
        self.lookback_candles = lookback_candles
        self.local_extrema_window = local_extrema_window

        if (
            lookback_candles
            > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "DAY", CANDLE_UNIT) * 0.5
        ):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({lookback_candles}) for meaningful evaluation to be done. "
                f"Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def _calc_rsi(self, close: pd.Series) -> pd.Series:
        """Calculate the RSI.

        Args:
            close: The close prices.

        Returns:
            The RSI.
        """

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        mean_gain = gain.rolling(self.lookback_candles).mean()
        mean_loss = loss.rolling(self.lookback_candles).mean()

        rs = mean_gain / (mean_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _find_extrema(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Find the local maxima and minima.

        Args:
            series: The series to find the extrema in.

        Returns:
            The local maxima and minima indices.
        """

        maximas = argrelextrema(
            series.values, np.greater_equal, order=self.local_extrema_window
        )[0]
        minimas = argrelextrema(
            series.values, np.less_equal, order=self.local_extrema_window
        )[0]
        return maximas, minimas

    def _get_divergence_signals(
        self, close: pd.Series, rsi: pd.Series, maximas: np.ndarray, minimas: np.ndarray
    ) -> pd.Series:
        """Get the divergence signals.

        Args:
            close: The close prices.
            rsi: The RSI values.
            maximas: The local maxima indices.
            minimas: The local minima indices.

        Returns:
            The divergence signals.
        """

        signal = pd.Series(index=close.index, data=0.0)

        # Bullish divergence: price makes lower low, RSI makes higher low
        for i in range(1, len(minimas)):
            if (
                close.iloc[minimas[i]] < close.iloc[minimas[i - 1]]
                and rsi.iloc[minimas[i]] > rsi.iloc[minimas[i - 1]]
            ):
                signal.iloc[minimas[i]] = 1.0

        # Bearish divergence: price makes higher high, RSI makes lower high
        for i in range(1, len(maximas)):
            if (
                close.iloc[maximas[i]] > close.iloc[maximas[i - 1]]
                and rsi.iloc[maximas[i]] < rsi.iloc[maximas[i - 1]]
            ):
                signal.iloc[maximas[i]] = -1.0

        return signal

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate the historical scores.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: The data to generate the scores from.
            symbol: The symbol to generate the scores for.

        Returns:
            The historical scores.
        """

        self.validate_data(data)

        close = data["Close"]

        rsi = self._calc_rsi(close)
        maximas, minimas = self._find_extrema(close)
        divergence_signals = self._get_divergence_signals(close, rsi, maximas, minimas)
        if np.isnan(divergence_signals.iloc[-1]):
            raise ValueError("Last score should not be NaN")

        # Optional: Smooth or decay signal over a few candles
        divergence_signals = divergence_signals.ffill(limit=3).fillna(0)

        return pd.Series(
            data=divergence_signals.clip(-1, 1).values,
            index=pd.to_datetime(data["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate the prediction score.

        NOTE: Assume that the data is sorted by datetime.

        Args:
            data: The data to generate the score from.
            symbol: The symbol to generate the score for.

        Returns:
            The prediction score.
        """

        return self.generate_historical_scores(
            data.iloc[-(self.lookback_candles + 1) :], symbol
        ).iloc[-1]
