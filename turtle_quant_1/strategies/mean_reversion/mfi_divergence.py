"""Money Flow Index (MFI) Divergence strategy implementation."""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema  # pyrefly: ignore[missing-module-attribute]

from turtle_quant_1.config import BACKTESTING_MAX_LOOKBACK_DAYS, CANDLE_UNIT
from turtle_quant_1.strategies.base import BaseStrategy
from turtle_quant_1.strategies.helpers.candle_units import convert_units


class MfiDivergence(BaseStrategy):
    def __init__(self, mfi_period: int = 14, local_extrema_window: int = 5):
        """Initialize the MFI Divergence strategy.

        Args:
            mfi_period: Number of candles for MFI calculation.
            local_extrema_window: Window for local maxima/minima detection.
        """

        super().__init__()
        self.mfi_period = mfi_period
        self.local_extrema_window = local_extrema_window

        if (
            mfi_period
            > convert_units(BACKTESTING_MAX_LOOKBACK_DAYS, "DAY", CANDLE_UNIT) * 0.5
        ):
            raise ValueError(
                f"This strategy relies on too many lookback candles ({mfi_period}) for meaningful evaluation. "
                f"Maximum lookback is {BACKTESTING_MAX_LOOKBACK_DAYS} days."
            )

    def _calc_mfi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate the Money Flow Index (MFI)."""
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
        money_flow = typical_price * data["Volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_mf = positive_flow.rolling(window=self.mfi_period).sum()
        negative_mf = negative_flow.rolling(window=self.mfi_period).sum()

        money_flow_ratio = positive_mf / (negative_mf + 1e-9)
        mfi = 100 - (100 / (1 + money_flow_ratio))
        return mfi

    def _find_extrema(self, series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Find local maxima and minima."""
        maximas = argrelextrema(
            series.values, np.greater_equal, order=self.local_extrema_window
        )[0]
        minimas = argrelextrema(
            series.values, np.less_equal, order=self.local_extrema_window
        )[0]
        return maximas, minimas

    def _get_divergence_signals(
        self, close: pd.Series, mfi: pd.Series, maximas: np.ndarray, minimas: np.ndarray
    ) -> pd.Series:
        """Get divergence signals based on MFI and price extrema."""
        signal = pd.Series(index=close.index, data=0.0)

        # Bullish divergence: price makes lower low, MFI makes higher low
        for i in range(1, len(minimas)):
            if (
                close.iloc[minimas[i]] < close.iloc[minimas[i - 1]]
                and mfi.iloc[minimas[i]] > mfi.iloc[minimas[i - 1]]
            ):
                signal.iloc[minimas[i]] = 1.0

        # Bearish divergence: price makes higher high, MFI makes lower high
        for i in range(1, len(maximas)):
            if (
                close.iloc[maximas[i]] > close.iloc[maximas[i - 1]]
                and mfi.iloc[maximas[i]] < mfi.iloc[maximas[i - 1]]
            ):
                signal.iloc[maximas[i]] = -1.0

        return signal

    def generate_historical_scores(self, data: pd.DataFrame, symbol: str) -> pd.Series:
        """Generate historical MFI divergence scores."""
        self.validate_data(data)

        data_sorted = data.sort_values("datetime").copy()
        close = data_sorted["Close"]

        mfi = self._calc_mfi(data_sorted)
        maximas, minimas = self._find_extrema(close)
        divergence_signals = self._get_divergence_signals(close, mfi, maximas, minimas)

        # Optional: Smooth or decay signal over a few candles
        divergence_signals = divergence_signals.ffill(limit=3).fillna(0)

        return pd.Series(
            data=divergence_signals.clip(-1, 1).values,
            index=pd.to_datetime(data_sorted["datetime"]),
        )

    def generate_prediction_score(self, data: pd.DataFrame, symbol: str) -> float:
        """Generate the prediction score for the latest candle."""
        return self.generate_historical_scores(data, symbol).iloc[-1]
