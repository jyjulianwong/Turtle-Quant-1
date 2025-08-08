"""Stationary Gaussian KDE support and resistance strategy."""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from turtle_quant_1.strategies.helpers.helpers import convert_to_daily_data

from .base import BaseSupResStrategy


class StnryGaussianKDE(BaseSupResStrategy):
    """Stationary Gaussian KDE support and resistance strategy.

    Refer to https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/mp_support_resist.py.

    This strategy identifies support and resistance levels by using Gaussian KDE to
    identify significant peaks in the market profile.
    """

    def __init__(self):
        """Initialize the StnryGaussianKDE strategy."""
        super().__init__()

    def _calc_log_atr(
        self, df: pd.DataFrame, period: int = 14, ema: bool = False
    ) -> float:
        """
        Calculate the Logarithmic Average True Range (Log ATR) for an OHLCV DataFrame.

        Parameters:
        - df (pd.DataFrame): Must include columns 'High', 'Low', 'Close'.
        - period (int): Lookback period for the ATR calculation.
        - ema (bool): If True, use Exponential Moving Average. If False, use Simple Moving Average.

        Returns:
        - pd.Series: Log ATR values.
        """
        if not {"High", "Low", "Close"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain 'High', 'Low', and 'Close' columns."
            )

        # pyrefly: ignore
        log_high: pd.Series = np.log(df["High"])
        # pyrefly: ignore
        log_low: pd.Series = np.log(df["Low"])
        # pyrefly: ignore
        log_close: pd.Series = np.log(df["Close"])
        # pyrefly: ignore
        prev_log_close: pd.Series = log_close.shift(1)

        # Calculate the log-based True Range
        log_tr = np.maximum.reduce(
            # pyrefly: ignore
            [
                (log_high - log_low).values,
                (log_high - prev_log_close).abs().values,
                (log_low - prev_log_close).abs().values,
            ]
        )

        log_tr_series = pd.Series(log_tr, index=df.index)

        # Compute ATR using SMA or EMA
        if ema:
            log_atr = log_tr_series.ewm(span=period, adjust=False).mean()
        else:
            log_atr = log_tr_series.rolling(window=period).mean()

        latest_value = log_atr.iloc[-1]
        return float(latest_value) if pd.notna(latest_value) else np.nan

    def _calc_kdf_values(
        self,
        price: np.ndarray,
        atr: float,  # Log closing price, and log atr
        first_w: float = 0.1,
        atr_mult: float = 3.0,
        prom_thresh: float = 0.1,
    ):
        if len(price) < 2 or np.all(price == price[0]):
            raise ValueError("Price array must contain at least 2 distinct values.")

        # Set up weights
        last_w = 1.0
        w_step = (last_w - first_w) / (len(price) - 1) if len(price) > 1 else 0.0
        weights = first_w + np.arange(len(price)) * w_step
        weights[weights < 0] = 0.0
        bandwidth = max(atr * atr_mult, 1e-6)

        # Get kernel of price
        kernal = scipy_stats.gaussian_kde(
            dataset=price, bw_method=bandwidth, weights=weights
        )

        # Construct market profile
        min_v = np.min(price)
        max_v = np.max(price)
        step = (max_v - min_v) / 200
        price_range = np.arange(min_v, max_v, step).astype(float)
        pdf = kernal(price_range)  # Market profile

        # Find significant peaks in the market profile
        pdf_max = np.max(pdf)
        prom_min = pdf_max * prom_thresh

        # pyrefly: ignore
        peaks, props = scipy_signal.find_peaks(pdf, prominence=prom_min)
        levels = []
        for peak in peaks:
            levels.append(np.exp(price_range[peak]))

        return levels, peaks, props, price_range, pdf, weights

    def _calc_sup_res_levels(
        self,
        data: pd.DataFrame,
        atr: float,
        first_w: float = 0.01,
        atr_mult: float = 3.00,
        prom_thresh: float = 0.25,
    ):
        vals = np.log(data["Close"].to_numpy())
        levels, peaks, props, price_range, pdf, weights = self._calc_kdf_values(
            vals, atr, first_w, atr_mult, prom_thresh
        )
        return levels

    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate Gaussian KDE levels for the entire dataset duration.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with one row containing all Fibonacci retracement levels
            for the entire dataset duration.
            The columns are: ['datetime_start', 'datetime_end', 'values'].
        """
        daily_data = convert_to_daily_data(data)

        # Get log average true range
        atr = self._calc_log_atr(daily_data)
        level_values = self._calc_sup_res_levels(daily_data, atr)

        # Create output DataFrame - single row since these are static levels
        result = pd.DataFrame(
            {
                "datetime_start": [
                    data["datetime"].iloc[0]
                    if "datetime" in data.columns
                    else data.index[0]
                ],
                "datetime_end": [
                    data["datetime"].iloc[-1]
                    if "datetime" in data.columns
                    else data.index[-1]
                ],
                "values": [level_values],
            }
        )

        return result
