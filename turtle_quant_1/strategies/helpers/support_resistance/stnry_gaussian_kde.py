"""Stationary Gaussian KDE support and resistance strategy."""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from turtle_quant_1.config import CANDLE_UNIT
from turtle_quant_1.strategies.helpers.candle_units import convert_units
from turtle_quant_1.strategies.helpers.helpers import calc_atr_value, round_to_sig_fig

from .base import BaseSupResStrategy


class StnryGaussianKde(BaseSupResStrategy):
    """Stationary Gaussian KDE support and resistance strategy.

    Refer to:
    - https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/mp_support_resist.py
    - https://www.youtube.com/watch?v=mNWPSFOVoYA

    This strategy identifies support and resistance levels by using Gaussian KDE to
    identify significant peaks in the market profile.
    """

    def __init__(self):
        """Initialize the StnryGaussianKde strategy."""
        super().__init__()

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
        price_pdf = kernal(price_range)  # Market profile

        # Find significant peaks in the market profile
        price_pdf_max = np.max(price_pdf)
        prom_min = price_pdf_max * prom_thresh

        # pyrefly: ignore
        peaks, props = scipy_signal.find_peaks(price_pdf, prominence=prom_min)
        levels = []
        for peak in peaks:
            levels.append(np.exp(price_range[peak]))

        return levels, peaks, props, price_range, price_pdf, weights

    def _calc_sup_res_levels(
        self,
        data: pd.DataFrame,
        lookback: int = convert_units(2, "MONTH", CANDLE_UNIT),
        first_w: float = 0.01,
        atr_mult: float = 3.00,
        prom_thresh: float = 0.25,
    ) -> list[np.ndarray]:
        """
        Calculate the support and resistance levels for a given DataFrame.

        The algorithm is inherently autoregressive and mitigates look-ahead bias.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            lookback: Lookback period for the ATR calculation.
            first_w: First weight for the Gaussian KDE.
            atr_mult: Multiplier for the ATR.
            prom_thresh: Prominence threshold for the Gaussian KDE.

        Returns:
            List of support and resistance levels for each timestamp in the dataset.
        """
        # Get log average true range
        log_atr = calc_atr_value(
            data=data,
            lookback=lookback,
            return_log_space=True,
        )

        level_values = [np.full(128, 0.0) for _ in range(len(data))]
        # TODO: Vectorize.
        for i in range(lookback, len(data)):
            i_start = i - lookback
            log_prices = np.log(data.iloc[i_start + 1 : i + 1]["Close"].to_numpy())
            levels, peaks, props, price_range, price_pdf, weights = (
                self._calc_kdf_values(
                    log_prices, log_atr, first_w, atr_mult, prom_thresh
                )
            )
            # Round values to reduce number of unique values
            # This is needed for the MultiLabelBinarizer to work later on
            rounded_levels = round_to_sig_fig(levels, 4)
            # Create fixed-size array with padding
            fixed_array = np.full(128, 0.0)
            fixed_array[: len(rounded_levels)] = rounded_levels
            level_values[i] = fixed_array

        return level_values

    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate Gaussian KDE levels for historical data.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with Gaussian KDE levels.
            The columns are: ['datetime', 'level_values'].
        """
        level_values = self._calc_sup_res_levels(data)

        # Create output DataFrame with 1-to-1 mapping to original data
        result = pd.DataFrame(
            {
                "datetime": pd.to_datetime(data["datetime"]).reset_index(drop=True),
                "level_values": level_values,
            }
        )

        return result
