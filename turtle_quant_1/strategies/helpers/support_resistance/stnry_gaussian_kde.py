"""Stationary Gaussian KDE support and resistance strategy."""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
from sklearn.preprocessing import MultiLabelBinarizer

from .base import BaseSupResStrategy


class StnryGaussianKDE(BaseSupResStrategy):
    """Stationary Gaussian KDE support and resistance strategy.

    Refer to:
    - https://github.com/neurotrader888/TechnicalAnalysisAutomation/blob/main/mp_support_resist.py
    - https://www.youtube.com/watch?v=mNWPSFOVoYA

    This strategy identifies support and resistance levels by using Gaussian KDE to
    identify significant peaks in the market profile.
    """

    def __init__(self):
        """Initialize the StnryGaussianKDE strategy."""
        super().__init__()

    def _round_to_sig_fig(self, x, p):
        """Round a list of numbers to a specified number of significant figures."""
        x = np.asarray(x)
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
        mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

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
        lookback: int = 360,  # TODO: Respect CANDLE_UNIT.
        first_w: float = 0.01,
        atr_mult: float = 3.00,
        prom_thresh: float = 0.25,
    ):
        """
        Calculate the support and resistance levels for a given DataFrame.

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
        atr = self._calc_log_atr(data)

        level_values = [[]] * len(data)
        for i in range(lookback, len(data)):
            i_start = i - lookback
            vals = np.log(data.iloc[i_start + 1 : i + 1]["Close"].to_numpy())
            levels, peaks, props, price_range, price_pdf, weights = (
                self._calc_kdf_values(vals, atr, first_w, atr_mult, prom_thresh)
            )
            # Round values to reduce number of unique values
            # This is needed for the MultiLabelBinarizer to work later on
            level_values[i] = self._round_to_sig_fig(levels, 4)

        return level_values

    def generate_historical_levels(
        self, data: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Generate Gaussian KDE levels for the entire dataset duration.

        The method intermediately transforms the output from _calc_sup_res_levels() to the following format:
        ```
             datetime_beg              datetime_end                   316.0  318.3  318.4  318.5  ...
        0    2023-08-31 09:30:00-04:00 2023-08-31 10:30:00-04:00      0      0      0      0      ...
        1    2023-08-31 10:30:00-04:00 2023-08-31 11:30:00-04:00      0      0      0      0      ...
        2    2023-08-31 11:30:00-04:00 2023-08-31 12:30:00-04:00      0      0      0      0      ...
        3    2023-08-31 12:30:00-04:00 2023-08-31 13:30:00-04:00      0      0      0      0      ...
        4    2023-08-31 13:30:00-04:00 2023-08-31 14:30:00-04:00      0      0      0      0      ...
             ...                       ...                            ...    ...    ...    ...    ...
        ```

        It then transforms it back to the expected format:
        ```
            datetime_beg               datetime_end                level_values
        0   2023-09-05 10:30:00-04:00  2023-09-08 14:30:00-04:00   [316.0]
        1   2023-09-12 09:30:00-04:00  2023-09-15 15:30:00-04:00   [318.3]
            ...                        ...                         ...
        ```

        This reduces the number of rows in the DataFrame,
        and increases the continuous duration of each level value when visualized.

        Args:
            data: DataFrame with OHLCV data containing columns:
                  ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            symbol: The symbol being analyzed.

        Returns:
            DataFrame with one row containing all Fibonacci retracement levels
            for the entire dataset duration.
            The columns are: ['datetime_beg', 'datetime_end', 'level_values'].
        """
        level_values = self._calc_sup_res_levels(data)

        # Create output DataFrame - single row since these are static levels
        result = pd.DataFrame(
            {
                "datetime_beg": pd.to_datetime(data.iloc[:-1]["datetime"]).reset_index(
                    drop=True
                ),
                "datetime_end": pd.to_datetime(data.iloc[+1:]["datetime"]).reset_index(
                    drop=True
                ),
                "level_values": level_values[1:],
            }
        )

        mlb = MultiLabelBinarizer(sparse_output=True)

        result = result.join(
            # pyrefly: ignore
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(result.pop("level_values")),
                index=result.index,
                columns=mlb.classes_,
            )
        )

        numeric_cols = result.select_dtypes(include="number").columns

        result_rows = []
        for col in numeric_cols:
            nonzero_mask = result[col] != 0
            if nonzero_mask.any():
                min_dt = result.loc[nonzero_mask, "datetime_beg"].min()
                max_dt = result.loc[nonzero_mask, "datetime_end"].max()
                result_rows.append(
                    {
                        "datetime_beg": min_dt,
                        "datetime_end": max_dt,
                        "level_values": [col],
                    }
                )

        result = pd.DataFrame(result_rows)

        return result
