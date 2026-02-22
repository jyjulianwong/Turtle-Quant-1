import numpy as np
import pandas as pd


def round_to_sig_fig(x: list[float], p: int) -> list[float]:
    """Round a list of numbers to a specified number of significant figures.

    Args:
        x: The list of numbers to round.
        p: The number of significant figures to round to.

    Returns:
        The rounded list of numbers.
    """
    x = np.asarray(x, dtype=float)
    x_pos = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    x_mag = 10 ** (p - 1 - np.floor(np.log10(x_pos)))
    result = np.round(x * x_mag) / x_mag
    return result.tolist()


def calc_atr_value(
    data: pd.DataFrame,
    lookback: int = 14,
    ema: bool = True,
    return_log_space: bool = False,
) -> float:
    """
    Calculate ATR (Average True Range) in either log-return or price space.

    Args:
        data: DataFrame with OHLCV data.
        lookback: Lookback period for ATR calculation.
        ema: If True, use exponential moving average; else simple moving average.
        return_log_space: If True, compute ATR in log-return space.

    Returns:
        Latest ATR value (NaN if insufficient data).
    """
    if not {"High", "Low", "Close"}.issubset(data.columns):
        raise ValueError("DataFrame must contain 'High', 'Low', and 'Close' columns.")

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    if return_log_space:
        # pyrefly: ignore
        high: pd.Series = np.log(high)
        # pyrefly: ignore
        low: pd.Series = np.log(low)
        # pyrefly: ignore
        close: pd.Series = np.log(close)

    prev_close = close.shift(1)

    tr = np.maximum.reduce(  # pyrefly: ignore[no-matching-overload]
        [
            (high - low).abs().values,
            (high - prev_close).abs().values,
            (low - prev_close).abs().values,
        ]
    )

    tr_series = pd.Series(tr, index=data.index)

    # Smoothing: EMA or SMA
    if ema:
        atr = tr_series.ewm(span=lookback, adjust=False).mean()
    else:
        atr = tr_series.rolling(window=lookback).mean()

    latest_value = atr.iloc[-1]
    return float(latest_value) if pd.notna(latest_value) else np.nan


def get_wick_direction(row: pd.Series) -> int:
    """Determine the direction of the candlestick wick for a single row of a DataFrame.

    Args:
        row: The row to check.

    Returns:
        The direction of the wick.
    """

    upper_wick = row["High"] - max(row["Close"], row["Open"])
    lower_wick = min(row["Close"], row["Open"]) - row["Low"]

    if upper_wick > lower_wick * 1.2:
        return +1  # "up"
    if lower_wick > upper_wick * 1.2:
        return -1  # "down"
    return 0  # "neutral"


def get_wick_directions_vecd(data: pd.DataFrame) -> pd.Series:
    """Vectorized version of get_wick_direction for entire DataFrame.

    Args:
        data: DataFrame with OHLC data

    Returns:
        Series with wick directions: +1 for "up", -1 for "down", 0 for "neutral"
    """
    # Calculate upper and lower wicks vectorized
    upper_wick = data["High"] - pd.DataFrame(
        {"Close": data["Close"], "Open": data["Open"]}
    ).max(axis=1)
    lower_wick = (
        pd.DataFrame({"Close": data["Close"], "Open": data["Open"]}).min(axis=1)
        - data["Low"]
    )

    # Vectorized direction logic
    wick_direction = pd.Series(0, index=data.index)  # Default to neutral (0)
    wick_direction.loc[upper_wick > lower_wick * 1.2] = +1  # "up"
    wick_direction.loc[lower_wick > upper_wick * 1.2] = -1  # "down"

    return wick_direction
