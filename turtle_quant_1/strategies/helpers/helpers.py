import numpy as np
import pandas as pd


def round_to_sig_fig(x, p):
    """Round a list of numbers to a specified number of significant figures."""
    x = np.asarray(x, dtype=float)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


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

    tr = np.maximum.reduce(
        # pyrefly: ignore
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


def get_wick_directions_vectorized(data: pd.DataFrame) -> pd.Series:
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


def convert_to_daily_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLC data to daily data.

    The daily data will have the following columns:
    - datetime: The date of the day.
    - Open: The open price of the day.
    - High: The high price of the day.
    - Low: The low price of the day.
    - Close: The close price of the day.
    - Volume: The volume of the day.

    The indices of the originalDataFrame will be retained,
    i.e. the row number of a day will be the row number of the last timestamp of the day,
    meaning the row numbers will not be contiguous.

    Args:
        data: The data to convert.

    Returns:
        The converted data.
    """
    if data.empty:
        return pd.DataFrame(
            columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
        )

    # Ensure datetime column is datetime type
    data = data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])

    # Extract date from datetime for grouping
    data["date"] = data["datetime"].dt.date

    daily_groups = data.groupby(pd.Grouper(key="date"))

    # Use pandas vectorized aggregation - much more efficient than loops
    daily_df = daily_groups.agg(
        {
            "datetime": "last",  # Last timestamp of the day, preserving timezone data
            "Open": "first",  # First open of the day
            "High": "max",  # Highest high of the day
            "Low": "min",  # Lowest low of the day
            "Close": "last",  # Last close of the day
            "Volume": "sum",  # Total volume of the day
        }
    )

    # Preserve original indices by using the last index of each group
    # Get the last index for each date group (avoiding deprecated behavior)
    # pyrefly: ignore
    last_indices = daily_groups.apply(lambda x: x.index[-1], include_groups=False)
    daily_df.index = last_indices.values

    # Reorder columns to match expected format
    daily_df = daily_df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

    return daily_df


def convert_to_weekly_data(data: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLC data to weekly data.

    The weekly data will have the following columns:
    - datetime: The date of the week.
    - Open: The open price of the week.
    - High: The high price of the week.
    - Low: The low price of the week.
    - Close: The close price of the week.
    - Volume: The volume of the week.

    The indices of the originalDataFrame will be retained,
    i.e. the row number of a week will be the row number of the last timestamp of the week,
    meaning the row numbers will not be contiguous.

    Args:
        data: The data to convert.

    Returns:
        The converted data.
    """
    if data.empty:
        return pd.DataFrame(
            columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
        )

    # Ensure datetime column is datetime type
    data = data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])

    # Group by week ending on Friday, preserving timezone info
    weekly_groups = data.groupby(pd.Grouper(key="datetime", freq="W-FRI"))

    # Use pandas vectorized aggregation - much more efficient than loops
    weekly_df = weekly_groups.agg(
        {
            "datetime": "last",  # Last timestamp of the week, preserving timezone data
            "Open": "first",  # First open of the week
            "High": "max",  # Highest high of the week
            "Low": "min",  # Lowest low of the week
            "Close": "last",  # Last close of the week
            "Volume": "sum",  # Total volume of the week
        }
    )

    # Rename the datetime index back to datetime column
    weekly_df = weekly_df.rename(columns={"datetime": "datetime"})

    # Preserve original indices by using the last index of each group
    # Get the last index for each week group (avoiding deprecated behavior)
    # pyrefly: ignore
    last_indices = weekly_groups.apply(lambda x: x.index[-1], include_groups=False)
    weekly_df.index = last_indices.values

    # Reorder columns to match expected format
    weekly_df = weekly_df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

    return weekly_df


def convert_to_yearly_data(data: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError()
