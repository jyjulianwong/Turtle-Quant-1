from typing import Literal

import pandas as pd


def get_wick_direction(row: pd.Series) -> Literal["up", "down", "neutral"]:
    """Determine the direction of the candlestick wick.

    Args:
        row: The row to check.

    Returns:
        The direction of the wick.
    """

    upper_wick = row["High"] - max(row["Close"], row["Open"])
    lower_wick = min(row["Close"], row["Open"]) - row["Low"]

    if upper_wick > lower_wick * 1.2:
        return "up"
    if lower_wick > upper_wick * 1.2:
        return "down"
    return "neutral"


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

    # Use pandas vectorized aggregation - much more efficient than loops
    daily_df = (
        data.groupby("date")
        .agg(
            {
                "datetime": "last",  # Last timestamp of the day, preserving timezone data
                "Open": "first",  # First open of the day
                "High": "max",  # Highest high of the day
                "Low": "min",  # Lowest low of the day
                "Close": "last",  # Last close of the day
                "Volume": "sum",  # Total volume of the day
            }
        )
        .reset_index()
    )

    # Drop the date column as we now have the full timestamp
    daily_df = daily_df.drop("date", axis=1)

    # Preserve original indices by using the last index of each group
    # Get the last index for each date group (avoiding deprecated behavior)
    # pyrefly: ignore
    last_indices = data.groupby("date", group_keys=False).apply(
        lambda x: x.index[-1], include_groups=False
    )
    daily_df.index = last_indices.values

    # Reorder columns to match expected format
    daily_df = daily_df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

    return daily_df
