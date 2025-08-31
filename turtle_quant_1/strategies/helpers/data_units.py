import pandas as pd


class DataUnitConverter:
    @staticmethod
    def convert_to_daily_data(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
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
            symbol: The symbol of the data.
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

    @staticmethod
    def convert_to_weekly_data(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
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
            symbol: The symbol of the data.
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

    @staticmethod
    def convert_to_yearly_data(symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
