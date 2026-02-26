import logging
from typing import Literal

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from turtle_quant_1.strategies.helpers.multiprocessing import FileCache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Global cache instance
_global_cache = FileCache()

_Freq = Literal["15M", "30M", "1H", "2H", "4H", "1D", "1W"]


def get_global_cache():
    """Get the global cache instance."""
    return _global_cache


def _resample_ohlcv(
    data: pd.DataFrame,
    grouper: pd.Grouper,
) -> tuple[pd.DataFrame, DataFrameGroupBy]:
    """Aggregate OHLCV data using a pandas Grouper.

    Returns the aggregated DataFrame and the group object so the caller can
    retrieve last-index information for index preservation.
    """
    groups = data.groupby(grouper)

    agg_df = groups.agg(
        {
            "datetime": "last",
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    agg_df = agg_df.loc[agg_df["datetime"].notna()]

    # pyrefly: ignore
    last_indices = groups.apply(
        lambda x: x.index[-1] if len(x) > 0 else None, include_groups=False
    )
    agg_df.index = last_indices.values[last_indices.notna()]
    agg_df = agg_df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

    return agg_df, groups


class DataUnitConverter:
    @classmethod
    def _get_cache_key(cls, symbol: str, freq: _Freq) -> str:
        return f"{symbol}_DataUnitConverter_{freq}"

    @classmethod
    def preload_global_instance_cache(
        cls, symbol: str, data: pd.DataFrame, freq: _Freq
    ) -> None:
        cache_key = cls._get_cache_key(symbol, freq)
        conversion_func = {
            "15M": cls.convert_to_15m_data,
            "30M": cls.convert_to_30m_data,
            "1H": cls.convert_to_1h_data,
            "2H": cls.convert_to_2h_data,
            "4H": cls.convert_to_4h_data,
            "1D": cls.convert_to_1d_data,
            "1W": cls.convert_to_1w_data,
        }.get(freq)

        if conversion_func is None:
            raise ValueError(f"Invalid frequency provided: {freq}")

        agg_data = conversion_func(symbol, data)
        get_global_cache().set(cache_key, agg_data)

        logger.info(f"Preloaded {freq} OHLCV data for {symbol} into cache")

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    @classmethod
    def _resample_with_datetime_grouper(
        cls,
        symbol: str,
        data: pd.DataFrame,
        freq: _Freq,
        pandas_freq: str,
    ) -> pd.DataFrame:
        """Resample 5M data to an intraday or higher frequency using pd.Grouper."""
        if data.empty:
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

        cache_key = cls._get_cache_key(symbol, freq)
        cached_data = get_global_cache().get(cache_key)
        if cached_data is not None:
            logger.debug(f"{cache_key} already exists. Using cached data...")
            return cached_data.loc[cached_data.index.intersection(data.index)]

        data = data.copy()
        data["datetime"] = pd.to_datetime(data["datetime"])

        grouper = pd.Grouper(key="datetime", freq=pandas_freq)
        agg_df, _ = _resample_ohlcv(data, grouper)

        return agg_df

    # ------------------------------------------------------------------
    # Public conversion methods
    # ------------------------------------------------------------------

    @classmethod
    def convert_to_15m_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to 15-minute bars."""
        return cls._resample_with_datetime_grouper(symbol, data, "15M", "15min")

    @classmethod
    def convert_to_30m_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to 30-minute bars."""
        return cls._resample_with_datetime_grouper(symbol, data, "30M", "30min")

    @classmethod
    def convert_to_1h_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to 1-hour bars."""
        return cls._resample_with_datetime_grouper(symbol, data, "1H", "1h")

    @classmethod
    def convert_to_2h_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to 2-hour bars."""
        return cls._resample_with_datetime_grouper(symbol, data, "2H", "2h")

    @classmethod
    def convert_to_4h_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to 4-hour bars."""
        return cls._resample_with_datetime_grouper(symbol, data, "4H", "4h")

    @classmethod
    def convert_to_1d_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to daily bars.

        The indices of the original DataFrame will be retained,
        i.e. the row number of a day will be the row number of the last
        timestamp of the day, so the row numbers will not be contiguous.

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

        cache_key = cls._get_cache_key(symbol, "1D")
        cached_data = get_global_cache().get(cache_key)
        if cached_data is not None:
            logger.debug(f"{cache_key} already exists. Using cached data...")
            return cached_data.loc[cached_data.index.intersection(data.index)]

        data = data.copy()
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["date"] = data["datetime"].dt.date

        daily_groups = data.groupby(pd.Grouper(key="date"))

        daily_df = daily_groups.agg(
            {
                "datetime": "last",
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

        # pyrefly: ignore
        last_indices = daily_groups.apply(lambda x: x.index[-1], include_groups=False)
        daily_df.index = last_indices.values
        daily_df = daily_df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

        return daily_df

    @classmethod
    def convert_to_1w_data(cls, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Convert 5M OHLCV data to weekly bars (week ending on Friday).

        The indices of the original DataFrame will be retained,
        i.e. the row number of a week will be the row number of the last
        timestamp of the week, so the row numbers will not be contiguous.

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

        cache_key = cls._get_cache_key(symbol, "1W")
        cached_data = get_global_cache().get(cache_key)
        if cached_data is not None:
            logger.debug(f"{cache_key} already exists. Using cached data...")
            return cached_data.loc[cached_data.index.intersection(data.index)]

        data = data.copy()
        data["datetime"] = pd.to_datetime(data["datetime"])

        weekly_groups = data.groupby(pd.Grouper(key="datetime", freq="W-FRI"))

        weekly_df = weekly_groups.agg(
            {
                "datetime": "last",
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

        # pyrefly: ignore
        last_indices = weekly_groups.apply(lambda x: x.index[-1], include_groups=False)
        weekly_df.index = last_indices.values
        weekly_df = weekly_df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

        return weekly_df
