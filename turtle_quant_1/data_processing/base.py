"""Base classes for data processing components."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pandas as pd


class BaseDataStorageAdapter(ABC):
    """Base class for data storage."""

    @abstractmethod
    def save_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
    ) -> None:
        """Save OHLCV data for a symbol.

        Args:
            symbol: Symbol to save data for.
            data: DataFrame with OHLCV data.
        """
        pass

    @abstractmethod
    def load_ohlcv(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data for a symbol.

        Args:
            symbol: Symbol to load data for.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            DataFrame with OHLCV data.
        """
        pass

    @abstractmethod
    def delete_data(self, symbol: str) -> None:
        """Delete data for a symbol from storage.

        Args:
            symbol: Symbol to delete data for.
        """
        pass


class BaseDataFetcher(ABC):
    """Base class for data fetchers."""

    def __init__(self, symbols: List[str]):
        """Initialize the data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        self.symbols = symbols

    @abstractmethod
    def fetch_hourly_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly OHLCV data for a symbol.

        Args:
            symbol: Symbol to fetch data for.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            DataFrame with OHLCV data.
        """
        pass


class BaseDataProcessor(ABC):
    """Base class for data processors."""

    @abstractmethod
    def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Load data for a symbol.

        Args:
            symbol: Symbol to load data for.
            start_date: Start date for the data.
            end_date: End date for the data.
        """
        pass

    @abstractmethod
    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data for a symbol.

        Args:
            symbol: Symbol to save data for.
            data: DataFrame with data.
        """
        pass


class BaseDataMaintainer(ABC):
    """Base class for data maintainers."""

    @abstractmethod
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        fetcher: Optional[BaseDataFetcher] = None,
    ):
        """Initialize the data maintainer.

        Args:
            symbols: List of symbols to maintain.
            fetcher: Data fetcher to use. If None, uses YFinanceDataFetcher.
        """
        pass

    @abstractmethod
    def impute_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Ensure continuous data for a symbol.

        Args:
            symbol: Symbol to impute data for.
            data: DataFrame with data.
            end_date: End date to ensure data up to. If None, uses current time.

        Returns:
            DataFrame with updated data.
        """
        pass
