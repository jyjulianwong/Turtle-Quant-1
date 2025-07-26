"""Base classes for data processing components."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

import pandas as pd


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
        start_date: datetime,
        end_date: datetime,
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


class BaseDataProcessor(ABC):
    """Base class for data processors."""

    @abstractmethod
    def update_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Update data for a symbol.

        Args:
            symbol: Symbol to update data for.
            start_date: Start date for the data.
            end_date: End date for the data.
        """
        pass


class BaseDataMaintainer(ABC):
    """Base class for data maintainers."""

    @abstractmethod
    def __init__(
        self,
        processor: Optional[BaseDataProcessor] = None,
        storage: Optional[BaseDataStorageAdapter] = None,
        symbols: Optional[List[str]] = None,
    ):
        """Initialize the data maintainer.

        Args:
            processor: BaseDataProcessor instance.
            storage: BaseDataStorage instance.
            symbols: List of symbols to maintain.
        """
        pass

    @abstractmethod
    def ensure_continuous_data(
        self,
        symbol: str,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Ensure continuous data for a symbol."""
        pass

    @abstractmethod
    def ensure_all_continuous_data(
        self,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Ensure continuous data for all symbols."""
        pass
