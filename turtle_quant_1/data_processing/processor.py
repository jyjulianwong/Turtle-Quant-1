"""Data processor for orchestrating data fetching and storage."""

from datetime import datetime, timedelta
from typing import List, Optional, Type

import pandas as pd

from turtle_quant_1.config import MAX_HISTORY_YEARS, SYMBOLS
from turtle_quant_1.data_processing.base import (
    BaseDataFetcher,
    BaseDataProcessor,
    BaseDataStorageAdapter,
)
from turtle_quant_1.data_processing.gcs_storage_adapter import GCSDataStorageAdapter
from turtle_quant_1.data_processing.yfinance_fetcher import YFinanceDataFetcher


class DataProcessor(BaseDataProcessor):
    """Data processor for orchestrating data fetching and storage."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        fetcher_type: Optional[Type[BaseDataFetcher]] = None,
        storage: Optional[BaseDataStorageAdapter] = None,
    ):
        """Initialize the data processor.

        Args:
            symbols: List of symbols to process. If None, uses SYMBOLS from config.
            fetcher_type: Data fetcher class to use. If None, uses YFinanceDataFetcher.
            storage: Data storage to use. If None, uses GCSDataStorage.
        """
        self.symbols = symbols or SYMBOLS
        self.fetcher_type = fetcher_type or YFinanceDataFetcher
        self.storage = storage or GCSDataStorageAdapter()

        # Initialize fetcher
        self.fetcher = self.fetcher_type(self.symbols)

    def update_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Update data for a symbol.

        Args:
            symbol: Symbol to update data for.
            start_date: Start date for the data. If None, uses MAX_HISTORY_YEARS ago.
            end_date: End date for the data. If None, uses current time.
        """
        # Set default dates if not provided
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365 * MAX_HISTORY_YEARS))

        # Check for existing data
        try:
            existing_data = self.storage.load_ohlcv(symbol)
        except Exception:
            # If loading fails, assume no existing data
            existing_data = pd.DataFrame()

        # Fetch data
        df = self.fetcher.fetch_hourly_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        # Combine with existing data if any
        if not existing_data.empty and not df.empty:
            # Convert datetime columns to same type for concatenation
            if "datetime" in existing_data.columns and "datetime" in df.columns:
                existing_data["datetime"] = pd.to_datetime(existing_data["datetime"])
                df["datetime"] = pd.to_datetime(df["datetime"])

                # Remove duplicates by datetime and combine
                combined_data = pd.concat([existing_data, df], ignore_index=True)
                combined_data = combined_data.drop_duplicates(
                    subset=["datetime"], keep="last"
                )
                combined_data = combined_data.sort_values("datetime").reset_index(
                    drop=True
                )
                df = combined_data

        # Save data
        self.storage.save_ohlcv(symbol=symbol, data=df)

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

        Returns:
            DataFrame with OHLCV data.
        """
        return self.storage.load_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    def update_all_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Update data for all symbols.

        Args:
            start_date: Start date for the data. If None, uses MAX_HISTORY_YEARS ago.
            end_date: End date for the data. If None, uses current time.
        """
        for symbol in self.symbols:
            self.update_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
