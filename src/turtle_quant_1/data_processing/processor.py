"""Data processor for orchestrating data fetching and storage."""

from datetime import datetime
from typing import List, Optional

import pandas as pd

from turtle_quant_1.config import LIVE_SYMBOLS
from turtle_quant_1.data_processing.adapters.gcs_storage_adapter import (
    GCSDataStorageAdapter,
)
from turtle_quant_1.data_processing.adapters.yfinance_fetcher import YFinanceDataFetcher
from turtle_quant_1.data_processing.base import (
    BaseDataFetcher,
    BaseDataMaintainer,
    BaseDataProcessor,
    BaseDataStorageAdapter,
)
from turtle_quant_1.data_processing.datetimes import get_expected_market_hours_bounds
from turtle_quant_1.data_processing.maintainer import DataMaintainer
from turtle_quant_1.logging import get_logger

logger = get_logger(__name__)

# !!!: Dangerous setting. Deletes and resets all data from cache.
DANGER_CLEAR_CACHE = False
# !!!: Dangerous setting. Deletes and resets all data from storage.
DANGER_CLEAR_STORAGE = False


class DataProcessor(BaseDataProcessor):
    """Data processor for orchestrating data fetching and storage."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        storage: Optional[BaseDataStorageAdapter] = None,
        fetcher: Optional[BaseDataFetcher] = None,
        maintainer: Optional[BaseDataMaintainer] = None,
    ):
        """Initialize the data processor.

        Args:
            symbols: List of symbols to process. If None, uses SYMBOLS from config.
            storage: Data storage to use. If None, uses GCSDataStorage.
            fetcher: Data fetcher to use. If None, uses YFinanceDataFetcher.
            maintainer: Data maintainer to use. If None, uses DataMaintainer.
        """
        self.symbols = symbols or LIVE_SYMBOLS

        self.storage = storage or GCSDataStorageAdapter()
        self.fetcher = fetcher or YFinanceDataFetcher(symbols=self.symbols)
        self.maintainer = maintainer or DataMaintainer(
            symbols=self.symbols, fetcher=self.fetcher
        )

        self.data_cache: dict[str, pd.DataFrame] = {}

    def load_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        impute_data: bool = True,
    ) -> pd.DataFrame:
        """Load data for a symbol.

        Auto-corrects start and end dates to be at the bounds of market hours for those days.

        Args:
            symbol: Symbol to load data for.
            start_date: Timezone-aware start date to fetch data from.
            end_date: Timezone-aware end date to fetch data up to.
            impute_data: Whether to impute data.

        Returns:
            DataFrame with OHLCV data.
        """
        if DANGER_CLEAR_CACHE:
            self.data_cache.pop(symbol, None)
        if DANGER_CLEAR_STORAGE:
            self.storage.delete_ohlcv_data(symbol)
            self.data_cache.pop(symbol, None)

        is_data_updated = False

        _start_date, _end_date = get_expected_market_hours_bounds(
            symbol, start_date, end_date
        )

        if symbol in self.data_cache:
            df = self.data_cache[symbol]
        else:
            logger.warning(
                f"No data found for {symbol} in cache. Fetching from {self.storage}..."
            )
            df = self.storage.load_ohlcv_data(
                symbol=symbol,
                start_date=None,  # TODO: Work out why.
                end_date=None,  # TODO: Work out why.
            )

        if df.empty:
            logger.warning(
                f"No data found for {symbol}. Fetching from {self.fetcher}..."
            )
            # TODO: Respect CANDLE_UNIT.
            df = self.fetcher.fetch_5min_ohlcv(
                symbol=symbol,
                start_date=_start_date,
                end_date=_end_date,
            )
            is_data_updated = True

        if impute_data:
            _df = self.maintainer.impute_data(symbol, df, _end_date)
            if not _df.index.equals(df.index):
                is_data_updated = True
            df = _df

        # Save updated data in relevant locations
        self.data_cache[symbol] = df
        if is_data_updated:
            logger.info(
                f"Data has been changed. Updating data for {symbol} in cache and storage..."
            )
            self.save_data(symbol, df)

        return df

    def save_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Save data to storage.

        Args:
            symbol: Symbol to save data for.
            data: DataFrame with data.
        """
        self.storage.save_ohlcv_data(symbol=symbol, data=data)
