"""Data maintainer for ensuring continuous historical data availability."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from turtle_quant_1.config import MAX_HISTORY_YEARS, SYMBOLS
from turtle_quant_1.data_processing.base import (
    BaseDataMaintainer,
    BaseDataProcessor,
    BaseDataStorageAdapter,
)
from turtle_quant_1.data_processing.gcs_storage_adapter import GCSDataStorageAdapter
from turtle_quant_1.data_processing.processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMaintainer(BaseDataMaintainer):
    """Maintains continuous historical data for all symbols."""

    def __init__(
        self,
        processor: Optional[BaseDataProcessor] = None,
        storage: Optional[BaseDataStorageAdapter] = None,
        symbols: Optional[List[str]] = None,
    ):
        """Initialize the data maintainer.

        Args:
            processor: BaseDataProcessor instance. If None, creates a new DataProcessor.
            storage: Data storage to use. If None, uses GCSDataStorage.
            symbols: List of symbols to maintain. If None, uses SYMBOLS from config.
        """
        self.symbols = symbols or SYMBOLS
        self.storage = storage or GCSDataStorageAdapter()
        self.processor = processor or DataProcessor(
            symbols=self.symbols,
            storage=self.storage,
        )

    def _get_data_gaps(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """Find gaps in the data for a symbol.

        Args:
            symbol: Symbol to check for gaps.
            start_date: Start date to check from.
            end_date: End date to check to.

        Returns:
            List of (gap_start, gap_end) tuples representing missing data periods.
        """
        try:
            # Load existing data
            df = self.storage.load_ohlcv(symbol, start_date, end_date)

            if df.empty:
                return [(start_date, end_date)]

            # Sort by datetime
            df = df.sort_values("datetime")

            # Calculate time differences between consecutive rows
            df["time_diff"] = df["datetime"].diff()

            # Find gaps (more than 1 hour difference)
            gaps = df[df["time_diff"] > timedelta(hours=1)]["datetime"]

            # Create list of gap periods
            gap_periods = []

            # Add gap if there's missing data at the start
            if df["datetime"].min() > start_date:
                gap_periods.append((start_date, df["datetime"].min()))

            # Add intermediate gaps
            for gap_end in gaps:
                gap_start = (
                    gap_end - df.loc[df["datetime"] == gap_end, "time_diff"].iloc[0]
                )
                gap_periods.append((gap_start, gap_end))

            # Add gap if there's missing data at the end
            if df["datetime"].max() < end_date:
                gap_periods.append((df["datetime"].max(), end_date))

            return gap_periods

        except Exception as e:
            logger.warning(f"No existing data found for {symbol}: {str(e)}")
            return [(start_date, end_date)]

    def _fill_data_gaps(
        self,
        symbol: str,
        gaps: List[Tuple[datetime, datetime]],
    ) -> None:
        """Fill gaps in the data for a symbol.

        Args:
            symbol: Symbol to fill gaps for.
            gaps: List of (gap_start, gap_end) tuples representing missing data periods.
        """
        for gap_start, gap_end in gaps:
            logger.info(f"Filling gap for {symbol} from {gap_start} to {gap_end}")
            try:
                self.processor.update_data(
                    symbol=symbol,
                    start_date=gap_start,
                    end_date=gap_end,
                )
            except Exception as e:
                logger.error(f"Failed to fill gap for {symbol}: {str(e)}")

    def ensure_continuous_data(
        self,
        symbol: str,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Ensure continuous data exists for a symbol up to the specified end date.

        Args:
            symbol: Symbol to ensure data for.
            end_date: End date to ensure data up to. If None, uses current time.
        """
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=365 * MAX_HISTORY_YEARS)

        logger.info(
            f"Ensuring continuous data for {symbol} from {start_date} to {end_date}"
        )

        # Find gaps in the data
        gaps = self._get_data_gaps(symbol, start_date, end_date)

        # Fill any gaps found
        if gaps:
            logger.info(f"Found {len(gaps)} gaps in data for {symbol}")
            self._fill_data_gaps(symbol, gaps)
        else:
            logger.info(f"No gaps found in data for {symbol}")

    def ensure_all_continuous_data(
        self,
        end_date: Optional[datetime] = None,
    ) -> None:
        """Ensure continuous data exists for all symbols.

        Args:
            end_date: End date to ensure data up to. If None, uses current time.
        """
        for symbol in self.symbols:
            self.ensure_continuous_data(symbol, end_date)
