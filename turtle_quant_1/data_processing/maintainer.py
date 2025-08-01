"""Data maintainer for ensuring continuous historical data availability."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import pytz

from turtle_quant_1.config import (
    LIVE_SYMBOLS,
    MAX_CANDLE_GAPS_TO_FILL,
    MAX_HISTORY_DAYS,
    HOST_TIMEZONE,
)
from turtle_quant_1.data_processing.base import (
    BaseDataFetcher,
    BaseDataMaintainer,
)
from turtle_quant_1.data_processing.datetimes import (
    get_expected_market_hours_index,
    get_symbol_timezone,
)
from turtle_quant_1.data_processing.adapters.yfinance_fetcher import YFinanceDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMaintainer(BaseDataMaintainer):
    """Maintains continuous historical data for all symbols."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        fetcher: Optional[BaseDataFetcher] = None,
    ):
        """Initialize the data maintainer.

        Args:
            symbols: List of symbols to maintain. If None, uses SYMBOLS from config.
            fetcher: Data fetcher to use. If None, uses YFinanceDataFetcher.
        """
        self.symbols = symbols or LIVE_SYMBOLS
        self.fetcher = fetcher or YFinanceDataFetcher(symbols=self.symbols)

    def _get_data_gaps(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """Find gaps in the data for a symbol during its specific market hours.

        Args:
            symbol: Symbol to check for gaps.
            data: DataFrame with data.
            start_date: Timezone-aware start date to check from.
            end_date: Timezone-aware end date to check to.

        Returns:
            List of (gap_start, gap_end) tuples representing missing data periods during this symbol's market hours.
            Each gap represents a continuous period of missing hourly data.
        """
        try:
            # Ensure datetime column is properly formatted and timezone-naive for comparisons
            if not data.empty:
                data["datetime"] = pd.to_datetime(data["datetime"])

            # Ensure start_date and end_date are timezone-naive for consistent comparisons
            if start_date.tzinfo is not None:
                start_date = start_date.astimezone(
                    pytz.timezone(get_symbol_timezone(symbol))
                )
            if end_date.tzinfo is not None:
                end_date = end_date.astimezone(
                    pytz.timezone(get_symbol_timezone(symbol))
                )
                end_date = min(
                    end_date, datetime.now().astimezone(pytz.timezone(HOST_TIMEZONE))
                )

            # Generate expected market hours for this symbol (hourly resolution)
            expected_datetimes = get_expected_market_hours_index(
                symbol, start_date, end_date
            )
            if not expected_datetimes:
                return []

            # Create DataFrame with all expected hours
            expected_df = pd.DataFrame({"datetime": expected_datetimes})

            # Get existing data hours (rounded to nearest hour)
            if data.empty:
                existing_datetimes = set()
            else:
                data_datetimes = data["datetime"]
                existing_datetimes = set(
                    data_datetimes.unique()
                )  # pyrefly: ignore[no-matching-overload]

            # Add exists flag
            expected_df["exists"] = expected_df["datetime"].isin(existing_datetimes)

            # Find gaps using groupby on consecutive missing periods
            return self._get_data_gaps_with_groupby(expected_df)

        except Exception as e:
            logger.warning(f"Error finding data gaps for {symbol}: {str(e)}")
            # If we can't process data properly, return empty gaps list
            return []

    def _get_data_gaps_with_groupby(
        self, expected_df: pd.DataFrame
    ) -> List[Tuple[datetime, datetime]]:
        """Find gaps using pandas groupby on consecutive missing periods.

        Args:
            expected_df: DataFrame with 'datetime' and 'exists' columns

        Returns:
            List of (gap_start, gap_end) tuples representing continuous missing periods
        """
        # Create groups for consecutive missing periods
        # Use cumsum on the difference between current index and row number to identify groups
        expected_df = expected_df.reset_index(drop=True)
        expected_df["group"] = (
            expected_df["exists"].ne(expected_df["exists"].shift()).cumsum()
        )

        # Filter to only missing hours
        missing_df = expected_df[~expected_df["exists"]]
        if missing_df.empty:
            return []

        # Group by consecutive periods and get min/max datetime for each gap
        gaps = []
        for _, group in missing_df.groupby("group"):
            gap_start = group["datetime"].min()
            gap_end = group["datetime"].max()
            gaps.append((gap_start, gap_end))

        return gaps

    def _fill_data_gaps(
        self,
        symbol: str,
        data: pd.DataFrame,
        gaps: List[Tuple[datetime, datetime]],
    ) -> pd.DataFrame:
        """Fill gaps in the data for a symbol.

        Args:
            symbol: Symbol to fill gaps for.
            data: DataFrame with data.
            gaps: List of (gap_start, gap_end) tuples representing missing data periods.

        Returns:
            DataFrame with gaps filled.
        """
        total_gaps = len(gaps)
        gaps_to_fill = gaps

        if total_gaps > MAX_CANDLE_GAPS_TO_FILL:
            logger.warning(
                f"Found {total_gaps} gaps for {symbol} (configured limit). "
                f"Entire dataset for symbol {symbol} will be filled."
            )
            gaps_to_fill = [(gaps[0][0], gaps[-1][1])]

        for i, (gap_start, gap_end) in enumerate(gaps_to_fill, 1):  # pyrefly: ignore
            logger.info(
                f"Filling gap {i}/{len(gaps_to_fill)} for {symbol} from {gap_start} to {gap_end}..."
            )
            try:
                data = self._append_gaps_to_data(
                    symbol=symbol,
                    data=data,
                    start_date=gap_start,
                    end_date=gap_end,
                )
            except Exception as e:
                logger.error(f"Failed to fill gap {i} for {symbol}: {str(e)}")

        return data

    def _append_gaps_to_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Append gaps to data for a symbol.

        Args:
            symbol: Symbol to append gaps to.
            data: DataFrame with data.
            start_date: Timezone-aware start date for the data. If None, uses MAX_HISTORY_DAYS ago.
            end_date: Timezone-aware end date for the data. If None, uses current time.

        Returns:
            DataFrame with gaps appended.
        """
        # Set default dates if not provided
        end_date = end_date or datetime.now().astimezone(pytz.timezone(HOST_TIMEZONE))
        start_date = start_date or (end_date - timedelta(days=MAX_HISTORY_DAYS))

        existing_data = data.copy()

        # Fetch data
        fetched_data = self.fetcher.fetch_hourly_ohlcv(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        # Combine with existing data if any
        if not existing_data.empty and not fetched_data.empty:
            # Convert datetime columns to same type for concatenation
            if (
                "datetime" in existing_data.columns
                and "datetime" in fetched_data.columns
            ):
                existing_data["datetime"] = pd.to_datetime(existing_data["datetime"])
                fetched_data["datetime"] = pd.to_datetime(fetched_data["datetime"])

                # Remove duplicates by datetime and combine
                combined_data = pd.concat(
                    [existing_data, fetched_data], ignore_index=True
                )
                combined_data = combined_data.drop_duplicates(
                    subset=["datetime"], keep="last"
                )
                combined_data = combined_data.sort_values("datetime").reset_index(
                    drop=True
                )
                existing_data = combined_data

        return existing_data

    def impute_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Ensure continuous data exists for a symbol up to the specified end date.

        Args:
            symbol: Symbol to ensure data for.
            data: DataFrame with data.
            end_date: Timezone-aware end date to ensure data up to. If None, uses current time.

        Returns:
            DataFrame with updated data.
        """
        end_date = end_date or datetime.now().astimezone(pytz.timezone(HOST_TIMEZONE))
        start_date = end_date - timedelta(days=MAX_HISTORY_DAYS)

        logger.info(
            f"Ensuring continuous data for {symbol} from {start_date} to {end_date}"
        )

        # Find gaps in the data
        gaps = self._get_data_gaps(symbol, data, start_date, end_date)

        # Fill any gaps found
        if gaps:
            logger.info(f"Found {len(gaps)} gaps in data for {symbol}")
            for i, gap in enumerate(gaps):
                logger.info(f"    - Gap {i}: {gap[0]} to {gap[1]}")
            data = self._fill_data_gaps(symbol, data, gaps)
        else:
            logger.info(f"No gaps found in data for {symbol}")

        return data
