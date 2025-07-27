"""Data maintainer for ensuring continuous historical data availability."""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from turtle_quant_1.config import (
    DEFAULT_MARKET_HOURS,
    LIVE_SYMBOLS,
    MAX_CANDLE_GAPS_TO_FILL,
    MAX_HISTORY_MONTHS,
    SYMBOL_MARKET_HOURS,
)
from turtle_quant_1.data_processing.base import (
    BaseDataFetcher,
    BaseDataMaintainer,
)
from turtle_quant_1.data_processing.yfinance_fetcher import YFinanceDataFetcher

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
        self.fetcher = fetcher or YFinanceDataFetcher()

    def _get_symbol_market_hours(self, symbol: str) -> dict:
        """Get market hours for a specific symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Dictionary with 'open', 'close', and 'timezone' keys
        """
        return SYMBOL_MARKET_HOURS.get(symbol, DEFAULT_MARKET_HOURS)

    def _get_symbol_open_time(self, symbol: str) -> str:
        """Get market open time for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Open time string in "HH:MM" format (UTC)
        """
        return self._get_symbol_market_hours(symbol)["open"]

    def _get_symbol_close_time(self, symbol: str) -> str:
        """Get market close time for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Close time string in "HH:MM" format (UTC)
        """
        return self._get_symbol_market_hours(symbol)["close"]

    def _get_expected_market_hours_index(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """Generate a list of all expected market hours for a symbol between start and end dates.

        Args:
            symbol: Stock symbol to get market hours for
            start_date: Start date
            end_date: End date

        Returns:
            List of datetime objects representing each expected market hour for this symbol
        """
        expected_hours = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        # Get symbol-specific market hours
        open_time_str = self._get_symbol_open_time(symbol)
        close_time_str = self._get_symbol_close_time(symbol)
        open_hour, open_minute = map(int, open_time_str.split(":"))
        close_hour, close_minute = map(int, close_time_str.split(":"))

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:  # Monday=0, Friday=4
                # Generate hourly timestamps for this trading day
                market_open = current_date.replace(hour=open_hour, minute=open_minute)
                market_close = current_date.replace(
                    hour=close_hour, minute=close_minute
                )

                current_hour = market_open
                while current_hour <= market_close:
                    if start_date <= current_hour <= end_date:
                        expected_hours.append(current_hour)
                    current_hour += timedelta(hours=1)

            # Move to next day
            current_date += timedelta(days=1)

        return expected_hours

    def _convert_dates_to_market_hours(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> Tuple[datetime, datetime]:
        """Convert dates to market hours timestamps.

        Args:
            symbol: Stock symbol to get market hours for
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (market_open, market_close) datetimes
        """
        # Get symbol-specific market hours
        open_time_str = self._get_symbol_open_time(symbol)
        close_time_str = self._get_symbol_close_time(symbol)
        open_hour, open_minute = map(int, open_time_str.split(":"))
        close_hour, close_minute = map(int, close_time_str.split(":"))

        market_open = datetime.combine(
            start_date, datetime.min.time().replace(hour=open_hour, minute=open_minute)
        )
        market_close = datetime.combine(
            end_date, datetime.min.time().replace(hour=close_hour, minute=close_minute)
        )

        return market_open, market_close

    def _is_consecutive_market_hours(
        self, symbol: str, hour1: datetime, hour2: datetime
    ) -> bool:
        """Check if two market hours are consecutive for a specific symbol considering market breaks.

        Args:
            symbol: Stock symbol to get market hours for
            hour1: First hour
            hour2: Second hour

        Returns:
            True if hours are consecutive market hours for this symbol, False otherwise
        """
        # Get symbol-specific market hours
        open_time_str = self._get_symbol_open_time(symbol)
        close_time_str = self._get_symbol_close_time(symbol)
        open_hour, open_minute = map(int, open_time_str.split(":"))
        close_hour, close_minute = map(int, close_time_str.split(":"))

        # If it's just the next hour on the same day
        if hour2 == hour1 + timedelta(hours=1):
            return True

        # Check if hour1 is market close and hour2 is market open next trading day
        market_close_time = hour1.replace(hour=close_hour, minute=close_minute)

        # If hour1 is at market close
        if hour1.time() == market_close_time.time():
            # Find next trading day
            next_day = hour1 + timedelta(days=1)
            while next_day.weekday() >= 5:  # Skip weekends
                next_day += timedelta(days=1)

            next_market_open = next_day.replace(hour=open_hour, minute=open_minute)
            return hour2 == next_market_open

        return False

    def _is_weekend_gap(self, date1: datetime.date, date2: datetime.date) -> bool:
        """Check if two dates are consecutive considering weekends.

        Args:
            date1: First date
            date2: Second date

        Returns:
            True if dates are consecutive considering weekends, False otherwise
        """
        days_between = (date2 - date1).days
        if days_between == 1:
            return True
        if days_between == 3:
            # Check if the gap is over a weekend (Fri->Mon)
            return date1.weekday() == 4 and date2.weekday() == 0
        return False

    def _is_holiday_gap(self, date1: datetime.date, date2: datetime.date) -> bool:
        """Check if two dates are consecutive considering bank holidays.

        Args:
            date1: First date
            date2: Second date

        Returns:
            True if dates are consecutive considering bank holidays, False otherwise
        """
        # TODO: Implement!
        return False

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
            start_date: Start date to check from.
            end_date: End date to check to.

        Returns:
            List of (gap_start, gap_end) tuples representing missing data periods during this symbol's market hours.
            Each gap starts at market open and ends at market close of the respective days.
        """
        try:
            # Ensure datetime column is properly formatted and timezone-naive for comparisons
            if not data.empty:
                data["datetime"] = pd.to_datetime(data["datetime"])
                if data["datetime"].dt.tz is not None:
                    # Convert timezone-aware datetime to naive (UTC) to avoid comparison issues
                    data["datetime"] = (
                        data["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
                    )

            # Ensure start_date and end_date are timezone-naive for consistent comparisons
            if start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)

            # Generate expected market days for this symbol
            current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            expected_market_days = []

            while current_date <= end_date:
                if current_date.weekday() < 5:  # Monday=0, Friday=4
                    expected_market_days.append(current_date)
                current_date += timedelta(days=1)

            if not expected_market_days:
                return []

            # Get existing data dates
            if data.empty:
                existing_days = set()
            else:
                # Get unique dates from the hourly data
                data["date"] = data["datetime"].dt.date
                existing_days = set(data["date"].unique())

            # Convert expected_market_days to date objects for comparison
            expected_days_set = set(day.date() for day in expected_market_days)
            missing_days = sorted(expected_days_set - existing_days)

            if not missing_days:
                return []

            # Group consecutive missing days into gap periods
            gap_periods = []
            gap_start = missing_days[0]
            gap_end = missing_days[0]

            for i in range(1, len(missing_days)):
                current_day = missing_days[i]
                prev_day = missing_days[i - 1]

                # Check if days are consecutive (accounting for weekends)
                days_between = (current_day - prev_day).days
                if days_between <= 3 and self._is_weekend_gap(prev_day, current_day):
                    gap_end = current_day
                else:
                    # End current gap and start a new one
                    gap_periods.append(
                        self._convert_dates_to_market_hours(symbol, gap_start, gap_end)
                    )
                    gap_start = current_day
                    gap_end = current_day

            # Add the final gap
            gap_periods.append(
                self._convert_dates_to_market_hours(symbol, gap_start, gap_end)
            )

            return gap_periods

        except Exception as e:
            logger.warning(f"Error finding data gaps for {symbol}: {str(e)}")
            # If we can't load data, consider all market days as gaps
            if not expected_market_days:
                return []
            return [
                self._convert_dates_to_market_hours(
                    symbol,
                    expected_market_days[0].date(),
                    expected_market_days[-1].date(),
                )
            ]

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
        gaps_to_fill = gaps[:MAX_CANDLE_GAPS_TO_FILL]

        if total_gaps > MAX_CANDLE_GAPS_TO_FILL:
            logger.warning(
                f"Found {total_gaps} gaps for {symbol}, but only filling the first {MAX_CANDLE_GAPS_TO_FILL} "
                f"(configured limit). Remaining {total_gaps - MAX_CANDLE_GAPS_TO_FILL} gaps will be skipped."
            )

        for i, (gap_start, gap_end) in enumerate(gaps_to_fill, 1):
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
            start_date: Start date for the data. If None, uses MAX_HISTORY_MONTHS ago.
            end_date: End date for the data. If None, uses current time.

        Returns:
            DataFrame with gaps appended.
        """
        # Set default dates if not provided
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30 * MAX_HISTORY_MONTHS))

        existing_data = data

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

                # Handle timezone-aware datetime data by converting to naive datetime
                if existing_data["datetime"].dt.tz is not None:
                    existing_data["datetime"] = (
                        existing_data["datetime"]
                        .dt.tz_convert("UTC")
                        .dt.tz_localize(None)
                    )
                if df["datetime"].dt.tz is not None:
                    df["datetime"] = (
                        df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
                    )

                # Remove duplicates by datetime and combine
                combined_data = pd.concat([existing_data, df], ignore_index=True)
                combined_data = combined_data.drop_duplicates(
                    subset=["datetime"], keep="last"
                )
                combined_data = combined_data.sort_values("datetime").reset_index(
                    drop=True
                )
                df = combined_data

        return df

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
            end_date: End date to ensure data up to. If None, uses current time.

        Returns:
            DataFrame with updated data.
        """
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=30 * MAX_HISTORY_MONTHS)

        logger.info(
            f"Ensuring continuous data for {symbol} from {start_date} to {end_date}"
        )

        # Find gaps in the data
        gaps = self._get_data_gaps(symbol, data, start_date, end_date)

        # Fill any gaps found
        if gaps:
            logger.info(f"Found {len(gaps)} gaps in data for {symbol}")
            data = self._fill_data_gaps(symbol, data, gaps)
        else:
            logger.info(f"No gaps found in data for {symbol}")

        return data
