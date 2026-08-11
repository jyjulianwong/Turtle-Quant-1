import math
from datetime import datetime, timedelta
from typing import List

import holidays
import pytz

from turtle_quant_1.config import MARKET_HOURS, MAX_HISTORY_DAYS, SYMBOL_MARKETS


def get_symbol_market_hours(symbol: str) -> dict:
    """Get market hours for a specific symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')

    Returns:
        Dictionary with 'open', 'close', and 'timezone' keys
    """
    return MARKET_HOURS.get(SYMBOL_MARKETS.get(symbol, "NYSE"), MARKET_HOURS["NYSE"])


def get_symbol_opening_time(symbol: str) -> str:
    """Get market open time for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Open time string in "HH:MM" format (UTC)
    """
    return get_symbol_market_hours(symbol)["opening"]


def get_symbol_closing_time(symbol: str) -> str:
    """Get market close time for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Close time string in "HH:MM" format (UTC)
    """
    return get_symbol_market_hours(symbol)["closing"]


def get_symbol_timezone(symbol: str) -> str:
    """Get timezone for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Timezone string
    """
    return get_symbol_market_hours(symbol)["timezone"]


def get_expected_market_hours_bounds(
    symbol: str, start_date: datetime, end_date: datetime
) -> tuple[datetime, datetime]:
    """Get the expected market hours boundaries for a symbol between start and end dates.

    TODO: Cache.

    Args:
        symbol: Stock symbol
        start_date: Timezone-aware start date
        end_date: Timezone-aware end date

    Returns:
        Tuple of timezone-aware datetime objects representing the start and end of the expected market hours
    """
    expected_datetimes = get_expected_market_hours_index(symbol, start_date, end_date)
    return expected_datetimes[0], expected_datetimes[-1]


def get_expected_market_hours_index(
    symbol: str, start_date: datetime, end_date: datetime
) -> List[datetime]:
    """Generate a list of all expected market hours for a symbol between start and end dates.

    TODO: Cache.

    Args:
        symbol: Stock symbol to get market hours for
        start_date: Timezone-aware start date
        end_date: Timezone-aware end date

    Returns:
        List of timezone-aware datetime objects representing each expected market hour for this symbol
    """
    expected_datetimes = []
    curr_datetime = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Get symbol-specific market hours
    o_time_str = get_symbol_opening_time(symbol)
    c_time_str = get_symbol_closing_time(symbol)
    o_hour, o_minute = map(int, o_time_str.split(":"))
    c_hour, c_minute = map(int, c_time_str.split(":"))

    while curr_datetime <= end_date:
        # Skip weekends and holidays
        if not is_weekend_date(curr_datetime) and not is_holiday_date(
            curr_datetime, symbol
        ):
            # Generate hourly timestamps for this trading day
            market_o_datetime = curr_datetime.replace(hour=o_hour, minute=o_minute)
            market_c_datetime = curr_datetime.replace(hour=c_hour, minute=c_minute)

            curr_hour = market_o_datetime
            while curr_hour <= market_c_datetime:
                if start_date <= curr_hour <= end_date:
                    expected_datetimes.append(curr_hour)
                # NOTE: Based on assumption that 5 minutes is the smallest unit of time for the data.
                curr_hour += timedelta(minutes=5)

        # Move to next day (DST-aware)
        if curr_datetime.tzinfo is not None:
            # For timezone-aware datetimes, use proper timezone handling for DST
            next_date = curr_datetime.date() + timedelta(days=1)

            # Get the timezone for this symbol to ensure proper DST handling
            timezone_str = get_symbol_timezone(symbol)
            symbol_tz = pytz.timezone(timezone_str)

            # Create midnight on the next day in the correct timezone
            naive_next_midnight = datetime.combine(next_date, datetime.min.time())
            curr_datetime = symbol_tz.localize(naive_next_midnight)
        else:
            # For naive datetimes, use simple timedelta
            curr_datetime += timedelta(days=1)

    return expected_datetimes


def is_within_market_hours(symbol: str, timestamp: datetime) -> bool:
    """Check if the given timestamp is within trading hours for the symbol.

    Args:
        symbol: Symbol to check trading hours for.
        timestamp: Timestamp to check. Timezone-aware.

    Returns:
        True if within trading hours (including 2-hour grace period after close), False otherwise.
    """
    # Check if it's a weekday (0=Monday, 6=Sunday)
    if timestamp.weekday() >= 5:  # Saturday or Sunday
        return False

    # Get market hours for this symbol in its local timezone
    market = SYMBOL_MARKETS.get(symbol, "NYSE")
    market_hours = MARKET_HOURS.get(market, MARKET_HOURS["NYSE"])

    # Parse market open and close times (format: "HH:MM") in its local timezone
    o_hour, o_minute = map(int, market_hours["opening"].split(":"))
    c_hour, c_minute = map(int, market_hours["closing"].split(":"))

    # Create market open and close times for the current day whilst preserving the timezone
    market_o_datetime = timestamp.replace(
        hour=o_hour, minute=o_minute, second=0, microsecond=0
    )
    market_c_datetime = timestamp.replace(
        hour=c_hour, minute=c_minute, second=0, microsecond=0
    )

    # Add 1-hour grace period after market close
    market_c_dt_with_grace = market_c_datetime + timedelta(hours=1)

    # Check if within trading hours (including grace period)
    return market_o_datetime <= timestamp <= market_c_dt_with_grace


def is_weekend_date(date: datetime) -> bool:
    """Check if a date is a weekend.

    Args:
        date: Date to check

    Returns:
        True if date is a weekend, False otherwise
    """
    return date.weekday() >= 5


def is_holiday_date(date: datetime, symbol: str) -> bool:
    """Check if a date is a bank holiday.

    Args:
        date: Date to check

    Returns:
        True if date is a bank holiday, False otherwise
    """
    # Get market code for this symbol
    market_code = SYMBOL_MARKETS.get(symbol, "NYSE")
    market_code = "XECB" if market_code == "ECB" else market_code

    # Determine the years to check for holidays
    curr_year = datetime.now().year
    years = [curr_year - i for i in range(math.ceil(MAX_HISTORY_DAYS / 365) + 1)]

    # Get holidays straight from the library
    if market_code == "LSE":
        # NOTE: LSE is not supported by holidays.financial_holidays. Substitute with country_holidays.
        holiday_dates = list(holidays.country_holidays("GB", years=years).keys())
    else:
        holiday_dates = list(
            holidays.financial_holidays(market_code, years=years).keys()
        )

    # Add overrides for holidays that are not in the library
    holiday_overrides = []
    if market_code == "NYSE":
        holiday_overrides = [
            datetime(2023, 11, 24).date(),
            datetime(2024, 7, 3).date(),
            datetime(2024, 11, 29).date(),
            datetime(2024, 12, 24).date(),
            datetime(2025, 7, 3).date(),
            datetime(2025, 11, 28).date(),
            datetime(2025, 12, 24).date(),
            # NOTE: This needs to be regularly updated by the maintainer.
        ]

    return date.date() in (holiday_dates + holiday_overrides)
