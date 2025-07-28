from datetime import datetime, timedelta
from typing import List

import holidays
import math
import pytz

from turtle_quant_1.config import MARKET_HOURS, SYMBOL_MARKETS, MAX_HISTORY_DAYS


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


def get_expected_market_hours_index(
    symbol: str, start_date: datetime, end_date: datetime
) -> List[datetime]:
    """Generate a list of all expected market hours for a symbol between start and end dates.

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
                curr_hour += timedelta(hours=1)

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
    market_code = SYMBOL_MARKETS.get(symbol, "NYSE")  # TODO: Handle LSE.
    curr_year = datetime.now().year
    years = [curr_year - i for i in range(math.ceil(MAX_HISTORY_DAYS / 365) + 1)]
    holiday_dates = list(holidays.financial_holidays(market_code, years=years).keys())
    return date.date() in holiday_dates
