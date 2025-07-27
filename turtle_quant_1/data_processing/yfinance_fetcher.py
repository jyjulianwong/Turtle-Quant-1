"""YFinance data fetcher implementation."""

import logging
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from turtle_quant_1.data_processing.base import BaseDataFetcher

logger = logging.getLogger(__name__)


class YFinanceDataFetcher(BaseDataFetcher):
    """YFinance implementation of data fetcher."""

    def __init__(self, symbols: List[str]):
        """Initialize the YFinance data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        super().__init__(symbols)
        self.tickers = {symbol: yf.Ticker(symbol) for symbol in symbols}

    def _fetch_hourly_ohlcv_batch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch data for a single year period.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            DataFrame with OHLCV data for the specified period.
        """
        if symbol not in self.tickers:
            raise ValueError(f"Symbol {symbol} not initialized")

        try:
            # Fetch data for this period
            df = self.tickers[symbol].history(
                start=start_date,
                end=end_date,
                interval="1h",
            )

            # Return empty DataFrame if no data
            if df.empty:
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}"
                )
                return pd.DataFrame(
                    columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
                )

            # Rename columns to match our standard format
            df = df.rename(
                columns={
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }
            )

            # Reset index to make datetime a column
            df = df.reset_index()

            # Handle different possible index column names from yfinance
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "datetime"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "datetime"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "datetime"})
            else:
                # If none of the above, the first column should be the datetime
                df.columns = ["datetime"] + list(df.columns[1:])

            # Ensure datetime column exists and convert timezone-aware datetime to naive
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                # Convert timezone-aware datetime to naive (UTC) to avoid comparison issues
                if df["datetime"].dt.tz is not None:
                    df["datetime"] = (
                        df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
                    )

            # Select only the columns we need
            return df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

        except Exception as e:
            logger.error(
                f"Error fetching data for {symbol} from {start_date} to {end_date}: {str(e)}"
            )
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

    def fetch_hourly_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly OHLCV data from YFinance year by year.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

        # List to store yearly data
        yearly_dataframes = []

        # Calculate year boundaries for chunked downloading
        current_start = start_date

        while current_start < end_date:
            # Calculate end of current year or end_date, whichever is earlier
            year_end = datetime(current_start.year + 1, 1, 1)
            current_end = min(year_end, end_date)

            logger.info(
                f"Fetching {symbol} data for period: {current_start} to {current_end}"
            )

            # Fetch data for this year
            year_df = self._fetch_hourly_ohlcv_batch(
                symbol=symbol,
                start_date=current_start,
                end_date=current_end,
            )

            # Add to list if not empty
            if not year_df.empty:
                yearly_dataframes.append(year_df)

            # Move to next year
            current_start = year_end

        # Combine all yearly dataframes
        if not yearly_dataframes:
            logger.warning(f"No data found for {symbol} in the specified date range")
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

        # Concatenate all yearly data
        combined_df = pd.concat(yearly_dataframes, ignore_index=True)

        # Ensure datetime column is properly formatted
        combined_df["datetime"] = pd.to_datetime(combined_df["datetime"])

        # Sort by datetime and remove any duplicates
        combined_df = (
            combined_df.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .reset_index(drop=True)
        )

        logger.info(f"Successfully fetched {len(combined_df)} records for {symbol}")

        return combined_df
