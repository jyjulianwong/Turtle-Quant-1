"""YFinance data fetcher implementation."""

from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from turtle_quant_1.data_processing.base import BaseDataFetcher
from turtle_quant_1.logging import get_logger

logger = get_logger(__name__)


class YFinanceDataFetcher(BaseDataFetcher):
    """YFinance implementation of data fetcher."""

    def __init__(self, symbols: List[str]):
        """Initialize the YFinance data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        super().__init__(symbols)
        self.tickers = {symbol: yf.Ticker(symbol) for symbol in symbols}

    def fetch_5min_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch 5-minute OHLCV data from YFinance.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Timezone-aware start date to fetch data from.
            end_date: Timezone-aware end date to fetch data up to.

        Returns:
            DataFrame with columns: datetime, Open, High, Low, Close, Volume.
        """
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

        if symbol not in self.tickers:
            raise ValueError(f"Symbol {symbol} not initialized")

        try:
            # Fetch all data in one go
            df = self.tickers[symbol].history(
                start=start_date,
                end=end_date,
                interval="5m",
            )

            # Return empty DataFrame if no data
            if df.empty:
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}"
                )
                return pd.DataFrame(
                    columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
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

            # Keep timezone-aware timestamps as they are with no conversions
            # Just ensure it's a datetime type
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])

            # Select only the columns we need and sort by datetime
            result_df = df[
                ["datetime", "Open", "High", "Low", "Close", "Volume"]
            ].copy()
            result_df = result_df.sort_values("datetime").reset_index(drop=True)

            logger.info(f"Successfully fetched {len(result_df)} records for {symbol}")

            return result_df

        except Exception as e:
            logger.error(
                f"Error fetching data for {symbol} from {start_date} to {end_date}: {str(e)}"
            )
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )
