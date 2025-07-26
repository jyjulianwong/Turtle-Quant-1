"""YFinance data fetcher implementation."""

from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from turtle_quant_1.data_processing.base import BaseDataFetcher


class YFinanceDataFetcher(BaseDataFetcher):
    """YFinance implementation of data fetcher."""

    def __init__(self, symbols: List[str]):
        """Initialize the YFinance data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        super().__init__(symbols)
        self.tickers = {symbol: yf.Ticker(symbol) for symbol in symbols}

    def fetch_hourly_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch hourly OHLCV data from YFinance.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
        """
        if symbol not in self.tickers:
            raise ValueError(f"Symbol {symbol} not initialized")

        # Fetch hourly data
        df = self.tickers[symbol].history(
            start=start_date,
            end=end_date,
            interval="1h",
        )

        # Return empty DataFrame if no data
        if df.empty:
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

        # Select only the columns we need
        return df[["datetime", "Open", "High", "Low", "Close", "Volume"]]
