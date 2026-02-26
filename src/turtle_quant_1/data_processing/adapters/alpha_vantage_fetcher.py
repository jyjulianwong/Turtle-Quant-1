"""Alpha Vantage data fetcher implementation."""

from datetime import datetime
from typing import List

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

from turtle_quant_1.config import ALPHA_VANTAGE_API_KEY
from turtle_quant_1.data_processing.base import BaseDataFetcher


class AlphaVantageDataFetcher(BaseDataFetcher):
    """Alpha Vantage implementation of data fetcher."""

    def __init__(self, symbols: List[str]):
        """Initialize the Alpha Vantage data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        super().__init__(symbols)
        self.client = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")

    def fetch_5min_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch 5-minute OHLCV data from Alpha Vantage.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Start date for the data.
            end_date: End date for the data.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.
        """
        # Get 5-minute data (returns tuple of (data, metadata))
        df, _ = self.client.get_intraday(  # pyrefly: ignore
            symbol=symbol, interval="5min", outputsize="full"
        )

        # Return empty DataFrame if no data
        if df.empty:
            return pd.DataFrame(
                columns=["datetime", "Open", "High", "Low", "Close", "Volume"]
            )

        # Filter by date range
        df = df.loc[start_date:end_date]

        # Rename columns to match our standard format
        df = df.rename(
            columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }
        )

        # Reset index to make datetime a column
        df = df.reset_index()

        # Handle different possible index column names
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
