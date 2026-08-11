"""YFinance data fetcher implementation."""

from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

from turtle_quant_1.data_processing.base import BaseDataFetcher
from turtle_quant_1.logging import get_logger
from turtle_quant_1.strategies.helpers.candle_units import CandleUnit

logger = get_logger(__name__)

# yfinance native interval strings for each CandleUnit.
# 2H and 4H are not supported natively; they are fetched at 1H and resampled.
_CANDLE_UNIT_TO_YF_INTERVAL: dict[CandleUnit, str] = {
    "5M": "5m",
    "15M": "15m",
    "30M": "30m",
    "1H": "1h",
    "2H": "1h",
    "4H": "1h",
    "1D": "1d",
    "1W": "1wk",
}

# pandas resample rule for CandleUnits that need a post-fetch aggregation step.
_CANDLE_UNIT_TO_RESAMPLE_RULE: dict[CandleUnit, str] = {
    "2H": "2h",
    "4H": "4h",
}

_EMPTY_DF = pd.DataFrame(columns=["datetime", "Open", "High", "Low", "Close", "Volume"])


class YFinanceDataFetcher(BaseDataFetcher):
    """YFinance implementation of data fetcher."""

    def __init__(self, symbols: List[str]):
        """Initialize the YFinance data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        super().__init__(symbols)
        self.tickers = {symbol: yf.Ticker(symbol) for symbol in symbols}

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        freq: CandleUnit = "5M",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from YFinance at the requested candle frequency.

        2H and 4H candles are not supported natively by yfinance; they are
        fetched at 1H resolution and resampled before being returned.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Timezone-aware start date to fetch data from.
            end_date: Timezone-aware end date to fetch data up to.
            freq: Candle frequency for the returned data.

        Returns:
            DataFrame with columns: datetime, Open, High, Low, Close, Volume.
        """
        logger.info(
            f"Fetching {freq} data for {symbol} from {start_date} to {end_date}"
        )

        if symbol not in self.tickers:
            raise ValueError(f"Symbol {symbol} not initialized")

        yf_interval = _CANDLE_UNIT_TO_YF_INTERVAL[freq]

        try:
            df = self.tickers[symbol].history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
            )

            if df.empty:
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}"
                )
                return _EMPTY_DF.copy()

            # Reset index and normalise the datetime column name
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "datetime"})
            elif "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "datetime"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "datetime"})
            else:
                df.columns = ["datetime"] + list(df.columns[1:])

            df["datetime"] = pd.to_datetime(df["datetime"])

            result_df = (
                df[["datetime", "Open", "High", "Low", "Close", "Volume"]]
                .copy()
                .sort_values("datetime")
                .reset_index(drop=True)
            )

            # Resample 1H base data up to 2H / 4H where yfinance has no native interval
            resample_rule = _CANDLE_UNIT_TO_RESAMPLE_RULE.get(freq)
            if resample_rule is not None:
                result_df = (
                    result_df.set_index("datetime")
                    .resample(resample_rule)
                    .agg(
                        Open=("Open", "first"),
                        High=("High", "max"),
                        Low=("Low", "min"),
                        Close=("Close", "last"),
                        Volume=("Volume", "sum"),
                    )
                    .dropna(subset=["Open"])
                    .reset_index()
                )

            logger.info(f"Successfully fetched {len(result_df)} records for {symbol}")
            return result_df

        except Exception as e:
            logger.error(
                f"Error fetching {freq} data for {symbol} "
                f"from {start_date} to {end_date}: {str(e)}"
            )
            return _EMPTY_DF.copy()
