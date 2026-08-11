"""Alpha Vantage data fetcher implementation."""

from datetime import datetime
from typing import List

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

from turtle_quant_1.config import ALPHA_VANTAGE_API_KEY, CANDLE_UNIT
from turtle_quant_1.data_processing.base import BaseDataFetcher
from turtle_quant_1.logging import get_logger
from turtle_quant_1.strategies.helpers.candle_units import CandleUnit

logger = get_logger(__name__)

# Alpha Vantage intraday interval strings for each intraday CandleUnit.
# 2H and 4H are not supported natively; they are fetched at 60min and resampled.
_INTRADAY_CANDLE_UNITS: dict[CandleUnit, str] = {
    "5M": "5min",
    "15M": "15min",
    "30M": "30min",
    "1H": "60min",
    "2H": "60min",
    "4H": "60min",
}

# pandas resample rule for CandleUnits that need a post-fetch aggregation step.
_CANDLE_UNIT_TO_RESAMPLE_RULE: dict[CandleUnit, str] = {
    "2H": "2h",
    "4H": "4h",
}

_EMPTY_DF = pd.DataFrame(columns=["datetime", "Open", "High", "Low", "Close", "Volume"])

_AV_COLUMN_RENAME = {
    "1. open": "Open",
    "2. high": "High",
    "3. low": "Low",
    "4. close": "Close",
    "5. volume": "Volume",
}


class AlphaVantageDataFetcher(BaseDataFetcher):
    """Alpha Vantage implementation of data fetcher."""

    def __init__(self, symbols: List[str]):
        """Initialize the Alpha Vantage data fetcher.

        Args:
            symbols: List of symbols to fetch data for.
        """
        super().__init__(symbols)
        self.client = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")

    def _normalise_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename Alpha Vantage columns and promote the index to a datetime column."""
        df = df.rename(columns=_AV_COLUMN_RENAME)
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
        return df[["datetime", "Open", "High", "Low", "Close", "Volume"]]

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        freq: CandleUnit = CANDLE_UNIT,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpha Vantage at the requested candle frequency.

        2H and 4H candles are not supported natively by Alpha Vantage; they are
        fetched at 60-minute resolution and resampled before being returned.
        Daily and weekly data use the dedicated Alpha Vantage endpoints.

        Args:
            symbol: The symbol to fetch data for.
            start_date: Start date for the data.
            end_date: End date for the data.
            freq: Candle frequency for the returned data.

        Returns:
            DataFrame with columns: datetime, Open, High, Low, Close, Volume.
        """
        logger.info(
            f"Fetching {freq} data for {symbol} from {start_date} to {end_date}"
        )

        try:
            if freq == "1W":
                raw, _ = self.client.get_weekly(symbol=symbol)  # pyrefly: ignore
            elif freq == "1D":
                raw, _ = self.client.get_daily(  # pyrefly: ignore
                    symbol=symbol, outputsize="full"
                )
            else:
                av_interval = _INTRADAY_CANDLE_UNITS[freq]
                raw, _ = self.client.get_intraday(  # pyrefly: ignore
                    symbol=symbol, interval=av_interval, outputsize="full"
                )

            if raw.empty:
                logger.warning(
                    f"No data returned for {symbol} from {start_date} to {end_date}"
                )
                return _EMPTY_DF.copy()

            df = self._normalise_df(raw)
            df = df.sort_values("datetime").reset_index(drop=True)

            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)

            if freq in ("1D", "1W"):
                # Compare at date granularity to avoid intraday-time vs midnight mismatch
                start_cmp = pd.Timestamp(start_ts.date())
                end_cmp = pd.Timestamp(end_ts.date())
                dt_cmp = pd.to_datetime(df["datetime"].dt.date)
            else:
                # Normalize to UTC-naive so tz-aware start/end compares cleanly
                if start_ts.tz is not None:
                    start_cmp = start_ts.tz_convert("UTC").tz_localize(None)
                    end_cmp = end_ts.tz_convert("UTC").tz_localize(None)
                else:
                    start_cmp = start_ts
                    end_cmp = end_ts
                dt_cmp = (
                    df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
                    if df["datetime"].dt.tz is not None
                    else df["datetime"]
                )

            df = df.loc[dt_cmp.between(start_cmp, end_cmp)].reset_index(drop=True)

            resample_rule = _CANDLE_UNIT_TO_RESAMPLE_RULE.get(freq)
            if resample_rule is not None:
                df = (
                    df.set_index("datetime")
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

            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(
                f"Error fetching {freq} data for {symbol} "
                f"from {start_date} to {end_date}: {str(e)}"
            )
            return _EMPTY_DF.copy()
